"""Thread/lifecycle proof for the experimental pool, without Torch or a GPU."""
import queue
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from posegraph.i18n import _lang
from posegraph.inference_pool import InferenceFailure
from posegraph.shared_inference import SharedInferencePool


class ModelDouble:
    def __init__(self):
        self.loaded = False
        self.active_device = self.active_precision = None
        self.device_reason = ("手动指定",)
        self.lock = threading.Lock()
        self.load_calls = self.unloads = self.active = self.peak = 0
        self.load_failures = self.unload_failures = 0
        self.load_gate = self.unload_gate = None
        self.load_started = threading.Event()
        self.unload_started = threading.Event()
        self.started = queue.Queue()
        self.gates = {}
        self.fail = set()
        self.language_barrier = None
        self.loads_language = []

    @staticmethod
    def wait(gate):
        if gate is not None and not gate.wait(timeout=5):
            raise TimeoutError("test gate was not released")

    def load(self):
        with self.lock:
            if self.active:
                raise AssertionError("load overlapped inference")
            self.load_calls += 1
            self.loads_language.append(_lang.get())
        self.load_started.set()
        self.wait(self.load_gate)
        with self.lock:
            if self.load_failures:
                self.load_failures -= 1
                raise ValueError("fake load failure")
            self.loaded = True
            self.active_device, self.active_precision = "cpu", "fp32"

    def unload(self):
        self.unload_started.set()
        self.wait(self.unload_gate)
        with self.lock:
            if self.active:
                raise AssertionError("unload overlapped inference")
            if self.unload_failures:
                self.unload_failures -= 1
                raise ValueError("fake unload failure")
            if self.loaded:
                self.unloads += 1
            self.loaded = False
            self.active_device = self.active_precision = None

    def estimate(self, marker):
        with self.lock:
            if not self.loaded:
                raise AssertionError("estimate ran without weights")
            self.active += 1
            self.peak = max(self.peak, self.active)
        language = _lang.get()
        self.started.put((marker, threading.get_ident()))
        try:
            self.wait(self.gates.get(marker))
            if self.language_barrier is not None:
                self.language_barrier.wait(timeout=5)
            if marker in self.fail:
                raise ValueError("fake request failure")
            return SimpleNamespace(marker=marker, language=language, after_language=_lang.get())
        finally:
            with self.lock:
                self.active -= 1

    def diagnostics(self, *, include_pointers=False):
        return {"sharedModel": True, "includePointers": include_pointers}


class SharedInferencePoolTests(unittest.TestCase):
    def pool(self, count=2, owner=None):
        owner = owner or ModelDouble()
        calls = []
        def factory(config):
            calls.append(config)
            return owner
        pool = SharedInferencePool(count, {"cuda_streams": True}, factory)
        self.addCleanup(pool.close)
        return pool, owner, calls

    def started(self, owner, count=1):
        return [owner.started.get(timeout=5) for _ in range(count)]

    def wait(self, condition):
        deadline = time.monotonic() + 5
        while not condition():
            if time.monotonic() > deadline:
                self.fail("Timed out waiting for lifecycle transition")
            time.sleep(.005)

    def test_one_factory_one_load_and_real_overlap_bounded_by_limit(self):
        for limit in (1, 3):
            with self.subTest(limit=limit):
                pool, owner, factories = self.pool(limit)
                total = 7
                owner.gates = {i: threading.Event() for i in range(total)}
                with ThreadPoolExecutor(max_workers=total) as executor:
                    futures = [executor.submit(pool.estimate, i, -1, "en") for i in range(total)]
                    running = self.started(owner, limit)
                    self.assertEqual(len({thread for _, thread in running}), limit)
                    self.assertEqual(owner.active, limit)
                    self.assertEqual(pool.states[0]["active"], limit)
                    self.assertEqual(owner.load_calls, 1)
                    for _ in range(total - limit):
                        marker, _ = running.pop()
                        owner.gates[marker].set()
                        running += self.started(owner)
                        self.assertLessEqual(owner.peak, limit)
                    for marker, _ in running:
                        owner.gates[marker].set()
                    self.assertEqual([f.result(timeout=5).marker for f in futures], list(range(total)))
                self.assertEqual(owner.peak, limit)
                self.assertEqual(len(factories), 1)
                self.assertEqual(factories[0], {"cuda_streams": True})
                self.assertEqual(owner.load_calls, 1)
                self.assertEqual(pool.states[0]["active"], 0)
                self.assertEqual(pool.processes, [])
                self.assertEqual(pool.states[0]["modelCopies"], 1)

    def test_release_zero_waits_last_active_even_with_mixed_policies(self):
        pool, owner, factories = self.pool()
        owner.gates = {marker: threading.Event() for marker in ("zero", "resident")}
        with ThreadPoolExecutor(max_workers=2) as executor:
            zero = executor.submit(pool.estimate, "zero", 0, "en")
            resident = executor.submit(pool.estimate, "resident", -1, "en")
            self.started(owner, 2)
            owner.gates["zero"].set()
            zero.result(timeout=5)
            self.assertEqual(owner.unloads, 0)
            self.assertTrue(owner.loaded)
            self.assertEqual(pool.states[0]["active"], 1)
            owner.gates["resident"].set()
            resident.result(timeout=5)
        self.assertFalse(owner.loaded)
        self.assertEqual(owner.unloads, 1)
        self.assertEqual(pool.estimate("next", -1).marker, "next")
        self.assertEqual(owner.load_calls, 2)
        self.assertEqual(len(factories), 1)

    def test_concurrent_preload_creates_one_copy_without_processes(self):
        pool, owner, factories = self.pool(8)
        owner.load_gate = threading.Event()
        with patch("multiprocessing.get_context", side_effect=AssertionError("must not spawn")):
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(pool.preload, "en") for _ in range(8)]
                self.assertTrue(owner.load_started.wait(timeout=5))
                state = pool.states[0]
                self.assertTrue(state["loading"])
                self.assertEqual(state["active"], 0)
                owner.load_gate.set()
                for future in futures:
                    future.result(timeout=5)
            pool.preload()
        self.assertEqual(owner.load_calls, 1)
        self.assertEqual(len(factories), 1)
        state = pool.states[0]
        self.assertEqual((state["strategy"], state["modelCopies"], state["loadedModelCopies"]), ("shared", 1, 1))
        self.assertTrue(pool.diagnostics(include_pointers=True)["owner"]["includePointers"])
        pool.close()
        pool.close()
        self.assertEqual(owner.unloads, 1)
        self.assertFalse(pool.states[0]["loaded"])

    def test_close_rejects_waiters_and_new_work_but_drains_active(self):
        pool, owner, _ = self.pool(1)
        owner.gates["active"] = threading.Event()
        with ThreadPoolExecutor(max_workers=3) as executor:
            active = executor.submit(pool.estimate, "active", -1, "en")
            self.started(owner)
            waiting = executor.submit(pool.estimate, "waiting", -1, "en")
            closing = executor.submit(pool.close)
            self.wait(lambda: pool.states[0]["closing"])
            with self.assertRaisesRegex(RuntimeError, "closed"):
                waiting.result(timeout=5)
            self.assertFalse(closing.done())
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pool.preload()
            owner.gates["active"].set()
            self.assertEqual(active.result(timeout=5).marker, "active")
            closing.result(timeout=5)
        self.assertEqual(owner.unloads, 1)
        self.assertEqual(pool.states[0]["active"], 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.estimate("new", -1)

    def test_close_waits_in_progress_load_without_admitting_its_request(self):
        pool, owner, _ = self.pool()
        owner.load_gate = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as executor:
            estimate = executor.submit(pool.estimate, "never-admitted", -1, "en")
            self.assertTrue(owner.load_started.wait(timeout=5))
            closing = executor.submit(pool.close)
            self.wait(lambda: pool.states[0]["closing"])
            self.assertFalse(closing.done())
            owner.load_gate.set()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                estimate.result(timeout=5)
            closing.result(timeout=5)
        self.assertTrue(owner.started.empty())
        self.assertEqual(owner.unloads, 1)

    def test_new_request_waits_for_release_unload_before_reloading(self):
        pool, owner, _ = self.pool()
        owner.unload_gate = threading.Event()
        owner.gates["second"] = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(pool.estimate, "first", 0, "en")
            self.assertTrue(owner.unload_started.wait(timeout=5))
            self.assertEqual(self.started(owner)[0][0], "first")
            second = executor.submit(pool.estimate, "second", -1, "en")
            self.assertEqual(owner.load_calls, 1)
            self.assertEqual(pool.states[0]["active"], 0)
            owner.unload_gate.set()
            first.result(timeout=5)
            self.assertEqual(self.started(owner)[0][0], "second")
            owner.gates["second"].set()
            second.result(timeout=5)
        self.assertEqual(owner.load_calls, 2)
        self.assertEqual(owner.unloads, 1)

    def test_failed_request_frees_slot_without_poisoning_model_or_health(self):
        pool, owner, _ = self.pool()
        owner.fail.add("bad")
        owner.gates = {name: threading.Event() for name in ("bad", "good")}
        with ThreadPoolExecutor(max_workers=2) as executor:
            bad = executor.submit(pool.estimate, "bad", -1, "en")
            good = executor.submit(pool.estimate, "good", -1, "en")
            self.started(owner, 2)
            for gate in owner.gates.values():
                gate.set()
            with self.assertRaisesRegex(InferenceFailure, "fake request failure") as caught:
                bad.result(timeout=5)
            self.assertEqual(caught.exception.status, 500)
            self.assertEqual(good.result(timeout=5).marker, "good")
        self.assertEqual(pool.estimate("recovered", -1).marker, "recovered")
        self.assertEqual(owner.load_calls, 1)
        self.assertIsNone(pool.states[0]["error"])
        self.assertEqual(pool.states[0]["active"], 0)

    def test_load_and_factory_failures_are_retryable_503_and_preserve_language(self):
        pool, owner, calls = self.pool()
        owner.load_failures = 1
        original = _lang.set("zh")
        try:
            with self.assertRaisesRegex(InferenceFailure, "fake load failure") as caught:
                pool.preload("en")
            self.assertEqual(caught.exception.status, 503)
            self.assertEqual(_lang.get(), "zh")
            self.assertIn("fake load failure", pool.states[0]["error"])
            pool.preload("zh")
            self.assertIsNone(pool.states[0]["error"])
            self.assertEqual(owner.loads_language, ["en", "zh"])
            self.assertEqual(owner.load_calls, 2)
            self.assertEqual(len(calls), 1)
        finally:
            _lang.reset(original)
        count = 0
        def factory(config):
            nonlocal count
            count += 1
            if count == 1:
                raise ValueError("fake factory failure")
            return ModelDouble()
        another = SharedInferencePool(2, {}, factory)
        self.addCleanup(another.close)
        with self.assertRaisesRegex(InferenceFailure, "fake factory failure"):
            another.preload()
        another.preload()
        self.assertEqual(count, 2)
        self.assertTrue(another.states[0]["loaded"])

    def test_thread_local_languages_do_not_leak_into_peer_or_reused_thread(self):
        pool, owner, _ = self.pool()
        owner.language_barrier = threading.Barrier(2)
        def request(language):
            original = _lang.set("zh" if language == "en" else "en")
            before = _lang.get()
            try:
                result = pool.estimate(language, -1, language)
                return before, _lang.get(), result.language, result.after_language
            finally:
                _lang.reset(original)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(request, ["en", "zh"]))
        self.assertEqual(results, [("zh", "zh", "en", "en"), ("en", "en", "zh", "zh")])

    def test_shutdown_failure_can_be_retried_without_reopening_admission(self):
        pool, owner, _ = self.pool()
        pool.preload()
        owner.unload_failures = 1
        with self.assertRaisesRegex(InferenceFailure, "fake unload failure"):
            pool.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.estimate("rejected")
        pool.close()
        self.assertFalse(owner.loaded)
        self.assertEqual(owner.unloads, 1)

    def test_invalid_limits_and_config_copy(self):
        for value in (0, 9, True, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SharedInferencePool(value, {})
        config = {"cuda_streams": False}
        seen = []
        def factory(snapshot):
            seen.append(snapshot)
            snapshot["factory_only"] = True
            return ModelDouble()
        pool = SharedInferencePool(2, config, factory)
        self.addCleanup(pool.close)
        config["cuda_streams"] = True
        pool.preload()
        self.assertFalse(seen[0]["cuda_streams"])
        self.assertNotIn("factory_only", pool.config)


if __name__ == "__main__":
    unittest.main()
