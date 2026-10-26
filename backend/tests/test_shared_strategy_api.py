"""Opt-in backend strategy contracts, without loading model weights or Torch."""
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from posegraph import app as service
from posegraph.inference_pool import InferenceFailure, ModelLifecycle


class FakeEstimator:
    name, device, precision = "fake", "cpu", "fp32"
    share_mhr, mhr_on_cpu, min_free_vram_mb = True, True, 3000
    loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def estimate(self, rgb):
        return SimpleNamespace(humans=[1])


class FakePool:
    def __init__(self, parallelism, config):
        self.parallelism, self.config = parallelism, dict(config)
        self.loaded, self.closed = False, False
        self.estimates = 0
        self.release_policies = []

    @property
    def states(self):
        return [{"loaded": self.loaded, "loading": False, "error": None,
                 "device": "cpu", "precision": "fp32", "device_reason": ("",)}]

    def preload(self, language):
        self.loaded = True

    def estimate(self, rgb, release_after, language):
        self.estimates += 1
        self.release_policies.append(release_after)
        self.loaded = True
        return SimpleNamespace(humans=[1])

    def close(self):
        self.closed, self.loaded = True, False


class SharedStrategyApiTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for key, value in {
            "_estimator": FakeEstimator(), "_parallel_pool": None,
            "_model_lifecycle": ModelLifecycle(), "_recognition_parallelism": 1,
            "_recognition_strategy": "process", "_release_after": -1,
            "_held": False, "_active_pool_requests": 0,
            "_model_state": {"loading": False, "error": None},
        }.items():
            self.stack.enter_context(patch.object(service, key, value))
        self.factory = self.stack.enter_context(patch.object(service, "SharedInferencePool", side_effect=FakePool))
        self.addCleanup(service._unload_model)

    def test_default_process_serial_path_remains_unchanged(self):
        service._load_model()
        self.assertTrue(service._estimator.loaded)
        self.factory.assert_not_called()
        self.assertEqual(service._model_info()["config"]["recognitionStrategy"], "process")

    def test_shared_one_slot_uses_shared_adapter_and_reports_one_copy(self):
        service.model_config(service.ModelConfig(recognition_strategy="shared"))
        service._load_model()
        pool = service._parallel_pool
        self.assertIsInstance(pool, FakePool)
        self.assertEqual(pool.parallelism, 1)
        self.assertFalse(service._estimator.loaded)
        self.assertIn("cuda_streams", pool.config)
        self.assertEqual(service._model_info()["modelCopies"], 1)
        with patch.object(service, "_decode", return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
            service._estimate(b"image")
        self.assertEqual(pool.estimates, 1)

    def test_plugin_partial_updates_preserve_explicit_shared_strategy(self):
        with TestClient(service.app) as client:
            response = client.post("/model/config", json={"recognition_strategy": "shared"})
            self.assertEqual(response.status_code, 200)
            service._load_model()
            previous = service._parallel_pool
            # Existing plugin requests do not include recognition_strategy.
            for payload in ({"recognition_parallelism": 3},
                            {"device": "cuda", "release_after": 60, "mhr_on_cpu": False}):
                response = client.post("/model/config", json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["model"]["config"]["recognitionStrategy"], "shared")
            self.assertTrue(previous.closed)
            config = response.json()["model"]["config"]
            self.assertEqual(config["recognitionParallelism"], 3)
            self.assertEqual(config["device"], "cuda")
            self.assertEqual(config["releaseAfter"], 60)
            self.assertFalse(config["mhrOnCpu"])

    def test_shared_capacity_is_not_reported_as_multiple_loaded_models(self):
        service.model_config(service.ModelConfig(recognition_strategy="shared", recognition_parallelism=3))
        self.assertEqual(service._model_info()["modelCopies"], 0)
        service._load_model()
        info = service._model_info()
        self.assertEqual(info["modelCopies"], 1)
        self.assertEqual(len(info["workers"]), 1)
        self.assertEqual(info["config"]["recognitionParallelism"], 3)

    def test_switching_back_closes_shared_owner_before_serial_load(self):
        service.model_config(service.ModelConfig(recognition_strategy="shared"))
        service._load_model()
        old = service._parallel_pool
        service.model_config(service.ModelConfig(recognition_strategy="process"))
        self.assertTrue(old.closed)
        self.assertIsNone(service._parallel_pool)
        service._load_model()
        self.assertTrue(service._estimator.loaded)

    def test_failed_close_preserves_config_and_same_request_retries_shutdown(self):
        class FlakyPool(FakePool):
            closes = 0
            def close(self):
                self.closes += 1
                self.closed = True  # Real shared shutdown also rejects new leases before unloading.
                if self.closes == 1:
                    raise InferenceFailure(500, "first shutdown failed")
                self.loaded = False
        self.factory.side_effect = FlakyPool
        service.model_config(service.ModelConfig(recognition_strategy="shared", recognition_parallelism=3))
        service._load_model()
        original = service._parallel_pool
        update = service.ModelConfig(recognition_strategy="process", recognition_parallelism=1,
                                     device="cuda", mhr_on_cpu=False, min_free_vram_mb=4000, release_after=60)
        with self.assertRaisesRegex(InferenceFailure, "first shutdown failed"):
            service.model_config(update)
        self.assertIs(service._parallel_pool, original)
        self.assertEqual(service._recognition_strategy, "shared")
        self.assertEqual(service._recognition_parallelism, 3)
        self.assertEqual(service._estimator.device, "cpu")
        self.assertTrue(service._estimator.mhr_on_cpu)
        self.assertEqual(service._estimator.min_free_vram_mb, 3000)
        self.assertEqual(service._release_after, -1)
        # Repeating the same JSON request must not be mistaken for an already
        # applied update while the retained original pool is closing.
        response = service.model_config(update)
        self.assertEqual(original.closes, 2)
        self.assertIsNone(service._parallel_pool)
        self.assertEqual(response["model"]["config"]["recognitionStrategy"], "process")
        self.assertEqual(service._recognition_parallelism, 1)
        self.assertEqual(service._estimator.device, "cuda")
        self.assertFalse(service._estimator.mhr_on_cpu)
        self.assertEqual(service._estimator.min_free_vram_mb, 4000)
        self.assertEqual(service._release_after, 60)
        service._load_model()
        self.assertTrue(service._estimator.loaded)

    def test_successful_pool_retry_clears_only_stale_global_preload_error(self):
        class RecoverablePool(FakePool):
            worker_error = None
            def preload(self, language):
                raise InferenceFailure(503, "fake preload failure")
            @property
            def states(self):
                states = super().states
                states[0]["error"] = self.worker_error
                return states
        self.factory.side_effect = RecoverablePool
        service.model_config(service.ModelConfig(recognition_strategy="shared"))
        with self.assertRaisesRegex(InferenceFailure, "fake preload failure"):
            service._load_model()
        self.assertIn("fake preload failure", service._model_info()["error"])
        with patch.object(service, "_decode", return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
            service._estimate(b"image")
        self.assertTrue(service._model_info()["loaded"])
        self.assertIsNone(service._model_info()["error"])
        # Another slot's real error remains visible even if this request succeeds.
        pool = service._parallel_pool
        pool.worker_error = "another worker failed"
        service._model_state["error"] = "older preload failure"
        with patch.object(service, "_decode", return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
            service._estimate(b"image")
        self.assertIsNone(service._model_state["error"])
        self.assertEqual(service._model_info()["error"], "another worker failed")

    def test_invalid_strategy_is_rejected_without_creating_a_model(self):
        with TestClient(service.app) as client:
            response = client.post("/model/config", json={"recognition_strategy": "unknown"})
        self.assertEqual(response.status_code, 422)
        self.factory.assert_not_called()
        self.assertFalse(service._estimator.loaded)

    def test_serial_hold_retains_model_until_nonheld_request(self):
        service._release_after = 0
        with patch.object(service, "_decode", return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
            service._estimate(b"image", hold=True)
            self.assertTrue(service._estimator.loaded)
            self.assertTrue(service._held)
            service._estimate(b"image", hold=False)
        self.assertFalse(service._estimator.loaded)
        self.assertFalse(service._held)

    def test_parallel_hold_protects_nonheld_peers_and_next_group_can_release(self):
        for strategy in ("process", "shared"):
            with self.subTest(strategy=strategy):
                entered = [threading.Event(), threading.Event()]
                release = [threading.Event(), threading.Event()]
                class BlockingPool(FakePool):
                    def estimate(self, rgb, release_after, language):
                        result = super().estimate(rgb, release_after, language)
                        if rgb < 2:
                            entered[rgb].set()
                            if not release[rgb].wait(3):
                                raise RuntimeError("test did not release inference")
                        return result
                service.model_config(service.ModelConfig(recognition_strategy=strategy,
                                                         recognition_parallelism=2, release_after=0))
                with patch.object(service, "InferencePool", side_effect=BlockingPool), \
                        patch.object(service, "SharedInferencePool", side_effect=BlockingPool), \
                        patch.object(service, "_decode", side_effect=lambda data: data), \
                        ThreadPoolExecutor(max_workers=2) as executor:
                    try:
                        first = executor.submit(service._estimate, 0, False)
                        self.assertTrue(entered[0].wait(3))
                        second = executor.submit(service._estimate, 1, True)
                        self.assertTrue(entered[1].wait(3))
                        pool = service._parallel_pool
                        release[0].set()
                        first.result(3)
                        self.assertFalse(pool.closed)
                        release[1].set()
                        second.result(3)
                        self.assertFalse(pool.closed)
                        self.assertTrue(service._held)
                        self.assertEqual(service._active_pool_requests, 0)
                        service._estimate(2, False)
                        self.assertTrue(pool.closed)
                        self.assertFalse(service._held)
                        self.assertEqual(pool.release_policies, [-1, -1, -1])
                    finally:
                        for event in release:
                            event.set()

    def test_hold_idle_timeout_releases_serial_and_parallel_models(self):
        for strategy, parallelism in (("process", 1), ("shared", 2)):
            with self.subTest(strategy=strategy):
                service.model_config(service.ModelConfig(recognition_strategy=strategy,
                                                         recognition_parallelism=parallelism, release_after=0))
                with patch.object(service, "_decode", return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
                    service._estimate(b"image", hold=True)
                self.assertTrue(service._model_info()["loaded"])
                with patch.object(service.time, "monotonic", return_value=service._last_used + service.HOLD_TIMEOUT - 1):
                    service._release_idle_models()
                self.assertTrue(service._model_info()["loaded"])
                with patch.object(service.time, "monotonic", return_value=service._last_used + service.HOLD_TIMEOUT + 1):
                    service._release_idle_models()
                self.assertFalse(service._model_info()["loaded"])
                self.assertFalse(service._held)

    def test_hold_does_not_bypass_exclusive_unload_or_configuration(self):
        for action in (service._unload_model,
                       lambda: service.model_config(service.ModelConfig(device="cuda"))):
            with self.subTest(action=action):
                entered, release, writer_started = threading.Event(), threading.Event(), threading.Event()
                class BlockingPool(FakePool):
                    def estimate(self, rgb, release_after, language):
                        result = super().estimate(rgb, release_after, language)
                        entered.set()
                        if not release.wait(3):
                            raise RuntimeError("test did not release inference")
                        return result
                def change():
                    writer_started.set()
                    return action()
                service.model_config(service.ModelConfig(recognition_strategy="shared", device="cpu", release_after=0))
                with patch.object(service, "SharedInferencePool", side_effect=BlockingPool), \
                        patch.object(service, "_decode", return_value=np.zeros((2, 2, 3), dtype=np.uint8)), \
                        ThreadPoolExecutor(max_workers=2) as executor:
                    try:
                        recognition = executor.submit(service._estimate, b"image", True)
                        self.assertTrue(entered.wait(3))
                        pool = service._parallel_pool
                        writer = executor.submit(change)
                        self.assertTrue(writer_started.wait(3))
                        self.assertFalse(release.wait(.05))
                        self.assertFalse(writer.done())
                        self.assertFalse(pool.closed)
                        release.set()
                        recognition.result(3)
                        writer.result(3)
                        self.assertTrue(pool.closed)
                        self.assertFalse(service._held)
                    finally:
                        release.set()

    def test_create_session_forwards_hold_query(self):
        with patch.object(service, "_estimate", side_effect=service.HTTPException(422, "fake result")) as estimate, \
                TestClient(service.app) as client:
            response = client.post("/sessions?hold=true", content=b"image")
        self.assertEqual(response.status_code, 422)
        estimate.assert_called_once_with(b"image", True)

    def test_multiview_writeback_uses_current_cache_and_keeps_delete_final(self):
        from posegraph.session_cache import Session, SessionCache
        cache = SessionCache(1)
        self.addCleanup(cache.close)
        original, auxiliary, fused = {"pose": "base"}, {"pose": "aux"}, {"pose": "fused"}
        cache["base"] = Session("base", "base", (2, 2), b"png", [original], 1)
        cache["aux"] = Session("aux", "aux", (2, 2), b"png", [auxiliary], 1)
        views = [{"kind": "back"}]
        with patch.object(service, "_sessions", cache), \
                patch("posegraph.multiview.describe", return_value=views), \
                patch("posegraph.multiview.fuse", return_value=fused):
            # Reading aux spills the main session while fusion still holds its old object.
            self.assertEqual(service.session_views("base", service.ViewsRequest(aux=["aux"])), {"views": views})
            self.assertEqual(cache["base"].fused, fused)
            self.assertEqual(service._human(cache["base"], 0), fused)
            self.assertEqual(service._human(cache["base"], 0, use_aux=False), original)
            self.assertEqual(service.session_views("base", service.ViewsRequest()), {"views": []})
            self.assertIsNone(cache["base"].fused)
            def delete_during_fusion(*args):
                cache.discard("base")
                return fused
            with patch("posegraph.multiview.fuse", side_effect=delete_during_fusion):
                with self.assertRaises(service.HTTPException) as failure:
                    service.session_views("base", service.ViewsRequest(aux=["aux"]))
            self.assertEqual(failure.exception.status_code, 404)
            self.assertNotIn("base", cache)


if __name__ == "__main__":
    unittest.main()
