"""Opt-in backend strategy contracts, without loading model weights or Torch."""
import sys
import unittest
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


class FakePool:
    def __init__(self, parallelism, config):
        self.parallelism, self.config = parallelism, dict(config)
        self.loaded, self.closed = False, False
        self.estimates = 0

    @property
    def states(self):
        return [{"loaded": self.loaded, "loading": False, "error": None,
                 "device": "cpu", "precision": "fp32", "device_reason": ("",)}]

    def preload(self, language):
        self.loaded = True

    def estimate(self, rgb, release_after, language):
        self.estimates += 1
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


if __name__ == "__main__":
    unittest.main()
