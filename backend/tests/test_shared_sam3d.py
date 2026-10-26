"""Small CPU Torch checks; no SAM weights, vendor modification or GPU work."""

import ast
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from posegraph.estimators import shared_sam3d as shared


_barrier = None
_block = None


class TinyBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.request = {"marker": None}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = TinyBranch()
        self.same_branch = self.branch
        self.register_buffer("constant", torch.ones(4))
        self.body_batch_idx = []
        self.hand_batch_idx = []
        self._batch_size = self._max_num_person = self._person_valid = None

    def forward(self, value, marker):
        self._batch_size = marker
        self._max_num_person = marker + 1
        self._person_valid = value > 0
        self.body_batch_idx = [marker]
        self.hand_batch_idx = [-marker]
        self.branch.request["marker"] = marker
        if _barrier is not None:
            _barrier.wait(timeout=5)
        if _block is not None:
            _block[0].set()
            if not _block[1].wait(5):
                raise TimeoutError("Test request was not released")
        return {"marker": self._batch_size, "person": self._max_num_person,
                "body": self.body_batch_idx[:], "hand": self.hand_batch_idx[:],
                "nested": self.branch.request["marker"],
                "values": (self.branch.linear(value) + self.constant).numpy(),
                "inference": torch.is_inference_mode_enabled()}


class TinyRequest:
    def __init__(self, model):
        self.model = model
        self.cfg = None
        self.transform = [{"request": []}]

    def process_one_image(self, image, bboxes=None):
        marker = int(image[0, 0, 0])
        self.prev_prompt = [marker]
        self.is_crop = bboxes is not None
        result = self.model(torch.full((1, 4), float(marker)), marker)
        result.update(prompt=self.prev_prompt[:], crop=self.is_crop)
        return result


class TinyShared(shared.SharedSam3dBodyEstimator):
    def __init__(self):
        super().__init__(Path("unused-vendor"), Path("unused-weights"), device="cpu")
        self.builds = 0

    def _load_template(self):
        self.builds += 1
        time.sleep(0.01)
        torch.manual_seed(42)  # One exclusive load, never inside an inference thread.
        self._estimator = TinyRequest(TinyModel().eval())
        self.active_device = "cpu"
        self.active_precision = "fp32"

    def _prepare_runtime(self, trees):
        self._process_method = TinyRequest.process_one_image
        self._model_methods = {}

    def _convert_result(self, view, outputs, width, height, elapsed):
        return outputs


class ReadOnlyScript(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("factor", torch.ones(4))

    def forward(self, value):
        return value * self.factor


class StatefulScript(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


class BufferWriteScript(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("factor", torch.ones(4))

    def forward(self, value):
        self.factor[:].add_(1)
        return value * self.factor


class LocalCopyScript(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("factor", torch.ones(4))

    def forward(self, value):
        temporary = torch.empty_like(value)
        temporary.copy_(self.factor)
        return value * temporary


class SharedSamTests(unittest.TestCase):
    def setUp(self):
        self.vendor = patch.object(shared, "validate_vendor", return_value={})
        self.vendor.start()
        self.addCleanup(self.vendor.stop)
        self.owner = TinyShared()
        self.addCleanup(self.owner.unload)

    def test_shells_share_exact_weights_buffers_and_module_aliases(self):
        model = TinyModel().eval()
        other = shared.fork_module_shells(model)
        self.assertIsNot(model, other)
        self.assertIsNot(model.branch, other.branch)
        self.assertIs(other.branch, other.same_branch)
        self.assertIsNot(model._modules, other._modules)
        self.assertIsNot(model._parameters, other._parameters)
        self.assertIsNot(model._buffers, other._buffers)
        self.assertIsNot(model.branch.request, other.branch.request)
        for name, tensor in model.named_parameters():
            self.assertIs(dict(other.named_parameters())[name], tensor)
            self.assertEqual(dict(other.named_parameters())[name].data_ptr(), tensor.data_ptr())
        self.assertIs(other.constant, model.constant)
        other.body_batch_idx.append(7)
        other.branch.request["marker"] = 12
        self.assertEqual(model.body_batch_idx, [])
        self.assertIsNone(model.branch.request["marker"])

    def test_multiple_threads_keep_batch_and_estimator_state_separate(self):
        global _barrier
        _barrier = threading.Barrier(3)
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                for markers in ([1, 2, 3], [9, 7, 4]):
                    images = [np.full((3, 4, 3), marker, dtype=np.uint8) for marker in markers]
                    results = list(executor.map(self.owner.estimate, images))
                    for marker, result in zip(markers, results):
                        self.assertEqual((result["marker"], result["person"], result["nested"]), (marker, marker + 1, marker))
                        self.assertEqual((result["body"], result["hand"], result["prompt"]), ([marker], [-marker], [marker]))
                        self.assertTrue(result["inference"])
                        with torch.inference_mode():
                            expected = self.owner._estimator.model.branch.linear(torch.full((1, 4), float(marker))) + 1
                        np.testing.assert_allclose(result["values"], expected.numpy(), atol=0, rtol=0)
            info = self.owner.diagnostics(include_pointers=True)
            self.assertEqual(info["load_count"], 1)
            self.assertEqual(self.owner.builds, 1)
            self.assertEqual(len(info["views"]), 3)
            self.assertTrue(all(v["shared_tensors"] == info["registered_tensors"] for v in info["views"]))
            self.assertTrue(all(v["cuda_stream"] is None for v in info["views"]))
            self.assertIsNone(self.owner._estimator.model._batch_size)
        finally:
            _barrier = None

    def test_unload_waits_for_active_request_and_reload_creates_new_generation(self):
        global _block
        entered, release, unloaded = threading.Event(), threading.Event(), threading.Event()
        _block = (entered, release)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                request = executor.submit(self.owner.estimate, np.ones((2, 2, 3), dtype=np.uint8))
                self.assertTrue(entered.wait(5))
                generation = self.owner.diagnostics()["generation"]
                unloading = executor.submit(lambda: (self.owner.unload(), unloaded.set()))
                self.assertFalse(unloaded.wait(0.05))
                release.set()
                request.result(5)
                unloading.result(5)
            self.assertFalse(self.owner.loaded)
            _block = None
            self.owner.estimate(np.ones((2, 2, 3), dtype=np.uint8))
            self.assertEqual(self.owner.builds, 2)
            self.assertGreater(self.owner.diagnostics()["generation"], generation)
        finally:
            release.set()
            _block = None

    def test_local_device_adapter_never_patches_torch_or_vendor_globals(self):
        source = """class Vendor:
    def process(self, value):
        torch.cuda.empty_cache()
        batch = recursive_to(value, "cuda")
        return batch.cuda()
"""
        original_cuda = torch.Tensor.cuda
        original_cache = torch.cuda.empty_cache
        recursive = lambda value, device: value.to(device)
        namespace = {"torch": torch, "recursive_to": recursive}
        exec(source, namespace)
        vendor = namespace["Vendor"]
        adapted = shared.compile_device_method(vendor.process, ast.parse(source), "Vendor", "process", torch.device("cpu"), (1, 1, 1))
        value = torch.ones(3)
        self.assertIs(adapted(vendor(), value), value)
        self.assertIs(torch.Tensor.cuda, original_cuda)
        self.assertIs(torch.cuda.empty_cache, original_cache)
        self.assertIs(namespace["recursive_to"], recursive)
        with self.assertRaises(shared.UnsupportedSharedRuntime):
            shared.compile_device_method(vendor.process, ast.parse(source), "Vendor", "process", torch.device("cpu"), (0, 0, 0))

    def test_torchscript_shares_readonly_leaf_and_rejects_state_assignment(self):
        scripted = torch.jit.script(ReadOnlyScript().eval())
        shared._check_script_readonly(scripted)
        model = TinyModel().eval()
        model.scripted = scripted
        other = shared.fork_module_shells(model)
        self.assertIs(other.scripted, scripted)
        with self.assertRaises(shared.UnsupportedSharedRuntime):
            shared._check_script_readonly(torch.jit.script(StatefulScript().eval()))
        with self.assertRaises(shared.UnsupportedSharedRuntime):
            shared._check_script_readonly(torch.jit.script(BufferWriteScript().eval()))
        shared._check_script_readonly(torch.jit.script(LocalCopyScript().eval()))

    def test_cpu_mhr_hidden_script_and_all_registered_storage_remain_shared(self):
        from posegraph.estimators.sam3d_body import _cpu_mhr
        scripted = torch.jit.script(ReadOnlyScript().eval())
        model = TinyModel().eval()
        model.external_mhr = _cpu_mhr(scripted).eval()
        model.other_mhr = model.external_mhr
        other = shared.fork_module_shells(model)
        self.assertIsNot(other.external_mhr, model.external_mhr)
        self.assertIs(other.external_mhr.inner, scripted)
        self.assertIs(other.external_mhr, other.other_mhr)
        source = shared._tensor_inventory(model)
        actual = shared._tensor_inventory(other)
        self.assertEqual(source.keys(), actual.keys())
        self.assertTrue(all(actual[name] is tensor for name, tensor in source.items()))

    def test_failed_stream_synchronization_still_releases_active_lease(self):
        class FailedStream:
            cuda_stream = 123
            def synchronize(self):
                raise RuntimeError("test stream failure")
        def convert(view, outputs, width, height, elapsed):
            view.stream = FailedStream()
            return outputs
        original_convert = self.owner._convert_result
        self.owner._convert_result = convert
        with self.assertRaisesRegex(RuntimeError, "test stream failure"):
            self.owner.estimate(np.ones((2, 2, 3), dtype=np.uint8))
        self.assertEqual(self.owner._active, 0)
        self.assertFalse(self.owner.diagnostics()["usable"])
        self.assertIn("test stream failure", self.owner.diagnostics()["stream_error"])
        with self.assertRaisesRegex(shared.UnsupportedSharedRuntime, "unload before reloading"):
            self.owner.estimate(np.ones((2, 2, 3), dtype=np.uint8))
        self.assertEqual(self.owner.builds, 1)
        self.owner.unload()
        self.owner._convert_result = original_convert
        self.owner.estimate(np.ones((2, 2, 3), dtype=np.uint8))
        self.assertEqual(self.owner.builds, 2)
        self.assertTrue(self.owner.diagnostics()["usable"])

    def test_registered_tensor_mutation_is_rejected(self):
        original = TinyModel.forward
        def mutating(model, *args):
            result = original(model, *args)
            model.constant.add_(1)
            return result
        with patch.object(TinyModel, "forward", mutating):
            with self.assertRaisesRegex(shared.UnsupportedSharedRuntime, "mutated"):
                self.owner.estimate(np.ones((2, 2, 3), dtype=np.uint8))
        self.assertEqual(self.owner._active, 0)

    def test_training_or_hooks_fail_instead_of_sharing_mutable_state(self):
        with self.assertRaises(shared.UnsupportedSharedRuntime):
            shared.fork_module_shells(TinyModel())
        model = TinyModel().eval()
        handle = model.register_forward_hook(lambda *args: None)
        try:
            with self.assertRaises(shared.UnsupportedSharedRuntime):
                shared.fork_module_shells(model)
        finally:
            handle.remove()

    def test_factory_is_lazy_and_rejects_unknown_options(self):
        owner = shared.create_shared_estimator({"repo_dir": "vendor", "checkpoint_dir": "weights", "device": "cpu", "cuda_streams": False})
        self.assertFalse(owner.loaded)
        self.assertEqual(owner.device, "cpu")
        self.assertFalse(owner.cuda_streams)
        with self.assertRaises(ValueError):
            shared.create_shared_estimator({"unknown": True})


if __name__ == "__main__":
    unittest.main()
