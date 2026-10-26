"""Experimental shared-weight SAM views with isolated request state and streams.

Only module shells and preprocessing objects are copied. Registered parameters,
buffers and audited read-only TorchScript MHR modules retain their exact storage.
The reviewed upstream sources are pinned by normalized AST fingerprints: an
upstream implementation change must be reviewed instead of silently patched.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import threading
import time
import types
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .base import EstimateResult
from .sam3d_body import Sam3dBodyEstimator


class UnsupportedSharedRuntime(RuntimeError):
    pass


_REVIEWED_AST = {
    "sam_3d_body/sam_3d_body_estimator.py": "ccb14a9bac4f6516fc361974073129a5ad172f8150f5e44649cae01e77164f07",
    "sam_3d_body/models/meta_arch/sam3d_body.py": "73f126c88805e11beb93ffd67cc4584a298dd4ffce6c63e64307e8e52735a3a9",
    "sam_3d_body/models/meta_arch/base_model.py": "78a69546f37df5ea456a1a4e0149537fb0cbdfa2064ccb418f9e91ecc3bc3bab",
    "sam_3d_body/models/heads/mhr_head.py": "4b81ee28e56931b971f3c401befb39077f115987b248052443f18741ec4f51d5",
}


def validate_vendor(repo: Path) -> dict[str, ast.Module]:
    result = {}
    for relative, expected in _REVIEWED_AST.items():
        path = repo / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        actual = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        if actual != expected:
            raise UnsupportedSharedRuntime(f"Shared SAM requires review of changed upstream source: {relative} ({actual})")
        result[relative] = tree
    return result


class _DeviceCalls(ast.NodeTransformer):
    def __init__(self):
        self.cuda = self.recursive = self.empty_cache = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            if node.args or node.keywords:
                raise UnsupportedSharedRuntime("Unexpected arguments to upstream Tensor.cuda")
            self.cuda += 1
            return ast.copy_location(ast.Call(ast.Attribute(node.func.value, "to", ast.Load()),
                                              [ast.Name("_posegraph_device", ast.Load())], []), node)
        if (isinstance(node.func, ast.Name) and node.func.id == "recursive_to" and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == "cuda"):
            self.recursive += 1
            node.args[1] = ast.Name("_posegraph_device", ast.Load())
        return node

    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "empty_cache"):
            if ast.unparse(call.func) != "torch.cuda.empty_cache" or call.args or call.keywords:
                raise UnsupportedSharedRuntime("Unexpected upstream cache operation")
            self.empty_cache += 1
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)


def compile_device_method(original, tree: ast.Module, class_name: str, method_name: str,
                          device: torch.device, expected: tuple[int, int, int]):
    """Compile a private function copy; the vendor class/module are never changed."""
    original = inspect.unwrap(original)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name), None) if cls else None
    if method is None or original.__code__.co_freevars:
        raise UnsupportedSharedRuntime(f"Unsupported upstream method: {class_name}.{method_name}")
    method = copy.deepcopy(method)
    method.decorator_list = []  # inference_mode is entered separately in each request thread.
    transform = _DeviceCalls()
    method = transform.visit(method)
    counts = (transform.cuda, transform.recursive, transform.empty_cache)
    if counts != expected:
        raise UnsupportedSharedRuntime(f"Changed device call contract for {method_name}: {counts} != {expected}")
    module = ast.fix_missing_locations(ast.Module([method], type_ignores=[]))
    namespace = dict(original.__globals__)
    namespace.update(_posegraph_device=device, print=lambda *args, **kwargs: None)
    exec(compile(module, original.__code__.co_filename + " [Posegraph private device adapter]", "exec"), namespace)
    return namespace[method_name]


def _is_scripted(module) -> bool:
    return isinstance(module, torch.jit.ScriptModule)


def _module_graph(model) -> list[torch.nn.Module]:
    modules, seen = [], set()
    def visit(module):
        if id(module) in seen:
            return
        seen.add(id(module))
        modules.append(module)
        if not _is_scripted(module):
            for child in module._modules.values():
                if child is not None:
                    visit(child)
            # CpuMhr.inner deliberately is not registered in _modules.
            for value in module.__dict__.values():
                if isinstance(value, torch.nn.Module):
                    visit(value)
    visit(model)
    return modules


def fork_module_shells(model: torch.nn.Module) -> torch.nn.Module:
    """Recursively isolate mutable Python state without copying registered tensors."""
    modules = _module_graph(model)
    memo = {}
    for module in modules:
        memo[id(module)] = module if _is_scripted(module) else copy.copy(module)
        for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
            memo[id(tensor)] = tensor
    cfg = getattr(model, "cfg", None)
    if cfg is not None and hasattr(cfg, "is_frozen"):
        if not cfg.is_frozen():
            raise UnsupportedSharedRuntime("Shared SAM requires a frozen upstream config")
        memo[id(cfg)] = cfg
    for module in modules:
        if _is_scripted(module):
            continue
        if module.training:
            raise UnsupportedSharedRuntime("Shared SAM only supports eval-mode modules")
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise UnsupportedSharedRuntime("Shared SAM does not support mutable module hooks")
        shell = memo[id(module)]
        attributes = {}
        for name, value in module.__dict__.items():
            if name in ("_parameters", "_buffers"):
                attributes[name] = value.copy()
            elif name == "_modules":
                attributes[name] = {key: memo[id(child)] if child is not None else None for key, child in value.items()}
            else:
                attributes[name] = copy.deepcopy(value, memo)
        shell.__dict__ = attributes
    return memo[id(model)]


def _check_script_readonly(module) -> None:
    """Reject persistent writes in the inlined MHR graph, including attribute views."""
    aliases = set()
    alias_ops = {"aten::view", "aten::reshape", "aten::slice", "aten::select", "aten::transpose", "aten::permute",
                 "aten::flatten", "aten::unflatten", "aten::detach", "aten::alias", "aten::expand", "aten::to",
                 "prim::ListConstruct", "prim::ListUnpack", "prim::TupleConstruct", "prim::TupleUnpack"}
    def walk(block):
        for node in block.nodes():
            kind = node.kind()
            if kind in ("prim::SetAttr", "prim::PythonOp"):
                raise UnsupportedSharedRuntime(f"MHR has unsupported persistent state operation: {kind}")
            inputs = list(node.inputs())
            depends = any(v.unique() in aliases for v in inputs)
            if kind == "prim::GetAttr" or (depends and kind in alias_ops):
                aliases.update(v.unique() for v in node.outputs())
            try:
                schema = node.schema()
            except RuntimeError:
                schema = ""
            if depends and "!" in schema:
                parsed = torch._C.parse_schema(schema)
                if any(argument.alias_info is not None and argument.alias_info.is_write
                       and index < len(inputs) and inputs[index].unique() in aliases
                       for index, argument in enumerate(parsed.arguments)):
                    raise UnsupportedSharedRuntime(f"MHR writes to an attribute or an attribute view: {kind}")
            for child in node.blocks():
                walk(child)
    walk(module.forward.inlined_graph)


def _tensor_inventory(model) -> dict[str, torch.Tensor]:
    result = dict(model.named_parameters())
    result.update({"buffer:" + name: value for name, value in model.named_buffers()})
    for index, module in enumerate(_module_graph(model)):
        if _is_scripted(module):
            result.update({f"script:{index}:" + name: value for name, value in module.named_parameters()})
            result.update({f"script:{index}:buffer:" + name: value for name, value in module.named_buffers()})
    return result


@dataclass
class _RequestView:
    generation: int
    request: object
    adapter: Sam3dBodyEstimator
    stream: object | None
    shared_tensors: int


class SharedSam3dBodyEstimator(Sam3dBodyEstimator):
    """One weight owner; each calling thread owns a shell, preprocessing and stream."""

    name = "sam3d_body_shared_experiment"

    def __init__(self, *args, cuda_streams: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_streams = cuda_streams
        self._condition = threading.Condition(threading.RLock())
        self._active = 0
        self._changing = False
        self._generation = 0
        self._local = threading.local()
        self._views: dict[int, _RequestView] = {}
        self._tensors = {}
        self._versions = {}
        self._load_count = 0
        self._sync_failure: str | None = None

    def _load_template(self):
        super().load()

    def _prepare_runtime(self, trees):
        request = self._estimator
        if any(getattr(request, name, None) is not None for name in ("detector", "sam", "fov_estimator")):
            raise UnsupportedSharedRuntime("Shared SAM supports only the local body model without optional detectors")
        device = torch.device(self.active_device)
        self._process_method = compile_device_method(type(request).process_one_image,
            trees["sam_3d_body/sam_3d_body_estimator.py"], "SAM3DBodyEstimator", "process_one_image", device, (0, 1, 1))
        model_type = type(request.model)
        self._model_methods = {name: compile_device_method(getattr(model_type, name),
            trees["sam_3d_body/models/meta_arch/sam3d_body.py"], "SAM3DBody", name, device, counts)
            for name, counts in (("get_ray_condition", (1, 0, 0)), ("run_inference", (4, 2, 0)))}
        for module in _module_graph(request.model):
            if _is_scripted(module):
                _check_script_readonly(module)

    def load(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: not self._changing)
            if self.loaded:
                if self._sync_failure is not None:
                    raise UnsupportedSharedRuntime("Shared CUDA stream failed; unload before reloading: " + self._sync_failure)
                return
            self._changing = True
            try:
                trees = validate_vendor(self.repo_dir)
                # Keep version counters available even if a caller entered its
                # own inference_mode before asking this lazy owner to load.
                with torch.inference_mode(False):
                    self._load_template()
                self._prepare_runtime(trees)
                if self.active_device == "cuda":
                    torch.cuda.synchronize()
                self._tensors = _tensor_inventory(self._estimator.model)
                self._versions = {name: value._version for name, value in self._tensors.items()}
                self._generation += 1
                self._load_count += 1
                self._sync_failure = None
            except BaseException:
                super().unload()
                raise
            finally:
                self._changing = False
                self._condition.notify_all()

    def _view(self) -> _RequestView:
        current = getattr(self._local, "view", None)
        if current is not None and current.generation == self._generation:
            return current
        template = self._estimator
        stream = (torch.cuda.Stream() if self.cuda_streams else torch.cuda.default_stream()) if self.active_device == "cuda" else None
        scope = torch.cuda.stream(stream) if stream is not None else nullcontext()
        try:
            # Any small, unregistered request tensors copied with Python state
            # must be created on this same stream, before its first inference.
            with scope:
                model = fork_module_shells(template.model)
                for name, method in self._model_methods.items():
                    object.__setattr__(model, name, types.MethodType(method, model))
                request = copy.copy(template)
                request.__dict__ = {name: (model if name == "model" else template.cfg if name == "cfg"
                                          else copy.deepcopy(value)) for name, value in template.__dict__.items()}
                request.process_one_image = types.MethodType(self._process_method, request)
                actual = _tensor_inventory(model)
                if actual.keys() != self._tensors.keys() or any(actual[name] is not value for name, value in self._tensors.items()):
                    raise UnsupportedSharedRuntime("A request shell unexpectedly copied or replaced registered weight/buffer storage")
                adapter = object.__new__(Sam3dBodyEstimator)
                adapter._estimator = request
                adapter.active_device = self.active_device
        except BaseException:
            if stream is not None:
                self._synchronize_stream(stream)
            raise
        current = _RequestView(self._generation, request, adapter, stream, len(actual))
        self._local.view = current
        with self._condition:
            self._views[threading.get_ident()] = current
        return current

    def _convert_result(self, view, outputs, width, height, elapsed):
        return view.adapter._make_result(outputs, width, height, elapsed)

    def _synchronize_stream(self, stream):
        try:
            stream.synchronize()
        except BaseException as error:
            with self._condition:
                self._sync_failure = f"{type(error).__name__}: {error}"
            raise

    def estimate(self, image_rgb: np.ndarray, bboxes: np.ndarray | None = None) -> EstimateResult:
        while True:
            self.load()
            with self._condition:
                self._condition.wait_for(lambda: not self._changing)
                if self._sync_failure is not None:
                    raise UnsupportedSharedRuntime("Shared CUDA stream failed; unload before reloading: " + self._sync_failure)
                if self.loaded:
                    self._active += 1
                    break
        view = None
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                view = self._view()
                scope = torch.cuda.stream(view.stream) if view.stream is not None else nullcontext()
                with scope:
                    outputs = view.request.process_one_image(image_rgb, bboxes=bboxes)
                    result = self._convert_result(view, outputs, image_rgb.shape[1], image_rgb.shape[0],
                                                  (time.perf_counter() - started) * 1000)
                if any(value._version != self._versions[name] for name, value in self._tensors.items()):
                    raise UnsupportedSharedRuntime("Shared inference mutated a registered parameter or buffer")
                return result
        finally:
            # Even exceptions must finish their stream before unload/reuse.
            try:
                if view is not None and view.stream is not None:
                    self._synchronize_stream(view.stream)
            finally:
                with self._condition:
                    self._active -= 1
                    self._condition.notify_all()

    def unload(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: not self._changing)
            self._changing = True
            try:
                self._condition.wait_for(lambda: self._active == 0)
                self._views.clear()
                self._local = threading.local()
                self._tensors.clear()
                self._versions.clear()
                self._generation += 1
                super().unload()
                self._sync_failure = None
            finally:
                self._changing = False
                self._condition.notify_all()

    def diagnostics(self, include_pointers: bool = False) -> dict:
        with self._condition:
            pointers = {name: value.data_ptr() for name, value in self._tensors.items()}
            unique_storage = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                              for t in self._tensors.values()}
            views = []
            for thread, view in self._views.items():
                mhr = view.request.model.head_pose.mhr if hasattr(view.request.model, "head_pose") else None
                views.append({"thread": thread, "generation": view.generation, "shared_tensors": view.shared_tensors,
                              "cuda_stream": view.stream.cuda_stream if view.stream is not None else None,
                              "mhr_identity": id(getattr(mhr, "inner", mhr)) if mhr is not None else None,
                              "mhr_wrapper_identity": id(mhr) if mhr is not None else None})
            device_bytes = {}
            for (device, _), size in unique_storage.items():
                device_bytes[device] = device_bytes.get(device, 0) + size
            return {"loaded": self.loaded, "load_count": self._load_count, "active": self._active,
                    "usable": self.loaded and self._sync_failure is None, "stream_error": self._sync_failure,
                    "cuda_streams_enabled": self.cuda_streams,
                    "generation": self._generation, "registered_tensors": len(pointers),
                    "unique_storage_bytes": sum(unique_storage.values()),
                    "storage_bytes_by_device": device_bytes,
                    "pointer_digest": hashlib.sha256(repr(sorted(pointers.items())).encode()).hexdigest(),
                    "pointers": pointers if include_pointers else dict(list(pointers.items())[:8]),
                    "views": views, "script_state_audit": "no attribute assignment or writes to attribute aliases"}


def create_shared_estimator(config: dict | None = None) -> SharedSam3dBodyEstimator:
    from ..config import settings
    options = {"repo_dir": settings.sam3d_repo, "checkpoint_dir": settings.sam3d_checkpoint_dir,
               "device": settings.device, "precision": settings.precision, "mhr_on_cpu": settings.mhr_on_cpu,
               "min_free_vram_mb": settings.min_free_vram_mb}
    supplied = dict(config or {})
    allowed = set(options) | {"share_mhr", "cuda_streams"}
    if supplied.keys() - allowed:
        raise ValueError(f"Unknown shared-estimator configuration: {sorted(supplied.keys() - allowed)}")
    options.update(supplied)
    return SharedSam3dBodyEstimator(**options)
