"""Real-model benchmark; run each strategy in a separate Python process.

No detector/segmentor/FOV model is constructed, and hub access is disabled.
Results and numeric arrays are written only below the ignored backend/outputs.
Comparison mode does not import Torch or load a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _torch_threads(count):
    import torch
    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _memory():
    try:
        import psutil
    except ImportError:
        return {"available": False, "reason": "psutil is not installed"}
    processes = [psutil.Process()]
    try:
        processes += processes[0].children(recursive=True)
    except psutil.Error:
        pass
    rows = []
    for process in processes:
        try:
            memory = process.memory_info()
            private = getattr(memory, "private", None)
            try:
                uss = getattr(process.memory_full_info(), "uss", None)
            except psutil.Error:
                uss = None
            rows.append({"pid": process.pid, "rss_bytes": memory.rss,
                         "private_bytes": private, "uss_bytes": uss})
        except psutil.Error:
            pass
    return {"available": True, "processes": rows,
            "tree_rss_bytes": sum(row["rss_bytes"] for row in rows),
            "tree_private_bytes": sum(row["private_bytes"] for row in rows)
            if all(row["private_bytes"] is not None for row in rows) else None,
            "tree_uss_bytes": sum(row["uss_bytes"] for row in rows)
            if all(row["uss_bytes"] is not None for row in rows) else None}


def _pointers(estimator):
    model = getattr(getattr(estimator, "_estimator", None), "model", None)
    if model is None:
        return {"available": False}
    tensors = [("parameter:" + name, tensor) for name, tensor in model.named_parameters()]
    tensors += [("buffer:" + name, tensor) for name, tensor in model.named_buffers()]
    registered_count = len(tensors)
    for head_name in ("head_pose", "head_pose_hand"):
        hidden = getattr(getattr(model, head_name).mhr, "inner", None)
        if hidden is not None:
            tensors += [(f"{head_name}.mhr.inner.parameter:{name}", tensor) for name, tensor in hidden.named_parameters()]
            tensors += [(f"{head_name}.mhr.inner.buffer:{name}", tensor) for name, tensor in hidden.named_buffers()]
    rows = [{"name": name, "data_ptr": tensor.data_ptr(), "device": str(tensor.device),
             "bytes": tensor.numel() * tensor.element_size()} for name, tensor in tensors]
    storages = {(str(tensor.device), tensor.untyped_storage().data_ptr()): tensor.untyped_storage().nbytes()
                for _, tensor in tensors}
    return {"available": True, "tensor_count": len(rows), "registered_tensor_count": registered_count,
            "pointer_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            "examples": rows[:16], "unique_storage_bytes": sum(storages.values()),
            "unique_storage_count": len(storages),
            "body_hand_mhr_same_object": model.head_pose.mhr is model.head_pose_hand.mhr}


def benchmark_worker_factory(config):
    """Spawn-safe adapter: collect per-process CUDA peaks where they actually live."""
    import torch
    from posegraph.inference_pool import create_worker_estimator
    config = dict(config)
    _torch_threads(config.pop("_bench_cpu_threads"))
    inner = create_worker_estimator(config)

    class InstrumentedEstimator:
        def __init__(self):
            self.load_seconds = None
            self.diagnostics = None

        def __getattr__(self, name):
            return getattr(inner, name)

        def load(self):
            start = time.perf_counter()
            inner.load()
            if inner.active_device == "cuda":
                torch.cuda.synchronize()
            self.load_seconds = time.perf_counter() - start
            self.diagnostics = _pointers(inner)

        def unload(self):
            inner.unload()

        def estimate(self, rgb):
            if inner.active_device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            result = inner.estimate(rgb)
            metadata = {"pid": os.getpid(), "load_seconds": self.load_seconds,
                        "pointers": self.diagnostics}
            if inner.active_device == "cuda":
                torch.cuda.synchronize()
                metadata["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                metadata["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            # EstimateResult is a plain dataclass, including across the process pipe.
            result._bench_worker = metadata
            return result

    return InstrumentedEstimator()


class Sampler:
    def __init__(self, nvidia, interval):
        self.nvidia, self.interval = nvidia, interval
        self.samples = []
        self.phase = "prepare"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            row = {"time": time.perf_counter(), "phase": self.phase, "memory": _memory()}
            if self.nvidia:
                try:
                    output = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                         "--format=csv,noheader,nounits"], text=True, timeout=3)
                    row["nvidia_smi"] = [{"index": int(parts[0]), "used_mb": int(parts[1]),
                                           "total_mb": int(parts[2]), "utilization_percent": int(parts[3])}
                                          for line in output.strip().splitlines()
                                          if len(parts := [part.strip() for part in line.split(",")]) == 4]
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    row["nvidia_smi_error"] = type(error).__name__
            self.samples.append(row)
            self.stop.wait(self.interval)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)


def _output_path(path):
    output = Path(path).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("--output must be below this worktree's ignored backend/outputs directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.with_suffix(".json")


def _images(paths, variants, output):
    import cv2
    import numpy as np
    images, manifest = [], []
    for path in paths:
        path = path.resolve()
        blob = path.read_bytes()
        bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot decode image: {path}")
        transforms = [("original", bgr)]
        if variants:
            h, w = bgr.shape[:2]
            transforms += [("flip", cv2.flip(bgr, 1)),
                           ("crop", bgr[h // 12:h - h // 12, w // 12:w - w // 12].copy())]
        for variant, transformed in transforms:
            rgb = np.ascontiguousarray(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
            index = len(images)
            images.append(rgb)
            generated = None
            if variant != "original":
                generated = output.parent / "inputs" / f"{index:03d}-{path.stem}-{variant}.png"
                generated.parent.mkdir(exist_ok=True)
                ok, encoded = cv2.imencode(".png", transformed)
                if not ok:
                    raise ValueError("Could not encode deterministic input variant")
                generated.write_bytes(encoded.tobytes())
            manifest.append({"image_index": index, "source": str(path), "variant": variant,
                             "source_sha256": hashlib.sha256(blob).hexdigest(),
                             "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                             "shape": list(rgb.shape), "generated": str(generated) if generated else None})
    return images, manifest


def _bundle(path):
    import numpy as np
    path = Path(path).resolve()
    if path.suffix == ".json":
        report = json.loads(path.read_text(encoding="utf-8"))
        arrays = np.load(path.parent / report["npz_file"], allow_pickle=False)
    else:
        arrays = np.load(path, allow_pickle=False)
        report = json.loads(str(arrays["manifest_json"]))
    return report, arrays


def compare(reference, candidate, atol, rtol):
    """Match by input index AND decoded input SHA, never by completion order/hash of outputs."""
    import numpy as np
    ref, ra = _bundle(reference)
    got, ga = _bundle(candidate)
    result = {"reference": str(reference), "candidate": str(candidate), "atol": atol, "rtol": rtol,
              "pass": True, "requests": [], "errors": []}
    ref_inputs = {row["image_index"]: row["rgb_sha256"] for row in ref["inputs"]}
    got_inputs = {row["image_index"]: row["rgb_sha256"] for row in got["inputs"]}
    if ref_inputs != got_inputs:
        result["errors"].append("Input index/SHA mapping differs")
    baseline = {(row["repeat_index"], row["image_index"]): row for row in ref["requests"]}
    candidate_rows = {(row["repeat_index"], row["image_index"]): row for row in got["requests"]}
    if set(baseline) != set(candidate_rows):
        result["errors"].append("Request/repeat keys differ")
    for key in sorted(set(baseline) & set(candidate_rows)):
        left, right = baseline[key], candidate_rows[key]
        row = {"repeat_index": key[0], "image_index": key[1], "pass": True, "arrays": {}}
        if left["humans"] != right["humans"]:
            row["pass"] = False
            row["human_count_mismatch"] = [left["humans"], right["humans"]]
        for human in range(min(left["humans"], right["humans"])):
            prefixes = [f"r{key[0]:03d}_i{key[1]:03d}_h{human:03d}_"]
            prefix = prefixes[0]
            names = {name[len(prefix):] for name in ra.files if name.startswith(prefix)}
            other_names = {name[len(prefix):] for name in ga.files if name.startswith(prefix)}
            if names != other_names:
                row["pass"] = False
                row["missing_fields"] = sorted(names ^ other_names)
            for name in sorted(names & other_names):
                a, b = ra[prefix + name], ga[prefix + name]
                same_shape = a.shape == b.shape
                finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
                passed = bool(same_shape and finite and np.allclose(a, b, atol=atol, rtol=rtol))
                difference = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if same_shape and a.size and finite else None
                row["arrays"][f"h{human}:{name}"] = {"pass": passed, "shape": list(a.shape),
                                                       "candidate_shape": list(b.shape), "max_abs": difference}
                row["pass"] &= passed
        result["requests"].append(row)
        result["pass"] &= row["pass"]
    result["pass"] &= not result["errors"] and bool(result["requests"])
    ra.close()
    ga.close()
    return result


def benchmark(args, output):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.cpu_threads)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    import cv2
    import numpy as np
    import torch
    from posegraph.config import settings
    from posegraph.estimators.sam3d_body import Sam3dBodyEstimator
    _torch_threads(args.cpu_threads)
    cv2.setNumThreads(1)
    config = {"repo_dir": args.repo or settings.sam3d_repo,
              "checkpoint_dir": args.checkpoint or settings.sam3d_checkpoint_dir,
              "device": args.device, "precision": args.precision, "share_mhr": True,
              "mhr_on_cpu": args.mhr_on_cpu, "min_free_vram_mb": 0}
    for path in [Path(config["repo_dir"]), Path(config["checkpoint_dir"]) / "model.ckpt",
                 Path(config["checkpoint_dir"]) / "assets/mhr_model.pt"]:
        if not path.exists():
            raise FileNotFoundError(f"Required local model asset does not exist: {path}")
    paths = args.images or [Path(config["repo_dir"]) / "notebook/images/dancing.jpg"]
    prepared = time.perf_counter()
    images, inputs = _images(paths, args.variants, output)
    report = {"schema_version": 1, "complete": False, "pid": os.getpid(), "strategy": args.strategy,
              "device_requested": args.device, "workers": args.workers, "cpu_threads": args.cpu_threads,
              "mhr_on_cpu": args.mhr_on_cpu, "precision_requested": args.precision,
              "cuda_streams": args.cuda_streams if args.strategy == "shared" else None,
              "warmup_rounds": args.warmup, "repeats": args.repeats, "inputs": inputs,
              "image_prepare_seconds": time.perf_counter() - prepared, "requests": [],
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "local_weights_only": True, "detector_enabled": False,
              "memory_metric_notes": {"rss": "Resident set; summing can double-count shared pages",
                                      "private_bytes": "Windows PrivateUsage (private committed bytes), not private working set",
                                      "uss_bytes": "Unique resident bytes; private working set on Windows"},
              "repo": str(Path(config["repo_dir"]).resolve()),
              "checkpoint_dir": str(Path(config["checkpoint_dir"]).resolve()),
              "npz_file": output.with_suffix(".npz").name}
    sampler = Sampler(args.nvidia_smi, args.sample_interval)
    sampler.thread.start()
    runner, arrays = None, {}
    executor = ThreadPoolExecutor(max_workers=args.workers)
    concurrency_lock = threading.Lock()
    active_calls = peak_calls = 0
    start = time.perf_counter()
    try:
        sampler.phase = "load"
        if args.strategy == "serial":
            runner = Sam3dBodyEstimator(**config)
            runner.load()
            estimate = runner.estimate
        elif args.strategy == "process":
            from posegraph.inference_pool import InferencePool
            runner = InferencePool(args.workers, {**config, "_bench_cpu_threads": args.cpu_threads},
                                   factory=benchmark_worker_factory)
            runner.preload(language="en")
            estimate = lambda rgb: runner.estimate(rgb, release_after=-1, language="en")
        else:
            from posegraph.shared_inference import SharedInferencePool
            runner = SharedInferencePool(args.workers, {**config, "cuda_streams": args.cuda_streams})
            runner.preload(language="en")
            estimate = lambda rgb: runner.estimate(rgb, release_after=-1, language="en")
        if args.device == "cuda" and args.strategy != "process":
            torch.cuda.synchronize()
        report["model_prepare_seconds"] = time.perf_counter() - start
        report["memory_after_load"] = _memory()
        print(json.dumps({"phase": "loaded", "device": args.device, "workers": args.workers}), flush=True)
        sampler.phase = "warmup"
        started = time.perf_counter()
        for _ in range(args.warmup):
            list(executor.map(estimate, [images[i % len(images)] for i in range(max(len(images), args.workers))]))
        report["warmup_seconds"] = time.perf_counter() - started
        print(json.dumps({"phase": "warmup_complete", "device": args.device, "workers": args.workers}), flush=True)
        if args.device == "cuda" and args.strategy != "process":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        sampler.phase = "measure"
        measured_start = time.perf_counter()

        def run(repeat, image_index, queued):
            nonlocal active_calls, peak_calls
            begin = time.perf_counter()
            with concurrency_lock:
                active_calls += 1
                peak_calls = max(peak_calls, active_calls)
            try:
                result = estimate(images[image_index])
            finally:
                with concurrency_lock:
                    active_calls -= 1
            return {"repeat_index": repeat, "image_index": image_index,
                    "latency_ms": (time.perf_counter() - begin) * 1000,
                    "queue_wait_ms": (begin - queued) * 1000, "humans": len(result.humans),
                    "device_actual": result.device, "model_timings_ms": result.timings_ms}, result

        futures = [executor.submit(run, repeat, image_index, time.perf_counter())
                   for repeat in range(args.repeats) for image_index in range(len(images))]
        results = []
        for future in as_completed(futures):
            row, result = future.result()
            results.append((row, result))
            if len(results) % len(images) == 0 or len(results) == len(futures):
                print(json.dumps({"phase": "measure", "device": args.device, "workers": args.workers,
                                  "completed": len(results), "total": len(futures)}), flush=True)
        if args.device == "cuda" and args.strategy != "process":
            torch.cuda.synchronize()
            report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["measured_wall_seconds"] = time.perf_counter() - measured_start
        report["peak_concurrent_calls"] = peak_calls
        # Serialize after the measured interval, so disk compression is not throughput.
        workers = {}
        for row, result in results:
            report["requests"].append(row)
            if hasattr(result, "_bench_worker"):
                metadata = result._bench_worker
                pid = str(metadata["pid"])
                previous = workers.get(pid, {})
                for name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                    if name in metadata:
                        metadata[name] = max(metadata[name], previous.get(name, 0))
                workers[pid] = metadata
            for human_index, human in enumerate(result.humans):
                prefix = f"r{row['repeat_index']:03d}_i{row['image_index']:03d}_h{human_index:03d}_"
                for name in ("bbox", "joint_coords", "joint_global_rots", "keypoints_2d", "keypoints_3d",
                             "cam_t", "rest_joint_coords", "rest_joint_rots"):
                    value = getattr(human, name, None)
                    if value is not None:
                        arrays[prefix + name] = np.asarray(value).copy()
                arrays[prefix + "focal_length"] = np.asarray(human.focal_length)
        report["worker_process_diagnostics"] = workers
        report["requests"].sort(key=lambda row: (row["repeat_index"], row["image_index"]))
        latencies = [row["latency_ms"] for row in report["requests"]]
        report["images_per_second"] = len(latencies) / report["measured_wall_seconds"]
        report["latency_ms"] = {"mean": statistics.mean(latencies), "median": statistics.median(latencies),
                                "p95": float(np.percentile(latencies, 95)), "min": min(latencies), "max": max(latencies)}
        if args.strategy == "serial":
            report["model_diagnostics"] = _pointers(runner)
        elif hasattr(runner, "diagnostics"):
            report["model_diagnostics"] = runner.diagnostics()
        report["states"] = getattr(runner, "states", None)
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        sampler.phase = "cleanup"
        report["memory_before_close"] = _memory()
        try:
            if runner is not None:
                runner.unload() if args.strategy == "serial" else runner.close()
        finally:
            sampler.close()
            report["samples"] = sampler.samples
            available = [sample["memory"] for sample in sampler.samples if sample["memory"].get("available")]
            report["sampled_peak_tree_rss_bytes"] = max((row["tree_rss_bytes"] for row in available), default=None)
            report["sampled_peak_tree_private_bytes"] = max((row["tree_private_bytes"] for row in available
                                                           if row["tree_private_bytes"] is not None), default=None)
            report["sampled_peak_tree_uss_bytes"] = max((row["tree_uss_bytes"] for row in available
                                                       if row["tree_uss_bytes"] is not None), default=None)
            arrays["manifest_json"] = np.asarray(json.dumps(report, ensure_ascii=False, default=str))
            np.savez_compressed(output.with_suffix(".npz"), **arrays)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("serial", "shared", "process"), default="serial")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--images", type=Path, nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--mhr-on-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-streams", action=argparse.BooleanOptionalAction, default=True,
                        help="Shared strategy: independent CUDA streams, or --no-cuda-streams for a default-stream control")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--variants", action="store_true", help="Also test deterministic crop/flip inputs")
    parser.add_argument("--nvidia-smi", action="store_true", help="Sample whole-GPU usage (includes other processes)")
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("REFERENCE", "CANDIDATE"))
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.cpu_threads < 1 or args.sample_interval < 0.05:
        parser.error("repeats/cpu-threads must be positive; warmup >= 0; sample-interval >= 0.05")
    if args.atol < 0 or args.rtol < 0:
        parser.error("comparison tolerances must be nonnegative")
    if args.strategy == "serial" and args.workers != 1 and not args.compare:
        parser.error("The serial reference uses exactly one worker")
    suffix = "-default-stream" if args.strategy == "shared" and not args.cuda_streams else ""
    output = _output_path(args.output or ROOT / "outputs/shared-model" /
                          ("comparison.json" if args.compare else f"{args.strategy}-{args.device}-{args.workers}{suffix}.json"))
    result = compare(*args.compare, args.atol, args.rtol) if args.compare else benchmark(args, output)
    if args.compare:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("strategy", "complete", "images_per_second", "latency_ms", "pass")}
                     | {"output": str(output)}, ensure_ascii=False))
    if args.compare and not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
