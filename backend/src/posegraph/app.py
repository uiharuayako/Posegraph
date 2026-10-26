"""本地识别服务（FastAPI），供 Posegraph 插件界面调用。

运行：python -m posegraph.app

接口：
  GET  /health                         服务、模型、显存状态
  POST /model/load | /model/unload     预加载 / 释放模型
  POST /model/config                   设备（auto/cuda/cpu）、释放策略（识别后空闲多少秒释放，0 立即、-1 常驻）、MHR 是否放 CPU
  POST /sessions?name=xx               请求体为图片原始字节（jpg/png/webp/bmp/gif 首帧/tiff/avif），识别一次并缓存
  GET  /sessions/{id}/image            预览图（PNG，长边不超过 1280）
  DELETE /sessions/{id}                删除图片及预识别缓存
  POST /sessions/{id}/retarget         按参数重定向（只算重定向，几十毫秒），返回 Brio 姿势、叠加层、火柴人、合理性提示
  POST /sessions/{id}/camera           按照片镜头设置 GPose 相机（经 Posegraph HTTP 闭环求解）
  GET  /presets                        参数预设
  GET  /races                          支持的种族骨架（retarget/races.py）
  POST /shutdown                       退出服务

会话里的 humans 预留多人：person 为人物序号（目前整图按一个人识别）。
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import session_archive, tuning
from .config import SERVICE_ROOT, settings
from .estimators import create_estimator
from . import RUNTIME_REQUIRED, __version__
from .gpu import query_gpu
from .i18n import set_language, tr
from .retarget import races
from .retarget.brio_pose import BrioPose
from .retarget.ffxiv import load_mhr_rest, to_brio_pose_json
from .session_cache import Session, SessionCache
from .inference_pool import InferenceFailure, InferencePool, ModelLifecycle
from .shared_inference import SharedInferencePool
from .i18n import _lang

app = FastAPI(title="posegraph", version=__version__)
_estimator = create_estimator()
# 默认单模型串行；并行槽使用独立进程，加载/配置/释放等待正在运行的识别。
_model_lock = threading.RLock()
_model_lifecycle = ModelLifecycle()
_model_state = {"loading": False, "error": None}
_release_after = settings.release_after
_recognition_parallelism = settings.recognition_parallelism
_recognition_strategy = settings.recognition_strategy
if _recognition_strategy not in ("process", "shared"):
    raise ValueError("POSEGRAPH_RECOGNITION_STRATEGY must be process or shared")
_parallel_pool: InferencePool | SharedInferencePool | None = None
_last_used = time.monotonic()

PREVIEW_MAX_SIDE = 1280
MAX_SESSIONS = 8
OUTPUTS = SERVICE_ROOT / "outputs" / "ui"


# 大批图片只保留最近使用的 8 份在内存，其余暂存磁盘；选择时直接恢复，不再推理。
_sessions = SessionCache(MAX_SESSIONS)


@app.middleware("http")
async def _language(request: Request, call_next):
    """插件在 Accept-Language 里带界面语言，返回的提示 / 错误按它翻译（i18n.py）。"""
    set_language(request.headers.get("accept-language"))
    return await call_next(request)


def _load_serial_model() -> None:
    with _model_lock:
        if _estimator.loaded:
            return
        _model_state.update(loading=True, error=None)
        try:
            _estimator.load()
        except Exception as ex:  # 显存不足、权重缺失等都要让界面看到
            _model_state["error"] = f"{type(ex).__name__}: {ex}"
            raise
        finally:
            _model_state["loading"] = False


def _worker_config() -> dict:
    return {key: getattr(_estimator, key) for key in
            ("device", "precision", "share_mhr", "mhr_on_cpu", "min_free_vram_mb")
            if hasattr(_estimator, key)}


def _uses_inference_pool() -> bool:
    return _recognition_parallelism > 1 or _recognition_strategy == "shared"


def _get_parallel_pool() -> InferencePool | SharedInferencePool:
    global _parallel_pool
    with _model_lock:
        if _parallel_pool is None:
            config = _worker_config()
            if _recognition_strategy == "shared":
                config["cuda_streams"] = settings.shared_cuda_streams
                _parallel_pool = SharedInferencePool(_recognition_parallelism, config)
            else:
                _parallel_pool = InferencePool(_recognition_parallelism, config)
        return _parallel_pool


def _close_parallel_pool() -> None:
    global _parallel_pool
    if _parallel_pool is not None:
        _parallel_pool.close()
        _parallel_pool = None


def _load_model() -> None:
    with _model_lifecycle.exclusive():
        if not _uses_inference_pool():
            _load_serial_model()
        else:
            _model_state.update(loading=True, error=None)
            try:
                _get_parallel_pool().preload(_lang.get())
            except Exception as error:
                _model_state["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                _model_state["loading"] = False


def _unload_model() -> None:
    with _model_lifecycle.exclusive(), _model_lock:
        _close_parallel_pool()
        _estimator.unload()


def _release_loop() -> None:
    """按策略释放模型：识别后空闲超过 release_after 秒就卸载（0 在识别结束时立即卸载，见 _estimate）。"""
    while True:
        time.sleep(2)
        _release_idle_models()


def _release_idle_models() -> None:
    idle = lambda: (_release_after > 0 and _model_info()["loaded"] and not _model_info()["loading"]
                    and time.monotonic() - _last_used > _release_after)
    if idle():
        with _model_lifecycle.exclusive(), _model_lock:
            if idle():
                _close_parallel_pool()
                _estimator.unload()


def _model_info() -> dict:
    info = {"loaded": _estimator.loaded, "loading": _model_state["loading"], "error": _model_state["error"],
            "device": getattr(_estimator, "active_device", None), "precision": getattr(_estimator, "active_precision", None),
            "deviceReason": _reason(getattr(_estimator, "device_reason", ("",))),
            "config": {"device": getattr(_estimator, "device", None), "releaseAfter": _release_after,
                       "mhrOnCpu": getattr(_estimator, "mhr_on_cpu", None),
                       "minFreeVramMb": getattr(_estimator, "min_free_vram_mb", None),
                       "recognitionParallelism": _recognition_parallelism,
                       "recognitionStrategy": _recognition_strategy},
            "modelCopies": int(_estimator.loaded)}
    pool = _parallel_pool
    if _uses_inference_pool():
        states = pool.states if pool is not None else []
        active = next((state for state in states if state["loaded"]), None)
        info.update(loaded=any(state["loaded"] for state in states),
                    loading=_model_state["loading"] or any(state["loading"] for state in states),
                    error=next((state["error"] for state in states if state["error"]), _model_state["error"]),
                    workers=states, activeInferences=_model_lifecycle.active,
                    modelCopies=sum(bool(state["loaded"]) for state in states))
        if active:
            info.update(device=active["device"], precision=active["precision"],
                        deviceReason=_reason(active["device_reason"]))
    return info


def _reason(r) -> str:
    """估计器的设备选择原因：(中文模板, 参数) → 当前语言的文字。"""
    if isinstance(r, str):
        return r
    return tr(r[0], **(r[1] if len(r) > 1 else {})) if r[0] else ""


@app.get("/health")
def health() -> dict:
    # 显卡信息走 nvidia-smi：模型不在 GPU 上时不能为了查询去初始化 CUDA（会常驻几百 MB 显存）
    return {"status": "ok", "version": app.version, "runtime": RUNTIME_REQUIRED, "estimator": _estimator.name, "model": _model_info(),
            "gpu": query_gpu(), "sessions": len(_sessions)}


@app.post("/model/load")
def model_load() -> dict:
    info = _model_info()
    if not info["loaded"] and not info["loading"]:
        threading.Thread(target=lambda: _safe(_load_model), daemon=True).start()
    return {"model": _model_info()}


def _safe(fn) -> None:
    try:
        fn()
    except Exception:
        pass


@app.post("/model/unload")
def model_unload() -> dict:
    _unload_model()
    return {"model": _model_info()}


class ModelConfig(BaseModel):
    device: str | None = None          # auto / cuda / cpu
    release_after: float | None = None  # 秒；0 立即释放，-1 常驻
    mhr_on_cpu: bool | None = None
    min_free_vram_mb: int | None = None
    recognition_parallelism: int | None = Field(default=None, ge=1, le=8, strict=True)
    recognition_strategy: Literal["process", "shared"] | None = None


@app.post("/model/config")
def model_config(cfg: ModelConfig) -> dict:
    """运行时修改设备与释放策略。设备相关的改动在下次加载时生效（已加载的模型会先卸载）。"""
    global _release_after, _recognition_parallelism, _recognition_strategy
    if cfg.device is not None and cfg.device not in ("auto", "cuda", "cpu"):
        raise HTTPException(400, tr("device 只能是 auto / cuda / cpu"))
    with _model_lifecycle.exclusive(), _model_lock:
        changes = {key: getattr(cfg, key) for key in ("device", "mhr_on_cpu", "min_free_vram_mb")
                   if getattr(cfg, key) is not None and getattr(_estimator, key) != getattr(cfg, key)}
        next_parallelism = cfg.recognition_parallelism if cfg.recognition_parallelism is not None else _recognition_parallelism
        next_strategy = cfg.recognition_strategy if cfg.recognition_strategy is not None else _recognition_strategy
        reload = bool(changes) or next_parallelism != _recognition_parallelism or next_strategy != _recognition_strategy
        if reload:
            # close() may fail after blocking new leases. Preserve both the old
            # configuration and pool handle so retrying this exact request will
            # attempt shutdown again, instead of treating it as a no-op update.
            _close_parallel_pool()
            _estimator.unload()
        for key, value in changes.items():
            setattr(_estimator, key, value)
        _recognition_parallelism = next_parallelism
        _recognition_strategy = next_strategy
        if cfg.release_after is not None:
            _release_after = cfg.release_after
    return {"model": _model_info()}


def _decode(data: bytes) -> np.ndarray:
    """任意常见格式 → RGB。按 EXIF 方向摆正（手机照片），GIF/动图取首帧，透明背景垫白。"""
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(data))
        img.seek(0)
        img = ImageOps.exif_transpose(img)
    except Exception as ex:
        raise HTTPException(400, tr("无法解码图片：{error}", error=ex))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.asarray(img.convert("RGB"))


def _preview_png(rgb: np.ndarray) -> bytes:
    from PIL import Image

    img = Image.fromarray(rgb)
    img.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _session(sid: str) -> Session:
    try:
        return _sessions[sid]
    except KeyError:
        raise HTTPException(404, tr("会话不存在（后端重启过？请重新识别）"))


@app.post("/sessions")
async def create_session(request: Request, name: str = "image") -> dict:
    data = await request.body()
    if not data:
        raise HTTPException(400, tr("请求体为空"))
    # 识别约几秒，放到线程池里跑，期间 /health 仍能响应
    rgb, result = await run_in_threadpool(_estimate, data)
    humans = []
    for h in result.humans:
        humans.append({"joint_global_rots": h.joint_global_rots, "joint_coords": h.joint_coords,
                       "rest_joint_coords": h.rest_joint_coords, "rest_joint_rots": h.rest_joint_rots,
                       "keypoints_2d": h.keypoints_2d, "focal_length": h.focal_length, "cam_t": h.cam_t,
                       "bbox": h.bbox, "image_size": np.array(result.image_size),
                       "vertices": h.extra["vertices"], "rest_vertices": h.extra["rest_vertices"]})
    session = await run_in_threadpool(_add_session, name, result.image_size, _preview_png(rgb), humans, result.timings_ms["inference"])

    if settings.save_sessions:   # 存一份 npz，命令行脚本（push_pose / preview_overlay / Blender 对比）可直接复用
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        np.savez(OUTPUTS / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}.npz", **humans[0],
                 faces=result.humans[0].extra["faces"])
    return _session_info(session, result.device)


def _add_session(name: str, image_size, preview_png: bytes, humans: list[dict], inference_ms: float) -> Session:
    sid = uuid.uuid4().hex[:12]
    session = Session(sid, name, tuple(image_size), preview_png, humans, inference_ms)
    _sessions[sid] = session
    return session


def _session_info(s: Session, device: str | None) -> dict:
    return {"sessionId": s.id, "name": s.name, "imageSize": list(s.image_size),
            "inferenceMs": round(s.inference_ms), "device": device,
            "humans": [{"bbox": [round(float(v), 1) for v in h["bbox"]]} for h in s.humans]}


@app.get("/sessions/{sid}/archive")
def session_archive_get(sid: str) -> Response:
    """会话存档（npz），插件存进图库的识别缓存。"""
    s = _session(sid)
    return Response(session_archive.pack(s.name, s.image_size, s.preview_png, s.humans, s.inference_ms),
                    media_type="application/octet-stream")


@app.post("/sessions/archive")
async def session_archive_restore(request: Request, name: str | None = None) -> dict:
    """从识别缓存恢复会话，不加载模型。"""
    try:
        a = session_archive.unpack(await request.body())
    except ValueError as ex:
        raise HTTPException(400, tr("识别缓存无法读取：{error}", error=ex))
    if not a["humans"]:
        raise HTTPException(400, tr("识别缓存无法读取：{error}", error="humans"))
    session = await run_in_threadpool(_add_session, name or a["name"], a["image_size"], a["preview_png"], a["humans"], a["inference_ms"])
    return _session_info(session, None)


@app.delete("/sessions/{sid}")
def session_delete(sid: str) -> dict:
    _sessions.discard(sid)
    return {"ok": True}


def _estimate(data: bytes):
    global _last_used
    rgb = _decode(data)
    with _model_lifecycle.inference():
        try:
            if _uses_inference_pool():
                try:
                    previous_load_error = _model_state["error"]
                    result = _get_parallel_pool().estimate(rgb, _release_after, _lang.get())
                    # A successful retry supersedes an earlier preload failure.
                    # Keep a newer global error and every per-worker/pool error.
                    with _model_lock:
                        if _model_state["error"] == previous_load_error:
                            _model_state["error"] = None
                except InferenceFailure as error:
                    raise HTTPException(error.status, str(error)) from error
            else:
                with _model_lock:
                    try:
                        _load_serial_model()
                    except Exception as ex:
                        raise HTTPException(503, tr("模型加载失败：{error}", error=ex))
                    try:
                        result = _estimator.estimate(rgb)
                    finally:
                        if _release_after == 0:
                            _estimator.unload()
        finally:
            _last_used = time.monotonic()
    if not result.humans:
        raise HTTPException(422, tr("未检测到人物"))
    return rgb, result


@app.get("/sessions/{sid}/image")
def session_image(sid: str) -> Response:
    return Response(_session(sid).preview_png, media_type="image/png")


class RetargetRequest(BaseModel):
    person: int = 0
    params: tuning.RetargetParams = tuning.RetargetParams()
    reference: str | None = None    # 游戏内角色当前姿势（Brio pose JSON），提供时骨长/缩放按游戏实际值
    overlay: bool = True
    stick: bool = True
    race: str | None = None         # 骨架编号 cXXXX（插件按角色外貌数据识别）；缺省时按 reference 的骨长推断
    trust_reference: bool = False   # 插件已清除过 Brio 姿势栈：骨长不符也按游戏实际值（Customize+、改体型 MOD）


def _ivcs(ref: BrioPose | None, race: dict) -> bool | None:
    """游戏角色是否加载了 IVCS 骨骼（Brio 读回的姿势含骨架里的全部骨骼，有 iv_ 骨骼就是装了）。
    没有游戏姿势时为 None（按装了计算）；没装时把提示加到 race["warnings"]。"""
    if ref is None:
        return None
    if any(name.startswith("iv_") for name in ref.bone_names):
        return True
    race["warnings"].append(tr("未检测到 IVCS 骨骼：手指第三节的弯曲已合并到第二节"))
    return False


def _reference_lengths(ref: BrioPose | None, race: dict, trust: bool) -> tuple[bool, bool]:
    """返回 (骨长是否取游戏值, 游戏骨长是否像残留的旧姿势)，并把提示加到 race["warnings"]。
    换种族后 Brio 会把按旧骨架算的位置增量继续叠加在新骨架上，读回的骨长是错的，而且重新应用也恢复不了
    （读回的是上次自己写进去的错误骨长）。"""
    if ref is None:
        return True, False
    dev = races.length_deviation(ref, race["code"])
    if dev is None or dev <= races.STALE_DEVIATION:
        return True, False
    if trust:
        race["warnings"].append(tr("游戏骨长与{name}相差 {pct}%（Customize+ 或改体型 MOD？），按游戏实际骨长计算",
                                   name=race["name"], pct=round(dev * 100)))
        return True, True
    race["warnings"].append(tr("游戏骨长与{name}相差 {pct}%，多半是换种族前残留的 Brio 姿势，已改用{name}的标准骨长",
                               name=race["name"], pct=round(dev * 100)))
    return False, True


def _resolve_race(requested: str | None, ref: BrioPose | None) -> dict:
    """确定使用的骨架。返回 races.describe 的内容，另加 source（plugin/guess/default）与 warnings。"""
    guessed = races.guess_race(ref) if ref is not None else None
    warnings = []
    if requested in races.RACES:
        code, source = requested, "plugin"
        # 骨长与识别出的种族明显不符：换种族/改体型 MOD、Customize+ 等
        if guessed and guessed[1] < 0.03 and races.RACES[guessed[0]].group != races.RACES[code].group:
            warnings.append(tr("游戏角色的骨长更像{group}，与所选{name}不符（刚换过种族？改体型 MOD？），身体比例会不对",
                               group=tr(races.RACES[guessed[0]].group), name=tr(races.RACES[code].name)))
    elif guessed and guessed[1] < 0.05:
        code, source = guessed[0], "guess"
        if requested:
            warnings.append(tr("没有 {code} 的骨架数据，按骨长推断为{group}", code=requested, group=tr(races.RACES[code].group)))
    else:
        code, source = races.DEFAULT_RACE, "default"
        warnings.append(tr("无法确定种族，按人族中原男计算") + (tr("（没有 {code} 的骨架数据）", code=requested) if requested else ""))
    info = races.describe(code)
    info.update(name=tr(info["name"]), group=tr(info["group"]))
    return {**info, "source": source, "warnings": warnings}


def _human(s: Session, person: int) -> dict:
    if not 0 <= person < len(s.humans):
        raise HTTPException(400, tr("人物序号超出范围（共 {count} 人）", count=len(s.humans)))
    return s.humans[person]


@app.post("/sessions/{sid}/retarget")
def session_retarget(sid: str, req: RetargetRequest) -> dict:
    s = _session(sid)
    human = _human(s, req.person)
    t0 = time.perf_counter()
    ref = BrioPose.loads(req.reference) if req.reference else None
    race = _resolve_race(req.race, ref)
    use_lengths, stale = _reference_lengths(ref, race, req.trust_reference)
    ivcs = _ivcs(ref, race)
    rt = tuning.run(human, s.image_size, req.params, ref, race["code"], reference_lengths=use_lengths,
                    ivcs=ivcs is not False)
    warnings, severity = tuning.check(rt, race["code"])
    out = {"pose": to_brio_pose_json(rt), "warnings": warnings, "severity": severity,
           "compensation": rt.hand_compensation, "race": race, "referenceStale": stale, "ivcs": ivcs}
    if req.overlay:
        out["overlay"] = tuning.overlay(human, s.image_size, rt, race["code"])
    if req.stick:
        out["stick"] = tuning.stick_figure(human, rt, race["code"])
    out["elapsedMs"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


class CameraRequest(BaseModel):
    person: int = 0
    params: tuning.RetargetParams = tuning.RetargetParams()
    actor: int | None = None
    race: str | None = None


@app.post("/sessions/{sid}/camera")
def session_camera(sid: str, req: CameraRequest) -> dict:
    """需要 Posegraph 的 HTTP 服务在运行（插件会调用本接口，本接口再回调插件读写相机）。"""
    from .bridge import BridgeClient
    from .camera import framing_error, photo_camera_in_world, solve_gpose_camera

    s = _session(sid)
    human = _human(s, req.person)
    client = BridgeClient()
    pose_json = client.get_pose(req.actor)
    ref = BrioPose.loads(pose_json)
    race = _resolve_race(req.race, ref)
    rt = tuning.run(human, s.image_size, req.params, ref, race["code"],
                    reference_lengths=_reference_lengths(ref, race, False)[0], ivcs=_ivcs(ref, race) is not False)
    model = ref.raw["ModelAbsoluteValues"]
    vec = lambda v: np.array([float(x) for x in v.split(",")])
    target = photo_camera_in_world(human, rt, vec(model["Position"]), vec(model["Rotation"]), vec(model["Scale"]),
                                   load_mhr_rest()[0])
    log: list[str] = []
    result = solve_gpose_camera(client, target, log=log.append)
    framing = framing_error(client, human, BrioPose.loads(client.get_pose(req.actor)), load_mhr_rest()[0])
    return {"positionErrorCm": round(result["position_error_cm"], 1),
            "forwardErrorDeg": round(result["forward_error_deg"], 2),
            "fovDeg": round(result["fov_deg"], 1), "targetFovDeg": round(result["target_fov_deg"], 1),
            "framingErrorPct": round(framing * 100, 1), "log": log}


@app.get("/races")
def race_list() -> dict:
    """插件的种族下拉框用。"""
    return {"default": races.DEFAULT_RACE,
            "races": [{**d, "name": tr(d["name"]), "group": tr(d["group"])} for d in map(races.describe, races.RACES)]}


@app.post("/shutdown")
def shutdown() -> dict:
    """插件"停止后端"用（后端可能不是插件启动的，拿不到进程句柄）。"""
    import os

    def stop():
        try:
            _unload_model()  # waits for active recognition, then reaps child workers
            _sessions.close()
        finally:
            os._exit(0)
    threading.Timer(0.3, stop).start()
    return {"ok": True}


@app.get("/presets")
def presets() -> dict:
    return {"presets": {k: v.model_dump() for k, v in tuning.PRESETS.items()},
            "amplitudeGroups": {k: list(v) for k, v in tuning.AMPLITUDE_GROUPS.items()}}


def main() -> None:
    import uvicorn

    threading.Thread(target=_release_loop, daemon=True).start()
    if settings.preload and settings.release_after != 0:
        threading.Thread(target=lambda: _safe(_load_model), daemon=True).start()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
