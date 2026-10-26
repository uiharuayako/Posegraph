"""服务配置。全部可用环境变量覆盖，前缀 POSEGRAPH_。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 注意：项目路径含中文。torch.jit / OpenCV 等 C 层库在 Windows 上打不开非 ASCII 路径，
# 传给它们的路径要先用 Python 读成字节（见 estimators/sam3d_body.py、bench 脚本）
SERVICE_ROOT = Path(os.path.abspath(__file__)).parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"POSEGRAPH_{name}", default)


@dataclass(frozen=True)
class Settings:
    # SAM 3D Body 源码（git clone 到 third_party）与权重目录
    sam3d_repo: Path = field(default_factory=lambda: Path(_env("SAM3D_REPO", str(SERVICE_ROOT / "third_party" / "sam-3d-body"))))
    sam3d_checkpoint_dir: Path = field(default_factory=lambda: Path(_env("SAM3D_CKPT", str(SERVICE_ROOT / "checkpoints" / "sam-3d-body-vith"))))

    # 推理设备：auto（按可用显存选 GPU/CPU）/ cuda / cpu
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    # 骨干网络精度：auto（GPU bf16、CPU fp32）/ bf16（上游默认）/ fp16 / fp32
    precision: str = field(default_factory=lambda: _env("PRECISION", "auto"))
    # GPU 模式下把 MHR（约 660MB）留在 CPU
    mhr_on_cpu: bool = field(default_factory=lambda: _env("MHR_ON_CPU", "1") == "1")
    # auto 模式下选 GPU 所需的最低可用显存（MB）：MHR 放 CPU 时模型峰值约 2GB，另加 CUDA 上下文与余量
    min_free_vram_mb: int = field(default_factory=lambda: int(_env("MIN_FREE_VRAM_MB", "3000")))
    # 识别完成后多少秒释放模型：0 立即释放，-1 常驻
    release_after: float = field(default_factory=lambda: float(_env("RELEASE_AFTER", "0")))
    # 多图识别并行数；默认串行。process 策略的并行槽各自持有完整模型。
    recognition_parallelism: int = field(default_factory=lambda: min(8, max(1, int(_env("RECOGNITION_PARALLELISM", "1")))))
    # Experimental shared-parameter inference; keep the published process strategy as the default.
    recognition_strategy: str = field(default_factory=lambda: _env("RECOGNITION_STRATEGY", "process"))
    shared_cuda_streams: bool = field(default_factory=lambda: _env("SHARED_CUDA_STREAMS", "1") == "1")

    # Posegraph 插件的本地 HTTP 地址
    bridge_url: str = field(default_factory=lambda: _env("BRIDGE_URL", "http://127.0.0.1:42429"))

    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "42430")))
    # 每次识别另存一份 npz 到 outputs/ui（命令行脚本、Blender 对比可直接复用），开发用
    save_sessions: bool = field(default_factory=lambda: _env("SAVE_SESSIONS", "0") == "1")
    # 启动后立即在后台加载模型（插件一键启动时开启，首次识别不用等加载）
    preload: bool = field(default_factory=lambda: _env("PRELOAD", "0") == "1")


settings = Settings()
