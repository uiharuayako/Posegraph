"""生成更新日志窗口的横幅（Posegraph/Changelog/banner.png）：几个识别出的姿势画成发光的骨架节点图。

    python make_banner.py <npz 文件...> -o ../../Posegraph/Changelog/banner.png

npz 是后端识别结果（backend/outputs 下，键同 image_to_pose），只用 joint_coords。需要 numpy、pillow。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H = 1280, 360
BG_TOP, BG_BOTTOM = (20, 22, 26), (28, 34, 38)
ACCENT = (47, 181, 168)

# 只画身体主干、四肢和头（手指太密，缩小后是一团）
BODY = ["root", "l_upleg", "l_lowleg", "l_foot", "r_upleg", "r_lowleg", "r_foot", "c_spine0", "c_spine1", "c_spine2",
        "c_spine3", "c_neck", "c_head", "l_clavicle", "l_uparm", "l_lowarm", "l_wrist", "r_clavicle", "r_uparm",
        "r_lowarm", "r_wrist"]


def skeleton() -> tuple[list[str], list[int]]:
    rest = json.loads((ROOT / "backend/src/posegraph/retarget/data/mhr_rest.json").read_text(encoding="utf-8"))
    return rest["names"], rest["parents"]


def body_edges(names: list[str], parents: list[int]) -> list[tuple[int, int]]:
    """主干关节之间的连线：沿父链往上找到最近的主干祖先。"""
    keep = {names.index(n) for n in BODY}
    edges = []
    for j in keep:
        p = parents[j]
        while p >= 0 and p not in keep:
            p = parents[p]
        if p >= 0:
            edges.append((p, j))
    return edges


def draw_pose(layer: Image.Image, glow: Image.Image, pts: np.ndarray, edges, alpha: int) -> None:
    d, g = ImageDraw.Draw(layer), ImageDraw.Draw(glow)
    color = (*ACCENT, alpha)
    for a, b in edges:
        seg = [tuple(pts[a]), tuple(pts[b])]
        g.line(seg, fill=(*ACCENT, alpha), width=10)
        d.line(seg, fill=color, width=3)
    for i in {i for e in edges for i in e}:
        x, y = pts[i]
        r = 5
        g.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(*ACCENT, alpha))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(235, 250, 248, alpha), outline=color, width=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "Posegraph/Changelog/banner.png")
    args = ap.parse_args()

    names, parents = skeleton()
    edges = body_edges(names, parents)
    keep = sorted({i for e in edges for i in e})

    bg = Image.new("RGBA", (W, H))
    top, bottom = np.array(BG_TOP), np.array(BG_BOTTOM)
    for y in range(H):
        c = tuple(int(v) for v in top + (bottom - top) * y / H)
        ImageDraw.Draw(bg).line([(0, y), (W, y)], fill=(*c, 255))
    # 右侧一圈柔光
    halo = Image.new("RGBA", (W, H))
    ImageDraw.Draw(halo).ellipse([W * 0.45, -H * 0.4, W * 1.25, H * 1.4], fill=(*ACCENT, 46))
    bg = Image.alpha_composite(bg, halo.filter(ImageFilter.GaussianBlur(120)))

    layer = Image.new("RGBA", (W, H))
    glow = Image.new("RGBA", (W, H))
    n = len(args.npz)
    for k, path in enumerate(args.npz):
        P = np.load(path)["joint_coords"][:, :2] * np.array([1.0, -1.0])   # 相机系 y 朝下 → 画面 y 朝上
        P = P[keep]
        lo, hi = P.min(0), P.max(0)
        h = H * 0.78
        s = h / (hi[1] - lo[1])
        cx = W * (0.5 + 0.45 * (k + 0.5) / n)   # 排在右半边
        pts = np.zeros((len(names), 2))
        pts[keep] = (P - (lo + hi) / 2) * s * np.array([1, -1]) + np.array([cx, H / 2])
        draw_pose(layer, glow, pts, edges, alpha=255 if k == n // 2 else 170)
    bg = Image.alpha_composite(bg, glow.filter(ImageFilter.GaussianBlur(8)))
    bg = Image.alpha_composite(bg, layer)
    # 左侧往右渐隐，放标题（标语和版本号在插件里画，方便翻译）
    fade = Image.new("L", (W, H))
    for x in range(W):
        ImageDraw.Draw(fade).line([(x, 0), (x, H)], fill=int(255 * min(1.0, max(0.0, (W * 0.62 - x) / (W * 0.25)))))
    bg = Image.composite(Image.new("RGBA", (W, H), (*BG_TOP, 255)), bg, fade.point(lambda v: int(v * 0.85)))
    d = ImageDraw.Draw(bg)
    title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 112)
    d.text((64, H / 2 - 18), "Posegraph", font=title, fill=(240, 244, 246), anchor="lm")
    d.rectangle([68, H / 2 + 52, 68 + 120, H / 2 + 58], fill=ACCENT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bg.convert("RGB").save(args.output, optimize=True)
    print(args.output)


if __name__ == "__main__":
    main()
