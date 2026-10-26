"""给教程截图加标注：截图上放编号圆标（点）或高亮框（区域），说明文字按编号列在右侧图例里。

    python annotate.py <截图目录> <输出目录> --lang zh|en [--spec tutorial_annotations.json]

标注描述（JSON）：{ "场景名": [ {"at": [x, y] 或 "box": [x0, y0, x1, y1], "dir": "up"|"down"|…（可选）, "zh": "…", "en": "…"}, … ] }
坐标是 UiPreview 输出截图里的像素（中英文共用；英文文字更长、位置不同时用 at_en / box_en）。编号按阅读顺序（先上后下）自动排。
需要 pillow。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ACCENT = (47, 181, 168)
BG = (24, 26, 30)
TEXT = (236, 240, 242)
FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
LEGEND_W = 340    # 右侧图例宽度
# 点标注的编号放在目标哪一侧（dir），默认右上；挤的地方挪到空白处
DIRS = {"up-right": (14, -14), "down-right": (14, 14), "up": (0, -24), "down": (0, 24), "left": (-24, 0), "right": (24, 0)}


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """按像素宽度换行：英文单词、数字和路径整体不拆，中文按字。"""
    lines: list[str] = []
    for para in text.split("\n"):
        line = ""
        for tok in re.findall(r"[A-Za-z0-9.#/()_'\-<>=…:]+|\s+|.", para):
            trial = line + tok
            if draw.textlength(trial, font=font) <= width or not line.strip():
                line = trial
            else:
                lines.append(line.rstrip())
                line = tok.lstrip()
        lines.append(line.rstrip())
    return lines


def badge(draw: ImageDraw.ImageDraw, center: tuple[float, float], n: int, font: ImageFont.FreeTypeFont) -> None:
    """编号圆标：深色描边 + 强调色底，在亮图和暗界面上都看得清。"""
    x, y = center
    r = 12
    draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(15, 18, 20))
    draw.ellipse([x - r, y - r, x + r, y + r], fill=ACCENT)
    draw.text((x, y), str(n), font=font, fill=(12, 16, 18), anchor="mm")


def annotate(src: Path, dst: Path, items: list[dict], lang: str) -> None:
    """截图上只放编号圆标（点）或高亮框（区域），说明文字按编号列在右侧图例里，不画长箭头。"""
    shot = Image.open(src).convert("RGB")
    font = ImageFont.truetype(FONT, 17)
    num_font = ImageFont.truetype(FONT_BOLD, 15)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    # 英文文字更长时可以用 at_en / box_en 单独指定坐标
    for item in items:
        for key in ("at", "box"):
            if f"{key}_{lang}" in item:
                item.pop("at", None)
                item.pop("box", None)
                item[key] = item[f"{key}_{lang}"]
        x, y = (item["box"][0], item["box"][1]) if "box" in item else item["at"]
        item["order"] = (round(y / 12), x)   # 阅读顺序：先上后下（同一行内先左后右）
    items.sort(key=lambda i: i["order"])

    # 图例排版
    legend_x = shot.width + 20 + 28
    text_w = LEGEND_W - 40
    entries, y = [], 20
    for n, item in enumerate(items, 1):
        lines = wrap(measure, item[lang], font, text_w)
        entries.append((n, y, lines))
        y += max(len(lines) * 24, 26) + 12
    height = max(shot.height + 40, y + 10)

    canvas = Image.new("RGB", (shot.width + 40 + LEGEND_W, height), BG)
    canvas.paste(shot, (20, 20))
    draw = ImageDraw.Draw(canvas)

    # 截图上的标记：区域画框、编号放在框的左上角外侧；点在目标右上方放编号，短线连到目标
    placed: list[tuple[float, float]] = []
    for n, item in enumerate(items, 1):
        if "box" in item:
            x0, y0, x1, y1 = item["box"]
            draw.rounded_rectangle([20 + x0 - 3, 20 + y0 - 3, 20 + x1 + 3, 20 + y1 + 3], radius=6, outline=ACCENT, width=3)
            center = [20 + x0 - 3, 20 + y0 - 3]
        else:
            tx, ty = 20 + item["at"][0], 20 + item["at"][1]
            dx, dy = DIRS[item.get("dir", "up-right")]
            center = [tx + dx, ty + dy]
            draw.line([(tx, ty), (center[0], center[1])], fill=ACCENT, width=3)
            draw.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=ACCENT)
        # 和已放的编号挤在一起时往右挪
        while any(abs(center[0] - px) < 27 and abs(center[1] - py) < 27 for px, py in placed):
            center[0] += 27
        placed.append((center[0], center[1]))
        badge(draw, center, n, num_font)

    # 图例
    for n, top, lines in entries:
        badge(draw, (legend_x - 2, top + 12), n, num_font)
        for k, line in enumerate(lines):
            draw.text((legend_x + 18, top + k * 24), line, font=font, fill=TEXT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--lang", choices=["zh", "en"], required=True)
    ap.add_argument("--spec", type=Path, default=HERE / "tutorial_annotations.json")
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    for png in sorted(args.src.glob("tut-*.png")):
        name = png.stem
        items = [dict(i) for i in spec.get(name, [])]
        if items:
            annotate(png, args.dst / f"{name}.png", items, args.lang)
        else:
            Image.open(png).save(args.dst / f"{name}.png", optimize=True)
        print(args.dst / f"{name}.png")


if __name__ == "__main__":
    main()
