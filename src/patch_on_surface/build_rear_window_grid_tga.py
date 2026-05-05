"""Generate a grid-mapped texture to visualize UV placement on the model.

Creates a square RGBA image with a configurable rows x cols grid. Each cell
is tinted with a unique color, outlined, and labeled (A1, B1, ...). This
helps identify where parts of the texture map end up on the mesh.

Usage (from repo root):
    python src/patch_on_surface/build_rear_window_grid_tga.py \\
        [--canvas 2048] [--rows 8] [--cols 8] [--fill-alpha 80]
"""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "assets" / "carla_rear_window"


def _cell_colors(n: int) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    if n <= 0:
        return colors
    for i in range(n):
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _row_label(idx: int) -> str:
    label = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def _find_font_path() -> Path | None:
    candidates = (
        Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Helvetica.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _choose_font(
    draw: ImageDraw.ImageDraw,
    rows: int,
    cols: int,
    cell_w: float,
    cell_h: float,
    font_size: int,
) -> ImageFont.ImageFont:
    font_path = _find_font_path()
    if font_path is None:
        return ImageFont.load_default()

    if font_size > 0:
        return ImageFont.truetype(str(font_path), font_size)

    max_label = f"{_row_label(rows)}{cols}"
    size = max(10, int(min(cell_w, cell_h) * 0.55))
    while size >= 10:
        font = ImageFont.truetype(str(font_path), size)
        bbox = draw.textbbox((0, 0), max_label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w + 4 <= cell_w and text_h + 4 <= cell_h:
            return font
        size -= 2

    return ImageFont.truetype(str(font_path), 10)


def build_grid_canvas(
    canvas: int,
    rows: int,
    cols: int,
    line_width: int,
    line_alpha: int,
    fill_alpha: int,
    label_alpha: int,
    font_size: int,
) -> Image.Image:
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    cell_w = canvas / cols
    cell_h = canvas / rows
    font = _choose_font(draw, rows, cols, cell_w, cell_h, font_size)
    colors = _cell_colors(rows * cols)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * cell_w))
            y0 = int(round(r * cell_h))
            x1 = int(round((c + 1) * cell_w))
            y1 = int(round((r + 1) * cell_h))

            color = colors[idx]
            idx += 1

            if fill_alpha > 0:
                draw.rectangle([x0, y0, x1, y1], fill=(*color, fill_alpha))
            if line_width > 0:
                draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, line_alpha), width=line_width)

            label = f"{_row_label(r + 1)}{c + 1}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            if text_w + 4 <= (x1 - x0) and text_h + 4 <= (y1 - y0):
                tx = x0 + (x1 - x0 - text_w) // 2
                ty = y0 + (y1 - y0 - text_h) // 2
                draw.text((tx, ty), label, fill=(255, 255, 255, label_alpha), font=font)

    return img


def save_tga(path: Path, img: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="TGA")
    print(f"  wrote {path}  ({img.size[0]}x{img.size[1]} RGBA)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--canvas", type=int, default=2048, help="Canvas side in pixels (square).")
    p.add_argument("--rows", type=int, default=8, help="Grid rows.")
    p.add_argument("--cols", type=int, default=8, help="Grid columns.")
    p.add_argument("--line-width", type=int, default=4, help="Grid line width in pixels.")
    p.add_argument("--line-alpha", type=int, default=220, help="Grid line alpha in [0, 255].")
    p.add_argument("--fill-alpha", type=int, default=80, help="Cell fill alpha in [0, 255].")
    p.add_argument("--label-alpha", type=int, default=220, help="Label alpha in [0, 255].")
    p.add_argument("--font-size", type=int, default=0,
                   help="Label font size in pixels. Use 0 for auto sizing.")
    p.add_argument("--out", type=Path, default=OUT_DIR / "rear_window_grid.TGA", help="Output TGA path.")
    args = p.parse_args()

    if args.canvas <= 0:
        raise SystemExit("--canvas must be > 0")
    if args.rows <= 0 or args.cols <= 0:
        raise SystemExit("--rows and --cols must be > 0")
    for name, val in ("line-alpha", args.line_alpha), ("fill-alpha", args.fill_alpha), ("label-alpha", args.label_alpha):
        if not 0 <= val <= 255:
            raise SystemExit(f"--{name} must be in [0, 255], got {val}")
    if args.font_size < 0:
        raise SystemExit("--font-size must be >= 0")

    print(f"Canvas   : {args.canvas} x {args.canvas}")
    print(f"Grid     : {args.rows} rows x {args.cols} cols")
    print(f"Out TGA  : {args.out}")
    print()

    img = build_grid_canvas(
        canvas=args.canvas,
        rows=args.rows,
        cols=args.cols,
        line_width=args.line_width,
        line_alpha=args.line_alpha,
        fill_alpha=args.fill_alpha,
        label_alpha=args.label_alpha,
        font_size=args.font_size,
    )
    save_tga(args.out, img)

    preview_path = args.out.with_suffix(".png")
    img.save(preview_path)
    print(f"  preview: {preview_path}")


if __name__ == "__main__":
    main()
