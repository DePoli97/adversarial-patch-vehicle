"""Generate a standalone yellow chroma-key marker image.

Just a solid yellow rectangle on a transparent background. You then drag
this on top of the vehicle's texture in GIMP/Photoshop, position it where
the rear surface lives in the UV map, flatten, and export the new TGA.

Outputs (default 512 x 256 yellow block on a 1024 x 1024 transparent canvas,
written to assets/chroma_key/):
    yellow_marker.PNG    standalone yellow patch (alpha = 255 on the block,
                         0 elsewhere). Good to paste over an existing TGA.
    yellow_marker.TGA    same, TGA format if you prefer.

Usage (from repo root):
    python src/chroma_key_dataset_generator/build_yellow_tga.py \\
        [--width 512] [--height 256] [--rgb 255 220 0] \\
        [--canvas 1024]   # optional transparent canvas size (else tight crop)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "assets" / "chroma_key"


def build_marker(width: int, height: int, rgb: tuple[int, int, int],
                 canvas: int | None) -> np.ndarray:
    """Solid yellow rectangle, opaque, on a transparent background.

    If `canvas` is None: tight crop (image is exactly width x height,
    fully opaque yellow). Easiest to paste.

    If `canvas` is set: the yellow block is centered on a `canvas x canvas`
    transparent canvas. Useful if you want positional headroom in your editor.
    """
    if canvas is None:
        out = np.zeros((height, width, 4), dtype=np.uint8)
        out[:, :, 0] = rgb[0]
        out[:, :, 1] = rgb[1]
        out[:, :, 2] = rgb[2]
        out[:, :, 3] = 255
        return out

    out = np.zeros((canvas, canvas, 4), dtype=np.uint8)
    y0 = (canvas - height) // 2
    x0 = (canvas - width) // 2
    out[y0:y0 + height, x0:x0 + width, 0] = rgb[0]
    out[y0:y0 + height, x0:x0 + width, 1] = rgb[1]
    out[y0:y0 + height, x0:x0 + width, 2] = rgb[2]
    out[y0:y0 + height, x0:x0 + width, 3] = 255
    return out


def save(path: Path, rgba: np.ndarray, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, format=fmt)
    H, W = rgba.shape[:2]
    print(f"  wrote {path}  ({W}x{H} RGBA)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=512,
                   help="Width of the yellow block in pixels (default 512).")
    p.add_argument("--height", type=int, default=256,
                   help="Height of the yellow block in pixels (default 256).")
    p.add_argument("--rgb", type=int, nargs=3, default=[255, 220, 0],
                   metavar=("R", "G", "B"),
                   help="Yellow marker color (default 255 220 0).")
    p.add_argument("--canvas", type=int, default=None,
                   help="Optional transparent canvas size (square). "
                        "If omitted, image is a tight crop of the yellow block.")
    p.add_argument("--name", default="yellow_marker",
                   help="Output filename stem (default yellow_marker).")
    args = p.parse_args()

    print(f"Block  : {args.width} x {args.height}")
    print(f"Canvas : {args.canvas or 'tight crop'}")
    print(f"Color  : RGB{tuple(args.rgb)}")
    print(f"Out dir: {OUT_DIR}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rgba = build_marker(args.width, args.height, tuple(args.rgb), args.canvas)

    save(OUT_DIR / f"{args.name}.PNG", rgba, "PNG")
    save(OUT_DIR / f"{args.name}.TGA", rgba, "TGA")


if __name__ == "__main__":
    main()
