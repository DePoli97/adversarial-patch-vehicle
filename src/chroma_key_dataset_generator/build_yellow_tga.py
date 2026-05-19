"""Generate the chroma-key (yellow marker) TGA for a target vehicle.

The yellow rectangle marks where the adversarial patch will eventually be
warped onto the vehicle's body/glass during training. Yellow is chosen
because it does not appear in CARLA natural scenes, so it can be reliably
isolated with an HSV mask at extraction time.

Layout: the yellow block sits in the same region of the canvas used by
`src/patch_on_surface/build_rear_window_tga.py` (rows 5H/16 : 9H/16,
cols 0 : 7W/16), so the patch we already trained on the rear window can be
overlaid back on top of it for visual demos.

Outputs (default canvas 2048 x 2048, written to assets/chroma_key/):
    rear_window_yellow.TGA   transparent canvas + opaque yellow block

Usage (from repo root):
    python src/chroma_key_dataset_generator/build_yellow_tga.py \\
        [--canvas 2048] [--rgb 255 220 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "assets" / "chroma_key"


def build_yellow_canvas(canvas: int, rgb: tuple[int, int, int]) -> np.ndarray:
    """Transparent canvas with a solid yellow block in the patch region.

    Layout matches the canvas-trick from build_rear_window_tga.py:
        rows [5H/16 : 9H/16], cols [0 : 7W/16]
    """
    out = np.zeros((canvas, canvas, 4), dtype=np.uint8)
    H, W = canvas, canvas
    row_start = 5 * H // 16
    row_end = 9 * H // 16
    col_start = 0
    col_end = 7 * W // 16
    out[row_start:row_end, col_start:col_end, 0] = rgb[0]
    out[row_start:row_end, col_start:col_end, 1] = rgb[1]
    out[row_start:row_end, col_start:col_end, 2] = rgb[2]
    out[row_start:row_end, col_start:col_end, 3] = 255  # fully opaque
    return out


def save_tga(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, format="TGA")
    H, W = rgba.shape[:2]
    print(f"  wrote {path}  ({W}x{H} RGBA)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--canvas", type=int, default=2048,
                   help="Canvas side in pixels (square). Match the vehicle's "
                        "shared-glass texture Resource Size in Unreal Editor.")
    p.add_argument("--rgb", type=int, nargs=3, default=[255, 220, 0],
                   metavar=("R", "G", "B"),
                   help="Yellow marker color (default 255 220 0).")
    p.add_argument("--name", default="rear_window_yellow",
                   help="Output filename stem (default rear_window_yellow).")
    args = p.parse_args()

    print(f"Canvas : {args.canvas} x {args.canvas}")
    print(f"Color  : RGB{tuple(args.rgb)}")
    print(f"Out dir: {OUT_DIR}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas = build_yellow_canvas(args.canvas, tuple(args.rgb))
    out_path = OUT_DIR / f"{args.name}.TGA"
    save_tga(out_path, canvas)

    # Sanity PNG preview
    Image.fromarray(canvas, mode="RGBA").save(OUT_DIR / f"{args.name}_preview.png")
    print(f"  preview: {OUT_DIR / f'{args.name}_preview.png'}")


if __name__ == "__main__":
    main()
