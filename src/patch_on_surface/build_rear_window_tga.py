"""Generate rear-window TGAs for the Nissan Micra adversarial patch experiment.

The Nissan Micra (and most CARLA vehicles) shares one glass material across
ALL glass surfaces (rear/front/side windows + headlight glass). UE splits
the texture into pieces and maps each onto a different glass element. Masoud
proposed a "canvas trick": divide the canvas into a 4-column x 2-row grid
and place the patch in the bottom row, columns 2-3 (the central-bottom
region). Empirically this region tends to land on the rear window.

The patch is the existing trained `patch_raw.pt` tensor (128 x 256), resized
to fit the bottom-half x middle-half rectangle of the canvas. Optional alpha
reduction makes the patch slightly transparent so the rear window stays
visually plausible.

Outputs (at default canvas 2048 x 2048):
    rear_window_none.TGA   fully transparent canvas (no patch)
    rear_window_raw.TGA    transparent canvas + patch in bottom-half cols 2-3,
                           with alpha = ALPHA (default 180/255 ~ 70%).

Usage (from repo root):
    python src/patch_on_surface/build_rear_window_tga.py \\
        [--canvas 2048] [--alpha 180] [--patch-pt assets/carla_plates/patch_raw.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATCH_PT = ROOT / "assets" / "carla_plates" / "patch_raw.pt"
OUT_DIR = ROOT / "assets" / "carla_rear_window"


def load_patch(pt_path: Path) -> np.ndarray:
    """Load a (1, 3, H, W) tensor saved during training and convert to HxWx3 uint8 RGB."""
    t = torch.load(pt_path, map_location="cpu")
    if isinstance(t, dict):
        # Some checkpoints store {'patch': tensor}; try common keys.
        for k in ("patch", "p", "data"):
            if k in t:
                t = t[k]
                break
    arr = t.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    # (3, H, W) -> (H, W, 3) and clip to [0, 1] then to uint8
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def build_canvas(canvas: int) -> np.ndarray:
    """Empty fully-transparent RGBA canvas (canvas x canvas)."""
    return np.zeros((canvas, canvas, 4), dtype=np.uint8)


def embed_patch_canvas_trick(canvas_rgba: np.ndarray, patch_rgb: np.ndarray, alpha: int) -> np.ndarray:
    """Place the patch in the bottom row, middle 2 columns of a 4-col x 3-row grid.

    The patch occupies rows [2H/3 : H], cols [W/4 : 3W/4]. Alpha = `alpha`/255.
    Returns a new RGBA canvas (does not mutate input). Channels are RGBA.

    Patch layout position is displayed in file: rear_window_grid_measurs.png
    """
    H, W, _ = canvas_rgba.shape
    row_start = 5 * H // 16
    row_end = 9 * H // 16
    col_start = 0
    col_end = 7 * W // 16
    target_h = row_end - row_start
    target_w = col_end - col_start

    patch_resized = cv2.resize(patch_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    out = canvas_rgba.copy()
    out[row_start:row_end, col_start:col_end, :3] = patch_resized  # RGB
    out[row_start:row_end, col_start:col_end, 3] = alpha
    return out


def save_tga(path: Path, rgba: np.ndarray):
    """Save an RGBA HxWx4 uint8 array as TGA via PIL (cv2 doesn't write TGA)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, format="TGA")
    print(f"  wrote {path}  ({rgba.shape[1]}x{rgba.shape[0]} RGBA, "
          f"alpha range [{rgba[:,:,3].min()}, {rgba[:,:,3].max()}])")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--canvas", type=int, default=2048,
                   help="Canvas side in pixels (square). Match the rear-window texture "
                        "Resource Size in Unreal Editor (default 2048).")
    p.add_argument("--alpha", type=int, default=180,
                   help="Patch alpha in [0, 255]. Lower = more transparent. Default 180 (~70%%).")
    p.add_argument("--patch-pt", type=Path, default=DEFAULT_PATCH_PT,
                   help="Path to the trained patch tensor .pt file.")
    args = p.parse_args()

    if not args.patch_pt.exists():
        raise SystemExit(f"Patch tensor not found: {args.patch_pt}")
    if not 0 <= args.alpha <= 255:
        raise SystemExit(f"--alpha must be in [0, 255], got {args.alpha}")

    print(f"Canvas    : {args.canvas} x {args.canvas}")
    print(f"Alpha     : {args.alpha} ({args.alpha / 255 * 100:.0f}%%)")
    print(f"Patch src : {args.patch_pt}")
    print(f"Out dir   : {OUT_DIR}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # rear_window_none: fully transparent canvas (fallback for the `none`
    # condition; in UE you can also simply skip the Reimport and keep the
    # default vanilla glass texture).
    none_canvas = build_canvas(args.canvas)
    save_tga(OUT_DIR / "rear_window_none.TGA", none_canvas)

    # rear_window_raw: canvas-trick layout with patch in bottom-mid quarter.
    patch_rgb = load_patch(args.patch_pt)
    print(f"  patch tensor shape: {patch_rgb.shape} (HxWx3 uint8 RGB)")
    raw_canvas = embed_patch_canvas_trick(none_canvas, patch_rgb, args.alpha)
    save_tga(OUT_DIR / "rear_window_raw.TGA", raw_canvas)

    # Quick PNG preview for sanity-check (humans don't read TGAs easily).
    Image.fromarray(raw_canvas, mode="RGBA").save(OUT_DIR / "rear_window_raw_preview.png")
    print(f"  preview: {OUT_DIR / 'rear_window_raw_preview.png'}")

    print(f"  none -> {OUT_DIR / 'car_rear_window_none.TGA'}")
    print(f"  raw  -> {OUT_DIR / 'car_rear_window_raw.TGA'}")

if __name__ == "__main__":
    main()
