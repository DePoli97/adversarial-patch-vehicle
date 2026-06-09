"""Export a trained patch tensor (.pt) as a TGA file ready to drop into the
CARLA marker_yellow package (replacing yellow_marker.TGA).

Patch tensor is (3, H, W) in [0,1] RGB. We save as RGBA TGA matching the
marker dimensions (W x H == 512 x 256 by default). Alpha is fully opaque
since the chroma-key marker had alpha=1 everywhere.

Usage:
    python -m src.yolo_chroma_attack.export_tga \\
        --patch experiments/yolo_attack/run02_20260609_093207/patch_final.pt \\
        --out assets/chroma_key/adv_patch_v1.TGA
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=256)
    args = p.parse_args()

    t = torch.load(args.patch, map_location="cpu")
    if t.dim() == 4:
        t = t.squeeze(0)
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)  # H,W,3 RGB
    if arr.shape[:2] != (args.height, args.width):
        from PIL import Image as _I
        arr = np.array(
            _I.fromarray(arr).resize((args.width, args.height), _I.LANCZOS)
        )
    rgba = np.dstack([arr, np.full(arr.shape[:2], 255, dtype=np.uint8)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(args.out)
    print(f"Saved {args.out}  ({rgba.shape[1]}x{rgba.shape[0]} RGBA)")


if __name__ == "__main__":
    main()
