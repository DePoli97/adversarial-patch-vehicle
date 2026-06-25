"""Export a trained CLIP-attack patch tensor (.pt) as a TGA file ready to drop
into the CARLA marker_yellow material slot (replacing yellow_marker.TGA).

Patch tensor is (3, H, W) in [0, 1] RGB. We save as RGBA TGA, alpha fully
opaque (the marker had alpha=1 everywhere).

Usage:
    python -m src.clip_chroma_attack.export_tga \\
        --patch experiments/clip_attack/run01/patch_final.pt \\
        --out assets/chroma_key/adv_patch_clip_v1.TGA
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch", type=Path, required=True)
    p.add_argument("--out",   type=Path, required=True)
    p.add_argument("--width",  type=int, default=512)
    p.add_argument("--height", type=int, default=256)
    args = p.parse_args()

    t = torch.load(args.patch, map_location="cpu")
    if t.dim() == 4:
        t = t.squeeze(0)
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    if arr.shape[:2] != (args.height, args.width):
        arr = np.array(
            Image.fromarray(arr).resize((args.width, args.height), Image.LANCZOS)
        )
    rgba = np.dstack([arr, np.full(arr.shape[:2], 255, dtype=np.uint8)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(args.out)
    print(f"Saved {args.out}  ({rgba.shape[1]}x{rgba.shape[0]} RGBA)")


if __name__ == "__main__":
    main()
