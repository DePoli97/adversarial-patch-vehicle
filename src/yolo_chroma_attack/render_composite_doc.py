"""Render a trained patch onto a day frame AND its night twin, to document the
train/deploy lighting gap: during training the patch is composited at its own
(bright) RGB values regardless of scene illumination, so on a night frame it
appears glaringly bright — brighter than it ever could be once baked into the
CarlaCola BaseColor (albedo) texture and lit by the night scene.

This produces the figure for the thesis report / defense: "what training sees at
night" vs "what training sees in daylight".

Usage on Vortex:
    conda activate PCLA15
    python -m src.yolo_chroma_attack.render_composite_doc \\
        --patch experiments/yolo_attack/fase1_20260711_153141/Town04_spawn273/patch_final.pt \\
        --day-marker   data/chroma_key_dataset/fase1/Town04_spawn273/day/dist10m/marker \\
        --night-marker data/chroma_key_dataset/fase1/Town04_spawn273/night/dist10m/marker \\
        --stem 000030 \\
        --out docs/lighting_gap
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.yolo_chroma_attack.patch_render import render_patch_on_image


def load_frame(marker_dir: Path, stem: str):
    img = cv2.imread(str(marker_dir / f"{stem}.png"))  # BGR
    index = json.loads((marker_dir / "quads_index.json").read_text())
    corners = np.asarray(index[stem]["corners"], dtype=np.float32)  # (4,2) TL/TR/BR/BL
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)         # (1,3,H,W)
    c = torch.from_numpy(corners).unsqueeze(0)                       # (1,4,2)
    return t, c


def composite(patch, marker_dir: Path, stem: str, out_path: Path):
    img, corners = load_frame(marker_dir, stem)
    out = render_patch_on_image(img, patch, corners)               # (1,3,H,W) [0,1]
    arr = (out.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    # also report mean scene luminance around the marker vs patch luminance
    scene_lum = float(img.mean())
    patch_lum = float(patch.mean())
    print(f"  {out_path.name}: scene_mean={scene_lum:.3f}  patch_mean={patch_lum:.3f}  "
          f"ratio patch/scene = {patch_lum/max(scene_lum,1e-6):.2f}x")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch", type=Path, required=True)
    p.add_argument("--day-marker", type=Path, required=True)
    p.add_argument("--night-marker", type=Path, required=True)
    p.add_argument("--stem", default="000030")
    p.add_argument("--out", type=Path, default=Path("docs/lighting_gap"))
    args = p.parse_args()

    patch = torch.load(args.patch, map_location="cpu")
    if patch.dim() == 4:
        patch = patch.squeeze(0)
    patch = patch.clamp(0, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    print("Lighting gap doc (patch composited at constant brightness):")
    composite(patch, args.day_marker, args.stem, args.out / f"day_{args.stem}.png")
    composite(patch, args.night_marker, args.stem, args.out / f"night_{args.stem}.png")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
