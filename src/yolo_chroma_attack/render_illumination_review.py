"""Generate a review gallery of illumination-aware compositing, to eyeball
realism BEFORE committing to a full retraining grid.

For each source (fase1 day / fase1 night / the old _014138 capture) it composites
a patch onto sampled frames with the per-frame illumination scale applied
(patch × s, s = ring_luminance / day_ref). Night frames also get a NO-FIX copy
(s = 1, the current-training behaviour) so the before/after is obvious.

The daylight reference `ref` is auto-computed as the median ring luminance over a
sample of fase1 day frames, unless --ref is given.

Usage on Vortex:
    conda activate PCLA15
    python -m src.yolo_chroma_attack.render_illumination_review \\
        --patch experiments/yolo_attack/generalist_full_20260711_164353/generalist_full/patch_final.pt \\
        --fase1-root data/chroma_key_dataset/fase1 \\
        --old-marker data/chroma_key_dataset/capture_20260609_014138_marker \\
        --out docs/illumination_review --n-per-source 30
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from src.yolo_chroma_attack.illumination import (
    illumination_map_ref, rectify_quad, _LUMA_BGR,
)
from src.yolo_chroma_attack.patch_render import render_patch_on_image

TOWNS = ["Town04_spawn273", "Town07_spawn38", "Town11_spawn1713"]


def load_frame(marker_dir: Path, stem: str):
    bgr = cv2.imread(str(marker_dir / f"{stem}.png"))
    index = json.loads((marker_dir / "quads_index.json").read_text())
    corners = np.asarray(index[stem]["corners"], dtype=np.float32)
    return bgr, corners


def composite(bgr, corners, patch_lit, out_path: Path):
    """patch_lit: (3,Ph,Pw) already illumination-scaled."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    c = torch.from_numpy(corners).unsqueeze(0)
    out = render_patch_on_image(img, patch_lit.clamp(0, 1), c)
    arr = (out.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def sample_stems(marker_dir: Path, n: int) -> list[str]:
    index = json.loads((marker_dir / "quads_index.json").read_text())
    stems = sorted(index.keys())
    if len(stems) <= n:
        return stems
    step = len(stems) / n
    return [stems[int(i * step)] for i in range(n)]


def calibrate_yellow_ref(fase1_root: Path, ph: int, pw: int, n: int = 12) -> float:
    """Yellow-marker albedo reference = p90 of the rectified marker luminance over
    fase1 DAY frames (full daylight ≈ illumination 1). A frame's illumination is
    then marker_luminance / yellow_ref, so day ≈ 1 and night < 1."""
    vals = []
    for town in TOWNS:
        md = fase1_root / town / "day" / "dist10m" / "marker"
        if not (md / "quads_index.json").exists():
            continue
        for stem in sample_stems(md, n):
            bgr, corners = load_frame(md, stem)
            L = rectify_quad(bgr, corners, ph, pw).astype(np.float32) / 255.0 @ _LUMA_BGR
            vals.append(float(np.percentile(L, 90)))
    return float(np.median(vals)) if vals else 0.80


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch", type=Path, required=True)
    p.add_argument("--fase1-root", type=Path, default=Path("data/chroma_key_dataset/fase1"))
    p.add_argument("--old-marker", type=Path,
                   default=Path("data/chroma_key_dataset/capture_20260609_014138_marker"))
    p.add_argument("--out", type=Path, default=Path("docs/illumination_review"))
    p.add_argument("--n-per-source", type=int, default=30)
    p.add_argument("--ref", type=float, default=None)
    args = p.parse_args()
    random.seed(0)

    patch = torch.load(args.patch, map_location="cpu")
    if patch.dim() == 4:
        patch = patch.squeeze(0)
    patch = patch.clamp(0, 1)
    Ph, Pw = patch.shape[-2], patch.shape[-1]

    args.out.mkdir(parents=True, exist_ok=True)
    yellow_ref = args.ref if args.ref is not None else calibrate_yellow_ref(args.fase1_root, Ph, Pw)
    print(f"yellow-marker albedo reference = {yellow_ref:.3f}")

    def as_map(m):  # (Ph,Pw) numpy -> (1,Ph,Pw) torch to broadcast over channels
        return torch.from_numpy(m).unsqueeze(0)

    # Every frame: per-pixel illumination map = its OWN marker luminance / yellow_ref.
    # The day frame is used ONLY to locate the quad (corners already in the index).
    sources = []
    for town in TOWNS:
        sources.append((f"{town}_day", args.fase1_root / town / "day" / "dist10m" / "marker", False))
        sources.append((f"{town}_night", args.fase1_root / town / "night" / "dist10m" / "marker", True))
    sources.append(("old_014138", args.old_marker, False))

    for label, md, is_night in sources:
        if not (md / "quads_index.json").exists():
            print(f"  skip {label} (no index)")
            continue
        for stem in sample_stems(md, args.n_per_source):
            bgr, corners = load_frame(md, stem)
            m = illumination_map_ref(bgr, corners, Ph, Pw, yellow_ref=yellow_ref)
            composite(bgr, corners, patch * as_map(m),
                      args.out / f"{label}_{stem}_s{m.mean():.2f}_FIX.png")
            if is_night:
                composite(bgr, corners, patch,
                          args.out / f"{label}_{stem}_NOFIX.png")
        print(f"  {label}: done")

    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
