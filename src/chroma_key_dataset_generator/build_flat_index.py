"""Build quads_index.json for a FLAT capture (no day/night split), using the
same diff-based corner detection validated on Fase 1.

Fase 1 exploited day->night transfer; this capture (e.g. capture_20260609_014138,
the big multi-city NPC dataset) is a single lighting condition per frame, so we
detect the marker corners directly on every marker frame:

  1. diff `marker[i] - noleader[i]`  ->  bounding box of the truck (+ shadow),
     which constrains the yellow search and rejects background yellows.
  2. tight saturated-yellow HSV mask inside that box  ->  the marker quad.

It also writes an --overlay-dir with the detected quad drawn on a sample of
frames, so you can eyeball whether the detection is good BEFORE training on it.

Usage on Vortex:
    conda activate PCLA15
    cd /home/vortex/adversarial-patch-vehicle
    python src/chroma_key_dataset_generator/build_flat_index.py \\
        --marker-dir   data/chroma_key_dataset/capture_20260609_014138_marker \\
        --noleader-dir data/chroma_key_dataset/capture_20260609_014138_noleader \\
        --overlay-dir  docs/corner_check_014138 \\
        --overlay-n 24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Reuse the exact detection primitives validated on Fase 1.
from build_fase1_indexes import (  # type: ignore
    HSV_LOW, HSV_HIGH, order_corners, truck_box, detect_marker_day,
)


def build(marker_dir: Path, noleader_dir: Path,
          overlay_dir: Path | None, overlay_n: int) -> None:
    index: dict[str, dict] = {}
    n_ok = n_miss = 0
    frames = sorted(marker_dir.glob("*.png"))
    # pick evenly-spaced frames for the overlay preview
    overlay_stems = set()
    if overlay_dir is not None and frames:
        step = max(1, len(frames) // overlay_n)
        overlay_stems = {frames[i].stem for i in range(0, len(frames), step)}
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for img_path in frames:
        stem = img_path.stem
        mk = cv2.imread(str(img_path))
        nl_path = noleader_dir / f"{stem}.png"
        nl = cv2.imread(str(nl_path)) if nl_path.exists() else None
        box = truck_box(mk, nl) if nl is not None else None
        corners = detect_marker_day(mk, box)
        if corners is None:
            n_miss += 1
            status = "MISS"
        else:
            index[stem] = {
                "corners": corners.tolist(),
                "shape": list(mk.shape),
                "area": float(cv2.contourArea(corners.astype(np.float32))),
                "source": "build_flat_index.py (diff marker-noleader + HSV in box)",
            }
            n_ok += 1
            status = "OK"

        if stem in overlay_stems:
            vis = mk.copy()
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 2)
            if corners is not None:
                cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 0, 255), 3)
                for k, (px, py) in enumerate(corners.astype(int)):
                    cv2.circle(vis, (px, py), 6, (0, 255, 0), -1)
                    cv2.putText(vis, str(k), (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis, f"{stem}  {status}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255) if status == "MISS" else (0, 220, 0), 2)
            cv2.imwrite(str(overlay_dir / f"{stem}_{status}.png"), vis)

    (marker_dir / "quads_index.json").write_text(json.dumps(index))
    total = n_ok + n_miss
    rec = 100.0 * n_ok / total if total else 0.0
    print(f"{marker_dir.name}: detected {n_ok}/{total} ({rec:.1f}% recall), "
          f"missed {n_miss}")
    print(f"wrote {marker_dir / 'quads_index.json'}")
    if overlay_dir is not None:
        print(f"overlay preview ({len(overlay_stems)} frames) -> {overlay_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--marker-dir", type=Path, required=True)
    p.add_argument("--noleader-dir", type=Path, required=True)
    p.add_argument("--overlay-dir", type=Path, default=None)
    p.add_argument("--overlay-n", type=int, default=24)
    args = p.parse_args()
    build(args.marker_dir, args.noleader_dir, args.overlay_dir, args.overlay_n)


if __name__ == "__main__":
    main()
