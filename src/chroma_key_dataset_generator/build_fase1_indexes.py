"""Build quads_index.json for every (town, light) combo of the Fase 1 dataset.

Detection strategy (validated 2026-07-07), exploiting the deterministic capture:
  1. The truck pose is identical between the day and night runs of a combo
     (same TM path, same ticks) — only the lighting differs. So the marker
     occupies the *same image pixels* in day and night.
  2. The yellow marker is easy and reliable to detect in the DAY frame via a
     tight saturated-yellow HSV mask, but collapses at night (brightness dies).
  3. So we detect the quad on the DAY frame and transfer the exact same corners
     to the matching NIGHT frame. No night-time colour detection at all.
  4. To reject false positives (Town04's yellow guardrails, road paint), the
     HSV search is restricted to the truck region, obtained from the
     illumination-consistent diff `marker_day - noleader_day` (computed
     light-with-light, never crossing day/night). The diff also catches the
     truck's shadow, but the marker is always inside that box, so as a search
     *constraint* the shadow is harmless.

Writes `quads_index.json` into every `<town>/<light>/marker/` folder, applying
the per-town frame cutoff (beyond it the vehicle drifts off the chosen road).

Usage on Vortex:
    conda activate PCLA15
    cd /home/vortex/adversarial-patch-vehicle
    python src/chroma_key_dataset_generator/build_fase1_indexes.py \\
        --fase1-root data/chroma_key_dataset/fase1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Tight saturated-yellow HSV band (same as extract_quad day path).
HSV_LOW = np.array([20, 120, 100], dtype=np.uint8)
HSV_HIGH = np.array([35, 255, 255], dtype=np.uint8)

# Per-town frame cutoff: beyond these indices the vehicle leaves the chosen
# road stretch (validated visually). Town11 has no cutoff (open fields).
FRAME_CUTOFF = {
    "Town04_spawn273": 111,
    "Town07_spawn38": 124,
    "Town11_spawn1713": 10**9,
}

COMBOS = list(FRAME_CUTOFF.keys())


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL by the sum/diff trick."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def truck_box(marker_day: np.ndarray, noleader_day: np.ndarray,
              thresh: int = 25) -> tuple[int, int, int, int] | None:
    """Bounding box of pixels that change between the leader and empty-scene
    day frames — the truck (plus its shadow). Used only to constrain the HSV
    marker search, so the shadow is harmless."""
    d = np.abs(marker_day.astype(int) - noleader_day.astype(int)).sum(2)
    m = (d > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(m)
    if len(xs) < 50:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def detect_marker_day(marker_day: np.ndarray,
                      box: tuple[int, int, int, int] | None) -> np.ndarray | None:
    """Detect the yellow marker quad in a day frame, optionally restricted to
    the truck bounding box. Returns ordered corners (4, 2) or None."""
    hsv = cv2.cvtColor(marker_day, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    if box is not None:
        x1, y1, x2, y2 = box
        region = np.zeros(m.shape, np.uint8)
        region[y1:y2 + 1, x1:x2 + 1] = 255
        m = cv2.bitwise_and(m, region)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < 200:
            break
        for eps in (0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True)
            if len(approx) == 4:
                return order_corners(approx.reshape(4, 2))
    return None


def build_combo(fase1_root: Path, combo: str) -> None:
    day_marker = fase1_root / combo / "day" / "marker"
    day_noleader = fase1_root / combo / "day" / "noleader"
    night_marker = fase1_root / combo / "night" / "marker"
    cutoff = FRAME_CUTOFF[combo]

    day_index: dict[str, dict] = {}
    night_index: dict[str, dict] = {}
    n_ok = n_miss = 0

    for img_path in sorted(day_marker.glob("*.png")):
        stem = img_path.stem
        if int(stem) > cutoff:
            continue
        mk = cv2.imread(str(img_path))
        nl_path = day_noleader / f"{stem}.png"
        nl = cv2.imread(str(nl_path)) if nl_path.exists() else None
        box = truck_box(mk, nl) if nl is not None else None
        corners = detect_marker_day(mk, box)
        if corners is None:
            n_miss += 1
            continue
        entry = {
            "corners": corners.tolist(),
            "shape": list(mk.shape),
            "area": float(cv2.contourArea(corners.astype(np.float32))),
            "source": "build_fase1_indexes.py (day HSV + truck-box constraint)",
        }
        day_index[stem] = entry
        # Transfer identical corners to the matching night frame (deterministic
        # pose → same pixels). Only if the night frame actually exists.
        if (night_marker / f"{stem}.png").exists():
            night_entry = dict(entry)
            night_entry["source"] = "build_fase1_indexes.py (corners transferred from day)"
            night_index[stem] = night_entry
        n_ok += 1

    (day_marker / "quads_index.json").write_text(json.dumps(day_index))
    (night_marker / "quads_index.json").write_text(json.dumps(night_index))
    print(f"{combo}: day {len(day_index)} / night {len(night_index)} "
          f"(detected {n_ok}, missed {n_miss}, cutoff {cutoff})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fase1-root", type=Path,
                   default=Path("data/chroma_key_dataset/fase1"))
    args = p.parse_args()
    for combo in COMBOS:
        build_combo(args.fase1_root, combo)
    print("done.")


if __name__ == "__main__":
    main()
