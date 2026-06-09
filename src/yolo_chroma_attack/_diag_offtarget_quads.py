"""Quantify how many chroma-key quads are off-target — i.e. detected somewhere
that does NOT overlap any vehicle in the paired clean frame.

For each frame in quads_index.json:
  1. Take the detected quad → bounding rect of the quad → 'quad_bbox'.
  2. Run YOLO on the paired CLEAN frame (no marker), collect all vehicle bboxes.
  3. Compute the best IoU between quad_bbox and any vehicle bbox.
  4. Classify:
       - on-target: IoU >= 0.10 with some vehicle (the marker actually sits on a car)
       - off-target: IoU < 0.10 with all vehicles (quad is on grass/road/etc.)
       - no-vehicle: YOLO sees no vehicle at all in clean (the leader is far/occluded)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = (2, 5, 7)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1.0, aa + bb - inter)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--marker-dir", type=Path, required=True)
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--yolo-weights", default="src/vehicle_counting_model/yolov8n.pt")
    p.add_argument("--iou-threshold", type=float, default=0.10)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    idx = json.loads((args.marker_dir / "quads_index.json").read_text())
    yolo = YOLO(args.yolo_weights)

    cat = Counter()
    rows = []
    items = list(idx.items())
    for i, (stem, entry) in enumerate(items):
        cp = args.clean_dir / f"{stem}.png"
        if not cp.exists():
            cat["missing_clean"] += 1
            continue
        bgr = cv2.imread(str(cp))
        corners = np.asarray(entry["corners"], np.float32)
        x1, y1 = float(corners[:, 0].min()), float(corners[:, 1].min())
        x2, y2 = float(corners[:, 0].max()), float(corners[:, 1].max())
        quad_box = (x1, y1, x2, y2)

        res = yolo.predict(bgr, verbose=False, conf=0.25)[0]
        veh_bboxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.cls.cpu().int().tolist()):
                if cls in VEHICLE_CLASSES:
                    veh_bboxes.append(box.tolist())

        if not veh_bboxes:
            best_iou = 0.0
            label = "no_vehicle"
        else:
            best_iou = max(iou_xyxy(quad_box, b) for b in veh_bboxes)
            label = "on_target" if best_iou >= args.iou_threshold else "off_target"
        cat[label] += 1
        rows.append({"stem": stem, "label": label, "best_iou": round(best_iou, 4),
                     "quad_box": quad_box, "n_vehicles_clean": len(veh_bboxes)})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(items)}  on={cat['on_target']}  off={cat['off_target']}  "
                  f"no_veh={cat['no_vehicle']}", flush=True)

    total = len(items)
    print("\n=== summary ===")
    for k in ("on_target", "off_target", "no_vehicle", "missing_clean"):
        n = cat.get(k, 0)
        print(f"  {k:14s}: {n:5d}  ({100*n/total:.1f}%)")
    if args.out_json:
        args.out_json.write_text(json.dumps({"summary": dict(cat), "rows": rows}, indent=2))
        print(f"saved -> {args.out_json}")


if __name__ == "__main__":
    main()
