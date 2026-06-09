"""Filter quads_index.json keeping only frames where YOLO actually detects the
leader vehicle in the paired clean frame. Frames where the leader is already
invisible to YOLO contribute zero gradient signal and add noise to multi-seed
runs — removing them sharpens training.

Output: writes `quads_index_visible.json` next to the original.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = (2, 5, 7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--marker-dir", type=Path, required=True)
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--yolo-weights", default="src/vehicle_counting_model/yolov8n.pt")
    p.add_argument("--conf-threshold", type=float, default=0.25)
    p.add_argument("--expand", type=float, default=2.5)
    args = p.parse_args()

    idx = json.loads((args.marker_dir / "quads_index.json").read_text())
    yolo = YOLO(args.yolo_weights)

    kept = {}
    n_seen = 0
    n_skip_missing = 0
    n_skip_blind = 0
    items = list(idx.items())
    for i, (stem, entry) in enumerate(items):
        clean_path = args.clean_dir / f"{stem}.png"
        if not clean_path.exists():
            n_skip_missing += 1
            continue
        bgr = cv2.imread(str(clean_path))
        corners = np.asarray(entry["corners"], np.float32)
        H, W = bgr.shape[:2]
        cx = corners[:, 0].mean()
        cy = corners[:, 1].mean()
        hw = (corners[:, 0].max() - corners[:, 0].min()) * 0.5 * args.expand
        hh = (corners[:, 1].max() - corners[:, 1].min()) * 0.5 * args.expand
        x1, y1, x2, y2 = max(0, cx - hw), max(0, cy - hh), min(W - 1, cx + hw), min(H - 1, cy + hh)

        res = yolo.predict(bgr, verbose=False, conf=args.conf_threshold)[0]
        detected = False
        if res.boxes is not None and len(res.boxes) > 0:
            for box, cls in zip(res.boxes.xywh.cpu().numpy(), res.boxes.cls.cpu().int().tolist()):
                if cls not in VEHICLE_CLASSES:
                    continue
                bx, by = float(box[0]), float(box[1])
                if x1 <= bx <= x2 and y1 <= by <= y2:
                    detected = True
                    break
        if detected:
            kept[stem] = entry
            n_seen += 1
        else:
            n_skip_blind += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(items)}  kept={n_seen}  blind={n_skip_blind}", flush=True)

    out_path = args.marker_dir / "quads_index_visible.json"
    out_path.write_text(json.dumps(kept))
    print(f"\nTotal: {len(items)}  kept (visible in clean): {n_seen}  "
          f"blind: {n_skip_blind}  missing: {n_skip_missing}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
