"""Diagnostic: which COCO class does YOLOv8n assign to the CarlaCola truck?"""
import glob
import json
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

yolo = YOLO("src/vehicle_counting_model/yolov8n.pt")
COCO_NAMES = yolo.model.names

clean_dir = "data/chroma_key_dataset/capture_20260609_014138_clean"
pngs = sorted(glob.glob(f"{clean_dir}/*.png"))[:100]
print(f"Testing {len(pngs)} clean frames")

with open("data/chroma_key_dataset/capture_20260609_014138_marker/quads_index.json") as f:
    idx = json.load(f)

cls_counts = Counter()
n_with_detect = 0
n_total = 0
all_confs = []
for png in pngs:
    stem = png.split("/")[-1].replace(".png", "")
    if stem not in idx:
        continue
    n_total += 1
    corners = np.array(idx[stem]["corners"])
    cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
    half_w = (corners[:, 0].max() - corners[:, 0].min()) * 1.25
    half_h = (corners[:, 1].max() - corners[:, 1].min()) * 1.25
    x1, y1, x2, y2 = cx - half_w, cy - half_h, cx + half_w, cy + half_h
    img = cv2.imread(png)
    res = yolo.predict(img, verbose=False, conf=0.25)[0]
    detected = False
    best_conf = 0.0
    if res.boxes is not None and len(res.boxes) > 0:
        for box, cls, conf in zip(
            res.boxes.xywh.cpu().numpy(),
            res.boxes.cls.cpu().int().tolist(),
            res.boxes.conf.cpu().numpy(),
        ):
            bx, by = float(box[0]), float(box[1])
            if x1 <= bx <= x2 and y1 <= by <= y2:
                cls_counts[COCO_NAMES[cls]] += 1
                detected = True
                best_conf = max(best_conf, float(conf))
    if detected:
        n_with_detect += 1
        all_confs.append(best_conf)

print(f"Frames where YOLO detects SOMETHING in the leader region: {n_with_detect}/{n_total} = {100*n_with_detect/n_total:.1f}%")
print(f"Class distribution of those detections:")
for k, v in cls_counts.most_common():
    print(f"  {k}: {v}")
if all_confs:
    print(f"Confidence of those detections: mean={np.mean(all_confs):.3f}  median={np.median(all_confs):.3f}  min={np.min(all_confs):.3f}  max={np.max(all_confs):.3f}")
