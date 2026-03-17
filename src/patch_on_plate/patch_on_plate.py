"""
Patch-on-Plate Experiment
=========================
Overlays the CARLA license plate texture onto real-world license plates
(CCPD dataset) using perspective warping, then compares YOLOv8 detection
scores between original and patched images.

Usage:
    python patch_on_plate.py [--n_images 50] [--output_dir ../../experiments/patch_on_plate]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --- Paths (relative to project root) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXTURE = PROJECT_ROOT / "assets" / "T_LicensePlate_d.TGA"
DEFAULT_WEIGHTS = PROJECT_ROOT / "src" / "vehicle_counting_model" / "yolov8n.pt"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "CCPD2020" / "ccpd_green" / "train"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "patch_on_plate"

# COCO vehicle class IDs
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


def parse_ccpd_filename(filename: str) -> dict:
    """Parse CCPD filename to extract annotations.

    CCPD filename format:
        <area>-<tilt>-<bbox>-<vertices>-<license>-<brightness>-<blurriness>.jpg

    Vertices (field 3, 0-indexed): four corners starting from bottom-right,
    counter-clockwise: BR, BL, TL, TR.
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Expected 7 fields in CCPD filename, got {len(parts)}: {filename}")

    vertices_str = parts[3]
    vertices = []
    for pt in vertices_str.split("_"):
        x, y = pt.split("&")
        vertices.append([int(x), int(y)])

    # Order: bottom-right, bottom-left, top-left, top-right
    return {
        "area": parts[0],
        "tilt": parts[1],
        "bbox": parts[2],
        "vertices": np.array(vertices, dtype=np.float32),  # shape (4, 2)
        "license": parts[4],
        "brightness": int(parts[5]),
        "blurriness": int(parts[6]),
    }


def load_texture(texture_path: str) -> np.ndarray:
    """Load the CARLA license plate texture as BGR, cropped to content area.

    The TGA texture is 512x512 but the actual plate content only occupies
    a horizontal band in the center (the rest is transparent). We crop to
    the alpha channel's bounding box so the warp fills the entire plate.
    """
    tex = Image.open(texture_path).convert("RGBA")
    arr = np.array(tex)
    alpha = arr[:, :, 3]
    rows = np.where(alpha.max(axis=1) > 0)[0]
    cols = np.where(alpha.max(axis=0) > 0)[0]
    cropped = arr[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1, :3]
    return cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)


def warp_texture_onto_image(
    image: np.ndarray, texture: np.ndarray, dst_corners: np.ndarray
) -> np.ndarray:
    """Perspective-warp the texture onto the image at the given 4 corners.

    Args:
        image: BGR image (H, W, 3).
        texture: BGR texture image.
        dst_corners: shape (4, 2) float32, order BR, BL, TL, TR (CCPD convention).

    Returns:
        Image with the texture overlaid on the license plate region.
    """
    h_tex, w_tex = texture.shape[:2]

    # Source corners of the texture (TL, TR, BR, BL)
    src_corners = np.array(
        [[0, 0], [w_tex - 1, 0], [w_tex - 1, h_tex - 1], [0, h_tex - 1]],
        dtype=np.float32,
    )

    # CCPD vertex order: BR(0), BL(1), TL(2), TR(3)
    # We need to map: src TL -> dst TL, src TR -> dst TR, src BR -> dst BR, src BL -> dst BL
    dst_reordered = np.array(
        [dst_corners[2], dst_corners[3], dst_corners[0], dst_corners[1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src_corners, dst_reordered)
    h_img, w_img = image.shape[:2]
    warped = cv2.warpPerspective(texture, M, (w_img, h_img))

    # Create mask as filled polygon from destination corners (not color-based,
    # which fails on dark/black regions of the texture)
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    poly = dst_reordered.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillConvexPoly(mask, poly, 255)

    # Blend: replace the plate region
    result = image.copy()
    result[mask > 0] = warped[mask > 0]
    return result


def run_yolo(model: YOLO, image: np.ndarray) -> list[dict]:
    """Run YOLO on an image and return vehicle detections.

    Returns list of dicts with keys: class_id, class_name, confidence, bbox.
    """
    results = model(image, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                detections.append({
                    "class_id": cls_id,
                    "class_name": VEHICLE_CLASSES[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                })
    return detections


def compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def match_detections(orig_dets: list, patched_dets: list, iou_threshold: float = 0.5):
    """Match original detections to patched detections by IoU.

    Returns list of tuples: (orig_det, matched_patched_det_or_None).
    """
    matched = []
    used_patched = set()

    for od in orig_dets:
        best_iou = 0
        best_idx = None
        for j, pd in enumerate(patched_dets):
            if j in used_patched:
                continue
            iou = compute_iou(od["bbox"], pd["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_threshold and best_idx is not None:
            matched.append((od, patched_dets[best_idx]))
            used_patched.add(best_idx)
        else:
            matched.append((od, None))

    return matched


def save_comparison_image(
    original: np.ndarray,
    patched: np.ndarray,
    orig_dets: list,
    patched_dets: list,
    output_path: str,
):
    """Save a side-by-side comparison image with detection boxes."""
    def draw_boxes(img, dets, color):
        vis = img.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f'{d["class_name"]} {d["confidence"]:.3f}'
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis

    vis_orig = draw_boxes(original, orig_dets, (0, 255, 0))
    vis_patched = draw_boxes(patched, patched_dets, (0, 0, 255))

    # Ensure same height
    h1, h2 = vis_orig.shape[0], vis_patched.shape[0]
    if h1 != h2:
        target_h = max(h1, h2)
        if h1 < target_h:
            vis_orig = cv2.resize(vis_orig, (int(vis_orig.shape[1] * target_h / h1), target_h))
        if h2 < target_h:
            vis_patched = cv2.resize(vis_patched, (int(vis_patched.shape[1] * target_h / h2), target_h))

    combined = np.hstack([vis_orig, vis_patched])
    cv2.imwrite(output_path, combined)


def main():
    parser = argparse.ArgumentParser(description="Patch-on-Plate experiment")
    parser.add_argument("--n_images", type=int, default=50, help="Number of images to process")
    parser.add_argument("--texture", type=str, default=str(DEFAULT_TEXTURE))
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)

    print(f"Loading texture from {args.texture}")
    texture = load_texture(args.texture)

    print(f"Loading YOLOv8 model from {args.weights}")
    model = YOLO(args.weights)

    # Collect image files
    dataset_path = Path(args.dataset)
    image_files = sorted([f for f in dataset_path.iterdir() if f.suffix == ".jpg"])
    if len(image_files) == 0:
        print(f"ERROR: No .jpg files found in {dataset_path}")
        sys.exit(1)

    n = min(args.n_images, len(image_files))
    print(f"Processing {n} images from {dataset_path}")

    results = []
    skipped = 0

    for i, img_path in enumerate(image_files[:n]):
        try:
            anno = parse_ccpd_filename(img_path.name)
        except ValueError as e:
            print(f"  SKIP {img_path.name}: {e}")
            skipped += 1
            continue

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  SKIP {img_path.name}: failed to load")
            skipped += 1
            continue

        # Run YOLO on original
        orig_dets = run_yolo(model, image)

        # Only process images where YOLO detects at least one vehicle
        if not orig_dets:
            skipped += 1
            continue

        # Warp texture onto plate
        patched_image = warp_texture_onto_image(image, texture, anno["vertices"])

        # Run YOLO on patched
        patched_dets = run_yolo(model, patched_image)

        # Match detections
        matches = match_detections(orig_dets, patched_dets)

        # Record results for each original vehicle detection
        for orig_det, patched_det in matches:
            entry = {
                "image": img_path.name,
                "class_name": orig_det["class_name"],
                "orig_confidence": orig_det["confidence"],
                "patched_confidence": patched_det["confidence"] if patched_det else 0.0,
                "confidence_delta": (
                    (patched_det["confidence"] - orig_det["confidence"])
                    if patched_det
                    else -orig_det["confidence"]
                ),
                "detection_lost": patched_det is None,
            }
            results.append(entry)

        # Save comparison image
        comp_path = output_dir / "comparisons" / f"{img_path.stem}_compare.jpg"
        save_comparison_image(image, patched_image, orig_dets, patched_dets, str(comp_path))

        processed = i + 1 - skipped
        if processed % 10 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] processed={processed}, skipped={skipped}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Images processed: {n - skipped}")
    print(f"Images skipped (no vehicle detected or parse error): {skipped}")
    print(f"Total vehicle detections compared: {len(results)}")

    if results:
        confs_orig = [r["orig_confidence"] for r in results]
        confs_patched = [r["patched_confidence"] for r in results]
        deltas = [r["confidence_delta"] for r in results]
        lost = sum(1 for r in results if r["detection_lost"])

        print(f"\nOriginal confidence:  mean={np.mean(confs_orig):.4f}, "
              f"std={np.std(confs_orig):.4f}")
        print(f"Patched confidence:   mean={np.mean(confs_patched):.4f}, "
              f"std={np.std(confs_patched):.4f}")
        print(f"Confidence delta:     mean={np.mean(deltas):.4f}, "
              f"std={np.std(deltas):.4f}")
        print(f"Detections lost:      {lost}/{len(results)} "
              f"({100*lost/len(results):.1f}%)")

        # Per-class breakdown
        classes = set(r["class_name"] for r in results)
        for cls in sorted(classes):
            cls_results = [r for r in results if r["class_name"] == cls]
            cls_deltas = [r["confidence_delta"] for r in cls_results]
            cls_lost = sum(1 for r in cls_results if r["detection_lost"])
            print(f"\n  {cls} (n={len(cls_results)}):")
            print(f"    mean delta: {np.mean(cls_deltas):.4f}")
            print(f"    lost: {cls_lost}/{len(cls_results)}")

    # Save detailed results to CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "class_name", "orig_confidence", "patched_confidence",
            "confidence_delta", "detection_lost"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to {csv_path}")

    # Save summary as JSON
    summary = {
        "n_images_processed": n - skipped,
        "n_images_skipped": skipped,
        "n_detections": len(results),
        "mean_orig_confidence": float(np.mean(confs_orig)) if results else 0,
        "mean_patched_confidence": float(np.mean(confs_patched)) if results else 0,
        "mean_confidence_delta": float(np.mean(deltas)) if results else 0,
        "detections_lost": lost if results else 0,
        "detections_lost_pct": 100 * lost / len(results) if results else 0,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
