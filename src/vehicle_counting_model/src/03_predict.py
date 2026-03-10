"""
Usage:
    python src/03_predict.py
    python src/03_predict.py --source path/to/image.jpg
"""

import argparse
from pathlib import Path
from collections import Counter

from model import load_model, CLASS_NAMES, RESULTS_DIR, TEST_IMAGES, IMG_SIZE, CONFIDENCE_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description="Vehicle detection and counting")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to image or folder. Defaults to test set.")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    args = parser.parse_args()

    model  = load_model()  # loads best.pt
    source = args.source if args.source else str(TEST_IMAGES)

    print(f"Source:     {source}")
    print(f"Confidence: {args.conf}\n")

    # Fai inferenza
    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=IMG_SIZE,
        save=True,
        project=str(RESULTS_DIR),
        name="predictions",
        exist_ok=True, # sovrascrivi
    )

    # Count vehicles across all images
    total_counts = Counter()
    print(f"\n{'='*60}")
    print("DETECTION RESULTS (per image)")
    print(f"{'='*60}")

    for r in results:
        img_name = Path(r.path).name
        img_counts = Counter()
        for box in r.boxes:
            cls_id = int(box.cls[0])
            img_counts[CLASS_NAMES[cls_id]] += 1
            total_counts[CLASS_NAMES[cls_id]] += 1

        total_in_img = sum(img_counts.values())
        breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(img_counts.items()))
        if total_in_img > 0:
            print(f"  {img_name:<45} -> {total_in_img} vehicles ({breakdown})")
        else:
            print(f"  {img_name:<45} -> 0 vehicles")

    # Overall summary
    grand_total = sum(total_counts.values())
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total images processed: {len(results)}")
    print(f"  Total vehicles found:   {grand_total}\n")
    for cls_name in CLASS_NAMES:
        count = total_counts.get(cls_name, 0)
        print(f"  {cls_name:<15} {count:>5}")
    print(f"{'='*60}")
    print(f"\nAnnotated images saved to: {RESULTS_DIR / 'predictions'}")


if __name__ == "__main__":
    main()
