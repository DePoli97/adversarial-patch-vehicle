from model import load_model, DATA_YAML, PROJECT_ROOT, CLASS_NAMES, IMG_SIZE, CONFIDENCE_THRESHOLD


def main():
    model = load_model()

    print("\n--- Evaluation on TEST set ---\n")
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMG_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        project=str(PROJECT_ROOT / "runs"),
        name=f"evaluation_conf{CONFIDENCE_THRESHOLD}",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:   {metrics.box.map:.4f}")
    print()

    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'mAP@50':>10} {'mAP@50-95':>10}")
    print(f"  {'-'*45}")
    for i, name in enumerate(CLASS_NAMES):
        p = metrics.box.p[i] if i < len(metrics.box.p) else 0
        r = metrics.box.r[i] if i < len(metrics.box.r) else 0
        ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
        ap50_95 = metrics.box.ap[i] if i < len(metrics.box.ap) else 0
        print(f"  {name:<15} {p:>10.4f} {r:>10.4f} {ap:>10.4f} {ap50_95:>10.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
