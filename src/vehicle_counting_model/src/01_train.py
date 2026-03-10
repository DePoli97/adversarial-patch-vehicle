from model import (
    load_model, BALANCED_DATA_YAML, PROJECT_ROOT,
    EPOCHS, IMG_SIZE, BATCH_SIZE, PATIENCE,
)


def main():
    print(f"Dataset:      {BALANCED_DATA_YAML}")
    print(f"Epochs:       {EPOCHS}")
    print(f"Image size:   {IMG_SIZE}")
    print()

    model = load_model("base")

    # Fine-tune sul dataset bilanciato (dopo EDA + undersampling di Car)
    model.train(
        data=str(BALANCED_DATA_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(PROJECT_ROOT / "runs"),
        name="vehicle_detection_balanced_100ep",
        exist_ok=True,
        verbose=True,
        device="mps"
    )

    print("\nTraining completed")
    print(f"Best model saved at: {PROJECT_ROOT / 'runs' / 'vehicle_detection_balanced' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
