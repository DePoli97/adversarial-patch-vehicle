from pathlib import Path
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML    = PROJECT_ROOT / "data" / "data.yaml"
WEIGHTS_DIR  = PROJECT_ROOT / "runs" / "vehicle_detection" / "weights"
BEST_MODEL   = WEIGHTS_DIR / "best.pt"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── Dataset splits ───────────────────────────────────────────────────────────────
TRAIN_IMAGES = PROJECT_ROOT / "data" / "train" / "images"
VAL_IMAGES   = PROJECT_ROOT / "data" / "valid" / "images"
TEST_IMAGES  = PROJECT_ROOT / "data" / "test" / "images"

# ── Balanced dataset (dopo EDA + undersampling) ─────────────────────────
BALANCED_DATA_YAML   = PROJECT_ROOT / "data" / "balanced_data.yaml"
BALANCED_WEIGHTS_DIR = PROJECT_ROOT / "runs" / "detect" / "runs" / "vehicle_detection_balanced" / "weights"
BALANCED_BEST_MODEL  = BALANCED_WEIGHTS_DIR / "best.pt"
BALANCED_TEST_IMAGES = PROJECT_ROOT / "data" / "balanced" / "test" / "images"

# ── Model configuration ─────────────────────────────────────────────────────
BASE_MODEL           = "yolov8n.pt"
CLASS_NAMES          = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]
NUM_CLASSES          = len(CLASS_NAMES)
IMG_SIZE             = 416
CONFIDENCE_THRESHOLD = 0.25

# ── Training configuration ──────────────────────────────────────────────────
EPOCHS     = 100
BATCH_SIZE = 16
PATIENCE   = 10


def load_model(weights: str | Path | None = None) -> YOLO:
    if weights is None:
        path = BEST_MODEL
    elif str(weights) == "base":
        path = BASE_MODEL
    else:
        path = Path(weights)

    if path != Path(BASE_MODEL) and not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}.\n"
        )

    model = YOLO(str(path))
    return model