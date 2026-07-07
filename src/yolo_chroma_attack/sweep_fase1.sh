#!/usr/bin/env bash
# Fase 1 sweep: train an adversarial patch against YOLOv8 on each
# (town, light) combo of the multi-condition dataset, plus one pooled
# run trained on all combos together. Same hyperparameters as the
# validated June run (multi_start.sh) — only the dataset changes.
#
# Logs every run to Weights & Biases (project + group) so all 7 curves
# can be compared side by side on wandb.ai.
#
# Usage:
#   wandb login   # once, paste your API key
#   bash src/yolo_chroma_attack/sweep_fase1.sh
#
# Env overrides:
#   EPOCHS=30 BATCH=16 LR=0.05 TOPK=20 EXPAND=2.5
#   WANDB_PROJECT=adversarial-patch-fase1

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-16}"
LR="${LR:-0.05}"
TOPK="${TOPK:-20}"
EXPAND="${EXPAND:-2.5}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8m.pt}"   # m detects the clean carlacola; n/s/x do not
MARGIN_TAU="${MARGIN_TAU:-0.2}"              # C&W hinge: stop once conf < detection threshold
WANDB_PROJECT="${WANDB_PROJECT:-adversarial-patch-fase1}"
WANDB_GROUP="sweep_$(date +%Y%m%d_%H%M%S)"

FASE1_ROOT="data/chroma_key_dataset/fase1"
OUT_ROOT="experiments/yolo_attack/fase1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"

COMBOS=(
  "Town04_spawn273 day"
  "Town04_spawn273 night"
  "Town07_spawn38 day"
  "Town07_spawn38 night"
  "Town11_spawn1713 day"
  "Town11_spawn1713 night"
)

echo "######################################################"
echo " FASE 1 sweep — patch vs YOLO"
echo "  WEIGHTS: $YOLO_WEIGHTS   MARGIN_TAU: $MARGIN_TAU"
echo "  EPOCHS/BATCH/LR/TOPK/EXPAND: $EPOCHS/$BATCH/$LR/$TOPK/$EXPAND"
echo "  WANDB_PROJECT : $WANDB_PROJECT"
echo "  WANDB_GROUP   : $WANDB_GROUP"
echo "  OUT_ROOT      : $OUT_ROOT"
echo "######################################################"
echo ""

# Build all quads_index.json: detect the marker on the DAY frame (HSV, restricted
# to the truck box from marker_day-noleader_day diff) and transfer the identical
# corners to the deterministic NIGHT twin. Also applies the per-town frame cutoff.
echo "=== building quads indexes (day detect + night transfer) ==="
python src/chroma_key_dataset_generator/build_fase1_indexes.py --fase1-root "$FASE1_ROOT"

train_one() {
  # $1 = run-dir  $2 = out-dir  $3 = wandb run name
  local run_dir="$1" out_dir="$2" name="$3"
  echo "=== $name  (dataset=${run_dir}  out=${out_dir}) ==="
  python -u -m src.yolo_chroma_attack.train \
      --run-dir "$run_dir" \
      --out-dir "$out_dir" \
      --yolo-weights "$YOLO_WEIGHTS" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH" \
      --lr "$LR" \
      --cosine-lr --lr-min 1e-3 \
      --patch-h 256 --patch-w 512 \
      --topk "$TOPK" \
      --margin-tau "$MARGIN_TAU" \
      --target-expand-x "$EXPAND" --target-expand-y "$EXPAND" \
      --eot-noise 0.05 \
      --seed 0 \
      --device cuda \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-name "$name" \
      --wandb-group "$WANDB_GROUP" \
      2>&1 | tee -a "$OUT_ROOT/all_runs.log"
}

# ---- 6 per-combo runs ----
for combo in "${COMBOS[@]}"; do
  read -r town light <<< "$combo"
  marker_dir="$FASE1_ROOT/$town/$light/marker"
  name="${town}_${light}"
  train_one "$marker_dir" "$OUT_ROOT/$name" "$name"
done

# ---- 1 pooled run: symlink all marker dirs' frames + merge quads_index ----
POOLED_DIR="$FASE1_ROOT/_pooled_marker"
if [ ! -d "$POOLED_DIR" ]; then
  echo "=== building pooled dataset -> $POOLED_DIR ==="
  mkdir -p "$POOLED_DIR"
  python - <<'PYEOF'
import json
from pathlib import Path

fase1_root = Path("data/chroma_key_dataset/fase1")
pooled_dir = fase1_root / "_pooled_marker"
combos = [
    ("Town04_spawn273", "day"), ("Town04_spawn273", "night"),
    ("Town07_spawn38", "day"), ("Town07_spawn38", "night"),
    ("Town11_spawn1713", "day"), ("Town11_spawn1713", "night"),
]

pooled_index = {}
for town, light in combos:
    marker_dir = fase1_root / town / light / "marker"
    index_path = marker_dir / "quads_index.json"
    with open(index_path) as f:
        index = json.load(f)
    prefix = f"{town}_{light}_"
    for stem, entry in index.items():
        new_stem = prefix + stem
        (pooled_dir / f"{new_stem}.png").symlink_to(
            (marker_dir / f"{stem}.png").resolve())
        pooled_index[new_stem] = entry

with open(pooled_dir / "quads_index.json", "w") as f:
    json.dump(pooled_index, f)
print(f"pooled dataset: {len(pooled_index)} frames -> {pooled_dir}")
PYEOF
fi
train_one "$POOLED_DIR" "$OUT_ROOT/pooled" "pooled_all"

echo ""
echo "ALL 7 RUNS DONE -> $OUT_ROOT"
echo "Compare on wandb.ai: project=$WANDB_PROJECT group=$WANDB_GROUP"
