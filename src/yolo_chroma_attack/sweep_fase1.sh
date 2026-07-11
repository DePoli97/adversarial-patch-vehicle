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
# Batch 32 (up from 16): the per-road datasets are now large (666-906 frames,
# ~2300 pooled) so a bigger batch uses the 20 GB GPU better; LR scaled up ~sqrt.
BATCH="${BATCH:-32}"
LR="${LR:-0.07}"
# Winning config from the 2026-07-07 Town04_day ablation:
#   topk=3  (attack the strongest anchors, not a diluted top-20 mean)
#   margin-tau=0.05  (hinge just above noise floor, not the crippling 0.2)
#   geom-eot  (the decisive lever: rotation/scale/translation forces the patch
#              to a spatially-coherent structured pattern instead of the
#              high-frequency noise the low-pose-variety dataset overfits to)
# TV loss helped smoothness but blurred the pattern with no perf gain, so it's
# off — geom alone reaches 99% confidence drop / 100% frames hidden.
TOPK="${TOPK:-3}"
EXPAND="${EXPAND:-2.5}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8m.pt}"   # m detects the clean carlacola; n/s/x do not
MARGIN_TAU="${MARGIN_TAU:-0.05}"
GEOM_EOT="${GEOM_EOT:-1}"                    # 1 = enable geometric EOT
TV_WEIGHT="${TV_WEIGHT:-0.0}"
WANDB_PROJECT="${WANDB_PROJECT:-adversarial-patch-fase1}"
WANDB_GROUP="sweep_$(date +%Y%m%d_%H%M%S)"

FASE1_ROOT="data/chroma_key_dataset/fase1"
OUT_ROOT="experiments/yolo_attack/fase1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"

# One patch per ROAD: the whole road context (day + night + all distances)
# pooled together. This tests Tonella's hypothesis that a patch trained
# specifically on the final deployment context beats a generalist patch. Do NOT
# split by lighting — a road's context includes both day and night.
TOWNS=(
  "Town04_spawn273"
  "Town07_spawn38"
  "Town11_spawn1713"
)
LIGHTS=(day night)

echo "######################################################"
echo " FASE 1 sweep — patch vs YOLO"
echo "  WEIGHTS: $YOLO_WEIGHTS   MARGIN_TAU: $MARGIN_TAU   GEOM_EOT: $GEOM_EOT   TV: $TV_WEIGHT"
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
  local geom_flag=""
  [ "$GEOM_EOT" = "1" ] && geom_flag="--geom-eot"
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
      $geom_flag \
      --tv-weight "$TV_WEIGHT" \
      --target-expand-x "$EXPAND" --target-expand-y "$EXPAND" \
      --eot-noise 0.05 \
      --seed 0 \
      --device cuda \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-name "$name" \
      --wandb-group "$WANDB_GROUP" \
      2>&1 | tee -a "$OUT_ROOT/all_runs.log"
}

# Pool a list of marker dirs (each a "town/light/distXm/marker") into one
# symlinked dataset dir with a merged quads_index.json. Used both to fuse the
# 3 distances of a single combo, and to fuse all 6 combos into the final
# pooled-all baseline.
pool_dirs() {
  # $1 = output pool dir   $2.. = source marker dirs
  local pool_dir="$1"; shift
  if [ -d "$pool_dir" ]; then return 0; fi
  mkdir -p "$pool_dir"
  python - "$pool_dir" "$@" <<'PYEOF'
import json, sys
from pathlib import Path

pool_dir = Path(sys.argv[1])
marker_dirs = [Path(p) for p in sys.argv[2:]]

pooled_index = {}
for marker_dir in marker_dirs:
    index_path = marker_dir / "quads_index.json"
    if not index_path.exists():
        continue
    with open(index_path) as f:
        index = json.load(f)
    # marker_dir looks like .../<town>/<light>/<distXm>/marker -> use the 3
    # parent components as a unique prefix so frames from different
    # distances/combos never collide.
    prefix = "_".join(marker_dir.parts[-4:-1]) + "_"
    for stem, entry in index.items():
        new_stem = prefix + stem
        (pool_dir / f"{new_stem}.png").symlink_to((marker_dir / f"{stem}.png").resolve())
        pooled_index[new_stem] = entry

with open(pool_dir / "quads_index.json", "w") as f:
    json.dump(pooled_index, f)
print(f"pooled dataset: {len(pooled_index)} frames -> {pool_dir}")
PYEOF
}

DISTANCES=(dist6m dist10m dist20m)

# ---- 3 per-ROAD runs: pool day+night x all distances for each town ----
ALL_MARKER_DIRS=()
for town in "${TOWNS[@]}"; do
  road_dirs=()
  for light in "${LIGHTS[@]}"; do
    for dist in "${DISTANCES[@]}"; do
      d="$FASE1_ROOT/$town/$light/$dist/marker"
      [ -d "$d" ] && road_dirs+=("$d")
    done
  done
  ALL_MARKER_DIRS+=("${road_dirs[@]}")
  pool_dir="$FASE1_ROOT/_pooled_${town}"
  echo "=== pooling day+night+distances for ${town} -> $pool_dir ==="
  pool_dirs "$pool_dir" "${road_dirs[@]}"
  train_one "$pool_dir" "$OUT_ROOT/$town" "$town"
done

# ---- 1 generalist run: all roads x lighting x distances together ----
POOLED_DIR="$FASE1_ROOT/_pooled_all"
echo "=== pooling ALL roads+lighting+distances -> $POOLED_DIR ==="
pool_dirs "$POOLED_DIR" "${ALL_MARKER_DIRS[@]}"
train_one "$POOLED_DIR" "$OUT_ROOT/pooled" "pooled_all"

echo ""
echo "ALL 4 RUNS DONE (3 per-road + 1 generalist) -> $OUT_ROOT"
echo "Compare on wandb.ai: project=$WANDB_PROJECT group=$WANDB_GROUP"
