#!/usr/bin/env bash
# Full Fase 1 training grid, with illumination-aware compositing (--illum-fix):
#   6 per (road x light) : Town04/07/11 x day/night  (pool the 3 distances)
#   3 per road           : Town04/07/11             (pool day+night x 3 dist)
#   1 pooled             : all roads x day/night x dist
# The generalist (fase1 + old _014138) is trained by train_generalist_full.sh.
#
# Every run: yolov8m surrogate, topk3, margin 0.05, geom-EOT, illum-fix on.
# All logged to W&B (group). Frozen loss/EOT config — only the dataset changes.
#
# Run on Vortex (after regenerating indexes without frame cutoff):
#   conda activate PCLA15
#   bash src/yolo_chroma_attack/grid_fase1.sh
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"

EPOCHS="${EPOCHS:-30}"; BATCH="${BATCH:-24}"; LR="${LR:-0.06}"
TOPK="${TOPK:-3}"; EXPAND="${EXPAND:-2.5}"; MARGIN_TAU="${MARGIN_TAU:-0.05}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8m.pt}"; YELLOW_REF="${YELLOW_REF:-0.65}"
WANDB_PROJECT="${WANDB_PROJECT:-adversarial-patch-fase1}"
# Overridable so a relaunch can join the SAME W&B group / output dir and skip
# already-finished runs (see train_one).
WANDB_GROUP="${WANDB_GROUP:-grid_$(date +%Y%m%d_%H%M%S)}"

FASE1_ROOT="data/chroma_key_dataset/fase1"
OUT_ROOT="${OUT_ROOT:-experiments/yolo_attack/grid_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT_ROOT"
TOWNS=(Town04_spawn273 Town07_spawn38 Town11_spawn1713)
LIGHTS=(day night); DISTANCES=(dist6m dist10m dist20m)

echo "=== rebuilding quads indexes (no frame cutoff) ==="
python src/chroma_key_dataset_generator/build_fase1_indexes.py --fase1-root "$FASE1_ROOT"

train_one() {  # $1=dataset dir  $2=out dir  $3=name
  if [ -f "$2/patch_final.pt" ]; then echo "=== skip $3 (already done) ==="; return 0; fi
  echo "=== $3  (dataset=$1) ==="
  python -u -m src.yolo_chroma_attack.train \
      --run-dir "$1" --out-dir "$2" --yolo-weights "$YOLO_WEIGHTS" \
      --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" --cosine-lr --lr-min 1e-3 \
      --patch-h 256 --patch-w 512 --topk "$TOPK" --margin-tau "$MARGIN_TAU" \
      --geom-eot --tv-weight 0.0 --illum-fix --illum-yellow-ref "$YELLOW_REF" \
      --target-expand-x "$EXPAND" --target-expand-y "$EXPAND" \
      --eot-noise 0.05 --seed 0 --device cuda \
      --wandb-project "$WANDB_PROJECT" --wandb-name "$3" --wandb-group "$WANDB_GROUP" \
      2>&1 | tee -a "$OUT_ROOT/all_runs.log"
}

pool_dirs() {  # $1=out pool dir  $2..=marker dirs
  local pool_dir="$1"; shift
  rm -rf "$pool_dir"; mkdir -p "$pool_dir"
  python - "$pool_dir" "$@" <<'PYEOF'
import json, sys
from pathlib import Path
pool_dir = Path(sys.argv[1]); marker_dirs = [Path(p) for p in sys.argv[2:]]
pooled = {}
for md in marker_dirs:
    idx = md / "quads_index.json"
    if not idx.exists(): continue
    prefix = "_".join(md.parts[-4:-1]) + "_"
    for stem, entry in json.loads(idx.read_text()).items():
        new = prefix + stem
        (pool_dir / f"{new}.png").symlink_to((md / f"{stem}.png").resolve())
        pooled[new] = entry
(pool_dir / "quads_index.json").write_text(json.dumps(pooled))
print(f"pooled {len(pooled)} frames -> {pool_dir}")
PYEOF
}

ALL_DIRS=(); DAY_DIRS=(); NIGHT_DIRS=()
# ---- 6 per (road x light) + 3 per road ----
for town in "${TOWNS[@]}"; do
  road_dirs=()
  for light in "${LIGHTS[@]}"; do
    combo_dirs=()
    for dist in "${DISTANCES[@]}"; do
      d="$FASE1_ROOT/$town/$light/$dist/marker"
      [ -d "$d" ] && combo_dirs+=("$d")
    done
    pd="$FASE1_ROOT/_pooled_${town}_${light}"
    pool_dirs "$pd" "${combo_dirs[@]}"
    train_one "$pd" "$OUT_ROOT/${town}_${light}" "${town}_${light}"
    road_dirs+=("${combo_dirs[@]}")
    [ "$light" = "day" ]   && DAY_DIRS+=("${combo_dirs[@]}")
    [ "$light" = "night" ] && NIGHT_DIRS+=("${combo_dirs[@]}")
  done
  ALL_DIRS+=("${road_dirs[@]}")
  pr="$FASE1_ROOT/_pooled_${town}"
  pool_dirs "$pr" "${road_dirs[@]}"
  train_one "$pr" "$OUT_ROOT/$town" "$town"
done

# ---- 2 pooled per light: all roads day-only / night-only ----
pool_dirs "$FASE1_ROOT/_pooled_all_day" "${DAY_DIRS[@]}"
train_one "$FASE1_ROOT/_pooled_all_day" "$OUT_ROOT/pooled_all_day" "pooled_all_day"
pool_dirs "$FASE1_ROOT/_pooled_all_night" "${NIGHT_DIRS[@]}"
train_one "$FASE1_ROOT/_pooled_all_night" "$OUT_ROOT/pooled_all_night" "pooled_all_night"

# ---- 1 pooled (all roads, day+night) ----
pool_dirs "$FASE1_ROOT/_pooled_all" "${ALL_DIRS[@]}"
train_one "$FASE1_ROOT/_pooled_all" "$OUT_ROOT/pooled" "pooled_all"

echo "GRID DONE (6 road-light + 3 road + 2 pooled-per-light + 1 pooled = 12) -> $OUT_ROOT"
echo "Next: bash src/yolo_chroma_attack/train_generalist_full.sh (also --illum-fix)"
