#!/usr/bin/env bash
# Train THE generalist baseline patch: the widest possible dataset.
#
# This is the patch we compare the per-road patches against (Tonella's question:
# does a road-specific patch beat a generalist trained on everything?). The
# generalist must therefore pool EVERY dataset we have:
#   - capture_20260609_014138  (the big June capture: many cities, random NPCs,
#     random spots — the "gigante" varied dataset)
#   - fase1/  (all 3 roads x day/night x 3 distances)
#
# The fase1-only "_pooled_all" run in sweep_fase1.sh is NOT this generalist — it
# is only a fase1 baseline. This script builds the real generalist.
#
# Note on corners: _014138 already ships a valid quads_index.json (used for the
# working June patch); fase1 corners come from build_fase1_indexes.py. Each frame
# keeps its own valid corners, so the patch warps correctly on every frame — we
# do NOT re-detect. Same winning hyperparameters as the per-road sweep.
#
# Run on Vortex, AFTER the per-road sweep frees the GPU:
#   conda activate PCLA15
#   cd /home/vortex/adversarial-patch-vehicle
#   bash src/yolo_chroma_attack/train_generalist_full.sh

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-24}"
LR="${LR:-0.06}"
TOPK="${TOPK:-3}"
EXPAND="${EXPAND:-2.5}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8m.pt}"
MARGIN_TAU="${MARGIN_TAU:-0.05}"
TV_WEIGHT="${TV_WEIGHT:-0.0}"
WANDB_PROJECT="${WANDB_PROJECT:-adversarial-patch-fase1}"
WANDB_GROUP="${WANDB_GROUP:-generalist_full_$(date +%Y%m%d_%H%M%S)}"

DATA_ROOT="data/chroma_key_dataset"
FASE1_ROOT="$DATA_ROOT/fase1"
OUT_ROOT="experiments/yolo_attack/generalist_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"

# --- collect every marker dir: the big June capture + all fase1 roads ---
MARKER_DIRS=("$DATA_ROOT/capture_20260609_014138_marker")
for town in Town04_spawn273 Town07_spawn38 Town11_spawn1713; do
  for light in day night; do
    for dist in dist6m dist10m dist20m; do
      d="$FASE1_ROOT/$town/$light/$dist/marker"
      [ -d "$d" ] && MARKER_DIRS+=("$d")
    done
  done
done

echo "######################################################"
echo " GENERALIST FULL — patch vs YOLO on EVERY dataset"
echo "  marker dirs: ${#MARKER_DIRS[@]}"
for d in "${MARKER_DIRS[@]}"; do echo "    - $d"; done
echo "######################################################"

# --- pool them: symlink frames + merge quads_index.json with a unique prefix ---
POOL_DIR="$FASE1_ROOT/_pooled_generalist_full"
rm -rf "$POOL_DIR"; mkdir -p "$POOL_DIR"
python - "$POOL_DIR" "${MARKER_DIRS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path

pool_dir = Path(sys.argv[1])
marker_dirs = [Path(p) for p in sys.argv[2:]]
pooled = {}
for md in marker_dirs:
    idx = md / "quads_index.json"
    if not idx.exists():
        print(f"  SKIP (no quads_index): {md}")
        continue
    index = json.loads(idx.read_text())
    # fase1 marker dir: .../<town>/<light>/<distXm>/marker -> town_light_dist_
    # flat capture dir:  .../capture_<ts>_marker            -> capture_<ts>_
    parts = md.parts
    if parts[-1] == "marker" and len(parts) >= 4 and parts[-4].startswith("Town"):
        prefix = "_".join(parts[-4:-1]) + "_"
    else:
        prefix = md.name + "_"
    for stem, entry in index.items():
        new = prefix + stem
        src = (md / f"{stem}.png").resolve()
        if not src.exists():
            continue
        (pool_dir / f"{new}.png").symlink_to(src)
        pooled[new] = entry
    print(f"  + {len([k for k in pooled if k.startswith(prefix)])} frames from {md}")

(pool_dir / "quads_index.json").write_text(json.dumps(pooled))
print(f"\nGENERALIST pool: {len(pooled)} frames -> {pool_dir}")
PYEOF

# --- train ---
python -u -m src.yolo_chroma_attack.train \
    --run-dir "$POOL_DIR" \
    --out-dir "$OUT_ROOT/generalist_full" \
    --yolo-weights "$YOLO_WEIGHTS" \
    --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
    --cosine-lr --lr-min 1e-3 \
    --patch-h 256 --patch-w 512 \
    --topk "$TOPK" --margin-tau "$MARGIN_TAU" --geom-eot --tv-weight "$TV_WEIGHT" \
    --illum-fix --illum-yellow-ref "${YELLOW_REF:-0.65}" \
    --target-expand-x "$EXPAND" --target-expand-y "$EXPAND" \
    --eot-noise 0.05 --seed 0 --device cuda \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "generalist_full" \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee -a "$OUT_ROOT/train.log"

echo ""
echo "GENERALIST FULL DONE -> $OUT_ROOT"
