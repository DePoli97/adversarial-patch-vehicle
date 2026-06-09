#!/usr/bin/env bash
# Multi-seed adversarial patch training. Runs N short trainings with
# different RNG seeds, each picking a different starting point in the
# non-convex patch optimization landscape, then keeps every patch_final.pt
# + train.log under a shared OUT_ROOT for downstream evaluation.
#
# Usage:
#   bash src/yolo_chroma_attack/multi_start.sh
#
# Env overrides:
#   N_RUNS=30 EPOCHS=30 BATCH=16 LR=0.05
#   RUN_TS=20260609_014138    # which captured dataset
#   INDEX_NAME=quads_index_visible.json   # filter to YOLO-visible frames

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

N_RUNS="${N_RUNS:-30}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-16}"
LR="${LR:-0.05}"
RUN_TS="${RUN_TS:-20260609_014138}"
INDEX_NAME="${INDEX_NAME:-quads_index_visible.json}"
TOPK="${TOPK:-20}"
EXPAND="${EXPAND:-2.5}"

DATASET="data/chroma_key_dataset/capture_${RUN_TS}_marker"
OUT_ROOT="experiments/yolo_attack/multi_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"

echo "######################################################"
echo " MULTI-START patch training"
echo "  N_RUNS    : $N_RUNS"
echo "  EPOCHS    : $EPOCHS"
echo "  BATCH     : $BATCH"
echo "  LR        : $LR (cosine)"
echo "  TOPK      : $TOPK"
echo "  EXPAND    : $EXPAND"
echo "  DATASET   : $DATASET"
echo "  INDEX     : $INDEX_NAME"
echo "  OUT_ROOT  : $OUT_ROOT"
echo "######################################################"
echo ""

for seed in $(seq 0 $((N_RUNS - 1))); do
  out="$OUT_ROOT/seed${seed}"
  echo "=== seed=${seed}  (out=${out}) ==="
  python -u -m src.yolo_chroma_attack.train \
      --run-dir "$DATASET" \
      --out-dir "$out" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH" \
      --lr "$LR" \
      --cosine-lr --lr-min 1e-3 \
      --patch-h 256 --patch-w 512 \
      --topk "$TOPK" \
      --target-expand-x "$EXPAND" --target-expand-y "$EXPAND" \
      --eot-noise 0.05 \
      --index-name "$INDEX_NAME" \
      --seed "$seed" \
      --device cuda 2>&1 | tee -a "$OUT_ROOT/all_runs.log"
done

echo ""
echo "ALL ${N_RUNS} RUNS DONE → $OUT_ROOT"
