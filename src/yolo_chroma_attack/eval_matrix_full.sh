#!/usr/bin/env bash
# Fair comparison: every Fase-1 grid patch (12) + the generalist, each evaluated
# on every road's held-out val split (same test set per road), WITH the
# illumination fix (deployment-realistic — matches how they were trained).
# Metric = detection_rate (lower = better attack).
#
# Run on Vortex:  conda activate PCLA15; bash src/yolo_chroma_attack/eval_matrix_full.sh
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
GRID="experiments/yolo_attack/grid_20260711_205811"
GEN="experiments/yolo_attack/generalist_full_20260711_231933/generalist_full"
OUT="experiments/yolo_attack/eval_full_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
YW="yolov8m.pt"

# patch label -> patch path
declare -A PATCH
for r in pooled pooled_all_day pooled_all_night \
         Town04_spawn273 Town04_spawn273_day Town04_spawn273_night \
         Town07_spawn38 Town07_spawn38_day Town07_spawn38_night \
         Town11_spawn1713 Town11_spawn1713_day Town11_spawn1713_night; do
  PATCH[$r]="$GRID/$r/patch_final.pt"
done
PATCH[generalist]="$GEN/patch_final.pt"

# road -> test dataset (that road's pooled day+night+distances)
declare -A ROAD
ROAD[Town04]="data/chroma_key_dataset/fase1/_pooled_Town04_spawn273"
ROAD[Town07]="data/chroma_key_dataset/fase1/_pooled_Town07_spawn38"
ROAD[Town11]="data/chroma_key_dataset/fase1/_pooled_Town11_spawn1713"

for pl in "${!PATCH[@]}"; do
  pp="${PATCH[$pl]}"
  [ -f "$pp" ] || { echo "SKIP $pl (no file)"; continue; }
  for rl in Town04 Town07 Town11; do
    o="$OUT/patch_${pl}__road_${rl}.json"
    [ -f "$o" ] && continue
    echo "=== patch=$pl road=$rl ==="
    python -m src.yolo_chroma_attack.evaluate \
      --run-dir "${ROAD[$rl]}" --patch "$pp" --yolo-weights "$YW" \
      --illum-fix --illum-yellow-ref 0.65 \
      --target-expand-x 2.5 --target-expand-y 2.5 \
      --batch-size 8 --device cuda --save-previews 0 --out "$o" \
      2>&1 | grep -E "trained|clean" | tail -2
  done
done
echo "matrix JSONs -> $OUT"
python src/yolo_chroma_attack/show_matrix_full.py "$OUT"
