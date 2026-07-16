#!/usr/bin/env bash
# Evaluate every patch on every road's val split, to answer Tonella's question:
# does the road-specific patch beat the generalist baselines ON its own road?
#
# Rows = patches (3 per-road + pooled-fase1 + generalist-full).
# Cols = roads (Town04, Town07, Town11), each pooled day+night+distances.
# Metric = detection_rate (lower = better attack: YOLO fails to see the vehicle).
#
# Run on Vortex:
#   conda activate PCLA15
#   bash src/yolo_chroma_attack/eval_matrix.sh
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"

SWEEP="experiments/yolo_attack/fase1_20260711_153141"
GEN_DIR=$(ls -d experiments/yolo_attack/generalist_full_*/generalist_full 2>/dev/null | head -1)
OUT="experiments/yolo_attack/eval_matrix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
YW="yolov8m.pt"

# patch label -> patch path
declare -A PATCH
PATCH[Town04]="$SWEEP/Town04_spawn273/patch_final.pt"
PATCH[Town07]="$SWEEP/Town07_spawn38/patch_final.pt"
PATCH[Town11]="$SWEEP/Town11_spawn1713/patch_final.pt"
PATCH[pooled]="$SWEEP/pooled/patch_final.pt"
[ -n "$GEN_DIR" ] && PATCH[generalist]="$GEN_DIR/patch_final.pt"

# road label -> pooled dataset dir (day+night+distances for that road)
declare -A ROAD
ROAD[Town04]="data/chroma_key_dataset/fase1/_pooled_Town04_spawn273"
ROAD[Town07]="data/chroma_key_dataset/fase1/_pooled_Town07_spawn38"
ROAD[Town11]="data/chroma_key_dataset/fase1/_pooled_Town11_spawn1713"

for pl in Town04 Town07 Town11 pooled generalist; do
  pp="${PATCH[$pl]:-}"
  [ -z "$pp" ] && { echo "SKIP patch $pl (not ready)"; continue; }
  [ -f "$pp" ] || { echo "SKIP patch $pl (no file: $pp)"; continue; }
  for rl in Town04 Town07 Town11; do
    echo "=== patch=$pl  road=$rl ==="
    python -m src.yolo_chroma_attack.evaluate \
      --run-dir "${ROAD[$rl]}" \
      --patch "$pp" \
      --yolo-weights "$YW" \
      --target-expand-x 2.5 --target-expand-y 2.5 \
      --batch-size 8 --device cuda \
      --out "$OUT/patch_${pl}__road_${rl}.json" 2>&1 | grep -E "trained|clean|random|val \(marker\)"
  done
done

echo ""
echo "=== MATRIX (detection_rate trained, lower=better attack) ==="
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
roads = ["Town04","Town07","Town11"]
patches = ["Town04","Town07","Town11","pooled","generalist"]
print(f"{'patch \\ road':14s} " + " ".join(f"{r:>9s}" for r in roads))
for p in patches:
    cells=[]
    for r in roads:
        f = out / f"patch_{p}__road_{r}.json"
        if not f.exists(): cells.append("   -   "); continue
        d=json.loads(f.read_text())
        tr=d.get("trained",{}).get("detection_rate")
        cells.append(f"{tr*100:6.1f}% " if tr is not None else "   ?   ")
    print(f"{p:14s} " + " ".join(f"{c:>9s}" for c in cells))
print("\nDiagonal (patch trained on its OWN road) vs pooled/generalist same column:")
for r in roads:
    own=out/f"patch_{r}__road_{r}.json"
    pool=out/f"patch_pooled__road_{r}.json"
    gen=out/f"patch_generalist__road_{r}.json"
    def rate(f):
        return json.loads(f.read_text())["trained"]["detection_rate"]*100 if f.exists() else None
    o,pl,g = rate(own),rate(pool),rate(gen)
    line=f"  {r}: own={o:.1f}%" if o is not None else f"  {r}: own=?"
    if pl is not None: line+=f"  pooled={pl:.1f}%"
    if g is not None: line+=f"  generalist={g:.1f}%"
    print(line)
PY
echo ""
echo "matrix JSONs -> $OUT"
