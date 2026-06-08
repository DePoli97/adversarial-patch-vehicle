#!/usr/bin/env bash
# Generate three paired datasets in one go, all with the same RUN_TS + SEED
# so scenes line up frame-for-frame:
#
#   capture_<ts>_marker     CarlaCola with yellow chroma-key marker (training)
#   capture_<ts>_clean      CarlaCola with original red skin       (true clean baseline)
#   capture_<ts>_noleader   Scene without CarlaCola at all         (OpenCLIP target)
#
# Two CARLA packages are required (built once via UE Editor + `make package`):
#   - PACKAGE_MARKER : skin with yellow marker on the rear panel
#   - PACKAGE_CLEAN  : skin with the original red CarlaCola texture
#
# Edit the two PACKAGE_* paths below for your machine and run:
#   bash src/chroma_key_dataset_generator/generate_all_datasets.sh
#
# To do a quick paired smoke test first (e.g. 4 frames/town = 20 frames per
# dataset = 60 total), override FRAMES_PER_TOWN:
#   FRAMES_PER_TOWN=4 bash src/chroma_key_dataset_generator/generate_all_datasets.sh

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# === Edit these paths to point at your two CARLA packages ===
PACKAGE_MARKER="${PACKAGE_MARKER:-/home/vortex/carla/Dist/CARLA_Shipping_0.9.15.2_marker_yellow/LinuxNoEditor}"
PACKAGE_CLEAN="${PACKAGE_CLEAN:-/home/vortex/carla/Dist/CARLA_Shipping_0.9.15.2_clean/LinuxNoEditor}"

# Shared timestamp & seed for all three runs.
export RUN_TS="$(date '+%Y%m%d_%H%M%S')"
export SEED="${SEED:-42}"
export FRAMES_PER_TOWN="${FRAMES_PER_TOWN:-200}"
export MODE="auto"

echo "###################################################################"
echo " GENERATING THREE PAIRED DATASETS"
echo "###################################################################"
echo "  shared RUN_TS  : $RUN_TS"
echo "  shared SEED    : $SEED"
echo "  frames/town    : $FRAMES_PER_TOWN"
echo "  package marker : $PACKAGE_MARKER"
echo "  package clean  : $PACKAGE_CLEAN"
echo "###################################################################"
echo ""

if [[ ! -x "${PACKAGE_MARKER}/CarlaUE4.sh" ]]; then
  echo "[ERR] marker package not found at ${PACKAGE_MARKER}/CarlaUE4.sh"
  exit 1
fi
if [[ ! -x "${PACKAGE_CLEAN}/CarlaUE4.sh" ]]; then
  echo "[ERR] clean package not found at ${PACKAGE_CLEAN}/CarlaUE4.sh"
  exit 1
fi

run_dataset () {
  local name="$1"; shift
  local package="$1"; shift
  local extra_env="$1"; shift  # e.g. "NO_LEADER=1" or empty
  echo ""
  echo "==================================================================="
  echo " DATASET: $name  (package=$(basename "$package"))"
  echo "==================================================================="
  if [[ -n "$extra_env" ]]; then
    env DATASET_NAME="$name" CARLA_PACKAGE_DIR="$package" $extra_env \
      bash src/chroma_key_dataset_generator/test_capture.sh
  else
    env DATASET_NAME="$name" CARLA_PACKAGE_DIR="$package" \
      bash src/chroma_key_dataset_generator/test_capture.sh
  fi
}

# 1. marker (yellow patch) – training input for the adversarial patch
run_dataset marker   "$PACKAGE_MARKER" ""

# 2. clean (original red skin) – true detection-rate baseline
run_dataset clean    "$PACKAGE_CLEAN"  ""

# 3. no-leader (clean package, leader skipped) – OpenCLIP target embedding
run_dataset noleader "$PACKAGE_CLEAN"  "NO_LEADER=1"

echo ""
echo "###################################################################"
echo " ALL THREE DATASETS DONE"
echo "###################################################################"
echo "  marker   : data/chroma_key_dataset/capture_${RUN_TS}_marker/"
echo "  clean    : data/chroma_key_dataset/capture_${RUN_TS}_clean/"
echo "  noleader : data/chroma_key_dataset/capture_${RUN_TS}_noleader/"
