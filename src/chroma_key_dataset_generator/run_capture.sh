#!/usr/bin/env bash
# Capture still frames of the leader vehicle wearing the yellow chroma-key marker.
#
# Prerequisites (on Vortex):
#   Terminal 1: cd /home/vortex/carla && conda activate carla && make launch
#               -> press Play in the Unreal Editor (server listens on :2000)
#               -> Reimport assets/chroma_key/rear_window_yellow.TGA on the
#                  target vehicle's glass material slot (Element 4)
#   Terminal 2: conda activate PCLA310 && cd /home/vortex/adversarial-patch-vehicle
#               bash src/chroma_key_dataset_generator/run_capture.sh
#
# Output:
#   data/chroma_key_dataset/capture_<timestamp>/
#       0001.png + 0001.json   per frame
#       run_capture.log         aggregated stdout/stderr

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SCRIPT="src/chroma_key_dataset_generator/capture_frames.py"

# Defaults; override via env, e.g.:
#   TOWNS="Town06 Town04" bash src/chroma_key_dataset_generator/run_capture.sh
#   LEADER="vehicle.nissan.micra" bash src/chroma_key_dataset_generator/run_capture.sh
TOWNS="${TOWNS:-Town06}"
WEATHER="${WEATHER:-ClearNoon WetCloudyNoon HardRainNoon}"
SUN_ALTITUDES="${SUN_ALTITUDES:-60 20 -10}"
DISTANCES="${DISTANCES:-8 15 25}"
LEADER="${LEADER:-vehicle.carlamotors.carlacola}"
FOLLOWER="${FOLLOWER:-vehicle.tesla.model3}"

TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="data/chroma_key_dataset"
LOG="${LOG_DIR}/run_capture_${TS}.log"
mkdir -p "$LOG_DIR"

{
    echo "################################################################"
    echo "# CHROMA-KEY CAPTURE — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"
    echo "  towns      : ${TOWNS}"
    echo "  weather    : ${WEATHER}"
    echo "  sun_alts   : ${SUN_ALTITUDES}"
    echo "  distances  : ${DISTANCES}"
    echo "  leader     : ${LEADER}"
    echo "  follower   : ${FOLLOWER}"
    echo "  log        : ${LOG}"
    echo ""
} | tee -a "$LOG"

# shellcheck disable=SC2086  # we want word-splitting on the env vars
if python "$SCRIPT" \
        --towns ${TOWNS} \
        --weather ${WEATHER} \
        --sun-altitudes ${SUN_ALTITUDES} \
        --distances ${DISTANCES} \
        --leader "${LEADER}" \
        --follower "${FOLLOWER}" \
        2>&1 | tee -a "$LOG"; then
    echo "[$(date '+%H:%M:%S')] DONE" | tee -a "$LOG"
else
    echo "[$(date '+%H:%M:%S')] FAILED (see $LOG)" | tee -a "$LOG"
    exit 1
fi
