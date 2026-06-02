#!/usr/bin/env bash
# Capture still frames of the leader vehicle wearing the yellow chroma-key marker.
#
# Prerequisites (on Vortex):
#   Terminal 1: cd /home/vortex/carla && conda activate carla && make launch
#               -> press Play in the Unreal Editor (server listens on :2000)
#               -> Reimport assets/chroma_key/*.TGA on the target vehicle in UE
#   Terminal 2: conda activate PCLA310 && cd /home/vortex/adversarial-patch-vehicle
#               bash src/chroma_key_dataset_generator/run_capture.sh
#
# Default grid (~1000 frames):
#   3 towns × 4 spawn-pts × 4 weather × 4 sun × 4 distances × 3 lateral × 3 heading
#   = 13824 combos. With --max-frames 1000 we stop at the first 1000.
#
# Override any axis via env, e.g.:
#   TOWNS="Town06" WEATHER="ClearNoon" bash run_capture.sh
#   MAX_FRAMES=500 bash run_capture.sh   # quick run
#
# Output:
#   data/chroma_key_dataset/capture_<ts>/
#       NNNNNN.png + NNNNNN.json   per frame
#       captures_index.csv         summary of all frames
#       run_capture.log            aggregated stdout/stderr

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SCRIPT="src/chroma_key_dataset_generator/capture_frames.py"

# Defaults (no darkness; daytime variations only).
TOWNS="${TOWNS:-Town04 Town06 Town10HD_Opt}"
WEATHER="${WEATHER:-ClearNoon CloudyNoon WetNoon MidRainyNoon}"
SUN_ALTITUDES="${SUN_ALTITUDES:-70 55 40 25}"
DISTANCES="${DISTANCES:-6 10 14 18}"
LATERAL_OFFSETS="${LATERAL_OFFSETS:--1.5 0 1.5}"
HEADING_OFFSETS="${HEADING_OFFSETS:--5 0 5}"
NPC_COUNT="${NPC_COUNT:-0}"
NPC_RADIUS="${NPC_RADIUS:-60}"
SPAWN_POOL_SIZE="${SPAWN_POOL_SIZE:-4}"
LEADER="${LEADER:-vehicle.carlamotors.carlacola}"
FOLLOWER="${FOLLOWER:-vehicle.tesla.model3}"
MAX_FRAMES="${MAX_FRAMES:-1000}"
SEED="${SEED:-0}"
# Set SHUFFLE=1 to sample combos randomly (recommended when MAX_FRAMES < full grid).
SHUFFLE="${SHUFFLE:-1}"

TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="data/chroma_key_dataset"
LOG="${LOG_DIR}/run_capture_${TS}.log"
mkdir -p "$LOG_DIR"

{
    echo "################################################################"
    echo "# CHROMA-KEY CAPTURE — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"
    echo "  towns          : ${TOWNS}"
    echo "  weather        : ${WEATHER}"
    echo "  sun_alts       : ${SUN_ALTITUDES}"
    echo "  distances      : ${DISTANCES}"
    echo "  lateral_offs   : ${LATERAL_OFFSETS}"
    echo "  heading_offs   : ${HEADING_OFFSETS}"
    echo "  npc_count      : ${NPC_COUNT}  (radius ${NPC_RADIUS} m)"
    echo "  spawn_pool     : ${SPAWN_POOL_SIZE}"
    echo "  leader         : ${LEADER}"
    echo "  follower       : ${FOLLOWER}"
    echo "  max_frames     : ${MAX_FRAMES}"
    echo "  shuffle        : ${SHUFFLE}"
    echo "  seed           : ${SEED}"
    echo "  log            : ${LOG}"
    echo ""
} | tee -a "$LOG"

# shellcheck disable=SC2086  # we want word-splitting on the env vars
if python "$SCRIPT" \
        --towns ${TOWNS} \
        --weather ${WEATHER} \
        --sun-altitudes ${SUN_ALTITUDES} \
        --distances ${DISTANCES} \
        --lateral-offsets ${LATERAL_OFFSETS} \
        --heading-offsets ${HEADING_OFFSETS} \
        --npc-count "${NPC_COUNT}" \
        --npc-radius "${NPC_RADIUS}" \
        --spawn-pool-size "${SPAWN_POOL_SIZE}" \
        --leader "${LEADER}" \
        --follower "${FOLLOWER}" \
        --max-frames "${MAX_FRAMES}" \
        --seed "${SEED}" \
        $( [[ "${SHUFFLE}" == "1" ]] && echo "--shuffle" ) \
        2>&1 | tee -a "$LOG"; then
    echo "[$(date '+%H:%M:%S')] DONE" | tee -a "$LOG"
else
    echo "[$(date '+%H:%M:%S')] FAILED (see $LOG)" | tee -a "$LOG"
    exit 1
fi
