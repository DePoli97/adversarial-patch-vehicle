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
LATERAL_OFFSETS="${LATERAL_OFFSETS:--1 0 1}"
HEADING_OFFSETS="${HEADING_OFFSETS:--5 0 5}"
NPC_COUNT="${NPC_COUNT:-0}"
NPC_RADIUS="${NPC_RADIUS:-60}"
NPC_SCATTERED="${NPC_SCATTERED:-0}"
SETTLE_TICKS="${SETTLE_TICKS:-50}"
SPAWN_POOL_SIZE="${SPAWN_POOL_SIZE:-4}"
LEADER="${LEADER:-vehicle.carlamotors.carlacola}"
FOLLOWER="${FOLLOWER:-vehicle.tesla.model3}"
MAX_FRAMES="${MAX_FRAMES:-1000}"
SEED="${SEED:-0}"
# Set SHUFFLE=1 to sample combos randomly (recommended when MAX_FRAMES < full grid).
SHUFFLE="${SHUFFLE:-1}"
# Continuous sampling mode (recommended): random sun/dist/heading within ranges,
# random weather/lateral from the discrete lists, N attempts per town.
CONTINUOUS="${CONTINUOUS:-0}"
FRAMES_PER_TOWN="${FRAMES_PER_TOWN:-300}"
SUN_ALTITUDE_RANGE="${SUN_ALTITUDE_RANGE:-25 70}"
DISTANCE_RANGE="${DISTANCE_RANGE:-6 18}"
HEADING_OFFSET_RANGE="${HEADING_OFFSET_RANGE:--5 5}"

TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="data/chroma_key_dataset"
# If RUN_DIR is set, test_capture.sh is already tee-logging everything to the
# master log inside RUN_DIR, so we skip writing a second log here.
# Standalone use of run_capture.sh still gets its own log.
if [[ -n "${RUN_DIR:-}" ]]; then
    mkdir -p "${RUN_DIR}"
    LOG="/dev/null"
else
    mkdir -p "$LOG_DIR"
    LOG="${LOG_DIR}/run_capture_${TS}.log"
fi

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
    echo "  npc_scattered  : ${NPC_SCATTERED}  (random across map)"
    echo "  spawn_pool     : ${SPAWN_POOL_SIZE}"
    echo "  leader         : ${LEADER}"
    echo "  follower       : ${FOLLOWER}"
    echo "  max_frames     : ${MAX_FRAMES}"
    echo "  shuffle        : ${SHUFFLE}"
    echo "  continuous     : ${CONTINUOUS}  (per-town: ${FRAMES_PER_TOWN})"
    echo "  sun_range      : ${SUN_ALTITUDE_RANGE}"
    echo "  dist_range     : ${DISTANCE_RANGE}"
    echo "  heading_range  : ${HEADING_OFFSET_RANGE}"
    echo "  seed           : ${SEED}"
    echo "  log            : ${LOG}"
    echo ""
} | tee -a "$LOG"

if [[ "$LOG" != "/dev/null" ]]; then
    echo ">>> Full output saved to: $LOG"
    echo ">>> Tail live with: tail -f $LOG"
fi
echo ""

# Force python stdout/stderr unbuffered so tee gets every line in real time.
# shellcheck disable=SC2086  # we want word-splitting on the env vars
if python -u "$SCRIPT" \
        --towns ${TOWNS} \
        --weather ${WEATHER} \
        --sun-altitudes ${SUN_ALTITUDES} \
        --distances ${DISTANCES} \
        --lateral-offsets ${LATERAL_OFFSETS} \
        --heading-offsets ${HEADING_OFFSETS} \
        --npc-count "${NPC_COUNT}" \
        --npc-radius "${NPC_RADIUS}" \
        --npc-scattered "${NPC_SCATTERED}" \
        --settle-ticks "${SETTLE_TICKS}" \
        --spawn-pool-size "${SPAWN_POOL_SIZE}" \
        --leader "${LEADER}" \
        --follower "${FOLLOWER}" \
        --max-frames "${MAX_FRAMES}" \
        --seed "${SEED}" \
        $( [[ "${SHUFFLE}" == "1" ]] && echo "--shuffle" ) \
        $( [[ "${CONTINUOUS}" == "1" ]] && echo "--continuous" ) \
        --frames-per-town "${FRAMES_PER_TOWN}" \
        --sun-altitude-range ${SUN_ALTITUDE_RANGE} \
        --distance-range ${DISTANCE_RANGE} \
        --heading-offset-range ${HEADING_OFFSET_RANGE} \
        $( [[ -n "${RUN_DIR:-}" ]] && echo "--run-dir ${RUN_DIR}" ) \
        2>&1 | tee -a "$LOG"; then
    echo "[$(date '+%H:%M:%S')] DONE" | tee -a "$LOG"
    if [[ "$LOG" != "/dev/null" ]]; then
        echo ""
        echo "=============================================================="
        echo " Full log saved to: $LOG"
        echo "=============================================================="
    fi
else
    echo "[$(date '+%H:%M:%S')] FAILED" | tee -a "$LOG"
    [[ "$LOG" != "/dev/null" ]] && echo "Log saved to: $LOG"
    exit 1
fi
