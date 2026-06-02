#!/usr/bin/env bash
# Capture frames in continuous random-sampling mode.
#
# TWO MODES:
#   MODE=manual (default): runs the ACTIVE TOWN below, then exits.
#                          You restart CARLA + edit the town + re-run.
#   MODE=auto            : loops over all 5 towns. Between towns it kills
#                          CARLA, restarts CARLA from the packaged
#                          CarlaUE4.sh, waits for the RPC port, then resumes.
#                          Set CARLA_PACKAGE_DIR to the path of the package.
#
# Continuous sampling (both modes):
#   sun altitude   uniform in SUN_ALTITUDE_RANGE  (degrees)
#   distance       uniform in DISTANCE_RANGE      (meters)
#   heading offset uniform in HEADING_OFFSET_RANGE (degrees)
#   weather        random.choice from WEATHER     (categorical)
#   lateral        random.choice from LATERAL_OFFSETS (lane shifts: -1/0/+1)
#
# Usage:
#   # Manual (UE Editor open, you press Play, you swap towns by hand)
#   bash src/chroma_key_dataset_generator/test_capture.sh
#
#   # Auto (uses packaged CarlaUE4.sh, no human interaction)
#   MODE=auto \
#   CARLA_PACKAGE_DIR=/home/vortex/carla/Dist/CARLA_<commit>/ \
#       bash src/chroma_key_dataset_generator/test_capture.sh

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-manual}"

# ===== ACTIVE TOWN (manual mode only) =====
# Uncomment ONE line below (or set TOWNS via env to override).
# In auto mode, this is ignored and AUTO_TOWNS below is used instead.
: "${TOWNS:=Town01}"
# : "${TOWNS:=Town03}"
# : "${TOWNS:=Town04}"
# : "${TOWNS:=Town06}"
# : "${TOWNS:=Town10HD_Opt}"
export TOWNS

# ===== AUTO MODE TOWN LIST =====
AUTO_TOWNS=(${AUTO_TOWNS:-Town01 Town03 Town04 Town06 Town10HD_Opt})

# ===== Common config =====
export FRAMES_PER_TOWN="${FRAMES_PER_TOWN:-50}"
export CONTINUOUS=1
export SUN_ALTITUDE_RANGE="${SUN_ALTITUDE_RANGE:-25 70}"
export DISTANCE_RANGE="${DISTANCE_RANGE:-6 18}"
export HEADING_OFFSET_RANGE="${HEADING_OFFSET_RANGE:--5 5}"
export WEATHER="${WEATHER:-ClearNoon CloudyNoon WetNoon MidRainyNoon}"
export LATERAL_OFFSETS="${LATERAL_OFFSETS:--1 0 1}"
export SPAWN_POOL_SIZE="${SPAWN_POOL_SIZE:-6}"
export NPC_COUNT="${NPC_COUNT:-0}"
export NPC_RADIUS="${NPC_RADIUS:-60}"
export SETTLE_TICKS="${SETTLE_TICKS:-50}"
export LEADER="${LEADER:-vehicle.carlamotors.carlacola}"
export FOLLOWER="${FOLLOWER:-vehicle.tesla.model3}"
export MAX_FRAMES="${MAX_FRAMES:-99999}"
export SHUFFLE="${SHUFFLE:-1}"

# ===== Helpers =====

wait_for_carla() {
    # Wait up to 90 s for port 2000 to start listening.
    local deadline=$((SECONDS + 90))
    while (( SECONDS < deadline )); do
        if ss -tln 2>/dev/null | grep -q ':2000 '; then
            sleep 3   # small extra delay for the world to be ready
            return 0
        fi
        sleep 2
    done
    echo "[ERR] CARLA did not open port 2000 within 90 s"
    return 1
}

start_carla_package() {
    if [[ -z "${CARLA_PACKAGE_DIR:-}" ]]; then
        echo "[ERR] auto mode requires CARLA_PACKAGE_DIR pointing at the CARLA package"
        exit 1
    fi
    if [[ ! -x "${CARLA_PACKAGE_DIR}/CarlaUE4.sh" ]]; then
        echo "[ERR] ${CARLA_PACKAGE_DIR}/CarlaUE4.sh not found / not executable"
        exit 1
    fi
    echo ">>> launching CarlaUE4.sh (headless, off-screen render)"
    (cd "${CARLA_PACKAGE_DIR}" && \
        nohup ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic \
              >/tmp/carla_server.log 2>&1 &)
    sleep 5
    wait_for_carla || exit 1
    echo ">>> CARLA up and listening on :2000"
}

stop_carla() {
    echo ">>> stopping CARLA"
    pkill -9 -f 'CarlaUE4-Linux\|CarlaUE4.sh' 2>/dev/null || true
    sleep 4
}

# ===== MAIN =====

if [[ "$MODE" == "manual" ]]; then
    echo "######################################################"
    echo " MANUAL CAPTURE — single town"
    echo "  active town    : $TOWNS"
    echo "  attempts       : $FRAMES_PER_TOWN"
    echo "  sun range      : $SUN_ALTITUDE_RANGE"
    echo "  dist range     : $DISTANCE_RANGE"
    echo "  heading range  : $HEADING_OFFSET_RANGE"
    echo "######################################################"
    echo ""
    export SEED="${SEED:-$RANDOM}"
    bash src/chroma_key_dataset_generator/run_capture.sh

elif [[ "$MODE" == "auto" ]]; then
    echo "######################################################"
    echo " AUTO CAPTURE — sweep ${#AUTO_TOWNS[@]} towns"
    echo "  towns          : ${AUTO_TOWNS[*]}"
    echo "  attempts/town  : $FRAMES_PER_TOWN"
    echo "  package dir    : ${CARLA_PACKAGE_DIR:-<unset>}"
    echo "######################################################"
    echo ""

    for town in "${AUTO_TOWNS[@]}"; do
        echo ""
        echo "============================================================"
        echo " TOWN: $town"
        echo "============================================================"

        stop_carla
        start_carla_package

        export TOWNS="$town"
        export SEED="$RANDOM"
        bash src/chroma_key_dataset_generator/run_capture.sh || \
            echo "[WARN] $town session exited non-zero — continuing"
    done

    stop_carla
    echo ""
    echo ">>> ALL TOWNS DONE."
    find data/chroma_key_dataset/ -name "*.png" -type f 2>/dev/null | wc -l

else
    echo "Unknown MODE: $MODE  (use 'manual' or 'auto')"
    exit 1
fi
