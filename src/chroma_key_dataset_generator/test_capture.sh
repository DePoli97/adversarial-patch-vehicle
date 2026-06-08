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

MODE="${MODE:-auto}"
# Default to the package built on Vortex. Override only if running elsewhere.
: "${CARLA_PACKAGE_DIR:=/home/vortex/carla/Dist/CARLA_Shipping_0.9.15.2/LinuxNoEditor}"
export CARLA_PACKAGE_DIR

# One run = one folder = one log. In auto mode we create the capture folder
# BEFORE re-exec so the master log can live inside it. In manual mode we still
# log to a top-level file because the capture folder is created by run_capture.
MASTER_LOG_DIR="data/chroma_key_dataset"
# RUN_TS can be set from outside (e.g. by generate_all_datasets.sh) so all
# three paired datasets share the same timestamp. DATASET_NAME, also set
# from outside, becomes a suffix on the capture folder (marker/clean/noleader).
MASTER_TS="${RUN_TS:-$(date '+%Y%m%d_%H%M%S')}"
DATASET_SUFFIX="${DATASET_NAME:+_${DATASET_NAME}}"
if [[ "$MODE" == "auto" ]]; then
  export RUN_DIR="${MASTER_LOG_DIR}/capture_${MASTER_TS}${DATASET_SUFFIX}"
  mkdir -p "$RUN_DIR"
  MASTER_LOG="${RUN_DIR}/run.log"
else
  mkdir -p "$MASTER_LOG_DIR"
  MASTER_LOG="${MASTER_LOG_DIR}/run_manual_${MASTER_TS}${DATASET_SUFFIX}.log"
fi

# Re-exec the rest of the script with all output tee'd to MASTER_LOG.
if [[ -z "${_RE_EXEC_:-}" ]]; then
  echo "Log: $MASTER_LOG"
  export _RE_EXEC_=1
  exec bash "$0" "$@" 2>&1 | tee "$MASTER_LOG"
fi

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
AUTO_TOWNS=(${AUTO_TOWNS:-Town01 Town03 Town04 Town05 Town10HD_Opt})

# ===== Common config =====
export FRAMES_PER_TOWN="${FRAMES_PER_TOWN:-200}"
# Fixed RNG seed for the whole run. KEEP THIS THE SAME across paired datasets
# (marker / clean / no-leader) so the scenes line up frame-for-frame.
export SEED="${SEED:-42}"
export CONTINUOUS=1
export SUN_ALTITUDE_RANGE="${SUN_ALTITUDE_RANGE:-25 70}"
export DISTANCE_RANGE="${DISTANCE_RANGE:-6 18}"
export HEADING_OFFSET_RANGE="${HEADING_OFFSET_RANGE:--5 5}"
export WEATHER="${WEATHER:-ClearNoon CloudyNoon WetNoon MidRainyNoon}"
export LATERAL_OFFSETS="${LATERAL_OFFSETS:--1 0 1}"
export SPAWN_POOL_SIZE="${SPAWN_POOL_SIZE:-6}"
export NPC_COUNT="${NPC_COUNT:-15}"
export NPC_RADIUS="${NPC_RADIUS:-80}"
export NPC_SCATTERED="${NPC_SCATTERED:-30}"
export SETTLE_TICKS="${SETTLE_TICKS:-50}"
export LEADER="${LEADER:-vehicle.carlamotors.carlacola}"
export FOLLOWER="${FOLLOWER:-vehicle.tesla.model3}"
export MAX_FRAMES="${MAX_FRAMES:-99999}"
export SHUFFLE="${SHUFFLE:-1}"

# ===== Helpers =====

wait_for_carla() {
  # Wait up to 90 s for port 2000 to start listening.
  local deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    if ss -tln 2>/dev/null | grep -q ':2000 '; then
      sleep 20  # UE4 opens the RPC port early; the world isn't actually ready
                # to serve load_world until shader compile finishes.
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
  # setsid puts CARLA in its own session so it cannot keep our terminal open
  # via inherited file descriptors; </dev/null closes stdin; redirect both
  # stdout and stderr to a log file, then background and disown.
  setsid bash -c "cd '${CARLA_PACKAGE_DIR}' && \
    exec ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic" \
    </dev/null >/tmp/carla_server.log 2>&1 &
  disown
  sleep 5
  wait_for_carla || exit 1
  echo ">>> CARLA up and listening on :2000"
}

stop_carla() {
  echo ">>> stopping CARLA"
  # Match any UE4/Carla helper process — main binary, launcher, render helpers.
  # Two-stage: SIGTERM first (clean shutdown), then SIGKILL to anything left.
  pkill -TERM -f 'CarlaUE4\|UE4-Linux\|UnrealEditor' 2>/dev/null || true
  sleep 3
  pkill -KILL -f 'CarlaUE4\|UE4-Linux\|UnrealEditor' 2>/dev/null || true
  # Wait for the RPC ports (2000) and TM (8000) to actually free up. Without
  # this, the next CARLA fails to bind the TM with 'bind error', leaving the
  # next town without a seeded TrafficManager (and no determinism).
  local deadline=$((SECONDS + 30))
  while ((SECONDS < deadline)); do
    if ! ss -tln 2>/dev/null | grep -qE ':(2000|8000) '; then
      return 0
    fi
    sleep 1
  done
  echo "[WARN] CARLA ports still busy after 30s"
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
  bash src/chroma_key_dataset_generator/run_capture.sh

elif [[ "$MODE" == "auto" ]]; then
  # RUN_DIR was already created above (before re-exec) so the master log
  # lives inside it. All towns share this single folder.

  echo "######################################################"
  echo " AUTO CAPTURE — sweep ${#AUTO_TOWNS[@]} towns"
  echo "  towns          : ${AUTO_TOWNS[*]}"
  echo "  attempts/town  : $FRAMES_PER_TOWN"
  echo "  npc per scene  : $NPC_COUNT (radius ${NPC_RADIUS} m)"
  echo "  run dir        : $RUN_DIR"
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
    bash src/chroma_key_dataset_generator/run_capture.sh ||
      echo "[WARN] $town session exited non-zero — continuing"
  done

  stop_carla
  echo ""
  echo ">>> ALL TOWNS DONE."
  echo ">>> Run directory: $RUN_DIR"
  printf ">>> PNG count   : "
  find "$RUN_DIR" -name "*.png" -type f 2>/dev/null | wc -l
  exit 0

else
  echo "Unknown MODE: $MODE  (use 'manual' or 'auto')"
  exit 1
fi
