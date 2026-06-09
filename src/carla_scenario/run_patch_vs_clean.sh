#!/usr/bin/env bash
# Run the scenario_two_vehicles experiment 2 x N times:
#   - N runs with the CLEAN package (CarlaCola with original red skin)
#   - N runs with the PATCH package (CarlaCola with the adversarial patch)
#
# Between conditions we stop CARLA, launch the other package, wait for the
# RPC port to come up, then repeat.
#
# Run on Vortex:
#   bash src/carla_scenario/run_patch_vs_clean.sh
#
# Env knobs:
#   N_RUNS=10           runs per condition
#   AGENT=tfv6_visiononly   PCLA agent
#   LEADER_SPEED=40     km/h (urban-tuned PCLA prefers <=40)
#   PACKAGE_CLEAN, PACKAGE_PATCH   override CARLA package paths

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

N_RUNS="${N_RUNS:-10}"
AGENT="${AGENT:-tfv6_visiononly}"
LEADER_SPEED="${LEADER_SPEED:-40}"

PACKAGE_CLEAN="${PACKAGE_CLEAN:-/home/vortex/carla/Dist/CARLA_Shipping_0.9.15.2_clean/LinuxNoEditor}"
PACKAGE_PATCH="${PACKAGE_PATCH:-/home/vortex/carla/Dist/CARLA_Shipping_0.9.15.2_patch/LinuxNoEditor}"

if [[ ! -x "${PACKAGE_CLEAN}/CarlaUE4.sh" ]]; then
  echo "[ERR] missing CARLA clean package: ${PACKAGE_CLEAN}/CarlaUE4.sh"
  exit 1
fi
if [[ ! -x "${PACKAGE_PATCH}/CarlaUE4.sh" ]]; then
  echo "[ERR] missing CARLA patch package: ${PACKAGE_PATCH}/CarlaUE4.sh"
  exit 1
fi

MASTER_TS="$(date '+%Y%m%d_%H%M%S')"
OUT_ROOT="experiments/carla_scenarios/patch_vs_clean_${MASTER_TS}"
mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"

wait_for_carla() {
  local deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    if ss -tln 2>/dev/null | grep -q ':2000 '; then
      sleep 15  # shader compile grace
      return 0
    fi
    sleep 2
  done
  echo "[ERR] CARLA did not open :2000 in 90s"
  return 1
}

stop_carla() {
  echo ">>> stopping CARLA"
  pkill -TERM -f 'CarlaUE4\|UE4-Linux\|UnrealEditor' 2>/dev/null || true
  sleep 3
  pkill -KILL -f 'CarlaUE4\|UE4-Linux\|UnrealEditor' 2>/dev/null || true
  sleep 2
  fuser -k 2000/tcp 8000/tcp 2>/dev/null || true
  local deadline=$((SECONDS + 60))
  while ((SECONDS < deadline)); do
    local procs ports
    procs=$(pgrep -f 'CarlaUE4\|UE4-Linux\|UnrealEditor' 2>/dev/null | wc -l)
    ports=$(ss -tln 2>/dev/null | grep -cE ':(2000|8000) ')
    if (( procs == 0 && ports == 0 )); then return 0; fi
    sleep 2
  done
  echo "[WARN] CARLA still has procs/ports busy"
}

start_carla() {
  local pkg="$1"
  echo ">>> launching CARLA from $(basename "$(dirname "$pkg")")"
  setsid bash -c "cd '${pkg}' && exec ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic" \
    </dev/null >/tmp/carla_scenario.log 2>&1 &
  disown
  sleep 5
  wait_for_carla || exit 1
}

run_condition() {
  local label="$1"   # "clean" or "patch"
  local pkg="$2"
  local out_dir="$OUT_ROOT/$label"
  mkdir -p "$out_dir"
  echo ""
  echo "###################################################################"
  echo " CONDITION: $label   (package=$(basename "$(dirname "$pkg")"))"
  echo "###################################################################"

  stop_carla
  start_carla "$pkg"

  for i in $(seq 1 "$N_RUNS"); do
    echo ""
    echo "--- $label run $i / $N_RUNS ---"
    python -u src/carla_scenario/scenario_two_vehicles.py \
        --condition "$label" \
        --agent "$AGENT" \
        --leader_speed "$LEADER_SPEED" \
        --host localhost --port 2000 \
        --out-subdir "patch_vs_clean_${MASTER_TS}/$label" \
        2>&1 | tee -a "$out_dir/all_runs.log" || \
      echo "[WARN] run $i exited non-zero"
  done
}

run_condition clean "$PACKAGE_CLEAN"
run_condition patch "$PACKAGE_PATCH"

stop_carla
echo ""
echo ">>> ALL DONE. Output: $OUT_ROOT"
