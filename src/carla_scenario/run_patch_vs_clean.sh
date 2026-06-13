#!/usr/bin/env bash
# Run the scenario_two_vehicles experiment across multiple towns and seeds,
# alternating between two CARLA packages (clean vs adversarial patch).
#
# Per (town, seed) the leader spawn index is picked from a per-town pool of
# top-K straight highway spawns produced by tools/scan_spawn.py --top-k.
# Different seeds → different starting points along the highway. The two towns
# (Town04, Town06 by default) give two genuinely different road geometries.
#
# Run on Vortex:
#   bash src/carla_scenario/run_patch_vs_clean.sh
#
# Env knobs:
#   N_RUNS=10           seeds per (town, agent, condition)
#   TOWNS="Town04 Town06"  whitespace-separated list of towns to sweep
#   AGENTS="tfv4_aim_0 tfv6_visiononly simlingo_simlingo"
#   LEADER_SPEED=40     km/h (PCLA agents are urban-tuned, keep <=40)
#   SPAWN_POOL_K=10     number of distinct straight spawns to scan per town
#   PACKAGE_CLEAN, PACKAGE_PATCH   override CARLA package paths

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

N_RUNS="${N_RUNS:-10}"
AGENTS="${AGENTS:-tfv4_aim_0 tfv6_visiononly simlingo_simlingo}"
LEADER_SPEED="${LEADER_SPEED:-40}"
# Town04 (long highway segment) + Town05 (multi-lane suburban with longer
# straight stretches). Town06 was NOT cooked into our shipping packages.
TOWNS="${TOWNS:-Town04 Town05}"
SPAWN_POOL_K="${SPAWN_POOL_K:-10}"

PACKAGE_CLEAN="${PACKAGE_CLEAN:-/home/vortex/carla/Dist/CARLA_Shipping_0.9.15.2_clean/LinuxNoEditor}"
PACKAGE_PATCH="${PACKAGE_PATCH:-/home/vortex/carla/Dist/CARLA_Shipping_0.9.15.2_patch/LinuxNoEditor}"

if [[ ! -x "${PACKAGE_CLEAN}/CarlaUE4.sh" ]]; then
  echo "[ERR] missing CARLA clean package: ${PACKAGE_CLEAN}/CarlaUE4.sh"; exit 1
fi
if [[ ! -x "${PACKAGE_PATCH}/CarlaUE4.sh" ]]; then
  echo "[ERR] missing CARLA patch package: ${PACKAGE_PATCH}/CarlaUE4.sh"; exit 1
fi

MASTER_TS="$(date '+%Y%m%d_%H%M%S')"
OUT_ROOT="experiments/carla_scenarios/multi_agent_${MASTER_TS}"
mkdir -p "$OUT_ROOT"

echo "###################################################################"
echo " MULTI-TOWN, MULTI-SPAWN scenario sweep"
echo "###################################################################"
echo "  TOWNS         : $TOWNS"
echo "  AGENTS        : $AGENTS"
echo "  N_RUNS / cell : $N_RUNS  (= seeds, each picks a different spawn)"
echo "  Spawn pool K  : $SPAWN_POOL_K  (top-K straight starts per town)"
echo "  LEADER_SPEED  : $LEADER_SPEED km/h"
echo "  OUT_ROOT      : $OUT_ROOT"
echo "###################################################################"

wait_for_carla() {
  local deadline=$((SECONDS + 120))
  while ((SECONDS < deadline)); do
    if ss -tln 2>/dev/null | grep -q ':2000 '; then
      sleep 15  # shader compile grace
      return 0
    fi
    sleep 2
  done
  echo "[ERR] CARLA did not open :2000 in 120s"
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

prepare_pools() {
  # Populate experiments/carla_scenarios/spawn_cache.json with per-town top-K
  # pools. Done once per CARLA boot (any package is fine for scanning).
  local town
  for town in $TOWNS; do
    echo ">>> scanning spawn pool for $town (top-${SPAWN_POOL_K})"
    python -u src/carla_scenario/tools/scan_spawn.py \
        --town "$town" --top-k "$SPAWN_POOL_K" \
        --host localhost --port 2000 \
        2>&1 | sed 's/^/  /'
  done
}

run_condition_town_agent() {
  local label="$1"     # clean | patch
  local cond="$2"      # --condition value
  local pkg="$3"
  local town="$4"
  local agent="$5"
  local out_dir="$OUT_ROOT/$agent/$label/$town"
  mkdir -p "$out_dir"
  echo ""
  echo "----[ $agent / $label / $town ]----"
  for i in $(seq 1 "$N_RUNS"); do
    echo "--- $agent / $label / $town  seed $i / $N_RUNS ---"
    python -u src/carla_scenario/scenario_two_vehicles.py \
        --condition "$cond" \
        --agent "$agent" \
        --town "$town" \
        --leader_speed "$LEADER_SPEED" \
        --seed "$i" \
        --host localhost --port 2000 \
        --out_subdir "multi_agent_${MASTER_TS}/$agent/$label/$town" \
        2>&1 | tee -a "$out_dir/all_runs.log" || \
      echo "[WARN] $agent $label $town seed=$i exited non-zero"
  done
}

# Outer loop: condition (boot CARLA once per package, expensive).
# Middle loop: town (load_world is cheap on a running server).
# Inner loop: agent x N seeds against current town.
for cond_label in clean patch; do
  case "$cond_label" in
    clean) pkg="$PACKAGE_CLEAN"; cond_arg="none" ;;
    patch) pkg="$PACKAGE_PATCH"; cond_arg="raw"  ;;
  esac
  echo ""
  echo "###################################################################"
  echo " CONDITION: $cond_label  (package=$(basename "$(dirname "$pkg")"))"
  echo "###################################################################"
  stop_carla
  start_carla "$pkg"
  prepare_pools
  for town in $TOWNS; do
    for agent in $AGENTS; do
      run_condition_town_agent "$cond_label" "$cond_arg" "$pkg" "$town" "$agent"
    done
  done
done

stop_carla
echo ""
echo ">>> ALL DONE. Output: $OUT_ROOT"
