#!/usr/bin/env bash
# Run the full N-seed x N-agent x 2-condition matrix of CARLA scenarios.
#
# Default: 5 seeds x 4 agents x 2 conditions = 40 runs.
#   - Conditions: `none` and `raw` (with reduced alpha; the camouflaged
#     condition is dropped — only raw + transparency for the rear-window
#     experiment, per Masoud's 2026-04-29 plan).
#   - Between conditions the rear-window texture must be manually Reimported
#     in the Unreal Editor. The script pauses before each condition.
#
# Usage (from repo root):
#   bash src/carla_scenario/run_all.sh
#
# Output:
#   experiments/carla_scenarios/patch_on_surface/micra_rear_window_<timestamp>/
#       <cond>_<agent>_seed<N>_<timestamp>/   per run
#       run_all.log                            aggregated stdout/stderr

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SCENARIO="src/carla_scenario/scenario_two_vehicles.py"

SESSION_TS="$(date '+%Y%m%d_%H%M%S')"
# Layout: experiments/carla_scenarios/patch_on_surface/micra_rear_window_<ts>/
# (parent groups all surface-patch experiments; subfolder identifies the
# vehicle + surface so future truck-side experiments live alongside.)
SESSION_SUBDIR="patch_on_surface/micra_rear_window_${SESSION_TS}"
LOG="experiments/carla_scenarios/${SESSION_SUBDIR}/run_all.log"
mkdir -p "experiments/carla_scenarios/${SESSION_SUBDIR}"

CONDITIONS=("none" "raw")
AGENTS=("tfv6_visiononly" "tfv5_l6_0" "tfv4_l6_0" "simlingo_simlingo")
#SEEDS=(0 1 2 3 4)
SEEDS=(0)

TGA_DIR="assets/carla_rear_window"
declare -A TGA_HINT=(
    [none]="$TGA_DIR/rear_window_none.TGA"
    [raw]="$TGA_DIR/rear_window_raw.TGA"
)

banner() {
    local msg="$1"
    {
        echo ""
        echo "################################################################"
        echo "# $msg"
        echo "# $(date '+%Y-%m-%d %H:%M:%S')"
        echo "################################################################"
        echo ""
    } | tee -a "$LOG"
}

banner "RUN ALL — starting batch"
echo "  conditions=(${CONDITIONS[*]})  agents=(${AGENTS[*]})  seeds=(${SEEDS[*]})" | tee -a "$LOG"

TOTAL=$(( ${#CONDITIONS[@]} * ${#AGENTS[@]} * ${#SEEDS[@]} ))
DONE=0
FAILED=0

for cond in "${CONDITIONS[@]}"; do
    banner "CONDITION: $cond"
    echo "Before starting, Reimport the rear-window texture in Unreal Editor:"
    echo "    ${TGA_HINT[$cond]}"
    echo "(carla -> static -> car -> four_wheeled -> Nissan_Micra -> Element 4)"
    echo ""
    read -r -p "Press Enter when the Reimport is done (or type 'skip' to skip this condition): " ans
    if [[ "$ans" == "skip" ]]; then
        banner "SKIPPED condition $cond"
        DONE=$((DONE + ${#AGENTS[@]} * ${#SEEDS[@]}))
        continue
    fi

    for agent in "${AGENTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            DONE=$((DONE + 1))
            banner "RUN $DONE/$TOTAL — condition=$cond  agent=$agent  seed=$seed"
            if python "$SCENARIO" \
                    --condition "$cond" \
                    --agent "$agent" \
                    --seed "$seed" \
                    --out_subdir "$SESSION_SUBDIR" \
                    >> "$LOG" 2>&1; then
                echo "[$(date '+%H:%M:%S')] OK   condition=$cond agent=$agent seed=$seed" | tee -a "$LOG"
            else
                FAILED=$((FAILED + 1))
                echo "[$(date '+%H:%M:%S')] FAIL condition=$cond agent=$agent seed=$seed (see $LOG)" | tee -a "$LOG"
            fi
        done
    done
done

banner "RUN ALL — done. ok=$((DONE - FAILED))/$TOTAL  failed=$FAILED"
echo "Full log: $LOG"
