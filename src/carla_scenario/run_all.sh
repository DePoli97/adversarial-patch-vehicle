#!/usr/bin/env bash
# Run the full 3-condition x 3-agent matrix of CARLA scenarios.
#
# Between conditions the plate texture must be manually Reimported in the
# Unreal Editor (the TGA file name is the same, so UE picks up the new
# content; you still need to right-click -> Reimport on the material slot).
# The script pauses and prompts you before each condition.
#
# Usage (from repo root):
#   bash src/carla_scenario/run_all.sh
#
# Output:
#   experiments/carla_scenarios/<condition>_<agent>_<timestamp>/   per run
#   experiments/carla_scenarios/run_all.log                        aggregated stdout/stderr
#
# The log file is APPENDED to, so re-running the script keeps history.

set -u  # error on unset vars, but DON'T -e: we want to survive single-run failures

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SCENARIO="src/carla_scenario/scenario_two_vehicles.py"

# One session timestamp shared by every run in this batch -> all 9 runs land
# under experiments/carla_scenarios/matrix_3x3_<timestamp>_with_probe/
SESSION_TS="$(date '+%Y%m%d_%H%M%S')"
SESSION_SUBDIR="matrix_3x3_${SESSION_TS}_with_probe"
LOG="experiments/carla_scenarios/${SESSION_SUBDIR}/run_all.log"
mkdir -p "experiments/carla_scenarios/${SESSION_SUBDIR}"

CONDITIONS=("none" "raw" "camouflaged")
AGENTS=("tfv6_visiononly" "tfv4_l6_0" "simlingo_simlingo")

TGA_DIR="assets/carla_plates"
# Mapping condition -> TGA file to Reimport before the condition's runs.
# In UE, right-click on T_LicensePlate_d -> Reimport With New File and point
# to the file below. File name on disk stays the same across conditions.
declare -A TGA_HINT=(
    [none]="$TGA_DIR/T_LicensePlate_d_none.TGA"
    [raw]="$TGA_DIR/T_LicensePlate_d_raw.TGA"
    [camouflaged]="$TGA_DIR/T_LicensePlate_d_camo.TGA"
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

mkdir -p "$(dirname "$LOG")"
banner "RUN ALL — starting batch"

TOTAL=$(( ${#CONDITIONS[@]} * ${#AGENTS[@]} ))
DONE=0
FAILED=0

for cond in "${CONDITIONS[@]}"; do
    banner "CONDITION: $cond"
    echo "Before starting, Reimport the plate texture in Unreal Editor:"
    echo "    ${TGA_HINT[$cond]}"
    echo ""
    read -r -p "Press Enter when the Reimport is done (or type 'skip' to skip this condition): " ans
    if [[ "$ans" == "skip" ]]; then
        banner "SKIPPED condition $cond"
        continue
    fi

    for agent in "${AGENTS[@]}"; do
        DONE=$((DONE + 1))
        banner "RUN $DONE/$TOTAL — condition=$cond  agent=$agent"
        if python "$SCENARIO" --condition "$cond" --agent "$agent" --out_subdir "$SESSION_SUBDIR" >> "$LOG" 2>&1; then
            echo "[$(date '+%H:%M:%S')] OK   condition=$cond agent=$agent" | tee -a "$LOG"
        else
            FAILED=$((FAILED + 1))
            echo "[$(date '+%H:%M:%S')] FAIL condition=$cond agent=$agent (see $LOG)" | tee -a "$LOG"
        fi
    done
done

banner "RUN ALL — done. ok=$((DONE - FAILED))/$TOTAL  failed=$FAILED"
echo "Full log: $LOG"
