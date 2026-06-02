#!/usr/bin/env bash
# Full-grid capture across 5 towns. Designed to run overnight on Vortex.
#
# Defaults: 5 towns x 4 spawns x 4 weather x 4 sun x 4 distance x 3 lateral x 3 heading
#         = ~11520 combinations. ~60-70% will produce a frame (the rest are
#         skipped: invalid lane shift, off-road spawn, etc).
#
# Output: data/chroma_key_dataset/capture_<ts>/ (one frame per accepted combo)
#         data/chroma_key_dataset/run_capture_<ts>.log (full stdout/stderr)
#
# Usage (on Vortex, with CARLA server running on :2000):
#   bash src/chroma_key_dataset_generator/test_capture.sh
#
# Override any axis via env, e.g.:
#   MAX_FRAMES=2000 bash src/chroma_key_dataset_generator/test_capture.sh
#   TOWNS="Town04 Town06" bash src/chroma_key_dataset_generator/test_capture.sh

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# 5 towns covering urban (Town01, Town03, Town10HD_Opt) and highway (Town04, Town06)
export TOWNS="${TOWNS:-Town01 Town03 Town04 Town06 Town10HD_Opt}"

# Full grid (4 x 4 x 4 = 64 weather/sun/dist combos per spawn)
export WEATHER="${WEATHER:-ClearNoon CloudyNoon WetNoon MidRainyNoon}"
export SUN_ALTITUDES="${SUN_ALTITUDES:-70 55 40 25}"
export DISTANCES="${DISTANCES:-6 10 14 18}"
export LATERAL_OFFSETS="${LATERAL_OFFSETS:--1 0 1}"     # lane shifts (integer)
export HEADING_OFFSETS="${HEADING_OFFSETS:--5 0 5}"     # degrees
export SPAWN_POOL_SIZE="${SPAWN_POOL_SIZE:-4}"

# No NPCs by default (TrafficManager + sync mode = unreliable). Set NPC_COUNT
# in the env if you want to experiment.
export NPC_COUNT="${NPC_COUNT:-0}"
export NPC_RADIUS="${NPC_RADIUS:-60}"

# Capture timing
export SETTLE_TICKS="${SETTLE_TICKS:-50}"

# Vehicles
export LEADER="${LEADER:-vehicle.carlamotors.carlacola}"
export FOLLOWER="${FOLLOWER:-vehicle.tesla.model3}"

# By default, no cap — go through the whole grid. Set MAX_FRAMES=N to cap.
export MAX_FRAMES="${MAX_FRAMES:-99999}"

# Shuffle so partial runs are still well-balanced across towns/weather/etc.
export SHUFFLE="${SHUFFLE:-1}"

export SEED="${SEED:-0}"

bash src/chroma_key_dataset_generator/run_capture.sh
