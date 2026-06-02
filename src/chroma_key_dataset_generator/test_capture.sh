#!/usr/bin/env bash
# Quick smoke-test of run_capture.sh — 5 frames in Town06, no NPCs, default leader (CarlaCola).
# Full output gets tee'd to data/chroma_key_dataset/run_capture_<ts>.log
# (printed at start and end of the run for easy grep).
#
# Usage:
#   bash src/chroma_key_dataset_generator/test_capture.sh
#
# Override individual params via env if needed, e.g.:
#   MAX_FRAMES=10 LEADER="vehicle.tesla.cybertruck" \
#       bash src/chroma_key_dataset_generator/test_capture.sh

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Test config (override any of these from the caller env if you want)
export TOWNS="${TOWNS:-Town06}"
export MAX_FRAMES="${MAX_FRAMES:-5}"
export NPC_COUNT="${NPC_COUNT:-0}"
export SHUFFLE="${SHUFFLE:-1}"
# Defaults below come from run_capture.sh; uncomment to override:
# export LEADER="vehicle.carlamotors.carlacola"
# export SETTLE_TICKS=50

bash src/chroma_key_dataset_generator/run_capture.sh
