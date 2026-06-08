#!/usr/bin/env bash
# Run chroma-key quad extraction on a dataset folder.
#
# Produces:
#   <RUN_DIR>/quads_index.json   ← one JSON with all corners (compact)
#   updates each <NNNNNN>.json   ← adds "detected_corners" to existing metadata
#
# Edit RUN_DIR below or override via env, then:
#   bash src/chroma_key_dataset_generator/extract_quads.sh

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Dataset folder to process. Override with: RUN_DIR=... bash ...
: "${RUN_DIR:=data/chroma_key_dataset/capture_20260602_211812}"

python src/chroma_key_dataset_generator/extract_quad.py \
    --image "${RUN_DIR}/" \
    --batch-index "${RUN_DIR}/quads_index.json"
