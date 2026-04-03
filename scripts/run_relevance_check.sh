#!/bin/bash
# Stage 6: LLM passage reliability check + reference selection
# Loads: LLM only (KGE scores already pre-computed in candidate_passages)
# Usage: bash scripts/run_relevance_check.sh <input.jsonl> <output.jsonl> [extra args]
#
# Environment variables:
#   CONFIG   path to config JSON  (default: config.json)
#   DATASET  override dataset_name in config
#   TOP_K    number of references to select  (default: from config)

set -euo pipefail
python framework/relevance_check.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --step all \
  ${TOP_K:+--top-k "$TOP_K"} \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"
