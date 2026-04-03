#!/bin/bash
# Stage 3: LLM triple extraction per candidate passage
# Loads: LLM only
# Usage: bash scripts/run_triple_extract.sh <input.jsonl> <output.jsonl> [extra args]
#
# Environment variables:
#   CONFIG   path to config JSON  (default: config.json)
#   DATASET  override dataset_name in config

set -euo pipefail
python framework/factual_check.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --step extract \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"
