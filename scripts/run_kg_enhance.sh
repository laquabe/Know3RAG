#!/bin/bash
# Stage 1: KG query enhancement
# Loads: EntityLinker + WikidataClient + KGEScorer (no LLM)
# Usage: bash scripts/run_kg_enhance.sh <input.jsonl> <output.jsonl> [extra args]
#
# Environment variables:
#   CONFIG   path to config JSON  (default: config.json)
#   DATASET  override dataset_name in config

set -euo pipefail
python framework/query_enhance.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --step kg \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"
