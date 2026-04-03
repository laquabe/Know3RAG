#!/bin/bash
# Stage 4: EntityLinker triple → Wikidata ID mapping
# Loads: EntityLinker (spaCy + SentenceTransformer) only
# Usage: bash scripts/run_triple_map.sh <input.jsonl> <output.jsonl> [extra args]
#
# Environment variables:
#   CONFIG   path to config JSON  (default: config.json)
#   DATASET  override dataset_name in config

set -euo pipefail
python framework/factual_check.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --step map \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"
