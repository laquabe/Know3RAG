#!/bin/bash
# Stage 5: KGE triple scoring per candidate passage
# Loads: KGEScorer (ComplEx on Wikidata5M) only
# Usage: bash scripts/run_triple_score.sh <input.jsonl> <output.jsonl> [extra args]
#
# Environment variables:
#   CONFIG   path to config JSON  (default: config.json)
#   DATASET  override dataset_name in config

set -euo pipefail
python framework/factual_check.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --step score \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"
