#!/bin/bash
# Stage: fetch Wikidata info for predicted tails and build KG knowledge card
# Loads: WikidataClient only; DocumentGenerator is used in no-LLM/no-retriever mode
# Usage: bash scripts/run_tail_wikidata_card.sh <input.jsonl> <output.jsonl> [extra args]

set -euo pipefail
"${PYTHON:-python3}" framework/query_enhance.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --stage tail-wikidata-card \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"