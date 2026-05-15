#!/bin/bash
# Stage: fetch Wikidata claims/descriptions for query entities
# Loads: WikidataClient only
# Usage: bash scripts/run_query_wikidata.sh <input.jsonl> <output.jsonl> [extra args]

set -euo pipefail
"${PYTHON:-python3}" framework/query_enhance.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --stage query-wikidata \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"