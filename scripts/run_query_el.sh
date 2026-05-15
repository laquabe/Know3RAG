#!/bin/bash
# Stage: query entity linking only
# Loads: EntityLinker EL components only
# Usage: bash scripts/run_query_el.sh <input.jsonl> <output.jsonl> [extra args]

set -euo pipefail
"${PYTHON:-python3}" framework/query_enhance.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --stage query-el \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"