#!/bin/bash
# Stage: predict KG relations and KGE tail entities
# Loads: EntityLinker relation resources + KGEScorer
# Usage: bash scripts/run_relation_tail.sh <input.jsonl> <output.jsonl> [extra args]

set -euo pipefail
"${PYTHON:-python3}" framework/query_enhance.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --stage relation-tail \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"