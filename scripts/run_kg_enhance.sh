#!/bin/bash
# Legacy combined KG query enhancement
# Loads: EntityLinker + WikidataClient + KGEScorer in one runtime (no LLM)
# Prefer split scripts when these environments are incompatible:
#   run_query_el.sh -> run_query_wikidata.sh -> run_relation_tail.sh -> run_tail_wikidata_card.sh
# Usage: bash scripts/run_kg_enhance.sh <input.jsonl> <output.jsonl> [extra args]
#
# Environment variables:
#   CONFIG   path to config JSON  (default: config.json)
#   DATASET  override dataset_name in config

set -euo pipefail
"${PYTHON:-python3}" framework/query_enhance.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --stage kg \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"
