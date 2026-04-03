#!/bin/bash
# Stage 2: Document / reference generation
# Loads: LLM + HybridRetriever (optional)
# Usage: bash scripts/run_doc_generate.sh <input.jsonl> <output.jsonl> [extra args]
#
# Environment variables:
#   CONFIG     path to config JSON  (default: config.json)
#   DATASET    override dataset_name in config
#   STEP       llm | retrieve | all  (default: all)
#   QUERY_KEY  question field name  (default: question)

set -euo pipefail
python framework/document_generation.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --step "${STEP:-all}" \
  --query-key "${QUERY_KEY:-question}" \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"
