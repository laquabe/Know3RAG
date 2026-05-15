#!/bin/bash
# Optional (multi-turn): Generate follow-up retrieval question from current answer
# Loads: LLM only
# Usage: bash scripts/run_followup_question.sh <input.jsonl> <output.jsonl> [extra args]
#
# Reads:  question, reference, llm_response
# Writes: new_question  (pass as --query-key new_question to run_doc_generate.sh)
#
# Environment variables:
#   CONFIG   path to config JSON  (default: config.json)
#   DATASET  override dataset_name in config

set -euo pipefail
"${PYTHON:-python3}" framework/query_enhance.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --stage followup \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"
