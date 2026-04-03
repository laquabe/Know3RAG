#!/bin/bash
# Stage 7: LLM answer generation
# Loads: LLM only
# Usage: bash scripts/run_qa.sh <input.jsonl> <output.jsonl> [extra args]
#
# Environment variables:
#   CONFIG      path to config JSON  (default: config.json)
#   DATASET     override dataset_name in config
#   ANSWER_KEY  key to store the answer under  (default: llm_response)
#               Use llm_response_0, llm_response_1, ... for multi-turn

set -euo pipefail
python framework/question_answer.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --answer-key "${ANSWER_KEY:-llm_response}" \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"
