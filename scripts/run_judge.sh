#!/bin/bash
# Stage 8 (optional): LLM multi-turn answer judge
# Loads: LLM only
# Usage: bash scripts/run_judge.sh <input.jsonl> <output.jsonl> [extra args]
#
# Environment variables:
#   CONFIG       path to config JSON  (default: config.json)
#   DATASET      override dataset_name in config
#   ANSWER_KEYS  space-separated list of answer keys to judge
#                (default: "llm_response_0 llm_response_1")

set -euo pipefail
python framework/question_answer.py \
  --config "${CONFIG:-config.json}" \
  --input "$1" --output "$2" \
  --step judge \
  --answer-keys ${ANSWER_KEYS:-llm_response_0 llm_response_1} \
  ${DATASET:+--dataset "$DATASET"} \
  "${@:3}"
