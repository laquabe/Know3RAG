#!/bin/bash
# QA test with references / RAG mode.
# Edit the variables below before running.

set -euo pipefail

CONFIG="configs/llama3_local.json"
DATASET="hotpotQA"
INPUT_FILE="path/to/input_with_reference.jsonl"
OUTPUT_FILE="path/to/output_qa_rag.jsonl"
COT_FILE="path/to/cot_with_reference.jsonl"
ANSWER_KEY="llm_response"

python framework/question_answer.py \
  --config "$CONFIG" \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --dataset "$DATASET" \
  --answer-key "$ANSWER_KEY" \
  --cot-file "$COT_FILE"