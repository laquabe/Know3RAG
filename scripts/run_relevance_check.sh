#!/bin/bash
# Relevance check stage: check whether a passage field is relevant/reliable for the question.

python framework/relevance_check.py \
  --config config.json \
  --input path/to/input.jsonl \
  --output path/to/output_relevance_checked.jsonl \
  --check-key passages \
  --output-key local_check \
  --raw-output-key local_check_raw \
  --question-key question
