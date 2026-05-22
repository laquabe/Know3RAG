#!/bin/bash
# Merge check stage: combine relevance and factual check results.

python framework/check_merge.py \
  --input path/to/input_factual_checked.jsonl \
  --output path/to/output_check_merged.jsonl \
  --relevance-key local_check \
  --factual-key factual_score \
  --threshold 10000 \
  --factual-output-key factual_check \
  --output-key final_check