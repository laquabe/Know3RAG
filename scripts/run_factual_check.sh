#!/bin/bash
# Factual check stage: check factual consistency of a passage field with KG.

python framework/factual_check.py \
  --config config.json \
  --input path/to/input_relevance_checked.jsonl \
  --output path/to/output_factual_checked.jsonl \
  --mode triple \
  --step all \
  --check-key passages