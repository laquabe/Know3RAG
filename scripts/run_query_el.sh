#!/bin/bash
# Query enhancement stage: query entity linking only.

python framework/query_enhance.py \
  --config config.json \
  --input path/to/input_raw.jsonl \
  --output path/to/output_query_el.jsonl \
  --stage query-el \
  --question-key question
