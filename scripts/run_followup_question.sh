#!/bin/bash
# Query enhancement stage: generate follow-up retrieval question.

python framework/query_enhance.py \
  --config config.json \
  --input path/to/input_tail_wikidata_card.jsonl \
  --output path/to/output_followup_question.jsonl \
  --stage followup
