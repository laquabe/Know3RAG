#!/bin/bash
# Query enhancement stage: fetch tail Wikidata info and build KG knowledge card.

python framework/query_enhance.py \
  --config config.json \
  --input path/to/input_relation_tail.jsonl \
  --output path/to/output_tail_wikidata_card.jsonl \
  --stage tail-wikidata-card
