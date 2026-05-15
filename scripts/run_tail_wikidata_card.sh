#!/bin/bash
# Query enhancement stage: fetch tail Wikidata info and build KG knowledge card.

python3 framework/query_enhance.py \
  --config config.json \
  --input datasets/test/turn0_el_add_tail.jsonl \
  --output datasets/test/turn0_query_enhanced.jsonl \
  --stage tail-wikidata-card
