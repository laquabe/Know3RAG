#!/bin/bash
# Query enhancement stage: fetch Wikidata claims/descriptions for query entities.

python framework/query_enhance.py \
  --config config.json \
  --input path/to/input_query_el.jsonl \
  --output path/to/output_query_wikidata.jsonl \
  --stage query-wikidata
