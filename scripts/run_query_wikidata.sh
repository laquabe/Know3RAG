#!/bin/bash
# Query enhancement stage: fetch Wikidata claims/descriptions for query entities.

python3 framework/query_enhance.py \
  --config config.json \
  --input datasets/test/turn0_query_el.jsonl \
  --output datasets/test/turn0_query_el_wiki.jsonl \
  --stage query-wikidata
