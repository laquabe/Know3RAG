#!/bin/bash
# Stage: predict KG relations and KGE tail entities

python framework/query_enhance.py \
  --config config.json \
  --input path/to/input_query_wikidata.jsonl \
  --output path/to/output_relation_tail.jsonl \
  --stage relation-tail