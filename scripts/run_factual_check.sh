#!/bin/bash
# Factual check stage: check factual consistency of a passage field with KG.
#
# There are two modes. Choose one command below and comment out the other.
#
# 1) triple mode:
#    LLM extracts triples from the passage, maps triples to Wikidata IDs,
#    then scores them with KGE. More explainable, but slower.
#    Main output score: factual_score
#
# 2) fast mode:
#    EntityLinker extracts passage entities, builds sentence-local entity pairs,
#    then scores entity pairs with KGE. Faster, but less semantically explicit.
#    Main output score: fast_factual_score

# -------- Option 1: triple mode, recommended for debugging --------
python framework/factual_check.py \
  --config config.json \
  --input path/to/input_relevance_checked.jsonl \
  --output path/to/output_factual_checked.jsonl \
  --mode triple \
  --step all \
  --check-key passages

# -------- Option 2: fast mode, recommended for faster batch checking --------
# python framework/factual_check.py \
#   --config config.json \
#   --input path/to/input_relevance_checked.jsonl \
#   --output path/to/output_factual_checked_fast.jsonl \
#   --mode fast \
#   --step all \
#   --check-key passages