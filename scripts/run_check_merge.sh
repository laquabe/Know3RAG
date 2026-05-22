#!/bin/bash
# Check-merge stage: combine per-passage relevance and factual results,
# then keep the top-k passages per query id.
#
# Inputs are DIRECTORIES (non-recursive). Every regular file in each
# directory is read as JSONL. Records are aligned by (id, passages).
#
# Filtering and ranking per query id:
#   - drop passages where local_check is false
#   - sort remaining passages by factual score ascending (lower is better)
#   - passages with score == None are placed last, but can still fill top-k
#   - keep the first --top-k passages
#
# Factual-key selection:
#   - triple mode output: --factual-key factual_score
#   - fast mode output:   --factual-key fast_factual_score

python framework/check_merge.py \
  --rel-check-dir path/to/rel_check_dir \
  --factual-check-dir path/to/factual_check_dir \
  --output path/to/output_check_merged.jsonl \
  --top-k 5 \
  --id-key id \
  --passage-key passages \
  --relevance-key local_check \
  --factual-key factual_score
