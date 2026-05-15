#!/bin/bash
# Document generation stage: generate one reference passage into `passages`.
#
# Input JSONL format:
#   - question: base query
#   - query_entity: optional entity context for entity-enhanced generation
#
# Choose generator with --step:
#   --step llm   : use LLM generator, e.g. Llama / Qwen / API model from config.json
#   --step card  : use fine-tuned knowledge-card generator, requires --card-model-path
#
# Choose whether to use entities:
#   without --add-entity : question-only generation
#   with    --add-entity : question + query_entity enhanced generation

python3 framework/document_generation.py \
  --config config.json \
  --input datasets/test/turn0_query_enhanced.jsonl \
  --output datasets/test/doc_generated.jsonl \
  --step llm

# Examples:
#
# 1) LLM + query_entity enhanced:
# python3 framework/document_generation.py \
#   --config config.json \
#   --input datasets/test/turn0_query_enhanced.jsonl \
#   --output datasets/test/doc_llm_entity.jsonl \
#   --step llm \
#   --add-entity
#
# 2) Knowledge-card model, question only:
# python3 framework/document_generation.py \
#   --config config.json \
#   --input datasets/test/turn0_query_enhanced.jsonl \
#   --output datasets/test/doc_card_question.jsonl \
#   --step card \
#   --card-model-path /path/to/knowledge-card-model \
#   --card-device 0
#
# 3) Knowledge-card model + query_entity enhanced:
# python3 framework/document_generation.py \
#   --config config.json \
#   --input datasets/test/turn0_query_enhanced.jsonl \
#   --output datasets/test/doc_card_entity.jsonl \
#   --step card \
#   --card-model-path /path/to/knowledge-card-model \
#   --card-device 0 \
#   --add-entity
