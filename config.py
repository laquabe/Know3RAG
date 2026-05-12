from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    # API-based models
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_api_model: str = "qwen2.5-32b-instruct"
    api_max_tokens: int = 1024
    api_timeout: int = 30
    api_batch_size: int = 10

    # Local HuggingFace models
    # Supported: Mistral | Llama | Qwen | GLM3 | GLM4 | Baichuan | Yi | Zephyr | Qwen_api | api
    local_model_name: str = "api"
    local_model_path: str = ""
    local_max_new_tokens: int = 1024
    local_do_sample: bool = False


@dataclass
class KGEConfig:
    checkpoint_path: str = ""           # path to wikidata5m-complex.pt
    entity_ids_path: str = ""           # wikidata5m entity_ids.json
    relation_ids_path: str = ""         # wikidata5m relation_ids.json
    dataset_path: str = ""              # directory containing train.txt / valid.txt / test.txt
    max_ref_num: int = 3


@dataclass
class EntityLinkerConfig:
    engine: str = "spacy"                 # spacy | refined

    # spaCy entityLinker config
    spacy_model: str = "en_core_web_md"

    # ReFinED config
    refined_model_name: str = "wikipedia_model"
    refined_entity_set: str = "wikidata"
    refined_device: str = ""

    # Relation mapping config
    sbert_model_path: str = ""          # e.g. all-mpnet-base-v2
    relation_file: str = ""             # datasets/relation.json
    relation_template_file: str = ""    # datasets/relation_template.json
    relation_sentence_template_file: str = ""
    ner_filter: bool = True
    add_description: bool = True
    topk_relations: int = 10
    count_num: int = 3


@dataclass
class RetrieverConfig:
    corpus_path: Optional[str] = None          # path to corpus JSONL: {id, text, title?}
    index_dir: str = "retriever_index/"

    # Retrieval mode:
    #   bm25    -> sparse BM25 only
    #   colbert -> ColBERTv2 only
    #   hybrid  -> BM25 + ColBERTv2 score fusion
    #   sbert   -> legacy SentenceTransformer dense retrieval only
    retrieval_mode: str = "hybrid"

    # Legacy SentenceTransformer dense retriever config.
    sbert_model_path: str = ""                  # e.g. all-mpnet-base-v2

    # ColBERTv2 config. `colbert_checkpoint` can be a HuggingFace model name
    # such as colbert-ir/colbertv2.0 or a local checkpoint path.
    colbert_checkpoint: str = "colbert-ir/colbertv2.0"
    colbert_index_name: str = "know3rag_colbert"
    colbert_root: str = ""                    # defaults to {index_dir}/colbert when empty
    colbert_nbits: int = 2

    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    top_k: int = 5
    show_progress: bool = True


@dataclass
class PipelineConfig:
    dataset_name: str = "hotpotQA"   # hotpotQA | 2wikimultihopQA | PopQA | MMLU
    dataset_path: str = ""
    output_dir: str = "result/"
    top_k_references: int = 5
    max_loop_turns: int = 2
    use_kg_query_enhance: bool = True
    use_kg_factual_check: bool = True
    use_llm_relevance_check: bool = True
    use_retriever: bool = False
    use_knowledge_card: bool = False
    knowledge_card_model_path: str = ""
    fast_mode_use_relation: bool = False


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    kge: KGEConfig = field(default_factory=KGEConfig)
    entity_linker: EntityLinkerConfig = field(default_factory=EntityLinkerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def load_config(path: Optional[str] = None) -> Config:
    """Load config from a JSON file, merging with defaults."""
    cfg = Config()
    if path is None:
        return cfg
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Config file not found: '{}'. Please pass an existing JSON file with "
            "--config, set CONFIG=/path/to/config.json when using scripts/*.sh, "
            "or create config.json in the project root.".format(path)
        )
    with open(path) as f:
        data = json.load(f)

    def _update(obj, updates: dict):
        for k, v in updates.items():
            if hasattr(obj, k):
                if isinstance(v, dict):
                    _update(getattr(obj, k), v)
                else:
                    setattr(obj, k, v)

    _update(cfg, data)
    return cfg
