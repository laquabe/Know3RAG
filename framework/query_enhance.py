"""
Query enhancement framework module.
Handles KG-based query enrichment, follow-up question generation, and self-ask.
"""
import os
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PipelineConfig
from utils import BaseLLMClient, EntityLinker, KGEScorer, WikidataClient
import prompt.query_enhance_decompose_question as decompose_mod
import prompt.query_enhance_selfask as selfask_mod
import prompt.query_enhance_generate_question as gen_question_mod


class QueryEnhancer:
    """
    Enriches a question with KG entity context and generates follow-up queries
    to drive the closed-loop retrieval cycle.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        entity_linker: EntityLinker,
        kge_scorer: KGEScorer,
        wikidata: WikidataClient,
        pipeline_cfg: PipelineConfig,
    ):
        self.llm = llm
        self.linker = entity_linker
        self.kge = kge_scorer
        self.wikidata = wikidata
        self.cfg = pipeline_cfg

    # ------------------------------------------------------------------
    # Entity linking
    # ------------------------------------------------------------------

    def link_query_entities(
        self,
        line: dict,
        question_key: str = 'question',
        ner_filter: bool = False,
    ) -> dict:
        """
        Run entity linking on the question text.
        Writes ``line['query_entity']`` — {mention: {id, entity, description}}.
        """
        question = line.get(question_key, line.get('Question', ''))
        line['query_entity'] = self.linker.link_entities(
            question, add_description=True, ner_filter=ner_filter
        )
        return line

    # ------------------------------------------------------------------
    # Wikidata enrichment
    # ------------------------------------------------------------------

    def fetch_entity_claims(self, line: dict) -> dict:
        """
        Fetch Wikidata claims for each entity in ``line['query_entity']``.
        Updates descriptions and adds ``claims`` / ``kg_triple_id`` fields.
        """
        line = self.wikidata.enrich_query_entities(line, entity_key='query_entity')
        return line

    # ------------------------------------------------------------------
    # KG relation prediction
    # ------------------------------------------------------------------

    def predict_related_relations(self, line: dict) -> dict:
        """
        For each query entity, predict the most relevant Wikidata relation(s)
        given the question.
        Writes ``line['query_relation']`` — {mention: [wiki_rel_id, ...]}.
        """
        question = line.get('question', line.get('Question', ''))
        query_relation: dict = {}
        for mention, ent_info in line.get('query_entity', {}).items():
            top_rels, local_rel = self.linker.convert_question_to_relations(
                question, mention, ent_info,
                topk=10, count_num=3,
            )
            query_relation[mention] = {
                'top_relations': top_rels,
                'local_relation': local_rel,
            }
        line['query_relation'] = query_relation
        return line

    # ------------------------------------------------------------------
    # KGE tail prediction
    # ------------------------------------------------------------------

    def predict_tail_entities(self, line: dict) -> dict:
        """
        For each query entity, use the KGE model to predict tail entities
        along the predicted relations.
        Writes ``line['kg_tail_pred']`` and ``line['kg_tail_id_set']``.
        """
        tail_preds: list = []
        tail_id_set: set = set()
        for mention, ent_info in line.get('query_entity', {}).items():
            tails, ids = self.kge.predict_tail(ent_info)
            tail_preds.extend(tails)
            tail_id_set.update(ids)
        line['kg_tail_pred'] = tail_preds
        line['kg_tail_id_set'] = tail_id_set
        return line

    # ------------------------------------------------------------------
    # Entity description expansion
    # ------------------------------------------------------------------

    def expand_entity_descriptions(self, line: dict, tail_map: dict) -> dict:
        """
        Enrich entity descriptions with KG neighbour information from *tail_map*.
        Updates description strings in ``line['query_entity']``.
        """
        for mention, ent_info in line.get('query_entity', {}).items():
            new_des, _ = self.linker.expand_entity_description(ent_info, tail_map)
            line['query_entity'][mention]['description'] = new_des
        return line

    # ------------------------------------------------------------------
    # LLM-based query operations
    # ------------------------------------------------------------------

    def generate_followup_question(
        self,
        line: dict,
        output_key: str = 'new_question',
    ) -> dict:
        """
        Generate a follow-up retrieval query based on the current answer.
        Writes ``line[output_key]``.
        """
        messages = gen_question_mod.build_prompt(line)
        response = self.llm.call(messages)
        line[output_key] = response.strip()
        return line

    def selfask(
        self,
        line: dict,
        have_choice: bool = False,
        output_key: str = 'need_more_info',
    ) -> dict:
        """
        Ask the LLM whether it needs more information.
        Writes ``line[output_key]`` as a raw string (starts with 'yes' or 'no').
        """
        prompt_str = selfask_mod.build_prompt(line, have_choice=have_choice)
        response = self.llm.call(prompt_str)
        line[output_key] = response.strip().lower()
        return line

    def decompose_question(
        self,
        line: dict,
        output_key: str = 'sub_questions',
    ) -> dict:
        """
        Decompose the question into entity-level sub-questions.
        Writes ``line[output_key]`` as a raw string (JSON dict from LLM).
        """
        prompt_str = decompose_mod.build_prompt(line)
        response = self.llm.call(prompt_str)
        line[output_key] = response.strip()
        return line

    # ------------------------------------------------------------------
    # Full KG enhancement pipeline
    # ------------------------------------------------------------------

    def run_kg_enhance(self, line: dict, tail_map: Optional[dict] = None) -> dict:
        """
        Run the full KG-based query enhancement pipeline:
        link → fetch claims → predict relations → predict tails →
        expand descriptions (if tail_map provided).
        """
        line = self.link_query_entities(line)
        line = self.fetch_entity_claims(line)
        line = self.predict_related_relations(line)
        line = self.predict_tail_entities(line)
        if tail_map is not None:
            line = self.expand_entity_descriptions(line, tail_map)
        return line


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from config import load_config
    from utils import (
        create_llm_client, EntityLinker, KGEScorer, WikidataClient, read_jsonl, write_jsonl
    )
    from framework.document_generation import DocumentGenerator

    parser = argparse.ArgumentParser(description="Know3RAG query enhancement stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--question-key", default="question",
                        help="Line field containing the question text")
    parser.add_argument(
        "--step", choices=["kg", "llm", "all"], default="kg",
        help="kg: entity link+Wikidata+KGE (no LLM); llm: followup question; all: both"
    )
    parser.add_argument("--test", action="store_true", help="Process first 5 lines only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.pipeline.dataset_name = args.dataset

    run_kg = args.step in ("kg", "all")
    run_llm = args.step in ("llm", "all")

    # Instantiate only the models actually needed
    llm = create_llm_client(cfg.llm) if run_llm else None
    linker = EntityLinker(cfg.entity_linker) if run_kg else None
    kge = KGEScorer(cfg.kge) if run_kg else None
    wikidata = WikidataClient() if run_kg else None

    enhancer = QueryEnhancer(
        llm=llm,
        entity_linker=linker,
        kge_scorer=kge,
        wikidata=wikidata,
        pipeline_cfg=cfg.pipeline,
    )

    data = list(read_jsonl(args.input))
    if args.test:
        data = data[:5]

    results = []
    for i, line in enumerate(data):
        print(f"[query_enhance] {i + 1}/{len(data)}", end="\r")
        if run_kg:
            line = enhancer.run_kg_enhance(line)
            # Append KG knowledge card (pure string-concat, no model needed)
            doc_gen = DocumentGenerator(llm=None, retriever=None, pipeline_cfg=cfg.pipeline)
            line = doc_gen.generate_knowledge_card(line)
        if run_llm:
            line = enhancer.generate_followup_question(line)
        results.append(line)

    print()
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
