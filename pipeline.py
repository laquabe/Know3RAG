"""
Know3RAG closed-loop pipeline orchestrator.
Wires together all framework modules and runs end-to-end QA.
"""
import argparse
import json
from typing import Optional

from config import Config, load_config
from utils import (
    create_llm_client,
    EntityLinker,
    KGEScorer,
    WikidataClient,
    HybridRetriever,
    read_jsonl,
    write_jsonl,
)
from framework import (
    QuestionAnswerer,
    FactualChecker,
    QueryEnhancer,
    DocumentGenerator,
    RelevanceChecker,
)


class Know3RAGPipeline:
    """
    Closed-loop RAG pipeline (paper Figure 2).

    Stages per question:
      1. KG-based query enhancement
      2. Multi-source reference generation (LLM + retriever + KG card)
      3. Relevance filtering (LLM check + KGE ranking)
      4. Turn-0 answer generation
      5. Loop up to max_loop_turns:
           a. Generate follow-up query
           b. Retrieve new passages
           c. Filter new passages
           d. Generate new turn answer
      6. Use the latest turn answer as final output
    """

    def __init__(self, config: Config):
        self.cfg = config

        # Shared utilities
        llm = create_llm_client(config.llm)
        linker = EntityLinker(config.entity_linker)
        kge = KGEScorer(config.kge)
        wikidata = WikidataClient()
        retriever: Optional[HybridRetriever] = None
        if config.pipeline.use_retriever:
            retriever = HybridRetriever(config.retriever)
            retriever.load_index()

        # Framework modules
        self.qa = QuestionAnswerer(llm, config.pipeline)
        self.factual = FactualChecker(llm, linker, kge, config.pipeline)
        self.query_enh = QueryEnhancer(llm, linker, kge, wikidata, config.pipeline)
        self.doc_gen = DocumentGenerator(llm, retriever, config.pipeline)
        self.relevance = RelevanceChecker(llm, kge, config.pipeline)

    # ------------------------------------------------------------------
    # Single-question pipeline
    # ------------------------------------------------------------------

    def run_single(self, line: dict) -> dict:
        cfg = self.cfg.pipeline
        have_choice = cfg.dataset_name == 'MMLU'

        # Stage 1 — KG query enhancement
        if cfg.use_kg_query_enhance:
            line = self.query_enh.run_kg_enhance(line)

        # Stage 2 — Reference generation
        line = self.doc_gen.generate_llm_reference(
            line, have_choice=have_choice, add_entity=cfg.use_kg_query_enhance
        )
        if cfg.use_retriever:
            line = self.doc_gen.retrieve_with_entities(line)

        if cfg.use_kg_query_enhance:
            line = self.doc_gen.generate_knowledge_card(line)

        # Merge all candidate sources
        sources = ['pseudo_doc']
        if cfg.use_retriever:
            sources.append('retrieved_passages')
        if cfg.use_kg_query_enhance:
            sources.append('pseudo_doc_entity')
        line = self.doc_gen.merge_references(line, sources=sources)

        # Stage 3 — Relevance filtering
        candidates = [
            {'text': t, 'factual_score': None}
            for t in line.get('candidate_passages', [])
        ]

        # Optional factual scoring per candidate
        if cfg.use_kg_factual_check:
            for cand in candidates:
                tmp = dict(line)
                tmp['passages'] = cand['text']
                tmp['passage_entity'] = {}
                tmp = self.factual.run(tmp, src_key='passages')
                cand['factual_score'] = self.factual.compute_factual_score(tmp)

        line = self.relevance.select_references(
            line,
            candidate_passages=candidates,
            have_choice=have_choice,
            use_llm_check=cfg.use_llm_relevance_check,
            use_kge_rank=cfg.use_kg_factual_check,
        )

        # Stage 4 — Turn-0 answer
        line = self.qa.answer(line, answer_key='llm_response_0')
        line['llm_response'] = line['llm_response_0']

        # Stage 5 — Closed-loop refinement turns
        for turn in range(cfg.max_loop_turns):
            # Check if more info is needed
            if cfg.use_kg_query_enhance:
                line = self.query_enh.selfask(line, have_choice=have_choice)
                if not line.get('need_more_info', 'yes').startswith('yes'):
                    break

            # Generate follow-up query
            line = self.query_enh.generate_followup_question(line)
            new_q = line.get('new_question', '')
            if not new_q:
                break

            # Retrieve new passages for the follow-up query
            tmp_line = dict(line)
            tmp_line['question'] = new_q
            tmp_line = self.doc_gen.retrieve_passages(tmp_line, query_key='question')
            new_candidates = [
                {'text': d.get('text', ''), 'factual_score': None}
                for d in tmp_line.get('retrieved_passages', [])
                if d.get('text')
            ]

            if new_candidates:
                if cfg.use_kg_factual_check:
                    for cand in new_candidates:
                        t2 = dict(line)
                        t2['passages'] = cand['text']
                        t2['passage_entity'] = {}
                        t2 = self.factual.run(t2, src_key='passages')
                        cand['factual_score'] = self.factual.compute_factual_score(t2)

                # Append good new passages to line['reference']
                tmp_line2 = dict(line)
                tmp_line2 = self.relevance.select_references(
                    tmp_line2,
                    candidate_passages=new_candidates,
                    have_choice=have_choice,
                    use_llm_check=cfg.use_llm_relevance_check,
                    use_kge_rank=cfg.use_kg_factual_check,
                    output_key='new_reference',
                )
                existing = line.get('reference', [])
                line['reference'] = existing + tmp_line2.get('new_reference', [])

            # Generate new turn answer
            ans_key = 'llm_response_{}'.format(turn + 1)
            line = self.qa.answer(line, answer_key=ans_key)
            line['llm_response'] = line[ans_key]

        return line

    # ------------------------------------------------------------------
    # Dataset-level runner
    # ------------------------------------------------------------------

    def run_dataset(
        self,
        input_path: str,
        output_path: str,
        test: bool = False,
    ) -> None:
        data = read_jsonl(input_path)
        if test:
            data = data[:5]

        results = []
        for i, line in enumerate(data):
            print("Processing {}/{}".format(i + 1, len(data)), end='\r')
            result = self.run_single(line)
            results.append(result)

        write_jsonl(output_path, results)
        print("\nDone. Wrote {} results to {}".format(len(results), output_path))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Know3RAG pipeline")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--dataset", default=None, help="Override dataset_name in config")
    parser.add_argument("--test", action="store_true", help="Run on first 5 examples only")
    parser.add_argument("--no-kg", action="store_true", help="Disable all KG features")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)

    if args.dataset:
        cfg.pipeline.dataset_name = args.dataset
    if args.no_kg:
        cfg.pipeline.use_kg_query_enhance = False
        cfg.pipeline.use_kg_factual_check = False

    pipeline = Know3RAGPipeline(cfg)
    pipeline.run_dataset(args.input, args.output, test=args.test)
