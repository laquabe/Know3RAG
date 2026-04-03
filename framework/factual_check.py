"""
Factual check framework module.
Chains: triple extraction → entity/relation mapping → KGE scoring.
"""
from typing import List, Optional

from config import PipelineConfig
from utils import (
    BaseLLMClient,
    EntityLinker,
    KGEScorer,
    triple_extraction_decode,
    triple_verification,
    score_feature,
)
import prompt.factual_check_triple_extraction as triple_prompt_mod


class FactualChecker:
    """
    Assesses the factual consistency of a passage against Wikidata KG
    by extracting triples from the passage and scoring them with a KGE model.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        entity_linker: EntityLinker,
        kge_scorer: KGEScorer,
        pipeline_cfg: PipelineConfig,
    ):
        self.llm = llm
        self.linker = entity_linker
        self.kge = kge_scorer
        self.cfg = pipeline_cfg

    # ------------------------------------------------------------------
    # Step 1 – Triple extraction
    # ------------------------------------------------------------------

    def extract_triples(
        self,
        line: dict,
        src_key: str = 'passages',
        ent_key: str = 'passage_entity',
    ) -> dict:
        """
        Call LLM to extract (subject, predicate, object) triples from the
        passage stored at *src_key*.  Decoded triples are stored under
        ``line['llm_triple']``.
        """
        messages = triple_prompt_mod.build_prompt(line, src_key=src_key, ent_key=ent_key)
        response = self.llm.call(messages)
        raw, ok = triple_extraction_decode(response)
        if ok:
            line['llm_triple'] = triple_verification(raw)
        else:
            line['llm_triple'] = []
        return line

    # ------------------------------------------------------------------
    # Step 2 – Entity / relation ID mapping
    # ------------------------------------------------------------------

    def map_triples_to_ids(self, line: dict) -> dict:
        """
        Map text triples in ``line['llm_triple']`` to Wikidata IDs.
        Writes:
          - ``line['triple_entity_mapping']`` — {mention: {id, entity}}
          - ``line['triple_relation_mapping']`` — {triple_str: wiki_rel_id}
          - ``line['llm_triple_id']`` — [(s_id, p_id, o_id), ...]
        """
        triples = line.get('llm_triple', [])
        if not triples:
            line['llm_triple_id'] = []
            return line

        # Start from query-level entity mapping if available
        entity_dict = dict(line.get('query_entity', {}))
        entity_dict = self.linker.map_entities_for_triples(triples, entity_dict)
        relation_dict = self.linker.map_relations_for_triples(triples)

        line['triple_entity_mapping'] = entity_dict
        line['triple_relation_mapping'] = relation_dict
        line['llm_triple_id'] = self.linker.map_triple_ids(
            triples, entity_dict, relation_dict
        )
        return line

    # ------------------------------------------------------------------
    # Step 3 – KGE scoring
    # ------------------------------------------------------------------

    def score_triples(self, line: dict) -> dict:
        """
        Score triples in ``line['llm_triple_id']`` with the KGE model.
        Writes ``line['llm_triple_score']`` — list of score dicts.
        """
        triple_ids = line.get('llm_triple_id', [])
        if not triple_ids:
            line['llm_triple_score'] = []
            return line
        line['llm_triple_score'] = self.kge.score_triples(triple_ids)
        return line

    # ------------------------------------------------------------------
    # Scoring helper
    # ------------------------------------------------------------------

    def compute_factual_score(self, line: dict) -> Optional[float]:
        """
        Aggregate triple scores into a single factual-consistency score.
        Returns None if no scored triples are available.
        """
        scores = line.get('llm_triple_score', [])
        entity_count = len(line.get('query_entity', {}))
        return score_feature(scores, entity_count)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        line: dict,
        src_key: str = 'passages',
        ent_key: str = 'passage_entity',
    ) -> dict:
        """
        Run the full factual-check pipeline:
        extract → map → score.
        """
        line = self.extract_triples(line, src_key=src_key, ent_key=ent_key)
        line = self.map_triples_to_ids(line)
        line = self.score_triples(line)
        return line


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from config import load_config
    from utils import create_llm_client, EntityLinker, KGEScorer, read_jsonl, write_jsonl

    parser = argparse.ArgumentParser(description="Know3RAG factual check stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--step", choices=["extract", "map", "score", "all"], default="all",
        help="extract: LLM triple extraction; map: EntityLinker ID mapping; "
             "score: KGE scoring; all: all three steps"
    )
    parser.add_argument("--test", action="store_true", help="Process first 5 lines only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.pipeline.dataset_name = args.dataset

    run_extract = args.step in ("extract", "all")
    run_map = args.step in ("map", "all")
    run_score = args.step in ("score", "all")

    llm = create_llm_client(cfg.llm) if run_extract else None
    linker = EntityLinker(cfg.entity_linker) if run_map else None
    kge = KGEScorer(cfg.kge) if run_score else None

    checker = FactualChecker(
        llm=llm,
        entity_linker=linker,
        kge_scorer=kge,
        pipeline_cfg=cfg.pipeline,
    )

    data = read_jsonl(args.input)
    if args.test:
        data = data[:5]

    results = []
    for i, line in enumerate(data):
        print(f"[factual_check --step {args.step}] {i + 1}/{len(data)}", end="\r")
        candidates = line.get("candidate_passages", [])
        for cand in candidates:
            # Build a temporary single-passage line for the checker
            tmp = {**line, "passages": cand.get("text", ""), "passage_entity": {}}
            if run_extract:
                tmp = checker.extract_triples(tmp, src_key="passages", ent_key="passage_entity")
                cand["llm_triple"] = tmp.get("llm_triple", [])
            else:
                tmp["llm_triple"] = cand.get("llm_triple", [])
            if run_map:
                tmp = checker.map_triples_to_ids(tmp)
                cand["llm_triple_id"] = tmp.get("llm_triple_id", [])
            else:
                tmp["llm_triple_id"] = cand.get("llm_triple_id", [])
            if run_score:
                tmp = checker.score_triples(tmp)
                cand["llm_triple_score"] = tmp.get("llm_triple_score", [])
                cand["factual_score"] = checker.compute_factual_score(tmp)
        line["candidate_passages"] = candidates
        results.append(line)

    print()
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
