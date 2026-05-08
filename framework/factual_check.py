"""
Factual check framework module.
Supports both:
1. triple extraction → entity/relation mapping → KGE scoring
2. fast entity-pair extraction → KGE scoring
"""
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    # Step 0 – Passage entity extraction
    # ------------------------------------------------------------------

    def extract_passage_entities(
        self,
        line: dict,
        src_key: str = 'passages',
        ent_key: str = 'passage_entity',
        add_description: bool = True,
        ner_filter: bool = False,
    ) -> dict:
        """
        Run entity linking on the passage stored at *src_key* and write the
        result to ``line[ent_key]``.
        """
        text = line.get(src_key, '')
        if not text or self.linker is None:
            line[ent_key] = line.get(ent_key, {})
            return line

        line[ent_key] = self.linker.link_entities(
            text,
            add_description=add_description,
            ner_filter=ner_filter,
        )
        return line

    # ------------------------------------------------------------------
    # Fast factual check helpers
    # ------------------------------------------------------------------

    @staticmethod
    def split_sentences(text: str) -> List[Dict]:
        """
        Split passage text into sentence spans.
        Returns [{'text', 'start', 'end'}, ...].
        """
        if not text:
            return []

        spans: List[Dict] = []
        for match in re.finditer(r'[^.!?。！？]+[.!?。！？]?', text, flags=re.MULTILINE):
            sent = match.group(0).strip()
            if not sent:
                continue
            spans.append({'text': sent, 'start': match.start(), 'end': match.end()})
        return spans

    @staticmethod
    def assign_entities_to_sentences(
        sentence_spans: List[Dict],
        entity_dict: Dict[str, Dict],
    ) -> List[Dict]:
        """
        Assign passage-level linked entities to sentence spans using start/end.
        Returns [{'text', 'start', 'end', 'entities': [...]}, ...].
        """
        results: List[Dict] = []
        for sent in sentence_spans:
            sent_entities = []
            for mention, info in entity_dict.items():
                start = info.get('start')
                end = info.get('end')
                if start is None or end is None:
                    continue
                if sent['start'] <= start and end <= sent['end']:
                    sent_entities.append({'mention': mention, **info})
            results.append({**sent, 'entities': sent_entities})
        return results

    @staticmethod
    def build_entity_pairs_from_sentences(sentence_entities: List[Dict]) -> List[Dict]:
        """
        Build unordered entity pairs within each sentence.
        Returns [{'sentence', 'head', 'tail'}, ...].
        """
        pair_records: List[Dict] = []
        seen: set[Tuple[str, str, int, int]] = set()
        for sent_idx, sent in enumerate(sentence_entities):
            entities = sent.get('entities', [])
            if len(entities) < 2:
                continue
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    h = entities[i]
                    t = entities[j]
                    h_id = h.get('id')
                    t_id = t.get('id')
                    if not h_id or not t_id or h_id == t_id:
                        continue
                    key = (str(h_id), str(t_id), sent_idx, i * 1000 + j)
                    if key in seen:
                        continue
                    seen.add(key)
                    pair_records.append({
                        'sentence': sent.get('text', ''),
                        'head': h.get('mention', ''),
                        'tail': t.get('mention', ''),
                        'head_id': h_id,
                        'tail_id': t_id,
                    })
        return pair_records

    def extract_entity_pairs(
        self,
        line: dict,
        src_key: str = 'passages',
        ent_key: str = 'passage_entity',
    ) -> dict:
        """
        Fast extract stage:
        passage EL → sentence split → sentence-local entity pairs.
        Writes sentence/entity/pair intermediate fields and ``entity_pair_ids``.
        """
        line = self.extract_passage_entities(line, src_key=src_key, ent_key=ent_key)
        text = line.get(src_key, '')
        sentence_spans = self.split_sentences(text)
        sentence_entities = self.assign_entities_to_sentences(
            sentence_spans,
            line.get(ent_key, {}),
        )
        pair_records = self.build_entity_pairs_from_sentences(sentence_entities)

        line['sentence_spans'] = sentence_spans
        line['sentence_entities'] = sentence_entities
        line['sentence_entity_pairs'] = pair_records
        line['entity_pair_ids'] = [
            (pair['head_id'], pair['tail_id'])
            for pair in pair_records
            if pair.get('head_id') and pair.get('tail_id')
        ]
        return line

    def score_entity_pairs(self, line: dict) -> dict:
        """
        Fast score stage: score ``entity_pair_ids`` with KGE without LLM or
        explicit relation extraction.
        """
        pair_ids = line.get('entity_pair_ids', [])
        if not pair_ids or self.kge is None:
            line['entity_pair_scores'] = []
            return line
        line['entity_pair_scores'] = self.kge.score_entity_pairs(pair_ids)
        return line

    def compute_fast_factual_score(self, line: dict) -> Optional[float]:
        """
        Aggregate fast entity-pair scores into a single passage score.
        """
        pair_scores = line.get('entity_pair_scores', [])
        converted_scores = [
            {
                'triple_id': item.get('pair_id', []),
                'triple_score': item.get('pair_score'),
                'ref_score': item.get('ref_score', []),
            }
            for item in pair_scores
        ]
        entity_count = len(line.get('passage_entity', {})) or len(line.get('query_entity', {}))
        return score_feature(converted_scores, entity_count)

    def run_fast(
        self,
        line: dict,
        src_key: str = 'passages',
        ent_key: str = 'passage_entity',
    ) -> dict:
        """
        Run the fast factual-check pipeline:
        passage EL → sentence-local entity pairs → KGE pair scoring.
        """
        line = self.extract_entity_pairs(line, src_key=src_key, ent_key=ent_key)
        line = self.score_entity_pairs(line)
        return line

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
        passage stored at *src_key*. If ``line[ent_key]`` is missing, entity
        extraction is run first so the prompt can include passage entities.
        Decoded triples are stored under ``line['llm_triple']``.
        """
        if self.llm is None:
            line['llm_triple'] = line.get('llm_triple', [])
            return line

        if ent_key not in line or not line.get(ent_key):
            line = self.extract_passage_entities(line, src_key=src_key, ent_key=ent_key)

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

        # Start from passage-level entity mapping if available; then fall back
        # to query-level entities for compatibility with the broader pipeline.
        entity_dict = dict(line.get('passage_entity', {}))
        for mention, info in line.get('query_entity', {}).items():
            entity_dict.setdefault(mention, info)
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
        entity_count = len(line.get('passage_entity', {})) or len(line.get('query_entity', {}))
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
        entity extract → triple extract → map → score.
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
    parser.add_argument("--mode", choices=["triple", "fast"], default="triple")
    parser.add_argument(
        "--step", choices=["extract", "map", "score", "all"], default="all",
        help="triple mode: extract/map/score/all; fast mode: extract=EL+pair build, score=pair KGE, all=both"
    )
    parser.add_argument("--test", action="store_true", help="Process first 5 lines only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.pipeline.dataset_name = args.dataset

    run_extract = args.step in ("extract", "all")
    run_map = args.step in ("map", "all") and args.mode == "triple"
    run_score = args.step in ("score", "all")

    llm = create_llm_client(cfg.llm) if (run_extract and args.mode == "triple") else None
    linker = EntityLinker(cfg.entity_linker) if (run_extract or run_map or args.mode == "fast") else None
    kge = KGEScorer(cfg.kge) if run_score else None

    checker = FactualChecker(
        llm=llm,
        entity_linker=linker,
        kge_scorer=kge,
        pipeline_cfg=cfg.pipeline,
    )

    data = list(read_jsonl(args.input))
    if args.test:
        data = data[:5]

    results = []
    for i, line in enumerate(data):
        print(f"[factual_check --mode {args.mode} --step {args.step}] {i + 1}/{len(data)}", end="\r")
        candidates = line.get("candidate_passages", [])
        for cand in candidates:
            # Build a temporary single-passage line for the checker
            tmp = {**line, "passages": cand.get("text", "")}
            if args.mode == "triple":
                if run_extract:
                    tmp = checker.extract_triples(tmp, src_key="passages", ent_key="passage_entity")
                    cand["passage_entity"] = tmp.get("passage_entity", {})
                    cand["llm_triple"] = tmp.get("llm_triple", [])
                else:
                    tmp["passage_entity"] = cand.get("passage_entity", {})
                    tmp["llm_triple"] = cand.get("llm_triple", [])
                if run_map:
                    tmp = checker.map_triples_to_ids(tmp)
                    cand["triple_entity_mapping"] = tmp.get("triple_entity_mapping", {})
                    cand["triple_relation_mapping"] = tmp.get("triple_relation_mapping", {})
                    cand["llm_triple_id"] = tmp.get("llm_triple_id", [])
                else:
                    tmp["llm_triple_id"] = cand.get("llm_triple_id", [])
                if run_score:
                    tmp = checker.score_triples(tmp)
                    cand["llm_triple_score"] = tmp.get("llm_triple_score", [])
                    cand["factual_score"] = checker.compute_factual_score(tmp)
            else:
                if run_extract:
                    tmp = checker.extract_entity_pairs(tmp, src_key="passages", ent_key="passage_entity")
                    cand["passage_entity"] = tmp.get("passage_entity", {})
                    cand["sentence_spans"] = tmp.get("sentence_spans", [])
                    cand["sentence_entities"] = tmp.get("sentence_entities", [])
                    cand["sentence_entity_pairs"] = tmp.get("sentence_entity_pairs", [])
                    cand["entity_pair_ids"] = tmp.get("entity_pair_ids", [])
                else:
                    tmp["passage_entity"] = cand.get("passage_entity", {})
                    tmp["entity_pair_ids"] = cand.get("entity_pair_ids", [])
                if run_score:
                    tmp = checker.score_entity_pairs(tmp)
                    cand["entity_pair_scores"] = tmp.get("entity_pair_scores", [])
                    cand["fast_factual_score"] = checker.compute_fast_factual_score(tmp)
        line["candidate_passages"] = candidates
        results.append(line)

    print()
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
