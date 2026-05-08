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

        linked_entities = self.linker.link_entities(
            text,
            add_description=add_description,
            ner_filter=ner_filter,
        )
        line[ent_key] = linked_entities
        line[f'_{ent_key}_expanded'] = self.expand_repeated_mentions(text, linked_entities)
        return line

    @staticmethod
    def expand_repeated_mentions(text: str, entity_dict: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Expand linked entities to all exact same mention occurrences in text.

        Some EL engines only link one occurrence of a repeated mention. Fast
        mode needs occurrence-level start/end offsets for sentence assignment,
        so this copies the linked entity info to every uncovered exact mention.
        Duplicate dictionary keys are suffixed as ``mention#N`` while the
        original mention text is stored in ``info['mention']``.
        """
        if not text or not entity_dict:
            return entity_dict

        expanded: Dict[str, Dict] = {}
        occupied: set[Tuple[int, int, str]] = set()
        key_counts: Dict[str, int] = {}

        def add_entry(base_key: str, info: Dict, start: int, end: int) -> None:
            raw_mention = info.get('mention', base_key.split('#', 1)[0])
            new_info = dict(info)
            new_info['mention'] = raw_mention
            new_info['start'] = start
            new_info['end'] = end

            count = key_counts.get(raw_mention, 0) + 1
            key_counts[raw_mention] = count
            out_key = raw_mention if count == 1 else f'{raw_mention}#{count}'
            expanded[out_key] = new_info
            occupied.add((start, end, raw_mention))

        # Preserve existing linked occurrences first.
        for key, info in entity_dict.items():
            mention = info.get('mention', key.split('#', 1)[0])
            start = info.get('start')
            end = info.get('end')
            if start is None or end is None:
                expanded[key] = dict(info)
                continue
            add_entry(mention, info, start, end)

        # Copy each linked mention to every identical uncovered text span.
        for key, info in entity_dict.items():
            mention = info.get('mention', key.split('#', 1)[0])
            if not mention:
                continue
            for match in re.finditer(re.escape(mention), text):
                start, end = match.start(), match.end()
                if (start, end, mention) in occupied:
                    continue
                add_entry(mention, info, start, end)

        return expanded

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
                    sent_entities.append({**info, 'mention': info.get('mention', mention)})
            sent_entities.sort(key=lambda x: (x.get('start', 1 << 30), x.get('end', 1 << 30)))
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
                        'pair_idx': len(pair_records),
                        'sentence_idx': sent_idx,
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
        entity_source = line.get(f'_{ent_key}_expanded', line.get(ent_key, {}))
        sentence_entities = self.assign_entities_to_sentences(
            sentence_spans,
            entity_source,
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

    def map_sentence_relations(self, line: dict) -> dict:
        """
        Fast relation stage: map each sentence to exactly one relation using
        its first and last linked entities as template placeholders. All
        sentence-local triples later share this sentence-level relation.
        """
        sentence_entities = line.get('sentence_entities', [])
        sentence_relations: Dict[int, Dict] = {}
        if self.linker is None:
            line['sentence_relations'] = sentence_relations
            return line

        for sent_idx, sent in enumerate(sentence_entities):
            entities = sent.get('entities', [])
            if len(entities) < 2:
                continue
            first = entities[0]
            last = entities[-1]
            relation = self.linker.map_relation_for_sentence(
                sentence=sent.get('text', ''),
                subject=first.get('mention', ''),
                object=last.get('mention', ''),
            )
            if relation:
                sentence_relations[sent_idx] = {
                    **relation,
                    'subject': first.get('mention', ''),
                    'object': last.get('mention', ''),
                    'subject_id': first.get('id'),
                    'object_id': last.get('id'),
                }

        line['sentence_relations'] = sentence_relations
        return line

    def score_entity_pairs(self, line: dict) -> dict:
        """
        Fast score stage: score sentence-local entity pairs as SPO triples.
        Each sentence gets one matched relation, shared by all entity pairs in
        that sentence. Pair direction is single-pass in sentence order, and the
        final output keeps only the best triple per sentence.
        """
        pair_records = line.get('sentence_entity_pairs', [])
        if not pair_records or self.kge is None:
            line['entity_pair_scores'] = []
            return line

        if 'sentence_relations' not in line:
            line = self.map_sentence_relations(line)
        sentence_relations = line.get('sentence_relations', {})

        triple_records: List[Dict] = []
        triple_ids: List[Tuple[str, str, str]] = []
        for pair in pair_records:
            relation_info = sentence_relations.get(pair.get('sentence_idx'))
            if not relation_info:
                continue
            relation_id = relation_info.get('relation_id')
            if not relation_id or not pair.get('head_id') or not pair.get('tail_id'):
                continue
            triple_id = (pair['head_id'], relation_id, pair['tail_id'])
            triple_ids.append(triple_id)
            triple_records.append({
                **pair,
                'relation_id': relation_id,
                'relation_score': relation_info.get('relation_score'),
                'sentence_relation_subject': relation_info.get('subject'),
                'sentence_relation_object': relation_info.get('object'),
                'triple_id': list(triple_id),
            })

        if not triple_ids:
            line['entity_pair_scores'] = []
            return line

        raw_scores = self.kge.score_triples(triple_ids, use_relation=True)
        records_by_triple = {
            tuple(record['triple_id']): record
            for record in triple_records
        }
        scored_triples = []
        for score in raw_scores:
            triple_key = tuple(score.get('triple_id', []))
            record = records_by_triple.get(triple_key)
            if record is None:
                continue
            scored_triples.append({**record, **score})

        line['entity_pair_scores'] = self.select_best_triples_by_sentence(scored_triples)
        return line

    @staticmethod
    def compute_pair_feature_score(pair_score_item: Dict) -> Optional[float]:
        """
        Compute the lower-is-better feature score for one fast-mode triple.
        Mirrors score_feature(): abs(score - avg(ref_score)) when references
        exist. If no references exist, fall back to negative raw KGE score so
        higher KGE plausibility is preferred during best-triple selection.
        """
        triple_score = pair_score_item.get('triple_score', pair_score_item.get('pair_score'))
        if triple_score is None:
            return None
        ref_score = pair_score_item.get('ref_score', [])
        if ref_score:
            import numpy as np
            return float(abs(triple_score - np.average(ref_score)))
        return -float(triple_score)

    @classmethod
    def select_best_triples_by_sentence(cls, triple_scores: List[Dict]) -> List[Dict]:
        """
        Attach lower-is-better ``pair_feature_score`` and keep only one best
        scored triple for each sentence.
        """
        best_by_sentence: Dict[int, Dict] = {}
        for item in triple_scores:
            feature = cls.compute_pair_feature_score(item)
            if feature is None:
                continue
            enriched = {**item, 'pair_feature_score': feature}
            sent_idx = enriched.get('sentence_idx')
            if sent_idx is None:
                continue
            current = best_by_sentence.get(sent_idx)
            if current is None or feature < current.get('pair_feature_score', float('inf')):
                best_by_sentence[sent_idx] = enriched
        return [best_by_sentence[k] for k in sorted(best_by_sentence)]

    def compute_fast_factual_score(self, line: dict) -> Optional[float]:
        """
        Aggregate fast entity-pair scores into a single passage score.
        """
        pair_scores = line.get('entity_pair_scores', [])
        converted_scores = [
            {
                'triple_id': item.get('triple_id', []),
                'triple_score': item.get('triple_score'),
                'ref_score': item.get('ref_score', []),
            }
            for item in pair_scores
        ]
        entity_count = len(line.get('passage_entity', {})) or len(line.get('query_entity', {}))
        return score_feature(converted_scores, entity_count)

    @staticmethod
    def simplify_pair_scores(pairs: List[Dict]) -> List[Dict]:
        """
        Simplify scored entity pairs for JSON output.
        Keeps only entity names, IDs, scoring direction, and the
        lower-is-better relative score.
        """
        return [
            {
                'sentence_idx': item.get('sentence_idx'),
                'head': item.get('head'),
                'tail': item.get('tail'),
                'head_id': item.get('head_id'),
                'tail_id': item.get('tail_id'),
                'relation_id': item.get('relation_id'),
                'relation_score': item.get('relation_score'),
                'triple_id': item.get('triple_id'),
                'triple_score': item.get('triple_score'),
                'relative_score': item.get('pair_feature_score'),
            }
            for item in pairs
        ]

    @classmethod
    def cleanup_fast_output(cls, line: dict, ent_key: str = 'passage_entity') -> dict:
        """
        Remove fast-mode intermediate fields from the output JSON.
        ``passage_entity`` remains the original EL result only; expanded
        repeated mentions are internal and removed here.
        """
        line['entity_pair_scores'] = cls.simplify_pair_scores(
            line.get('entity_pair_scores', [])
        )
        for key in [
            'sentence_spans',
            'sentence_entities',
            'sentence_entity_pairs',
            'sentence_relations',
            'entity_pair_ids',
            f'_{ent_key}_expanded',
        ]:
            line.pop(key, None)
        return line

    def run_fast(
        self,
        line: dict,
        src_key: str = 'passages',
        ent_key: str = 'passage_entity',
    ) -> dict:
        """
        Run the fast factual-check pipeline:
        passage EL → sentence-local entity pairs → sentence relation mapping
        → SPO KGE scoring → one best triple per sentence.
        """
        line = self.extract_entity_pairs(line, src_key=src_key, ent_key=ent_key)
        line = self.map_sentence_relations(line)
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

        # Each input record contains a single passage under line['passages'].
        # Factual-check fields are therefore written directly at the record level.
        if args.mode == "triple":
            if run_extract:
                line = checker.extract_triples(line, src_key="passages", ent_key="passage_entity")
            if run_map:
                line = checker.map_triples_to_ids(line)
            if run_score:
                line = checker.score_triples(line)
                line["factual_score"] = checker.compute_factual_score(line)
        else:
            if run_extract:
                line = checker.extract_entity_pairs(line, src_key="passages", ent_key="passage_entity")
            if run_score:
                line = checker.score_entity_pairs(line)
                line["fast_factual_score"] = checker.compute_fast_factual_score(line)
            line = checker.cleanup_fast_output(line, ent_key="passage_entity")

        results.append(line)

    print()
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
