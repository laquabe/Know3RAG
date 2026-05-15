"""
Query enhancement framework module.
Handles KG-based query enrichment, follow-up question generation, and self-ask.
"""
import os
import sys
import json
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PipelineConfig
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
        llm,
        entity_linker,
        kge_scorer,
        wikidata,
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
            # KGEScorer.predict_tail() consumes these fields from ent_info.
            # Keep line['query_relation'] for inspection/backward compatibility,
            # and also write the prediction back to the entity record.
            ent_info['pred_relation_rank'] = top_rels
            ent_info['local_pred_r'] = local_rel
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
        # JSONL output cannot serialize set; keep deterministic list output.
        line['kg_tail_id_set'] = sorted(tail_id_set)
        return line

    def fetch_tail_entity_info(
        self,
        line: dict,
        tail_ids_key: str = 'kg_tail_id_set',
        output_key: str = 'tail_entity_info',
    ) -> dict:
        """
        Fetch Wikidata information for predicted tail entities.

        Reads ``line[tail_ids_key]`` and writes ``line[output_key]`` as
        {tail_wiki_id: {labels, descriptions, aliases, claims}}.
        """
        tail_ids = line.get(tail_ids_key, []) or []
        if isinstance(tail_ids, set):
            tail_ids = sorted(tail_ids)

        tail_info = {}
        for tail_id in tail_ids:
            if not tail_id:
                continue
            info = self.wikidata.query_entity(str(tail_id))
            if info:
                tail_info[str(tail_id)] = info
        line[output_key] = tail_info
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

    # ------------------------------------------------------------------
    # Split-stage runners for incompatible runtime environments
    # ------------------------------------------------------------------

    def run_query_el(self, line: dict, question_key: str = 'question') -> dict:
        """Stage: query entity linking only. Requires EntityLinker EL env."""
        return self.link_query_entities(line, question_key=question_key)

    def run_query_wikidata(self, line: dict) -> dict:
        """Stage: fetch Wikidata claims/descriptions for query entities only."""
        return self.fetch_entity_claims(line)

    def run_relation_tail(self, line: dict) -> dict:
        """Stage: relation prediction + KGE tail prediction only."""
        line = self.predict_related_relations(line)
        line = self.predict_tail_entities(line)
        return line

    def run_tail_wikidata(self, line: dict) -> dict:
        """Stage: fetch Wikidata info for predicted tail entities only."""
        return self.fetch_tail_entity_info(line)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from config import load_config

    parser = argparse.ArgumentParser(description="Know3RAG query enhancement stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--question-key", default="question",
                        help="Line field containing the question text")
    parser.add_argument(
        "--step", choices=["kg", "llm", "all"], default="kg",
        help="Deprecated coarse mode: kg, llm, or all. Prefer --stage."
    )
    parser.add_argument(
        "--stage",
        choices=[
            "query-el",
            "query-wikidata",
            "relation-tail",
            "tail-wikidata",
            "tail-wikidata-card",
            "followup",
            "kg",
            "all",
        ],
        default=None,
        help=(
            "Split runtime stage. Use query-el, query-wikidata, relation-tail, "
            "tail-wikidata-card, followup for incompatible environments. "
            "kg/all keep the legacy combined behavior."
        ),
    )
    parser.add_argument("--test", action="store_true", help="Process first 5 lines only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.pipeline.dataset_name = args.dataset

    stage = args.stage
    if stage is None:
        stage = "followup" if args.step == "llm" else args.step

    needs_llm = stage in ("followup", "all")
    needs_linker = stage in ("query-el", "relation-tail", "kg", "all")
    needs_kge = stage in ("relation-tail", "kg", "all")
    needs_wikidata = stage in (
        "query-wikidata", "tail-wikidata", "tail-wikidata-card", "kg", "all"
    )
    needs_card = stage in ("tail-wikidata-card", "kg", "all")

    # Instantiate only the models actually needed
    llm = None
    if needs_llm:
        from utils.llm_client import create_llm_client
        llm = create_llm_client(cfg.llm)

    linker = None
    if needs_linker:
        from utils.entity_linker import EntityLinker
        linker = EntityLinker(cfg.entity_linker)

    kge = None
    if needs_kge:
        from utils.kge_scorer import KGEScorer
        kge = KGEScorer(cfg.kge)

    wikidata = None
    if needs_wikidata:
        from utils.wikidata_client import WikidataClient
        wikidata = WikidataClient()

    enhancer = QueryEnhancer(
        llm=llm,
        entity_linker=linker,
        kge_scorer=kge,
        wikidata=wikidata,
        pipeline_cfg=cfg.pipeline,
    )

    data = list(_read_jsonl(args.input))
    if args.test:
        data = data[:5]

    results = []
    for i, line in enumerate(data):
        print(f"[query_enhance] {i + 1}/{len(data)}", end="\r")
        if stage == "query-el":
            line = enhancer.run_query_el(line, question_key=args.question_key)
        elif stage == "query-wikidata":
            line = enhancer.run_query_wikidata(line)
        elif stage == "relation-tail":
            line = enhancer.run_relation_tail(line)
        elif stage == "tail-wikidata":
            line = enhancer.run_tail_wikidata(line)
        elif stage == "tail-wikidata-card":
            line = enhancer.run_tail_wikidata(line)
            line = _append_tail_info_to_kg_tail_pred(line)
            line = _generate_knowledge_card(line)
        elif stage == "followup":
            line = enhancer.generate_followup_question(line)
        elif stage == "kg":
            line = enhancer.run_kg_enhance(line)
            # Append KG knowledge card (pure string-concat, no model needed)
            line = _generate_knowledge_card(line)
        elif stage == "all":
            line = enhancer.run_kg_enhance(line)
            line = _generate_knowledge_card(line)
            line = enhancer.generate_followup_question(line)
        else:
            raise ValueError(f"Unsupported query enhancement stage: {stage}")
        results.append(line)

    print()
    _write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


def _read_jsonl(path: str):
    """Local JSONL reader to avoid importing utils aggregate modules per stage."""
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: str, records: list) -> None:
    """Local JSONL writer to keep split runtime imports minimal."""
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _generate_knowledge_card(
    line: dict,
    output_key: str = 'pseudo_doc_entity',
) -> dict:
    """
    Lightweight KG knowledge-card builder equivalent to
    DocumentGenerator.generate_knowledge_card(), kept local so split stages do
    not need to import document-generation dependencies.
    """
    parts = []
    for mention, ent_info in line.get('query_entity', {}).items():
        desc = ent_info.get('description', '')
        if desc:
            parts.append("{}: {}".format(mention, desc))

    for tail in line.get('kg_tail_pred', []):
        if isinstance(tail, str) and tail:
            parts.append(tail)
        elif isinstance(tail, dict):
            label = tail.get('labels', tail.get('label', ''))
            desc = tail.get('description', tail.get('descriptions', ''))
            if label:
                parts.append("{}: {}".format(label, desc) if desc else label)

    line[output_key] = " ".join(parts)
    return line


def _append_tail_info_to_kg_tail_pred(line: dict) -> dict:
    """
    Adds tail Wikidata label/description dicts to kg_tail_pred so the existing
    DocumentGenerator.generate_knowledge_card() can include tail information
    without changing its public interface.
    """
    tail_info = line.get('tail_entity_info', {}) or {}
    if not tail_info:
        return line

    kg_tail_pred = list(line.get('kg_tail_pred', []) or [])
    existing_tail_info = {
        item.get('id') or item.get('wiki_id')
        for item in kg_tail_pred
        if isinstance(item, dict)
    }
    for tail_id, info in tail_info.items():
        if tail_id in existing_tail_info:
            continue
        item = dict(info)
        item['id'] = tail_id
        kg_tail_pred.append(item)
    line['kg_tail_pred'] = kg_tail_pred
    return line


if __name__ == "__main__":
    main()
