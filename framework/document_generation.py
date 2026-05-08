"""
Document generation framework module.
Handles LLM-based reference generation, local retrieval, and knowledge card building.
"""
import os
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PipelineConfig
from utils import BaseLLMClient, HybridRetriever
import prompt.document_generation_reference_generate as ref_gen_mod


class DocumentGenerator:
    """
    Generates reference documents for a question from three sources:
      1. LLM self-generation
      2. Local hybrid retriever (BM25 + dense)
      3. KG-based knowledge card (entity descriptions + tail predictions)
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        retriever: Optional[HybridRetriever],
        pipeline_cfg: PipelineConfig,
    ):
        self.llm = llm
        self.retriever = retriever
        self.cfg = pipeline_cfg

    # ------------------------------------------------------------------
    # LLM reference generation
    # ------------------------------------------------------------------

    def generate_llm_reference(
        self,
        line: dict,
        have_choice: bool = False,
        add_entity: bool = False,
        cot_prompt: str = None,
        output_key: str = 'pseudo_doc',
    ) -> dict:
        """
        Ask the LLM to write a reference paragraph for the question.
        Result stored under *output_key*.
        """
        prompt = ref_gen_mod.build_prompt(
            line,
            have_choice=have_choice,
            add_entity=add_entity,
            cot_prompt=cot_prompt,
        )
        response = self.llm.call(prompt)
        # Strip "Reference: " prefix if present (MMLU / open-ended)
        resp = response.strip()
        if resp.startswith("Reference:"):
            resp = resp[len("Reference:"):].strip()
        line[output_key] = resp
        return line

    # ------------------------------------------------------------------
    # Retriever-based passage retrieval
    # ------------------------------------------------------------------

    def retrieve_passages(
        self,
        line: dict,
        query_key: str = 'question',
        top_k: Optional[int] = None,
        output_key: str = 'retrieved_passages',
    ) -> dict:
        """
        Retrieve passages from the local corpus using the query stored at
        *query_key*.  Results stored as a list of dicts under *output_key*.
        """
        if self.retriever is None:
            line[output_key] = []
            return line
        query = line.get(query_key, line.get('Question', ''))
        k = top_k or self.cfg.top_k_references
        results = self.retriever.retrieve(query, top_k=k)
        line[output_key] = results
        return line

    def retrieve_with_entities(
        self,
        line: dict,
        top_k: Optional[int] = None,
        output_key: str = 'retrieved_passages',
    ) -> dict:
        """
        Retrieve passages using entity-enriched queries.
        Merges results from the base question and each entity mention.
        """
        if self.retriever is None:
            line[output_key] = []
            return line

        k = top_k or self.cfg.top_k_references
        base_q = line.get('question', line.get('Question', ''))

        # Build entity-enriched queries
        queries = [base_q]
        for mention, ent_info in line.get('query_entity', {}).items():
            desc = ent_info.get('description', '')
            if desc:
                queries.append("{} {}".format(mention, desc))

        # Batch retrieve, de-duplicate by doc id
        all_results: Dict[str, dict] = {}
        for results in self.retriever.retrieve_batch(queries, top_k=k):
            for doc in results:
                doc_id = str(doc['id'])
                if doc_id not in all_results or doc['score'] > all_results[doc_id]['score']:
                    all_results[doc_id] = doc

        # Sort by score descending, trim to top_k
        sorted_docs = sorted(all_results.values(), key=lambda d: d['score'], reverse=True)[:k]
        line[output_key] = sorted_docs
        return line

    # ------------------------------------------------------------------
    # KG knowledge card
    # ------------------------------------------------------------------

    def generate_knowledge_card(
        self,
        line: dict,
        output_key: str = 'pseudo_doc_entity',
    ) -> dict:
        """
        Build a knowledge card from entity descriptions and KGE tail predictions.
        Writes a single concatenated string under *output_key*.
        """
        parts: List[str] = []
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

    # ------------------------------------------------------------------
    # Reference merging
    # ------------------------------------------------------------------

    def merge_references(
        self,
        line: dict,
        sources: List[str],
        output_key: str = 'candidate_passages',
    ) -> dict:
        """
        Merge passages from multiple source keys into a single candidate list.
        Each entry in *sources* names a key in *line* holding a str or list.
        Writes the merged list under *output_key*.
        """
        merged: List[str] = []
        for src_key in sources:
            val = line.get(src_key)
            if val is None:
                continue
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        merged.append(item.get('text', ''))
                    elif isinstance(item, str):
                        merged.append(item)
            elif isinstance(val, str) and val:
                merged.append(val)
        line[output_key] = [p for p in merged if p]
        return line


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from config import load_config
    from utils import create_llm_client, HybridRetriever, read_jsonl, write_jsonl

    parser = argparse.ArgumentParser(description="Know3RAG document generation stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--query-key", default="question",
                        help="Line field used as retrieval query")
    parser.add_argument(
        "--step", choices=["llm", "retrieve", "all"], default="all",
        help="llm: LLM reference only; retrieve: retriever only; all: both"
    )
    parser.add_argument("--have-choice", action="store_true",
                        help="MMLU multiple-choice mode")
    parser.add_argument("--add-entity", action="store_true",
                        help="Include entity context in LLM prompt")
    parser.add_argument("--test", action="store_true", help="Process first 5 lines only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.pipeline.dataset_name = args.dataset

    run_llm = args.step in ("llm", "all")
    run_retrieve = args.step in ("retrieve", "all") and cfg.pipeline.use_retriever

    llm = create_llm_client(cfg.llm) if run_llm else None
    retriever = None
    if run_retrieve:
        retriever = HybridRetriever(cfg.retriever)
        retriever.load_index()

    gen = DocumentGenerator(llm=llm, retriever=retriever, pipeline_cfg=cfg.pipeline)

    data = read_jsonl(args.input)
    if args.test:
        data = data[:5]

    results = []
    for i, line in enumerate(data):
        print(f"[doc_generate] {i + 1}/{len(data)}", end="\r")
        if run_llm:
            line = gen.generate_llm_reference(
                line,
                have_choice=args.have_choice,
                add_entity=args.add_entity,
            )
        if run_retrieve:
            line = gen.retrieve_with_entities(line)

        # Build candidate_passages as List[Dict] (handoff format for downstream scripts)
        candidates = []
        if line.get("pseudo_doc"):
            candidates.append({"text": line["pseudo_doc"], "source": "llm"})
        for doc in line.get("retrieved_passages", []):
            candidates.append({"text": doc.get("text", ""), "source": "retriever",
                                "score": doc.get("score", 0.0)})
        if line.get("pseudo_doc_entity"):
            candidates.append({"text": line["pseudo_doc_entity"], "source": "kg"})
        line["candidate_passages"] = [c for c in candidates if c["text"]]

        results.append(line)

    print()
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
