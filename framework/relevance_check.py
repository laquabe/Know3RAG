"""
Relevance check framework module.
Combines LLM reliability filtering and KGE-based passage ranking.
"""
import os
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PipelineConfig
from utils import BaseLLMClient, KGEScorer, local_check_str, score_feature
import prompt.relevance_check_kg_local_check as local_check_mod


class RelevanceChecker:
    """
    Filters and ranks candidate passages by:
      1. LLM reliability check (yes/no per passage)
      2. KGE triple-score ranking
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        kge_scorer: KGEScorer,
        pipeline_cfg: PipelineConfig,
    ):
        self.llm = llm
        self.kge = kge_scorer
        self.cfg = pipeline_cfg

    # ------------------------------------------------------------------
    # LLM-based passage reliability check
    # ------------------------------------------------------------------

    def llm_check_passage(
        self,
        line: dict,
        check_key: str = 'passages',
        have_choice: bool = False,
        output_key: str = 'local_check',
    ) -> dict:
        """
        Ask the LLM whether the passage at *check_key* is reliable for the
        question.  Writes a bool to ``line[output_key]``.
        """
        messages = local_check_mod.build_prompt(
            line, have_choice=have_choice, check_key=check_key
        )
        response = self.llm.call(messages)
        line[output_key] = local_check_str(response)
        return line

    def llm_check_passage_batch(
        self,
        lines: List[dict],
        check_key: str = 'passages',
        have_choice: bool = False,
        output_key: str = 'local_check',
    ) -> List[dict]:
        """Batch version of llm_check_passage()."""
        batch_messages = [
            local_check_mod.build_prompt(line, have_choice=have_choice, check_key=check_key)
            for line in lines
        ]
        responses = self.llm.call_batch(batch_messages)
        for line, resp in zip(lines, responses):
            line[output_key] = local_check_str(resp)
        return lines

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def filter_by_llm_check(passages_with_checks: List[dict]) -> List[dict]:
        """
        Return only entries where ``entry['local_check']`` is True.
        Each entry is expected to have at least 'text' and 'local_check'.
        """
        return [p for p in passages_with_checks if p.get('local_check')]

    def rank_by_kge_score(
        self,
        passages_with_scores: List[dict],
        top_k: Optional[int] = None,
        entity_count: bool = True,
    ) -> List[str]:
        """
        Rank passages by their pre-computed factual score (lower = more
        consistent with KG).  Returns a list of passage texts, best first.

        Each entry in *passages_with_scores* should have:
          - 'text': str
          - 'factual_score': float or None  (None → treated as worst)
        """
        k = top_k or self.cfg.top_k_references

        def sort_key(p):
            s = p.get('factual_score')
            return float('inf') if s is None else s

        ranked = sorted(passages_with_scores, key=sort_key)
        return [p['text'] for p in ranked[:k]]

    # ------------------------------------------------------------------
    # Full reference selection
    # ------------------------------------------------------------------

    def select_references(
        self,
        line: dict,
        candidate_passages: List[dict],
        top_k: Optional[int] = None,
        have_choice: bool = False,
        use_llm_check: bool = True,
        use_kge_rank: bool = True,
        output_key: str = 'reference',
    ) -> dict:
        """
        Select the final reference passages from *candidate_passages*.

        Each candidate should be a dict with at minimum a 'text' key.
        Optionally runs LLM check and/or KGE ranking before selecting top_k.
        Writes a list of passage strings to ``line[output_key]``.
        """
        k = top_k or self.cfg.top_k_references

        if use_llm_check and candidate_passages:
            # Run LLM check on each candidate
            for cand in candidate_passages:
                tmp_line = dict(line)
                tmp_line['passages'] = cand.get('text', '')
                tmp_line = self.llm_check_passage(
                    tmp_line, check_key='passages', have_choice=have_choice
                )
                cand['local_check'] = tmp_line.get('local_check', False)
            candidate_passages = self.filter_by_llm_check(candidate_passages)

        if use_kge_rank and candidate_passages:
            texts = self.rank_by_kge_score(candidate_passages, top_k=k)
        else:
            texts = [p['text'] for p in candidate_passages[:k]]

        line[output_key] = texts
        return line


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from config import load_config
    from utils import create_llm_client, KGEScorer, read_jsonl, write_jsonl

    parser = argparse.ArgumentParser(description="Know3RAG relevance check stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--step", choices=["llm", "rank", "all"], default="all",
        help="llm: LLM passage check; rank: filter+select using pre-computed scores; all: both"
    )
    parser.add_argument("--have-choice", action="store_true", help="MMLU mode")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Process first 5 lines only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.pipeline.dataset_name = args.dataset

    run_llm = args.step in ("llm", "all")
    run_rank = args.step in ("rank", "all")
    top_k = args.top_k or cfg.pipeline.top_k_references

    # KGEScorer is NOT needed here — factual scores are pre-computed in candidate_passages
    llm = create_llm_client(cfg.llm) if run_llm else None
    checker = RelevanceChecker(llm=llm, kge_scorer=None, pipeline_cfg=cfg.pipeline)

    data = read_jsonl(args.input)
    if args.test:
        data = data[:5]

    results = []
    for i, line in enumerate(data):
        print(f"[relevance_check --step {args.step}] {i + 1}/{len(data)}", end="\r")
        candidates = line.get("candidate_passages", [])

        if run_llm:
            for cand in candidates:
                tmp = {**line, "passages": cand.get("text", "")}
                tmp = checker.llm_check_passage(
                    tmp, check_key="passages", have_choice=args.have_choice
                )
                cand["local_check"] = tmp.get("local_check", False)

        if run_rank:
            passed = [c for c in candidates if c.get("local_check", True)]
            ranked = sorted(
                passed,
                key=lambda c: c.get("factual_score") if c.get("factual_score") is not None
                              else float("inf")
            )
            line["reference"] = [c["text"] for c in ranked[:top_k]]

        line["candidate_passages"] = candidates
        results.append(line)

    print()
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
