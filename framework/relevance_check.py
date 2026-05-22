"""
Relevance check framework module.

Runs an LLM-based field-level relevance / reliability check: given a question
and a passage-like field, decide whether that field is useful and reliable for
answering the question.
"""
import os
import sys
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PipelineConfig
from utils import BaseLLMClient, local_check_str
import prompt.relevance_check_kg_local_check as local_check_mod


class RelevanceChecker:
    """
    Checks whether a passage-like field is relevant and reliable for answering
    the record's question.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        pipeline_cfg: PipelineConfig,
    ):
        self.llm = llm
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

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from config import load_config
    from utils import create_llm_client, read_jsonl, write_jsonl

    parser = argparse.ArgumentParser(description="Know3RAG relevance check stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--question-key", default="question", help="Input field containing the question")
    parser.add_argument("--check-key", default="passages", help="Input field to check for relevance/reliability")
    parser.add_argument("--output-key", default="local_check", help="Output boolean field for the check result")
    parser.add_argument("--have-choice", action="store_true", help="MMLU mode")
    parser.add_argument("--test", action="store_true", help="Process first 5 lines only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.pipeline.dataset_name = args.dataset

    llm = create_llm_client(cfg.llm)
    checker = RelevanceChecker(llm=llm, pipeline_cfg=cfg.pipeline)

    data = list(read_jsonl(args.input))
    if args.test:
        data = data[:5]

    results = []
    for i, line in enumerate(data):
        print(f"[relevance_check --check-key {args.check_key}] {i + 1}/{len(data)}", end="\r")

        # The prompt module expects open-domain questions under ``question`` and
        # MMLU questions under ``Question``.  Keep input schemas flexible by
        # copying the configured question field into the expected key.
        if args.have_choice:
            if args.question_key != "Question" and args.question_key in line:
                line["Question"] = line[args.question_key]
        else:
            if args.question_key != "question" and args.question_key in line:
                line["question"] = line[args.question_key]

        line = checker.llm_check_passage(
            line,
            check_key=args.check_key,
            have_choice=args.have_choice,
            output_key=args.output_key,
        )
        results.append(line)

    print()
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
