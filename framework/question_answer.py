"""
Question answering framework module.
Wraps LLM calls for answering questions.
"""
import os
import sys
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PipelineConfig
from utils import BaseLLMClient, extract_answer_by_dataset
import prompt.question_answer_qa as qa_prompt_mod


class QuestionAnswerer:
    """
    Generates answers to questions (single and multi-turn).
    """

    DEFAULT_ANSWER_KEY = 'llm_response'

    def __init__(
        self,
        llm: BaseLLMClient,
        pipeline_cfg: PipelineConfig,
    ):
        self.llm = llm
        self.cfg = pipeline_cfg

    def answer(
        self,
        line: dict,
        cot_prompt: str = None,
        output_reason: bool = True,
        add_ref: bool = True,
        answer_key: str = DEFAULT_ANSWER_KEY,
    ) -> dict:
        """
        Generate an answer for the question in *line* and store it under
        *answer_key*.  Works for hotpotQA, 2wikimultihopQA, PopQA, MMLU,
        Temporal_QA.  Also extracts the parsed answer into line['answer'].
        """
        messages = qa_prompt_mod.build_prompt(
            line,
            dataset=self.cfg.dataset_name,
            cot_prompt=cot_prompt,
            output_reason=output_reason,
            add_ref=add_ref,
        )
        response = self.llm.call(messages)
        line[answer_key] = response
        pred, _ = self.extract_answer(line, answer_key=answer_key)
        line['answer'] = pred
        return line

    def answer_batch(
        self,
        lines: List[dict],
        cot_prompt: str = None,
        output_reason: bool = True,
        add_ref: bool = True,
        answer_key: str = DEFAULT_ANSWER_KEY,
    ) -> List[dict]:
        """Batch version of answer(). Also extracts parsed answers into line['answer']."""
        batch_messages = [
            qa_prompt_mod.build_prompt(
                line,
                dataset=self.cfg.dataset_name,
                cot_prompt=cot_prompt,
                output_reason=output_reason,
                add_ref=add_ref,
            )
            for line in lines
        ]
        responses = self.llm.call_batch(batch_messages)
        for line, resp in zip(lines, responses):
            line[answer_key] = resp
            pred, _ = self.extract_answer(line, answer_key=answer_key)
            line['answer'] = pred
        return lines

    def extract_answer(self, line: dict, answer_key: str = DEFAULT_ANSWER_KEY):
        """Parse the LLM response into a dataset-appropriate answer format."""
        pred, success = extract_answer_by_dataset(
            line.get(answer_key, ''),
            dataset=self.cfg.dataset_name,
        )
        return pred, success


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from config import load_config
    from utils import create_llm_client, read_jsonl, write_jsonl

    parser = argparse.ArgumentParser(description="Know3RAG QA stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--answer-key", default=QuestionAnswerer.DEFAULT_ANSWER_KEY,
                        help="Output key for generated answers")
    parser.add_argument("--no-ref", action="store_true",
                        help="Answer without references (baseline)")
    parser.add_argument("--no-reason", action="store_true",
                        help="Skip chain-of-thought reasoning in answer")
    parser.add_argument("--test", action="store_true", help="Process first 5 lines only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.pipeline.dataset_name = args.dataset

    llm = create_llm_client(cfg.llm)
    qa = QuestionAnswerer(llm=llm, pipeline_cfg=cfg.pipeline)

    data = read_jsonl(args.input)
    if args.test:
        data = data[:5]

    results = []
    for i, line in enumerate(data):
        print(f"[question_answer] {i + 1}/{len(data)}", end="\r")
        line = qa.answer(
            line,
            add_ref=not args.no_ref,
            output_reason=not args.no_reason,
            answer_key=args.answer_key,
        )
        results.append(line)

    print()
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
