"""
Question answering framework module.
Wraps LLM calls for answering questions and judging multi-turn responses.
"""
from typing import Dict, List, Optional

from config import LLMConfig, PipelineConfig
from utils import BaseLLMClient, answer_phrase
import prompt.question_answer_qa as qa_prompt_mod
import prompt.question_answer_judge as judge_prompt_mod


class QuestionAnswerer:
    """
    Generates answers to questions (single and multi-turn) and
    reconciles multiple answers via an LLM judge.
    """

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
        answer_key: str = 'llm_response',
    ) -> dict:
        """
        Generate an answer for the question in *line* and store it under
        *answer_key*.  Works for hotpotQA, 2wikimultihopQA, PopQA, MMLU,
        Temporal_QA.
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
        return line

    def answer_batch(
        self,
        lines: List[dict],
        cot_prompt: str = None,
        output_reason: bool = True,
        add_ref: bool = True,
        answer_key: str = 'llm_response',
    ) -> List[dict]:
        """Batch version of answer()."""
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
        return lines

    def judge(
        self,
        line: dict,
        answer_key_list: List[str],
        have_choice: bool = False,
        output_key: str = 'judge_response',
    ) -> dict:
        """
        Judge multiple answers stored under *answer_key_list* keys and pick
        the best one.  Result stored under *output_key*.
        Uses a Llama-template string — caller's LLM must handle raw strings.
        """
        prompt_str = judge_prompt_mod.build_prompt(
            line, answer_key_list=answer_key_list, have_choice=have_choice
        )
        # Judge prompt is a raw Llama template string, not a messages list.
        # Wrap in a single user message so BaseLLMClient can handle it.
        response = self.llm.call(prompt_str)
        line[output_key] = response
        return line

    @staticmethod
    def extract_answer(line: dict, answer_key: str = 'llm_response'):
        """Parse the LLM response to extract a structured answer / choice."""
        pred, success = answer_phrase(line.get(answer_key, ''))
        return pred, success


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from config import load_config
    from utils import create_llm_client, read_jsonl, write_jsonl

    parser = argparse.ArgumentParser(description="Know3RAG QA / judge stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--step", choices=["answer", "judge"], default="answer",
        help="answer: generate answer; judge: select best from multi-turn answers"
    )
    parser.add_argument("--answer-key", default="llm_response",
                        help="Output key for --step answer (e.g. llm_response_0)")
    parser.add_argument("--answer-keys", nargs="+", default=["llm_response_0", "llm_response_1"],
                        help="Input answer keys for --step judge")
    parser.add_argument("--have-choice", action="store_true", help="MMLU mode")
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
        print(f"[question_answer --step {args.step}] {i + 1}/{len(data)}", end="\r")
        if args.step == "answer":
            line = qa.answer(
                line,
                add_ref=not args.no_ref,
                output_reason=not args.no_reason,
                answer_key=args.answer_key,
            )
        else:  # judge
            line = qa.judge(
                line,
                answer_key_list=args.answer_keys,
                have_choice=args.have_choice,
            )
        results.append(line)

    print()
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
