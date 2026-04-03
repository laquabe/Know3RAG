"""
Prompt package — re-exports each module's build_prompt() under a unique alias.
"""
from prompt.factual_check_triple_extraction import build_prompt as triple_extraction_prompt
from prompt.relevance_check_kg_local_check import build_prompt as kg_local_check_prompt
from prompt.document_generation_reference_generate import build_prompt as reference_generate_prompt
from prompt.query_enhance_decompose_question import build_prompt as decompose_question_prompt
from prompt.query_enhance_selfask import build_prompt as selfask_prompt
from prompt.question_answer_judge import build_prompt as judge_prompt
from prompt.query_enhance_generate_question import build_prompt as generate_question_prompt
from prompt.question_answer_qa import build_prompt as qa_prompt

__all__ = [
    "triple_extraction_prompt",
    "kg_local_check_prompt",
    "reference_generate_prompt",
    "decompose_question_prompt",
    "selfask_prompt",
    "judge_prompt",
    "generate_question_prompt",
    "qa_prompt",
]
