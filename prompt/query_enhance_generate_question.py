"""
Follow-up question generation prompt.
Source: prompt_generate_question() in code/main.py (lines 302-316).
"""
from typing import Dict, List


def build_prompt(line: dict) -> List[Dict]:
    """
    Prompt for generating a new follow-up question based on references and the
    current (possibly incomplete) answer.
    Returns an OpenAI messages list (single user turn).
    """
    user_1 = (
        "You will be given references, a question, and an answer. The answer may be incomplete "
        "or incorrect. Identify the most critical missing or incorrect information in the "
        "references and the answer. Formulate one most important new question that will most "
        "effectively help retrieve the necessary information to answer the original question.\n"
    )

    if line.get('reference'):
        for ref_id, ref in enumerate(line['reference']):
            user_1 += "Reference {}: {}\n".format(ref_id + 1, ref)
    else:
        user_1 += "Reference: No reference available.\n"

    user_1 += "Question: {}\n".format(line.get('question', ''))
    user_1 += "Answer: {}\n".format(line.get('llm_response', ''))
    user_1 += "Please directly output the new question:"

    return [{"role": "user", "content": user_1}]
