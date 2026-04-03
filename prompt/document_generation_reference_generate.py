"""
Reference generation prompt.
Source: prompt_fomular_reference_generate() in code/main.py (lines 126-179).
"""
from typing import Dict, List, Union


def build_prompt(
    line: dict,
    have_choice: bool = False,
    add_entity: bool = False,
    cot_prompt: str = None,
) -> Union[List[Dict], str]:
    """
    Prompt for generating a reference paragraph for a question.

    - For MMLU (have_choice=True): returns a Llama-template string
      (1-shot CoT + actual question).
    - For open-ended (have_choice=False): returns an OpenAI messages list
      (single user turn, no shot).
    """
    if have_choice:
        # Build or reuse the CoT example
        if cot_prompt is None:
            user_0 = (
                "I have a list of multiple-choice questions, and I'd like you to write a "
                "reference paragraph for each question. These paragraphs will assist the "
                "person coming after me in understanding the context of the question and "
                "choices, enabling them to amplify and answer the questions concisely. "
                "You don't need to answer the questions directly, just provide enough "
                "information to guide the next person.\n"
            )
            if add_entity and line.get('query_entity'):
                user_0 += (
                    "To make your reference passages more accurate, I'm going to provide "
                    "you with some entities inside the question that you can refer to them, "
                    "but they're not necessarily accurate.\n"
                )
            user_0 += "\nQuestion: Which city is the capital of France?\n"
            user_0 += "A. Paris\nB. London\nC. Berlin\nD. Madrid\n"
            if add_entity and line.get('query_entity'):
                user_0 += "\nRelated Entities:\n"
                user_0 += "1. France: country in Western Europe\n"
            user_0 += (
                'Your response should start with "Reference: [reference_paragraph]" '
                "where the [reference_paragraph] is the reference you write.\n"
            )
            assist_0 = (
                "Reference: The capital of France is Paris. Paris, known for its historical "
                "landmarks such as the Eiffel Tower and the Louvre Museum, is located in the "
                "northern part of the country along the Seine River. It is a major European "
                "city and a global center for art, fashion, and culture."
            )
            cot_block = (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "{}<|eot_id|>\n"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
                "{}<|eot_id|>"
            ).format(user_0, assist_0)
        else:
            cot_block = cot_prompt

        user_1 = (
            "I have a list of multiple-choice questions, and I'd like you to write a "
            "reference paragraph for each question. These paragraphs will assist the "
            "person coming after me in understanding the context of the question and "
            "choices, enabling them to amplify and answer the questions concisely. "
            "You don't need to answer the questions directly, just provide enough "
            "information to guide the next person.\n"
        )
        if add_entity and line.get('query_entity'):
            user_1 += (
                "To make your reference passages more accurate, I'm going to provide "
                "you with some entities inside the question that you can refer to them, "
                "but they're not necessarily accurate.\n"
            )
        user_1 += "Question: {}\n".format(line.get('Question', ''))
        user_1 += "A. {}\nB. {}\nC. {}\nD. {}\n".format(
            line.get('A', ''), line.get('B', ''), line.get('C', ''), line.get('D', '')
        )
        if add_entity and line.get('query_entity'):
            user_1 += "\nRelated Entities:\n"
            for i, ent in enumerate(line['query_entity'].values()):
                user_1 += "{}. {}: {}\n".format(i + 1, ent['entity'], ent['description'])
        user_1 += (
            'Your response should start with "Reference: [reference_paragraph]" '
            "where the [reference_paragraph] is the reference you write.\n"
        )
        assist_1 = "Reference:"
        question_block = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{}"
        ).format(user_1, assist_1)

        return cot_block + question_block

    else:
        # Open-ended: single user turn, no shot
        user_prompt = (
            "I have a list of open-ended questions, and I'd like you to write a reference "
            "paragraph for each question. These paragraphs should provide sufficient "
            "background, key concepts, or context to guide the next person in answering "
            "the question effectively. You do not need to provide an answer directly, just "
            "enough information to help the next person frame their answer concisely and "
            "accurately.\n"
        )
        if line.get('query_entity'):
            user_prompt += (
                "To make your reference passages more accurate, I'm going to provide "
                "you with some entities inside the question that you can refer to them, "
                "but they're not necessarily accurate.\n"
            )
        user_prompt += "Question: {}\n".format(line.get('question', ''))
        if line.get('query_entity'):
            user_prompt += "\nRelated Entities:\n"
            for i, ent in enumerate(line['query_entity'].values()):
                user_prompt += "{}. {}: {}\n".format(i + 1, ent['entity'], ent['description'])
        user_prompt += (
            'You should just output "[reference_paragraph]", '
            "where the [reference_paragraph] is the reference you write.\n"
        )
        return [{"role": "user", "content": user_prompt}]
