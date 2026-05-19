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
    cot_messages: List[Dict] = None,
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
        # Open-ended: optional few-shot demos + current user turn
        messages = list(cot_messages or [])
        messages.append({
            "role": "user",
            "content": build_open_reference_user_prompt(line, add_entity=add_entity),
        })
        return messages


def build_open_reference_user_prompt(line: dict, add_entity: bool = False) -> str:
    """Build the open-ended reference-generation user prompt for one sample."""
    user_prompt = (
        "I have a list of open-ended questions, and I'd like you to write a reference "
        "paragraph for each question. These paragraphs should provide sufficient "
        "background, key concepts, or context to guide the next person in answering "
        "the question effectively. You do not need to provide an answer directly, just "
        "enough information to help the next person frame their answer concisely and "
        "accurately.\n"
    )
    if add_entity and line.get('query_entity'):
        user_prompt += (
            "To make your reference passages more accurate, I'm going to provide "
            "you with some entities inside the question that you can refer to them, "
            "but they're not necessarily accurate.\n"
        )
    user_prompt += "Question: {}\n".format(line.get('question', line.get('Question', '')))
    if add_entity and line.get('query_entity'):
        user_prompt += "\nRelated Entities:\n"
        for i, ent in enumerate(line['query_entity'].values()):
            user_prompt += "{}. {}: {}\n".format(
                i + 1, ent.get('entity', ''), ent.get('description', '')
            )
    user_prompt += (
        'You should just output "[reference_paragraph]", '
        "where the [reference_paragraph] is the reference you write.\n"
    )
    return user_prompt


def build_open_reference_cot_messages(
    examples: List[Dict],
    add_entity: bool = False,
) -> List[Dict]:
    """Build few-shot messages from open-QA reference-generation JSONL examples."""
    system_prompt = (
        "You are an intelligent assistant specialized in generating reference "
        "paragraphs for open-ended questions. Your task is to provide clear and "
        "concise reference paragraphs that contextualize the question and guide "
        "the answerer in understanding the context, key concepts, and relevant "
        "details. These paragraphs are meant to provide sufficient information "
        "for the next person to answer the question accurately and thoroughly.\n"
        "Hints:\n"
        "1. Provide background information that is relevant to the question.\n"
        "2. Clarify any key terms or concepts that might be important for answering the question.\n"
        "3. Provide context such as important dates, figures, or events if applicable.\n"
        "4. Keep the paragraph concise but detailed enough to guide the next person in framing an answer.\n"
        "5. If there are provided entities and the entities mentioned in the question are accurate, "
        "ensure they are consistent with your reference. If the entities are inaccurate, you may disregard them."
    )
    messages: List[Dict] = [{"role": "system", "content": system_prompt}]
    for ex in examples:
        pseudo_doc = str(ex.get('query_pseudo_doc', '')).strip()
        if not pseudo_doc:
            continue
        messages.append({
            "role": "user",
            "content": build_open_reference_user_prompt(ex, add_entity=add_entity),
        })
        messages.append({"role": "assistant", "content": pseudo_doc})
    return messages
