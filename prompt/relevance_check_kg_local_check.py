"""
KG local reliability check prompt.
Source: prompt_fomular_kg_local_check() in code/main.py (lines 49-124).
"""
from typing import Dict, List


def build_prompt(
    line: dict,
    have_choice: bool = False,
    check_key: str = 'passages',
) -> List[Dict]:
    """
    4-shot prompt for assessing whether a passage is reliable for answering a question.
    Returns an OpenAI messages list.
    Response should end with "The reliability of the passage is [yes or no]."
    """
    system_prompt = (
        'I need your help determining the reliability of a passage in the context of its ability to answer '
        'a specific question. I might provide some entities to help you better understand the problem. '
        'Here are the key considerations:\n'
        '1. Passage Relevance: Check if the passage provides information relevant to answering the question. '
        'Even if the passage does not directly mention the entities, it should address the key concepts or '
        'ideas related to the question. If the passage does not contribute meaningfully to answering the '
        'question, it may be unreliable.\n'
        '2. Entity Accuracy: The entities provided are from the question and may not appear in the passage. '
        'These entities are meant to help you understand the context of the question. If the passage conflicts '
        'with the entities provided (e.g., incorrect descriptions or relationships), this could affect its '
        'reliability.\n'
        '3. Overall Reliability: Based on the relevance of the passage to the question and the accuracy of '
        'the entities, assess whether the passage is reliable for answering the question. If there are doubts '
        'or inconsistencies, provide a clear explanation.\n'
    )

    def _question_block(q_text: str, choice: bool) -> str:
        if choice:
            return 'Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\n'.format(
                q_text,
                line.get('A', ''), line.get('B', ''), line.get('C', ''), line.get('D', '')
            )
        return 'Question: {}\n'.format(q_text)

    instruction = (
        'Confirm that the article is reliable for the question. Provide your reasoning for the reliability '
        'decision, and end your response with: "The reliability of the passage is [yes or no]."\n'
    )

    # Shot 0: apple nutrition — yes
    u0 = instruction
    if have_choice:
        u0 += 'Question: What is the nutritional value of an apple?\nA. High in fiber and vitamins.\nB Low in calories but high in protein.\nC. Rich in fats.D. No nutritional value\n'
    else:
        u0 += 'Question: What is the nutritional value of an apple?\n'
    u0 += 'Entities:\n1. Apple: A fruit known for its nutritional benefits, such as fiber and vitamins.\n'
    u0 += 'Passage: An apple is a nutritious fruit rich in fiber, vitamins, and antioxidants.\n'
    a0 = ('The passage provides relevant information about the nutritional value of an apple, aligning with '
          'the question. The entity "Apple" refers to the fruit, which matches the context of the question. '
          'The reliability of the passage is yes.')

    # Shot 1: Apple CEO vs fruit passage — no
    u1 = instruction
    if have_choice:
        u1 += 'Question: What is the CEO of Apple Inc.?\nA. Tim Cook\nB. Steve Jobs\nC. Elon Musk\nD. Satya Nadella\n'
    else:
        u1 += 'Question: What is the CEO of Apple Inc.?\n'
    u1 += 'Entities:\n1. Apple Inc.: A technology company, known for products like the iPhone and Mac computers.\n'
    u1 += 'Passage: Apples are widely consumed fruits that come in different varieties, including Granny Smith and Red Delicious.\n'
    a1 = ('The passage discusses apples as a fruit, which is unrelated to the question about the CEO of '
          'Apple Inc. The passage does not address the company or its leadership. '
          'The reliability of the passage is no.')

    # Shot 2: Apple CEO, entity is wrong fruit — no
    u2 = instruction
    if have_choice:
        u2 += 'Question: What is the CEO of Apple Inc.?\nA. Tim Cook\nB. Steve Jobs\nC. Elon Musk\nD. Satya Nadella\n'
    else:
        u2 += 'Question: What is the CEO of Apple Inc.?\n'
    u2 += 'Entities:\n1. Apple: A fruit known for its sweet taste and variety of colors, including red, green, and yellow.\n'
    u2 += 'Passage: Apples are a popular fruit consumed worldwide. They are rich in fiber and vitamins, often enjoyed raw or in various dishes.\n'
    a2 = ('The passage discusses the fruit *Apple*, which is unrelated to the question about the CEO of '
          '*Apple Inc.*, a technology company. The entity provided incorrectly refers to the fruit, not the '
          'tech company. This makes the passage unreliable for answering the question. '
          'The reliability of the passage is no.')

    # Shot 3: Apple CEO, entity is fruit but passage answers correctly — yes
    u3 = instruction
    if have_choice:
        u3 += 'Question: What is the CEO of Apple Inc.?\nA. Tim Cook\nB. Steve Jobs\nC. Elon Musk\nD. Satya Nadella\n'
    else:
        u3 += 'Question: What is the CEO of Apple Inc.?\n'
    u3 += 'Entities:\n1. Apple: A fruit known for its sweet taste and variety of colors, including red, green, and yellow.\n'
    u3 += 'Passage: Tim Cook is the current CEO of Apple Inc., a technology company known for its products such as the iPhone and Mac computers.\n'
    a3 = ('Despite the provided entity referring to the fruit *Apple*, the passage directly answers the '
          'question by stating that Tim Cook is the CEO of *Apple Inc.* The passage is relevant to the '
          'question, and the reliability is unaffected by the unrelated entity description. '
          'The reliability of the passage is yes.')

    # Actual query
    u4 = instruction
    if have_choice:
        u4 += 'Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\n'.format(
            line.get('Question', ''), line.get('A', ''), line.get('B', ''),
            line.get('C', ''), line.get('D', '')
        )
    else:
        u4 += 'Question: {}\n'.format(line.get('question', ''))

    query_entities = line.get('query_entity', {})
    if query_entities:
        u4 += 'Entities:\n'
        for i, (_, ent) in enumerate(query_entities.items()):
            u4 += '{}. {}: {}\n'.format(i + 1, ent.get('entity', ''), ent.get('description', ''))
    u4 += 'Passage: {}\n'.format(line.get(check_key, ''))

    return [
        {'role': 'system',    'content': system_prompt},
        {'role': 'user',      'content': u0},
        {'role': 'assistant', 'content': a0},
        {'role': 'user',      'content': u1},
        {'role': 'assistant', 'content': a1},
        {'role': 'user',      'content': u2},
        {'role': 'assistant', 'content': a2},
        {'role': 'user',      'content': u3},
        {'role': 'assistant', 'content': a3},
        {'role': 'user',      'content': u4},
    ]
