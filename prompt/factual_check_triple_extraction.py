"""
Triple extraction prompt.
Source: prompt_fomular_triple_extraction() in code/main.py (lines 13-47).
"""
from typing import Dict, List


def build_prompt(line: dict, src_key: str = 'passages', ent_key: str = 'passage_entity') -> List[Dict]:
    """
    2-shot prompt for extracting (subject, predicate, object) triples from text.
    Returns an OpenAI messages list.
    """
    system_prompt = (
        'I have a task to extract relationships between entities from a given text. '
        'You will be provided with a list of possible entities as a reference. Your task is to extract triples '
        'representing relationships between entities in the text. Each triple should include a subject, predicate, '
        'and object directly from the text. '
        'The entities in the triples may or may not match the provided reference list, but use the list to guide '
        'your extraction process.'
        'If no meaningful relationships are found, return None.\n'
        'Instructions:\n'
        '- Extract the subject, predicate, and object exactly as they appear in the text.\n'
        '- If no valid relationships are found, return None.\n'
        '- Output only the extracted triples in the format: [list of triples].\n'
    )

    # Shot 1: positive example
    user_0 = (
        'Text: Albert Einstein was born in Ulm, Germany in 1879.\n'
        'Entities: Albert Einstein, Ulm, Germany\n'
    )
    assist_0 = (
        '[{"subject": "Albert Einstein", "predicate": "was born in", "object": "Ulm"}, '
        '{"subject": "Albert Einstein", "predicate": "was born in", "object": "Germany"}]'
    )

    # Shot 2: negative example (pronoun → None)
    user_1 = (
        'Text: She is a member of the organization.\n'
        'Entities: the organization\n'
    )
    assist_1 = 'None'

    # Actual query
    entity_str = ', '.join(line.get(ent_key, {}).keys())
    user_2 = 'Text: {}\nEntities: {}\n'.format(line[src_key], entity_str)

    return [
        {'role': 'system',    'content': system_prompt},
        {'role': 'user',      'content': user_0},
        {'role': 'assistant', 'content': assist_0},
        {'role': 'user',      'content': user_1},
        {'role': 'assistant', 'content': assist_1},
        {'role': 'user',      'content': user_2},
    ]
