"""
Question decomposition prompt.
Source: prompt_fomular_decompose_question() in code/main.py (lines 181-190).
"""


def build_prompt(line: dict) -> str:
    """
    Prompt for breaking a question into entity-specific sub-questions.
    Returns a plain string (not a messages list) — caller wraps in user turn.
    Response should be a JSON dict: {entity: sub_question, ...}.
    """
    content = (
        "I have a problem that I need to break down into sub-problems. please extract "
        "the key entities from the problem and come up with one piece of information that "
        "needs to be collected for each entity in JSON format. You don't need to answer "
        "these questions; just identify what information should be gathered.\n"
    )
    content += "To give you a clearer idea, here's an example problem and how it should be broken down:\n\n"
    content += "**Example Problem:**\nWho was the current President of the United States when Zootopia was released?\n"
    content += "**Entities and Information Needed:**\n"
    content += '{"President of the United States": "Who was the President of the United States?", "Zootopia": "When was Zootopia released?"}\n\n'
    content += "Here's the problem I need your help with:\n"
    content += "**Problem:**\n{}".format(line.get('Question', ''))
    return content
