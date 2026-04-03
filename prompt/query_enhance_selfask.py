"""
Self-ask prompt.
Source: prompt_fomular_selfask() in code/main.py (lines 200-222).
"""


def build_prompt(line: dict, have_choice: bool = False) -> str:
    """
    Prompt asking whether more information is needed before answering.
    Returns a Llama-template string (user turn only, no assistant header).
    Response should start with 'yes' or 'no'.
    """
    if line.get('reference'):
        user_0 = (
            "Given the following references and question, you should first assess if you have "
            "enough information to answer. If you feel your own knowledge and the provided "
            "references are insufficient, respond with 'yes'. If you believe you can answer "
            "based on the current data, respond with 'no'.\n"
        )
        for ref_id, ref in enumerate(line['reference']):
            user_0 += "Reference {}: {}\n".format(ref_id + 1, ref)
    else:
        user_0 = (
            "Given the following question, you should first assess if you have enough "
            "information to answer. If you feel your own knowledge is insufficient, respond "
            "with 'yes'. If you believe you can answer, respond with 'no'.\n"
        )

    if have_choice:
        user_0 += "\nQuestion: {}\nA. {}\nB. {}\nC. {}\nD. {}\n".format(
            line.get('Question', ''),
            line.get('A', ''), line.get('B', ''), line.get('C', ''), line.get('D', '')
        )
    else:
        user_0 += "\nQuestion: {}\n".format(line.get('Question', ''))

    user_0 += "\nDo you need more information?\n"

    return "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>".format(user_0)
