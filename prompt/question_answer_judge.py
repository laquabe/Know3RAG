"""
Judge prompt (multi-turn answer reconciliation).
Source: prompt_fomular_judge() in code/main.py (lines 262-281).
"""


def build_prompt(line: dict, answer_key_list: list, have_choice: bool = False) -> str:
    """
    Prompt for judging multiple answers to a question and selecting the best one.
    Returns a Llama-template string (system + user turns, no assistant header).
    Response should end with "The best answer is [A/B/C/D]".
    """
    system_prompt = "You will act as a judge to evaluate answers provided by multiple sources for a given question."
    user_0 = (
        "I will provide you with the question and some respective answers. Your task is to "
        "analyze their reasoning, compare their validity, and provide a final, well-reasoned "
        "answer based on the evidence and logic presented.\n"
    )
    if have_choice:
        user_0 += "\nQuestion: {}\nA. {}\nB. {}\nC. {}\nD. {}\n".format(
            line.get('Question', ''),
            line.get('A', ''), line.get('B', ''), line.get('C', ''), line.get('D', '')
        )
    else:
        user_0 += "\nQuestion: {}\n".format(line.get('Question', ''))

    for idx, ans_key in enumerate(answer_key_list):
        user_0 += "Answer {}: {}\n".format(idx + 1, line.get(ans_key, ''))

    user_0 += (
        '\nYour response should end with "The best answer is [the_answer_letter]" '
        "where the [the_answer_letter] is one of A, B, C or D.\n"
    )

    content = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>".format(system_prompt)
    content += "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>".format(user_0)
    return content
