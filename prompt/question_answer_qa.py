"""
Main QA prompt.
Source: prompt_fomular() in code/main.py (lines 319-391).
"""
from typing import Dict, List


def build_prompt(
    line: dict,
    dataset: str,
    cot_prompt: str = None,
    output_reason: bool = True,
    add_ref: bool = True,
) -> List[Dict]:
    """
    Build the main question-answering prompt.

    dataset options: 'hotpotQA', '2wikimultihopQA', 'PopQA', 'MMLU', 'Temporal_QA'

    Always returns OpenAI messages list format: [{"role": "user", "content": str}].
    For MMLU, content includes Llama3 template tokens; cot_prompt defaults to '' if None.
    """
    if dataset in ['hotpotQA', '2wikimultihopQA', 'PopQA']:
        if add_ref:
            user_prompt = (
                "Given the following question, references (may or may not be available), "
                "explain your reasoning step-by-step based on the references and then provide "
                "your best possible answer. If there is no reference or you find the reference "
                "irrelevant, please provide an answer based on your knowledge.\n\n"
            )
            if line.get('reference'):
                for ref_id, ref in enumerate(line['reference']):
                    user_prompt += "Reference {}: {}\n".format(ref_id + 1, ref)
            else:
                user_prompt += "Reference: No reference available.\n"
            user_prompt += "\nQuestion: {}\n".format(line.get('question', ''))
            user_prompt += (
                "\nYour response should end with \"The answer is [your_answer_text]\", "
                "where the [your_answer_text] should be yes, no, or a few words directly "
                "answering the question.\n Let's think step by step."
            )
            return [{"role": "user", "content": user_prompt}]

        elif output_reason:
            user_prompt = (
                "Given the following open-ended question, explain your reasoning step-by-step "
                "and then provide your final answer.\n"
                "Question: {}\nYour response should end with \"The answer is [your_answer_text]\". "
                "Let's think step by step."
            ).format(line.get('question', ''))
            return [{"role": "user", "content": user_prompt}]

        else:
            user_prompt = (
                "Given the following question, provide a clear and concise answer. Your answer "
                "should be \"yes,\" \"no,\" or a few words directly answering the question.\n"
                "Question: {}\nYour response should be concise and directly related to the "
                "question. End your response with: \"The answer is [your_answer].\""
            ).format(line.get('question', ''))
            return [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "The answer is"},
            ]

    elif dataset == 'Temporal_QA':
        content = (
            "You are tasked with a question-answer task. For each question, you need to "
            "provide the reason and then output the answer in the following JSON format.\n"
            '{"reason": "<detailed reasoning>", "answer": "<the answer>"}\n'
            "\nHere are some examples of how you should respond:\n"
            "**Question:** What is the capital of France?\n"
            "**Response:**\n"
            '{"reason": "France\'s capital city, Paris, is widely recognized and documented '
            'in various reliable sources including encyclopedias and official government '
            'websites.","answer": "Paris"}\n'
        )
        content += "\nAnswer the following questions using the format and guidelines provided above.\n"
        content += "**Question:** {}\n**Response:**".format(line.get('Question', ''))
        return [{"role": "user", "content": content}]

    elif dataset == 'MMLU':
        content = cot_prompt or ''
        if add_ref:
            content += (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Given the following question, references (may or may not be available), and "
                "four candidate answers (A, B, C, and D), explain your reasoning step-by-step "
                "based on the references and then choose the best answer. If there is no "
                "reference or you find the reference irrelevant, please choose the correct "
                "option based on your knowledge.\n\n"
            )
            if line.get('reference'):
                for ref_id, ref in enumerate(line['reference']):
                    content += "Reference {}: {}\n".format(ref_id + 1, ref)
            else:
                content += "Reference: No reference available.\n"
            content += "\nQuestion: {}\nA. {}\nB. {}\nC. {}\nD. {}\n".format(
                line.get('Question', ''),
                line.get('A', ''), line.get('B', ''), line.get('C', ''), line.get('D', '')
            )
            content += (
                "\nYour response should include the reasoning \"Reasoning: [reasoning_text]\" "
                "based on the references (or your knowledge if no references are available), "
                "and end with \"The best answer is [the_answer_letter]\" where "
                "[the_answer_letter] is one of A, B, C, or D."
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReasoning: "
            )
        elif output_reason:
            content += (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Given the following question and four candidate answers (A, B, C and D), "
                "explain your reasoning step-by-step and then choose the best answer.\n"
                "Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\n"
                "Your response should include the reasoning \"Reasoning: [reasoning_text]\" "
                "and end with \"The best answer is [the_answer_letter]\" where the "
                "[the_answer_letter] is one of A, B, C or D."
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReasoning: "
            ).format(
                line.get('Question', ''),
                line.get('A', ''), line.get('B', ''), line.get('C', ''), line.get('D', '')
            )
        else:
            content += (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Given the following question and four candidate answers (A, B, C and D), "
                "choose the best answer.\nQuestion: {}\nA. {}\nB. {}\nC. {}\nD. {}\n"
                "Your response should end with \"The best answer is [the_answer_letter]\" "
                "where the [the_answer_letter] is one of A, B, C or D."
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is"
            ).format(
                line.get('Question', ''),
                line.get('A', ''), line.get('B', ''), line.get('C', ''), line.get('D', '')
            )
        return [{"role": "user", "content": content}]

    raise ValueError("Unsupported dataset: {}".format(dataset))
