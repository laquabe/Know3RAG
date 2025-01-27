from LLM_calls import load_llm, llm_call
from utils import read_data
import random
import json
from tqdm import tqdm
import argparse
import re
import copy
'''
Here’s a revised version of your prompt, tailored to be simpler and more intuitive for the LLM to follow:

---

I have a task that requires your help in extracting relationships between predefined entities in a given text. You will be provided with a list of specific entities, and your job is to extract triples that represent relationships between these entities. Each triple should include a **subject**, **predicate**, and **object** taken directly from the text, where the **subject** and **object** must be from the predefined list of entities. If no meaningful relationships are found between the entities, return **None**.

### Hint:
- The **subject** and **object** should **only** come from the list of predefined entities.
- Extract the subject, predicate, and object exactly as they appear in the text.
- If no valid relationships are found between the provided entities, return **None**.
- Output the extracted triples in the format:  
  **Extracted triples: [list of triples]**.

Here’s an example with a multi-round dialog to guide you:

**User:**  
**Input Text:**  
"Albert Einstein was born in Ulm, Germany in 1879."  
**Entities Provided:**  
["Albert Einstein", "Ulm", "Germany"]

**Assistant:**  
**Extracted triples:**  
[{"subject": "Albert Einstein", "predicate": "was born in", "object": "Ulm"}, {"subject": "Albert Einstein", "predicate": "was born in", "object": "Germany"}]  

**User:**  
**Input Text:**  
"She is a member of the organization."  
**Entities Provided:**  
["the organization"]

**Assistant:**  
**Extracted triples:**  
None  

Now, please extract the relationships between the provided entities in the following text:

**Input Text:**  
{line['passages']}  
**Entities Provided:**  
{get_entity_list(line)}

**Extracted triples:**  

---

This version introduces the task in a clearer, more direct way, with an example that aligns with the multi-round dialog format. It also specifies the output format and simplifies the instructions. Would you like any further changes?
'''
def get_entity_list(line:dict):
    return list(line['query_entity'].keys())
'''
def prompt_fomular_triple_extraction(line:dict, src_key='passages', ent_key='passage_entity'):
    system_prompt = 'I have a task that requires your help in extracting relationships between predefined entities in a given text. You will be provided with a list of specific entities, and your job is to extract triples that represent relationships between these entities. Each triple should include a subject, predicate, and object taken directly from the text, where the subject and object must be from the predefined list of entities. If no meaningful relationships are found between the entities, return None.\n'
    system_prompt += 'Hint:\n'
    system_prompt += '- The subject and object should only come from the list of predefined entities.\n'
    system_prompt += '- Extract the subject, predicate, and object exactly as they appear in the text.\n'
    system_prompt += '- If no valid relationships are found between the provided entities, return None.\n'
    system_prompt += '- Output the extracted triples in the format: Extracted triples: [list of triples].\n'
    user_0 = 'Please extract the triples in the text. If no hint is matched, output None. the output format is Extracted triples: [list of triples].\n'
    user_0 += 'Text: Albert Einstein was born in Ulm, Germany in 1879.\n'
    user_0 += 'Entities: Albert Einstein, Ulm, Germany\n'
    assist_0 = '[{"subject": "Albert Einstein", "predicate": "was born in", "object": "Ulm"}, {"subject": "Albert Einstein", "predicate": "was born in", "object": "Germany"}]'
    user_1 = 'Please extract the triples in the text. If no hint is matched, output None. the output format is Extracted triples: [list of triples].\n'
    user_1 += 'Text: She is a member of the organization.\n'
    user_1 += 'Entities: the organization\n'
    assist_1 = 'None\n'
    user_2 = 'Please extract the triples in the text. If no hint is matched, output None. the output format is Extracted triples: [list of triples].\n'
    user_2 += 'Text: {}\n'.format(line[src_key])
    user_2 += 'Entities:'
    for ent in line[ent_key].keys():
        user_2 += ' {},'.format(ent)
    user_2.rstrip(',')
    user_2 += '\n'
    '''

def prompt_fomular_triple_extraction(line: dict, src_key='passages', ent_key='passage_entity'):
    system_prompt = 'I have a task to extract relationships between entities from a given text. '
    system_prompt += 'You will be provided with a list of possible entities as a reference. Your task is to extract triples representing relationships between entities in the text. Each triple should include a subject, predicate, and object directly from the text. '
    system_prompt += 'The entities in the triples may or may not match the provided reference list, but use the list to guide your extraction process.'
    system_prompt += 'If no meaningful relationships are found, return None.\n'
    system_prompt += 'Instructions:\n'
    system_prompt += '- Extract the subject, predicate, and object exactly as they appear in the text.\n'
    system_prompt += '- If no valid relationships are found, return None.\n'
    system_prompt += '- Output only the extracted triples in the format: [list of triples].\n'

    user_0 = 'Text: Albert Einstein was born in Ulm, Germany in 1879.\n'
    user_0 += 'Entities: Albert Einstein, Ulm, Germany\n'
    
    assist_0 = '[{"subject": "Albert Einstein", "predicate": "was born in", "object": "Ulm"}, {"subject": "Albert Einstein", "predicate": "was born in", "object": "Germany"}]'

    user_1 = 'Text: She is a member of the organization.\n'
    user_1 += 'Entities: the organization\n'
    assist_1 = 'None'

    user_2 = 'Text: {}\n'.format(line[src_key])
    user_2 += 'Entities: '
    for ent in line[ent_key].keys():
        user_2 += ' {},'.format(ent)
    user_2.rstrip(',')

    content = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": user_0},
        {"role":"assistant", "content": assist_0},
        {"role":"user", "content": user_1},
        {"role":"assistant", "content": assist_1},
        {"role":"user", "content": user_2}
    ]

    return content

def prompt_fomular_kg_local_check(line:dict, have_choice=False, check_key='passages'):
    system_prompt = 'I need your help determining the reliability of a passage in the context of its ability to answer a specific question. I might provide some entities to help you better understand the problem. Here are the key considerations:\n'
    system_prompt += '1. Passage Relevance: Check if the passage provides information relevant to answering the question. Even if the passage does not directly mention the entities, it should address the key concepts or ideas related to the question. If the passage does not contribute meaningfully to answering the question, it may be unreliable.\n'
    system_prompt += '2. Entity Accuracy: The entities provided are from the question and may not appear in the passage. These entities are meant to help you understand the context of the question. If the passage conflicts with the entities provided (e.g., incorrect descriptions or relationships), this could affect its reliability.\n'
    system_prompt += '3. Overall Reliability: Based on the relevance of the passage to the question and the accuracy of the entities, assess whether the passage is reliable for answering the question. If there are doubts or inconsistencies, provide a clear explanation.\n'
    
    user_0 = 'Confirm that the article is reliable for the question. Provide your reasoning for the reliability decision, and end your response with: "The reliability of the passage is [yes or no]."\n'
    if have_choice:
        user_0 += 'Question: What is the nutritional value of an apple?\nA. High in fiber and vitamins.\nB Low in calories but high in protein.\nC. Rich in fats.D. No nutritional value\n'
    else:
        user_0 += 'Question: What is the nutritional value of an apple?\n'
    user_0 += 'Entities:\n'
    user_0 += '1. Apple: A fruit known for its nutritional benefits, such as fiber and vitamins.\n'
    user_0 += 'Passage: An apple is a nutritious fruit rich in fiber, vitamins, and antioxidants.\n'
     
    assist_0 = 'The passage provides relevant information about the nutritional value of an apple, aligning with the question. The entity "Apple" refers to the fruit, which matches the context of the question. The reliability of the passage is yes.'
    
    user_1 = 'Confirm that the article is reliable for the question. Provide your reasoning for the reliability decision, and end your response with: "The reliability of the passage is [yes or no]."\n'
    if have_choice:
        user_1 += 'Question: What is the CEO of Apple Inc.?\nA. Tim Cook\nB. Steve Jobs\nC. Elon Musk\nD. Satya Nadella\n'
    else:
        user_1 += 'Question: What is the CEO of Apple Inc.?\n'
    user_1 += 'Entities:\n'
    user_1 += '1. Apple Inc.: A technology company, known for products like the iPhone and Mac computers.\n'
    user_1 += 'Passage: Apples are widely consumed fruits that come in different varieties, including Granny Smith and Red Delicious.\n'
     
    assist_1 = 'The passage discusses apples as a fruit, which is unrelated to the question about the CEO of Apple Inc. The passage does not address the company or its leadership. The reliability of the passage is no.'

    user_2 = 'Confirm that the article is reliable for the question. Provide your reasoning for the reliability decision, and end your response with: "The reliability of the passage is [yes or no]."\n'
    if have_choice:
        user_2 += 'Question: What is the CEO of Apple Inc.?\nA. Tim Cook\nB. Steve Jobs\nC. Elon Musk\nD. Satya Nadella\n'
    else:
        user_2 += 'Question: What is the CEO of Apple Inc.?\n'
    user_2 += 'Entities:\n'
    user_2 += '1. Apple: A fruit known for its sweet taste and variety of colors, including red, green, and yellow.\n'
    user_2 += 'Passage: Apples are a popular fruit consumed worldwide. They are rich in fiber and vitamins, often enjoyed raw or in various dishes.\n'

    assist_2 = 'The passage discusses the fruit *Apple*, which is unrelated to the question about the CEO of *Apple Inc.*, a technology company. The entity provided incorrectly refers to the fruit, not the tech company. This makes the passage unreliable for answering the question. The reliability of the passage is no.'

    user_3 = 'Confirm that the article is reliable for the question. Provide your reasoning for the reliability decision, and end your response with: "The reliability of the passage is [yes or no]."\n'
    if have_choice:
        user_3 += 'Question: What is the CEO of Apple Inc.?\nA. Tim Cook\nB. Steve Jobs\nC. Elon Musk\nD. Satya Nadella\n'
    else:
        user_3 += 'Question: What is the CEO of Apple Inc.?\n'
    user_3 += 'Entities:\n'
    user_3 += '1. Apple: A fruit known for its sweet taste and variety of colors, including red, green, and yellow.\n'
    user_3 += 'Passage: Tim Cook is the current CEO of Apple Inc., a technology company known for its products such as the iPhone and Mac computers.\n'

    assist_3 = 'Despite the provided entity referring to the fruit *Apple*, the passage directly answers the question by stating that Tim Cook is the CEO of *Apple Inc.* The passage is relevant to the question, and the reliability is unaffected by the unrelated entity description. The reliability of the passage is yes.'

    user_4 = 'Confirm that the article is reliable for the question. Provide your reasoning for the reliability decision, and end your response with: "The reliability of the passage is [yes or no]."\n'
    if have_choice:
        user_4 += 'Question: {}\n'.format(line['Question'])
        user_4 += 'A. {}\nB. {}\nC. {}\nD. {}\n'.format(line['A'], line['B'], line['C'], line['D'])
    else:
        user_4 += 'Question: {}\n'.format(line['question'])
    if (len(line['query_entity'])):
        user_4 += 'Entities:\n'
        for i, ent in enumerate(line['query_entity'].values()):
            user_4 += '{}. {}: {}\n'.format(i + 1, ent['entity'], ent['description'])
    user_4 += 'Passage: {}\n'.format(line[check_key])
    
    content = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": user_0},
        {"role":"assistant", "content": assist_0},
        {"role":"user", "content": user_1},
        {"role":"assistant", "content": assist_1},
        {"role":"user", "content": user_2},
        {"role":"assistant", "content": assist_2},
        {"role":"user", "content": user_3},
        {"role":"assistant", "content": assist_3},
        {"role":"user", "content": user_4},
    ]

    return content

def prompt_fomular_reference_generate(line:dict, sub=False, add_entity=False, have_choice=False, CoT_prompt=None):
    if have_choice:
        if CoT_prompt == None:
            user_0 = 'I have a list of multiple-choice questions, and I\'d like you to write a reference paragraph for each question. These paragraphs will assist the person coming after me in understanding the context of the question and choices, enabling them to amplify and answer the questions concisely. You don\'t need to answer the questions directly, just provide enough information to guide the next person.\n'
            if add_entity and (len(line['query_entity'])):
                user_0 += 'To make your reference passages more accurate, I\'m going to provide you with some entities inside the question that you can refer to them, but they\'re not necessarily accurate.\n'

            user_0 += '\nQuestion: Which city is the capital of France?\n'
            user_0 += 'A. Paris\nB. London\nC. Berlin\nD. Madrid\n'
            if add_entity and (len(line['query_entity'])):
                user_0 += '\nRelated Entities:\n'
                user_0 += '1. France: country in Western Europe\n'
            user_0 += 'Your response should start with "Reference: [reference_paragraph]" where the [reference_paragraph] is the reference you write.\n'

            assist_0 = "Reference: The capital of France is Paris. Paris, known for its historical landmarks such as the Eiffel Tower and the Louvre Museum, is located in the northern part of the country along the Seine River. It is a major European city and a global center for art, fashion, and culture."

            CoT = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>'.format(user_0, assist_0)
        else:
            CoT = CoT_prompt
        
        user_1 = 'I have a list of multiple-choice questions, and I\'d like you to write a reference paragraph for each question. These paragraphs will assist the person coming after me in understanding the context of the question and choices, enabling them to amplify and answer the questions concisely. You don\'t need to answer the questions directly, just provide enough information to guide the next person.\n'
        if add_entity and (len(line['query_entity'])):
            user_1 += 'To make your reference passages more accurate, I\'m going to provide you with some entities inside the question that you can refer to them, but they\'re not necessarily accurate.\n'

        user_1 += 'Question: {}\n'.format(line['Question'])
        user_1 += 'A. {}\nB. {}\nC. {}\nD. {}\n'.format(line['A'], line['B'], line['C'], line['D'])
        if add_entity and (len(line['query_entity'])):
            user_1 += '\nRelated Entities:\n'
            for i, ent in enumerate(line['query_entity'].values()):
                user_1 += '{}. {}: {}\n'.format(i + 1, ent['entity'], ent['description'])
        user_1 += 'Your response should start with "Reference: [reference_paragraph]" where the [reference_paragraph] is the reference you write.\n'

        assist_1 = "Reference:"

        question_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{}'.format(user_1, assist_1)

        content = CoT + question_prompt

        return content 
    
    else:
        user_prompt = 'I have a list of open-ended questions, and I\'d like you to write a reference paragraph for each question. These paragraphs should provide sufficient background, key concepts, or context to guide the next person in answering the question effectively. You do not need to provide an answer directly, just enough information to help the next person frame their answer concisely and accurately.\n'
        if len(line['query_entity']) != 0:
            user_prompt += 'To make your reference passages more accurate, I\'m going to provide you with some entities inside the question that you can refer to them, but they\'re not necessarily accurate.\n'

        user_prompt += 'Question: {}\n'.format(line['question'])
        if len(line['query_entity']) != 0:
            user_prompt += '\nRelated Entities:\n'
            for i, ent in enumerate(line['query_entity'].values()):
                user_prompt += '{}. {}: {}\n'.format(i + 1, ent['entity'], ent['description'])
        user_prompt += 'You should just output "[reference_paragraph]", where the [reference_paragraph] is the reference you write.\n'
        content = [{"role":"user", "content": user_prompt}]

    return content

def prompt_fomular_decompose_question(line:dict):
    content = 'I have a problem that I need to break down into sub-problems. please extract the key entities from the problem and come up with one piece of information that needs to be collected for each entity in JSON format. You don\'t need to answer these questions; just identify what information should be gathered.\n'
    content += 'To give you a clearer idea, here\'s an example problem and how it should be broken down:\n\n'
    content += '**Example Problem:**\nWho was the current President of the United States when Zootopia was released?\n'
    content += '**Entities and Information Needed:**\n'
    content += '{"President of the United States": "Who was the President of the United States?", "Zootopia": "When was Zootopia released?"}\n\n'
    content += 'Here\'s the problem I need your help with:\n'
    content += '**Problem:**\n{}'.format(line['Question'])

    return content

def prompt_fomular_summary(line:dict):
    content = '<|start_header_id|>user<|end_header_id|>\n\nPlease read the text and provide a summary that captures the key information.\n'
    content += 'Your response should directly output the summary "Summary: [summary of the text]"\n'
    content += 'Text:\n'
    content += '{}\n'.format(line['passages'])
    content += '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nSummary:'
    return content

def prompt_fomular_selfask(line:dict, have_choice=False):
    '''
    "Given the following question and references, you should first assess if you have enough information to answer. 
    If you feel your own knowledge and the provided references are insufficient, respond with 'yes' and indicate what additional information is needed. 
    If you believe you can answer based on the current data, respond with 'no'."
    '''
    if len(line.get('reference', [])) > 0:
        user_0 = "Given the following references and question, you should first assess if you have enough information to answer. If you feel your own knowledge and the provided references are insufficient, respond with 'yes'. If you believe you can answer based on the current data, respond with 'no'.\n"
        for ref_id, ref in enumerate(line['reference']):
            user_0 += "Reference {}: {}\n".format(ref_id + 1, ref)
    else:
        user_0 = "Given the following question, you should first assess if you have enough information to answer. If you feel your own knowledge is insufficient, respond with 'yes'. If you believe you can answer, respond with 'no'.\n"
    
    if have_choice:
        user_0 += '\nQuestion: {}\nA. {}\nB. {}\nC. {}\nD. {}\n'.format(line['Question'], line['A'], line['B'], line['C'], line['D'])
    else:
        user_0 += '\nQuestion: {}\n'.format(line['Question'])
    
    user_0 += "\nDo you need more information?\n"

    content = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>'.format(user_0)

    return content

def find_options_positions(text, options=['a', 'b', 'c', 'd']):
    positions = {}
    
    for option in options:
        match = re.search(rf'{option}', text)
        if match:
            positions[option] = match.start()
    
    first_option = min(positions, key=positions.get) if positions else None

    return positions, first_option

def answer_phrase(line:dict, answer_key):
    error_flag = True
    possible_prefix = ["The best answer is ", "the best answer is ", "answer:", 'answer is:', 'answer is ']
    pred = line[answer_key]
    for prefix in possible_prefix:
        if prefix in pred.lower():
            idx = pred.lower().rfind(prefix)
            pred_ans = pred[idx + len(prefix) : ]
            pred_ans = pred_ans.strip()
            if len(pred_ans) > 0:
                error_flag = False
                break

    if error_flag:
        pred_ans = pred.strip()
        '''text'''
        _, pred_choice = find_options_positions(pred_ans.lower(), ['a.', 'b.', 'c.', 'd.'])
        if pred_choice != None:
            pred_choice = pred_choice[0]
        else:
            _, pred_choice = find_options_positions(pred_ans.lower())
    else:
        _, pred_choice = find_options_positions(pred_ans.lower())

    return pred_choice, not error_flag

def prompt_fomular_judge(line:dict, answer_key_list, have_choice=False):
    '''
    "You will act as a judge to evaluate answers provided by multiple sources for a given question. I will provide you with the question and their respective answers. Your task is to analyze their reasoning, compare their validity, and provide a final, well-reasoned answer based on the evidence and logic presented."
    '''
    system_prompt = "You will act as a judge to evaluate answers provided by multiple sources for a given question."
    user_0 = "I will provide you with the question and some respective answers. Your task is to analyze their reasoning, compare their validity, and provide a final, well-reasoned answer based on the evidence and logic presented.\n"
    if have_choice:
        user_0 += '\nQuestion: {}\nA. {}\nB. {}\nC. {}\nD. {}\n'.format(line['Question'], line['A'], line['B'], line['C'], line['D'])
    else:
        user_0 += '\nQuestion: {}\n'.format(line['Question'])
    
    for idx, ans_key in enumerate(answer_key_list):
        user_0 += 'Answer {}: {}\n'.format(idx + 1, line[ans_key])
    
    user_0 += "\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.\n"

    content = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>'.format(system_prompt)
    content += '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>'.format(user_0)
    
    return content

def external_knowledge_prompt(line, src_key, local_check=False):
    ex_know_list = line[src_key]
    if local_check == True:
        ex_know_list = [line['pseudo_doc_raw']]
    elif src_key == 'pseudo_doc_entity':
        ex_know_list = [ex_know_list]
    context = ''
    for idx, ex_know in enumerate(ex_know_list):
        context += 'Reference {}: {}\n'.format(idx + 1, ex_know)
    context += '\n'
    return context

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def prompt_generate_question(line:dict):
    user_1 = 'You will be given references, a question, and an answer. The answer may be incomplete or incorrect. Identify the most critical missing or incorrect information in the references and the answer. Formulate one most important new question that will most effectively help retrieve the necessary information to answer the original question.\n'

    if len(line['reference']) > 0:
        for ref_id, ref in enumerate(line['reference']):
            user_1 += "Reference {}: {}\n".format(ref_id + 1, ref)
    else:
        user_1 += 'Reference: No reference available.\n'
    user_1 += 'Question: {}\n'.format(line['question'])
    user_1 += 'Answer: {}\n'.format(line['llm_response'])
    user_1 += 'Please directly output the new question:'

    content = [{"role":"user", "content": user_1}]

    return content


def prompt_fomular(line:dict, dataset, model=None, shuffle=True, rag=False, src_key='passages',
                   subject=None, CoT_prompt=None, logits=False, output_reason=True, add_ref=True):
    if dataset == 'Truthful_QA':
        content = 'I will give a question and some answer choices, please select the only correct answer.\n\n'
        content += 'Question:{}\n'.format(line['question'])
        
        candidates_list = list(line['mc1_targets'].keys())
        if shuffle:
            random.shuffle(candidates_list)
        for i, cand in enumerate(candidates_list):
            content += '{}.{}\n'.format(i, cand)
        
        '''raw'''
        content += 'The answer is therefore:'
        '''explain'''

        return content
    if dataset == 'hotpotQA':
        if add_ref:
            user_prompt = 'Given the following question, references (may or may not be available), explain your reasoning step-by-step based on the references and then provide your best possible answer. If there is no reference or you find the reference irrelevant, please provide an answer based on your knowledge.\n\n'
            if len(line['reference']) > 0:
                for ref_id, ref in enumerate(line['reference']):
                    user_prompt += "Reference {}: {}\n".format(ref_id + 1, ref)
            else:
                user_prompt += 'Reference: No reference available.\n'
            user_prompt += '\nQuestion: {}\n'.format(line['question'])
            user_prompt += '\nYour response should end with "The answer is [your_answer_text]", where the [your_answer_text] should be yes, no, or a few words directly answering the question.\n Let\'s think step by step.'
            content = [{"role":"user", "content": user_prompt}]

        elif output_reason:
            user_prompt = 'Given the following open-ended question, explain your reasoning step-by-step and then provide your final answer.\n'
            user_prompt += 'Question: {}\nYour response should end with "The answer is [your_answer_text]". Let\'s think step by step.'.format(line['question'])
            content = [{"role":"user", "content": user_prompt}]
        else:
            user_prompt = 'Given the following question, provide a clear and concise answer. Your answer should be "yes," "no," or a few words directly answering the question.\nQuestion: {}\nYour response should be concise and directly related to the question. End your response with: "The answer is [your_answer]."'.format(line['question'])
            assist_prompt = 'The answer is'
            content = [{"role":"user", "content": user_prompt},
                       {"role":"assistant", "content": assist_prompt}]
            
        return content
    
    elif dataset == 'Temporal_QA':
        content = 'You are tasked with a question-answer task. For each question, you need to provide the reason and then output the answer in the following JSON format.\n'
        content += '{"reason": "<detailed reasoning>", "answer": "<the answer>"}\n'
        content += '\nHere are some examples of how you should respond:\n'
        content += '**Question:** What is the capital of France?\n'
        content += '**Response:**\n{"reason": "France\'s capital city, Paris, is widely recognized and documented in various reliable sources including encyclopedias and official government websites.","answer": "Paris"}\n'
        if rag and (len(line[src_key]) != 0):
            content += '\nNow, before you answer the question, you can read the following references to ensure your answers are accurate. It is worth noting that when there is no relevant information in the reference, you can rely on the knowledge you have to answer the question:\n\n'
            content += external_knowledge_prompt(line, src_key, local_check=False)
        else:
            content += '\n'
        content += 'Answer the following questions using the format and guidelines provided above.\n**Question:** {}\n**Response:**'.format(line['Question'])

        return content
    elif dataset == 'MMLU':
        content = CoT_prompt
        if add_ref:
            content += '<|start_header_id|>user<|end_header_id|>\n\nGiven the following question, references (may or may not be available), and four candidate answers (A, B, C, and D), explain your reasoning step-by-step based on the references and then choose the best answer. If there is no reference or you find the reference irrelevant, please choose the correct option based on your knowledge.\n\n'
            if len(line['reference']) > 0:
                for ref_id, ref in enumerate(line['reference']):
                    content += "Reference {}: {}\n".format(ref_id + 1, ref)
            else:
                content += 'Reference: No reference available.\n'
            content += '\nQuestion: {}\nA. {}\nB. {}\nC. {}\nD. {}\n'.format(line['Question'], line['A'], line['B'], line['C'], line['D'])
            content += '\nYour response should include the reasoning "Reasoning: [reasoning_text]" based on the references (or your knowledge if no references are available), and end with "The best answer is [the_answer_letter]" where [the_answer_letter] is one of A, B, C, or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReasoning: '
        elif output_reason:
            content += '<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), explain your reasoning step-by-step and then choose the best answer.\n'
            content += 'Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\nYour response should include the reasoning \"Reasoning: [reasoning_text]\" and end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReasoning: '.format(line['Question'], line['A'], line['B'], line['C'], line['D'])
        else:
            content += '<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: {}\nA. {}\nB. {}\nC. {}\nD. {}\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is'.format(line['Question'], line['A'], line['B'], line['C'], line['D'])
        
        return content

def process_file(data, output_file, args, model=None, tokenizer=None, pipeline=None, 
                dev_file=None, subject=None):
    if args.dataset_name == 'MMLU':
        if args.local_check:
            CoT_prompt = ''
        elif args.extract_triple:
            CoT_prompt = ''
        elif args.summary:
            CoT_prompt = ''
        elif args.selfask:
            CoT_prompt = ''
        elif args.judge:
            CoT_prompt = ''
        else:
            CoT_prompt = []
            for dev_line in dev_file:
                dev_line = json.loads(dev_line)
                
                if args.rag:
                    user_prompt += 'Given the following question, references (may or may not be available), and four candidate answers (A, B, C, and D), explain your reasoning step-by-step based on the references and then choose the best answer. If there is no reference or you find the reference irrelevant, please choose the correct option based on your knowledge.\n\n'
                    if len(dev_line['reference']) > 0:
                        for ref_id, ref in enumerate(dev_line['reference']):
                            user_prompt += "Reference {}: {}\n".format(ref_id + 1, ref)
                    else:
                        user_prompt += 'Reference: No reference available.\n'
                    user_prompt += '\nQuestion: {}\nA. {}\nB. {}\nC. {}\nD. {}\n'.format(dev_line['Question'], dev_line['A'], dev_line['B'], dev_line['C'], dev_line['D'])
                    user_prompt += '\nYour response should include the reasoning "Reasoning: [reasoning_text]" based on the references (or your knowledge if no references are available), and end with "The best answer is [the_answer_letter]" where [the_answer_letter] is one of A, B, C, or D'
                    assist_prompt = 'Reasoning: {}\n\nThe best answer is {}.'.format(dev_line['llm_response'] ,dev_line['Answer'])
                    CoT_prompt.extend([{"role":"user", "content": user_prompt},
                                        {"role":"assistant", "content": assist_prompt}])
                    
                elif args.generate_reference:
                    user_prompt = 'I have a list of multiple-choice questions, and I\'d like you to write a reference paragraph for each question. These paragraphs will assist the person coming after me in understanding the context of the question and choices, enabling them to amplify and answer the questions concisely. You don\'t need to answer the questions directly, just provide enough information to guide the next person.\n'
                    if len(dev_line['query_entity']) != 0:
                        user_prompt += 'To make your reference passages more accurate, I\'m going to provide you with some entities inside the question that you can refer to them, but they\'re not necessarily accurate.\n'

                    user_prompt += 'Question: {}\n'.format(dev_line['Question'])
                    user_prompt += 'A. {}\nB. {}\nC. {}\nD. {}\n'.format(dev_line['A'], dev_line['B'], dev_line['C'], dev_line['D'])
                    if len(dev_line['query_entity']) != 0:
                        user_prompt += '\nRelated Entities:\n'
                        for i, ent in enumerate(dev_line['query_entity'].values()):
                            user_prompt += '{}. {}: {}\n'.format(i + 1, ent['entity'], ent['description'])
                    user_prompt += 'Your response should start with "Reference: [reference_paragraph]" where the [reference_paragraph] is the reference you write.\n'

                    question_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nReference: {}<|eot_id|>'.format(user_prompt, dev_line['query_pseudo_doc'])

                    CoT_prompt += question_prompt

                elif args.output_reason:
                    reason_str = dev_line['reason']
                    reason_str = reason_str.lstrip('Reasoning:')
                    reason_str = reason_str.strip()
                    CoT_prompt += '<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), explain your reasoning step-by-step and then choose the best answer.\n'
                    CoT_prompt += 'Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\nYour response should include the reasoning \"Reasoning: [reasoning_text]\" and end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReasoning: {}\n\nThe best answer is {}.<|eot_id|>'.format(dev_line['Question'], dev_line['A'], dev_line['B'], dev_line['C'], dev_line['D'], reason_str ,dev_line['Answer'])
                else:
                    CoT_prompt += '<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\n'
                    CoT_prompt += 'Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is {}.<|eot_id|>'.format(dev_line['Question'], dev_line['A'], dev_line['B'], dev_line['C'], dev_line['D'], dev_line['Answer'])
    
    if args.dataset_name == 'hotpotQA':
        if args.local_check:
            CoT_prompt = []
        elif args.extract_triple:
            CoT_prompt = []
        elif args.summary:
            CoT_prompt = []
        elif args.selfask:
            CoT_prompt = []
        elif args.judge:
            CoT_prompt = []
        else:
            if args.generate_reference:
                system_prompt = "You are an intelligent assistant specialized in generating reference paragraphs for open-ended questions. Your task is to provide clear and concise reference paragraphs that contextualize the question and guide the answerer in understanding the context, key concepts, and relevant details. These paragraphs are meant to provide sufficient information for the next person to answer the question accurately and thoroughly."
                system_prompt += '\nHints:\n'
                system_prompt += '1. Provide background information that is relevant to the question.\n'
                system_prompt += '2. Clarify any key terms or concepts that might be important for answering the question.\n'
                system_prompt += '3. Provide context such as important dates, figures, or events if applicable.\n'
                system_prompt += '4. Keep the paragraph concise but detailed enough to guide the next person in framing an answer.\n'
                system_prompt += '5. If there are provded entities and the entities mentioned in the question are accurate, ensure they are consistent with your reference. If the entities are inaccurate, you may disregard them.\n'
                CoT_prompt = [{"role":"system", "content": system_prompt}]
            else:
                CoT_prompt = []
            for dev_line in dev_file:
                dev_line = json.loads(dev_line)
                
                if args.rag:
                    user_prompt = 'Given the following question, references (may or may not be available), explain your reasoning step-by-step based on the references and then provide your best possible answer. If there is no reference or you find the reference irrelevant, please provide an answer based on your knowledge.\n\n'
                    if len(dev_line['reference']) > 0:
                        for ref_id, ref in enumerate(dev_line['reference']):
                            user_prompt += "Reference {}: {}\n".format(ref_id + 1, ref)
                    else:
                        user_prompt += 'Reference: No reference available.\n'
                    user_prompt += '\nQuestion: {}\n'.format(dev_line['question'])
                    user_prompt += '\nYour response should end with "The answer is [your_answer_text]", where the [your_answer_text] should be yes, no, or a few words directly answering the question.\n Let\'s think step by step.'
                    assist_prompt = '{}\n\nThe answer is {}.'.format(dev_line['reason'], dev_line['answer'])
                    CoT_prompt.extend([{"role":"user", "content": user_prompt},
                                        {"role":"assistant", "content": assist_prompt}])
                elif args.generate_question:
                    user_prompt = 'You will be given references, a question, and an answer. The answer may be incomplete or incorrect. Identify the most critical missing or incorrect information in the references and the answer. Formulate one most important new question that will most effectively help retrieve the necessary information to answer the original question.\n'
                    if len(dev_line['reference']) > 0:
                        for ref_id, ref in enumerate(dev_line['reference']):
                            user_prompt += "Reference {}: {}\n".format(ref_id + 1, ref)
                    else:
                        user_prompt += 'Reference: No reference available.\n'
                    user_prompt += 'Question: {}\n'.format(dev_line['question'])
                    user_prompt += 'Answer: {}\n\nThe answer is {}.\n'.format(dev_line['reason'], dev_line['answer'])
                    user_prompt += 'Please directly output the new question:'
                    assist_prompt = dev_line['new_question']

                    CoT_prompt.extend([{"role":"user", "content": user_prompt},
                                        {"role":"assistant", "content": assist_prompt}])

                elif args.generate_reference:
                    user_prompt = 'I have a list of open-ended questions, and I\'d like you to write a reference paragraph for each question. These paragraphs should provide sufficient background, key concepts, or context to guide the next person in answering the question effectively. You do not need to provide an answer directly, just enough information to help the next person frame their answer concisely and accurately.\n'
                    if len(dev_line['query_entity']) != 0:
                        user_prompt += 'To make your reference passages more accurate, I\'m going to provide you with some entities inside the question that you can refer to them, but they\'re not necessarily accurate.\n'

                    user_prompt += 'Question: {}\n'.format(dev_line['question'])
                    if len(dev_line['query_entity']) != 0:
                        user_prompt += '\nRelated Entities:\n'
                        for i, ent in enumerate(dev_line['query_entity'].values()):
                            user_prompt += '{}. {}: {}\n'.format(i + 1, ent['entity'], ent['description'])
                    user_prompt += 'You should just output "[reference_paragraph]", where the [reference_paragraph] is the reference you write.\n'
                    assist_prompt = '{}'.format(dev_line['query_pseudo_doc'])

                    CoT_prompt.extend([{"role":"user", "content": user_prompt},
                                        {"role":"assistant", "content": assist_prompt}])

                elif args.output_reason:
                    reason_str = dev_line['reason']
                    user_prompt = 'Given the following open-ended question, explain your reasoning step-by-step and then provide your final answer.\n'
                    user_prompt += 'Question: {}\nYour response should end with "The answer is [your_answer_text]". Let\'s think step by step.'.format(dev_line['question'], )
                    assist_prompt = '{}\n\nThe answer is {}.'.format(reason_str, dev_line['answer'])
                    CoT_prompt.extend([{"role":"user", "content": user_prompt},
                                        {"role":"assistant", "content": assist_prompt}])
                    
                else:
                    user_prompt = 'Given the following question, provide a clear and concise answer. Your answer should be "yes," "no," or a few words directly answering the question.\nQuestion: {}\nYour response should be concise and directly related to the question. End your response with: "The answer is [your_answer]."'.format(dev_line['question'])
                    assist_prompt = 'The answer is {}.'.format(dev_line['answer'])
                    CoT_prompt.extend([{"role":"user", "content": user_prompt},
                                        {"role":"assistant", "content": assist_prompt}])
    i = 0
    for line in tqdm(data):
        prompt_final = copy.deepcopy(CoT_prompt)
        if args.line:
            line = json.loads(line)
        
        if args.summary:
            prompt = prompt_fomular_summary(line)
        elif args.decompose:
            prompt = prompt_fomular_decompose_question(line)
        elif args.selfask:
            if args.dataset_name == 'MMLU':
                prompt = prompt_fomular_selfask(line, have_choice=True)
            else:
                prompt = prompt_fomular_selfask(line)
        elif args.judge:
            answer_key_list = ['turn1_response', 'turn0_response']
            phrase_key= True
            ans_set = set()
            for ans_key in answer_key_list:
                ans, err = answer_phrase(line, ans_key)
                phrase_key = phrase_key & err
                ans_set.add(ans)
            if phrase_key and (len(ans_set) == 1):
                line['llm_response'] = line[answer_key_list[0]]
                output_file.write(json.dumps(line, ensure_ascii=False) + '\n')
                continue

            if args.dataset_name == 'MMLU':
                prompt = prompt_fomular_judge(line, answer_key_list, have_choice=True)
            else:
                prompt = prompt_fomular_judge(line, answer_key_list)
        elif args.generate_reference:
            if args.dataset_name == 'MMLU':
                prompt = prompt_fomular_reference_generate(line, add_entity=True, have_choice=True, CoT_prompt=CoT_prompt)
            else:
                prompt = prompt_fomular_reference_generate(line, add_entity=True, CoT_prompt=CoT_prompt)
        elif args.local_check:
            if args.dataset_name == 'MMLU':
                prompt = prompt_fomular_kg_local_check(line, have_choice=True, check_key=args.src_key)
            else:
                prompt = prompt_fomular_kg_local_check(line, check_key=args.src_key)
        elif args.extract_triple:
            prompt = prompt_fomular_triple_extraction(line, src_key=args.src_key, ent_key=args.entity_key)
        elif args.generate_question:
            prompt = prompt_generate_question(line)
        else:
            prompt = prompt_fomular(line, args.dataset_name, model=args.model_name, rag=args.rag, 
                                    CoT_prompt=CoT_prompt, output_reason=args.output_reason, add_ref=args.rag)

        prompt_final.extend(prompt)
        if args.model_name == 'Mistral':
            response = llm_call(prompt_final, args.model_name, model=model, tokenizer=tokenizer)
        elif args.model_name == 'Llama':
            if args.logits:
                response = llm_call(prompt_final, args.model_name, model=model, tokenizer=tokenizer, output_logit=True)
            else:
                response = llm_call(prompt_final, args.model_name, pipeline=pipeline)
        elif args.model_name == 'api':
            response = prompt_final
        elif args.model_name == 'GLM4':
            response = llm_call(prompt_final, args.model_name, model=model, tokenizer=tokenizer)
        elif args.model_name == 'Qwen':
            response = llm_call(prompt_final, args.model_name, model=model, tokenizer=tokenizer)

        line['llm_response'] = response
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')

        if args.test:
            print('-'*50 + 'PROMPT' + '-'*50)
            for prompt_test in prompt_final:
                print(prompt_test['content'])
            print('-'*50 + 'RESPONSE' + '-'*50)
            print(response)
            i += 1
            if i >= 3:
                break

def main(args):
    if args.model_name == 'Mistral':
        assert args.model_name.lower() in args.model_path.lower()
        model, tokenizer = load_llm(args.model_name, args.model_path)
        pipeline = None
    elif args.model_name == 'Llama':
        assert args.model_name.lower() in args.model_path.lower()
        if args.logits:
            model, tokenizer = load_llm(args.model_name, args.model_path, logit=True)
            pipeline = None
        else:
            pipeline = load_llm(args.model_name, args.model_path)
            model, tokenizer = None, None
    elif args.model_name == 'api':
        model, tokenizer, pipeline = None, None, None
    elif args.model_name == 'GLM4':
        model, tokenizer = load_llm(args.model_name, args.model_path)
        pipeline = None
    elif args.model_name == 'Qwen':
        model, tokenizer = load_llm(args.model_name, args.model_path)
        pipeline = None

    if args.dataset_name == 'Temporal':
        # input_file
        dataset = '{}_QA'.format(args.dataset_name)
        if args.dataset_path:
            dataset_path = args.dataset_path
        else:
            dataset_path = '/data/xkliu/LLMs/DocFixQA/datasets/{}QA/dev.json'.format(args.dataset_name)

        if not args.line:
            data = read_data(dataset, dataset_path)
        else:
            data = open(dataset_path)
        # output_file
        output_file_name = 'result/{}QA/{}/{}.json'.format(args.dataset_name, args.model_name, args.exp_name)
        output_file = open(output_file_name, 'w')
        # process
        process_file(data, output_file, args, model=model, tokenizer=tokenizer, pipeline=pipeline)

    if args.dataset_name == 'Truthful':
        # input_file
        dataset = '{}_QA'.format(args.dataset_name)
        if args.dataset_path:
            dataset_path = args.dataset_path
        else:
            dataset_path = '/data/xkliu/LLMs/DocFixQA/datasets/TruthfulQA/truthfulqa_mc_task.json'

        if not args.line:
            data = read_data(dataset, dataset_path)
        else:
            data = open(dataset_path)
        # output_file
        output_file_name = 'result/{}QA/{}/{}.json'.format(args.dataset_name, args.model_name, args.exp_name)
        output_file = open(output_file_name, 'w')
        # process
        process_file(data, output_file, args, model=model, tokenizer=tokenizer, pipeline=pipeline)

    if args.dataset_name == 'MMLU':
        import os
        from mmlu_categories import subcategories, categories
        
        #load src dir
        subjects = sorted([f.split("_dev.json")[0] for f in os.listdir(os.path.join('datasets', 'MMLU', 'data', "dev")) if "_dev.json" in f])
        if args.exp_name == '':
            exp_name = 'test'
        else:
            exp_name = args.exp_name
        # mkdir save dir
        save_dir = os.path.join('result', 'MMLU', exp_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # read category
        train_categories = []
        for c in args.MMLU_categories:
            train_categories.extend(categories[c])
        
        for sub in subjects:
            if len(set(subcategories[sub]) & set(train_categories)) == 0:
                continue
            print(sub)
            if len(args.dev_input) > 0:
                dev_file_name = os.path.join(args.dataset_path, "dev", args.dev_input, sub + "_dev.json")
            else:
                dev_file_name = os.path.join(args.dataset_path, "dev", sub + "_dev.json")

            if len(args.test_input) > 0:
                input_file_name = os.path.join(args.dataset_path, "test", args.test_input, sub + "_test.json")
            else:
                input_file_name = os.path.join(args.dataset_path, "test", sub + "_test.json")
            
            if os.path.exists(dev_file_name):
                dev_file = open(dev_file_name)
            else:
                dev_file = None
            input_file = open(input_file_name)

            output_file_name = os.path.join(save_dir, "{}_result.json".format(sub))
            output_file = open(output_file_name, 'w')
            process_file(input_file, output_file, args, model=model, tokenizer=tokenizer, pipeline=pipeline, dev_file=dev_file, subject=sub)

            if args.test:
                break
        
    if args.dataset_name == 'hotpotQA':
        import os
        
        if args.split_num == -1:
            save_dir = os.path.join('result', 'hotpotQA', args.model_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            output_file_name = os.path.join(save_dir, '{}.json'.format(args.exp_name))
        else:
            save_dir = os.path.join('result', 'hotpotQA', args.model_name, args.exp_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            output_file_name = os.path.join(save_dir, '{}_{}.json'.format(args.exp_name, args.split_num))
        
        output_file = open(output_file_name, 'w')
        if args.split_num == -1:
            input_file_name = os.path.join(args.dataset_path, '{}.json'.format(args.test_input))
        else:
            input_file_name = os.path.join(args.dataset_path, args.test_input, '{}_{}.json'.format(args.test_input, args.split_num))
        data = open(input_file_name)
        dev_file = open(os.path.join(args.dataset_path, args.dev_input + '.json'))
        process_file(data, output_file, args, model=model, tokenizer=tokenizer, pipeline=pipeline, dev_file=dev_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocFixQA args')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help="Dataset Name")
    parser.add_argument('--dataset_path', type=str, help="Dataset Path", default=None)
    parser.add_argument('--model_name', '-m', type=str, required=True, help='Model Name')
    parser.add_argument('--exp_name','-e',type=str, default='test', help='Exp Name')
    parser.add_argument('--model_path','-p',type=str, required=True, help="Path to model")
    parser.add_argument('--MMLU_categories', type=str, help='MMLU category', choices=["STEM", "humanities", "social sciences", "other (business, health, misc.)", "history", "philosophy", "law"],
                        default=["STEM", "humanities", "social sciences", "other (business, health, misc.)"], nargs="+") # --MMLU_categories STEM humanities
    parser.add_argument('--split_num',type=int, help="split data number", default=-1)
    parser.add_argument('--dev_input',type=str, help="MMLU input dev sub dir", default='')
    parser.add_argument('--test_input',type=str, help="MMLU input test sub dir", default='')
    parser.add_argument('--test', action='store_true', help="if Test")
    parser.add_argument('--line', action='store_true', help="if Process by line")
    parser.add_argument('--rag', action='store_true', help="if Rag")
    parser.add_argument('--summary', action='store_true', help="Summary Process")
    parser.add_argument('--selfask', action='store_true', help="Self Ask for More Information")
    parser.add_argument('--judge', action='store_true', help="Judge the different answer")
    parser.add_argument('--decompose', action='store_true', help="Decompose the Question into Subqustion")
    parser.add_argument('--generate_reference', action='store_true', help="Generate Reference by LLM")
    parser.add_argument('--generate_question', action='store_true', help="Generate Question by LLM")
    parser.add_argument('--local_check', action='store_true', help="Chech the reliability of generate passages with local entity")
    parser.add_argument('--extract_triple', action='store_true', help="Extract triples in Text")
    parser.add_argument('--logits', action='store_true', help="For mult-choice QA, use logits to choose answer")
    parser.add_argument('--output_reason', action='store_true', help="If LLM output_reason")
    parser.add_argument('--src_key', help="Passage Key", default='passages')
    parser.add_argument('--entity_key', help="Entity Key", default='passage_entity')

    args = parser.parse_args()
    main(args)
