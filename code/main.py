from LLM_calls import load_llm, llm_call
from utils import read_data
import random
import json
from tqdm import tqdm
import argparse
'''
Here’s the revised prompt based on your updated task description:

---

## Task Name:  
Extract Relationships between Provided Entities from a Text

## Task Description:  
Your task is to extract triples that represent relationships between a set of **predefined entities** based on the provided text. The subject and object of each triple must come **only** from the list of entities that I provide. If no meaningful relationships are found between the entities, return **None**.

### Task Input:  
1. A piece of text.
2. A list of possible entities.

### Task Output:  
A list of triples in JSON format, or **None** if no valid relationships are found.

### JSON Format for Triples:
```json
[
    {
        "subject": "subject_entity",
        "predicate": "predicate_relation",
        "object": "object_entity"
    }
]
```

### Guidelines:
- The **subject** and **object** of the triple must come from the list of predefined entities. Do not extract any other words as entities.
- Use **exact phrases** from the text for the predicate (relationship) between the entities. Do not rephrase or summarize.
- If there are no valid relationships between the provided entities, return **None**.

### Examples:

#### Example 1:  
**Input Text:**  
"Paris is the capital of France."  
**Entities Provided:**  
["Paris", "France"]

**Output JSON:**
```json
[
    {
        "subject": "Paris",
        "predicate": "is the capital of",
        "object": "France"
    }
]
```

#### Example 2:  
**Input Text:**  
"Einstein developed the theory of relativity."  
**Entities Provided:**  
["Einstein", "the theory of relativity"]

**Output JSON:**
```json
[
    {
        "subject": "Einstein",
        "predicate": "developed",
        "object": "the theory of relativity"
    }
]
```

#### Example 3:  
**Input Text:**  
"She is a member of the organization."  
**Entities Provided:**  
["the organization"]

**Output JSON:**  
```json
None
```

#### Example 4:  
**Input Text:**  
"The Nile is the longest river in the world, flowing through Egypt."  
**Entities Provided:**  
["The Nile", "Egypt"]

**Output JSON:**
```json
[
    {
        "subject": "The Nile",
        "predicate": "flows through",
        "object": "Egypt"
    }
]
```

### Additional Notes:
- Ensure that the subject and object are always chosen from the provided entity list.
- If no valid relationships exist between the entities in the text, return **None**.
- Ensure all outputs are in valid JSON format.

---

This prompt reflects the updated requirement of working with predefined entities, helping ensure your friend extracts valid relationships only between those entities.
'''
def get_entity_list(line:dict):
    return list(line['passage_entity'].keys())

def prompt_fomular_triple_extraction(line:dict, provide_entity=False):
    if provide_entity == False:
        content = '## Task Name:\nExtract Triples from a Text\n\n'
        content += '## Task Description:\nA **triple** is the fundamental unit in a knowledge graph, represented in the form of **subject-predicate-object**. Your task is to extract meaningful triples from a provided text. Each triple should consist of **entities** as the subject and object, avoiding generic or abstract nouns such as "a member" or "the person." The words inside the triple (subject, predicate, object) should come **directly from the text**, without rephrasing or summarizing. If no meaningful triples can be extracted, return **None**.\n\n'
        content += '### Task Input: \nYou will be given a piece of text.\n\n'
        content += '### Task Output: \nA list of triples in JSON format, or **None** if no valid triples are found.\n\n'
        content += '### JSON Format for Triples:\n[{"subject": "subject_name", "predicate": "predicate_relation", "object": "object_name"}, {"subject": "subject_name","predicate": "predicate_relation","object": "object_name"}]\n\n'
        content += '### Guidelines: - **Entities** in the subject and object fields should be specific and identifiable, like "Einstein" or "France." Avoid extracting generic or abstract nouns like "a member" or "the person".\n- Use **exact phrases** from the original text for the subject, predicate, and object. Do not rephrase or summarize.\n- If the text contains no useful triples (e.g., only generic nouns or no clear entities), return **None**.\n\n'
        content += '### Examples:'
        content += '#### Example 1: \n**Input Text:**  \n"Paris is the capital of France."\n**Output JSON:**\n[{"subject": "Paris", "predicate": "is the capital of", "object": "France"}]\n\n'
        content += '#### Example 2: \n**Input Text:**  \n"She is a member of the organization.\n"**Output JSON:**\nNone\n\n'
        content += '### Additional Notes:\n- Ensure that the subject, predicate, and object are extracted exactly as they appear in the text.\n- If no valid entities are found in a sentence, do not force the extraction of triples. Simply return **None**\n- Ensure all outputs are in valid JSON format.\n\n\n'
        content += 'Now, Here is the text, please follow the above instruction to extract the triples:\n'
        content += '**Input Text:** \n{}\n'.format(line['passages'])
        content += '**Output JSON:** \n'

    else:
        content = '## Task Name: \nExtract Relationships between Provided Entities from a Text\n\n'
        content += '## Task Description: \nYour task is to extract triples that represent relationships between a set of **predefined entities** based on the provided text. The subject and object of each triple must come **only** from the list of entities that I provide. If no meaningful relationships are found between the entities, return **None**.\n\n'
        content += '### Task Input: \n1. A piece of text.\n2. A list of possible entities.\n\n'
        content += '### Task Output:  \nA list of triples in JSON format, or **None** if no valid relationships are found.\n\n'
        content += '### JSON Format for Triples:\n[{"subject": "subject_name", "predicate": "predicate_relation", "object": "object_name"}, {"subject": "subject_name","predicate": "predicate_relation","object": "object_name"}]\n\n'
        content += '### Guidelines: \n- The **subject** and **object** of the triple must come from the list of predefined entities. Do not extract any other words as entities.\n- If there are no valid relationships between the provided entities, return **None**.\n\n'
        content += '### Examples:\n\n'
        content += '#### Example 1: \n**Input Text:** \n"Paris is the capital of France."\n**Entities Provided:** \n["Paris", "France"]\n\n'
        content += '**Output JSON:**\n[{"subject": "Paris", "predicate": "is the capital of", "object": "France"}]\n\n'
        content += '#### Example 2: \n**Input Text:**  \n"Einstein developed the theory of relativity."  \n**Entities Provided:**  \n["Einstein", "the theory of relativity"]\n\n'
        content += '**Output JSON:**[{"subject": "Einstein", "predicate": "developed", "object": "the theory of relativity"}]\n\n'
        content += '#### Example 3:  \n**Input Text:**  \n"She is a member of the organization."  \n**Entities Provided:**  \n["the organization"]\n\n'
        content += '**Output JSON:** \nNone\n\n'
        content += '#### Example 4:  **Input Text:**  \n"The Nile is the longest river in the world, flowing through Egypt."  \n**Entities Provided:**  \n["The Nile", "Egypt"]\n\n'
        content += '**Output JSON:**\n[{"subject": "The Nile", "predicate": "flows through", "object": "Egypt"}]\n\n'
        content += '### Real Data:\n\n'
        content += 'Input Text:**  \n{}\n**Entities Provided:**\n{}\n\n'.format(line['passages'], get_entity_list(line))
        content += '**Output JSON:**\n'
    return content

def prompt_fomular_kg_local_check(line:dict):
    content = 'I need your help determining the reliability of a passage, and I’ve provided descriptions of the relevant entities mentioned in the text. Here are some hints:\n'
    content += '1. **Entity Accuracy**: Check if the details about the entities (e.g., names, attributes, roles) match the descriptions I’ve provided. Note that we do not require all relevant entities to appear in the article, you only need to verify that the entities present in the text do not conflict with the entities provided. You also do not need to check the relationships between the entities. However, if none of the entities appear, then the passage should be unreliable.\n'
    # content += '2. **Relation Validity**: Review the relationships between the entities. Are these relationships correctly represented according to the descriptions, or do they seem inconsistent?\n'
    # content += '3. **Consistency with Known Information**: Does the passage align with what is commonly known or accepted about the entities and their interactions?\n'
    # content += '4. **Context and Reliability**: Based on the above checks, please assess whether the passage is reliable overall. If there are any doubts or questionable points, provide your reasoning.\n\n'
    content += 'Please output your final evaluation in the following JSON format:\n'
    content += '{"reason": [your explanation for the decision],"reliability": "yes" or "no"}\n\n'
    content += 'Here is the Passages:\n{}\n\n'.format(line['passages'])
    content += 'Here are the relevant entities:\n'
    for i, ent in enumerate(line['question_entity'].values()):
        content += '{}.{}: {}\n'.format(i + 1, ent['entity'], ent['description'])
    content += '\nOutput:\n'

    return content

def prompt_fomular_reference_generate(line:dict, sub=False, add_entity=False, have_choice=False, CoT_prompt=None):
    if have_choice:
        system_prompt = "You are an intelligent assistant specialized in generating reference paragraphs for multiple-choice questions. Your task is to provide clear and concise paragraphs that contextualize each question and its answer choices. These paragraphs are meant to guide the next person in understanding the question and amplifying it effectively."
                
        system_prompt += '\nHints:\n'
        system_prompt += '1. Provide context or background information relevant to the question and choices.\n'
        system_prompt += '2. Include key facts, important dates if applicable.\n'
        system_prompt += '3. Keep the paragraphs concise but informative enough to guide a more detailed response.\n'
        if add_entity:
            system_prompt += '4. Please make sure that the paragraphs you generate do not conflict with the relevant entities.\n'
        
        system_prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\n'.format(system_prompt)

        if CoT_prompt == None:
            user_0 = 'I have a list of multiple-choice questions, and I\'d like you to write a reference paragraph for each question. These paragraphs will assist the person coming after me in understanding the context of the question and choices, enabling them to amplify and answer the questions concisely. You don\'t need to answer the questions directly, just provide enough information to guide the next person.\n'
            if add_entity and (len(line['query_entity'])):
                user_0 += 'To make your reference paragraph more accurate, I will provide you with the entities related to the question and you can refer to them.\n'

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
            user_1 += 'To make your reference paragraph more accurate, I will provide you with the entities related to the question and you can refer to them.\n'

        user_1 += 'Question: {}\n'.format(line['Question'])
        user_1 += 'A. {}\nB. {}\nC. {}\nD. {}\n'.format(line['A'], line['B'], line['C'], line['D'])
        if add_entity and (len(line['query_entity'])):
            user_1 += '\nRelated Entities:\n'
            for i, ent in enumerate(line['query_entity'].values()):
                user_1 += '{}. {}: {}\n'.format(i + 1, ent['entity'], ent['description'])
        user_1 += 'Your response should start with "Reference: [reference_paragraph]" where the [reference_paragraph] is the reference you write.\n'

        assist_1 = "Reference:"

        question_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{}'.format(user_1, assist_1)

        content = system_prompt + CoT + question_prompt

        return content 
    
    else:
        content = 'I have a list of questions and I\'d like you to write a reference paragraph for each question. These paragraphs will assist the person coming after me in amplifying and answering the questions concisely. You don\'t need to answer the questions directly, just provide enough information to guide the next person.\n'
        if add_entity:
            content += 'To make your reference paragraph more accurate, I will provide you with the entities related to the question and you can refer to them.\n'
        content += 'Please output your responses in JSON format containing a key named "reference_paragraph".\n'
        content += 'Here\'s an example to illustrate:\n\n'
        content += '### Example Question:\n'
        content += "Where is the capital of France?\n"
        if add_entity:
            content += '### Example Related Entities\n'
            content += '1. France: country in Western Europe'
        content += '### Example Output:\n'
        content += '{"question": "Where is the capital of France?", "reference_paragraph": "The capital of France is Paris. Paris, known for its historical landmarks such as the Eiffel Tower and the Louvre Museum, is located in the northern part of the country along the Seine River. It is a major European city and a global center for art, fashion, and culture."}\n\n'
        content += '### Hints:\n'
        content += '1. Provide context or background information relevant to the question.\n'
        content += '2. Include key facts, important dates, or notable figures if applicable.\n'
        content += '3. Keep the paragraphs concise but informative enough to guide a more detailed response.\n'
        if add_entity:
            content += '4. Please make sure that the paragraphs you generate do not conflict with the relevant entities.\n'
        content += '\nHere is the question:'
        if sub:
            content += '### Question:\n{}\n'.format(line['sub_question'])
        else:
            content += '### Question:\n{}\n'.format(line['Question'])
        if add_entity:
            content += '### Related Entities\n'
            for i, ent in enumerate(line['query_entity'].values()):
                content += '{}. {}: {}\n'.format(i + 1, ent['entity'], ent['description'])
        content += 'Output:\n'

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

def prompt_fomular_retrive_judge(line:dict):
    # content = 'I will provide you with a question and a reference that may or may not be useful. Your task is to determine if the reference is useful in answering the question. A reference is considered useful if it provides information that helps answer the question, even if it doesn\'t fully answer it.\n'
    content = 'I will provide you with a question and a reference that may or may not be useful. Your task is to determine if the reference is useful in answering the question. A reference is considered useful if it provides information that helps answer the question.\n'
    content += 'Please respond with a JSON object containing a key named "useful" with a value of either "yes" or "no". If you\'re not sure whether the reference is useful, please answer "yes", which is like:'
    content += '{"reason": "<detailed reasoning why the reference is useful or not>", "useful": "<yes or no>"}\n\n'
    content += '**Hints to Determine Usefulness:**\n\n'
    # content += '1. **Relevance**: Does the reference directly relate to the topic of the question? Even partial relevance counts.\n'
    content += '1. **Relevance**: Does the reference directly relate to the topic of the question?\n'
    content += '2. **Information Quality**: Does the reference provide accurate and credible information?\n'
    content += '3. **Detail**: Does the reference include details that help explain or support the answer to the question?\n'
    content += '4. **Context**: Does the reference give contextual information that is helpful for understanding the question better?\n'
    # content += '5. **Breadth**: Does the reference cover a broad aspect of the topic that might indirectly help answer the question?\n\n'
    content += 'Below are two specific examples to guide you:\n\n'
    content += '**Example 1: Positive Case**\n'
    content += '*Question:* What is the capital of France?\n'
    content += '*Reference:* France is a unitary semi-presidential republic with its capital in Paris, the country\'s largest city and main cultural and commercial centre.\n'
    content += '*Response:* {"reason": "The reference directly states that Paris is the capital of France, providing clear and relevant information to answer the question.", "useful": "yes"}\n\n'
    content += '**Example 2: Negative Case**\n'
    content += '*Question:* What is the capital of France?\n'
    content += '*Reference:* France retains its centuries-long status as a global centre of art, science, and philosophy. It hosts the third-largest number of UNESCO World Heritage Sites and is the world\'s leading tourist destination, receiving over 89 million foreign visitors in 2018.\n'
    content += '*Response:* {"reason": "The reference does not mention the capital of France or provide any information that helps answer the question. It talks about France\'s cultural and tourist significance but does not address the specific query.", "useful": "no"}\n\n'
    content += 'Now here are the question and reference.\n'
    content += '*Question:* {}\n'.format(line['Question'])
    content += '*Reference:* {}\n'.format(line['passages'])
    content += '*Response:* '

    return content


def prompt_fomular_summary(line:dict):
    content = 'Please read the text and provide a summary that captures the key information. Specifically, I\'m looking for details on the key information ,such as main characters, the time period, the location, and any major events or themes.\n'
    content += 'Please format your summary in JSON like this:\n'
    content += '{"summary": "Your summary here, including key information like main characters, time period, location, major events, and themes."}\n'
    content += 'Now here is the text:\n'
    content += '{}\n'.format(line['passages'])

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
            content += '<|start_header_id|>user<|end_header_id|>\n\nGiven the following question, relevant references, and four candidate answers (A, B, C, and D), explain your reasoning step-by-step based on the references and then choose the best answer.\n\n'
            content += 'Reference 1:\n{}\n\n'.format(line['query_pseudo_doc'])
            content += 'Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\n'.format(line['Question'], line['A'], line['B'], line['C'], line['D'])
            content += 'Your response should include the reasoning "Reasoning: [reasoning_text]" based on the references, and end with "The best answer is [the_answer_letter]" where [the_answer_letter] is one of A, B, C, or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReasoning: '
        elif output_reason:
            content += '<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), explain your reasoning step-by-step and then choose the best answer.\n'
            content += 'Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\nYour response should include the reasoning \"Reasoning: [reasoning_text]\" and end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReasoning: '.format(line['Question'], line['A'], line['B'], line['C'], line['D'])
        else:
            content += '<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: {}\nA. {}\nB. {}\nC. {}\nD. {}\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is'.format(line['Question'], line['A'], line['B'], line['C'], line['D'])
        
        return content

def process_file(data, output_file, args, model=None, tokenizer=None, pipeline=None, 
                dev_file=None, subject=None):
    if args.dataset_name == 'MMLU':
        CoT_prompt = ''

        for dev_line in dev_file:
            dev_line = json.loads(dev_line)
            if args.rag:
                CoT_prompt += '<|start_header_id|>user<|end_header_id|>\n\nGiven the following question, relevant references, and four candidate answers (A, B, C, and D), explain your reasoning step-by-step based on the references and then choose the best answer.\n\n'
                CoT_prompt += 'Reference 1:\n{}\n\n'.format(dev_line['query_pseudo_doc'])
                CoT_prompt += 'Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\n'.format(dev_line['Question'], dev_line['A'], dev_line['B'], dev_line['C'], dev_line['D'])
                CoT_prompt += 'Your response should include the reasoning "Reasoning: [reasoning_text]" based on the references, and end with "The best answer is [the_answer_letter]" where [the_answer_letter] is one of A, B, C, or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReasoning: {}\n\nThe best answer is {}.<|eot_id|>'.format(dev_line['query_pseudo_doc'] ,dev_line['Answer'])
            
            elif args.generate_reference:
                user_prompt = 'I have a list of multiple-choice questions, and I\'d like you to write a reference paragraph for each question. These paragraphs will assist the person coming after me in understanding the context of the question and choices, enabling them to amplify and answer the questions concisely. You don\'t need to answer the questions directly, just provide enough information to guide the next person.\n'
                if len(dev_line['query_entity']) != 0:
                    user_prompt += 'To make your reference paragraph more accurate, I will provide you with the entities related to the question and you can refer to them.\n'

                user_prompt += 'Question: {}\n'.format(dev_line['Question'])
                user_prompt += 'A. {}\nB. {}\nC. {}\nD. {}\n'.format(dev_line['A'], dev_line['B'], dev_line['C'], dev_line['D'])
                if len(dev_line['query_entity']) != 0:
                    user_prompt += '\nRelated Entities:\n'
                    for i, ent in enumerate(dev_line['query_entity'].values()):
                        user_prompt += '{}. {}: {}\n'.format(i + 1, ent['entity'], ent['description'])
                user_prompt += 'Your response should start with "Reference: [reference_paragraph]" where the [reference_paragraph] is the reference you write.\n'

                question_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{}'.format(user_prompt, dev_line['query_pseudo_doc'])

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
        
    i = 0
    for line in tqdm(data):
        if args.line:
            line = json.loads(line)
        if args.exinfo_judge:
            prompt = prompt_fomular_retrive_judge(line)
        elif args.summary:
            prompt = prompt_fomular_summary(line)
        elif args.decompose:
            prompt = prompt_fomular_decompose_question(line)
        elif args.generate_reference:
            if args.dataset_name == 'MMLU':
                prompt = prompt_fomular_reference_generate(line, add_entity=True, have_choice=True, CoT_prompt=CoT_prompt)
            else:
                prompt = prompt_fomular_reference_generate(line, add_entity=True)
        elif args.local_check:
            prompt = prompt_fomular_kg_local_check(line)
        elif args.extract_triple:
            prompt = prompt_fomular_triple_extraction(line, provide_entity=True)
        else:
            prompt = prompt_fomular(line, args.dataset_name, model=args.model_name, rag=args.rag, 
                                    CoT_prompt=CoT_prompt, subject=subject)

        messages = [{"role": "user", "content": prompt}]
        if args.model_name == 'Mistral':
            response = llm_call(messages, args.model_name, model=model, tokenizer=tokenizer)
        elif args.model_name == 'Llama':
            if args.logits:
                response = llm_call(messages, args.model_name, model=model, tokenizer=tokenizer, output_logit=True)
            else:
                response = llm_call(messages, args.model_name, pipeline=pipeline)
        line['llm_response'] = response
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')

        if args.test:
            print('-'*50 + 'PROMPT' + '-'*50)
            print(prompt)
            print('-'*50 + 'RESPONSE' + '-'*50)
            print(response)
            i += 1
            if i >= 3:
                break

def main(args):
    assert args.model_name.lower() in args.model_path.lower()
    if args.model_name == 'Mistral':
        model, tokenizer = load_llm(args.model_name, args.model_path)
        pipeline = None
    elif args.model_name == 'Llama':
        if args.logits:
            model, tokenizer = load_llm(args.model_name, args.model_path, logit=True)
            pipeline = None
        else:
            pipeline = load_llm(args.model_name, args.model_path)
            model, tokenizer = None, None

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
        subjects = sorted([f.split("_dev.json")[0] for f in os.listdir(os.path.join(args.dataset_path, "dev")) if "_dev.json" in f])
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
            dev_file = open(dev_file_name)
            input_file = open(input_file_name)

            output_file_name = os.path.join(save_dir, "{}_result.json".format(sub))
            output_file = open(output_file_name, 'w')
            process_file(input_file, output_file, args, model=model, tokenizer=tokenizer, pipeline=pipeline, dev_file=dev_file, subject=sub)

            if args.test:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocFixQA args')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help="Dataset Name")
    parser.add_argument('--dataset_path', type=str, help="Dataset Path", default=None)
    parser.add_argument('--model_name', '-m', type=str, required=True, help='Model Name')
    parser.add_argument('--exp_name','-e',type=str, default='test', help='Exp Name')
    parser.add_argument('--model_path','-p',type=str, required=True, help="Path to model")
    parser.add_argument('--MMLU_categories', type=str, help='MMLU category', choices=["STEM", "humanities", "social sciences", "other (business, health, misc.)"],
                        default=["STEM", "humanities", "social sciences", "other (business, health, misc.)"], nargs="+") # --MMLU_categories STEM humanities
    parser.add_argument('--dev_input',type=str, help="MMLU input dev sub dir", default='')
    parser.add_argument('--test_input',type=str, help="MMLU input test sub dir", default='')
    parser.add_argument('--test', action='store_true', help="if Test")
    parser.add_argument('--line', action='store_true', help="if Process by line")
    parser.add_argument('--rag', action='store_true', help="if Rag")
    parser.add_argument('--exinfo_judge', action='store_true', help="if External Information Filter by Pair")
    parser.add_argument('--summary', action='store_true', help="Summary Process")
    parser.add_argument('--decompose', action='store_true', help="Decompose the Question into Subqustion")
    parser.add_argument('--generate_reference', action='store_true', help="Generate Reference by LLM")
    parser.add_argument('--local_check', action='store_true', help="Chech the reliability of generate passages with local entity")
    parser.add_argument('--extract_triple', action='store_true', help="Extract triples in Text")
    parser.add_argument('--logits', action='store_true', help="For mult-choice QA, use logits to choose answer")
    parser.add_argument('--output_reason', action='store_true', help="If LLM output_reason")

    args = parser.parse_args()
    main(args)
