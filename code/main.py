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

def prompt_fomular_reference_generate(line:dict, sub=False, add_entity=False):
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
        for i, ent in enumerate(line['question_entity'].values()):
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

def prompt_fomular(line:dict, dataset, model=None, shuffle=True, extral_ask=True, rag=False, src_key='passages'):
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
        # if extral_ask:
        #     content += 'If you encounter a question that requires additional external information that you don\'t currently have, you must state in your reasoning what external knowledge is required. In such cases, the answer field should be: "I need external knowledge.".Here is an example of the expected output format:\n'
        #     content += '{"reason": "I need information about the historical significance of the event to answer this question accurately.","answer": "I need external knowledge."}\n'
        content += '\nHere are some examples of how you should respond:\n'
        content += '**Question:** What is the capital of France?\n'
        content += '**Response:**\n{"reason": "France\'s capital city, Paris, is widely recognized and documented in various reliable sources including encyclopedias and official government websites.","answer": "Paris"}\n'
        # if extral_ask:
        #     content += '**Question:** What was the impact of the Treaty of Versailles on Germany?\n'
        #     content += '**Response:**\n{"reason": "To accurately discuss the impact of the Treaty of Versailles on Germany, I need detailed information on the economic, political, and social consequences it had on the country post-World War I.","answer": "I need external knowledge."}\n\n'
        if rag and (len(line[src_key]) != 0):
            content += '\nNow, before you answer the question, you can read the following references to ensure your answers are accurate. It is worth noting that when there is no relevant information in the reference, you can rely on the knowledge you have to answer the question:\n\n'
            content += external_knowledge_prompt(line, src_key, local_check=False)
        else:
            content += '\n'
        content += 'Answer the following questions using the format and guidelines provided above.\n**Question:** {}\n**Response:**'.format(line['Question'])

        return content
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocFixQA args')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help="Dataset Name")
    parser.add_argument('--dataset_path', type=str, help="Dataset Path", default=None)
    parser.add_argument('--model_name', '-m', type=str, required=True, help='Model Name')
    parser.add_argument('--exp_name','-e',type=str, required=True, default='test', help='Exp Name')
    parser.add_argument('--model_path','-p',type=str, required=True, help="Path to model")
    parser.add_argument('--test', action='store_true', help="if Test", default=None)
    parser.add_argument('--self_ask', action='store_true', help="if Self Ask", default=None)
    parser.add_argument('--rag', action='store_true', help="if Rag", default=None)
    parser.add_argument('--exinfo_judge', action='store_true', help="if External Information Filter by Pair", default=None)
    parser.add_argument('--line', action='store_true', help="if Process by line", default=None)
    parser.add_argument('--summary', action='store_true', help="Summary Process", default=None)
    parser.add_argument('--decompose', action='store_true', help="Decompose the Question into Subqustion", default=None)
    parser.add_argument('--generate_reference', action='store_true', help="Generate Reference by LLM", default=None)
    parser.add_argument('--local_check', action='store_true', help="Chech the reliability of generate passages with local entity", default=None)
    parser.add_argument('--extact_triple', action='store_true', help="Extract triples in Text", default=None)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    exp_name = args.exp_name
    full_flag = False if args.test else True
    extral_ask = True if args.self_ask else False
    rag_flag = True if args.rag else False
    exinfo_judge = True if args.exinfo_judge else False
    line_flag = True if args.line else False
    summary_flag = True if args.summary else False
    decompose_flag = True if args.decompose else False
    gen_reference_flag = True if args.generate_reference else False
    local_check_flag = True if args.local_check else False
    extract_triple_flag = True if args.extact_triple else False

    dataset = '{}_QA'.format(dataset_name)
    if args.dataset_path:
        dataset_path = args.dataset_path
    elif dataset_name == 'Temporal':
        dataset_path = '/data/xkliu/LLMs/DocFixQA/datasets/{}QA/dev.json'.format(dataset_name)
    elif dataset_name == 'Truthful':
        dataset_path = '/data/xkliu/LLMs/DocFixQA/datasets/TruthfulQA/truthfulqa_mc_task.json'
        

    output_file_name = 'result/{}QA/{}/{}.json'.format(dataset_name, model_name, exp_name)
    output_file = open(output_file_name, 'w')
    
    assert model_name.lower() in args.model_path.lower()
    if model_name == 'Mistral':
        model, tokenizer = load_llm(model_name, args.model_path)
    elif model_name == 'Llama':
        pipeline = load_llm(model_name, args.model_path)
    
    if not line_flag:
        data = read_data(dataset, dataset_path)
    else:
        data = open(dataset_path)
    
    if full_flag:
        for line in tqdm(data):
            if line_flag:
                line = json.loads(line)

            if exinfo_judge:
                prompt = prompt_fomular_retrive_judge(line)
            elif summary_flag:
                prompt = prompt_fomular_summary(line)
            elif decompose_flag:
                prompt = prompt_fomular_decompose_question(line)
            elif gen_reference_flag:
                prompt = prompt_fomular_reference_generate(line, add_entity=True)
            elif local_check_flag:
                prompt = prompt_fomular_kg_local_check(line)
            elif extract_triple_flag:
                prompt = prompt_fomular_triple_extraction(line, provide_entity=True)
            else:
                prompt = prompt_fomular(line, dataset, model=model_name, extral_ask=extral_ask, rag=rag_flag, src_key='pseudo_doc_entity')

            messages = [{"role": "user", "content": prompt}]
            if model_name == 'Mistral':
                response = llm_call(messages, model_name, model=model, tokenizer=tokenizer)
            elif model_name == 'Llama':
                response = llm_call(messages, model_name, pipeline=pipeline)
            line['llm_response'] = response
            output_file.write(json.dumps(line, ensure_ascii=False) + '\n')
    else:
        for i, line in enumerate(data):
            if line_flag:
                line = json.loads(line)

            if exinfo_judge:
                prompt = prompt_fomular_retrive_judge(line)
            elif summary_flag:
                prompt = prompt_fomular_summary(line)
            elif decompose_flag:
                prompt = prompt_fomular_decompose_question(line)
            elif gen_reference_flag:
                prompt = prompt_fomular_reference_generate(line, add_entity=True)
            elif local_check_flag:
                prompt = prompt_fomular_kg_local_check(line)
            elif extract_triple_flag:
                prompt = prompt_fomular_triple_extraction(line, provide_entity=True)
            else:
                prompt = prompt_fomular(line, dataset, model=model_name, extral_ask=extral_ask, rag=rag_flag, src_key='pseudo_doc_entity')

            print('-'*50 + 'PROMPT' + '-'*50)
            print(prompt)
            messages = [{"role": "user", "content": prompt}]
            if model_name == 'Mistral':
                response = llm_call(messages, model_name, model=model, tokenizer=tokenizer)
            elif model_name == 'Llama':
                response = llm_call(messages, model_name, pipeline=pipeline)
            line['llm_response'] = response
            output_file.write(json.dumps(line, ensure_ascii=False) + '\n')
            print('-'*50 + 'RESPONSE' + '-'*50)
            print(response)
            if i >= 3:
                break
