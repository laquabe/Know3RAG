import os
from mmlu_categories import subcategories, categories
from api import chat_with_openai, run_batch
from tqdm import tqdm
import json
import asyncio

'''
Here’s a sample prompt you can use to generate reasoning directly based on your multiple-choice questions and answers:  

---

**Prompt:**  
You are a reasoning expert tasked with explaining answers to multiple-choice questions clearly and concisely. For each question, use the provided correct answer to write a reasoning that:  
- Explains why the correct answer is valid.  
- If necessary, briefly contrasts it with the incorrect options to highlight key differences.  
Output the reasoning directly in full sentences, formatted as follows:  
1. **Reasoning for Question [Question Number]:** [Detailed Reasoning]  

### Example Input:  
**Question 1:** What is the capital of France?  
**Options:**  
A. Berlin  
B. Madrid  
C. Paris  
D. Rome  
**Correct Answer:** C  

**Expected Output:**  
1. **Reasoning for Question 1:** Paris is the capital of France. It is the largest city in France and is well-known as a major cultural and historical center. The other options (Berlin, Madrid, Rome) are capitals of other European countries.  

---  

You can adapt the template to suit your exact needs, like adding specific formatting preferences or emphasizing certain reasoning styles.'''

def prompt_fomular_triple_extraction(line:dict):
    system_prompt = 'I have a task that requires your help in extracting relationships between predefined entities in a given text. You will be provided with a list of specific entities, and your job is to extract triples that represent relationships between these entities. Each triple should include a subject, predicate, and object taken directly from the text, where the subject and object must be from the predefined list of entities. If no meaningful relationships are found between the entities, return None.\n'
    system_prompt += 'Hint:\n'
    system_prompt += '- The subject and object should only come from the list of predefined entities.\n'
    system_prompt += '- Extract the subject, predicate, and object exactly as they appear in the text.\n'
    system_prompt += '- If no valid relationships are found between the provided entities, return None.\n'
    system_prompt += '- Output the extracted triples in the format: Extracted triples: [list of triples].\n'
    user_0 = 'Please extract the triples in the text. If no hint is matched, output None. the output format is Extracted triples: [list of triples].\n'
    user_0 += 'Text: Albert Einstein was born in Ulm, Germany in 1879.\n'
    user_0 += 'Entities: Albert Einstein, Ulm, Germany\n'
    assist_0 = 'Extracted triples: [{"subject": "Albert Einstein", "predicate": "was born in", "object": "Ulm"}, {"subject": "Albert Einstein", "predicate": "was born in", "object": "Germany"}]'
    user_1 = 'Please extract the triples in the text. If no hint is matched, output None. the output format is Extracted triples: [list of triples].\n'
    user_1 += 'Text: She is a member of the organization.\n'
    user_1 += 'Entities: the organization\n'
    assist_1 = 'Extracted triples: None\n'
    user_2 = 'Please extract the triples in the text. If no hint is matched, output None. the output format is Extracted triples: [list of triples].\n'
    user_2 += 'Text: {}\n'.format(line['passages'])
    user_2 += 'Entities:'
    for ent in line['passage_entity']:
        user_2 += ' {},'.format(ent)
    user_2.rstrip(',')
    user_2 += '\n'
    assist_2 = 'Extracted triples:'

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_0},
        {"role": "assistant", "content": assist_0},
        {"role": "user", "content": user_1},
        {"role": "assistant", "content": assist_1},
        {"role": "user", "content": user_2},
        {"role": "assistant", "content": assist_2}
    ]

    return messages

def prompt_fomular_kg_local_check(line:dict, have_choice=False):
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
     
    assist_0 = 'Explanation: The passage provides relevant information about the nutritional value of an apple, aligning with the question. The entity "Apple" refers to the fruit, which matches the context of the question. The reliability of the passage is yes.'
    
    user_1 = 'Confirm that the article is reliable for the question. Provide your reasoning for the reliability decision, and end your response with: "The reliability of the passage is [yes or no]."\n'
    if have_choice:
        user_1 += 'Question: What is the CEO of Apple Inc.?\n'
    else:
        user_1 += 'Question: What is the CEO of Apple Inc.?\nA. Tim Cook\nB. Steve Jobs\nC. Elon Musk\nD. Satya Nadella\n'
    user_1 += 'Entities:\n'
    user_1 += '1. Apple Inc.: A technology company, known for products like the iPhone and Mac computers.\n'
    user_1 += 'Passage: Apples are widely consumed fruits that come in different varieties, including Granny Smith and Red Delicious.\n'
     
    assist_1 = 'Explanation: The passage discusses apples as a fruit, which is unrelated to the question about the CEO of Apple Inc. The passage does not address the company or its leadership. The reliability of the passage is no.'


    user_2 = 'Confirm that the article is reliable for the question. Provide your reasoning for the reliability decision, and end your response with: "The reliability of the passage is [yes or no]."\n'
    if have_choice:
        user_2 += 'Question: {}\n'.format(line['Question'])
        user_2 += 'A. {}\nB. {}\nC. {}\nD. {}\n'.format(line['A'], line['B'], line['C'], line['D'])
    else:
        user_2 += 'Question: {}\n'.format(line['Question'])
    if (len(line['query_entity'])):
        user_2 += 'Entities:\n'
        for i, ent in enumerate(line['query_entity'].values()):
            user_2 += '{}. {}: {}\n'.format(i + 1, ent['entity'], ent['description'])
    user_2 += 'Passage: {}\n'.format(line['passages'])
    assist_2 = 'Explanation: '
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_0},
        {"role": "assistant", "content": assist_0},
        {"role": "user", "content": user_1},
        {"role": "assistant", "content": assist_1},
        {"role": "user", "content": user_2},
        {"role": "assistant", "content": assist_2}
    ]

    return messages

def reason_message_with_references(line: dict, have_choice):
    if have_choice == True:
        system_prompt = 'You are a reasoning expert tasked with explaining answers to multiple-choice questions clearly and concisely, using provided references to enhance your explanation. If no reference is provided, use logical reasoning and general knowledge to support your explanation.'
        prompt_0 = "I've got some questions, and I need you to help me fill in the gaps in my reasoning. For each question, use the provided references (if available) and the correct answer to write reasoning that:\n"
        prompt_0 += "- Explains why the correct answer is valid, supported by relevant points from the references (if provided).\n"
        prompt_0 += "- If no reference is available, rely on logical reasoning and general knowledge to justify the correct answer.\n"
        prompt_0 += "- If necessary, briefly contrasts it with the incorrect options to highlight key differences, using the references where applicable.\n"
        prompt_0 += "Output the reasoning directly in full sentences, formatted as follows:\n"
        prompt_0 += "Reasoning: [Detailed Reasoning]\n\n"
        prompt_0 += "Now here is the question and its references (if any):\n"
        for ref_id, ref in enumerate(line['reference']):
            prompt_0 += "Reference {}: {}\n".format(ref_id, ref)
        prompt_0 += "Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer: {}\n".format(
            line['Question'], line['A'], line['B'], line['C'], line['D'], line['Answer']
        )
        assist_0 = "Reasoning: "

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_0},
            {"role": "assistant", "content": assist_0}
        ]

        return messages
    else:
        system_prompt = 'You are a reasoning expert to answer an open-ended question. Use the provided answer only as a reference to help guide the reasoning process, but do not assume the answer is known to you. If references are available, incorporate them into the reasoning process. If no references are provided, rely on logical reasoning and general knowledge to build your answer.'

        prompt_0 = "I need your help in writing a reasoning to answer the following question. Use the provided answer only as a reference to guide the reasoning process, but do not assume the answer is known to you.\n"
        # prompt_0 += "- Begin by quickly analyzing the question and identifying the key concepts.\n"
        prompt_0 += "- Use references (if available) to help refine your reasoning; otherwise, rely on logical reasoning and general knowledge.\n"
        # prompt_0 += "- Focus on the essential steps needed to arrive at the correct answer, avoiding unnecessary details.\n"
        # prompt_0 += "- Keep the explanation concise, highlighting only the most important points of the reasoning.\n"
        # prompt_0 += "- The provided answer should be used only to check if the reasoning aligns with the correct conclusion.\n"
        prompt_0 += "Output the reasoning directly in full sentences, formatted as follows:\n"
        prompt_0 += "Reasoning: [Concise and focused reasoning]\n\n"
        prompt_0 += "Now here is the question and its references (if any):\n"
        for ref_id, ref in enumerate(line['reference']):
            prompt_0 += "Reference {}: {}\n".format(ref_id, ref)
        prompt_0 += "Question: {}\nAnswer: {}\nLet's think step by step.\n".format(line['question'], line['answer'])
        assist_0 = "Reasoning: "

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_0},
            {"role": "assistant", "content": assist_0}
        ]

        return messages

def prompt_generate_question(line:dict):
    system_prompt = 'You are an expert assistant.'
    user_1 = 'You will be given references, a question, and an answer. The answer may be incomplete or incorrect. Identify the most critical missing or incorrect information in the references and the answer. Formulate one most important new question that will most effectively help retrieve the necessary information to answer the original question.'
    if len(line['reference']) > 0:
        for ref_id, ref in enumerate(line['reference']):
            user_1 += "Reference {}: {}\n".format(ref_id + 1, ref)
    else:
        user_1 += 'Reference: No reference available.\n'
    user_1 += 'Question: {}\n'.format(line['question'])
    user_1 += 'Answer: {}\n\nThe answer is {}.'.format(line['reason'], line['answer'])
    user_1 += 'Please directly output the new question:'

    content = [{"role":"system", "content": system_prompt,
        "role":"user", "content": user_1}]

    return content

def prompt_fomular_reference_generate(line:dict, sub=False, add_entity=False, have_choice=False):
    if have_choice:
        system_prompt = "You are an intelligent assistant specialized in generating reference paragraphs for multiple-choice questions. Your task is to provide clear and concise paragraphs that contextualize each question and its answer choices. These paragraphs are meant to guide the next person in understanding the question and amplifying it effectively."
        user_0 = 'I have a list of multiple-choice questions, and I\'d like you to write a reference paragraph for each question. These paragraphs will assist the person coming after me in understanding the context of the question and choices, enabling them to amplify and answer the questions concisely. You don\'t need to answer the questions directly, just provide enough information to guide the next person.\n'
        if add_entity and (len(line['query_entity'])):
            user_0 += 'To make your reference passages more accurate, I\'m going to provide you with some entities inside the question that you can refer to them, but they\'re not necessarily accurate.\n'
        
        user_0 += '\nHints:\n'
        user_0 += '1. Provide context or background information relevant to the question and choices.\n'
        user_0 += '2. Include key facts, important dates if applicable.\n'
        user_0 += '3. Keep the paragraphs concise but informative enough to guide a more detailed response.\n'
        if add_entity:
            user_0 += '4. Please make sure that the paragraphs you generate do not conflict with the relevant entities if entities are accurate. (If the entities are inaccurate, ignore them.)\n'

        user_0 += '\nQuestion: Which city is the capital of France?\n'
        user_0 += 'A. Paris\nB. London\nC. Berlin\nD. Madrid\n'
        if add_entity and (len(line['query_entity'])):
            user_0 += '\nRelated Entities:\n'
            user_0 += '1. France: country in Western Europe\n'
            user_0 += '2. Paris: capital city of France\n'
            user_0 += '3. London: capital and largest city of England and the United Kingdom\n'
            user_0 += '4. Berlin: federated state, capital and largest city of Germany\n'
            user_0 += '5. Madrid: municipality and capital of Spain\n'

        user_0 += 'Your response should start with "Reference: [reference_paragraph]" where the [reference_paragraph] is the reference you write.\n'

        assist_0 = "Reference: The capital of France is Paris. Paris, known for its historical landmarks such as the Eiffel Tower and the Louvre Museum, is located in the northern part of the country along the Seine River. It is a major European city and a global center for art, fashion, and culture."
        
        user_1 = 'Question: {}\n'.format(line['Question'])
        user_1 += 'A. {}\nB. {}\nC. {}\nD. {}\n'.format(line['A'], line['B'], line['C'], line['D'])
        if add_entity and (len(line['query_entity'])):
            user_1 += '\nRelated Entities:\n'
            for i, ent in enumerate(line['query_entity'].values()):
                user_1 += '{}. {}: {}\n'.format(i + 1, ent['entity'], ent['description'])
        user_1 += 'Your response should start with "Reference: [reference_paragraph]" where the [reference_paragraph] is the reference you write.\n'
        assist_1 = "Reference:"

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_0},
            {"role": "assistant", "content": assist_0},
            {"role": "user", "content": user_1},
            {"role": "assistant", "content": assist_1},
        ]
        
        return message
    
    else:
        system_prompt = "You are an intelligent assistant specialized in generating reference paragraphs for open-ended questions. Your task is to provide clear and concise reference paragraphs that contextualize the question and guide the answerer in understanding the context, key concepts, and relevant details. These paragraphs are meant to provide sufficient information for the next person to answer the question accurately and thoroughly."
        user_0 = 'I have a list of open-ended questions, and I\'d like you to write a reference paragraph for each question. These paragraphs should provide sufficient background, key concepts, or context to guide the next person in answering the question effectively. You do not need to provide an answer directly, just enough information to help the next person frame their answer concisely and accurately.\n'
        if add_entity and (len(line['query_entity'])):
            user_0 += 'To make your reference passages more accurate, I\'m going to provide you with some entities mentioned in the question. These entities may not necessarily be accurate.\n'

        user_0 += '\nHints:\n'
        user_0 += '1. Provide background information that is relevant to the question.\n'
        user_0 += '2. Clarify any key terms or concepts that might be important for answering the question.\n'
        user_0 += '3. Provide context such as important dates, figures, or events if applicable.\n'
        user_0 += '4. Keep the paragraph concise but detailed enough to guide the next person in framing an answer.\n'
        if add_entity:
            user_0 += '5. If the entities mentioned in the question are accurate, ensure they are consistent with your reference. If the entities are inaccurate, you may disregard them.\n'

        user_0 += '\nQuestion: What is the capital of France?\n'
        if add_entity and (len(line['query_entity'])):
            user_0 += '\nRelated Entities:\n'
            user_0 += '1. France: country in Western Europe\n'
            user_0 += '2. Paris: capital city of France\n'
        user_0 += 'Answer: Prais.\n'

        user_0 += 'Your response should start with "Reference: [reference_paragraph]" where the [reference_paragraph] is the reference you write.\n'

        assist_0 = "Reference: The capital of France is Paris. Paris is one of the most famous cities in the world, known for its cultural and historical significance. It is located in the northern part of the country along the Seine River and has been the political, economic, and cultural center of France for centuries. Key landmarks include the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral."

        user_1 = 'Question: {}\n'.format(line['question'])
        if add_entity and (len(line['query_entity'])):
            user_1 += '\nRelated Entities:\n'
            for i, ent in enumerate(line['query_entity'].values()):
                user_1 += '{}. {}: {}\n'.format(i + 1, ent['entity'], ent['description'])
        user_1 += 'Answer: {}'.format(line['answer'])
        user_1 += 'Your response should start with "Reference: [reference_paragraph]" where the [reference_paragraph] is the reference you write.\n'

        assist_1 = "Reference:"

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_0},
            {"role": "assistant", "content": assist_0},
            {"role": "user", "content": user_1},
            {"role": "assistant", "content": assist_1},
        ]

        return message

def prompt_relation_templates(line):
    system_prompt = 'You are an expert in natural language understanding and knowledge graph design. Your task is to assist in converting predicates from knowledge graphs into natural language problem templates.'
    user_0 = 'I am working on converting predicates from knowledge graphs into natural language problem templates. The goal is to transform each predicate into a question format with a placeholder "{}" for the entity.\n'
    user_0 += 'Here is the predicate:\n'
    user_0 += 'Predicate: instance of.\n'
    user_0 += 'Descriptions: that class of which this subject is a particular example and member; different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain (and an instance of volcanic landform)\n'
    user_0 += 'Aliases: is a, is an, unique individual of, unitary element of class, rdf:type, type, ∈, example of, is of type, has type, item of type\n'
    user_0 += 'Please output several templates directly and split them with \n.'
    assist_0 = 'What is {} an instance of?\nWhat type of entity is {}?\n{} is an example of what?\nWhat class does {} belong to?\nWhat category is {} a member of?\nWhat kind of thing is {}?\n{} is of what type?'
    user_1 = 'I am working on converting predicates from knowledge graphs into natural language problem templates. The goal is to transform each predicate into a question format with a placeholder "{}" for the entity.\n'
    user_1 += 'Here is the predicate:\n'
    user_1 += 'Predicate: {}.\n'.format(line['labels'])
    user_1 += 'Descriptions: {}\n'.format(line['descriptions'])
    user_1 += 'Aliases:'
    for aliase in line['aliases']:
        user_1 += ' {},'.format(aliase)
    user_1 = user_1.rstrip(',')
    user_1 += '\n'
    user_1 += 'Please output several templates directly and split them with \n.'

    message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_0},
            {"role": "assistant", "content": assist_0},
            {"role": "user", "content": user_1},
        ]

    return message

def prompt_relation_templates_sentence(line):
    system_prompt = 'You are an expert in natural language understanding and knowledge graph design. Your task is to assist in converting predicates from knowledge graphs into natural language sentence templates.'
    user_0 = 'I am working on converting predicates from knowledge graphs into natural language sentence templates. The goal is to transform each predicate into a sentence format with placeholders for the subject and object.\n'
    user_0 += 'Here is the predicate:\n'
    user_0 += 'Predicate: instance of.\n'
    user_0 += 'Descriptions: that class of which this subject is a particular example and member; different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain (and an instance of volcanic landform)\n'
    user_0 += 'Aliases: is a, is an, unique individual of, unitary element of class, rdf:type, type, ∈, example of, is of type, has type, item of type\n'
    user_0 += 'Please output the most common sentence template with the placeholders "{subject}" and "{object}".\n'
    assist_0 = '{subject} is {object}.'
    user_1 = 'I am working on converting predicates from knowledge graphs into natural language sentence templates. The goal is to transform each predicate into a sentence format with placeholders for the subject and object.\n'
    user_1 += 'Here is the predicate:\n'
    user_1 += 'Predicate: {}.\n'.format(line['labels'])
    user_1 += 'Descriptions: {}\n'.format(line['descriptions'])
    user_1 += 'Aliases:'
    for aliase in line['aliases']:
        user_1 += ' {},'.format(aliase)
    user_1 = user_1.rstrip(',')
    user_1 += '\n'
    user_1 += 'Please output the most common sentence template with the placeholders "{subject}" and "{object}".\n'

    message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_0},
            {"role": "assistant", "content": assist_0},
            {"role": "user", "content": user_1},
        ]

    return message

dataset_path = '/data/xkliu/LLMs/DocFixQA/datasets/MMLU/data'
input_path = '/data/xkliu/LLMs/DocFixQA/datasets/hotpotQA'
input_dir = 'example_rag_top6'
output_path = '/data/xkliu/LLMs/DocFixQA/datasets/hotpotQA'
output_dir = 'example_rag_top6_reason'
model_name = 'gpt-4o-mini'

input_file_name = os.path.join(input_path, 'example_rag_top6_reason.json')
input_file = open(input_file_name)
output_file_name = os.path.join(output_path, 'example_generate_question.json')

processed_lines = set()
if os.path.exists(output_file_name):
    with open(output_file_name, 'r', encoding='utf-8') as output_file:
        for line in output_file:
            data = json.loads(line)
            processed_lines.add(data['id'])  # Assuming 'id' is the unique identifier for each entry

output_file = open(output_file_name, 'a', encoding='utf-8')

batch_num = 3
batch = []
src_line = []
for line in tqdm(input_file):
    line = json.loads(line)
    if line['id'] in processed_lines:
        continue
    # message = prompt_fomular_reference_generate(line, have_choice=False, add_entity=True)
    # message = reason_message_with_references(line, have_choice=False)
    # message = prompt_fomular_kg_local_check(line, have_choice=True)
    # message = prompt_fomular_triple_extraction(line)
    # message = prompt_relation_templates(line)
    # message = prompt_relation_templates_sentence(line)
    message = prompt_generate_question(line)

    # for m in message:
    #     print(m['content'] + '\n')

    '''one'''
    # llm_response = chat_with_openai(message, max_tokens=1024)

    # llm_response = llm_response.lstrip('Reasoning:')
    # llm_response = llm_response.strip()
    # # print(llm_response)

    # line['reason'] = llm_response
    # output_file.write(json.dumps(line, ensure_ascii=False) + '\n')

    '''batch'''
    batch.append(message)
    src_line.append(line)

    if len(batch) >= batch_num:
        llm_response = asyncio.run(run_batch(batch, model=model_name))
        for l, r in zip(src_line, llm_response):
            l['new_question'] = r.strip()
            output_file.write(json.dumps(l, ensure_ascii=False) + '\n')
        batch = []
        src_line = []

if len(batch) > 0:
    llm_response = asyncio.run(run_batch(batch, model=model_name))
    for l, r in zip(src_line, llm_response):
        l['new_question'] = r.strip()
        output_file.write(json.dumps(l, ensure_ascii=False) + '\n')


exit()
'''MMLU'''

#load src dir
subjects = sorted([f.split("_dev.json")[0] for f in os.listdir(os.path.join(dataset_path, "dev")) if "_dev.json" in f])

# mkdir save dir
save_dir = os.path.join(output_path, 'dev', output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

exsit_sub = sorted([f.split("_dev.json")[0] for f in os.listdir(os.path.join(output_path, 'dev', output_dir)) if "_dev.json" in f])

for sub in subjects:
    if sub in exsit_sub:
        continue
    print(sub)
    dev_file_name = os.path.join(input_path, "dev", input_dir , sub + "_dev.json")
    dev_file = open(dev_file_name)

    output_file_name = os.path.join(save_dir, "{}_dev.json".format(sub))
    output_file = open(output_file_name, 'w')

    for line in tqdm(dev_file):
        line = json.loads(line)
        message = prompt_fomular_reference_generate(line, have_choice=True, add_entity=True)
        # message = reason_message_with_references(line)
        # message = prompt_fomular_kg_local_check(line, have_choice=True)
        # message = prompt_fomular_triple_extraction(line)

        # for m in message:
        #     print(m['content'] + '\n')

        llm_response = chat_with_openai(message, max_tokens=1024)

        llm_response = llm_response.lstrip('Reasoning:')
        llm_response = llm_response.strip()
        # print(llm_response)

        line['llm_response'] = llm_response
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')
        # exit()
