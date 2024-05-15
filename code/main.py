from LLM_calls import load_llm, llm_call
from utils import read_data
import random
import json
from tqdm import tqdm

def prompt_fomular(line:dict, dataset, shuffle=True):
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
        
if __name__ == '__main__':
    # messages = [
    #     {"role": "user", "content": "What is your favourite condiment?"},
    #     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    #     {"role": "user", "content": "Do you have mayonnaise recipes?"}
    # ]
    # model_name = 'Mistral'
    # '''Model & Tokenizer'''
    # model, tokenizer = load_llm(model_name, '/data/share_weight/mistral-7B-v0.2-instruct')
    # response = llm_call(messages, model_name, model=model, tokenizer=tokenizer)
    # print(response)
    '''Pipeline'''
    # pipeline = load_llm(model_name, '/data/share_weight/Meta-Llama-3-8B-Instruct')
    # response = llm_call(messages, model_name, pipeline=pipeline)
    # print(response)
    dataset = 'Truthful_QA'
    dataset_path = '/data/xkliu/LLMs/DocFixQA/datasets/truthfulqa_mc_task.json'
    model_name = 'Llama'
    output_file = open('result/Llama3_8B_raw.json', 'w')

    if model_name == 'Mistral':
        model, tokenizer = load_llm(model_name, '/data/share_weight/mistral-7B-v0.2-instruct')
    elif model_name == 'Llama':
        pipeline = load_llm(model_name, '/data/share_weight/Meta-Llama-3-8B-Instruct')

    data = read_data(dataset, dataset_path)
    for line in tqdm(data):
        prompt = prompt_fomular(line, dataset)
        # print('-'*50 + 'PROMPT' + '-'*50)
        # print(prompt)
        messages = [{"role": "user", "content": prompt}]
        if model_name == 'Mistral':
            response = llm_call(messages, model_name, model=model, tokenizer=tokenizer)
        elif model_name == 'Llama':
            response = llm_call(messages, model_name, pipeline=pipeline)
        line['llm_response'] = response
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')
        # print('-'*50 + 'RESPONSE' + '-'*50)
        # print(response)
        # if i >= 5:
        #     break
