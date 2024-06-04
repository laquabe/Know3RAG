from LLM_calls import load_llm, llm_call
from utils import read_data
import random
import json
from tqdm import tqdm
'''
Sure! Here's a sample prompt you can use to instruct a chatbot to provide answers in the specified JSON format:

---

You are a highly intelligent and helpful assistant. When asked a question, you will provide the answer in a JSON format with the key named "answer". Here is the structure you should use:

```json
{
  "answer": "Your response here"
}
```

For example, if asked "What is the capital of France?", you should respond with:

```json
{
  "answer": "The capital of France is Paris."
}
```

Let's begin!

'''


def prompt_fomular(line:dict, dataset, model=None,shuffle=True):
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
        content = 'You are a highly intelligent and helpful assistant. When asked a question, you will provide the answer in a JSON format with the key named "answer". Here is the structure you should use:\n'
        content += '{"answer": "Your response here"}\n'
        content += 'For example, if asked "who is the first husband of julia roberts?", you should respond with:\n'
        content += '{"answer": "Lyle Lovett"}\n'
        content += 'Here is the question:'
        content += 'Question: {}\n'.format(line['Question'])

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
    dataset = 'Temporal_QA'
    dataset_path = '/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/dev.json'
    model_name = 'Mistral'
    output_file = open('result/TemporalQA/{}_raw.json'.format(model_name), 'w')
    full_flag = False
    if model_name == 'Mistral':
        model, tokenizer = load_llm(model_name, '/data/share_weight/mistral-7B-v0.2-instruct')
    elif model_name == 'Llama':
        pipeline = load_llm(model_name, '/data/share_weight/Meta-Llama-3-8B-Instruct')

    data = read_data(dataset, dataset_path)
    if full_flag:
        for line in tqdm(data):
            prompt = prompt_fomular(line, dataset, model=model_name)
            messages = [{"role": "user", "content": prompt}]
            if model_name == 'Mistral':
                response = llm_call(messages, model_name, model=model, tokenizer=tokenizer)
            elif model_name == 'Llama':
                response = llm_call(messages, model_name, pipeline=pipeline)
            line['llm_response'] = response
            output_file.write(json.dumps(line, ensure_ascii=False) + '\n')
    else:
        for i, line in enumerate(data):
            prompt = prompt_fomular(line, dataset, model=model_name)
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
            if i >= 5:
                break
