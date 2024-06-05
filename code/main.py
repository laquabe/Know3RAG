from LLM_calls import load_llm, llm_call
from utils import read_data
import random
import json
from tqdm import tqdm
import argparse
'''
Sure, here's a prompt you can give to your friend for the question-answer task:

---

**Prompt:**

You are tasked with a question-answer task. For each question, you need to provide the reasoning behind your answer and then output the answer in a JSON format. The JSON should include the following fields: "reason" and "answer".

1. **Reasoning**: Write a detailed explanation of how you arrived at your answer.
2. **Answer**: Provide the final answer based on your reasoning.

If you encounter a question that requires additional external information that you don't currently have, you must state in your reasoning what external knowledge is required. In such cases, the answer field should be: "I need external knowledge."

Here is an example of the expected output format:

```json
{
  "reason": "I need information about the historical significance of the event to answer this question accurately.",
  "answer": "I need external knowledge."
}
```

Please ensure that each response follows this format.

**Example:**

**Question:** What is the capital of France?

**Response:**

```json
{
  "reason": "The capital city of France is a well-known fact. Paris is the most populous city and has been the capital since the 10th century.",
  "answer": "Paris"
}
```

**Question:** What was the impact of the Treaty of Versailles on Germany?

**Response:**

```json
{
  "reason": "To accurately discuss the impact of the Treaty of Versailles on Germany, I need detailed information on the economic, political, and social consequences it had on the country post-World War I.",
  "answer": "I need external knowledge."
}
```

Use this format for all responses. 

---

This prompt should guide your friend to provide the necessary reasoning and format the answers correctly.

'''


def prompt_fomular(line:dict, dataset, model=None, shuffle=True, extral_ask=True):
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
        if extral_ask:
            content += 'If you encounter a question that requires additional external information that you don\'t currently have, you must state in your reasoning what external knowledge is required. In such cases, the answer field should be: "I need external knowledge.".Here is an example of the expected output format:\n'
            content += '{"reason": "I need information about the historical significance of the event to answer this question accurately.","answer": "I need external knowledge."}\n'
        content += 'Here are some examples of how you should respond:\n'
        content += '**Question:** What is the capital of France?\n'
        content += '**Response:**\n{"reason": "France\'s capital city, Paris, is widely recognized and documented in various reliable sources including encyclopedias and official government websites.","answer": "Paris"}\n'
        if extral_ask:
            content += '**Question:** What was the impact of the Treaty of Versailles on Germany?\n'
            content += '**Response:**\n{"reason": "To accurately discuss the impact of the Treaty of Versailles on Germany, I need detailed information on the economic, political, and social consequences it had on the country post-World War I.","answer": "I need external knowledge."}\n\n'
        content += 'Answer the following questions using the format and guidelines provided above.\n**Question:**{}\n**Response:**'.format(line['Question'])

        return content
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocFixQA args')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help="Dataset Name")
    parser.add_argument('--model_name', '-m', type=str, required=True, help='Model Name')
    parser.add_argument('--exp_name','-e',type=str, required=True, default='test', help='Exp Name')
    parser.add_argument('--model_path','-p',type=str, required=True, help="Path to model")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    exp_name = args.exp_name
    full_flag = False

    dataset = '{}_QA'.format(dataset_name)
    dataset_path = '/data/xkliu/LLMs/DocFixQA/datasets/{}QA/dev.json'.format(dataset_name)
    output_file_name = 'result/{}QA/{}/{}.json'.format(dataset_name, model_name, exp_name)
    output_file = open(output_file_name, 'w')
    
    assert model_name.lower() in args.model_path.lower()
    if model_name == 'Mistral':
        model, tokenizer = load_llm(model_name, args.model_path)
    elif model_name == 'Llama':
        pipeline = load_llm(model_name, args.model_path)

    data = read_data(dataset, dataset_path)
    if full_flag:
        for line in tqdm(data):
            prompt = prompt_fomular(line, dataset, model=model_name, extral_ask=False)
            messages = [{"role": "user", "content": prompt}]
            if model_name == 'Mistral':
                response = llm_call(messages, model_name, model=model, tokenizer=tokenizer)
            elif model_name == 'Llama':
                response = llm_call(messages, model_name, pipeline=pipeline)
            line['llm_response'] = response
            output_file.write(json.dumps(line, ensure_ascii=False) + '\n')
    else:
        for i, line in enumerate(data):
            prompt = prompt_fomular(line, dataset, model=model_name, extral_ask=False)
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
            if i >= 1:
                break
