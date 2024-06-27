from LLM_calls import load_llm, llm_call
from utils import read_data
import random
import json
from tqdm import tqdm
import argparse
'''
Sure! Here's a prompt you can send to your friend:

---

Hi [Friend's Name],

I have a problem that I need to break down into sub-problems, and I need your help. Could you please extract the key entities from the problem and come up with one piece of information that needs to be collected for each entity? You don't need to answer these questions; just identify what information should be gathered.

To give you a clearer idea, here's an example problem and how it should be broken down:

**Example Problem:**
"Who was the current President of the United States when Zootopia was released?"

**Entities and Information Needed:**
```json
{
  "President of the United States": "Name of the President",
  "Zootopia": "Release date of the movie"
}
```

Here's the problem I need your help with:

**Problem:**
[Insert your problem here]

Could you please provide the output in a similar JSON format?

Thank you so much for your help!

Best,
[Your Name]

---

Feel free to customize the message further as needed!
'''
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
    content += '*Question:* {}\n'.format(line['sub_question'])
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

def external_knowledge_prompt(line, src_key):
    ex_know_list = line[src_key]
    # if src_key == 'passages':
    #     ex_know_list = ex_know_list[0][0]
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
        if extral_ask:
            content += 'If you encounter a question that requires additional external information that you don\'t currently have, you must state in your reasoning what external knowledge is required. In such cases, the answer field should be: "I need external knowledge.".Here is an example of the expected output format:\n'
            content += '{"reason": "I need information about the historical significance of the event to answer this question accurately.","answer": "I need external knowledge."}\n'
        content += 'Here are some examples of how you should respond:\n'
        content += '**Question:** What is the capital of France?\n'
        content += '**Response:**\n{"reason": "France\'s capital city, Paris, is widely recognized and documented in various reliable sources including encyclopedias and official government websites.","answer": "Paris"}\n'
        if extral_ask:
            content += '**Question:** What was the impact of the Treaty of Versailles on Germany?\n'
            content += '**Response:**\n{"reason": "To accurately discuss the impact of the Treaty of Versailles on Germany, I need detailed information on the economic, political, and social consequences it had on the country post-World War I.","answer": "I need external knowledge."}\n\n'
        if rag and (len(line[src_key]) != 0):
            content += '\nNow, before you answer the question, you can read the following references to ensure your answers are accurate. It is worth noting that when there is no relevant information in the reference, you can rely on the knowledge you have to answer the question:\n\n'
            content += external_knowledge_prompt(line, src_key)
        
        content += 'Answer the following questions using the format and guidelines provided above.\n**Question:**{}\n**Response:**'.format(line['Question'])

        return content
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocFixQA args')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help="Dataset Name")
    parser.add_argument('--dataset_path', type=str, help="Dataset Path", default=None)
    parser.add_argument('--model_name', '-m', type=str, required=True, help='Model Name')
    parser.add_argument('--exp_name','-e',type=str, required=True, default='test', help='Exp Name')
    parser.add_argument('--model_path','-p',type=str, required=True, help="Path to model")
    parser.add_argument('--test', action='store_true', help="if Test", default=None)
    parser.add_argument('--extral_ask', action='store_true', help="if Self Ask", default=None)
    parser.add_argument('--rag', action='store_true', help="if Rag", default=None)
    parser.add_argument('--exinfo_judge', action='store_true', help="if External Information Filter by Pair", default=None)
    parser.add_argument('--line', action='store_true', help="if Process by line", default=None)
    parser.add_argument('--summary', action='store_true', help="Summary Process", default=None)
    parser.add_argument('--decompose', action='store_true', help="Decompose the Question into Subqustion")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    exp_name = args.exp_name
    full_flag = False if args.test else True
    extral_ask = True if args.extral_ask else False
    rag_flag = True if args.rag else False
    exinfo_judge = True if args.exinfo_judge else False
    line_flag = True if args.line else False
    summary_flag = True if args.summary else False
    decompose_flag = True if args.decompose else False

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
            else:
                prompt = prompt_fomular(line, dataset, model=model_name, extral_ask=extral_ask, rag=rag_flag)

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
            else:
                prompt = prompt_fomular(line, dataset, model=model_name, extral_ask=extral_ask, rag=rag_flag)

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
            if i >= 10:
                break
