import openai
import asyncio
from tqdm import tqdm
import json
import aiohttp



def chat_with_openai(messages, max_tokens=200):

    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=messages,
        max_tokens=max_tokens   
    )

    # print(response)
    return response.choices[0].message.content.strip()


async def fetch_openai_response(messages, model="gpt-4o-mini", max_tokens=1024, timeout=30):
    if 'gpt' in model:
        try:
            response = await asyncio.wait_for(asy_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens
            ), timeout)
        except:
            print(f"Request for model {model} error.")
            return ''  # Or handle the timeout gracefully
    elif 'qwen' in model:
        try:
            response = await asyncio.wait_for(asy_client_qwen.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens
            ), timeout)
        except:
            print(f"Request for model {model} error.")
            return '' # Or handle the timeout gracefully
    else:
        print('not support model')
        exit()

    return response.choices[0].message.content.strip()

# Function to run a batch of API requests concurrently
async def run_batch(prompts, model="gpt-4o-mini", max_tokens=1024):
    # Create a list of asynchronous tasks for each prompt
    tasks = [fetch_openai_response(prompt, model, max_tokens) for prompt in prompts]
    
    # Use asyncio.gather() to run the tasks concurrently
    responses = await asyncio.gather(*tasks)
    
    return responses

# Main function to orchestrate the batch execution
async def batch_run():
    prompts = [
        [{"role":"user", "content":"Tell me a joke."}],
        [{"role":"user", "content":"What is the capital of France?"}],
    ]
    
    # Run the batch of requests
    responses = await run_batch(prompts, model='qwen2.5-32b-instruct')

    for i, response in enumerate(responses):
        print(f"Response {i + 1}: {response}")
        
    return responses
    
def run_file(input_file_name, output_file_name, model_name, res_key='llm_response', test=True, batch_size=10):
    with open(input_file_name) as input_file, \
        open(output_file_name, 'w') as output_file:
        batch = []
        src_line = []
        error_num = 0
        for line in tqdm(input_file):
            line = json.loads(line)
            message = line[res_key]

            '''batch'''
            batch.append(message)
            src_line.append(line)

            if len(batch) >= batch_size:
                llm_response = asyncio.run(run_batch(batch, model=model_name))
                for l, r in zip(src_line, llm_response):
                    l[res_key] = r
                    if r == '':
                        error_num += 1
                    output_file.write(json.dumps(l, ensure_ascii=False) + '\n')
                batch = []
                src_line = []

                if test:
                    print(r)
                    break

        if len(batch) > 0:
            llm_response = asyncio.run(run_batch(batch, model=model_name))
            for l, r in zip(src_line, llm_response):
                l[res_key] = r
                if r == '':
                    error_num += 1
                output_file.write(json.dumps(l, ensure_ascii=False) + '\n')

        print('Bad Request: {}'.format(error_num))

if __name__ == "__main__":
    # user_input = [{"role":"user", "content":"hello"}]
    # reply = chat_with_openai(user_input)
    # print("R：", reply)

    # asyncio.run(batch_run())

    run_file(
        '/data/xkliu/LLMs/DocFixQA/result/hotpotQA/api/rag.json',
        'result/tmp.json',
        'gpt-4o-mini',
        batch_size=1
    )
