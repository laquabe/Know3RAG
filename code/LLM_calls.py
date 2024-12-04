from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import transformers
import torch
from transformers.generation.utils import GenerationConfig  #baichuan
# device = "cuda" the device to load the model onto


def load_llm(model_name, model_path, logit=False):
    if model_name == 'Mistral':
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto')
        return model, tokenizer
    elif model_name == 'Llama':
        if logit:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto')
            return model, tokenizer
        else:
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_path,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map='auto',
                )
            return pipeline
    elif model_name == 'GLM3':
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, device_map='auto', trust_remote_code=True).half().cuda()
        model = model.eval()
        return model, tokenizer
    elif model_name == 'Baichuan':
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        model = model.eval()
        return model, tokenizer
    elif model_name == 'Yi':
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto").eval()
        return model, tokenizer
    elif model_name == 'Qwen':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto").eval()
        return model, tokenizer
    elif model_name == 'GLM4':
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to('cuda').eval()
        return model, tokenizer
    elif model_name == 'Zephyr':
        pipe = transformers.pipeline("text-generation", model=model_path, torch_dtype=torch.bfloat16, device_map="auto")
        return pipe
    else:
        print('Error! No support models')
    print('Model load sucessfully!')

def llm_call(messages, model_name, model=None, tokenizer=None, pipeline=None, do_sample=False, max_new_tokens=1024, output_logit=False, logit_topk=100):
    if model_name == 'Mistral':
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
        generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        decoded = tokenizer.batch_decode(generated_ids)
        response = decoded[0]
        res_pos = response.find('[/INST]')
        response = response[res_pos + len('[/INST]'):]
        response = response.strip()
        return response
    elif model_name == 'Llama':
        if output_logit:
            input_ids = tokenizer(messages[-1]['content'], return_tensors="pt").input_ids.to('cuda')
            # 禁用梯度计算
            with torch.no_grad():
                # 使用 generate 方法并返回所有生成步骤的 logits
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=do_sample
                )
            # 获取最后生成 token 的 logits
            logits = outputs.scores[0] # ?

            # 计算概率并选出前100个最高的
            probs = torch.softmax(logits, dim=-1)
            # top_probs, top_indices = torch.topk(probs, 100)
            # for i in range(100):
            #     print('{}:{}'.format(top_indices[0][i], tokenizer.decode(top_indices[0][i], skip_special_tokens=True)))

            # 获取候选词及其对应的 token IDs
            candidate_tokens = ["A", "B", "C", "D"]
            candidate_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(option)[0]) for option in candidate_tokens]
            # print(candidate_ids)
            
            # 筛选候选项的概率
            # candidate_probs = {candidate_tokens[i]: probs[0, candidate_id].item() for i, candidate_id in enumerate(candidate_ids) if candidate_id in top_indices}
            candidate_probs = {candidate_tokens[i]: probs[0, candidate_id].item() for i, candidate_id in enumerate(candidate_ids)}

            # 找出概率最大的候选项或返回 None
            max_option = max(candidate_probs, key=candidate_probs.get) if candidate_probs else None

            # 输出结果
            # print("Probabilities (Top 100):", candidate_probs)
            # print("Selected Option:", max_option)
            return max_option

        else:
            prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            if do_sample:
                outputs = pipeline(prompt, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=do_sample, temperature=0.6, top_p=0.9)
            else:
                outputs = pipeline(prompt, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=do_sample)
        
            return outputs[0]["generated_text"][len(prompt):]

    elif model_name == 'GLM3':
        message = messages[-1]['content']
        history = messages[:-1]
        input_length = len(tokenizer.build_chat_input(message, history=history)['input_ids'][0])
        response, history = model.chat(tokenizer, message, history=history, do_sample=do_sample, max_length = (input_length + max_new_tokens))
        return response
    elif model_name == 'Baichuan':
        response = model.chat(tokenizer, messages)
        return response
    elif model_name == 'Yi':
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id, do_sample=do_sample, max_new_tokens=max_new_tokens)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response
    elif model_name == 'Qwen':
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    elif model_name == 'GLM4':
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
        input_length = len(inputs['input_ids'][0])
        inputs = inputs.to('cuda')
        gen_kwargs = {"max_length": input_length + max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs['top_k'] = 1
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
        
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif model_name == 'Zephyr':
        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if do_sample:
            outputs = pipeline(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        else:
            outputs = pipeline(prompt, max_new_tokens=1024, do_sample=True)

        gen_text = outputs[0]["generated_text"]
        gen_start_pos = gen_text.rfind('<|assistant|>')  # zephyr
        gen_text = gen_text[gen_start_pos:]
        gen_text = gen_text.lstrip('<|assistant|>').strip()
        
        return gen_text
    else:
        print('Error! No models use')

if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Read the following question and provide only the correct option letter (e.g., A, B, C, or D) without adding any additional text.\n\n Question: Where is London?\nA. China\n B. America\n C. England \n D.France \n Answer:"},
    ]
    model_name = 'Llama'
    '''Mistarl'''
    # model, tokenizer = load_llm(model_name, '')
    # response = llm_call(messages, model_name, model=model, tokenizer=tokenizer)
    # print(response)
    '''Llama'''
    model, tokenizer = load_llm(model_name, '/data/share_weight/Meta-Llama-3-8B-Instruct', logit=True)
    response = llm_call(messages, model_name, model=model, tokenizer=tokenizer, output_logit=True)
    print(response)
    '''GLM3'''
    # model, tokenizer = load_llm(model_name, '/data/xkliu/LLMs/models/chatglm3-6b')
    # response = llm_call(messages, model_name, model=model, tokenizer=tokenizer)
    # print(response)
    '''Baichuan'''
    # model, tokenizer = load_llm(model_name, '/data/xkliu/LLMs/models/Baichuan2-7B-Chat')
    # response = llm_call(messages, model_name, model=model, tokenizer=tokenizer)
    # print(response)
    '''Qwen'''
    # model, tokenizer = load_llm(model_name, '/data/xkliu/LLMs/models/Qwen2-7B-Instruct')
    # response = llm_call(messages, model_name, model=model, tokenizer=tokenizer, do_sample=True)
    # print(response)
    '''Yi'''
    # model, tokenizer = load_llm(model_name, '/data/xkliu/LLMs/models/Yi-1.5-9B-Chat')
    # response = llm_call(messages, model_name, model=model, tokenizer=tokenizer, do_sample=True)
    # print(response)
    '''GLM4'''
    # model, tokenizer = load_llm(model_name, '/data/xkliu/LLMs/models/glm-4-9b-chat')
    # response = llm_call(messages, model_name, model=model, tokenizer=tokenizer, do_sample=True)
    # print(response)