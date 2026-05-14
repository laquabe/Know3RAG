"""Local HuggingFace LLM loading and inference utilities.

This module is the refactored replacement for the legacy ``code/LLM_calls.py``
local-model path. New framework code should import from here instead of the
legacy ``code`` directory.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def load_local_llm(model_name: str, model_path: str, logit: bool = False):
    """Load a supported local HuggingFace chat model.

    Supported model names follow the legacy names used by the project:
    Mistral | Llama | Qwen | GLM3 | GLM4 | Baichuan | Yi | Zephyr.
    """
    import torch
    import transformers
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from transformers.generation.utils import GenerationConfig

    if model_name == 'Mistral':
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto')
        return model, tokenizer

    if model_name == 'Llama':
        if logit:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto')
            return model, tokenizer
        return transformers.pipeline(
            'text-generation',
            model=model_path,
            model_kwargs={'torch_dtype': torch.bfloat16},
            device_map='auto',
        )

    if model_name == 'GLM3':
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, device_map='auto', trust_remote_code=True).half().cuda()
        return model.eval(), tokenizer

    if model_name == 'Baichuan':
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            device_map='auto',
            use_fast=False,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        return model.eval(), tokenizer

    if model_name == 'Yi':
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype='auto',
        ).eval()
        return model, tokenizer

    if model_name == 'Qwen':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map='auto',
        ).eval()
        return model, tokenizer

    if model_name == 'GLM4':
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to('cuda').eval()
        return model, tokenizer

    if model_name == 'Zephyr':
        return transformers.pipeline(
            'text-generation',
            model=model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )

    raise ValueError(f'Unsupported local model_name: {model_name}')


def call_local_llm(
    messages: List[Dict[str, str]],
    model_name: str,
    model=None,
    tokenizer=None,
    pipeline=None,
    do_sample: bool = False,
    max_new_tokens: int = 1024,
    output_logit: bool = False,
    logit_topk: int = 100,
) -> str:
    """Run inference for a supported local HuggingFace chat model."""
    import torch

    if model_name == 'Mistral':
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors='pt').to('cuda')
        generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        decoded = tokenizer.batch_decode(generated_ids)
        response = decoded[0]
        res_pos = response.find('[/INST]')
        response = response[res_pos + len('[/INST]'):]
        return response.strip()

    if model_name == 'Llama':
        if output_logit:
            input_ids = tokenizer(messages[-1]['content'], return_tensors='pt').input_ids.to('cuda')
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=do_sample,
                )
            logits = outputs.scores[0]
            probs = torch.softmax(logits, dim=-1)
            candidate_tokens = ['A', 'B', 'C', 'D']
            candidate_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(option)[0]) for option in candidate_tokens]
            candidate_probs = {
                candidate_tokens[i]: probs[0, candidate_id].item()
                for i, candidate_id in enumerate(candidate_ids)
            }
            return max(candidate_probs, key=candidate_probs.get) if candidate_probs else ''

        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
        ]
        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'eos_token_id': terminators,
            'do_sample': do_sample,
        }
        if do_sample:
            generation_kwargs.update({'temperature': 0.6, 'top_p': 0.9})
        outputs = pipeline(prompt, **generation_kwargs)
        return outputs[0]['generated_text'][len(prompt):]

    if model_name == 'GLM3':
        message = messages[-1]['content']
        history = messages[:-1]
        input_length = len(tokenizer.build_chat_input(message, history=history)['input_ids'][0])
        response, _ = model.chat(
            tokenizer,
            message,
            history=history,
            do_sample=do_sample,
            max_length=input_length + max_new_tokens,
        )
        return response

    if model_name == 'Baichuan':
        return model.chat(tokenizer, messages)

    if model_name == 'Yi':
        input_ids = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            return_tensors='pt',
        )
        output_ids = model.generate(
            input_ids.to('cuda'),
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )
        return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    if model_name == 'Qwen':
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors='pt').to('cuda')
        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': do_sample,
        }
        if not do_sample:
            generation_kwargs.update({'temperature': None, 'top_p': None, 'top_k': None})
        generated_ids = model.generate(model_inputs.input_ids, **generation_kwargs)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if model_name == 'GLM4':
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors='pt',
            return_dict=True,
        )
        input_length = len(inputs['input_ids'][0])
        inputs = inputs.to('cuda')
        gen_kwargs = {'max_length': input_length + max_new_tokens, 'do_sample': do_sample}
        if do_sample:
            gen_kwargs['top_k'] = 1
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    if model_name == 'Zephyr':
        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if do_sample:
            outputs = pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
        else:
            outputs = pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        gen_text = outputs[0]['generated_text']
        gen_start_pos = gen_text.rfind('<|assistant|>')
        return gen_text[gen_start_pos:].lstrip('<|assistant|>').strip()

    raise ValueError(f'Unsupported local model_name: {model_name}')