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


def call_local_llm_batch(
    batch: List[List[Dict[str, str]]],
    model_name: str,
    model=None,
    tokenizer=None,
    pipeline=None,
    do_sample: bool = False,
    max_new_tokens: int = 1024,
    batch_size: int = 1,
    batch_events: Optional[List[Dict]] = None,
) -> List[str]:
    """True padded GPU batches for Llama/GLM4; preserve single-call behavior.

    Sort only within this call and scatter outputs back to the original order.
    CUDA OOM retries split the affected batch, never truncate its inputs.
    """
    import gc
    import logging

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not batch:
        return []
    events = batch_events if batch_events is not None else []

    def single(messages):
        return call_local_llm(
            messages, model_name, model=model, tokenizer=tokenizer, pipeline=pipeline,
            do_sample=do_sample, max_new_tokens=max_new_tokens,
        )

    if batch_size == 1 or model_name not in ("Llama", "GLM4"):
        results = []
        for messages in batch:
            results.append(single(messages))
            events.append({"batch_size": 1, "status": "complete"})
        return results

    import torch
    tok = pipeline.tokenizer if model_name == "Llama" else tokenizer
    if model_name == "Llama":
        encoded = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in batch]
        lengths = [len(tok(text)["input_ids"]) for text in encoded]
    else:
        # Use the original GLM chat template; do not tokenize its rendered text a
        # second time, which could duplicate model-specific special tokens.
        encoded = [tok.apply_chat_template(
            m, add_generation_prompt=True, tokenize=True, return_dict=True,
        ) for m in batch]
        lengths = [len(item["input_ids"]) for item in encoded]
    order = sorted(range(len(batch)), key=lambda i: (lengths[i], i))
    results = [""] * len(batch)

    old_side, old_pad = tok.padding_side, tok.pad_token_id
    if old_pad is None and tok.eos_token_id is None:
        raise ValueError("Batch generation requires a pad token or an EOS token")
    tok.padding_side = "left"
    if old_pad is None:
        tok.pad_token_id = tok.eos_token_id

    def generate(indices):
        if len(indices) == 1:
            return [single(batch[indices[0]])]
        if model_name == "Llama":
            prompts = [encoded[i] for i in indices]
            kwargs = dict(
                batch_size=len(indices), num_workers=0, max_new_tokens=max_new_tokens,
                eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")],
                pad_token_id=tok.pad_token_id, do_sample=do_sample,
            )
            if do_sample:
                kwargs.update(temperature=0.6, top_p=0.9)
            outputs = pipeline(prompts, **kwargs)
            if len(outputs) != len(prompts):
                raise ValueError("Model returned the wrong number of batch outputs")
            return [output[0]["generated_text"][len(prompt):]
                    for prompt, output in zip(prompts, outputs)]
        inputs = tok.pad([encoded[i] for i in indices], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        width = inputs["input_ids"].shape[1]
        kwargs = dict(max_length=width + max_new_tokens, do_sample=do_sample,
                      pad_token_id=tok.pad_token_id)
        if do_sample:
            kwargs["top_k"] = 1
        with torch.no_grad():
            outputs = model.generate(**inputs, **kwargs)
        return tok.batch_decode(outputs[:, width:], skip_special_tokens=True)

    def run_group(indices):
        try:
            outputs = generate(indices)
            if len(outputs) != len(indices):
                raise ValueError("Model returned the wrong number of batch outputs")
            for index, output in zip(indices, outputs):
                results[index] = output
            events.append({"batch_size": len(indices), "status": "complete"})
            return
        except RuntimeError as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in str(exc).lower()
            if not is_oom or len(indices) == 1:
                raise
            events.append({"batch_size": len(indices), "status": "cuda_oom"})
        # Leave the except block before clearing memory: its traceback may retain
        # the tensors from the failed generate call.
        gc.collect()
        torch.cuda.empty_cache()
        logging.warning("CUDA OOM: splitting batch of %d", len(indices))
        middle = len(indices) // 2
        run_group(indices[:middle])
        run_group(indices[middle:])

    try:
        for start in range(0, len(order), batch_size):
            run_group(order[start:start + batch_size])
    finally:
        tok.padding_side = old_side
        if old_pad is None:
            tok.pad_token_id = old_pad
    return results


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
