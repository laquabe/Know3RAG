from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
# device = "cuda" the device to load the model onto


def load_llm(model_name, model_path):
    if model_name == 'Mistral':
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    elif model_name == 'Llama':
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
            )
        return pipeline
    else:
        print('Error! No support models')

def llm_call(messages, model_name, model=None, tokenizer=None, pipeline=None):
    if model_name == 'Mistral':
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        
        return decoded[0]
    if 
    else:
        print('Error! No models use')

if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

