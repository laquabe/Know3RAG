from LLM_calls import load_llm, llm_call

if __name__ == '__main__':
    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    model_name = 'Mistral'
    '''Model & Tokenizer'''
    model, tokenizer = load_llm(model_name, '/data/share_weight/mistral-7B-v0.2-instruct')
    response = llm_call(messages, model_name, model=model, tokenizer=tokenizer)
    print(response)
    '''Pipeline'''
    # pipeline = load_llm(model_name, '/data/share_weight/Meta-Llama-3-8B-Instruct')
    # response = llm_call(messages, model_name, pipeline=pipeline)
    # print(response)