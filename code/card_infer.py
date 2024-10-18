import transformers
import json
from tqdm import tqdm
from utils import read_data

max_new_tokens = 128    # default 100
task = 'entity'
card_name = 'knowledge-card-1btokens'   # knowledge-card-1btokens, knowledge-card-atomic, knowledge-card-reddit, knowledge-card-wikidata, knowledge-card-wikipedia
card_path = '/data/xkliu/LLMs/Knowledge_Card-main/cards/{}'.format(card_name)
card_device = 3
k = 3

card = transformers.pipeline('text-generation', model=card_path, device = card_device, num_return_sequences=k, do_sample=True, max_new_tokens = max_new_tokens)
data = open('/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/pseudo_doc_generate_question_entity.json')
output_f = open('/data/xkliu/LLMs/DocFixQA/result/TemporalQA/cards/{}_{}.json'.format(task, card_name), 'w')


for line in tqdm(data):
    line = json.loads(line.strip())
    if task == 'entity':
        for ent in line['question_entity'].values():
            prompt = '{}, {}'.format(ent['entity'], ent['description'])
            knowl = card(prompt)
            knowl = [obj["generated_text"][len(prompt)+1:] for obj in knowl]
            line['generate'] = knowl
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
        continue
    
    if task == 'pseduo':
        prompt = line['pseudo_doc']
    if task == 'summary':
        prompt = line['summary']
    else:
        prompt = line['Question']
    knowl = card(prompt)
    knowl = [obj["generated_text"][len(prompt)+1:] for obj in knowl]
    line['generate'] = knowl
    output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

    # for ent, prompt in line['sub_questions'].items():
    #     knowl = card(prompt)
    #     knowl = [obj["generated_text"][len(prompt)+1:] for obj in knowl]
    #     line['generate'] = knowl
    #     output_f.write(json.dumps(line, ensure_ascii=False) + '\n') 
    
    