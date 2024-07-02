import transformers
import json
from tqdm import tqdm
from utils import read_data

card_name = 'knowledge-card-wikipedia'
card_path = '/data/xkliu/LLMs/Knowledge_Card-main/cards/{}'.format(card_name)
card_device = 4
k = 5

card = transformers.pipeline('text-generation', model=card_path, device = card_device, num_return_sequences=k, do_sample=True, max_new_tokens = 100)
data = read_data('Temporal_QA',
                 '/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/dev.json')
output_f = open('/data/xkliu/LLMs/DocFixQA/result/TemporalQA/cards/{}.json'.format(card_name), 'w')


for line in tqdm(data):
    prompt = line['Question']
    knowl = card(prompt)
    knowl = [obj["generated_text"][len(prompt)+1:] for obj in knowl]
    line['generate'] = knowl
    output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    