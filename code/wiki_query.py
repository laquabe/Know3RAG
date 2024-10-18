import requests
import json
from tqdm import tqdm

import spacy  # version 3.5

# initialize language model
nlp = spacy.load("en_core_web_md")
# add pipeline (declared through entry_points in setup.py)
nlp.add_pipe("entityLinker", last=True)


def query_entity(entity_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

    response = requests.get(url)
    data = response.json()

    # 打印实体的标签和描述
    entity_data = data['entities'][entity_id]


    info_dict = {
        'labels': entity_data['labels']['en']['value'],
        'descriptions': entity_data['descriptions']['en']['value'],
        'aliases': [i['value'] for i in entity_data['aliases']['en']],
    }

    return info_dict

def falcon_query(doc):

    url = 'https://labs.tib.eu/falcon/falcon2/api?mode=long'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "text": doc
    }

    response = requests.post(url, headers=headers, json=data)

    print(response)
    print(json.dumps(response.json(), ensure_ascii=False))
    return response.json()

def entity_linking_with_spacy(sentence:str, add_description=False):
    # returns all entities in the whole document
    doc = nlp(sentence)
    # iterates over sentences and prints linked entities
    ent_dict = {}
    for ent in list(doc._.linkedEntities):
        # print('ID:Q{}. Ent: {}. Mention: {}.'.format(ent.get_id(), ent.get_label(), ent.get_span()))
        if add_description:
            ent_dict[ent.get_span().text] = {'id': 'Q{}'.format(ent.get_id()), 'entity': ent.get_label(), 
                                         'start':ent.get_span().start, 'end':ent.get_span().end,
                                         'description':ent.get_description()}
        else:
            ent_dict[ent.get_span().text] = {'id': 'Q{}'.format(ent.get_id()), 'entity': ent.get_label(), 
                                         'start':ent.get_span().start, 'end':ent.get_span().end}
    
    return ent_dict

def process_by_line(input_file_path, output_file_path, func, src_key):
    with open(input_file_path) as input_f, \
        open(output_file_path, 'w') as output_f:
        i = 0
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            if func == 'el':
                ques = line[src_key]
                ent_dict = entity_linking_with_spacy(ques, add_description=True)
                line['question_entity'] = ent_dict
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    input_file_path = '/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/decompose.json'
    output_file_path = '/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/question_entity.json'
    process_by_line(input_file_path, output_file_path, func='el')
