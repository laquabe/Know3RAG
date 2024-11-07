import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
import requests
import json
from tqdm import tqdm

mapping_flag = True
el_flag = mapping_flag
re_flag = mapping_flag

if el_flag:
    import spacy  # version 3.5
    # initialize language model
    nlp = spacy.load("en_core_web_md")
    # add pipeline (declared through entry_points in setup.py)
    nlp.add_pipe("entityLinker", last=True)

if re_flag:
    from sentence_transformers import SentenceTransformer, util
    import torch
    sent_model = SentenceTransformer('/data/xkliu/hf_models/all-mpnet-base-v2')

def read_KG_relation(relation_file_name):
    r_dict = {}
    r_name_dict = {}
    tmp2wiki = {}
    r_des_list = []
    with open(relation_file_name) as r_file:
        for line in tqdm(r_file):
            line = json.loads(line.strip())
            if len(line['labels']) == 0:
                continue

            r_dict[line['wiki_id']] = line
            
            r_name_dict[line['labels']] = line['wiki_id']
            for ali in line['aliases']:
                r_name_dict[ali] = line['wiki_id']
            
            aliases = ';'.join(line['aliases'])
            relation_des = '{}. {}. {}.'.format(line['labels'], line['descriptions'], aliases)
            tmp2wiki[len(r_des_list)] = line['wiki_id']
            r_des_list.append(relation_des)

    r_des_embedding = sent_model.encode(r_des_list)

    return r_dict, r_name_dict, tmp2wiki, r_des_embedding

if re_flag:
    r_dict, r_name_dict, tmp2wiki, r_des_embedding = read_KG_relation('datasets/relation.json')
    
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
        # filter one word small, as they usually not specific entity
        entity_name = ent.get_label()
        if entity_name == None:
            continue
        if len(entity_name.split()) < 2 and entity_name.islower():
            continue

        if add_description:
            ent_dict[ent.get_span().text] = {'id': 'Q{}'.format(ent.get_id()), 'entity': ent.get_label(), 
                                         'start':ent.get_span().start, 'end':ent.get_span().end,
                                         'description':ent.get_description()}
        else:
            ent_dict[ent.get_span().text] = {'id': 'Q{}'.format(ent.get_id()), 'entity': ent.get_label(), 
                                         'start':ent.get_span().start, 'end':ent.get_span().end}
    
    return ent_dict

def entity_mapping_for_line(triple_list:list, entity_dict:dict):
    # entity_dict : passage_entity
    # triple: llm_triple
    
    # get the llm link entity first
    local_entity_set = set()
    for t in triple_list:
        local_entity_set.add(t['subject'])
        local_entity_set.add(t['object'])
    
    # search the local dict (and global_dict: determine time cost)
    link_entity_keys = set(entity_dict.keys())
    unlink_entity_list = []
    for e in local_entity_set:
        if e in link_entity_keys:
            continue
        unlink_entity_list.append(e)

    # use el link the miss entity
    for e in unlink_entity_list:
        el_dict = entity_linking_with_spacy(e)
        el_id_list = []
        el_entity_list = []
        for v in el_dict.values():
            el_id_list.append(v['id'])
            el_entity_list.append(v['entity'])

        if len(el_id_list) > 0:
            res_dict = {
                e: {
                    'id':el_id_list,
                    'entity':el_entity_list
                }
            }
            entity_dict.update(res_dict)

    return entity_dict

def relation_mapping_for_line(triple_list:list):
    # get all relation
    local_relation_set = set()
    for t in triple_list:
        local_relation_set.add(t['predicate'])
    local_relation_list = list(local_relation_set)

    # sentence sim
    local_relation_embedding = sent_model.encode(local_relation_list)
    sim_matrix = util.pytorch_cos_sim(local_relation_embedding, r_des_embedding)
    sim_matrix = torch.argmax(sim_matrix, dim=1)
    sim_matrix = sim_matrix.tolist()

    # mapping: 1. hit name. 2. top sim 
    local_relation_map_dict = {}
    for local_relation, sim_id in zip(local_relation_list, sim_matrix):
        if local_relation in r_name_dict.keys():
            local_relation_map_dict[local_relation] = r_name_dict[local_relation]
        else:
            local_relation_map_dict[local_relation] = tmp2wiki[sim_id]

    return local_relation_map_dict

def triple_mapping(triple_text_list:list, entity_id_mapping:dict, relation_id_mapping:dict):
    triple_id_list = []
    for t in triple_text_list:
        # entity_mapping
        # relation_mapping
        s = entity_id_mapping.get(t['subject'])
        if s == None:
            continue
        else:
            s = s['id']
        if isinstance(s, str):
            s = [s]

        o = entity_id_mapping.get(t['object'])
        if o == None:
            continue
        else:
            o = o['id']
        if isinstance(o, str):
            o = [o]

        p = relation_id_mapping.get(t['predicate'])
        if p == None:
            continue

        for ss in s:
            for oo in o:
                if ss == oo:
                    continue
                triple_id_list.append((ss, p, oo))
        # triple_id_list.append((s, p, o))
    
    return triple_id_list

def process_by_line(input_file_path, output_file_path, func, src_key, tgt_key):
    with open(input_file_path) as input_f, \
        open(output_file_path, 'w') as output_f:
        i = 0
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            if func == 'el':
                ques = line[src_key]
                ent_dict = entity_linking_with_spacy(ques, add_description=True)
                line[tgt_key] = ent_dict
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
            if func == 'entity_map':
                if len(line['llm_triple']) == 0:
                    line[tgt_key] = []
                    output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                    continue
                if el_flag:
                    entity_map_dict = entity_mapping_for_line(line['llm_triple'], line['passage_entity'])
                    # print(json.dumps(entity_map_dict))
                if re_flag:
                    relation_map_dict = relation_mapping_for_line(line['llm_triple'])
                    # print(json.dumps(relation_map_dict))
                triple_id_list = triple_mapping(line['llm_triple'], entity_map_dict, relation_map_dict)
                line[tgt_key] = triple_id_list
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

def read_KGC_id_dict(entity_file_name, relation_file_name):
    with open(entity_file_name) as e_f,\
        open(relation_file_name) as r_f:
        e_kgc_id_dict = {}
        r_kgc_id_dict = {}

        for line in tqdm(e_f):
            line = json.loads(line)
            e_kgc_id_dict[line['wiki_id']] = line['map_id']
        
        for line in tqdm(r_f):
            line = json.loads(line)
            r_kgc_id_dict[line['wiki_id']] = line['map_id']
    
    return e_kgc_id_dict, r_kgc_id_dict

if score_flag:
    e_file_path = '/data/xkliu/kge/data/wikidata5m/entity_ids.json'
    r_file_path = '/data/xkliu/kge/data/wikidata5m/relation_ids.json'
    e_kgc_id_dict, r_kgc_id_dict = read_KGC_id_dict(e_file_path, r_file_path)

if __name__ == '__main__':
    input_file_path = '/data/xkliu/LLMs/DocFixQA/result/TemporalQA/wiki/test.json'
    output_file_path = '/data/xkliu/LLMs/DocFixQA/result/TemporalQA/wiki/test_score.json'
    process_by_line(input_file_path, output_file_path, func='entity_map',src_key='passages', tgt_key='triple_id_list')