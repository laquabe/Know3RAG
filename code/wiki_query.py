import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import requests
import json
from tqdm import tqdm
from collections import Counter
import copy

def most_frequent_elements_sorted(lst):
    # 统计元素出现次数
    counts = Counter(lst)
    # 对元素按照出现次数从大到小排序
    sorted_elements = sorted(counts.items(), key=lambda x: (-x[1], lst.index(x[0])))
    return [element for element, count in sorted_elements]

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

def read_relation_template(template_file_name):
    template_info = []
    template_dict = {}
    with open(template_file_name) as template_file:
        for line in template_file:
            line = json.loads(line)
            template_info.append(line)
            r_list = template_dict.get(line['wiki_id'], [])
            r_list.append(line)
            template_dict[line['wiki_id']] = r_list

    return template_info, template_dict

if re_flag:
    template_info, template_dict = read_relation_template('datasets/relation_template.json')
    template_sentence_info, template_sentence_dict = read_relation_template('datasets/relation_sentence_template.json')

def read_map_dict(file_name):
    info_dict = {}
    with open(file_name) as file:
        for line in file:
            line = json.loads(line)
            info_dict[line['wiki_id']] = line
    
    return info_dict

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

def entity_linking_with_spacy(sentence:str, add_description=False, ner=False):
    # returns all entities in the whole document
    doc = nlp(sentence)
    # iterates over sentences and prints linked entities
    ent_dict = {}
    ner_str = ''
    ner_list = []
    # print(len(doc.ents))
    for ent in doc.ents:
        # print(f"{ent.text} ({ent.label_}) - Start: {ent.start_char}, End: {ent.end_char}")
        ner_str += ent.text
        ner_list.append(ent.text)

    for ent in list(doc._.linkedEntities):
        # print('ID:Q{}. Ent: {}. Mention: {}.'.format(ent.get_id(), ent.get_label(), ent.get_span()))
        # filter one word small, as they usually not specific entity
        entity_name = ent.get_label()
        mention = ent.get_span().text
        if entity_name == None:
            continue
        if len(mention) < 2:
            continue
        if ner:
            if mention not in ner_list:
                if mention not in ner_str:
                    continue
                else:
                    if len(mention) <= 2:
                        continue
        else:
            if len(mention) <= 2:
                continue
        
        if add_description:
            ent_dict[mention] = {'id': 'Q{}'.format(ent.get_id()), 'entity': ent.get_label(), 
                                         'start':ent.get_span().start, 'end':ent.get_span().end,
                                         'description':ent.get_description()}
        else:
            ent_dict[mention] = {'id': 'Q{}'.format(ent.get_id()), 'entity': ent.get_label(), 
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
        el_dict = entity_linking_with_spacy(e, ner=True)
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
    triple_relation_map_dict = {}
    for t in triple_list:
        triple_str = t['subject'] + ' ' + t['predicate'] + ' ' + t['object'] + '.'
        if t['predicate'] in r_name_dict.keys():
            triple_relation_map_dict[triple_str] = r_name_dict[t['predicate']]
        else:
            triple_str_embed = sent_model.encode(triple_str)
            relation_embed = sent_model.encode([r['sentence_template'].format_map({'subject': t['subject'], 'object':t['object']}) for r in template_sentence_info])
            sim_matrix = util.pytorch_cos_sim(triple_str_embed, relation_embed)
            sim_matrix = torch.argmax(sim_matrix, dim=-1).item()
            triple_relation_map_dict[triple_str] = template_sentence_info[sim_matrix]['wiki_id']

    return triple_relation_map_dict

def convert_question_to_triple(question:str, entity:str, entity_info, topk=10, count_num=3):
    entity_question_list = [temp['template'].format(entity) for temp in template_info]
    entity_question_embedding = sent_model.encode(entity_question_list)
    question_embedding = sent_model.encode(question)
    sim_matrix = util.pytorch_cos_sim(question_embedding, entity_question_embedding)
    top_similarities, top_indices = torch.topk(sim_matrix, k=topk)
    top_indices = top_indices.tolist()[0]
    top_relation = []
    for idx in top_indices:
        top_relation.append(template_info[idx]['wiki_id'])
    
    local_question_index = []
    local_question_mapping = []
    for local_r in entity_info['claims'].keys():
        local_r_list = template_dict.get(local_r, [])
        if len(local_r_list) > 0:
            local_question_index.extend([temp['template_id'] for temp in local_r_list])
            local_question_mapping.extend([local_r] * len(local_r_list))
    
    if len(local_question_index) > 0:
        local_sim_matrix = sim_matrix[:,local_question_index]
        max_idx = torch.argmax(local_sim_matrix).item()
        local_pred_r = local_question_mapping[max_idx]
    else:
        local_pred_r = ''

    return most_frequent_elements_sorted(top_relation)[:count_num], local_pred_r


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

        p = relation_id_mapping.get(t['subject'] + ' ' + t['predicate'] + ' ' + t['object'] + '.')
        if p == None:
            continue

        for ss in s:
            for oo in o:
                if ss == oo:
                    continue
                triple_id_list.append((ss, p, oo))
        # triple_id_list.append((s, p, o))
    
    return triple_id_list

def update_head_entity(ent_info:dict, tail_dict:dict):
    des = ent_info['description'] + '.'
    valid_tail_set = set()
    for h_id, r_id, t_id in ent_info['kg_triple_id']:
        temp = template_sentence_dict[r_id][0]['sentence_template']
        tail_info = tail_dict[t_id]
        if len(tail_info['labels']) > 0:
            des += ' '
            des += temp.format_map({'subject':ent_info['entity'], 'object':tail_info['labels']})
            valid_tail_set.add(t_id)

    return des.rstrip('.'), valid_tail_set

def process_by_line(input_file_path, output_file_path, func, src_key, tgt_key, entity_key='passage_entity', triple_key='llm_triple', 
                    question_type='open', ner_flag=False, tail_map_dict=None):
    with open(input_file_path) as input_f, \
        open(output_file_path, 'w') as output_f:
        i = 0
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            if func == 'el':
                ques = line[src_key]
                if question_type == 'choice':
                    ques += '\n{}\n{}\n{}\n{}'.format(line['A'], line['B'], line['C'], line['D'])
                ent_dict = entity_linking_with_spacy(ques, add_description=True, ner=ner_flag)
                line[tgt_key] = ent_dict
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
            if func == 'entity_map':
                if len(line[triple_key]) == 0:
                    line[tgt_key] = []
                    output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                    continue
                if el_flag:
                    entity_map_dict = entity_mapping_for_line(line[triple_key], line[entity_key])
                    # print(json.dumps(entity_map_dict))
                if re_flag:
                    relation_map_dict = relation_mapping_for_line(line[triple_key])
                    # print(json.dumps(relation_map_dict))
                triple_id_list = triple_mapping(line[triple_key], entity_map_dict, relation_map_dict)
                line[tgt_key] = triple_id_list
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
            if func == 'convert_triple':
                question = line['question']
                for ent in line['query_entity'].keys():
                    relation_list, local_pred_r = convert_question_to_triple(question, ent, line['query_entity'][ent])
                    ent_dict = line['query_entity'][ent]
                    ent_dict['pred_relation_rank'] = relation_list
                    ent_dict['local_pred_r'] = local_pred_r
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
            if func == 'expand_entity':
                # add triple in head entity
                tail_set = set()
                head_set = set([e['id'] for e in line['query_entity'].values()])
                for ent in line['query_entity'].keys():
                    ent_dict = line['query_entity'][ent]
                    new_des, tail_set_tmp = update_head_entity(ent_dict, tail_map_dict)
                    ent_dict['description'] = new_des
                    tail_set.update(tail_set_tmp)
                    del ent_dict['kg_triple_id']
                # add tail in el
                for t_id in tail_set:
                    if t_id in head_set:
                        continue
                    t_info = copy.deepcopy(tail_dict[t_id])
                    t_info["id"] = t_info.pop("wiki_id")
                    t_info["description"] = t_info.pop("descriptions")
                    t_info["entity"] = t_info.pop("labels")
                    del t_info["claims"]
                    del t_info["aliases"]
                    line['query_entity'][t_info["entity"]] = t_info
                
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    func_name = 'entity_map'
    input_file_path = '/data/xkliu/LLMs/DocFixQA/reference/PopQA/test/gpt4omini_turn1_triple.json'
    output_file_path = '/data/xkliu/LLMs/DocFixQA/reference/PopQA/test/gpt4omini_turn1_triple_id.json'
    map_file_path = '/data/xkliu/LLMs/DocFixQA/datasets/PopQA/turn1_tail_map_full.json'
    if func_name == 'expand_entity':
        tail_dict = read_map_dict(map_file_path)
    else:
        tail_dict = None
    process_by_line(input_file_path, output_file_path, func=func_name, src_key='passages', tgt_key='llm_triple_id', ner_flag=False,
                    entity_key='passage_entity', triple_key='llm_triple', tail_map_dict=tail_dict)
    exit()
    
    '''MMLU'''

    dataset_path = '/data/xkliu/LLMs/DocFixQA/datasets/MMLU/data'
    input_path = '/data/xkliu/LLMs/DocFixQA/reference/MMLU'
    output_path = '/data/xkliu/LLMs/DocFixQA/reference/MMLU'
    mmlu_input = 'triple_llama'
    exp_name = 'triple_id_llama'

    #load src dir
    subjects = sorted([f.split("_dev.json")[0] for f in os.listdir(os.path.join(dataset_path, "dev")) if "_dev.json" in f])

    # mkdir save dir
    save_dir = os.path.join(output_path, 'dev', exp_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for sub in subjects:
        print(sub)
        # if sub != 'high_school_european_history':
        #     continue

        input_file_name = os.path.join(input_path, 'dev', mmlu_input, sub + "_dev.json")

        output_file_name = os.path.join(save_dir, "{}_dev.json".format(sub))

        process_by_line(input_file_name, output_file_name, func='entity_map', src_key='passages', tgt_key='llm_triple_id', 
                        entity_key='passage_entity', triple_key='llm_triple', question_type=exp_name, ner_flag=False)