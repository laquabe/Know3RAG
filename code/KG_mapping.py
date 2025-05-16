import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import requests
import json
from tqdm import tqdm
from collections import Counter
import copy
import argparse
from sentence_transformers import SentenceTransformer, util
import torch
import spacy  # version 3.5

# nlp = spacy.load("en_core_web_md")
# nlp.add_pipe("entityLinker", last=True)
# sent_model = SentenceTransformer('/data/xkliu/hf_models/all-mpnet-base-v2')

def most_frequent_elements_sorted(lst):
    counts = Counter(lst)
    sorted_elements = sorted(counts.items(), key=lambda x: (-x[1], lst.index(x[0])))
    return [element for element, count in sorted_elements]

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

# r_dict, r_name_dict, tmp2wiki, r_des_embedding = read_KG_relation('datasets/relation.json')

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

# template_info, template_dict = read_relation_template('datasets/relation_template.json')
# template_sentence_info, template_sentence_dict = read_relation_template('datasets/relation_sentence_template.json')

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
                entity_map_dict = entity_mapping_for_line(line[triple_key], line[entity_key])
                relation_map_dict = relation_mapping_for_line(line[triple_key])
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
                    t_info = copy.deepcopy(tail_map_dict[t_id])
                    t_info["id"] = t_info.pop("wiki_id")
                    t_info["description"] = t_info.pop("descriptions")
                    t_info["entity"] = t_info.pop("labels")
                    del t_info["claims"]
                    del t_info["aliases"]
                    line['query_entity'][t_info["entity"]] = t_info
                
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    # func_name = 'entity_map'
    # input_file_path = '/data/xkliu/LLMs/DocFixQA/reference/PopQA/test/gpt4omini_turn1_triple.json'
    # output_file_path = '/data/xkliu/LLMs/DocFixQA/reference/PopQA/test/gpt4omini_turn1_triple_id.json'
    # map_file_path = '/data/xkliu/LLMs/DocFixQA/datasets/PopQA/turn1_tail_map_full.json'
    # if func_name == 'expand_entity':
    #     tail_dict = read_map_dict(map_file_path)
    # else:
    #     tail_dict = None
    # process_by_line(input_file_path, output_file_path, func=func_name, src_key='passages', tgt_key='llm_triple_id', ner_flag=False,
    #                 entity_key='passage_entity', triple_key='llm_triple', tail_map_dict=tail_dict)

    parser = argparse.ArgumentParser(description='Script for KG process.')
    import sys
    # --- Global Arguments ---
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device ID for model placement (-1 for CPU). Sets CUDA_VISIBLE_DEVICES.')

    # --- Model and Data Paths (Required as models are loaded unconditionally) ---
    parser.add_argument('--spacy_model', type=str, default='en_core_web_md',
                        help='spaCy model name or path for entity linking.')
    parser.add_argument('--sbert_model', type=str, required=True,
                        help='Path to the Sentence-BERT model for relation mapping.')
    parser.add_argument('--relation_file', type=str, required=True,
                        help='Path to the KG relation definition file.')
    parser.add_argument('--relation_template_file', type=str, required=True,
                        help='Path to the relation template file.')
    parser.add_argument('--relation_sentence_template_file', type=str, required=True,
                        help='Path to the relation sentence template file.')

    # --- Processing Arguments ---
    parser.add_argument('--command', type=str, required=True,
                        choices=['el', 'entity_map', 'convert_triple', 'expand_entity'],
                        help='Processing command to execute: el, entity_map, convert_triple, or expand_entity.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input JSONL file.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path for the output JSONL file.')

    # --- Command-Specific Arguments (Optional or required based on command) ---
    parser.add_argument('--src_key', type=str,
                        help='Source key in the input JSON line (e.g., "passages" for "el"). Required by "el".')
    parser.add_argument('--tgt_key', type=str,
                        help='Target key in the output JSON line (e.g., "passage_entity" for "el", "llm_triple_id" for "entity_map", "pred_relation_rank" for "convert_triple"). Required by "el", "entity_map", "convert_triple".')
    parser.add_argument('--entity_key', type=str, default='passage_entity',
                        help='Key for entity dictionary in input line (default: passage_entity). Used by "entity_map", "expand_entity".')
    parser.add_argument('--triple_key', type=str, default='llm_triple',
                        help='Key for textual triple list in input line (default: llm_triple). Used by "entity_map".')
    parser.add_argument('--question_type', type=str, default='open', choices=['open', 'choice'],
                        help='Type of question format for "el" command (default: open).')
    parser.add_argument('--ner_flag', action='store_true',
                        help='if the el result filter by ner. query entity is set True, while passage entity is set False.')
    parser.add_argument('--map_file', type=str,
                        help='Path to a map file (e.g., tail entity map). Required by "expand_entity".')

    # Arguments for 'convert_triple'
    parser.add_argument('--topk', type=int, default=10,
                        help='Top K similar relation templates to consider for "convert_triple".')
    parser.add_argument('--count_num', type=int, default=3,
                        help='Number of top relation IDs to keep after frequency sorting for "convert_triple".')


    # --- Parse arguments ---
    args = parser.parse_args()

    # --- Set CUDA device environment variable ---
    if args.device >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        print(f"Setting CUDA_VISIBLE_DEVICES = {args.device}")
    else:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        print("Running on CPU.")

    # --- Load Global Resources (Unconditional based on simplified logic) ---
    # Import libraries after setting CUDA_VISIBLE_DEVICES
    try:
        import spacy
        nlp = spacy.load(args.spacy_model)
        if "entityLinker" not in nlp.pipe_names:
            try:
                nlp.add_pipe("entityLinker", last=True)
                print("Added entityLinker pipeline.")
            except Exception as e:
                 print(f"Warning: Could not add entityLinker pipeline: {e}. spaCy EL might be limited.")
        print(f"spaCy model '{args.spacy_model}' loaded.")
    except ImportError:
        print("Error: spacy library not found. Install it.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer, util
        import torch
        sent_model = SentenceTransformer(args.sbert_model)
        print(f"Sentence-BERT model '{args.sbert_model}' loaded.")
    except ImportError:
        print("Error: sentence-transformers or torch library not found. Install them.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading Sentence-BERT model: {e}")
        sys.exit(1)

    # Load relation data and templates
    r_dict, r_name_dict, tmp2wiki, r_des_embedding = read_KG_relation(args.relation_file)
    if r_dict is None:
        print("Error: Failed to load KG relation data.")
        sys.exit(1)

    template_info, template_dict = read_relation_template(args.relation_template_file)
    if template_info is None:
        print("Error: Failed to load relation templates.")
        sys.exit(1)

    template_sentence_info, template_sentence_dict = read_relation_template(args.relation_sentence_template_file)
    if template_sentence_info is None:
        print("Error: Failed to load relation sentence templates.")
        sys.exit(1)


    # --- Load map file if required by the command ---
    if args.command == 'expand_entity':
        if not args.map_file:
            print("Error: 'expand_entity' command requires --map_file.")
            sys.exit(1)
        print(f"Reading map file for expand_entity from: {args.map_file}")
        tail_dict_from_map = read_map_dict(args.map_file)
        if tail_dict_from_map is None:
            print("Error: Failed to read map file.")
            sys.exit(1)


    # --- Validate arguments based on command ---
    if args.command == 'el':
        if not args.src_key or not args.tgt_key:
            print("Error: 'el' command requires --src_key and --tgt_key.")
            sys.exit(1)
    elif args.command == 'entity_map':
        if not args.tgt_key:
            print("Error: 'entity_map' command requires --tgt_key.")
            sys.exit(1)
    elif args.command == 'convert_triple':
        if not args.tgt_key:
            print("Error: 'convert_triple' command requires --tgt_key.")
            sys.exit(1)
    # 'expand_entity' map_file requirement checked above. entity_key has default.


    # --- Execute the main processing function ---
    process_by_line(
        input_file_path=args.input_file,
        output_file_path=args.output_file,
        func_name=args.command, # Use the command as the function name
        src_key=args.src_key,
        tgt_key=args.tgt_key,
        entity_key=args.entity_key,
        triple_key=args.triple_key,
        question_type=args.question_type,
        ner_flag=args.ner_flag,
        tail_map_dict_loaded=tail_dict_from_map, # Pass the loaded map if any
        topk=args.topk,
        count_num=args.count_num
    )

    print("Script execution finished.")