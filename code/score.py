import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import json
from tqdm import tqdm
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import argparse

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

def load_dataset(dataset_path):
    file_list = ['train', 'valid', 'test']
    ref_triple_dict = {}
    for file_type in file_list:
        file_name = dataset_path + file_type + '.txt'
        with open(file_name) as file:
            for line in file:
                line = line.strip()
                s, p, o = line.split('\t')

                ref_list = ref_triple_dict.get(s, [])
                ref_list.append((s,p,o))
                ref_triple_dict[s] = ref_list

                # ref_list = ref_triple_dict.get(o, [])
                # ref_list.append((s,p,o))
                # ref_triple_dict[o] = ref_list

    return ref_triple_dict

# load mapping dict
e_file_path = '/data/xkliu/kge/data/wikidata5m/entity_ids.json'
r_file_path = '/data/xkliu/kge/data/wikidata5m/relation_ids.json'
e_kgc_id_dict, r_kgc_id_dict = read_KGC_id_dict(e_file_path, r_file_path)

# download link for this checkpoint given under results above
checkpoint = load_checkpoint('/data/xkliu/kge/checkpoints/wikidata5m-complex.pt')
model = KgeModel.create_from(checkpoint)

ref_dict = load_dataset('/data/xkliu/kge/data/wikidata5m/')

def kg_perd_tail(ent_dict:dict, max_ref_num=3):
    tail_set = set()
    ref_triple_id = []
    head_id = e_kgc_id_dict.get(ent_dict['id'], -1)
    if len(ent_dict['local_pred_r']) > 0:
        local_r_id = r_kgc_id_dict[ent_dict['local_pred_r']]
        local_cand_tail = ent_dict['claims'][ent_dict['local_pred_r']]

        if len(local_cand_tail) > max_ref_num:
            if head_id == -1:
                local_cand_tail = local_cand_tail[:max_ref_num]
            else:
                new_local_cand_tail = []
                for l_c_t in local_cand_tail:
                    tail_list = []
                    l_c_t_id = e_kgc_id_dict.get(l_c_t, -1)
                    if l_c_t_id != -1:
                        tail_list.append(l_c_t_id)
                        new_local_cand_tail.append(l_c_t)
                if len(tail_list) < max_ref_num:
                    local_cand_tail = local_cand_tail[:max_ref_num]
                else:
                    head_list = [int(head_id)] * len(tail_list)
                    relation_list = [int(local_r_id)] * len(tail_list)
                    o_tensor = torch.Tensor(tail_list).long()
                    s_tensor = torch.Tensor(head_list).long()
                    p_tensor = torch.Tensor(relation_list).long()
                    scores = model.score_spo(s_tensor, p_tensor, o_tensor)
                    topk_values, topk_indices = torch.topk(scores, k=max_ref_num)
                    local_cand_tail = new_local_cand_tail[topk_indices]

        tail_set.update(local_cand_tail)
        ref_triple_id = [(ent_dict['id'], ent_dict['local_pred_r'], t) for t in local_cand_tail]

    if ent_dict['local_pred_r'] in ent_dict['pred_relation_rank']:
        return ref_triple_id, tail_set
    
    if head_id == -1:
        return ref_triple_id, tail_set
    
    head_list = [int(head_id)] * len(ent_dict['pred_relation_rank'])
    relation_list = [int(r_kgc_id_dict[r]) for r in ent_dict['pred_relation_rank']]
    s_tensor = torch.Tensor(head_list).long()
    p_tensor = torch.Tensor(relation_list).long()
    scores = model.score_sp(s_tensor, p_tensor)
    max_values, max_indices = torch.max(scores, dim=-1)
    global_max_index = torch.argmax(max_values)
    pred_id = model.dataset.entity_ids(max_indices[global_max_index]).item()
    if pred_id not in tail_set:
        tail_set.add(pred_id)
        ref_triple_id.append((ent_dict['id'], ent_dict['pred_relation_rank'][global_max_index], pred_id))
    
    return ref_triple_id, tail_set

def kg_prediction_by_line(input_file_path, output_file_path, map_file_path):
    with open(input_file_path) as input_f, \
        open(output_file_path, 'w') as output_f, \
        open(map_file_path, 'w') as map_file:
        global_tail_set = set()
        for line in tqdm(input_f):
            line = json.loads(line.strip())

            # pred tail
            for ent_name, ent in line['query_entity'].items():
                ref_triple_list, tail_set = kg_perd_tail(ent)
                global_tail_set.update(tail_set)
                line['query_entity'][ent_name]['kg_triple_id'] = ref_triple_list
                del line['query_entity'][ent_name]['claims']
                del line['query_entity'][ent_name]['pred_relation_rank']
                del line['query_entity'][ent_name]['local_pred_r']
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
        for t in global_tail_set:
            map_file.write(json.dumps({'wiki_id':t}, ensure_ascii=False) + '\n')


def process_by_line(input_file_path, output_file_path, src_key, tgt_key, relation=True, ref_dict=None):
    with open(input_file_path) as input_f, \
        open(output_file_path, 'w') as output_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            triples = line[src_key]
            s_id_list = []
            p_id_list = []
            o_id_list = []

            valid_head_set = set()
            valid_triple = []

            for t in triples:
                s, p, o = t
                
                o_id = e_kgc_id_dict.get(o, -1)
                if o_id == -1:
                    continue

                s_id = e_kgc_id_dict.get(s, -1)
                if s_id == -1:
                    continue
                else:
                    valid_head_set.add(s)

                p_id = r_kgc_id_dict[p] # we use the relation in kgc model, so we do not check

                s_id_list.append(int(s_id))
                p_id_list.append(int(p_id))
                o_id_list.append(int(o_id))
                valid_triple.append(t)

            if len(s_id_list) == 0:
                # none link
                line[tgt_key] = []
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                continue

            else:
                if relation:
                    s_tensor = torch.Tensor(s_id_list).long()
                    p_tensor = torch.Tensor(p_id_list).long()
                    o_tensor = torch.Tensor(o_id_list).long()
                    scores = model.score_spo(s_tensor, p_tensor, o_tensor)
                    line_scores = scores.tolist()
                else:
                    s_tensor = torch.Tensor(s_id_list).long()
                    o_tensor = torch.Tensor(o_id_list).long()
                    scores = model.score_so(s_tensor, o_tensor)
                    scores, _ = torch.max(scores, dim=1)
                    line_scores = scores.tolist()

            # ref score

            ref_score_dict = {}

            for ss in valid_head_set:
                ref_triple = ref_dict.get(ss, [])
                if len(ref_triple) == 0:
                    ref_score_dict[ss] = []
                else:
                    s_ref_list = []
                    p_ref_list = []
                    o_ref_list = []
                    for s, p, o in ref_triple:
                        s_ref_list.append(int(e_kgc_id_dict[s]))
                        p_ref_list.append(int(r_kgc_id_dict[p]))
                        o_ref_list.append(int(e_kgc_id_dict[o]))
                    if relation:
                        s_tensor = torch.Tensor(s_ref_list).long()
                        p_tensor = torch.Tensor(p_ref_list).long()
                        o_tensor = torch.Tensor(o_ref_list).long()
                        scores = model.score_spo(s_tensor, p_tensor, o_tensor)
                        scores = scores.tolist()
                    else:
                        s_tensor = torch.Tensor(s_ref_list).long()
                        o_tensor = torch.Tensor(o_ref_list).long()
                        scores = model.score_so(s_tensor, o_tensor)
                        scores, _ = torch.max(scores, dim=1)
                        scores = scores.tolist()

                    ref_score_dict[ss] = scores
            
            tgt_list = []
            assert len(valid_triple) == len(line_scores)

            for t, score in zip(valid_triple, line_scores):
                s, p, o = t
                tgt_list.append(
                    {'triple_id': t,
                     'triple_score': score,
                     'ref_score': ref_score_dict[s]}
                )

            line[tgt_key] = tgt_list
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    # flag = 'score'
    # input_file_path = '/data/xkliu/LLMs/DocFixQA/result/PopQA/qwen2.5-32b-instruct/Qwen32B_turn01_merge_triple_id.json'
    # output_file_path = '/data/xkliu/LLMs/DocFixQA/result/PopQA/qwen2.5-32b-instruct/Qwen32B_turn01_merge_triple_score.json'
    # if flag == 'score':
    #     process_by_line(input_file_path, output_file_path, src_key='llm_triple_id', tgt_key='llm_triple_score', relation=True, ref_dict=ref_dict)
    # elif flag == 'pred':
    #     map_file_path = '/data/xkliu/LLMs/DocFixQA/datasets/PopQA/turn1_tail_map.json'
    #     kg_prediction_by_line(input_file_path, output_file_path, map_file_path)
    # exit()

    parser = argparse.ArgumentParser(description="Process or score knowledge graph triples using a KGE model.")

    # Use subparsers to define different modes of operation
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Operation mode (score or predict)')

    # --- Subparser for 'score' mode (using process_by_line) ---
    parser_score = subparsers.add_parser('score', help='Score existing triples in an input file.')
    parser_score.add_argument('--input_file', help='Path to the input JSONL file containing triples.')
    parser_score.add_argument('--output_file', help='Path to the output JSONL file to save scored results.')
    parser_score.add_argument('--src-key', default='llm_triple_id',
                              help=f'Key in the input JSON object containing the list of triples to score. Default: %(default)s')
    parser_score.add_argument('--tgt-key', default='llm_triple_score',
                              help=f'Key to add to the output JSON object for the scored triples. Default: %(default)s')

    # --- Subparser for 'predict' mode (using kg_prediction_by_line) ---
    parser_predict = subparsers.add_parser('predict', help='Generate predicted tails for entities in an input file.')
    parser_predict.add_argument('--input_file', help='Path to the input JSONL file containing entities with claims/predictions.')
    parser_predict.add_argument('--output_file', help='Path to the output JSONL file to save entity info with predicted triples.')
    parser_predict.add_argument('--map_file', help='Path to a JSONL file to save the collected wiki IDs of predicted tails.')


    args = parser.parse_args()

    # --- Execute based on the chosen mode ---
    if args.mode == 'score':
        process_by_line(
            input_file_path=args.input_file,
            output_file_path=args.output_file,
            src_key=args.src_key,
            tgt_key=args.tgt_key,
            relation=True,
            ref_dict=ref_dict # Pass the globally loaded ref_dict
        )
    elif args.mode == 'predict':
        kg_prediction_by_line(
            input_file_path=args.input_file,
            output_file_path=args.output_file,
            map_file_path=args.map_file
        )

    print("Operation complete.")
