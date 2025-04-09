import json
import copy
import re
from tqdm import tqdm
import numpy as np
import copy
MAX_SCORE = 10000

def read_data(dataset_name, file_path):
    if dataset_name in ['Truthful_QA', 'Temporal_QA']:
        with open(file_path) as f:
            data = f.read()
            data = json.loads(data)
            return data
        if dataset_name in ['PopQA']:
            pass

def is_discontinuous_substring(main_string:str, substring:str):
    substring = substring.lower()
    main_string = main_string.lower()
    substring = substring.split(' ')
    for ss in substring:
        if ss not in main_string:
            return False
    
    return True

def list2pair(input_file_name, output_file_name, func, line_mode=True):
    '''
    raw: list2pair
    knowledge_card: k generate list to pair
    decompose: sub question list to pair with passages
    subques_only: only sub question
    '''
    with open(input_file_name) as input_f, \
        open(output_file_name, 'w') as output_f:
        if not line_mode:
            data = input_f.read()
            input_f = json.loads(data)
        error_num = 0
        for line in input_f:
            if line_mode:
                line = json.loads(line.strip())
            if func == 'raw':
                src = line['passages']
                src = src[0][0]
                for s in src:
                    new_line = copy.deepcopy(line)
                    del new_line['passages']
                    new_line['passages'] = s
                    output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')
            if func == 'knowledge_card':
                src = line['generate']
                for s in src:
                    new_line = copy.deepcopy(line)
                    if 'passages' in new_line.keys():
                        del new_line['passages']
                    del new_line['generate']
                    # s = s.split('.')
                    # s = '.'.join(s[:-1])
                    # s += '.'
                    # if len(s) == 0:
                    #     continue
                    s = s.strip().lstrip('Answer:').strip()
                    new_line['passages'] = s
                    if 'orginal_question' in line.keys():
                        new_line['question']  = line['orginal_question']
                        del new_line['orginal_question']
                    output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')
            if func == 'subques_only':
                new_line = copy.deepcopy(line)
                new_line['sub_question'] = line['Question']
                output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')
                for sub_q in line['sub_questions'].values():
                    new_line['sub_question'] = sub_q
                    output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')
            if func == 'decompose':
                res, json_flag = json_decode(line['llm_response'])
                if json_flag:
                    src = line['passages']
                    src = src[0][0]
                    for s in src:
                        write_flag = False
                        # first search str in reference
                        for ent, qes in res.items():
                            if ent.lower() in s.lower():
                                new_line = copy.deepcopy(line)
                                del new_line['passages']
                                new_line['passages'] = s
                                new_line['sub_question'] = qes
                                output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')
                                write_flag = True
                        if write_flag:
                            continue
                        new_line = copy.deepcopy(line)
                        del new_line['passages']
                        new_line['passages'] = s
                        new_line['sub_question'] = new_line['Question']
                        output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')
                else:
                    error_num += 1
                    src = line['passages']
                    src = src[0][0]
                    for s in src:
                        new_line = copy.deepcopy(line)
                        del new_line['passages']
                        new_line['passages'] = s
                        new_line['sub_question'] = new_line['Question']
                        output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')

    print(error_num)

def filter_exinfo(prediction, firstorlast='last', check_key='useful'):
    '''
    filter the external information
    1. decode json
    2. find yes/no
    '''
    json_pattern = r'\{.*?\}'
    match = re.search(json_pattern, prediction, re.DOTALL)  # re.DOTALL 允许 . 匹配换行符

    if match:
        json_str = match.group()
        try:
            json_str = json.loads(json_str)
            if json_str[check_key] == 'no':
                return False
            return True
        except:
            if isinstance(json_str, dict):
                prediction = json_str['reason']
            else:
                prediction = json_str  
    
    
        if firstorlast == 'last':
            neg_index = prediction.rfind('no')
            if neg_index == -1:
                return True
            pos_index = prediction.rfind('yes')
            if pos_index < neg_index:
                return False
            return True
        else:   # first
            neg_index = prediction.find('no')
            if neg_index == -1:
                return True
            pos_index = prediction.find('yes')
            if pos_index == -1:
                return False
            if pos_index < neg_index:
                return True
            return False

def score_feature(score_list:list, entity_num, entity_count=True):
    feature_score_list = []
    for triple in score_list:
        # feature_score_list.append(triple['triple_score'])
        if len(triple['ref_score']) == 0:
            continue
        ref_avg = np.average(triple['ref_score'])
        score = np.abs(triple['triple_score'] - ref_avg)
        feature_score_list.append(score)
    
    if entity_count:
        if len(feature_score_list) == 0:
            feature_score_list = [MAX_SCORE - entity_num]
    else:
        if len(feature_score_list) == 0:
            return None
        
    return np.average(feature_score_list)

def pair_merge(input_file_name, output_file_name, func, dataset, summary_map_dict = None, top_k=3):
    '''
    raw: just pair to list. input file is the summary file(processed)
    filter: use llm pair judge to filter external info. input file is the filter file(raw)
    '''
    id2src = {} # id to question
    id2info = {} # id to ref

    with open(input_file_name) as input_f, \
        open(output_file_name, 'w') as output_f:

        if dataset == 'Temporal_QA':

            for line in input_f:
                line = json.loads(line.strip())
                l_id = line['Id']
                if l_id not in id2src.keys():
                    new_l = copy.deepcopy(line)
                    del new_l['passages']
                    if 'llm_response' in new_l.keys():
                        del new_l['llm_response']
                    id2src[l_id] = new_l
                    id2info[l_id] = []
                if func == 'filter':
                    # if no then ignore the exinfo
                    try:
                        line['summary'] = summary_map_dict[line['Id']][line['passages']]
                    except:
                        line['summary'] = line['passages']
                    if filter_exinfo(line['llm_response'], check_key='reliability'):
                        ex_info_list = id2info[l_id]
                        if line['summary'] not in ex_info_list:
                            # ex_info_list.append(line['summary'])
                            ex_info_list.append(line['passages'])
                            if l_id == 3:
                                print(json.dumps(line))
                        id2info[l_id] = ex_info_list
                    # else:
                    #     ex_info_list = id2info[l_id]
                    #     ex_info_list.append(line['summary'])
                    #     id2info[l_id] = ex_info_list
                if func == 'raw':
                    ex_info_list = id2info[l_id]
                    ex_info_list.append(line['summary'])
                    id2info[l_id] = ex_info_list
            
            for k, v in id2src.items():
                exinfo = id2info[k]
                v['passages'] = exinfo
                output_f.write(json.dumps(v, ensure_ascii=False) + '\n')
        
        elif dataset in ['MMLU', 'hotpotQA']:
            global_id = 0
            for line in input_f:
                # gather reference
                line = json.loads(line.strip())
                if dataset == 'MMLU':
                    key_word = line['Question'] + str(line['A']) + str(line['B']) + str(line['C']) + str(line['D'])
                elif dataset == 'hotpotQA':
                    key_word = line['id']

                if key_word not in id2src.keys():
                    if dataset == 'MMLU':
                        id2src[key_word] = {'Question':line['Question'], 'Id': global_id, 'A': line['A'], 'B': line['B'], 'C': line['C'], 'D': line['D'], 'Answer':line['Answer'], 'query_entity':line['query_entity']}
                    elif dataset == 'hotpotQA':
                        id2src[key_word] = {'id': line['id'], 'Id': global_id, 'question':line['question'], 'answer':line['answer'], 'query_entity':line['query_entity']}
                    global_id += 1
                quiz_id = id2src[key_word]['Id']
                ref_list = id2info.get(quiz_id, [])
                ref_list.append({'passages':line['passages'], 'passage_entity':line['passage_entity'], 'local_check':line['local_check'], 'triple_score':score_feature(line['llm_triple_score'], len(line['passage_entity']), entity_count=True)})
                id2info[quiz_id] = ref_list
            
            # filter ref
            for question, question_info in id2src.items():
                ref_list = id2info[question_info['Id']]
                ref_p_valid = []
                ref_s_valid = []
                # score filter
                for ref in ref_list:
                    if func == 'filter':
                        if ref['local_check'] == True:
                            # if ref['triple_score'] == None:
                            #     continue
                            # elif ref['triple_score'] > 10:
                            #     continue

                            ref_p_valid.append(ref['passages'])
                            # ref_p_valid.append(ref['summary'])
                            ref_s_valid.append(ref['triple_score'])
                    elif func == 'raw':
                        ref_p_valid.append(ref['passages'])
                        # ref_p_valid.append(ref['summary'])
                        ref_s_valid.append(ref['triple_score'])
                    else:
                        print('func error')
                        exit()

                
                # write json
                del question_info['Id']
                del question_info['query_entity']
                # print(len(ref_s_valid))
                if len(ref_s_valid) == 0:
                    question_info['reference'] = []
                    output_f.write(json.dumps(question_info, ensure_ascii=False) + '\n')
                else:
                    index = np.argsort(ref_s_valid)
                    index = index[:np.min([top_k, len(ref_s_valid)])]
                    question_info['reference'] = [ref_p_valid[i] for i in index]
                    # question_info['reference'].reverse()
                    output_f.write(json.dumps(question_info, ensure_ascii=False) + '\n')
                    # print(ref_s_valid)
                    # print(index)
                    # print(question_info['reference'], len(question_info['reference']))
                    # exit()


def json_decode(ans:str):
    json_pattern = r'\{.*?\}'
    match = re.search(json_pattern, ans, re.DOTALL)  # re.DOTALL 允许 . 匹配换行符

    if match:
        json_str = match.group()
        json_str = json_str.replace('\n', '')
        try:
            json_str = json.loads(json_str)
            return json_str, True
        except:
            return json_str, False  
    
    return ans, False

def triple_extraction_decode(ans:str):
    json_pattern = r'\[.*?\]'
    match = re.match(json_pattern, ans, re.DOTALL)  # re.DOTALL 允许 . 匹配换行符
    phrase_flag = False

    if match:
        json_str = match.group()
        json_str = json_str.replace('\n', '')
        # print(json.dumps(json_str))
        try:
            json_str = json.loads(json_str)
            phrase_flag = True
            return json_str, phrase_flag
        except:
            pass  

    json_pattern = r'\{.*?\}'
    match = re.findall(json_pattern, ans)
    triple_list = []
    for m in match:
        try:
            m = json.loads(m)
            triple_list.append(m)
        except:
            continue
    
    if len(triple_list) > 0:
        return triple_list, True
    else:
        return ans, False

    return ans, False

forbidden_list = set(['--','I', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those', 
                      'anyone', 'everyone', 'someone', 'no one', 'nobody', 'somebody', 'everybody', 'anything', 'something', 'everything', 'nothing',
                      'the', 'a', 'an', 'one', 'the two', 'the other', 'other', 'another',
                      'book', 'song', 'country', 'school', 'friend', 'pet', 'job', 'event', 'restaurant', 'app', 'company', 'film', 'people', 'person',
                      'language', 'city', 'family member', 'hobby', 'sport', 'project', 'skill', 'neighborhood', 'website', 'community', 'judge', 'court'])

def triple_verication(list_raw:list):
    '''纯数字我们不处理，他仍然是合法的'''
    list_new = []
    
    for t in list_raw:
        head_list, tail_list = [], []
        if isinstance(t, str):
            continue
        if 'subject' not in t.keys():
            continue
        if 'object' not in t.keys():
            continue
        if 'predicate' not in t.keys():
            continue
        if isinstance(t['subject'], str):
            head_list = [t['subject']]
        elif isinstance(t['subject'], list):
            head_list = t['subject']

        if isinstance(t['object'], str):
            tail_list = [t['object']]
        elif isinstance(t['object'], list):
            tail_list = t['object']

        for s in head_list:
            if s.lower() in forbidden_list:
                continue
            for o in tail_list:
                if o.lower() in forbidden_list:
                    continue
                list_new.append({'subject':s, 'predicate':t['predicate'], 'object':o})
    
    return list_new

def summary_process(line:dict):
    res_json, flag = json_decode(line['llm_response'])
    if flag:
        if 'summary' in res_json.keys():
            return res_json['summary'], flag
        elif 'Summary' in res_json.keys():
            return res_json['Summary'], flag
        else:
            print(res_json)
            exit()
    return res_json, flag

def local_check_str(response:str):
    prefix = 'The reliability of the passage is'
    ans_index = response.find(prefix)
    ans = response[ans_index:]
    if 'yes' in ans:
        return True
    else:
        return False

def process_by_line(input_file_name, output_file_name, func, id2subq_dict=None, tgt_key_name=None, src_key_name=None):
    '''
    summary: phrase llm response to key summary\n
    map_decompose: deliver sub-question to passages\n
    reference: phrase reference\n
    add_reference: add pseudo reference as passages\n
    add_question_entity: add_question_entity\n
    reliability_phrase: local_ckeck\n
    add_key: add src to tgt key\n
    triple_extract: phrase llm triple extraction\n
    self_ask: phrase llm self ask\n
    result_merge: final result merge\n
    count: count line\n
    phrase_question: phrase the generated answer by LLM
    '''
    with open(input_file_name) as input_f, \
        open(output_file_name, 'w') as output_f:
        error_num = 0
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            if func == 'summary':
                summary_text, flag = summary_process(line)
                if flag:
                    line['summary'] = summary_text
                else:
                    line['summary'] = line['passages']
                    error_num += 1
            if func == 'phrase_question':
                line['orginal_question'] = line['question']
                line['question'] = line['llm_response'].strip()
                del line['llm_response']
                # del line['passages']
            if func == 'reference':
                '''old for json phrase'''
                # res, json_flag = json_decode(line['llm_response'])
                # if json_flag:
                #     line['pseudo_doc'] = res['reference_paragraph']
                # else:
                #     line['pseudo_doc'] = line['Question']
                #     error_num += 1
                '''new for str'''
                line[tgt_key_name] = line[src_key_name].strip()
            if func == 'triple_extract':
                res, json_flag = triple_extraction_decode(line['llm_response'])
                if json_flag:
                    res = triple_verication(res)
                    line['llm_triple'] = res
                
                else:
                    line['llm_triple'] = []
                    error_num += 1
            if func == 'add_reference':
                reference = line[src_key_name]
                idx= reference.rfind('The answer is')
                reference = reference[:idx]
                line[tgt_key_name] = reference.strip()
                if line[src_key_name] == line[tgt_key_name]:
                    error_num += 1
            if func == 'add_question_entity':
                line['question_entity'] = id2subq_dict[line['Id']]
            if func == 'add_key':
                line[tgt_key_name] = line[src_key_name]
            if func == 'add_pseudo_doc_simple':
                passages = line['passages'][0][0]
                new_p_list = []
                for p in passages:
                    new_p_list.append(id2subq_dict[line['Id']][p])
                new_p_list.append(line['pseudo_doc'])
                line['passages'] = new_p_list
            if func == 'reliability_phrase':
                line['local_check'] = local_check_str(line['llm_response'])
            if func == 'self_ask':
                llm_ans = line[src_key_name].lower()
                if 'yes' in llm_ans:
                    line[tgt_key_name] = True
                else:
                    line[tgt_key_name] = False
            if func == 'get_decompose':
                res, json_flag = json_decode(line['llm_response'])
                if json_flag:
                    line['sub_questions'] = res
                else:
                    line['sub_questions'] = {}
                    error_num += 1
            if func == 'map_decompose':
                subq_dict = id2subq_dict[line['Id']]
                s = line['passages']
                if len(subq_dict) > 0:
                    write_flag = False
                    # first search str in reference
                    for ent, qes in subq_dict.items():
                        if ent.lower() in s.lower():
                            new_line = copy.deepcopy(line)
                            new_line['sub_question'] = qes
                            output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')
                            write_flag = True
                    if write_flag:
                        continue
                    new_line = copy.deepcopy(line)
                    new_line['sub_question'] = new_line['Question']
                    output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')
                    continue
                else:
                    error_num += 1
                    new_line = copy.deepcopy(line)
                    new_line['sub_question'] = new_line['Question']
                    output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')
                    continue
            if func == 'count':
                error_num += 1
                continue
            if func == 'result_merge':
                if line['local_check'] == True:
                    line['llm_response'] = line['turn0_response']
                else:
                    kg_score = score_feature(line['llm_triple_score'], 0, entity_count=False)
                    if kg_score == None or kg_score >= 1:
                        line['llm_response'] = line['turn1_response']
                    else:
                        line['llm_response'] = line['turn1_response']

            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
        print(error_num)
        return error_num

def read_map(input_file_name, map_func):
    '''all map use id as key'''
    id2passage_dict = {}
    with open(input_file_name) as inpuf_f:
        for line in inpuf_f:
            line = json.loads(line.strip())
            if map_func == 'summary':
                passage2summary_dict = id2passage_dict.get(line['Id'], {})
                passage2summary_dict[line['passages']] = line['summary']
                id2passage_dict[line['Id']] = passage2summary_dict
            elif map_func == 'sub_question':
                id2passage_dict[line['Id']] = line['sub_questions']
            elif map_func == 'question_entity':
                id2passage_dict[line['Id']] = line['question_entity']
            elif map_func == 'pseudo_doc':
                id2passage_dict[line['Id']] = line['pseudo_doc']
            elif map_func == 'local_check':
                id2passage_dict[line['Id']] = line['local_check']

    return id2passage_dict

def merge_file(input_file_list:list, output_file_name, type='concat', merge_key={}, index_key='passages', have_choice=False):
    '''kc for entity & question
    concat\n
    merge\n
    reference_extend
    '''
    output_f = open(output_file_name, 'w')
    count_num = 0
    if type == 'concat':
        for input_file_name in input_file_list:
            # if not os.path.exists(input_file_name):
            #     continue
            with open(input_file_name, 'r') as input_f:
                for line in input_f:
                    output_f.write(line)
    elif type == 'merge':
        '''file_list = [src_file, merge_file]'''
        assert len(input_file_list) == 2
        file1_name, file2_name = input_file_list
        with open(file1_name) as file1, \
            open(file2_name) as file2:
            src_dict = {}
            for line in file1:
                line = json.loads(line.strip())
                if have_choice and (index_key == 'Question'):
                    key_word = line[index_key] + line['A'] + line['B'] + line['C'] + line['D']
                else:
                    key_word = line[index_key]
                src_dict[key_word] = line
            
            for line in file2:
                line = json.loads(line.strip())
                if have_choice and (index_key == 'Question'):
                    key_word = line[index_key] + line['A'] + line['B'] + line['C'] + line['D']
                else:
                    key_word = line[index_key]
                src_line = src_dict.get(key_word, None)
                if src_line == None:
                    continue
                for k,v in merge_key.items():
                    line[v] = src_line[k]
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
    elif type == 'reference_extend':
        assert len(input_file_list) == 2
        file1_name, file2_name = input_file_list
        with open(file1_name) as file1, \
            open(file2_name) as file2:
            src_dict = {}
            for line in file1:
                line = json.loads(line.strip())
                if have_choice and (index_key == 'Question'):
                    key_word = line[index_key] + line['A'] + line['B'] + line['C'] + line['D']
                else:
                    key_word = line[index_key]
                src_dict[key_word] = line
            
            for line in file2:
                line = json.loads(line.strip())
                if have_choice and (index_key == 'Question'):
                    key_word = line[index_key] + line['A'] + line['B'] + line['C'] + line['D']
                else:
                    key_word = line[index_key]
                src_line = src_dict.get(key_word, None)
                if src_line == None:
                    continue
                line['reference'].extend(src_line['reference'])
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # read_data('Truthful_QA', '/data/xkliu/LLMs/DocFixQA/datasets/truthfulqa_mc_task.json')
    # list2pair('/data/xkliu/LLMs/DocFixQA/result/TemporalQA/process_data/entity_knowledge-card-all_raw.json',
    #           '/data/xkliu/LLMs/DocFixQA/result/TemporalQA/process_data/entity_knowledge-card-all.json',
    #           'knowledge_card')
    # summary_dict = read_map('/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/summary_all.json', map_func='local')
    # pair_merge('/data/xkliu/LLMs/DocFixQA/result/TemporalQA/process_data/pseudoEntity_card_question_local_check.json',
    #              '/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/pseudoEntity_card_question_local_check_filter_dev.json',
    #              'filter', summary_map_dict=None)
    # subq_dict = read_map('/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/pseudo_doc_generate_question_entity.json', 'pseudo_doc')
    # process_by_line('/data/xkliu/LLMs/DocFixQA/result/TemporalQA/Llama/entity_knowledge-card-all_triples_woentity.json',
    #                 '/data/xkliu/LLMs/DocFixQA/result/TemporalQA/process_data/entity_knowledge-card-all_triples_woentity.json',
    #                 'triple_extract', id2subq_dict=None)
    # res, _ = triple_extraction_decode("I'll extract the relationships between the provided entities from the given text.\n\nAfter analyzing the text and the provided entities, I found the following relationships:\n\n[{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"got completed\", \"object\": \"None\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"must be created\", \"object\": \"murtis\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"made\", \"object\": \"murtis\"}, \n{\"subject\": \"shanthamadurai\", \"predicate\": \"is brought out\", \"object\": \"moon light\"}, \n{\"subject\": \"moon light\", \"predicate\": \"is brought\", \"object\": \"house\"}]\n\nNote that the entity \"reference\" is not related to any other entity in the text, so it's not included in the output. Also, the entity \"question\" is not directly related to any other entity, but it's mentioned in the context of \"lakshmi kalyanam\", so I didn't include it as a separate entity.\n\nThe entity \"purpose\" is related to the creation of murtis, but it's not a direct relationship, so I didn't include it as a separate triple. Similarly, the entity \"reason\" is mentioned as the same reason as they made murtis for kalyanam, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"ceremony\" is mentioned in the context of creating murtis, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"kalyanam\" is mentioned multiple times, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"light\" is mentioned as \"moon light\", but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"place\" is mentioned as the place where shanthamadurai was staying, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"house\" is mentioned as the place where the moon light is brought, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"moon\" is mentioned as the source of the moon light, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nSo, the final output is:\n\n[{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"got completed\", \"object\": \"None\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"must be created\", \"object\": \"murtis\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"made\", \"object\": \"murtis\"}, \n{\"subject\": \"shanthamadurai\", \"predicate\": \"is brought out\", \"object\": \"moon light\"}, \n{\"subject\": \"moon light\", \"predicate\": \"is brought\", \"object\": \"house\"}]I'll extract the relationships between the provided entities from the given text.\n\nAfter analyzing the text and the provided entities, I found the following relationships:\n\n[{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"got completed\", \"object\": \"None\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"must be created\", \"object\": \"murtis\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"made\", \"object\": \"murtis\"}, \n{\"subject\": \"shanthamadurai\", \"predicate\": \"is brought out\", \"object\": \"moon light\"}, \n{\"subject\": \"moon light\", \"predicate\": \"is brought\", \"object\": \"house\"}]\n\nNote that the entity \"reference\" is not related to any other entity in the text, so it's not included in the output. Also, the entity \"question\" is not directly related to any other entity, but it's mentioned in the context of \"lakshmi kalyanam\", so I didn't include it as a separate entity.\n\nThe entity \"purpose\" is related to the creation of murtis, but it's not a direct relationship, so I didn't include it as a separate triple. Similarly, the entity \"reason\" is mentioned as the same reason as they made murtis for kalyanam, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"ceremony\" is mentioned in the context of creating murtis, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"kalyanam\" is mentioned multiple times, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"light\" is mentioned as \"moon light\", but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"place\" is mentioned as the place where shanthamadurai was staying, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"house\" is mentioned as the place where the moon light is brought, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"moon\" is mentioned as the source of the moon light, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nSo, the final output is:\n\n[{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"got completed\", \"object\": \"None\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"must be created\", \"object\": \"murtis\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"made\", \"object\": \"murtis\"}, \n{\"subject\": \"shanthamadurai\", \"predicate\": \"is brought out\", \"object\": \"moon light\"}, \n{\"subject\": \"moon light\", \"predicate\": \"is brought\", \"object\": \"house\"}]")
    # print(res)

    import os

    dataset_path = '/data/xkliu/LLMs/DocFixQA/datasets/2wikimultihopQA'
    input_path = '/data/xkliu/LLMs/DocFixQA/result/hotpotQA/gpt-4o-mini/baseline'
    input_dir = 'recitation_0324'
    output_path = '/data/xkliu/LLMs/DocFixQA/datasets/hotpotQA'
    output_dir = 'baseline_gpt4omini_recitation_0324'
    kc_name = 'knowledge-card-wikipedia'
    ref_src = 'triple_score'

    '''hotpotQA'''
    input_file_list = [os.path.join(input_path, f) for f in os.listdir(os.path.join(input_path))]
    # Turn 0 
    self_inner_file_list = [os.path.join(input_path, 'gpt4omini_turn0_{}.json'.format(ref_src))]
    self_extra_file_list = [os.path.join(input_path, 'pseudo_entity_gpt4omini_turn0_{}.json'.format(ref_src))]

    kc_inner_file_list = [os.path.join(input_path, 'knowledge-card-wikidata_turn0_question_{}.json'.format(ref_src)),
                          os.path.join(input_path, 'knowledge-card-wikipedia_turn0_question_{}.json'.format(ref_src)),
                          os.path.join(input_path, 'knowledge-card-yago_turn0_question_{}.json'.format(ref_src)),]
    kc_extra_file_list = [os.path.join(input_path, 'knowledge-card-wikidata_turn0_entity_{}.json'.format(ref_src)),
                          os.path.join(input_path, 'knowledge-card-wikipedia_turn0_entity_{}.json'.format(ref_src)),
                          os.path.join(input_path, 'knowledge-card-yago_turn0_entity_{}.json'.format(ref_src)),]
    llm_inner_file_list = [os.path.join(input_path, 'Llama_turn0_{}.json'.format(ref_src)),
                           os.path.join(input_path, 'Qwen_turn0_{}.json'.format(ref_src)),]
    llm_extra_file_list = [os.path.join(input_path, 'pseudo_entity_Llama_turn0_{}.json'.format(ref_src)),
                           os.path.join(input_path, 'pseudo_entity_Qwen_turn0_{}.json'.format(ref_src)),]
    # Turn 1
    turn1_self_inner_file_list = [os.path.join(input_path, 'gpt4omini_turn1_{}.json'.format(ref_src))]
    turn1_self_extra_file_list = [os.path.join(input_path, 'pseudo_entity_gpt4omini_turn1_{}.json'.format(ref_src))]

    turn1_kc_inner_file_list = [os.path.join(input_path, 'knowledge-card-wikidata_turn1_question_{}.json'.format(ref_src)),
                          os.path.join(input_path, 'knowledge-card-wikipedia_turn1_question_{}.json'.format(ref_src)),
                          os.path.join(input_path, 'knowledge-card-yago_turn1_question_{}.json'.format(ref_src)),]
    turn1_kc_extra_file_list = [os.path.join(input_path, 'knowledge-card-wikidata_turn1_entity_{}.json'.format(ref_src)),
                          os.path.join(input_path, 'knowledge-card-wikipedia_turn1_entity_{}.json'.format(ref_src)),
                          os.path.join(input_path, 'knowledge-card-yago_turn1_entity_{}.json'.format(ref_src)),]
    turn1_llm_inner_file_list = [os.path.join(input_path, 'Llama_turn1_{}.json'.format(ref_src)),
                           os.path.join(input_path, 'Qwen_turn1_{}.json'.format(ref_src)),]
    turn1_llm_extra_file_list = [os.path.join(input_path, 'pseudo_entity_Llama_turn1_{}.json'.format(ref_src)),
                           os.path.join(input_path, 'pseudo_entity_Qwen_turn1_{}.json'.format(ref_src)),]


    input_file_list = []
    # input_file_list.extend(self_inner_file_list)
    # input_file_list.extend(self_extra_file_list)
    # input_file_list.extend(kc_inner_file_list)
    # input_file_list.extend(kc_extra_file_list)
    # input_file_list.extend(llm_inner_file_list)
    # input_file_list.extend(llm_extra_file_list)

    input_file_list.extend(turn1_self_inner_file_list)
    input_file_list.extend(turn1_self_extra_file_list)
    input_file_list.extend(turn1_kc_inner_file_list)
    input_file_list.extend(turn1_kc_extra_file_list)
    input_file_list.extend(turn1_llm_inner_file_list)
    input_file_list.extend(turn1_llm_extra_file_list)

    input_file_list = [os.path.join(input_path, 'turn1_reference_local_check.json'),
                     os.path.join(input_path, 'turn1_reference_triple_score.json'),]
    
    input_file_list = ['/data/xkliu/LLMs/DocFixQA/reference/PopQA/test/gpt4omini_turn1_rag_top6.json',
                       '/data/xkliu/LLMs/DocFixQA/datasets/PopQA/gpt4omini_turn0_rag.json',]
    
    # input_file_list = [os.path.join(input_path,  'baseline_CoT_{}.json'.format(i)) for i in range(4)]
    
    input_file_name = os.path.join(input_path,'{}.json'.format(input_dir))
    output_file_name = os.path.join(output_path, '{}.json'.format(output_dir))
    
    # list2pair(input_file_name, output_file_name, 'knowledge_card', line_mode=True)
    # merge_file(input_file_list, output_file_name, type='reference_extend', merge_key={'local_check': 'local_check'}, index_key='id')
    process_by_line(input_file_name, output_file_name, 'reference', tgt_key_name='passages', src_key_name='llm_response')
    # pair_merge(input_file_name, output_file_name, 'filter', dataset='hotpotQA', top_k=6)
    exit()

    '''MMLU'''
    #load src dir
    subjects = sorted([f.split("_dev.json")[0] for f in os.listdir(os.path.join(dataset_path, "dev")) if "_dev.json" in f])

    # mkdir save dir
    save_dir = os.path.join(output_path, output_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    all_num = 0
    for sub in subjects:
        # if sub != 'moral_scenarios':
        #     continue
        print(sub)
        input_file_name = os.path.join(input_path, 'dev', input_dir, sub + "_dev.json")
        # input_file_list = [
        #     os.path.join(input_path, 'dev', 'entity', "{}_knowledge-card-wikipedia.json".format(sub)),
        #     os.path.join(input_path, 'dev', 'entity', "{}_knowledge-card-yago.json".format(sub)),
        #     os.path.join(input_path, 'dev', 'pseudo_doc', "{}_pseudo_doc.json".format(sub)),
        #     os.path.join(input_path, 'dev', 'question', "{}_knowledge-card-wikipedia.json".format(sub)),
        #     os.path.join(input_path, 'dev', 'question', "{}_knowledge-card-yago.json".format(sub)),
        #     os.path.join(input_path, 'dev', 'choice', "{}_knowledge-card-wikipedia.json".format(sub)),
        #     os.path.join(input_path, 'dev', 'choice', "{}_knowledge-card-yago.json".format(sub))
        # ]
        input_file_list = [os.path.join(input_path,  'question_el_{}_kc.json'.format(i)) for i in range(4)]

        output_file_name = os.path.join(save_dir, "{}.json".format(sub))

        merge_file(input_file_list, output_file_name, type='concat', merge_key={'llm_response': 'turn1_response', 'reference':'reference'}, index_key='Question')
        # all_num += process_by_line(input_file_name, output_file_name, 'triple_extract', tgt_key_name='llm_triple', src_key_name='llm_response')
        # list2pair(input_file_name, output_file_name, 'knowledge_card', line_mode=True)
        # pair_merge(input_file_name, output_file_name, 'filter', dataset='MMLU', summary_map_dict=None)
    
    print(all_num)