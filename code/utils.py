import json
import copy
import re
from tqdm import tqdm

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
                    del new_line['passages']
                    del new_line['generate']
                    # s = s.split('.')
                    # s = '.'.join(s[:-1])
                    # s += '.'
                    # if len(s) == 0:
                    #     continue
                    new_line['passages'] = s
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

def pair_merge(input_file_name, output_file_name, func, summary_map_dict = None):
    '''
    raw: just pair to list. input file is the summary file(processed)
    filter: use llm pair judge to filter external info. input file is the filter file(raw)
    '''
    id2src = {} #no_change
    id2info = {}    #change info

    with open(input_file_name) as input_f, \
        open(output_file_name, 'w') as output_f:

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
    json_pattern = r'\[.*\]'
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
        if 'subject' not in t.keys():
            continue
        if 'object' not in t.keys():
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

def process_by_line(input_file_name, output_file_name, func, id2subq_dict=None, tgt_key_name=None):
    '''
    summary: phrase llm response to key summary\n
    map_decompose: deliver sub-question to passages\n
    reference: phrase reference\n
    add_reference: add pseudo reference as passages\n
    add_question_entity: add_question_entity\n
    reliability_phrase: local_ckeck\n
    add_key: add src to tgt key\n
    triple_extract: phrase llm triple extraction
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
            if func == 'reference':
                res, json_flag = json_decode(line['llm_response'])
                if json_flag:
                    line['pseudo_doc'] = res['reference_paragraph']
                else:
                    line['pseudo_doc'] = line['Question']
                    error_num += 1
            if func == 'triple_extract':
                res, json_flag = triple_extraction_decode(line['llm_response'])
                if json_flag:
                    res = triple_verication(res)
                    line['llm_triple'] = res
                else:
                    line['llm_triple'] = []
                    error_num += 1
            if func == 'add_reference':
                line['passages'] = line['pseudo_doc_entity']
            if func == 'add_question_entity':
                line['question_entity'] = id2subq_dict[line['Id']]
            if func == 'add_key':
                line[tgt_key_name] = id2subq_dict[line['Id']]
            if func == 'add_pseudo_doc_simple':
                passages = line['passages'][0][0]
                new_p_list = []
                for p in passages:
                    new_p_list.append(id2subq_dict[line['Id']][p])
                new_p_list.append(line['pseudo_doc'])
                line['passages'] = new_p_list
            if func == 'reliability_phrase':
                if filter_exinfo(line['llm_response'], check_key='reliability'):
                    line['local_check'] = True
                else:
                    line['local_check'] = False
                    error_num += 1
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

            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
        print(error_num)

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
    process_by_line('/data/xkliu/LLMs/DocFixQA/result/TemporalQA/Llama/entity_knowledge-card-all_triples_woentity.json',
                    '/data/xkliu/LLMs/DocFixQA/result/TemporalQA/process_data/entity_knowledge-card-all_triples_woentity.json',
                    'triple_extract', id2subq_dict=None)
    # res, _ = triple_extraction_decode("I'll extract the relationships between the provided entities from the given text.\n\nAfter analyzing the text and the provided entities, I found the following relationships:\n\n[{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"got completed\", \"object\": \"None\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"must be created\", \"object\": \"murtis\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"made\", \"object\": \"murtis\"}, \n{\"subject\": \"shanthamadurai\", \"predicate\": \"is brought out\", \"object\": \"moon light\"}, \n{\"subject\": \"moon light\", \"predicate\": \"is brought\", \"object\": \"house\"}]\n\nNote that the entity \"reference\" is not related to any other entity in the text, so it's not included in the output. Also, the entity \"question\" is not directly related to any other entity, but it's mentioned in the context of \"lakshmi kalyanam\", so I didn't include it as a separate entity.\n\nThe entity \"purpose\" is related to the creation of murtis, but it's not a direct relationship, so I didn't include it as a separate triple. Similarly, the entity \"reason\" is mentioned as the same reason as they made murtis for kalyanam, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"ceremony\" is mentioned in the context of creating murtis, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"kalyanam\" is mentioned multiple times, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"light\" is mentioned as \"moon light\", but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"place\" is mentioned as the place where shanthamadurai was staying, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"house\" is mentioned as the place where the moon light is brought, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"moon\" is mentioned as the source of the moon light, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nSo, the final output is:\n\n[{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"got completed\", \"object\": \"None\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"must be created\", \"object\": \"murtis\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"made\", \"object\": \"murtis\"}, \n{\"subject\": \"shanthamadurai\", \"predicate\": \"is brought out\", \"object\": \"moon light\"}, \n{\"subject\": \"moon light\", \"predicate\": \"is brought\", \"object\": \"house\"}]I'll extract the relationships between the provided entities from the given text.\n\nAfter analyzing the text and the provided entities, I found the following relationships:\n\n[{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"got completed\", \"object\": \"None\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"must be created\", \"object\": \"murtis\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"made\", \"object\": \"murtis\"}, \n{\"subject\": \"shanthamadurai\", \"predicate\": \"is brought out\", \"object\": \"moon light\"}, \n{\"subject\": \"moon light\", \"predicate\": \"is brought\", \"object\": \"house\"}]\n\nNote that the entity \"reference\" is not related to any other entity in the text, so it's not included in the output. Also, the entity \"question\" is not directly related to any other entity, but it's mentioned in the context of \"lakshmi kalyanam\", so I didn't include it as a separate entity.\n\nThe entity \"purpose\" is related to the creation of murtis, but it's not a direct relationship, so I didn't include it as a separate triple. Similarly, the entity \"reason\" is mentioned as the same reason as they made murtis for kalyanam, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"ceremony\" is mentioned in the context of creating murtis, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"kalyanam\" is mentioned multiple times, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"light\" is mentioned as \"moon light\", but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"place\" is mentioned as the place where shanthamadurai was staying, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"house\" is mentioned as the place where the moon light is brought, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nThe entity \"moon\" is mentioned as the source of the moon light, but it's not a direct relationship, so I didn't include it as a separate triple.\n\nSo, the final output is:\n\n[{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"got completed\", \"object\": \"None\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"must be created\", \"object\": \"murtis\"}, \n{\"subject\": \"lakshmi kalyanam\", \"predicate\": \"made\", \"object\": \"murtis\"}, \n{\"subject\": \"shanthamadurai\", \"predicate\": \"is brought out\", \"object\": \"moon light\"}, \n{\"subject\": \"moon light\", \"predicate\": \"is brought\", \"object\": \"house\"}]")
    # print(res)