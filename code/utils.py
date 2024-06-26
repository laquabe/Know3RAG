import json
import copy
import re

def read_data(dataset_name, file_path):
    if dataset_name in ['Truthful_QA', 'Temporal_QA']:
        with open(file_path) as f:
            data = f.read()
            data = json.loads(data)
            return data
        if dataset_name in ['PopQA']:
            pass

def list2pair(input_file_name, output_file_name, phrase_key, line_mode=True):
    with open(input_file_name) as input_f, \
        open(output_file_name, 'w') as output_f:
        if not line_mode:
            data = input_f.read()
            input_f = json.loads(data)

        for line in input_f:
            if line_mode:
                line = json.loads(line.strip())
            if phrase_key == 'passages':
                src = line['passages']
                src = src[0][0]
                for s in src:
                    new_line = copy.deepcopy(line)
                    del new_line['passages']
                    new_line['passages'] = s
                    output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')

def filter_exinfo(prediction, firstorlast='last'):
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
            if json_str['useful'] == 'no':
                return False
            return True
        except:
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

def pair_merge(input_file_name, output_file_name, func):
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
                del new_l['llm_response']
                id2src[l_id] = new_l
                id2info[l_id] = []
            if func == 'filter':
                # if no then ignore the exinfo
                if filter_exinfo(line['llm_response']):
                    ex_info_list = id2info[l_id]
                    ex_info_list.append(line['passages'])
                    id2info[l_id] = ex_info_list
            if func == 'summary':
                pass
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

def summary_process(line:dict):
    res_json, flag = json_decode(line['llm_response'])
    if flag:
        return res_json['summary'], flag
    return res_json, flag

def process_by_line(input_file_name, output_file_name, func):
    with open(input_file_name) as input_f, \
        open(output_file_name, 'w') as output_f:
        error_num = 0
        for line in input_f:
            line = json.loads(line.strip())
            if func == 'summary':
                summary_text, flag = summary_process(line)
                if flag:
                    line['summary'] = summary_text
                else:
                    line['summary'] = line['passages']
                    error_num += 1
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
        print(error_num)


if __name__ == "__main__":
    # read_data('Truthful_QA', '/data/xkliu/LLMs/DocFixQA/datasets/truthfulqa_mc_task.json')
    # list2pair('/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/dev.json',
    #           '/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/passage_pair_dev.json',
    #           'passages', line_mode=False)
    pair_merge('/data/xkliu/LLMs/DocFixQA/result/TemporalQA/process_data/summary.json',
                 '/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/summary_dev.json',
                 'raw')
    # process_by_line('/data/xkliu/LLMs/DocFixQA/result/TemporalQA/Llama/summary.json',
    #                 '/data/xkliu/LLMs/DocFixQA/result/TemporalQA/process_data/summary.json',
    #                 'summary')