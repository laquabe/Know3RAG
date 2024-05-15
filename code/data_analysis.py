from eval import eval_line
import pandas as pd
from tqdm import tqdm
import json

def csv2dict(filename):
    '''key:question; value:type&category'''
    df = pd.read_csv(filename)
    new_dict = {}
    for index, row in tqdm(df.iterrows()):
        new_dict[row['Question'].strip()] = {'Type':row['Type'], 'Category':row['Category']}
    return new_dict

def badcase_analysis(dataset, model, result_file, src_file=None):
    if dataset == 'TruthfulQA':
        src_dict = csv2dict(src_file)
        
        res_dict = {}
        with open(result_file) as input_f:
            for line in tqdm(input_f):
                line = json.loads(line.strip())
                choice, flag = eval_line(line, dataset=dataset, model=model)
                line_category = src_dict[line['question']]['Category']
                cate_dict = res_dict.get(line_category, {'all':0, 'hit':0})
                cate_dict['all'] = cate_dict['all'] + 1
                if flag:
                    cate_dict['hit'] = cate_dict['hit'] + 1
                res_dict[line_category] = cate_dict
        for k, v in res_dict.items():
            print('{} Acc: {:.4f}'.format(k, v['hit']/v['all']))     

if __name__ == '__main__':
    src_file = '/data/xkliu/LLMs/DocFixQA/datasets/TruthfulQA/TruthfulQA.csv'
    res_file = '/data/xkliu/LLMs/DocFixQA/result/Mistral_7B_raw.json'
    dataset = 'TruthfulQA'
    model = 'Mistral'
    badcase_analysis(dataset, model, res_file, src_file)
    