import json
from tqdm import tqdm
'''OpenQA follow skr $ midtermQA'''

from collections import Counter
import string
import re
import json
import spacy
import numpy as np
# from scipy import stats

# nlp = spacy.load('en_core_web_md')

# # for mc1 sim caculate
# import torch
# torch.set_num_threads(10)
# from sentence_transformers import SentenceTransformer
# sim_model = SentenceTransformer('/data/xkliu/hf_models/all-mpnet-base-v2')

import argparse

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_answer_ref(prediction, answer):
    '''ToDo: our phrase methods with json format'''
    json_pattern = r'\{.*?\}'
    match = re.search(json_pattern, prediction, re.DOTALL)  # re.DOTALL 允许 . 匹配换行符

    if match:
        json_str = match.group()
        try:
            json_dict = json.loads(json_str)
            if json_dict['answer'] == None:
                return ''
            return json_dict['answer']
        except:
            prediction = json_str   
    # if have answer, return answer
    # if not or phrase badly, return the last sentence as we first output reason
    nlp_text = nlp(prediction).sents
    sentences = [str(sent).strip() for sent in nlp_text]
    for sent in sentences:
        if answer.lower() in sent.lower():
            return answer
    if len(sentences) > 0:
        return sentences[-1].strip()
    else:
        return ''

def find_options_positions(text, options=['a', 'b', 'c', 'd']):
    positions = {}
    
    for option in options:
        match = re.search(rf'{option}', text)
        if match:
            positions[option] = match.start()
    
    first_option = min(positions, key=positions.get) if positions else None

    return positions, first_option

def eval_line(line, dataset, model_name, answer_key, firstorlast='first'):
    if dataset == 'TruthfulQA':
        ans = line[answer_key]
        candidates = line['mc1_targets']
        if model_name == 'Mistral':
            start_str = '[/INST]'
            start_pos = ans.rfind(start_str)
            ans = ans[start_pos:].lstrip(start_str).strip()
        ans_pos = -1
        choice = None
        for cand in candidates.keys():
            if firstorlast == 'last':
                cand_pos = ans.rfind(cand)
                if cand_pos == -1:
                    continue
                if ans_pos == -1:
                    ans_pos = cand_pos
                    choice = cand
                else:
                    if cand_pos > ans_pos:
                        ans_pos = cand_pos
                        choice = cand
                    elif cand_pos == ans_pos:
                        if len(choice) < len(cand):
                            choice = cand
            else:   # first
                cand_pos = ans.find(cand)
                if cand_pos == -1:
                    continue
                if ans_pos == -1:
                    ans_pos = cand_pos
                    choice = cand
                else:
                    if cand_pos < ans_pos:
                        ans_pos = cand_pos
                        choice = cand
                    elif cand_pos == ans_pos:
                        if len(choice) < len(cand):
                            choice = cand

        # if no direct choice, we use max_sim
        if choice == None:
            ans_list_raw = ans.split('\n')
            ans_list = []
            for ans in ans_list_raw:
                if len(ans) == 0:
                    continue
                ans_list.append(ans)
            cand_list = list(candidates.keys())
            ans_embed = sim_model.encode(ans_list)
            cand_embed = sim_model.encode(cand_list)
            sim = sim_model.similarity(cand_embed, ans_embed)
            sim, _ = torch.max(sim, dim=1)
            cand_num = torch.argmax(sim)
            choice = cand_list[cand_num]

        # eval result
        if candidates[choice] == 1:
            return choice, True
        else:
            return choice, False
    
    if dataset == 'MMLU':
        # multi choice
        error_flag = True
        possible_prefix = ["The best answer is ", "answer:", 'answer is:', 'answer is ']
        pred = line[answer_key]
        for prefix in possible_prefix:
            if prefix in pred.lower():
                idx = pred.lower().rfind(prefix)
                # print ("extracted ans string: ", pred[idx + len(prefix) : ])
                pred_ans = pred[idx + len(prefix) : ]
                pred_ans = pred_ans.strip()
                if len(pred_ans) > 0:
                    error_flag = False
                    break

        if error_flag:
            pred_ans = pred.strip()
            '''logits'''
            pred_choice = pred_ans
            '''text'''
            _, pred_choice = find_options_positions(pred_ans.lower(), ['a.', 'b.', 'c.', 'd.'])
            if pred_choice != None:
                pred_choice = pred_choice[0]
            else:
                _, pred_choice = find_options_positions(pred_ans.lower())
        else:
            _, pred_choice = find_options_positions(pred_ans.lower())
    
        if pred_choice != None:
            return pred_ans, pred_choice.upper() == line['Answer'], error_flag
        else:
            return pred_ans, False, error_flag

def eval_file(file_name, dataset, model_name, answer_key):
    '''
    TruthfulQA mc1
    TemporalQA annwer_key is needed
    '''
    if dataset == 'TruthfulQA':
        '''acc'''
        all_num = 0
        hit_num = 0
        with open(file_name) as input_f:
            for line in tqdm(input_f):
                line = json.loads(line.strip())
                all_num += 1
                choice, flag = eval_line(line, dataset, model_name, answer_key)
                if flag:
                    hit_num += 1
        print('{}.{} Acc for mc1: {}'.format(dataset, model_name, hit_num / all_num))
    
    elif dataset == 'TemporalQA':
        cot=[]
        test_file = open(file_name, 'r')
        f1 = exact_match = total = 0
        local_true_P, local_false_P ,local_true_N, local_false_N = 0, 0, 0, 0
        for idx, test_case in tqdm(enumerate(test_file)):
            test_case = json.loads(test_case)
            total += 1
            prediction = test_case[answer_key]
            ground_truths = test_case['Gold answer']

            prediction = prediction.replace('\n', '').strip()
            prediction = get_answer_ref(prediction, ground_truths[0].strip())
            
            cot.append(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths)
            if metric_max_over_ground_truths(exact_match_score, prediction, ground_truths):
                if test_case['local_check'] == True:
                    local_true_P += 1
                else:
                    local_false_P += 1
            else:
                if test_case['local_check'] == True:
                    local_true_N += 1
                else:
                    local_false_N += 1
            f1 += metric_max_over_ground_truths(
                f1_score, prediction, ground_truths)

        print('{}.{} Acc for EM: {}'.format(dataset, model_name, exact_match/total))
        print('{}.{} Acc for F1: {}'.format(dataset, model_name, f1/total))
        print(local_true_P, local_true_N, local_false_P, local_false_N)

    elif dataset == 'MMLU':
        # file_name -> file_path
        import os
        from mmlu_categories import subcategories, categories
        
        #load test dir
        subjects = sorted([f.split("_result.json")[0] for f in os.listdir(os.path.join(file_name)) if "_result.json" in f])
        result_dict = {}
        all_sub_dict = {}
        for sub in subjects:
            for c, c_list in categories.items():
                if len(set(subcategories[sub]) & set(c_list)) != 0:
                    cat = c
                    break

            input_file_name = os.path.join(file_name, sub + "_result.json")
            input_file = open(input_file_name)
            for line in tqdm(input_file):
                line = json.loads(line)
                pred, cor, err = eval_line(line, dataset, model_name, answer_key)
                cat_dict = result_dict.get(cat, {})
                cat_dict['all_num'] = cat_dict.get('all_num', 0) + 1
                if cor:
                    cat_dict['correct_num'] = cat_dict.get('correct_num', 0) + 1
                if err:
                    cat_dict['error_num'] = cat_dict.get('error_num', 0) + 1
                    # print('-'*50)
                    # print(pred)
                    # if cat_dict['error_num'] > 10:
                    #     exit()
                result_dict[cat] = cat_dict
        
        all_num, cor_num, err_num = 0, 0, 0
        acc_list = []
        for sub, v_dict in result_dict.items():
            print('{}: All: {}, Correct: {}, Error: {}, Acc: {:.3f}'.format(sub, v_dict['all_num'], v_dict['correct_num'], v_dict['error_num'], v_dict['correct_num'] / v_dict['all_num']))
            all_num += v_dict['all_num']
            cor_num += v_dict['correct_num']
            err_num += v_dict['error_num']
            acc_list.append(v_dict['correct_num'] / v_dict['all_num'])
        
        print('Micro: All: {}, Correct: {}, Error: {}, Acc: {:.3f}'.format(all_num, cor_num, err_num, cor_num / all_num))
        print('4 Categories Macro Acc: {:.3f}'.format(sum(acc_list) / len(acc_list)))
        # print('All Sub Categories Macro Acc: {:.3f}'.format(sum(acc_list) / len(acc_list)))

    else:
        print('Not supported Datasets')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocFixQA args')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help="Dataset Name for Test")
    parser.add_argument('--model_name', '-m', type=str, required=True, help='Model Name for Process')
    parser.add_argument('--file_path','-f',type=str, required=True, help="Path to Eval File")
    parser.add_argument('--answer_key','-k',type=str, required=True, help="Answer Key in File")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    file_path = args.file_path
    answer_key = args.answer_key

    eval_file(file_path , dataset_name, model_name, answer_key)
    