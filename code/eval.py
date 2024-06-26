import json
from tqdm import tqdm
'''OpenQA follow skr $ midtermQA'''

from collections import Counter
import string
import re
import json
import spacy
import numpy as np
from scipy import stats

nlp = spacy.load('en_core_web_md')

# for mc1 sim caculate
import torch
torch.set_num_threads(10)
from sentence_transformers import SentenceTransformer
sim_model = SentenceTransformer('/data/xkliu/hf_models/all-mpnet-base-v2')

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
            json_str = json.loads(json_str)
            if json_str['answer'] == None:
                return ''
            return json_str['answer']
        except:
            prediction = json_str   
    # if have answer, return answer
    # if not or phrase badly, return the first sentence (here we may use last)
    nlp_text = nlp(prediction).sents
    sentences = [str(sent).strip() for sent in nlp_text]
    for sent in sentences:
        if answer.lower() in sent.lower():
            return answer
    return sentences[0].strip()

def eval_line(line, dataset, model_name, answer_key,firstorlast='first'):
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
            f1 += metric_max_over_ground_truths(
                f1_score, prediction, ground_truths)

        print('{}.{} Acc for EM: {}'.format(dataset, model_name, exact_match/total))
        print('{}.{} Acc for F1: {}'.format(dataset, model_name, f1/total))
    
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
    