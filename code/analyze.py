import json
from tqdm import tqdm
import re

import matplotlib.pyplot as plt
import numpy as np

def draw_bar(values1, values2, categories, save_path):
    # categories = ['A', 'B', 'C', 'D']
    # values1 = [20, 35, 30, 35]
    # values2 = [25, 32, 34, 20]

    bar_width = 0.35

    x = np.arange(len(categories))

    bars1 = plt.bar(x - bar_width / 2, values1, bar_width, label='Tunr0 True', color='blue')
    bars2 = plt.bar(x + bar_width / 2, values2, bar_width, label='Merge True', color='orange')

    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, str(yval), ha='center', va='bottom')

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, str(yval), ha='center', va='bottom')

    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Double Column Bar Chart')
    plt.xticks(x, categories)
    plt.legend()

    plt.savefig(save_path)

def compare_entity_cover(input_file_name):
    num_dict = {}
    with open(input_file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            llm_ent = line['sub_questions']
            model_ent = line['question_entity']
            llm_ent = set([ent.lower() for ent in llm_ent.keys()])
            model_ent = set([ent.lower() for ent in model_ent.keys()])
            num_together = len(llm_ent & model_ent)
            if num_together == 0:
                print(json.dumps(line, ensure_ascii=False))
            num_dict[num_together] = num_dict.get(num_together, 0) + 1
    
    # for k, v in num_dict.items():
    #     print('{}:{}'.format(k, v))

def find_options_positions(text, options=['a', 'b', 'c', 'd']):
    positions = {}
    
    for option in options:
        match = re.search(rf'{option}', text)
        if match:
            positions[option] = match.start()
    
    first_option = min(positions, key=positions.get) if positions else None

    return positions, first_option

def eval_line(line, dataset, model_name, answer_key, firstorlast='first'):
    if dataset == 'MMLU':
        # multi choice
        error_flag = True
        possible_prefix = ["The best answer is ", "the best answer is ", "answer:", 'answer is:', 'answer is ']
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
            # print(pred)
            # exit()
            # '''logits'''
            # pred_choice = pred_ans
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

def score_feature(score_list:list):
    feature_score_list = []
    for triple in score_list:
        # feature_score_list.append(triple['triple_score'])
        ref_avg = np.average(triple['ref_score'])
        score = np.abs(triple['triple_score'] - ref_avg)
        feature_score_list.append(score)
    
    return np.min(feature_score_list)

def score_category(score):
    if score < 0:
        return 0
    elif score <= 1:
        return 1
    elif score <= 5:
        return 2
    elif score <= 10:
        return 3
    elif score <= 50:
        return 4
    else:
        return 5

def eval_file(file_name, dataset, model_name):
    '''
    TruthfulQA mc1
    TemporalQA annwer_key is needed
    '''
    if dataset == 'MMLU':
        # file_name -> file_path
        import os
        from mmlu_categories import subcategories, categories
        
        #load test dir
        subjects = sorted([f.split("_result.json")[0] for f in os.listdir(os.path.join(file_name)) if "_result.json" in f])
        result_dict = {}
        all_sub_dict = {}

        x_set = ['< 0', '0 - 1' , '1 - 5', '5 - 10', '10-50', '>50','None']
        true_score_list = [0] * len(x_set)
        false_score_list = [0] * len(x_set)
        ref0_num = 0

        for sub in subjects:
            for c, c_list in categories.items():
                if len(set(subcategories[sub]) & set(c_list)) != 0:
                    cat = c
                    break

            input_file_name = os.path.join(file_name, sub + "_result.json")
            input_file = open(input_file_name)
            for line in tqdm(input_file):
                line = json.loads(line)
                pred_1, cor_1, err_1 = eval_line(line, dataset, model_name, 'turn0_response')
                pred_2, cor_2, err_2 = eval_line(line, dataset, model_name, 'turn1_response')
                # pred_3, cor_3, err_3 = eval_line(line, dataset, model_name, 'llm_response')
                cat_dict = result_dict.get(cat, {})
                if cor_1 & cor_2 :
                    # if len(line['llm_triple_score']) == 0:
                    #     true_score_list[-1] = true_score_list[-1] + 1
                    # else:
                    #     score = score_feature(line['llm_triple_score'])
                    #     score_index = score_category(score)
                    #     true_score_list[score_index] = true_score_list[score_index] + 1
                    cat_dict['correct_num'] = cat_dict.get('correct_num', 0) + 1
                elif cor_1:
                    cat_dict['turn0_correct_num'] = cat_dict.get('turn0_correct_num', 0) + 1
                    if len(line['reference']) == 0:
                        ref0_num += 1
                    if line['local_check'] == False:
                        continue
                    if len(line['llm_triple_score']) == 0:
                        true_score_list[-1] = true_score_list[-1] + 1
                    else:
                        score = score_feature(line['llm_triple_score'])
                        score_index = score_category(score)
                        true_score_list[score_index] = true_score_list[score_index] + 1
                elif cor_2:
                    if len(line['reference']) == 0:
                        ref0_num -= 1
                    cat_dict['turn1_correct_num'] = cat_dict.get('turn1_correct_num', 0) + 1
                    if line['local_check'] == False:
                        continue
                    if len(line['llm_triple_score']) == 0:
                        false_score_list[-1] = false_score_list[-1] + 1
                    else:
                        score = score_feature(line['llm_triple_score'])
                        score_index = score_category(score)
                        false_score_list[score_index] = false_score_list[score_index] + 1
                else:
                    # if len(line['llm_triple_score']) == 0:
                    #     false_score_list[-1] = false_score_list[-1] + 1
                    # else:
                    #     score = score_feature(line['llm_triple_score'])
                    #     score_index = score_category(score)
                    #     false_score_list[score_index] = false_score_list[score_index] + 1
                    cat_dict['false_num'] = cat_dict.get('false_num', 0) + 1
                result_dict[cat] = cat_dict

        # draw pic
        draw_bar(true_score_list, false_score_list, x_set, 'pic/std_min_Turn1Filter_LocalChcek_True_0103.png')
        
        correct_num, cor1_num, cor2_num, false_num = 0, 0, 0, 0
        acc_list = []
        for sub, v_dict in result_dict.items():
            print('{}: Correct_Both: {}, Correct_turn0: {}, Correct_turn1: {}, False: {}'.format(sub, v_dict['correct_num'], v_dict['turn0_correct_num'], v_dict['turn1_correct_num'], v_dict['false_num']))
            correct_num += v_dict['correct_num']
            cor1_num += v_dict['turn0_correct_num']
            cor2_num += v_dict['turn1_correct_num']
            false_num += v_dict['false_num']
        
        print('Micro: Correct_Both: {}, Correct_turn0: {}, Correct_turn1: {}, False: {}, Acc: {:.3f}'.format(correct_num, cor1_num, cor2_num, false_num, (correct_num + cor1_num + cor2_num) / (correct_num + cor1_num + cor2_num + false_num)))
        # print('4 Categories Macro Acc: {:.3f}'.format(sum(acc_list) / len(acc_list)))
        # print('All Sub Categories Macro Acc: {:.3f}'.format(sum(acc_list) / len(acc_list)))
        print(ref0_num)

    else:
        print('Not supported Datasets')

if __name__ == '__main__':
    eval_file('/data/xkliu/LLMs/DocFixQA/result/MMLU/turn01_filter_rag_feature/', dataset='MMLU', model_name='Llama')
