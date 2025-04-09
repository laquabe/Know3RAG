import transformers
import json
from tqdm import tqdm
from utils import read_data

max_new_tokens = 128    # default 100
task = 'question'
card_name = 'knowledge-card-yago'   # knowledge-card-1btokens, knowledge-card-atomic, knowledge-card-reddit, knowledge-card-wikidata, knowledge-card-wikipedia, knowledge-card-yago
card_path = '/data/xkliu/LLMs/Knowledge_Card-main/cards/{}'.format(card_name)
card_device = 7
k = 1

card = transformers.pipeline('text-generation', model=card_path, device = card_device, num_return_sequences=k, do_sample=True, max_new_tokens = max_new_tokens)

def process_line(data, output_f):
    for line in tqdm(data):
        line = json.loads(line.strip())
        if task == 'entity_old':
            for ent in line['query_entity'].values():
                prompt = '{}, {}'.format(ent['entity'], ent['description'])
                knowl = card(prompt)
                knowl = [obj["generated_text"][len(prompt)+1:] for obj in knowl]
                line['generate'] = knowl
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
            continue
        elif task == 'entity':
            prompt = 'Knowledge:'
            for ent in line['query_entity'].values():
                prompt += ' {}, {}.'.format(ent['entity'], ent['description'])
            prompt += '\nQuestion: {}'.format(line['question'])
        elif task == 'pseduo':
            prompt = line['pseudo_doc']
        elif task == 'summary':
            prompt = line['summary']
        elif task == 'choice':
            choice_list = ['A', 'B', 'C', 'D']
            for choice in choice_list:
                choice_str = str(line[choice])
                if len(choice_str.split()) < 20:
                    continue
                else:
                    prompt = choice_str
                    knowl = card(prompt)
                    knowl = [obj["generated_text"][len(prompt)+1:] for obj in knowl]
                    line['generate'] = knowl
                    output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

            continue
        else:
            prompt = 'Question: {}'.format(line['question'])
        

        knowl = card(prompt)
        knowl = [obj["generated_text"][len(prompt):] for obj in knowl]
        line['generate'] = knowl
        output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
        # print(prompt)
        # print(knowl[0])
        # exit()

if __name__ == '__main__':
    import os
    from mmlu_categories import subcategories, categories

    # for split_num in range(4):
    dataset_path = 'datasets/PopQA'
    input_dir = 'turn1_question_kg'

    save_dir = os.path.join('knowledge_card_result', 'PopQA', task, "test", card_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    input_file_name = os.path.join(dataset_path, input_dir + '.json')
    input_file = open(input_file_name)

    output_file_name = os.path.join(save_dir, "{}_kc.json".format(input_dir))
    output_file = open(output_file_name, 'w')
    process_line(input_file, output_file)
    '''MMLU'''

    # dataset_path = 'datasets/MMLU/data'
    # input_dir = 'query_el_raw' 
    # MMLU_categories = ["STEM", "humanities", "social sciences", "other (business, health, misc.)"]  # "STEM", "humanities", "social sciences", "other (business, health, misc.)"
    # subjects = sorted([f.split("_dev.json")[0] for f in os.listdir(os.path.join(dataset_path, "dev")) if "_dev.json" in f])

    # # mkdir save dir
    # save_dir = os.path.join('knowledge_card_result', 'MMLU', task, "dev", card_name)
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # # read category
    # train_categories = []
    # for c in MMLU_categories:
    #     train_categories.extend(categories[c])

    # for sub in subjects:
    #     if len(set(subcategories[sub]) & set(train_categories)) == 0:
    #         continue
    #     print(sub)
    #     input_file_name = os.path.join(dataset_path, "dev", input_dir , sub + "_dev.json")
    #     input_file = open(input_file_name)

    #     output_file_name = os.path.join(save_dir, "{}_kc.json".format(sub))
    #     output_file = open(output_file_name, 'w')
    #     process_line(input_file, output_file)
    
    