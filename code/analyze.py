import json
from tqdm import tqdm

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

if __name__ == '__main__':
    compare_entity_cover('/data/xkliu/LLMs/DocFixQA/datasets/TemporalQA/question_entity.json')
