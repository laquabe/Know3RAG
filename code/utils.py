import json

def read_data(dataset_name, file_path):
    with open(file_path) as f:
        if dataset_name == 'Truthful_QA':
            data = f.read()
            data = json.loads(data)
            return data

if __name__ == "__main__":
    read_data('Truthful_QA', '/data/xkliu/LLMs/DocFixQA/datasets/truthfulqa_mc_task.json')