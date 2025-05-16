import requests
import json
from tqdm import tqdm
import os
import multiprocessing
from multiprocessing import Manager, Process, Pool

def query_entity(entity_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

    response = requests.get(url)
    if response.status_code == 404:
        info_dict = {
            'labels': '',
            'descriptions': '',
            'aliases': [],
            'claims': [],
        }
        return info_dict
    
    data = response.json()

    try:
        entity_data = data['entities'][entity_id]
    except:
        info_dict = {
            'labels': '',
            'descriptions': '',
            'aliases': [],
            'claims': {},
        }
        return info_dict

    try:
        label_data = entity_data['labels']['en']['value']
    except:
        label_data = ''
    
    try:
        des_data = entity_data['descriptions']['en']['value']
    except:
        des_data = ''

    try:
        aliases_data = [i['value'] for i in entity_data['aliases']['en']]
    except:
        aliases_data = []

    claim_data = {}
    if 'claims' in entity_data:
        for r, e_list in entity_data['claims'].items():
            r_list = []
            for e in e_list:
                try:
                    r_list.append(e['mainsnak']['datavalue']['value']['id'])
                except:
                    continue
            if len(r_list) > 0:
                claim_data[r] = r_list

    info_dict = {
        'labels': label_data,
        'descriptions': des_data,
        'aliases': aliases_data,
        'claims': claim_data
    }

    return info_dict

def process_line(line, processed_ids, queue):
    if line['wiki_id'] in processed_ids:
        return

    try:
        ent_info = query_entity(line['wiki_id'])
        line.update(ent_info)

        queue.put(line)
    except Exception as e:
        print(f"Error processing line with id {line['wiki_id']}: {e}")

def write_output(queue, output_file_name):
    with open(output_file_name, 'a') as output_f:
        while True:
            line = queue.get()
            if line is None:
                break
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

def main(input_file_name, output_file_name, process_num=5):
    processed_ids = set()
    if os.path.exists(output_file_name):
        with open(output_file_name, 'r') as output_f:
            for line in output_f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data['wiki_id'])
                except json.JSONDecodeError:
                    continue

    manager = Manager()
    queue = manager.Queue()

    write_process = Process(target=write_output, args=(queue, output_file_name))
    write_process.start()

    lines = []
    with open(input_file_name, 'r') as input_f:
        for line in tqdm(input_f):
            try:
                lines.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    with Pool(processes=process_num) as pool:
        for line in lines:
            pool.apply_async(process_line, args=(line, processed_ids, queue))

        pool.close()
        pool.join()

    queue.put(None)  
    write_process.join()

if __name__ == "__main__":
    input_file_name = 'PopQA/turn1_tail_map.json'
    output_file_name = 'PopQA/turn1_tail_map_full.json'
    main(input_file_name, output_file_name, process_num=3)
