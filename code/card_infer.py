import transformers
import json
from tqdm import tqdm
from utils import read_data
import argparse
import os

def process_line(card, task, data, output_f):
    for line in tqdm(data):
        line = json.loads(line.strip())
        if task == 'entity':
            prompt = 'Knowledge:'
            for ent in line['query_entity'].values():
                prompt += ' {}, {}.'.format(ent['entity'], ent['description'])
            prompt += '\nQuestion: {}'.format(line['question'])
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
    parser = argparse.ArgumentParser(description='Knowledge card generation')

    # Define arguments
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='Maximum number of new tokens to generate per prompt.')
    parser.add_argument('--task', type=str, default='question',
                        choices=['question', 'entity_old', 'entity', 'pseduo', 'summary', 'choice'],
                        help='Task type controlling the prompt format. '
                             'Choices: question, entity_old, entity, pseduo, summary, choice.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model.')
    parser.add_argument('--device', type=int, default=7,
                        help='CUDA device ID for model placement (-1 for CPU).')
    parser.add_argument('--k', type=int, default=1,
                        help='Number of sequences to return (num_return_sequences).')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input JSONL file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path for the output JSONL file ')

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir: # Check if output_file contains a directory path
        os.makedirs(output_dir, exist_ok=True)

    # Load the knowledge card model pipeline
    try:
        print(f"Loading model from: {args.model_path} on device: {args.device}")
        card_pipeline = transformers.pipeline(
            'text-generation',
            model=args.model_path,
            device=args.device,
            num_return_sequences=args.k,
            do_sample=True, # Using do_sample=True as in original code
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=True # Add if loading custom models
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1) # Exit if model fails to load

    # Open input and output files
    try:
        with open(args.input_file, 'r', encoding='utf-8') as input_file, \
             open(args.output_file, 'w', encoding='utf-8') as output_file:

            print(f"Processing data from '{args.input_file}' and writing to '{args.output_file}'...")
            process_line(card_pipeline, args.task, input_file, output_file)
            print("Processing complete.")

    except FileNotFoundError as e:
        print(f"Error: Input or output file not found: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        exit(1)


    
    