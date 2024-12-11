export CUDA_VISIBLE_DEVICES=4
python code/eval.py \
    --dataset_name MMLU \
    --model_name Llama \
    --file_path /data/xkliu/LLMs/DocFixQA/result/MMLU/dev_gpt4o_mini_reason/ \
    --answer_key llm_response