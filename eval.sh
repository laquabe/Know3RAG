export CUDA_VISIBLE_DEVICES=6
python code/eval.py \
    --dataset_name MMLU \
    --model_name Llama \
    --file_path /data/xkliu/LLMs/DocFixQA/result/MMLU/raw/ \
    --answer_key llm_response