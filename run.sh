export CUDA_VISIBLE_DEVICES=2,3
python code/main.py \
    --dataset_name Temporal \
    --model_name Mistral \
    --exp_name raw \
    --model_path /data/xkliu/LLMs/models/mistral-7B-v0.2-instruct

# Llama
# /data/xkliu/LLMs/models/Meta-Llama-3-8B-Instruct
# Mistral
# /data/xkliu/LLMs/models/mistral-7B-v0.2-instruct