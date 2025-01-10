export CUDA_VISIBLE_DEVICES=2
python code/eval.py \
    --dataset_name MMLU \
    --model_name Llama \
    --file_path /data/xkliu/LLMs/DocFixQA/result/MMLU/turn1_rag/ \
    --answer_key llm_response