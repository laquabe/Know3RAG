export CUDA_VISIBLE_DEVICES=0
python code/eval.py \
    --dataset_name TemporalQA \
    --model_name Llama \
    --file_path /data/xkliu/LLMs/DocFixQA/result/TemporalQA/Llama/SelfQuery_AddRaw_rag.json \
    --answer_key llm_response