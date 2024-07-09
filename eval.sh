export CUDA_VISIBLE_DEVICES=4
python code/eval.py \
    --dataset_name TemporalQA \
    --model_name Llama \
    --file_path /data/xkliu/LLMs/DocFixQA/result/TemporalQA/Llama/knowledge_card_filter_rag.json \
    --answer_key llm_response