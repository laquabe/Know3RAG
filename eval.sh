export CUDA_VISIBLE_DEVICES=2
python code/eval.py \
    --dataset_name TemporalQA \
    --model_name Llama \
    --file_path /data/xkliu/LLMs/DocFixQA/result/TemporalQA/Llama/rag_filter.json \
    --answer_key llm_response