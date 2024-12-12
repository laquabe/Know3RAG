export CUDA_VISIBLE_DEVICES=3
python code/eval.py \
    --dataset_name MMLU \
    --model_name Llama \
    --file_path /data/xkliu/LLMs/DocFixQA/result/MMLU/query_doc_rag_new/ \
    --answer_key llm_response