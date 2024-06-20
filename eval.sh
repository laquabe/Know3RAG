export CUDA_VISIBLE_DEVICES=4
python code/eval.py \
    --dataset_name TemporalQA \
    --model_name Mistarl \
    --file_path /data/xkliu/LLMs/DocFixQA/result/TemporalQA/Mistral/raw.json \
    --answer_key llm_response