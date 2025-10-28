export PYTHONPATH=your_MENTOR_path:$PYTHONPATH
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

CUDA_VISIBLE_DEVICES=0,1,2,3 python batch_inference_multiturn_bench.py \
    --dataset_name YOUR_DATASET_NAME \
    --model_path YOUR_MODEL_PATH \
    --tensor_parallel_size 4 \
    --max_turns 5 \
    --output_file ./results/output.jsonl \
    --batch_size 10000 \
    --categories YOUR_CATEGORIES
