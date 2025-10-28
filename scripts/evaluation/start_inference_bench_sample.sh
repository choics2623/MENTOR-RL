export PYTHONPATH=your_MENTOR_path:$PYTHONPATH
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

CUDA_VISIBLE_DEVICES=0,1,2,3 python batch_inference_multiturn_bench.py \
    --dataset_name MENTOR-RL/math-bench \
    --model_path your_model_path \
    --hf_token "" \
    --tensor_parallel_size 4 \
    --max_turns 5 \
    --output_file ./math_bench_sample/sample.jsonl \
    --batch_size 10000 \
    --categories Math-Forge-Hard Omni-MATH-512 aime24 aime25 amc23 minervamath