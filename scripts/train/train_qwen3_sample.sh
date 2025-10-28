export MKL_THREADING_LAYER=GNU
export RAY_BACKEND_LOG_LEVEL=WARNING
# export PYTHONPATH=/path/to/mentor/src:$PYTHONPATH

PROMPT_KEY=question
TRAIN_BATCH_SIZE=8  # e.g., 32, 64, 128
PPO_MINI_BATCH_SIZE=4  # e.g., 4, 8, 32
LR=1e-6
MAX_PROMPT_LENGTH=1536
MAX_RESPONSE_LENGTH=12000  # e.g., 12000
MAX_NUM_BATCHED_TOKENS=128000  # e.g., 64000, 128000
USE_MENTOR=True
PROMPT_TEMPLATE_NAME=qwen3_native
MODEL_NAME=qwen3
ACTOR_MODEL_PATH=Qwen/Qwen3-1.7B # Qwen/Qwen2.5-1.5B-Instruct
ROLLOUT_NAME=vllm_with_tool
REWARD_MANAGER=mentor
ABLATION_EXP_ID=ABLATION4  # Options: SPARSE, MENTOR, ABLATION2, ABLATION3, ABLATION4
ROLLOUT_N=5  # e.g., 5, 10
ROLLOUT_TP=2  # e.g., 2
ROLLOUT_GPU_UTIL=0.8  # e.g., 0.8
MAX_TURNS=5  # e.g., 5
SEARCH_URL=http://0.0.0.0:7777  # e.g., http://0.0.0.0:7777
SANDBOX_URL=http://0.0.0.0:2623  # e.g., http://0.0.0.0:2623
PROJECT_NAME=MENTOR-qwen3


NNODES=1  # e.g., 1
N_GPUS_PER_NODE=4  # e.g., 8
SAVE_FREQ=15  # e.g., 10, 15, 80
TEST_FREQ=4  # e.g., 4, 10, 15
TOTAL_EPOCHS=2  # e.g., 2, 5
WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
TRAIN_FILES="['../../data/train_dataset/train.parquet']"  # Update with your training data path
TEST_FILES="['../../data/train_dataset/test.parquet']"  # Update with your test data path

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train_files) TRAIN_FILES="$2"; shift 2;;
        --test_files) TEST_FILES="$2"; shift 2;;
        --prompt_key) PROMPT_KEY="$2"; shift 2;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2;;
        --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2;;
        --use_mentor) USE_MENTOR="$2"; shift 2;;
        --prompt_template_name) PROMPT_TEMPLATE_NAME="$2"; shift 2;;
        --model_name) MODEL_NAME="$2"; shift 2;;
        --actor_model_path) ACTOR_MODEL_PATH="$2"; shift 2;;
        --rollout_name) ROLLOUT_NAME="$2"; shift 2;;
        --max_turns) MAX_TURNS="$2"; shift 2;;
        --reward_manager) REWARD_MANAGER="$2"; shift 2;;
        --ablation_exp_id) ABLATION_EXP_ID="$2"; shift 2;;
        --rollout_n) ROLLOUT_N="$2"; shift 2;;
        --rollout_tp) ROLLOUT_TP="$2"; shift 2;;
        --rollout_gpu_util) ROLLOUT_GPU_UTIL="$2"; shift 2;;
        --search_url) SEARCH_URL="$2"; shift 2;;
        --sandbox_url) SANDBOX_URL="$2"; shift 2;;
        --project_name) PROJECT_NAME="$2"; shift 2;;
        --experiment_name) EXPERIMENT_NAME="$2"; shift 2;;
        --nnodes) NNODES="$2"; shift 2;;
        --n_gpus_per_node) N_GPUS_PER_NODE="$2"; shift 2;;
        --save_freq) SAVE_FREQ="$2"; shift 2;;
        --test_freq) TEST_FREQ="$2"; shift 2;;
        --total_epochs) TOTAL_EPOCHS="$2"; shift 2;;
        --wandb_api_key) WANDB_API_KEY="$2"; shift 2;;
        --save_path) SAVE_PATH="$2"; shift 2;;
        *)
            echo "unknown argument '$1'" >&2
            exit 1;;
    esac
done

# Base directory for saving models and checkpoints
SAVE_BASE_DIR=YOUR_SAVE_BASE_DIR  # e.g., ../../saved_models or /path/to/your/models

# Automatically generated experiment name from hyperparameters
EXPERIMENT_NAME=ABLATION-FINAL-8B-$ABLATION_EXP_ID-${NNODES}Node-${N_GPUS_PER_NODE}GPUs-$MODEL_NAME-bs-$TRAIN_BATCH_SIZE-mini-$PPO_MINI_BATCH_SIZE-rollout-$ROLLOUT_N
SAVE_PATH=$SAVE_BASE_DIR/qwen3-8b/$EXPERIMENT_NAME
CHECKPOINT_SAVE=$SAVE_PATH

if [ "$WANDB_API_KEY" != "None" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

ROLLOUT_SAVE_PATH=${SAVE_PATH}/rollout
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.use_mentor=${USE_MENTOR} \
    data.prompt_template_name=${PROMPT_TEMPLATE_NAME} \
    data.search_url=${SEARCH_URL} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.sandbox_url=${SANDBOX_URL} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.model_name=${MODEL_NAME} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    reward_model.ablation_exp_id=${ABLATION_EXP_ID} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.rollout_save_path=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=$CHECKPOINT_SAVE/outputs | tee $CHECKPOINT_SAVE/run.log