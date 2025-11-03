WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=8

DATAPATH='/share/project/yuqi.wang/datasets/post_train_data/meta/world_model_post_train_v3.pkl'
ACTION_TOKENIZER_PATH="/share/project/yuqi.wang/UniVLA/pretrain/fast"
EXP_NAME="WORLD_MODEL_PRETRAIN_DEBUG"

export PYTHONPATH=$(pwd)

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    train/train_moe.py \
    --model_name_or_path /share/project/yuqi.wang/UniVLA/pretrain/Emu3-Base \
    --model_config_path /share/project/yuqi.wang/UniVLA/configs/moe_fast_video_pretrain.json \
    --deepspeed scripts/sft/zero3_offload.json \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 8e-5 \
    --null_prompt_prob 0.15 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 30000 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 50 \
    --per_device_train_batch_size 1 \
    --frames 6 \
    --action_frames 5 \
    --max_position_embeddings 6400 \
    --seed 42 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 2 \
    --save_strategy steps \
    --save_steps 5000 \
    --eval_strategy no \
    --apply_loss_on_only_vision True \
    --apply_loss_on_only_action False \
    --actions False \
    --use_gripper False \
    --video_format "interleave" \
    --post_training True \
    # --report_to wandb tensorboard \