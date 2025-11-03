WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=8

DATAPATH='/share/project/yuqi.wang/datasets/processed_data/meta/simplerenv_bridge_trainval.pkl'
ACTION_TOKENIZER_PATH="/share/project/yuqi.wang/UniVLA/pretrain/fast_bridge_t5_s50"
EXP_NAME="UNIVLA_SIMPLERENV_BRIDGE_VIDEO_BS128_20k"

export PYTHONPATH=$(pwd)

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} \
    train/train_moe.py \
    --model_name_or_path /data/user/wsong890/user68/project/UniVLA/logs/ckpts/WORLD_MODEL_POSTTRAIN\
    --model_config_path /data/user/wsong890/user68/project/UniVLA/configs/moe_fast_video.json \
    --deepspeed scripts/sft/zero3_offload.json \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 8e-5 \
    --null_prompt_prob 0.15 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 2.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 20000 \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 500 \
    --per_device_train_batch_size 4 \
    --frames 2 \
    --action_frames 5 \
    --max_position_embeddings 2400 \
    --seed 42 \
    --logging_steps 20 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 4000 \
    --eval_strategy no \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_action True \
    --actions True \
    --actions_format "fast" \
    --use_gripper False \
    --video_format "interleave" \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
