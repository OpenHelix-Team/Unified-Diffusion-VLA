#!/bin/bash
module load cuda/12.1
# export CUDA_VISIBLE_DEVICES=1
# export ARNOLD_WORKER_GPU=1
# export ARNOLD_WORKER_NUM=1
# export ARNOLD_ID=0
# export RANK=0

# export OMP_NUM_THREADS=16

# echo "total workers: ${ARNOLD_WORKER_NUM}"
# echo "cur worker id: ${ARNOLD_ID}"
# echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
# ckpt_dir="/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_LIBERO_VIDEO_BS48_24k_mi_0912/checkpoint-24000"
# ckpt_dir="/data/user/wsong890/user68/project/UniVLA/logs/libero10/train/UNIVLA_LIBERO_10_VIDEO_BS48_24k_mi_0926/checkpoint-24000"
ckpt_dir="/data/user/wsong890/user68/project/UniVLA/logs/libero/train/UNIVLA_LIBERO_VIDEO_BS64_32k_mi_w80_1004/checkpoint-32000"
steps=72
TIMESTAMP=$(date +%Y%m%d%H%M)
# GPUS_PER_NODE=$ARNOLD_WORKER_GPU

torchrun --nproc_per_node=4 --master_addr="localhost" --master_port=29881 eval/libero/evaluate_libero_emu_ddp.py \
--emu_hub $ckpt_dir \
--no_nccl \
--no_action_ensemble \
--task_suite_name libero_spatial \
--dis_i2a \
--steps $steps \
--cache_root /data/user/wsong890/user68/project/UniVLA/logs/libero_spatial/eval/${TIMESTAMP} \
--action_tokenizer /data/user/wsong890/user68/project/UniVLA/pretrain/fast_libero_all_t10_s50 \
> ./log/libero/eval_libero_udvla_dis_32k_mi_1012_libero_spatial_${steps}steps.log 2>&1 
# --debug
