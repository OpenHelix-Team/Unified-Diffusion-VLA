#!/bin/bash
module load cuda/12.1
# export CUDA_VISIBLE_DEVICES=1
# export ARNOLD_WORKER_GPU=1
# export ARNOLD_WORKER_NUM=1
# export ARNOLD_ID=0
# export RANK=0

export OMP_NUM_THREADS=16


ckpt_dir="/data/user/wsong890/user68/project/UniVLA/logs/libero/train/UNIVLA_LIBERO_VIDEO_BS64_32k_mi_w80_1004/checkpoint-64000"

steps=72
TIMESTAMP=$(date +%Y%m%d%H%M)
# GPUS_PER_NODE=$ARNOLD_WORKER_GPU
# export ROBOSUITE_LOG_FILE="./robosuite.log"
# export CUDA_VISIBLE_DEVICES=0,2
torchrun --nproc_per_node=4 --master_addr="localhost" --master_port=29688 eval/libero/evaluate_libero_emu_ddp.py \
--emu_hub $ckpt_dir \
--no_nccl \
--no_action_ensemble \
--task_suite_name libero_10 \
--dis_i2a \
--steps $steps \
--cache_root Unified-Diffusion-VLA/logs/libero_goal/eval/${TIMESTAMP} \
--action_tokenizer Unified-Diffusion-VLA/pretrain/fast_libero_all_t10_s50 \
--debug \
> ./log/libero/eval_libero_udvla_dis_64k_mi_1016_w80_libero10_${steps}steps.log 2>&1 
# 
