#!/bin/bash
module load cuda/12.1
trap "echo 'Ctrl+C detected, killing all subprocesses...'; kill 0" SIGINT
ckpt_dir="/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_LIBERO_VIDEO_BS48_24k_mi_0912/checkpoint-48000"
steps=96
# ckpt_dir="/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_LIBERO_VIDEO_BS48_24k_mi_0915/checkpoint-24000"
TIMESTAMP=$(date +%Y%m%d%H%M)

CUDA_VISIBLE_DEVICES=0 python eval/libero/evaluate_libero_emu.py \
--emu_hub $ckpt_dir \
--no_nccl \
--no_action_ensemble \
--task_suite_name libero_goal \
--dis_i2a \
--steps $steps \
--cache_root /data/user/wsong890/user68/project/UniVLA/logs/libero_goal/${TIMESTAMP} \
> ./log/libero/eval_libero_univla_dis_24k_mi_0918_goal_${steps}steps.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python eval/libero/evaluate_libero_emu.py \
--emu_hub $ckpt_dir \
--no_nccl \
--no_action_ensemble \
--task_suite_name libero_10 \
--dis_i2a \
--steps $steps \
--cache_root /data/user/wsong890/user68/project/UniVLA/logs/libero10/${TIMESTAMP} \
> ./log/libero/eval_libero_univla_dis_24k_mi_0918_libero10_${steps}steps.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python eval/libero/evaluate_libero_emu.py \
--emu_hub $ckpt_dir \
--no_nccl \
--no_action_ensemble \
--task_suite_name libero_object \
--dis_i2a \
--steps $steps \
--cache_root /data/user/wsong890/user68/project/UniVLA/logs/libero_object/${TIMESTAMP} \
> ./log/libero/eval_libero_univla_dis_24k_mi_0918_object_${steps}steps.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python eval/libero/evaluate_libero_emu.py \
--emu_hub $ckpt_dir \
--no_nccl \
--no_action_ensemble \
--task_suite_name libero_spatial \
--dis_i2a \
--steps $steps \
--cache_root /data/user/wsong890/user68/project/UniVLA/logs/libero_spatial/${TIMESTAMP} \
> ./log/libero/eval_libero_univla_dis_24k_mi_0918_spatial_${steps}steps.log 2>&1 &

# --debug

wait