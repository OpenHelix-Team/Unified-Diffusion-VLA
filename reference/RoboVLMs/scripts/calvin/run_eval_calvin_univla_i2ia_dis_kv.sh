#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,3
# Run
# export MESA_GL_VERSION_OVERRIDE=4.1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN          # 启用 NCCL 调试信息
export NCCL_P2P_DISABLE=1        # 禁用 P2P 直连通信
export NCCL_IB_DISABLE=1         # 关闭 InfiniBand
export NCCL_SHM_DISABLE=1        # 禁用共享内存
export NCCL_LAUNCH_MODE=PARALLEL # 避免 NCCL 死锁
unset DISPLAY
export PYOPENGL_PLATFORM=egl
# export EGL_DEVICE_ID=0
# emu_hub=$1
# emu_hub=/share/user/iperror/data/univla/UniVLA/UNIVLA_CALVIN_ABCD_VIDEO_BS192_8K
# emu_hub=/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_CALVIN_ABCD_VIDEO_BS128_12k_NI2NA/checkpoint-9000
# emu_hub=/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_CALVIN_ABCD_VIDEO_BS128_12k_NI2NA/checkpoint-24000
# emu_hub="/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_CALVIN_ABCD_VIDEO_BS128_12k_I2IA_0825/checkpoint-18000"
# emu_hub="/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_CALVIN_ABCD_VIDEO_BS128_12k_I2IA_0827/checkpoint-9000"
# emu_hub="/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_CALVIN_ABCD_VIDEO_BS64_24k_I2IA_mi_0904/checkpoint-21000"
emu_hub="/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_CALVIN_ABCD_VIDEO_BS64_24k_I2IA_mi_0905/checkpoint-24000"
# emu_hub="/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_CALVIN_ABCD_VIDEO_BS64_24k_I2IA_mi_0913/checkpoint-24000"#不预测开始符和结束符


GPUS_PER_NODE=4

torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE --master_port=6099 eval/calvin/evaluate_ddp-emu.py \
--emu_hub $emu_hub \
--dis_i2a_kv \
--steps 72 \
--CACHE_ROOT /data/user/wsong890/user68/project/UniVLA/logs/calvin_exp_main/univla_calvin_abcd_video_i2ia_dis_mnt64 \
> ./log/calvin/eval_calvin_univla_i2ia_24k_jacobi_dis_i2a_mi_0915_72steps_kv.log 2>&1 
# --debug \
