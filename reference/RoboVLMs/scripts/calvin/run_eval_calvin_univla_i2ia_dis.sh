#!/bin/bash

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

emu_hub="chenpyyy/UD-VLA_CALVIN-ABCD" 

# export CUDA_VISIBLE_DEVICES=1
GPUS_PER_NODE=4

torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE --master_port=6099 eval/calvin/evaluate_ddp-emu.py \
--emu_hub $emu_hub \
--use_jacobi_generate \
--dis_i2a \
--steps 72 \
--CACHE_ROOT /data/user/wsong890/user68/project/UniVLA/logs/calvin_exp_main/univla_calvin_abcd_video_i2ia_dis \
# --debug \
