# Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process
[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.01718) [![Page](https://img.shields.io/badge/Project--Page-blue?style=for-the-badge&logo=homepage&logoColor=white)](https://irpn-eai.github.io/UD-VLA.github.io) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/chenpyyy/UD-VLA_CALVIN_ABCD_D) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)]([https://github.com/OpenHelix-Team/VLA-Adapter/issues/1](https://github.com/OpenHelix-Team/UD-VLA/issues/1))

<!-- <a href="https://arxiv.org/abs/2511.01718" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arxiv-red?logo=arxiv&label=UD-VLA" height="25" />
</a>
<a href="https://irpn-eai.github.io/UD-VLA.github.io/" target="_blank">
    <img alt="project" src="https://img.shields.io/badge/Project-blue?logo=probot&label=UD-VLA" height="25" />
</a>
<a href="https://huggingface.co/chenpyyy/UD-VLA_CALVIN_ABCD_D" target="_blank">
    <img alt="HF Model" src="https://img.shields.io/badge/Model-ffd400?logo=huggingface&label=UD-VLA" height="25" />
</a>
<a href="https://github.com/OpenHelix-Team/UD-VLA/issues/1" target="_blank">
    <img alt="Wechat" src="https://img.shields.io/badge/Wechat-green?logo=wechat&label=UD-VLA" height="25" />
</a> -->


<br>

 **Jiayi Chen¹\***,**Wenxuan Song¹\***, **Pengxiang Ding²˒³**, **Ziyang Zhou¹**, **Han Zhao²˒³**, **Feilong Tang⁴**,**Donglin Wang²**,  **Haoang Li¹†**

¹ IRPN Lab, HKUST(GZ) <br>
² MiLab, Westlake University  <br> 
³ Zhejiang University  <br> 
⁴ Monash University

\* Equal Contribution  † Project leader  ‡ Corresponding Author 


## List
 
[**Overview**](#overview) | 
[**TODO**](#TODO-List) | 
[**Installation**](#installation) | 
[**Download checkpoint**](#download-pretraning-checkpoint) | 
[**Train**](#model-training) | 
[**Evaluation**](#model-evaluation) | 
[**Results**](#experiment-result)


<hr style="border: 2px solid gray;"></hr>

## Overview

This repository is an improved fork of [UniVLA](https://github.com/baaivision/UniVLA). You can set up the environment by following UniVLA’s instructions. Below, we provide a step-by-step guide to using UD-VLA in the CALVIN setting.
![](overeview.png)

## TODO List<a name="todo"></a>

- [ ]  The code of real world implementaion.
- [ ]  Research of distillation of discrete diffusion VLA for more efficient Inference.


## Installation


### 1. Install Base environment.

```bash
# Create and activate conda environment
conda create -n udvla-calvin python=3.10 -y
conda activate udvla-calvin
# Clone and install the openvla repo
git clone https://github.com/OpenHelix-Team/UD-VLA.git
cd UD-VLA
pip install -r requirements.txt

```
 
### 2. Install Calvin environment

This setup is only for evaluation. The following steps are required to set up the environment:

```shell
# Install dependencies
cd reference/RoboVLMs

# This will install the required environment and download the calvin dataset.
bash scripts/setup_calvin.sh

# Only for rendering environment.
bash scripts/setup_calvin_vla.sh

# Check if the environment is set up correctly
python eval/calvin/env_test.py
```
## Download  pretraning checkpoint
> [Emu3-base](https://huggingface.co/BAAI/Emu3-Stage1)

Emu3-base for text tokenizer.
> [Emu3-vision](https://huggingface.co/BAAI/Emu3-VisionTokenizer)

Emu3-vision for visual tokenizer.

> [Autoregressive World model](https://huggingface.co/Yuqi1997/UniVLA/tree/main/WORLD_MODEL_POSTTRAIN)

This model is trained in an autoregressive manner with causal attention.

## Dataset Preparation


```shell
# 1. process the dataset
python tools/process/calvin_process.py

# 2. extract the vq tokens, need to change the dataset & output path
bash scripts/tokenizer/extract_vq_emu3.sh 

# 3. pickle generation for training
python tools/pickle_gen/pickle_generation_calvin.py
```

## Model Training

### FAST Tokenizer
You can fit the FAST tokenizer on the corresponding dataset. Also, you can adjust the scale in tokenizer for more fine-grained tokenization.For CALVIN ABCD→D, we set the action chunk size to 10.
```shell
python tools/action_tokenizer/fit_fast.py
```
### Discreate difussion training
```shell
bash scripts/simulator/calvin/train_calvin_abcd_video_i-ia_bid_mi.sh
```
> We recommend **at least 4×80-GB GPUs** (e.g., A100/H100 80GB). Each sample contains many **image tokens**, which results in long **sequence lengths** and increased memory usage.


We also release our checkpoint fituned on CALVIN ABCD→D at [UD-VLA_CALVIN](https://huggingface.co/chenpyyy/UD-VLA_CALVIN-ABCD)


```bash
# train_calvin_abcd_video_i-ia_bid_mi.sh
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=4
MODEL_PATH="logs/ckpts/WORLD_MODEL_POSTTRAIN"
DATAPATH='/share/user/iperror/data/udvla/processed_data/meta/calvin_abcd_norm.pkl'
ACTION_TOKENIZER_PATH="./pretrain/fast_calvin_abcd_a10_s50"

EXP_NAME="UNIVLA_CALVIN_ABCD_VIDEO_BS64_32k_I2IA_mi_0915"

export PYTHONPATH=$(pwd)

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} \
    train/train_moe.py \
    --model_name_or_path ${MODEL_PATH}\
    --model_config_path configs/moe_fast_video.json \
    --deepspeed scripts/sft/zero3.json \
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
    --max_steps 24000 \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 50 \
    --per_device_train_batch_size 4 \
    --frames 2 \
    --action_frames 10 \
    --max_position_embeddings 1650 \
    --seed 42 \
    --logging_steps 20 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 3000 \
    --eval_strategy no \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_action False \
    --actions True \
    --actions_format "fast" \
    --use_gripper True \
    --video_format "interleave" \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
    --with_i_ia True \
    --mask_image True \
    --use_blockwise_attn_mask True \
    --attn_type "None" \

```
Key training flags

- `--with_i_ia True` Enable **joint generation** of *future image tokens * and *action tokens * in the same denoising step (JD3P).  

- `--mask_image True` Apply `<MASK>` to **future-image positions** during training and compute cross-entropy **only on masked positions** (single-step mask-predict objective).

- `--use_blockwise_attn_mask True` Use **blockwise hybrid attention**:
    - **Bidirectional** within the future-image block and within the action block.
    - **Causal** across blocks

- `--attn_type "None"`  Not use `"flash"` (Flash-Attention).

## Model Evaluation
You also can dowload our checkpoint fituned on CALVIN ABCD→D at [UD-VLA_CALVIN](https://huggingface.co/chenpyyy/UD-VLA_CALVIN-ABCD) 
```shell
cd reference/RoboVLMs

# 4 GPUs inference,we set difussion step to 72.
bash scripts/run_eval_calvin_univla_i2ia_dis.sh 

# above command will generate the 4 results in the `results` folder, calculate the final average score
python tools/evaluation/calvin_score.py
```


## Experiment Result
### Performance on CALVIN ABCD→D Benchmark.
<em>UniVLA*</em> denotes the variant without historical frames for fair comparison. We evaluate 500 rollouts for our model, where each rollout involves a sequence of 5 consecutive sub-tasks.
<div align="center">

<table>
  <thead>
    <tr>
      <th>Method</th><th>Task</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>Avg. Len ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>MCIL</td><td>ABCD→D</td><td>0.373</td><td>0.027</td><td>0.002</td><td>0.000</td><td>0.000</td><td>0.40</td></tr>
    <tr><td>RT-1</td><td>ABCD→D</td><td>0.844</td><td>0.617</td><td>0.438</td><td>0.323</td><td>0.227</td><td>2.45</td></tr>
    <tr><td>Robo-Flamingo</td><td>ABCD→D</td><td>0.964</td><td>0.896</td><td>0.824</td><td>0.740</td><td>0.660</td><td>4.09</td></tr>
    <tr><td>GR-1</td><td>ABCD→D</td><td>0.949</td><td>0.896</td><td>0.844</td><td>0.789</td><td>0.731</td><td>4.21</td></tr>
    <tr><td>ReconVLA</td><td>ABCD→D</td><td>0.980</td><td>0.900</td><td>0.845</td><td>0.785</td><td>0.705</td><td>4.23</td></tr>
    <tr><td>UniVLA*</td><td>ABCD→D</td><td>0.958</td><td>0.918</td><td>0.874</td><td>0.846</td><td>0.702</td><td>4.24</td></tr>
    <tr><td>UP-VLA</td><td>ABCD→D</td><td>0.962</td><td>0.921</td><td>0.879</td><td>0.842</td><td>0.812</td><td>4.42</td></tr>
    <tr><td><b>UD-VLA (ours)</b></td><td>ABCD→D</td><td><b>0.992</b></td><td><b>0.968</b></td><td><b>0.936</b></td><td><b>0.904</b></td><td><b>0.840</b></td><td><b>4.64</b></td></tr>
  </tbody>
</table>
</div>

### Performance on  Real-world.
Our real-world setup consists of a 6-DoF UR5e robotic arm equipped with a 6-DoF Inspire RH56E2 robotic hand for dexterous manipulation. 
![](real-world.png)

## Other Simulation Benchmark Setup
- [LIBERO](docs/libero.md)
- [SimplerEnv](docs/simpler.md)
<!-- ### Performance on Libero Benchmark.
### Performance on Simplerenv Benchmark. -->
## ❤️ Acknowledgment

We thank [Univla](https://github.com/baaivision/UniVLA), [Emu3](https://github.com/baaivision/Emu3), [RobotVLM](https://github.com/Robot-VLAs/RoboVLMs) and [Show-o](https://github.com/showlab/Show-o) for their open-sourced work!

We thank [Yuqi Wang](https://github.com/Robertwyq) and [Zhide zhong](https://scholar.google.com/citations?user=msy4tL4AAAAJ&hl=zh-CN) for their guidance about experiment!
## 📖Citation
If you find UD-VLA useful, please consider citing our work🤗:
```bibtex
@article{udvla2025,
title={Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process},
author={Jiayi Chen, Wenxuan Song, Pengxiang Ding, Ziyang Zhou, Han Zhao, Feilong Tang, Donglin Wang, Haoang Li},
year={2025},
journal={arXiv preprint arXiv:2511.01718}
}
```
