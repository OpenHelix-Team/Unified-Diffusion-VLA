import argparse
import os
import torch
import pickle
import numpy as np
from time import time
from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import sys
from PIL import Image, ImageDraw, ImageFont
from torch.nn.functional import cross_entropy
from random import shuffle
import random

sys.path.append("/share/project/yuqi.wang/UniVLA/reference/Emu3")
from emu3.mllm.processing_emu3 import Emu3Processor
from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM
# action related
from emu3.mllm import Emu3MoE
from transformers import LogitsProcessor

class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        """
        :param allowed_token_ids: 允许的token ID列表
        """
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        # 创建掩码：允许的token位置为True，其他为False
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            mask[self.allowed_token_ids] = True
        else:
            mask[:, self.allowed_token_ids] = True
        
        # 将不允许的token概率设为负无穷
        scores[~mask] = -float("inf")
        return scores

train_pickle = '/share/project/yuqi.wang/datasets/processed_data/meta/libero_all_norm.pkl'
with open(train_pickle, 'rb') as f:
    train_meta = pickle.load(f)

EMU_HUB = "/share/project/yuqi.wang/UniVLA/logs/huggingface_univla_ckpt/UniVLA/UNIVLA_LIBERO_IMG_BS192_8K"
VQ_HUB = EMU_HUB
VISION_HUB = "/share/project/yuqi.wang/UniVLA/pretrain/Emu3-VisionVQ"
fast_path = "/share/project/yuqi.wang/UniVLA/pretrain/fast"


# Base config
#####################
action_predict_frame = 10
gripper = True
video_average_nums = 512
debug = False
####################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Emu3MoE.from_pretrained(
    EMU_HUB,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
model.to(device).eval()
kwargs = dict(
    mode='VLA',
    padding="longest",
)
eoa_token_id=151845
GENERATION_CONFIG = GenerationConfig(
    pad_token_id=model.config.pad_token_id,
    bos_token_id=model.config.bos_token_id,
    eos_token_id=eoa_token_id,
    do_sample=False,
)

tokenizer = Emu3Tokenizer.from_pretrained(
        VQ_HUB,
        model_max_length=model.config.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )
image_processor = AutoImageProcessor.from_pretrained(VISION_HUB, trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(VISION_HUB, trust_remote_code=True).to(device).eval()
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
action_tokenizer = AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)
image_processor.min_pixels = 80 * 80

last_token_id = tokenizer.pad_token_id - 1
allowed_token_ids = list(range(last_token_id - action_tokenizer.vocab_size, last_token_id + 1)) + [eoa_token_id]
action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
action_errors_per_dimension = [[] for _ in range(7)]  

for i in range(video_average_nums):
    task_idx = random.randint(0, len(train_meta) - 1)
    task_data = train_meta[task_idx]
    text = task_data['text']
    image_list = task_data['image']
    action_list = task_data['action']
    rand_idx = random.randint(0, len(image_list) - action_predict_frame - 1)
    image = image_list[rand_idx]
    if gripper:
        gripper_list = task_data['gripper_image']
        gripper = gripper_list[rand_idx]
        gripper_code = np.load(gripper)
        gripper_code = torch.tensor(gripper_code).unsqueeze(0)
    else:
        gripper_code = None
    action_gt = action_list[rand_idx:rand_idx + action_predict_frame]
    video_code = np.load(image)
    video_code = torch.tensor(video_code).unsqueeze(0)

    pos_inputs = processor.video_process(text=text, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=1, frames = 1, return_tensors="pt", **kwargs)

    outputs = model.generate(
        pos_inputs.input_ids.to(device),
        GENERATION_CONFIG,
        max_new_tokens=50,
        logits_processor=[action_id_processor],
        attention_mask=pos_inputs.attention_mask.to(device),
    )
    outputs = outputs[:, pos_inputs.input_ids.shape[-1]:-1]
    last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
    processed_outputs = last_token_id_tensor - outputs
    action_outputs = action_tokenizer.decode(processed_outputs, time_horizon=action_predict_frame, action_dim=7)
    action = torch.from_numpy(action_outputs[0])

    action_gt_id = action_tokenizer(action_gt)
    action_gt_decode = action_tokenizer.decode(action_gt_id, time_horizon=action_predict_frame, action_dim=7)

    if debug:
        print(f"Ground Truth: {action_gt}")
        print(f"Ground Truth: {action_gt_id}")
        print(f"Action Prediction: {processed_outputs}")
        print(f"Action Prediction: {action}")
        print(f"Ground Truth: {action_gt_decode}")
        import ipdb
        ipdb.set_trace()

    # Calculate the error for each dimension over time
    for t in range(min(action.shape[0], action_gt.shape[0])):  # Iterate over the time horizon
        for dim in range(action.shape[1]):  # Iterate over the dimensions
            error = (action[t, dim] - action_gt[t, dim]).abs()
            action_errors_per_dimension[dim].append(error.item())

# Calculate the average error for each dimension
average_errors_per_dimension = [np.mean(errors) if errors else 0 for errors in action_errors_per_dimension]
print(f"Average Action Error per Dimension: {average_errors_per_dimension}")