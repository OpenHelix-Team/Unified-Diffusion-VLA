from typing import Optional, Sequence, Callable
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2 as cv
import json
from transforms3d.euler import euler2axangle
from pathlib import Path
import time
import sys
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import sys
sys.path.append("/data/user/wsong890/user68/project/UniVLA/reference")
# from robovlms.train.base_trainer import BaseTrainer
from RoboVLMs.eval.calvin.model_wrapper import CustomModel
from queue import Queue
from PIL import Image


from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import sys
sys.path.append("/data/user/wsong890/user68/project/UniVLA/reference/Emu3")
from emu3.mllm import Emu3Tokenizer, Emu3ForCausalLM, Emu3Processor
from emu3.mllm import Emu3MoE
from transformers import LogitsProcessor
from emu3.sampling import get_mask_chedule
import time

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

class EmuVLAInference(CustomModel):
    def __init__(
        self,
        emu_hub,
        vq_hub,
        vision_hub,
        device,
        policy_setup="widowx_bridge",
        fast_path="/data/user/wsong890/user68/project/UniVLA/pretrain/fast_bridge_t5_s50",
    ):

        self.emu_hub = emu_hub
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.device = device
        self.policy_setup = policy_setup

        if self.policy_setup == "google_robot":
            self.close_gripper_act = -1
            self.image_size = (160, 128)
        elif self.policy_setup == "widowx_bridge":
            self.close_gripper_act = 1
            self.image_size = (256,256)
        
        self.sticky_gripper_num_repeat = 2
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.close_gripper_num = 0
        self.late_close_gripper = 1 # 2

        ## hard code here
        self.window_size = 2
        self.predict_action_frames = 5
        self.context_frames = 1
        self.predict_frames = 1
        self.action_dim = 7
        self.use_gripper = False
        self.use_fast = True
        self.use_one_step = False
        self.eoa_token_id = 151845
        self.use_cot = False

        self.boa_token_id = 151844
        self.eoa_token_id = 151845
        self.boi_token_id = 151852
        self.bos_token_id = 151849
        self.bov_token_id = 151854
        self.eof_token_id = 151847
        self.eoi_token_id = 151853
        self.eol_token_id = 151846
        self.eos_token_id = 151850
        self.eov_token_id = 184621
        self.img_token_id = 151851
        self.inference_time=[]
        self.video_mode = True
        self.fast_path = fast_path
    
        # load model and tokenizer
        self.init_config(device=device)
        self.image_processor.min_pixels = 80 * 80

        if self.use_cot:
            self.kwargs = dict(
                mode='VLA_COT',
                padding="longest",
            )
        else:
            self.kwargs = dict(
                mode='VLA',
                padding="longest",
            )
        if self.use_fast:
            self.GENERATION_CONFIG = GenerationConfig(
                    pad_token_id=self.model.config.pad_token_id,
                    bos_token_id=self.model.config.bos_token_id,
                    eos_token_id=self.eoa_token_id,
                    do_sample=False,
                )
        
        else:
            self.GENERATION_CONFIG = GenerationConfig(
                use_cache=True,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                max_new_tokens=800, # hard code here
                do_sample=True,
                top_k=2048,
                temperature=0.8,
            )


    def init_config(self, device):
        
        self.model = Emu3MoE.from_pretrained(
            self.emu_hub,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            attn_implementation=None,
            trust_remote_code=True,
        )
        self.model.to(device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.vq_hub,
            model_max_length=self.model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True).to(device).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)

        # # fast tokenization
        # if self.policy_setup == "widowx_bridge":
            
        #     fast_path = "/data/user/wsong890/user68/project/UniVLA/pretrain/fast_bridge_t5_s50"
        #     # fast_path = "/data/user/wsong890/user68/project/UniVLA/pretrain/fast_bridge_t5_s50_old"
        #     print("fast_path:",fast_path)
        # elif self.policy_setup == "google_robot":
        #     fast_path = "/share/project/yuqi.wang/UniVLA/pretrain/fast_google_a5_s50"
        # else:
        #     fast_path = "/data/user/wsong890/user68/project/UniVLA/pretrain/fast"
        print("self.fast_path:",self.fast_path)
        self.action_tokenizer = AutoProcessor.from_pretrained(self.fast_path, trust_remote_code=True)

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0

        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

    
    def add_image(self, image):
        if self.vision_queue.full():
            self.vision_queue.get()
        self.vision_queue.put(image)
    
    def get_history(self):
        return list(self.vision_queue.queue) 

    def add_action(self, action):
        if self.action_queue.full():
            self.action_queue.get()
        self.action_queue.put(action)
    
    def get_action_history(self):
        return list(self.action_queue.queue)


    def reset(self):

        self.rgb_list = []
        self.hand_rgb_list = []
        self.rollout_step_counter = 0
        self.action_hist_list = []

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.close_gripper_num = 0
        self.previous_gripper_action = None

        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()

    def preprocess(self, image):
        # preprocess image
        agent_view = image
        agent_view = Image.fromarray(agent_view)
        agent_view = agent_view.resize(self.image_size)
        image_x = self.image_processor(agent_view, return_tensors="pt")["pixel_values"].cuda()
        image_code = self.image_tokenizer.encode(image_x)

        gripper_code = None

        return (
            image_code,
            gripper_code,
        )

    def step(self, image, goal):
        input_dict = dict()
      
        image_code, gripper_code = self.preprocess(image)

        prompt,neg_prompt = goal,""

        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + prompt]
        text_tokens = self.processor.tokenizer(text_prompt)
        
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type='pt')

        if self.video_mode:
            kwargs = dict(
                    mode='VLA_Video',
                    padding="longest",
                )
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **kwargs)
        else:
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **self.kwargs)
        
        if self.video_mode:
            self.add_image(pos_inputs)
            
            # 获取历史图像和动作
            history = self.get_history()
            action_history = self.get_action_history()

            # 初始化输入ID、token类型ID和attention mask
            all_input_ids = []
            all_token_type_ids = []
            all_attention_mask = []

            # Add text
            all_input_ids.append(text_tokens['input_ids'])
            all_token_type_ids.append(text_tokens['token_type_ids'])
            all_attention_mask.append(text_tokens['attention_mask'])

            # 遍历历史图像
            for i in range(len(history)):
                img_input_ids = history[i]['input_ids']
                img_token_type_ids = history[i]['token_type_ids']
                img_attention_mask = history[i]['attention_mask']
                
                # 对应的动作
                if i < len(action_history):
                    act_input_ids = action_history[i]
                    
                    # 动作的token_type_ids和attention_mask分别填充为全0和全1
                    act_token_type_ids = torch.zeros_like(act_input_ids)
                    act_attention_mask = torch.ones_like(act_input_ids)
                    
                    # 交替添加图像和动作数据
                    all_input_ids.extend([img_input_ids, act_input_ids])
                    all_token_type_ids.extend([img_token_type_ids, act_token_type_ids])
                    all_attention_mask.extend([img_attention_mask, act_attention_mask])
                else:
                    # 若没有对应的动作，添加图像数据
                    all_input_ids.append(img_input_ids)
                    all_token_type_ids.append(img_token_type_ids)
                    all_attention_mask.append(img_attention_mask)
            # 拼接所有的input_ids、token_type_ids和attention_mask
            concatenated_input_ids = torch.cat(all_input_ids, dim=1)
            concatenated_token_type_ids = torch.cat(all_token_type_ids, dim=1)
            concatenated_attention_mask = torch.cat(all_attention_mask, dim=1)
            
            # 更新pos_inputs
            final_inputs = pos_inputs.copy()
            final_inputs['input_ids'] = concatenated_input_ids
            final_inputs['token_type_ids'] = concatenated_token_type_ids
            final_inputs['attention_mask'] = concatenated_attention_mask
        else:
            final_inputs = pos_inputs
        # print("final_inputs:", final_inputs["input_ids"])
        if self.use_fast: 
            last_token_id = self.tokenizer.pad_token_id - 1
            allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)

            if self.use_cot:
                self.COT_CONFIG = GenerationConfig(
                    pad_token_id=self.model.config.pad_token_id,
                    bos_token_id=self.model.config.bos_token_id,
                    eos_token_id=151843,
                    do_sample=False,
                )
                cot_allowed_token_ids = list(range(0, self.tokenizer.pad_token_id)) + [151843]
                cot_id_processor = ActionIDConstraintLogitsProcessor(cot_allowed_token_ids)
                with torch.no_grad():
                    cot_outputs = self.model.generate(
                        final_inputs.input_ids.to(self.device),
                        self.COT_CONFIG,
                        max_new_tokens=500,
                        logits_processor=[cot_id_processor],
                        attention_mask=final_inputs.attention_mask.to(self.device),
                    )
                cot_text = cot_outputs[:, final_inputs.input_ids.shape[-1]:-1]
                reasoning = self.tokenizer.decode(cot_text[0], skip_special_tokens=True)

                boa = self.tokenizer.convert_tokens_to_ids(self.tokenizer.boa_token)
                final_inputs.input_ids = torch.cat([cot_outputs, torch.tensor([[boa]], device=cot_outputs.device)], dim=1)
                final_inputs.attention_mask = torch.ones_like(final_inputs.input_ids)
            with torch.no_grad():
                outputs = self.model.generate(
                    final_inputs.input_ids.to(self.device),
                    self.GENERATION_CONFIG,
                    max_new_tokens=100,
                    logits_processor=[action_id_processor],
                    attention_mask=final_inputs.attention_mask.to(self.device),
                )
            # omit the eoa token
            orig_outputs = outputs[:, final_inputs.input_ids.shape[-1]:]
            outputs = outputs[:, final_inputs.input_ids.shape[-1]:-1]
            last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
            processed_outputs = last_token_id_tensor - outputs
            action_outputs = self.action_tokenizer.decode(
                processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
            )
            action = action_outputs[0]
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

        else:
            pass

        # unnormalize action
        action = self.unormalize_action(action)
        # print("action:", action)

        if self.use_one_step:
            # only one step
            action_pred = action[0:1]
        else:
            # action chunk
            action_pred = action
        
        res = [self.transform_action(action[[i],:]) for i in range(action.shape[0])]
        raw_actions = [_[0] for _ in res]
        env_actions = [_[1] for _ in res]

        return raw_actions, env_actions
    
    def unormalize_action(self, action):

        if self.policy_setup == "google_robot":
            action_high = np.array([
                0.17753939667156038,
                0.14669284061245857,
                0.2179850059450077,
                0.5882710857765758,
                0.35334834816471683,
                0.4470693223284772,
                0.99980000009996
            ])
            action_low = np.array([
                -0.22624227067544056,
                -0.15126218201085617,
                -0.23251856873127252,
                -0.3538952427136002,
                -0.4202906595250484,
                -0.43766197340888247,
                -1e-10
            ])
        else:
            if self.use_cot:
                action_high = np.array([
                    0.028418295088990186,
                    0.04023438128952678,
                    0.040332944217942646,
                    0.08114985513931172,
                    0.07819607377472315,
                    0.2025440130266567,
                    0.99980000009996
                ])
                action_low = np.array([
                    -0.029267571877272303,
                    -0.04094358115000042,
                    -0.026200699773410135,
                    -0.07969299698147969,
                    -0.0938352885296676,
                    -0.20914677715629448,
                    -1e-10
                ])
            else:
                # print("use this")
                action_high = np.array([
                    0.02819482065335177,
                    0.04079562563528227,
                    0.04015785568112329,
                    0.08070396399877966,
                    0.07745134926258301,
                    0.2016542930635028,
                    0.99980000009996
                ])
                action_low = np.array([
                    -0.02887803485105428,
                    -0.04178320091122349,
                    -0.026113155505000457,
                    -0.08117201867235568,
                    -0.09309056401752747,
                    -0.20778717060421048,
                    -1e-10
                ])
        action = 0.5 * (action + 1) * (action_high - action_low) + action_low
        return action
    
    def transform_action(self, raw_actions):
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(
                raw_actions[0, 6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] 
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle 

        if self.policy_setup == "google_robot":
            current_gripper_action = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                        self.previous_gripper_action - current_gripper_action
                )
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action
            # print(f'action gripper: {action["gripper"]}')

        elif self.policy_setup == "widowx_bridge":
            relative_gripper_action = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            if relative_gripper_action[0] > 0:
                self.close_gripper_num += 1
            else:
                self.close_gripper_num = 0

            if self.close_gripper_num >= self.late_close_gripper:
                relative_gripper_action[0] = 1
            else:
                relative_gripper_action[0] = -1

            action["gripper"] = relative_gripper_action

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    
class EmuVLAInference_dis(EmuVLAInference):
    def __init__(
        self,
        emu_hub,
        vq_hub,
        vision_hub,
        device,
        policy_setup="widowx_bridge",
        window_size=1,
        fast_path="/data/user/wsong890/user68/project/UniVLA/pretrain/fast_bridge_t5_s50",
        **kwargs,
    ):
        super().__init__(emu_hub, vq_hub, vision_hub, device, policy_setup, fast_path)
        self.window_size=window_size
        self.use_dis = True
        self.vision_queue = Queue(maxsize=self.window_size)  
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)
        # self.use_mutil_maxnewtokens=use_mutil_maxnewtokens
        self.max_new_tokens=kwargs.get("max_new_tokens",625+11+72)#1024+11+40
        self.predict_action_frames=kwargs.get("action_chunk",10)
        self.steps=kwargs.get("denoise_steps",72)
        self.inference_time=[]
        self.debug_image=kwargs.get("debug_image",False)
        self.debug_image=True
        self.new_action_len=kwargs.get("new_action_len",72)
        self.count=0
        self.image_size=(200,200)
        print("self.window_size:",self.window_size)
        # print("self.predict_action_frames:",self.predict_action_frames)
        print("self.max_new_tokens:",self.max_new_tokens)
        # print("self.use_mutil_maxnewtokens:",self.use_mutil_maxnewtokens)
        # print("self.use_jacobi_generate:",self.use_jacobi_generate)
        print("self.steps:",self.steps)
        print("use video_mode:",self.video_mode)
        print("fast_path:",fast_path)
        print("use_dis:",self.use_dis)
        print("image_size:",self.image_size)
        print("new_action_len:",self.new_action_len)
    def step(self, image, goal):
        input_dict = dict()
      
        image_code, gripper_code = self.preprocess(image)

        prompt,neg_prompt = goal,""

        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + prompt]
        text_tokens = self.processor.tokenizer(text_prompt)
        
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type='pt')

        if self.video_mode:
            kwargs = dict(
                    mode='VLA_Video',
                    padding="longest",
                )
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **kwargs)
        else:
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **self.kwargs)
        
        if self.video_mode:
            # print("use video_mode")
            self.add_image(pos_inputs)
            
            # 获取历史图像和动作
            history = self.get_history()
            action_history = self.get_action_history()

            # 初始化输入ID、token类型ID和attention mask
            all_input_ids = []
            all_token_type_ids = []
            all_attention_mask = []

            # Add text
            all_input_ids.append(text_tokens['input_ids'])
            all_token_type_ids.append(text_tokens['token_type_ids'])
            all_attention_mask.append(text_tokens['attention_mask'])

            # 遍历历史图像
            for i in range(len(history)):
                # print("len(history):",len(history))
                img_input_ids = history[i]['input_ids']#自带boa
                # print("img_input_ids.shape:", img_input_ids.shape)
                img_token_type_ids = history[i]['token_type_ids']
                img_attention_mask = history[i]['attention_mask']
                if i==0 :#i2ia 第一个i不需要添加boa_token在最后
                    img_input_ids=img_input_ids[:,:-1]
                    img_token_type_ids=img_token_type_ids[:,:-1]
                    img_attention_mask=img_attention_mask[:,:-1]

                
                # 对应的动作
                if i>0 and i < len(action_history)+1 and self.vision_queue.maxsize > 1:
                    print("len(action_history):",len(action_history))
                    # print("action_history[i-1]:",action_history[i-1])
                    act_input_ids = action_history[i-1]
                    
                    # 动作的token_type_ids和attention_mask分别填充为全0和全1
                    act_token_type_ids = torch.zeros_like(act_input_ids)
                    act_attention_mask = torch.ones_like(act_input_ids)
                    
                    # 交替添加图像和动作数据
                    all_input_ids.extend([img_input_ids, act_input_ids])
                    all_token_type_ids.extend([img_token_type_ids, act_token_type_ids])
                    all_attention_mask.extend([img_attention_mask, act_attention_mask])
                else:
                    # 若没有对应的动作，添加图像数据
                    all_input_ids.append(img_input_ids)
                    all_token_type_ids.append(img_token_type_ids)
                    all_attention_mask.append(img_attention_mask)
            # all_input_ids.append(img_input_ids)
            # all_token_type_ids.append(img_token_type_ids)
            # all_attention_mask.append(img_attention_mask)
            # 拼接所有的input_ids、token_type_ids和attention_mask
            concatenated_input_ids = torch.cat(all_input_ids, dim=1)
            concatenated_token_type_ids = torch.cat(all_token_type_ids, dim=1)
            concatenated_attention_mask = torch.cat(all_attention_mask, dim=1)
            
            # 更新pos_inputs
            final_inputs = pos_inputs.copy()
            final_inputs['input_ids'] = concatenated_input_ids
            # print("final_inputs['input_ids']:", final_inputs['input_ids'])
            final_inputs['token_type_ids'] = concatenated_token_type_ids
            final_inputs['attention_mask'] = concatenated_attention_mask
        else:
            final_inputs = pos_inputs
        
        # print("final_inputs['input_ids']:", final_inputs['input_ids'])

        if self.use_fast: 
            last_token_id = self.tokenizer.pad_token_id - 1#150618->151642
            # allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            allowed_action_token_ids=list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            # allowed_image_token_ids = list(range(self.tokenizer.pad_token_id+1 , self.tokenizer.pad_token_id+1 + 32768))+\
            # [self.boi_token_id,self.eoi_token_id,self.img_token_id,self.eof_token_id]+\
            # [16, 9, 15,17, 20]
            allowed_image_token_ids = list(range(151854 , 184622))+\
            [self.img_token_id,self.eof_token_id]+\
            [9,15, 16,17, 18,19,20,21]

            # action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            mask_schedule = get_mask_chedule("cosine")
            temperature=1
            seq_len=self.max_new_tokens
            uncond_input_ids = None
            mask_token_id=151848
            top_k=None
            input_ids_gen = make_mask(history[0]['input_ids'], mask_token_id,mask_begin=151852,mask_end=151853,new_action_len=self.new_action_len).to(self.device)
            # print("input_ids.shape:",final_inputs['input_ids'].shape)
            # print("input_ids_gen.shape:",input_ids_gen.shape)
            input_ids=final_inputs['input_ids'].to(self.device)
            
            input_ids=torch.cat([input_ids,input_ids_gen],dim=1).to(self.device)
            # print("input_ids.shape:",input_ids.shape)
            # print("input_ids_gen:",input_ids_gen)
            torch.set_printoptions(threshold=2300)
            # print("input_ids:",input_ids)
            generator = torch.Generator(device=self.device).manual_seed(42)

            
            with torch.no_grad():
            
                    # print("not use vllm")
                    # print("final_inputs.input_ids:", final_inputs.input_ids)
                    # print("final_inputs.attention_mask:", final_inputs.attention_mask)
                    #记录时间 
                start = time.perf_counter()
                total_steps=self.steps
                # eoa_index=None
                for step in range(total_steps):
                    ratio = 1.0 * (step + 1) / total_steps
                    noise_schedule = mask_schedule
                    input_ids, input_ids_gen, temperature, sampled_ids = denoise(
                        self.model, 
                        input_ids, 
                        input_ids_gen,
                        # uncond_input_ids,
                        # uncond_prefix, 
                        None, 
                        # config,
                        generator, 
                        ratio, 
                        mask_token_id, 
                        noise_schedule, 
                        seq_len,
                        temperature,
                        allowed_action_token_ids,
                        allowed_image_token_ids,
                        action_len=self.new_action_len    
                        
                        )
                outputs = sampled_ids#
                
                # print("outputs:",outputs)
        
                end = time.perf_counter()
                elapsed = end - start
                # print("converge_step:", outputs["converge_step"])

                # outputs = outputs["output_token_ids"]#
                    # print("outputs.shape:", outputs.shape)
                speed = (outputs.shape[-1])/elapsed
                # print("speed:", speed)#token/s
                self.inference_time.append(speed)
  
           
            # eoa_token_index=torch.where(outputs==self.eoa_token_id)[1]
            # if len(eoa_token_index) > 0:
            #     orig_outputs = outputs[:, :eoa_token_index[0]+1].clone()
            #     outputs = outputs[:, :eoa_token_index[0]+1]
            #     # print("outputs:",outputs)
            # else:
            #     print("find no eoa")
            #     orig_outputs = outputs.clone()
            #     outputs = outputs
  
            boa_index=torch.where(outputs==torch.tensor(self.boa_token_id,dtype=outputs.dtype))#"boa_token_id": 151844,
            eoa_index=torch.where(outputs==torch.tensor(self.eoa_token_id,dtype=outputs.dtype))#"eoa_token_id": 151845,
            # print("boa_index:", boa_index)
            # print("eoa_index:", eoa_index)
            # print("outputs:", outputs)
            # print("outputs.shape:", outputs.shape)
            # if self.debug_image:
            #     img_index=torch.where(outputs==torch.tensor(self.img_token_id,dtype=outputs.dtype))
            #     eof_index=torch.where(outputs==torch.tensor(self.eof_token_id,dtype=outputs.dtype))
            #     image_1=outputs[:,img_index[1][0]+1:eof_index[1][0]]-self.bov_token_id
            #     # image_2=outputs[:,img_index[1][1]+1:eof_index[1][1]]-self.bov_token_id
            #     # print("image_1:",image_1)
            #     # print("image_2:",image_2)
            #     image_1=image_1.view(-1, 32,32)
            #     recon = self.image_tokenizer.decode(image_1)
            #     # image_2_text = self.image_tokenizer.decode(image_2)
            #     # recon = recon.view(-1, 25,25)
            #     recon_images = self.image_processor.postprocess(recon)["pixel_values"]
            #     for idx, im in enumerate(recon_images):
            #         im.save(f"./log/image_simpler_dis/image_dis_{self.count:03d}.jpg")
            #         self.count+=1
            if len(eoa_index[1])>0:
                # print("boa_index:",boa_index)
                orig_outputs=outputs.clone()
                outputs=outputs[:,boa_index[1][0]+1:eoa_index[1][0]]
                
                orig_outputs=orig_outputs[:,boa_index[1][0]+1:eoa_index[1][0]+1]#保留 eoa 去掉boa，因为图像token结束自带一个boa
            else:
                print("find no eoa")
                outputs=outputs[:,boa_index[1][0]+1:]
                # orig_outputs=orig_outputs[:,boa_index[1][0]+1:]#保留 eoa
                # 创建一个（1，10，7）的numpy，值为0
                # action_tensor = np.zeros((1,10,7),dtype=np.float32)
                # action_token_ids = self.action_tokenizer(action_tensor)
                orig_outputs = torch.cat([outputs, torch.tensor(self.eoa_token_id,dtype=outputs.dtype,device=outputs.device).view(1,1)], dim=1)
            
            # if outputs.shape[-1] > 70:
            #     print("outputs:", outputs)
            # print("outputs:", outputs)
            last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
            processed_outputs = last_token_id_tensor - outputs
            action_outputs = self.action_tokenizer.decode(
                processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
            )#[1,10,7]
            action = action_outputs[0]
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

        else:
            pass

        # unnormalize action
        action = self.unormalize_action(action)
        # print("action:", action)

        if self.use_one_step:
            # only one step
            action_pred = action[0:1]
        else:
            # action chunk
            action_pred = action
        
        res = [self.transform_action(action[[i],:]) for i in range(action.shape[0])]
        raw_actions = [_[0] for _ in res]
        env_actions = [_[1] for _ in res]

        return raw_actions, env_actions
def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    confidence = log(probs) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking

def denoise(
    model, 
    input_ids,
    input_ids_gen, 
    # uncond_input_ids, 
    # uncond_prefix, 
    attention_mask, 
    # config, 
    generator, 
    ratio, 
    mask_token_id, 
    noise_schedule, 
    seq_len=747+70, 
    temperature=1,
    allowed_action_token_ids=None,
    allowed_image_token_ids=None,
    action_len=40,
    # eoa_index=None,
    ):

    eoa_index=None
    eoa_index_tuple=torch.where(input_ids_gen==151845)
    if len(eoa_index_tuple[1])>0:
        # input_ids_gen[:,eoa_index_tuple[1][0]+1:]=mask_token_id
        eoa_index=eoa_index_tuple[1][0]
        # print("-seq_len+eoa_index+1:",-seq_len+eoa_index+1)
        input_ids[:,-seq_len+eoa_index+1:]=mask_token_id
        

    logits = forward(model, input_ids, attention_mask=attention_mask)
    logits = logits[:, -(seq_len + 1):-1, :]
    # print("logits.shape:", logits.shape)
    if allowed_action_token_ids is not None:
        mask = torch.full_like(logits[:,-action_len:,:], float('-inf'))
        allowed_action_token_ids_tensor = torch.tensor(allowed_action_token_ids, device=logits.device)  # (vocab_size, )
        mask[:,:,allowed_action_token_ids_tensor]=0
        logits[:,-action_len:,:] += mask
    if allowed_image_token_ids is not None:
        mask = torch.full_like(logits[:,:-action_len,:], float('-inf'))
        allowed_image_token_ids_tensor = torch.tensor(allowed_image_token_ids, device=logits.device)  # (vocab_size, )
        mask[:,:,allowed_image_token_ids_tensor]=0
        logits[:,:-action_len,:] += mask
    probs = logits.softmax(dim=-1)
    sampled = probs.reshape(-1, logits.size(-1))#(b*seq,vocabsize)
    top_k=None
    if top_k is not None:
        topk_probs, topk_indices = torch.topk(sampled, top_k, dim=-1)#[B*T, k]
        topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
        sampled_ids = torch.multinomial(topk_probs, 1, generator=generator)[:, 0]#[B*seq_len] 内部下标
        sampled_ids = topk_indices.gather(-1, sampled_ids.view(-1, 1)).view(*logits.shape[:-1])#[B, T]

    else:
        sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

    unknown_map = input_ids_gen == mask_token_id
    # print("eoa_index:",eoa_index)
    # if eoa_index is not None:
    #     # print("unknown_map.shape:",unknown_map.shape)
    #     # print("before unknown_map:",unknown_map)
    #     unknown_map[:,eoa_index+1:]=False
    #     # print("after unknown_map:",unknown_map)
    # print("eoa_index:",eoa_index)
    # print("unknown_map:",unknown_map)
    sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_gen)

    mask_ratio = noise_schedule(torch.tensor(ratio))
    selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])#[B, T, 1]
    selected_probs = selected_probs.squeeze(-1)#[B, T]
    selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

    mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)#(1,1)
    mask_len = torch.max(
        torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
    )#(B,1) 1<=masklen<=unkonm_map-1
    temperature = temperature * (1.0 - ratio)
    masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)

    input_ids[:, -seq_len:] = torch.where(masking, mask_token_id,sampled_ids )
    input_ids_gen = torch.where(masking, mask_token_id, sampled_ids)

    return input_ids, input_ids_gen, temperature, sampled_ids


def denoise_image(
    model, 
    input_ids,
    input_ids_gen, 
    # uncond_input_ids, 
    # uncond_prefix, 
    attention_mask, 
    # config, 
    generator, 
    ratio, 
    mask_token_id, 
    noise_schedule, 
    seq_len=747, 
    temperature=1,
    allowed_action_token_ids=None,
    allowed_image_token_ids=None,
    # eoa_index=None,
    ):
    


    logits = forward(model, input_ids, attention_mask=attention_mask)
    logits = logits[:, -(seq_len + 1):-1, :]
    # print("logits.shape:", logits.shape)
    # if allowed_action_token_ids is not None:
    #     mask = torch.full_like(logits[:,-1:,:], float('-inf'))
    #     allowed_action_token_ids_tensor = torch.tensor(allowed_action_token_ids, device=logits.device)  # (vocab_size, )
    #     mask[:,:,allowed_action_token_ids_tensor]=0
    #     logits[:,-1:,:] += mask
    if allowed_image_token_ids is not None:
        mask = torch.full_like(logits[:,:,:], float('-inf'))
        allowed_image_token_ids_tensor = torch.tensor(allowed_image_token_ids, device=logits.device)  # (vocab_size, )
        mask[:,:,allowed_image_token_ids_tensor]=0
        logits[:,:,:] += mask
    probs = logits.softmax(dim=-1)
    sampled = probs.reshape(-1, logits.size(-1))#(b*seq,vocabsize)
    top_k=None
    if top_k is not None:
        topk_probs, topk_indices = torch.topk(sampled, top_k, dim=-1)#[B*T, k]
        topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
        sampled_ids = torch.multinomial(topk_probs, 1, generator=generator)[:, 0]#[B*seq_len] 内部下标
        sampled_ids = topk_indices.gather(-1, sampled_ids.view(-1, 1)).view(*logits.shape[:-1])#[B, T]

    else:
        sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

    unknown_map = input_ids_gen == mask_token_id

    sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_gen)

    mask_ratio = noise_schedule(torch.tensor(ratio))
    selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])#[B, T, 1]
    selected_probs = selected_probs.squeeze(-1)#[B, T]
    selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

    mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)#(1,1)
    mask_len = torch.max(
        torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
    )#(B,1) 1<=masklen<=unkonm_map-1
    temperature = temperature * (1.0 - ratio)
    masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)

    input_ids[:, -seq_len:] = torch.where(masking, mask_token_id,sampled_ids )
    input_ids_gen = torch.where(masking, mask_token_id, sampled_ids)
    return input_ids, input_ids_gen, temperature, sampled_ids

def denoise_action(
    model, 
    input_ids,
    input_ids_gen, 
    # uncond_input_ids, 
    # uncond_prefix, 
    attention_mask, 
    # config, 
    generator, 
    ratio, 
    mask_token_id, 
    noise_schedule, 
    seq_len=70, 
    temperature=1,
    allowed_action_token_ids=None,

    # eoa_index=None,
    ):
   
    eoa_index=None
    eoa_index_tuple=torch.where(input_ids_gen==151845)
    if len(eoa_index_tuple[1])>0:
        # input_ids_gen[:,eoa_index_tuple[1][0]+1:]=mask_token_id
        eoa_index=eoa_index_tuple[1][0]
        # print("-seq_len+eoa_index+1:",-seq_len+eoa_index+1)
        input_ids[:,-seq_len+eoa_index+1:]=mask_token_id
        

    logits = forward(model, input_ids, attention_mask=attention_mask)
    logits = logits[:, -(seq_len + 1):-1, :]
    # print("logits.shape:", logits.shape)
    if allowed_action_token_ids is not None:
        mask = torch.full_like(logits, float('-inf'))
        allowed_action_token_ids_tensor = torch.tensor(allowed_action_token_ids, device=logits.device)  # (vocab_size, )
        mask[:,:,allowed_action_token_ids_tensor]=0
        logits += mask
    
    probs = logits.softmax(dim=-1)
    sampled = probs.reshape(-1, logits.size(-1))#(b*seq,vocabsize)
    top_k=None
    if top_k is not None:
        topk_probs, topk_indices = torch.topk(sampled, top_k, dim=-1)#[B*T, k]
        topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
        sampled_ids = torch.multinomial(topk_probs, 1, generator=generator)[:, 0]#[B*seq_len] 内部下标
        sampled_ids = topk_indices.gather(-1, sampled_ids.view(-1, 1)).view(*logits.shape[:-1])#[B, T]

    else:
        sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

    unknown_map = input_ids_gen == mask_token_id
    # print("eoa_index:",eoa_index)
    if eoa_index is not None:

        unknown_map[:,eoa_index+1:]=False

    sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_gen)

    mask_ratio = noise_schedule(torch.tensor(ratio))
    selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])#[B, T, 1]
    selected_probs = selected_probs.squeeze(-1)#[B, T]
    selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

    mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)#(1,1)
    mask_len = torch.max(
        torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
    )#(B,1) 1<=masklen<=unkonm_map-1
    temperature = temperature * (1.0 - ratio)
    masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)

    input_ids[:, -seq_len:] = torch.where(masking, mask_token_id,sampled_ids )
    input_ids_gen = torch.where(masking, mask_token_id, sampled_ids)

    return input_ids, input_ids_gen, temperature, sampled_ids

def forward(
    model,
    input_ids,
    input_embeddings=None,
    attention_mask=None,
    labels=None,
    label_smoothing=0.0,
    config=None,
    labels_mask_text=None,
    labels_mask_image=None,
    **kwargs,
):
    if attention_mask is not None:
        attention_mask = attention_mask.to(dtype=model.dtype)
    # if input_embeddings is None:
    #     logits = model.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
    # else:
    #     logits = model.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']
    dict=model(input_ids=input_ids, attention_mask=attention_mask)
    logits=dict['logits']
    past=dict['past_key_values']

    if labels is not None:
        raise NotImplementedError

    return logits

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t, generator=None):
    noise = torch.zeros_like(t).uniform_(0, 1, generator=generator)
    return -log(-log(noise))
def make_mask_image(input_ids, mask_token_id,mask_begin=151852,mask_end=151853):
    is_boi =(input_ids == mask_begin)
    is_eoi=(input_ids == mask_end)
    starts_cum = torch.cumsum(is_boi, dim=1)
    ends_cum   = torch.cumsum(is_eoi, dim=1)
    is_mask=(starts_cum-ends_cum>0) & (~is_boi) & (~is_eoi)
    mask_inputs=torch.full_like(input_ids,mask_token_id)
    mask_inputs=torch.where(is_mask,mask_inputs,input_ids)
    mask_inputs=mask_inputs[:,:-1]
    return mask_inputs
def make_mask_action(mask_token_id,new_action_len=70):
    mask_action=torch.full((1, new_action_len),mask_token_id)
    mask_action[:,0]=151844
    return mask_action

def make_mask(input_ids, mask_token_id,mask_begin=151852,mask_end=151853,new_action_len=70):
    is_boi =(input_ids == mask_begin)
    # print("is_boi:",~is_boi)
    is_eoi=(input_ids == mask_end)
    # is_boa=(input_ids == 151844)
    starts_cum = torch.cumsum(is_boi, dim=1)
    ends_cum   = torch.cumsum(is_eoi, dim=1)
    is_mask=(starts_cum-ends_cum>0) & (~is_boi) & (~is_eoi) 
    # print("is_mask:",is_mask)
    mask_inputs=torch.full_like(input_ids,mask_token_id)
    mask_inputs=torch.where(is_mask,mask_inputs,input_ids)
    mask_action=torch.full((1, new_action_len-1),mask_token_id)
    mask_inputs=torch.cat([mask_inputs,mask_action],dim=1)
    # print("mask_inputs:",mask_inputs)
    return mask_inputs
    # mask_inputs[is_mask]=input_ids[is_mask]
    # prefix=[151852, 16, 9, 17, 20, 9, 17, 20, 151851]
    # postfix=[]