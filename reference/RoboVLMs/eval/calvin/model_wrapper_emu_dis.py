import json
import os.path
from copy import deepcopy
import torch
from PIL import Image
from typing import Literal
import numpy as np
import functools

from lightning.pytorch.trainer import Trainer

from eval_utils import init_trainer_config, euler2rotm, rotm2euler
# from robovlms.train.base_trainer import BaseTrainer
# from robovlms.utils.model_utils import build_tokenizer
# from robovlms.data.datamodule.gr_datamodule import GRDataModule
# from robovlms.data.data_utils import get_text_function
# from robovlms.data.data_utils import (
#     preprocess_image,
#     get_prompt_builder,
#     tcp_to_world_frame,
# )
# from robovlms.model.policy_head.action_tokenizer import ActionTokenizer
# from robovlms.data.data_utils import unnoramalize_action
from queue import Queue


# emu3
from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import sys
sys.path.append("/data/user/wsong890/user68/project/UniVLA/reference/Emu3")
from emu3.sampling import get_mask_chedule
from emu3.mllm import Emu3Tokenizer, Emu3ForCausalLM, Emu3Processor
from emu3.mllm import Emu3MoE
from transformers import LogitsProcessor
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

class EmuVLAModel:
    def __init__(self,emu_hub,vq_hub,vision_hub,device,raw_calvin=True,use_jacobi_generate=False):
        self.emu_hub = emu_hub
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.device = device
        self.raw_calvin = raw_calvin

        # vllm
        self.vllm_accelerator = False
        ## hard code here
        self.window_size = 2
        self.predict_action_frames = 10
        self.normalize_action = True
        self.video_mode = True
        # load model and tokenizer
        self.init_config(device=device)
        self.action_space = "continuous"
        self.image_processor.min_pixels = 80 * 80

        self.predict_frames = 1
        self.context_frames = 1
        self.action_dim = 7
        self.diffusion_steps = 30
        self.classifier_free_guidance = 1.0
        self.target_length = 800

        # flow matching
        # self.use_gripper = True
        # self.use_fast = False
        # self.use_one_step = True

        self.use_gripper = True
        self.use_fast = True
        self.use_one_step = False
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
        self.use_jacobi_generate=use_jacobi_generate
        self.inference_time=[]

        self.kwargs = dict(
            mode='VLA',
            padding="longest",
        )
        if self.use_fast:
            if self.vllm_accelerator:
                pass
            else:
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
        
        if self.vllm_accelerator:
            from vllm import LLM, SamplingParams, TokensPrompt
            self.tokenizer = Emu3Tokenizer.from_pretrained(
                self.vq_hub,
                model_max_length=2000,
                padding_side="right",
                use_fast=False,
            )
            self.model = LLM(
                model=self.emu_hub,
                skip_tokenizer_init=True,
                trust_remote_code=True,
                dtype='bfloat16',
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                disable_log_stats=False,
                enable_chunked_prefill=True,
                max_seq_len_to_capture=2000,
                gpu_memory_utilization=0.8,
            )
        else:
            self.model = Emu3MoE.from_pretrained(
                self.emu_hub,
                torch_dtype=torch.bfloat16,
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

        # fast tokenization
        if self.normalize_action:
            fast_path = '/data/user/wsong890/user68/project/UniVLA/pretrain/fast_calvin_abcd_a10_s50'
        else:
            fast_path = "/data/user/wsong890/user68/project/UniVLA/pretrain/fast_calvin_abcd_a10_s50"

        self.action_tokenizer = AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)

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

    def step(self, obs, goal):
        """Step function."""
        # preprocess observation
        torch.set_printoptions(threshold=2000)

        image_code, gripper_code = self.preprocess(obs, goal, self.action_space)

        input_dict = dict()
        prompt,neg_prompt = goal, ""

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
            neg_inputs = self.processor.video_process(text=neg_prompt, video_tokens=video_code, gripper_tokens=gripper_code, context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt",**self.kwargs)
        
        if self.video_mode:
            # print("pos_inputs:", pos_inputs)
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
                # print(f"img_input_ids{i}:{img_input_ids}")

                # 对应的动作
                if i < len(action_history):
                    act_input_ids = action_history[i]
                    # print(f"act_input_ids{i}:{act_input_ids}")
                    
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

        if self.use_fast: 
            last_token_id = self.tokenizer.pad_token_id - 1#151643-1
            allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            # print("last_token_id:",last_token_id)
            # print("self.action_tokenizer.vocab_size:",self.action_tokenizer.vocab_size)
            with torch.no_grad():
                if self.vllm_accelerator:
                    print("use vllm")
                    from vllm import LLM, SamplingParams, TokensPrompt
                    self.GENERATION_CONFIG = SamplingParams(
                        temperature=0.0,            # Set to 0 for deterministic behavior (do_sample=False equivalent)
                        top_p=1.0,                  # No probability filtering
                        top_k=-1,                   # No token count limiting
                        max_tokens=50,             # Maximum tokens to generate
                        stop_token_ids=[self.eoa_token_id],  # End generation at this token
                        skip_special_tokens=False,  # Keep special tokens
                        logits_processors=[action_id_processor],
                    )
                    input_ids_list = final_inputs.input_ids.detach().cpu().tolist()[0]
                    inputs = TokensPrompt(prompt_token_ids=input_ids_list)
                    # print("inputs:", inputs)
                    outputs = self.model.generate(inputs,self.GENERATION_CONFIG)
                    token_ids_tuple = outputs[0].outputs[0].token_ids
                    # print("token_ids_tuple:", token_ids_tuple)
                    token_ids_list = list(token_ids_tuple)
                    outputs = torch.tensor([token_ids_list], dtype=torch.long)
                elif self.use_jacobi_generate:
                    # print("not use vllm")
                    # print("final_inputs.input_ids:", final_inputs.input_ids)
                    # print("final_inputs.attention_mask:", final_inputs.attention_mask)
                    #记录时间 
                    start = time.perf_counter()
                    outputs = self.model.generate_jacobi_kv(
                        final_inputs.input_ids.to(self.device),
                        max_new_tokens=70,
                        max_iter=70,
                        attention_mask=final_inputs.attention_mask.to(self.device),
                        eos_token=self.eoa_token_id,
                    )
                    end = time.perf_counter()
                    elapsed = end - start
                    # print("converge_step:", outputs["converge_step"])

                    outputs = outputs["output_token_ids"]#
                    speed = (outputs.shape[1])/elapsed
                    self.inference_time.append(speed)
                    # print("speed:", speed)
                    # print("outputs:", outputs)
                else:
                    # print("not use vllm")
                    # print("final_inputs.input_ids:", final_inputs.input_ids)
                    # print("final_inputs.attention_mask:", final_inputs.attention_mask)
                    start = time.perf_counter()
                    outputs = self.model.generate(
                        final_inputs.input_ids.to(self.device),
                        self.GENERATION_CONFIG,
                        max_new_tokens=100,
                        logits_processor=[action_id_processor],
                        attention_mask=final_inputs.attention_mask.to(self.device),
                    )
                    end = time.perf_counter()
                    elapsed = end - start
                    speed = (outputs.shape[-1]-final_inputs.input_ids.shape[-1])/elapsed
                    self.inference_time.append(speed)
                    print("speed:", speed)
                    # print("outputs:", outputs)
            # omit the eoa token
            
            if self.vllm_accelerator:
                orig_outputs = outputs
                outputs = outputs[:, :-1]
            elif self.use_jacobi_generate:
                eoa_token_index=torch.where(outputs==self.eoa_token_id)[1]
                # print("eoa_token_index:", eoa_token_index)
                if len(eoa_token_index) > 0:
                    eoa_token_index = eoa_token_index[0]
                    orig_outputs = outputs[:, :eoa_token_index+1]
                    outputs = outputs[:, :eoa_token_index]
                else:
                    orig_outputs = outputs
                    outputs = outputs
                # print("outputs:", outputs)

            else:
                # print("outputs:", outputs)
                orig_outputs = outputs[:, final_inputs.input_ids.shape[-1]:]#action+eoa
                outputs = outputs[:, final_inputs.input_ids.shape[-1]:-1]
            # print("outputs:", outputs)
            last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
            processed_outputs = last_token_id_tensor - outputs
            # print("processed_outputs:", processed_outputs)
            action_outputs = self.action_tokenizer.decode(
                processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
            )
            # print("action_outputs:", action_outputs)
            action = torch.from_numpy(action_outputs[0])
            # print("action:", action)
            if self.video_mode:
                # print("use video_mode")
                # print("orig_outputs:", orig_outputs)
                self.add_action(orig_outputs.detach().cpu())

        else:
            # logits_processor
            h = pos_inputs.video_size[:, 0]
            w = pos_inputs.video_size[:, 1]
            t = pos_inputs.video_size[:, 2]
            constrained_fn = self.processor.build_prefix_constrained_fn_video(h, w, t)
            logits_processor = LogitsProcessorList([
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    self.classifier_free_guidance,
                    self.model,
                    unconditional_ids=neg_inputs.input_ids.to(self.device),
                ),
                PrefixConstrainedLogitsProcessor(
                    constrained_fn,
                    num_beams=1,
                ),
            ])
            # model inference
            with torch.no_grad():
                worldmodel_outputs = self.model.generate(
                            pos_inputs.input_ids.to(self.device),
                            self.GENERATION_CONFIG,
                            logits_processor=logits_processor,
                            attention_mask=pos_inputs.attention_mask.to(self.device),
                        )   
                # worldmodel_outputs = pos_inputs.input_ids.to(self.device)
                padding = torch.full((1, self.target_length - worldmodel_outputs.shape[1]), self.model.config.pad_token_id, dtype=worldmodel_outputs.dtype).to(worldmodel_outputs.device)
                worldmodel_outputs = torch.cat([worldmodel_outputs, padding], dim=1)        
                action_outputs = self.model.generate_action(
                        outputs = worldmodel_outputs,
                        sample_steps = self.diffusion_steps,
                        frames = self.predict_action_frames,
                        action_dim = self.action_dim,
                    )    
            action = action_outputs[0].detach().cpu()

        self.rollout_step_counter += 1

        if self.normalize_action:
            action = self.unnormalize_action(action)

        if action.ndim >= 2:
            action[..., -1] = torch.where(action[..., -1] > 0, 1, -1)
        else:
            action[-1] = 1 if action[-1] > 0 else -1

        if self.use_one_step:
            # only one step
            action_pred = np.array(action[0].to(torch.float32))
        else:
            # action chunk
            action_pred = np.array(action.to(torch.float32))
        # print(f"step {self.rollout_step_counter} action {action_pred}")
        return action_pred

    def unnormalize_action(self, action):
        # partial
        # action_high = torch.tensor([
        #     0.68240000006824,
        #     0.5500000000549998,
        #     0.5940000000593999,
        #     0.4292000000429199,
        #     0.40320000004032,
        #     0.9996000000999599,
        #     0.9996000000999599
        # ], dtype=action.dtype, device=action.device)
        # action_low = torch.tensor([
        #     -0.70240000007024,
        #     -0.56760000005676,
        #     -0.430000000043,
        #     -0.42280000004228,
        #     -0.45240000004524006,
        #     -1.0000000001,
        #     -1.0000000001
        # ], dtype=action.dtype, device=action.device)

        # ABC
        # action_high = torch.tensor([
        #     0.68480000006848,
        #     0.5612000000561199,
        #     0.5952000000595199,
        #     0.4340000000433999,
        #     0.42280000004228,
        #     0.9996000000999599,
        #     0.9996000000999599
        # ], dtype=action.dtype, device=action.device)
        
        # action_low = torch.tensor([
        #     -0.7080000000708,
        #     -0.57840000005784,
        #     -0.43320000004332004,
        #     -0.42760000004276,
        #     -0.47640000004764005,
        #     -1.0000000001,
        #     -1.0000000001
        # ], dtype=action.dtype, device=action.device)

        # ABCD
        action_high = torch.tensor([
            0.67640000006764,
            0.5560000000555998,
            0.5944000000594398,
            0.42640000004264,
            0.41200000004119985,
            0.9996000000999599,
            0.9996000000999599
        ], dtype=action.dtype, device=action.device)
        
        action_low = torch.tensor([
            -0.69960000006996,
            -0.57760000005776,
            -0.4336000000433601,
            -0.42320000004232006,
            -0.46520000004652007,
            -1.0000000001,
            -1.0000000001
        ], dtype=action.dtype, device=action.device)
        
        action = 0.5 * (action + 1) * (action_high - action_low) + action_low
        return action

    def preprocess(self, obs, lang, mode="continuous"):
        # preprocess image
        image = obs["rgb_obs"]["rgb_static"]
        image = Image.fromarray(image)
        image_x = self.image_processor(image, return_tensors="pt")["pixel_values"].cuda()
        image_code = self.image_tokenizer.encode(image_x)
        # print("image_code,shape:",image_code.shape)
        
        gripper_x = None
        if "rgb_gripper" in obs["rgb_obs"]:
            gripper = obs["rgb_obs"]["rgb_gripper"]
            gripper = Image.fromarray(gripper)
            gripper = gripper.resize((80, 80))
            gripper_x = self.image_processor(gripper, return_tensors="pt")["pixel_values"].cuda()
            gripper_code = self.image_tokenizer.encode(gripper_x)

        return (
            image_code,
            gripper_code,
        )

    def reset(self):

        self.rgb_list = []
        self.hand_rgb_list = []
        self.rollout_step_counter = 0
        self.action_hist_list = []

        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()




class EmuVLAModel_i_ia_dis(EmuVLAModel):
    def __init__(self,emu_hub,vq_hub,vision_hub,device,raw_calvin=True,use_jacobi_generate=False,window_size=1,use_mutil_maxnewtokens=False,**kwargs):
        super().__init__(emu_hub,vq_hub,vision_hub,device,raw_calvin,use_jacobi_generate)
        self.window_size=window_size
        
        torch.set_printoptions(threshold=2000)
        self.vision_queue = Queue(maxsize=self.window_size)  
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)
        self.use_mutil_maxnewtokens=use_mutil_maxnewtokens
        self.max_new_tokens=kwargs.get("max_new_tokens",747+70)
        self.predict_action_frames=kwargs.get("action_chunk",10)
        self.steps=kwargs.get("steps",72)
        self.debug_image=kwargs.get("debug_image",False)
        # self.debug_image=True
        self.count=0
        print("self.window_size:",self.window_size)
        print("self.predict_action_frames:",self.predict_action_frames)
        print("self.max_new_tokens:",self.max_new_tokens)
        print("self.use_mutil_maxnewtokens:",self.use_mutil_maxnewtokens)
        print("self.use_jacobi_generate:",self.use_jacobi_generate)
        print("self.steps:",self.steps)
        print("debug_image",self.debug_image)


    def step(self, obs, goal):
        """Step function."""
        # preprocess observation
        if self.debug_image:
            image = obs["rgb_obs"]["rgb_static"]
            image = Image.fromarray(image)
            image.save(f"./log/image_origin/obs_{self.count:03d}.jpg")
            # self.count+=1
        image_code, gripper_code = self.preprocess(obs, goal, self.action_space)

        # input_dict = dict()
        prompt,neg_prompt = goal, ""

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
            neg_inputs = self.processor.video_process(text=neg_prompt, video_tokens=video_code, gripper_tokens=gripper_code, context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt",**self.kwargs)
        
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

        if self.use_fast: 
            last_token_id = self.tokenizer.pad_token_id - 1#150618->151642
            # allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            allowed_action_token_ids=list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            # allowed_image_token_ids = list(range(self.tokenizer.pad_token_id+1 , self.tokenizer.pad_token_id+1 + 32768))+\
            # [self.boi_token_id,self.eoi_token_id,self.img_token_id,self.eof_token_id]+\
            # [16, 9, 15,17, 20]
            allowed_image_token_ids = list(range(151854 , 184622))+\
            [self.img_token_id,self.eof_token_id]+\
            [16, 9, 15,17, 20]

            # action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            mask_schedule = get_mask_chedule("cosine")
            temperature=1
            seq_len=747+70
            uncond_input_ids = None
            mask_token_id=151848
            top_k=None
            input_ids_gen = make_mask(history[0]['input_ids'], mask_token_id,mask_begin=151852,mask_end=151853,new_action_len=70).to(self.device)
            # print("input_ids.shape:",final_inputs['input_ids'].shape)
            # print("input_ids_gen.shape:",input_ids_gen.shape)
            input_ids=final_inputs['input_ids'].to(self.device)
            input_ids=torch.cat([input_ids,input_ids_gen],dim=1).to(self.device)

            generator = torch.Generator(device=self.device).manual_seed(42)

            
            with torch.no_grad():
            
                    # print("not use vllm")
                    # print("final_inputs.input_ids:", final_inputs.input_ids)
                    # print("final_inputs.attention_mask:", final_inputs.attention_mask)
                    #记录时间 
                start = time.perf_counter()
                total_steps=self.steps
                eoa_index=None
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
  
            # omit the eoa token
           
        
            eoa_token_index=torch.where(outputs==self.eoa_token_id)[1]
            if len(eoa_token_index) > 0:
                orig_outputs = outputs[:, :eoa_token_index[0]+1].clone()
                outputs = outputs[:, :eoa_token_index[0]+1]
                # print("outputs:",outputs)
            else:
                print("find no eoa")
                orig_outputs = outputs.clone()
                outputs = outputs
  
            boa_index=torch.where(outputs==torch.tensor(self.boa_token_id,dtype=outputs.dtype))#"boa_token_id": 151844,
            eoa_index=torch.where(outputs==torch.tensor(self.eoa_token_id,dtype=outputs.dtype))#"eoa_token_id": 151845,
            if self.debug_image:
                img_index=torch.where(outputs==torch.tensor(self.img_token_id,dtype=outputs.dtype))
                eof_index=torch.where(outputs==torch.tensor(self.eof_token_id,dtype=outputs.dtype))
                image_1=outputs[:,img_index[1][0]+1:eof_index[1][0]]-self.bov_token_id
                # image_2=outputs[:,img_index[1][1]+1:eof_index[1][1]]-self.bov_token_id
                # print("image_1:",image_1)
                # print("image_2:",image_2)
                image_1=image_1.view(-1, 25,25)
                recon = self.image_tokenizer.decode(image_1)
                # image_2_text = self.image_tokenizer.decode(image_2)
                # recon = recon.view(-1, 25,25)
                recon_images = self.image_processor.postprocess(recon)["pixel_values"]
                for idx, im in enumerate(recon_images):
                    im.save(f"./log/image_dis/image_dis_{self.count:03d}.jpg")
                    self.count+=1



            # print("boa_index:", boa_index)
            # print("eoa_index:", eoa_index)
            # print("outputs:", outputs)
            # print("outputs.shape:", outputs.shape)
            if len(eoa_index[1])>0:
                # print("boa_index:",boa_index)
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
            
            if outputs.shape[-1] > 70:
                print("outputs:", outputs)
            # print("outputs:", outputs)
            last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
            processed_outputs = last_token_id_tensor - outputs
            action_outputs = self.action_tokenizer.decode(
                processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
            )#[1,10,7]
            action = torch.from_numpy(action_outputs[0])
            # if torch.all(action == 0): 
            #     action_tensor = np.zeros((1,10,7),dtype=np.float32)
            #     action_token_ids = self.action_tokenizer(action_tensor)
            #     orig_outputs = torch.cat(
            #         [torch.as_tensor(action_token_ids,dtype=orig_outputs.dtype),
            #           torch.tensor(self.eoa_token_id,dtype=orig_outputs.dtype).view(1,1)], 
            #           dim=1)

            # print("action:", action)
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

    

        self.rollout_step_counter += 1

        if self.normalize_action:
            action = self.unnormalize_action(action)

        if action.ndim >= 2:
            action[..., -1] = torch.where(action[..., -1] > 0, 1, -1)
        else:
            action[-1] = 1 if action[-1] > 0 else -1

        if self.use_one_step:
            # only one step
            action_pred = np.array(action[0].to(torch.float32))
        else:
            # action chunk
            action_pred = np.array(action.to(torch.float32))
        # print(f"step {self.rollout_step_counter} action {action_pred}")
        return action_pred
    
    
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
    # eoa_index=None,
    ):
    # if uncond_input_ids is not None and config.training.guidance_scale > 0:
    #     uncond_input_ids = torch.cat(
    #         [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
    #     model_input = torch.cat([input_ids, uncond_input_ids])
    #     cond_logits, uncond_logits = forward(model, model_input, attention_mask=attention_mask).chunk(2)
    #     logits = (1 + config.training.guidance_scale) * cond_logits - config.training.guidance_scale * uncond_logits
    #     logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]
    # else:
    #     logits = forward(model, input_ids, attention_mask=attention_mask)
    #     logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]
    # if input_ids存在eoa_token_id,则把第一个eoa后面的token都变成mask_token_id
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
        mask = torch.full_like(logits[:,-70:,:], float('-inf'))
        allowed_action_token_ids_tensor = torch.tensor(allowed_action_token_ids, device=logits.device)  # (vocab_size, )
        mask[:,:,allowed_action_token_ids_tensor]=0
        logits[:,-70:,:] += mask
    if allowed_image_token_ids is not None:
        mask = torch.full_like(logits[:,:-70,:], float('-inf'))
        allowed_image_token_ids_tensor = torch.tensor(allowed_image_token_ids, device=logits.device)  # (vocab_size, )
        mask[:,:,allowed_image_token_ids_tensor]=0
        logits[:,:-70,:] += mask
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


def build_blockwise_attn_mask(
    sequence: torch.Tensor,                     # [B, L]
    boi_token_id: int = 151852,
    eoi_token_id: int = 151853,
    boa_token_id: int = 151844,
    eoa_token_id: int = 151845,
    pad_token_id: int = 151643,
    include_boundary_as_image: bool = True,     # True: <BOI>/<EOI> 也算图像块
    return_bool_mask: bool = True,              # False: 返回加性 bias(允许=0/屏蔽=-inf)
    past_key_values_length: int = 0,            # <-- KV cache 长度（列前缀）
):
    B, L = sequence.shape
    device = sequence.device
    P = int(past_key_values_length)             # 过去的 key 列数
    K = P + L                                   # 总 key 列

    # 标注
    is_pad = (sequence == pad_token_id)
    is_boi = (sequence == boi_token_id) | (sequence == boa_token_id)
    is_eoi = (sequence == eoi_token_id) | (sequence == eoa_token_id)

    # 图像/动作块定位
    starts_cum = torch.cumsum(is_boi, dim=1)
    ends_cum   = torch.cumsum(is_eoi, dim=1)
    level = (starts_cum - ends_cum).clamp(min=0)
    is_image = (level > 0) | is_boi | is_eoi if include_boundary_as_image else (level > 0)
    is_text = ~is_image

    # 切块 + 块序号
    cur = is_image
    prev = torch.zeros_like(cur); prev[:, 1:] = cur[:, :-1]
    block_start = (cur != prev) | torch.zeros((B,1), dtype=torch.bool, device=device)
    block_id = torch.cumsum(block_start.to(torch.int32), dim=1)  # [B,L] 1-based

    # 块间因果（当前段内）
    query_block = block_id[:, :, None]   # [B,L,1]
    key_block   = block_id[:, None, :]   # [B,1,L]
    allowed_cur = (key_block <= query_block)      # [B,L,L]

    # 文本块内因果（当前段内）
    tril = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
    allowed_cur &= ((~is_text)[:, :, None] | tril[None, :, :])

    # 屏蔽当前段内的 PAD 列（列维度）
    allowed_cur &= ~is_pad[:, None, :]   # [B,L,L]

    # ===== 关键：拼接过去的列（KV cache）到左侧，得到 [B,L,K] =====
    if P > 0:
        # 过去列：对所有 query 允许（它们都属于“之前的块”）；如需更严格规则，可在这里加限制
        allowed_past = torch.ones((B, L, P), dtype=torch.bool, device=device)
        allowed_full = torch.cat([allowed_past, allowed_cur], dim=-1)  # [B,L,P+L]
    else:
        allowed_full = allowed_cur  # [B,L,L]

    # 如果你还想禁止“PAD 行”作为查询（较少见），可再： allowed_full[is_pad] = False

    mask4d = allowed_full.unsqueeze(1)  # [B,1,L,K]
    if return_bool_mask:
        return mask4d
    else:
        bias = (~mask4d).to(torch.float32)
        bias = bias.masked_fill(bias.bool(), float("-inf"))
        return bias


# def denoise(
#     model, 
#     input_ids,
#     input_ids_gen, 
#     # uncond_input_ids, 
#     # uncond_prefix, 
#     attention_mask, 
#     # config, 
#     generator, 
#     ratio, 
#     mask_token_id, 
#     noise_schedule, 
#     seq_len=747+70, 
#     temperature=1,
#     allowed_action_token_ids=None,
#     allowed_image_token_ids=None,
#     # eoa_index=None,
#     ):
#     # if uncond_input_ids is not None and config.training.guidance_scale > 0:
#     #     uncond_input_ids = torch.cat(
#     #         [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
#     #     model_input = torch.cat([input_ids, uncond_input_ids])
#     #     cond_logits, uncond_logits = forward(model, model_input, attention_mask=attention_mask).chunk(2)
#     #     logits = (1 + config.training.guidance_scale) * cond_logits - config.training.guidance_scale * uncond_logits
#     #     logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]
#     # else:
#     #     logits = forward(model, input_ids, attention_mask=attention_mask)
#     #     logits = logits[:, -(seq_len + 1):-1, config.model.showo.llm_vocab_size + 10:-1]
#     # if input_ids存在eoa_token_id,则把第一个eoa后面的token都变成mask_token_id
#     eoa_index=None
#     eoa_index_tuple=torch.where(input_ids==151845)
#     if len(eoa_index_tuple[1])>0:
#         # input_ids_gen[:,eoa_index_tuple[1][0]+1:]=mask_token_id
#         eoa_index=eoa_index_tuple[1][0]
#         # print("-seq_len+eoa_index+1:",-seq_len+eoa_index+1)
#         input_ids[:,eoa_index+1:]=mask_token_id
        

#     logits = forward(model, input_ids, attention_mask=attention_mask)
#     logits = logits[:, -(seq_len + 1):-1, :]
#     # print("logits.shape:", logits.shape)
#     if allowed_action_token_ids is not None:
#         mask = torch.full_like(logits[:,-71:,:], float('-inf'))
#         allowed_action_token_ids_tensor = torch.tensor(allowed_action_token_ids, device=logits.device)  # (vocab_size, )
#         mask[:,:,allowed_action_token_ids_tensor]=0
#         logits[:,-71:,:] += mask
#     if allowed_image_token_ids is not None:
#         mask = torch.full_like(logits[:,:-71,:], float('-inf'))
#         allowed_image_token_ids_tensor = torch.tensor(allowed_image_token_ids, device=logits.device)  # (vocab_size, )
#         mask[:,:,allowed_image_token_ids_tensor]=0
#         logits[:,:-71,:] += mask
#     probs = logits.softmax(dim=-1)
#     sampled = probs.reshape(-1, logits.size(-1))#(b*seq,vocabsize)
#     top_k=None
#     if top_k is not None:
#         topk_probs, topk_indices = torch.topk(sampled, top_k, dim=-1)#[B*T, k]
#         topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
#         sampled_ids = torch.multinomial(topk_probs, 1, generator=generator)[:, 0]#[B*seq_len] 内部下标
#         sampled_ids = topk_indices.gather(-1, sampled_ids.view(-1, 1)).view(*logits.shape[:-1])#[B, T]

#     else:
#         sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

#     unknown_map = input_ids_gen == mask_token_id
    
#     # print("eoa_index:",eoa_index)
#     if eoa_index is not None:
#         # print("unknown_map.shape:",unknown_map.shape)
#         # print("before unknown_map:",unknown_map)
#         len_prompt=input_ids.shape[1]-seq_len
#         unknown_map[:,eoa_index-len_prompt+1:]=False
#         # unknown_map[:,eoa_index+1:]=False
#         # print("after unknown_map:",unknown_map)
#     # print("eoa_index:",eoa_index)
#     # print("unknown_map:",unknown_map)
#     sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_gen)

#     mask_ratio = noise_schedule(torch.tensor(ratio))
#     selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])#[B, T, 1]
#     selected_probs = selected_probs.squeeze(-1)#[B, T]
#     selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

#     mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)#(1,1)
#     mask_len = torch.max(
#         torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
#     )#(B,1) 1<=masklen<=unkonm_map-1
#     temperature = temperature * (1.0 - ratio)
#     masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)

#     input_ids[:, -seq_len:] = torch.where(masking, mask_token_id,sampled_ids )
#     input_ids_gen = torch.where(masking, mask_token_id, sampled_ids)

#     return input_ids, input_ids_gen, temperature, sampled_ids


class EmuVLAModel_i_ia_dis_2stage(EmuVLAModel):
    def __init__(self,emu_hub,vq_hub,vision_hub,device,raw_calvin=True,use_jacobi_generate=False,window_size=1,use_mutil_maxnewtokens=False,**kwargs):
        super().__init__(emu_hub,vq_hub,vision_hub,device,raw_calvin,use_jacobi_generate)
        self.window_size=window_size
        
        torch.set_printoptions(threshold=2000)
        self.vision_queue = Queue(maxsize=self.window_size)  
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)
        self.use_mutil_maxnewtokens=use_mutil_maxnewtokens
        self.max_new_tokens=kwargs.get("max_new_tokens",747+70)
        self.predict_action_frames=kwargs.get("action_chunk",10)
        self.action_steps=kwargs.get("action_steps",28)
        self.image_steps=kwargs.get("image_steps",64)
        self.steps=kwargs.get("steps",72)
        print("use EmuVLAModel_i_ia_dis_2stage")
        print("self.window_size:",self.window_size)
        print("self.predict_action_frames:",self.predict_action_frames)
        print("self.max_new_tokens:",self.max_new_tokens)
        print("self.use_mutil_maxnewtokens:",self.use_mutil_maxnewtokens)
        print("self.use_jacobi_generate:",self.use_jacobi_generate)
        print("self.action_steps:",self.action_steps)
        print("self.image_steps:",self.image_steps)
        # print("self.steps:",self.steps)


    def step(self, obs, goal):
        """Step function."""
        # preprocess observation
        image_code, gripper_code = self.preprocess(obs, goal, self.action_space)

        # input_dict = dict()
        prompt,neg_prompt = goal, ""

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
            neg_inputs = self.processor.video_process(text=neg_prompt, video_tokens=video_code, gripper_tokens=gripper_code, context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt",**self.kwargs)
        
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

        if self.use_fast: 
            last_token_id = self.tokenizer.pad_token_id - 1#150618->151642
            # allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            allowed_action_token_ids=list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            # allowed_image_token_ids = list(range(self.tokenizer.pad_token_id+1 , self.tokenizer.pad_token_id+1 + 32768))+\
            # [self.boi_token_id,self.eoi_token_id,self.img_token_id,self.eof_token_id]+\
            # [16, 9, 15,17, 20]
            allowed_image_token_ids = list(range(151854 , 184622))+\
            [self.img_token_id,self.eof_token_id]+\
            [16, 9, 15,17, 20]

            # action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            mask_schedule = get_mask_chedule("cosine")
            temperature=1
            
            uncond_input_ids = None
            mask_token_id=151848
            top_k=None
            
            # print("input_ids.shape:",final_inputs['input_ids'].shape)
            # print("input_ids_gen.shape:",input_ids_gen.shape)
            input_ids=final_inputs['input_ids'].to(self.device)
            

            generator = torch.Generator(device=self.device).manual_seed(42)

            
            with torch.no_grad():
            
                    # print("not use vllm")
                    # print("final_inputs.input_ids:", final_inputs.input_ids)
                    # print("final_inputs.attention_mask:", final_inputs.attention_mask)
                    #记录时间 
                start = time.perf_counter()
                iamge_steps=self.image_steps
                seq_len=747
                eoa_index=None
                input_ids_gen = make_mask_image(history[0]['input_ids'], mask_token_id,mask_begin=151852,mask_end=151853).to(self.device)
                # print("input_ids_gen,shape:",input_ids_gen.shape)
                input_ids=torch.cat([input_ids,input_ids_gen],dim=1).to(self.device)
                for step in range(iamge_steps):
                    ratio = 1.0 * (step + 1) / iamge_steps
                    noise_schedule = mask_schedule
                    input_ids, input_ids_gen, temperature, sampled_ids = denoise_image(
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
                        
                        )
                # print("output_iamge_ids:",sampled_ids)
                input_ids[:,-seq_len:] = sampled_ids#
                input_ids_gen=make_mask_action(mask_token_id,new_action_len=70).to(self.device)
                # print("input_ids_gen.shape:",input_ids_gen.shape)
                input_ids=torch.cat([input_ids,input_ids_gen],dim=1).to(self.device)
                action_steps=self.action_steps
                seq_len=70
                for step in range(action_steps):
                    ratio = 1.0 * (step + 1) / action_steps
                    noise_schedule = mask_schedule
                    input_ids, input_ids_gen, temperature, sampled_ids = denoise_action(
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
                        )
                # print("sampled_ids.shape:",sampled_ids.shape)
                outputs=sampled_ids
                # print("outputs:",outputs)
                end = time.perf_counter()
                elapsed = end - start
                # print("converge_step:", outputs["converge_step"])

                # outputs = outputs["output_token_ids"]#
                    # print("outputs.shape:", outputs.shape)
                speed = (outputs.shape[-1])/elapsed
                # print("speed:", speed)#token/s
                self.inference_time.append(speed)
  
            # omit the eoa token
           
        
            eoa_token_index=torch.where(outputs==self.eoa_token_id)[1]
            if len(eoa_token_index) > 0:
                orig_outputs = outputs[:, :eoa_token_index[0]+1].clone()
                outputs = outputs[:, :eoa_token_index[0]+1]
                # print("outputs:",outputs)
            else:
                print("find no eoa")
                orig_outputs = outputs.clone()
                outputs = outputs
  
            boa_index=torch.where(outputs==torch.tensor(self.boa_token_id,dtype=outputs.dtype))#"boa_token_id": 151844,
            eoa_index=torch.where(outputs==torch.tensor(self.eoa_token_id,dtype=outputs.dtype))#"eoa_token_id": 151845,
            # print("boa_index:", boa_index)
            # print("eoa_index:", eoa_index)
            # print("outputs:", outputs)
            # print("outputs.shape:", outputs.shape)
            if len(eoa_index[1])>0:
                # print("boa_index:",boa_index)
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
            
            if outputs.shape[-1] > 70:
                print("outputs:", outputs)
            # print("outputs:", outputs)
            last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
            processed_outputs = last_token_id_tensor - outputs
            action_outputs = self.action_tokenizer.decode(
                processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
            )#[1,10,7]
            action = torch.from_numpy(action_outputs[0])
            # if torch.all(action == 0): 
            #     action_tensor = np.zeros((1,10,7),dtype=np.float32)
            #     action_token_ids = self.action_tokenizer(action_tensor)
            #     orig_outputs = torch.cat(
            #         [torch.as_tensor(action_token_ids,dtype=orig_outputs.dtype),
            #           torch.tensor(self.eoa_token_id,dtype=orig_outputs.dtype).view(1,1)], 
            #           dim=1)

            # print("action:", action)
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

    

        self.rollout_step_counter += 1

        if self.normalize_action:
            action = self.unnormalize_action(action)

        if action.ndim >= 2:
            action[..., -1] = torch.where(action[..., -1] > 0, 1, -1)
        else:
            action[-1] = 1 if action[-1] > 0 else -1

        if self.use_one_step:
            # only one step
            action_pred = np.array(action[0].to(torch.float32))
        else:
            # action chunk
            action_pred = np.array(action.to(torch.float32))
        # print(f"step {self.rollout_step_counter} action {action_pred}")
        return action_pred
    
    
def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    confidence = log(probs) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking