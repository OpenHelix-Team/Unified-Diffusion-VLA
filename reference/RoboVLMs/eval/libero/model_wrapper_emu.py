import json
import torch
import numpy as np
from queue import Queue
from PIL import Image

from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import sys
# sys.path.append("/data/user/wsong890/user68/project/UniVLA/reference/Emu3")
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[3] / "Emu3"))
from emu3.mllm import Emu3Tokenizer, Emu3ForCausalLM, Emu3Processor
from emu3.mllm import Emu3MoE
from transformers import LogitsProcessor
from emu3.sampling import get_mask_chedule
import time
import logging
import matplotlib.pyplot as plt
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s - %(name)s - %(levelname)s - %(message)s]"
)
logger = logging.getLogger(__name__)
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
    # model option
    def __init__(
        self,
        emu_hub,
        vq_hub,
        vision_hub,
        device,
        fast_path="/data/user/wsong890/user68/project/UniVLA/pretrain/fast",
    ):

        self.emu_hub = emu_hub
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.device = device

        ## hard code here
        self.window_size = 2
        self.predict_action_frames = 10
        self.context_frames = 1
        self.predict_frames = 1
        self.action_dim = 7
        self.use_gripper = True
        self.use_fast = True
        self.use_one_step = False
        # self.eoa_token_id = 151845
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
        self.use_cot = False  # always disable CoT
        self.fast_path = fast_path

        self.video_mode = True
    
        # load model and tokenizer
        self.init_config(device=device)
        self.image_processor.min_pixels = 80 * 80

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
                max_new_tokens=800,
                do_sample=True,
                top_k=2048,
                temperature=0.8,
            )

    def init_config(self, device):
        
        self.model = Emu3MoE.from_pretrained(
            self.emu_hub,
            torch_dtype=torch.bfloat16,
            attn_implementation=None,
            trust_remote_code=True,
        )
        # self.model = Emu3MoE.from_pretrained(
        #     self.emu_hub,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     trust_remote_code=True,
        # )
        self.model.to(device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.vq_hub,#文本tokenizer
            model_max_length=self.model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True).to(device).eval()#图像tokenizer
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)

        # fast tokenization
        # fast_path = "/data/user/wsong890/user68/project/UniVLA/pretrain/fast"
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

        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()

    def preprocess(self, image,img_size=(200,200),wrist_img_size=(200,200)):
        # preprocess image
        agent_view = image['full_image']
        agent_view = Image.fromarray(agent_view)
        # agent_view = agent_view.resize((200, 200))
        agent_view = agent_view.resize(img_size)
        image_x = self.image_processor(agent_view, return_tensors="pt")["pixel_values"].cuda()
        image_code = self.image_tokenizer.encode(image_x)

        gripper_code = None
        if "wrist_image" in image:
            gripper_view = image['wrist_image']
            gripper_view = Image.fromarray(gripper_view)
            # gripper_view = gripper_view.resize((200, 200))
            gripper_view = gripper_view.resize(wrist_img_size)
            gripper_x = self.image_processor(gripper_view, return_tensors="pt")["pixel_values"].cuda()  
            gripper_code = self.image_tokenizer.encode(gripper_x)

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
        # print("self.video_mode",self.video_mode)
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

        if self.use_fast: 
            last_token_id = self.tokenizer.pad_token_id - 1
            allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            # print("final_inputs.input_ids:",final_inputs.input_ids)
            with torch.no_grad():
                outputs = self.model.generate(
                    final_inputs.input_ids.to(self.device),
                    self.GENERATION_CONFIG,
                    max_new_tokens=80,
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
            # print("action:",action)
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

        else:
            pass
        
        # unnormalize action
        action = self.unormalize_action(action)
        # print("action:",action)

        # NOTE(zbzhu): Flip the gripper action here
        # refer to https://github.com/openvla/openvla/blob/1b024f242eda833dc8e321953f25cfd5f3d2f76d/experiments/robot/libero/run_libero_eval.py#L225
        action[..., -1] = np.where(action[..., -1] > 0, 1, -1)

        
        if self.use_one_step:
            # only one step
            action_pred = action[0:1]
        else:
            # action chunk
            action_pred = action
        
        if self.use_cot:
            return action_pred, thought
        else:
            return action_pred
    
    def unormalize_action(self, action):
        action_high = np.array([
            0.93712500009996,
            0.86775000009256,
            0.93712500009996,
            0.13175314309916836,
            0.19275000005139997,
            0.3353504997073735,
            0.9996000000999599
        ])
        action_low = np.array([
            -0.7046250000751599,
            -0.80100000008544,
            -0.9375000001,
            -0.11467779149968735,
            -0.16395000004372,
            -0.2240490058320433,
            -1.0000000001
        ])
        action = 0.5 * (action + 1) * (action_high - action_low) + action_low
        return action


class EmuVLAModel_i_ia_dis(EmuVLAModel):
    def __init__(self,emu_hub,vq_hub,vision_hub,device,window_size=1,fast_path="/data/user/wsong890/user68/project/UniVLA/pretrain/fast",**kwargs):
        super().__init__(emu_hub,vq_hub,vision_hub,device,fast_path)
        self.window_size=window_size
        
        torch.set_printoptions(threshold=2000)
        self.vision_queue = Queue(maxsize=self.window_size)  
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)
        # self.use_mutil_maxnewtokens=use_mutil_maxnewtokens
        
        self.predict_action_frames=kwargs.get("action_chunk",10)
        self.steps=kwargs.get("denoise_steps",72)
        self.inference_time=[]
        self.debug_image=kwargs.get("debug_image",False)
        self.debug_score=kwargs.get("debug_score",False)
        # self.debug_score=True
        # self.debug_image=True
        self.count=0
        self.use_norm_single=kwargs.get("use_norm_single",False)
        self.max_action_len=kwargs.get("max_action_len",70)
        self.max_new_tokens=kwargs.get("max_new_tokens",747+self.max_action_len)
        self.img_size=kwargs.get("img_size",(200,200))
        self.wrist_img_size=kwargs.get("wrist_img_size",(80,80))
        self.action_decay_rate=kwargs.get("action_decay_rate",None)
        self.mask_token_id=151848
        # self.action_decay_rate=0.75
        logger.info(f"self.window_size:{self.window_size}")
        logger.info(f"self.predict_action_frames:{self.predict_action_frames}")
        logger.info(f"self.max_new_tokens:{self.max_new_tokens}")
        logger.info(f"self.steps:{self.steps}")
        logger.info(f"self.action_decay_rate:{self.action_decay_rate}")
        logger.info(f"use video_mode:{self.video_mode}")
        logger.info(f"use_norm_single:{self.use_norm_single}")
        logger.info(f"max_action_len:{self.max_action_len}")
        logger.info(f"img_size:{self.img_size}")
        logger.info(f"wrist_img_size:{self.wrist_img_size}")
        logger.info(f"fast_path:{self.fast_path}")


    def step(self, image, goal):
        """Step function."""
        score_all=[]
        # preprocess observation
        image_code, gripper_code = self.preprocess(image,self.img_size,self.wrist_img_size)
        
        prompt,neg_prompt = goal,""

        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + prompt]
        text_tokens = self.processor.tokenizer(text_prompt)
        
        token_levels = []
        text_token_levels = torch.zeros(len(text_tokens['input_ids'][0]), dtype=torch.long)
        token_levels.append(text_token_levels)
        cur_level = 1
        
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
            
            #
            history = self.get_history()
            action_history = self.get_action_history()

            #
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
                img_input_ids = history[i]['input_ids']#
                # print("img_input_ids.shape:", img_input_ids.shape)
                img_token_type_ids = history[i]['token_type_ids']
                img_attention_mask = history[i]['attention_mask']
 
                img_input_ids=img_input_ids[:,:-1]
                img_token_type_ids=img_token_type_ids[:,:-1]
                img_attention_mask=img_attention_mask[:,:-1]

                image_token_levels = torch.full((img_input_ids.shape[-1],), fill_value=cur_level, dtype=torch.long)
                cur_level += 1
                token_levels.append(image_token_levels)
                
                # 对应的动作
                if i>0 and i < len(action_history)+1 and self.vision_queue.maxsize > 1:
                    # print("len(action_history):",len(action_history))
                    # print("action_history[i-1]:",action_history[i-1])
                    act_input_ids = action_history[i-1]
                    # 
                    act_input_ids = torch.cat([act_input_ids, torch.full((1, self.max_action_len-1 - act_input_ids.shape[1]), self.mask_token_id, dtype=act_input_ids.dtype, device=act_input_ids.device)], dim=1)
                    # 
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
            # 
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
            [9, 15, 16,17,18,19 ,20]

            # action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            mask_schedule = get_mask_chedule("cosine")
            # mask_schedule = get_mask_chedule("linear")
            temperature=1
            seq_len=self.max_new_tokens
            uncond_input_ids = None
            mask_token_id=151848
            top_k=None
            
            input_ids_gen, token_levels_gen = make_mask_with_token_levels(
                history[0]["input_ids"],
                mask_token_id,
                cur_level,
                new_action_len=self.max_action_len,
            )
            input_ids_gen = input_ids_gen.to(self.device)
            
            token_levels.extend(token_levels_gen)
            token_levels = torch.cat(token_levels).unsqueeze(0).to(self.device) # [batch, seq_len]
            
            # print("input_ids.shape:",final_inputs['input_ids'].shape)
            # print("input_ids_gen.shape:",input_ids_gen.shape)
            input_ids=final_inputs['input_ids'].to(self.device)
            # print("input_ids:",input_ids)
            input_ids=torch.cat([input_ids,input_ids_gen],dim=1).to(self.device)
            # print("input_ids.shape:",input_ids.shape)
            # print("input_ids_gen:",input_ids_gen)

            generator = torch.Generator(device=self.device).manual_seed(42)

            noise_schedule = mask_schedule
            
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
                    input_ids, input_ids_gen, temperature, sampled_ids = denoise(
                        self.model, 
                        input_ids, 
                        input_ids_gen,
                        token_levels,
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
                        self.max_action_len,
                        self.action_decay_rate,
                        )
                    # score_all.append(mean_all)
                outputs = sampled_ids#
                # print("outputs:",outputs)
        
                end = time.perf_counter()
                elapsed = end - start
                # print("converge_step:", outputs["converge_step"])

                # outputs = outputs["output_token_ids"]#
                    # print("outputs.shape:", outputs.shape)
                speed = (outputs.shape[-1])/elapsed
                print("speed:", speed)#token/s
                self.inference_time.append(speed)
  
           
            # if self.debug_image:
            #     img_index=torch.where(outputs==torch.tensor(self.img_token_id,dtype=outputs.dtype))
            #     eof_index=torch.where(outputs==torch.tensor(self.eof_token_id,dtype=outputs.dtype))
            #     image_1=outputs[:,img_index[1][0]+1:eof_index[1][0]]-self.bov_token_id
            #     # image_2=outputs[:,img_index[1][1]+1:eof_index[1][1]]-self.bov_token_id
            #     # print("image_1:",image_1)
            #     # print("image_2:",image_2)
            #     image_1=image_1.view(-1, 25,25)
            #     recon = self.image_tokenizer.decode(image_1)
            #     # image_2_text = self.image_tokenizer.decode(image_2)
            #     # recon = recon.view(-1, 25,25)
            #     recon_images = self.image_processor.postprocess(recon)["pixel_values"]
            #     for idx, im in enumerate(recon_images):
            #         im.save(f"./log/image_dis/image_dis_{self.count:03d}.jpg")
            #         self.count+=1
  
            boa_index=torch.where(outputs==torch.tensor(self.boa_token_id,dtype=outputs.dtype))#"boa_token_id": 151844,
            eoa_index=torch.where(outputs==torch.tensor(self.eoa_token_id,dtype=outputs.dtype))#"eoa_token_id": 151845,
            # print("boa_index:", boa_index)
            # print("eoa_index:", eoa_index)
            # print("outputs:", outputs)
            # print("outputs.shape:", outputs.shape)
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

          # unnormalize action
        if self.use_norm_single:
            action = self.unormalize_action_libero10(action)
        else:
            action = self.unormalize_action(action)

        # NOTE(zbzhu): Flip the gripper action here
        # refer to https://github.com/openvla/openvla/blob/1b024f242eda833dc8e321953f25cfd5f3d2f76d/experiments/robot/libero/run_libero_eval.py#L225
        action[..., -1] = np.where(action[..., -1] > 0, 1, -1)
        # print("action:",action)
        
        if self.use_one_step:
            # only one step
            action_pred = action[0:1]
        else:
            # action chunk
            action_pred = action
        # 绘制score_all
        # if self.debug_score:
        #     # score_all=score_all.detach().cpu().numpy()
        #     score_all = torch.stack(score_all).detach().cpu().numpy()
        #     # print("score_all.shape:",score_all.shape)
        #     plt.plot(score_all)
        #     # plt.savefig(f"./log/score/score_all_{self.count:03d}.png")
        #     plt.savefig(f"./log/score/score_all.png")
        #     self.count+=1
        #     plt.close()
        if self.use_cot:
            return action_pred, thought
        else:
            return action_pred
    
    def unormalize_action_libero10(self, action):

        ##libero10
        # logger.info("use libero10 norm")
        action_high = np.array([
            0.7713750600814819,
            0.8463749885559082,
            0.9371249675750732,
            0.13917870819568634,
            0.15953123569488525,
            0.32450997829437256,
            0.9995999336242676
        ])
        action_low = np.array([
            -0.6348749995231628,
            -0.7743749618530273,
            -0.7634999752044678,
            -0.09760086238384247,
            -0.14903677999973297,
            -0.2743905186653137,
            -1.0
        ])
        
        action = 0.5 * (action + 1) * (action_high - action_low) + action_low
        return action
    def unormalize_action(self, action):
        ##libero all
        action_high = np.array([
            0.93712500009996,
            0.86775000009256,
            0.93712500009996,
            0.13175314309916836,
            0.19275000005139997,
            0.3353504997073735,
            0.9996000000999599
        ])
        action_low = np.array([
            -0.7046250000751599,
            -0.80100000008544,
            -0.9375000001,
            -0.11467779149968735,
            -0.16395000004372,
            -0.2240490058320433,
            -1.0000000001
        ])

        
        action = 0.5 * (action + 1) * (action_high - action_low) + action_low
        return action
    
    
    
def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    confidence = log(probs) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    # print("sorted_confidence[,-10:]:",sorted_confidence[:,-10:])
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking

def denoise(
    model, 
    input_ids,
    input_ids_gen,
    token_levels,
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
    max_action_len=70,
    action_decay_rate=None,

    ):

    eoa_index=None
    eoa_index_tuple=torch.where(input_ids_gen==151845)
    if len(eoa_index_tuple[1])>0:
        # input_ids_gen[:,eoa_index_tuple[1][0]+1:]=mask_token_id
        eoa_index=eoa_index_tuple[1][0]
        # print("-seq_len+eoa_index+1:",-seq_len+eoa_index+1)
        input_ids[:,-seq_len+eoa_index+1:]=mask_token_id
        

    logits = forward(model, input_ids, attention_mask=attention_mask, token_levels=token_levels)
    logits = logits[:, -(seq_len + 1):-1, :]
    # print("logits.shape:", logits.shape)
    if allowed_action_token_ids is not None:
        mask = torch.full_like(logits[:,-max_action_len:,:], float('-inf'))
        allowed_action_token_ids_tensor = torch.tensor(allowed_action_token_ids, device=logits.device)  # (vocab_size, )
        mask[:,:,allowed_action_token_ids_tensor]=0
        logits[:,-max_action_len:,:] += mask
    if allowed_image_token_ids is not None:
        mask = torch.full_like(logits[:,:-max_action_len,:], float('-inf'))
        allowed_image_token_ids_tensor = torch.tensor(allowed_image_token_ids, device=logits.device)  # (vocab_size, )
        mask[:,:,allowed_image_token_ids_tensor]=0
        logits[:,:-max_action_len,:] += mask
    probs = logits.softmax(dim=-1)
    sampled = probs.reshape(-1, logits.size(-1))#(b*seq,vocabsize)
    top_k=None
    if top_k is not None:
        topk_probs, topk_indices = torch.topk(sampled, top_k, dim=-1)#[B*T, k]
        topk_probs /= topk_probs.sum(dim=-1, keepdim=True)
        sampled_ids = torch.multinomial(topk_probs, 1, generator=generator)[:, 0]#[B*seq_len] 内部下标
        sampled_ids = topk_indices.gather(-1, sampled_ids.view(-1, 1)).view(*logits.shape[:-1])#[B, T]

    else:
        sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])#挑选每个位置概率最大的下标

    unknown_map = input_ids_gen == mask_token_id
    # if action_decay_rate and ratio<action_decay_rate:
        # print(f"ratio:{ratio} action_decay_rate:{action_decay_rate}")
        # unknown_map[:,-max_action_len:]=False

    # print("eoa_index:",eoa_index)
    # if eoa_index is not None:
    #     # print("unknown_map.shape:",unknown_map.shape)
    #     # print("before unknown_map:",unknown_map)
    #     unknown_map[:,eoa_index+1:]=False
    #     # print("after unknown_map:",unknown_map)
    # print("eoa_index:",eoa_index)
    # print(f"ration:{ratio*72} unknown_map:{unknown_map[:,-70:]}")
    sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_gen)

    mask_ratio = noise_schedule(torch.tensor(ratio))
    selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])#[B, T, 1]#提取预测下标对应的概率
    selected_probs = selected_probs.squeeze(-1)#[B, T]#去掉最后一维
    # score = selected_probs[unknown_map].sort().values[-10:]
    # print("score:",score)
    # num = unknown_map.sum()
    # print("num:",num)
    # mean_all = (selected_probs * unknown_map).sum() / num.clamp_min(1)
    # print("mean_all:",mean_all)

    selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
    unknown_map = input_ids_gen == mask_token_id
    if action_decay_rate and ratio<action_decay_rate:
        # print(f"ratio:{ratio} action_decay_rate:{action_decay_rate}")
        selected_probs[:,-max_action_len+1:]=-100#抑制动作生成

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
    token_levels=None,
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
    dict=model(input_ids=input_ids, attention_mask=attention_mask, token_levels=token_levels)
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

def make_mask_with_token_levels(
    input_ids, mask_token_id, cur_level,
    boi_id=151852, eoi_id=151853, boa_id=151844, new_action_len=70,
):
    '''
    mask the input image(the same shape) + action(boa + [mask_token] * (new_action_len-1))
    '''
    input_ids = input_ids[..., :-1] # the last token of input_ids is boa
    is_boi = input_ids == boi_id
    is_eoi = input_ids == eoi_id
    
    starts_cum = torch.cumsum(is_boi, dim=1)
    ends_cum = torch.cumsum(is_eoi, dim=1)
    
    is_mask = (starts_cum - ends_cum > 0) & (~is_boi) & (~is_eoi)
    
    mask_inputs = torch.full_like(input_ids, mask_token_id)
    mask_inputs = torch.where(is_mask, mask_inputs, input_ids)
    
    mask_action = torch.full((1, new_action_len - 1), mask_token_id) # the first token is boa
    
    token_levels = []

    image_token_levels = torch.full((mask_inputs.shape[-1], ), cur_level, dtype=torch.long)
    token_levels.append(image_token_levels)
    cur_level += 1
    action_token_levels = torch.full((new_action_len, ), cur_level, dtype=torch.long)
    token_levels.append(action_token_levels)
    mask_inputs = torch.cat([mask_inputs, torch.tensor([[boa_id]], dtype=torch.long), mask_action], dim=-1)
        
    return mask_inputs, token_levels

class EmuVLAModel_i_ia_dis_2stage(EmuVLAModel):
    def __init__(self,emu_hub,vq_hub,vision_hub,device,window_size=1,fast_path="/data/user/wsong890/user68/project/UniVLA/pretrain/fast",**kwargs):
        super().__init__(emu_hub,vq_hub,vision_hub,device,fast_path)
        self.window_size=window_size
        
        torch.set_printoptions(threshold=2000)
        self.vision_queue = Queue(maxsize=self.window_size)  
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)
        # self.use_mutil_maxnewtokens=use_mutil_maxnewtokens
        self.max_new_tokens=kwargs.get("max_new_tokens",747+70)
        self.predict_action_frames=kwargs.get("action_chunk",10)
        self.steps=kwargs.get("denoise_steps",72)
        self.inference_time=[]
        self.debug_image=kwargs.get("debug_image",False)
        self.debug_score=kwargs.get("debug_score",False)
        # self.debug_score=True
        # self.debug_image=True
        self.count=0
        self.use_norm_single=kwargs.get("use_norm_single",False)
        self.max_action_len=kwargs.get("max_action_len",70)
        self.img_size=kwargs.get("img_size",(200,200))
        self.wrist_img_size=kwargs.get("wrist_img_size",(80,80))
        self.action_steps=kwargs.get("action_steps",24)
        self.image_steps=kwargs.get("image_steps",64)
        logger.info(f"self.window_size:{self.window_size}")
        logger.info(f"self.predict_action_frames:{self.predict_action_frames}")
        logger.info(f"self.max_new_tokens:{self.max_new_tokens}")
        logger.info(f"self.steps:{self.steps}")
        logger.info(f"use video_mode:{self.video_mode}")
        logger.info(f"use_norm_single:{self.use_norm_single}")
        logger.info(f"max_action_len:{self.max_action_len}")
        logger.info(f"img_size:{self.img_size}")
        logger.info(f"wrist_img_size:{self.wrist_img_size}")
        logger.info(f"fast_path:{self.fast_path}")
        logger.info(f"action_steps:{self.action_steps}")
        logger.info(f"image_steps:{self.image_steps}")
    def step(self, image, goal):
        """Step function."""
        score_all=[]
        # preprocess observation
        image_code, gripper_code = self.preprocess(image,self.img_size,self.wrist_img_size)
        
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
            allowed_image_token_ids = list(range(151854 , 184622))+\
            [self.img_token_id,self.eof_token_id]+\
            [9, 15, 16,17, 20]

            # action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            mask_schedule = get_mask_chedule("cosine")
            # mask_schedule = get_mask_chedule("linear")
            temperature=1
            # seq_len=self.max_new_tokens
            uncond_input_ids = None
            mask_token_id=151848
            top_k=None
            input_ids=final_inputs['input_ids'].to(self.device)



            generator = torch.Generator(device=self.device).manual_seed(42)

            noise_schedule = mask_schedule
            
            with torch.no_grad():
                input_ids_gen = make_mask_image(history[0]['input_ids'], mask_token_id,mask_begin=151852,mask_end=151853).to(self.device)
                input_ids=torch.cat([input_ids,input_ids_gen],dim=1).to(self.device)
                start = time.perf_counter()
                iamge_steps=self.image_steps
                seq_len=747
                # eoa_index=None
                for step in range(iamge_steps):
                    ratio = 1.0 * (step + 1) / iamge_steps
                    input_ids, input_ids_gen, temperature, sampled_ids = denoise_image(
                        self.model, 
                        input_ids, 
                        input_ids_gen,
                        None, 
                        generator, 
                        ratio, 
                        mask_token_id, 
                        noise_schedule, 
                        seq_len,
                        temperature,
                        allowed_action_token_ids,
                        allowed_image_token_ids,    
                        )
                input_ids[:,-seq_len:] = sampled_ids#
                # print("sampled_ids:",sampled_ids)
                input_ids_gen=make_mask_action(mask_token_id,new_action_len=70).to(self.device)
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
                        None, 
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
  
  
            boa_index=torch.where(outputs==torch.tensor(self.boa_token_id,dtype=outputs.dtype))#"boa_token_id": 151844,
            eoa_index=torch.where(outputs==torch.tensor(self.eoa_token_id,dtype=outputs.dtype))#"eoa_token_id": 151845,

            if len(eoa_index[1])>0:
                # print("boa_index:",boa_index)
                orig_outputs=outputs.clone()
                outputs=outputs[:,boa_index[1][0]+1:eoa_index[1][0]]
                
                orig_outputs=orig_outputs[:,boa_index[1][0]+1:eoa_index[1][0]+1]#保留 eoa 去掉boa，因为图像token结束自带一个boa
            else:
                print("find no eoa")
                outputs=outputs[:,boa_index[1][0]+1:]
     
                orig_outputs = torch.cat([outputs, torch.tensor(self.eoa_token_id,dtype=outputs.dtype,device=outputs.device).view(1,1)], dim=1)
   
            last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
            processed_outputs = last_token_id_tensor - outputs
            action_outputs = self.action_tokenizer.decode(
                processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
            )#[1,10,7]
            action = action_outputs[0]
  
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

          # unnormalize action
        if self.use_norm_single:
            action = self.unormalize_action_libero10(action)
        else:
            action = self.unormalize_action(action)

        # NOTE(zbzhu): Flip the gripper action here
        # refer to https://github.com/openvla/openvla/blob/1b024f242eda833dc8e321953f25cfd5f3d2f76d/experiments/robot/libero/run_libero_eval.py#L225
        action[..., -1] = np.where(action[..., -1] > 0, 1, -1)
        # print("action:",action)
        
        if self.use_one_step:
            # only one step
            action_pred = action[0:1]
        else:
            # action chunk
            action_pred = action
 
        if self.use_cot:
            return action_pred, thought
        else:
            return action_pred
    
    def unormalize_action_libero10(self, action):

        ##libero10
        # logger.info("use libero10 norm")
        action_high = np.array([
            0.7713750600814819,
            0.8463749885559082,
            0.9371249675750732,
            0.13917870819568634,
            0.15953123569488525,
            0.32450997829437256,
            0.9995999336242676
        ])
        action_low = np.array([
            -0.6348749995231628,
            -0.7743749618530273,
            -0.7634999752044678,
            -0.09760086238384247,
            -0.14903677999973297,
            -0.2743905186653137,
            -1.0
        ])
        
        action = 0.5 * (action + 1) * (action_high - action_low) + action_low
        return action
    def unormalize_action(self, action):
        ##libero all
        action_high = np.array([
            0.93712500009996,
            0.86775000009256,
            0.93712500009996,
            0.13175314309916836,
            0.19275000005139997,
            0.3353504997073735,
            0.9996000000999599
        ])
        action_low = np.array([
            -0.7046250000751599,
            -0.80100000008544,
            -0.9375000001,
            -0.11467779149968735,
            -0.16395000004372,
            -0.2240490058320433,
            -1.0000000001
        ])

        
        action = 0.5 * (action + 1) * (action_high - action_low) + action_low
        return action