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
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            self.model.to(device).eval()

            self.tokenizer = Emu3Tokenizer.from_pretrained(
                self.vq_hub,#从模型加载tokenizer
                model_max_length=self.model.config.max_position_embeddings,
                padding_side="right",
                use_fast=False,
            )
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)#从vqvae加载image_processor
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



class CustomModel:
    # model option
    def __init__(
        self,
        ckpt_path,
        configs,
        device,
        save_dir=None,
        raw_calvin=True,
        debug=False,
        action_ensemble=False,
    ):
        self.model = BaseTrainer(configs=configs)
        self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug)
        # self.model.model.lm_head.window_size = 1

    def init_config(
        self, ckpt_path, configs, device, save_dir=None, raw_calvin=False, debug=False
    ):
        ### load and convert checkpoint
        self.debug = debug
        if configs["model"] == "kosmos":
            import transformers

            import pdb 
            pdb.set_trace()
            package_dir = transformers.__path__[0]
            os.system(
                "cp tools/modeling_kosmos2.py {}/models/kosmos2/modeling_kosmos2.py".format(
                    package_dir
                )
            )

        if not self.debug:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in ckpt:
                new_state_dict = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                new_state_dict = ckpt["model_state_dict"]
            else:
                raise KeyError("no checkpoint dict in the loaded pretrain parameters")

            new_state_dict = self.convert_old_state_dict(new_state_dict)
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"CKPT Loaded \n {msg}")

            ckpt_dir = os.path.dirname(ckpt_path)
            ckpt_name = os.path.basename(ckpt_path)
            save_dir = ckpt_dir if save_dir is None else save_dir
            load_info_path = os.path.join(save_dir, f"{ckpt_name}_loading_msg.json")
            if os.path.exists(load_info_path):
                os.system(f"rm {load_info_path}")
            with open(load_info_path, "w") as f:
                _info = {
                    "missing_keys": msg.missing_keys,
                    "unexpected_keys": msg.unexpected_keys,
                }
                json.dump(_info, f, indent=2)
                print(f"Model loading msg is updated to: {load_info_path}")

        self.configs = configs

        dtype = torch.float32
        if self.configs["trainer"]["precision"] == "bf16":
            dtype = torch.bfloat16
        elif self.configs["trainer"]["precision"] == "fp16":
            dtype = torch.float16
        self.dtype = dtype
        self.act_head_configs = self.configs["act_head"]
        self.raw_calvin = raw_calvin
        self.tcp_rel = self.configs.get("tcp_rel", False)

        print(f"raw action: {self.raw_calvin}")

        self.device = device
        self.policy = self.model
        self.policy = self.policy.to(self.dtype)
        # self.policy = self.policy.float()
        self.policy.to(self.device)
        self.policy.eval()

        if not hasattr(self.policy.model, "lm_head"):
            self.policy.model.lm_head = self.policy.model.act_head

        self.tokenizer = build_tokenizer(self.configs["tokenizer"])

        self.window_size = configs["window_size"]
        self.fwd_pred_next_n = configs["fwd_pred_next_n"]
        self.act_step = self.fwd_pred_next_n + 1
        self.seq_len = self.configs["seq_len"]
        self.use_hand_rgb = self.configs["use_hand_rgb"]

        if hasattr(self, "policy_setup"):
            data_mix = "bridge" if self.policy_setup == "widowx_bridge" else "rt_1"
            configs["train_dataset"]["data_mix"] = data_mix
            configs["val_dataset"]["data_mix"] = data_mix
        if configs["model"] == "kosmos":
            image_preprocess = self.model.model.image_processor
        else:
            image_preprocess = self.model.model.processor
        self.image_preprocess = functools.partial(
            preprocess_image,
            image_processor=image_preprocess,
            model_type=configs["model"],
        )

        self.text_preprocess = get_text_function(
            self.model.model.tokenizer, configs["model"]
        )

        self.action_space = self.configs["act_head"].get("action_space", "continuous")
        if self.action_space == "discrete":
            self.action_tokenizer = ActionTokenizer(
                self.tokenizer,
                bins=self.act_head_configs["n_bin"],
                min_action=self.act_head_configs["min_action"],
                max_action=self.act_head_configs["max_action"],
            )

        print(f"Evaluating checkpoint {ckpt_path}")

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0

        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

    def ensemble_action(self, action):
        if action.ndim >= 3:
            action = action.squeeze()

        if action.ndim == 1:
            action = action.unsqueeze(0)

        self.action_hist_list.append(action)

        act_cache = []
        # self.fwd_pred_next_n = 1
        max_len = self.fwd_pred_next_n
        # max_len = 1
        # if self.tcp_rel:
        #     max_len = 1
        while len(self.action_hist_list) > max_len:
            self.action_hist_list.pop(0)

        idx = 0
        for act in self.action_hist_list[::-1]:
            # print(act.shape)
            act_cache.append(act[idx])
            idx += 1

        act_cache = torch.stack(act_cache, dim=0)

        weights = torch.tensor([fwd_decay_ratio**i for i in range(len(act_cache))])
        weights = weights / weights.sum()

        weighted_act = (act_cache * weights.unsqueeze(1)).sum(dim=0)

        return weighted_act

    @staticmethod
    def convert_old_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_k = k.replace("module.", "")
            else:
                new_k = k

            if not new_k.startswith("model."):
                new_k = "model." + new_k

            new_state_dict[new_k] = state_dict[k]
        return new_state_dict

    def _get_default_calvin_config(self):
        return {
            "type": "DiskCalvinDataset",
            "data_dir": "CALVIN/task_ABCD_D/val",
            "c_act_scaler": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }

    def add_element_to_queue(self, q: Queue, element):
        while q.qsize() >= q.maxsize:
            q.get()
        q.put(element)

    def get_history(self, q: Queue, pad: Literal["zero", "first"] = "zero"):
        queue_list = list(q.queue)
        if len(queue_list) == 0:
            return queue_list, None
        history_type = self.configs["act_head"].get("history_type", "pre")
        if history_type == "pre":
            pad_len = 0
        else:
            raise ValueError(f"Unsupported history type {history_type}")
        element = queue_list[0]
        if pad == "zero":
            if isinstance(element, torch.Tensor):
                element = torch.zeros_like(element)
            elif isinstance(element, np.ndarray):
                element = np.zeros_like(element)
            else:
                raise ValueError("This type is not supported")
            queue_list = [element for _ in range(pad_len)] + queue_list
        else:
            if isinstance(element, torch.Tensor):
                pad_list = [element.clone() for _ in range(pad_len)]
            elif isinstance(element, np.ndarray):
                pad_list = [deepcopy(element) for _ in range(pad_len)]
            queue_list = pad_list + queue_list
        pad_mask = np.ones(q.maxsize, dtype=bool)
        pad_mask[:pad_len] = False
        return queue_list, pad_mask

    def preprocess(self, obs, lang, mode="continuous"):
        # preprocess image
        image = obs["rgb_obs"]["rgb_static"]
        image = Image.fromarray(image)
        image_x = self.image_preprocess([image]).unsqueeze(0)

        gripper_x = None
        if "rgb_gripper" in obs["rgb_obs"]:
            gripper = obs["rgb_obs"]["rgb_gripper"]
            gripper = Image.fromarray(gripper)
            gripper_x = self.image_preprocess([gripper]).unsqueeze(0)
            gripper_x = gripper_x.to(self.device).to(self.dtype)

        if self.configs["act_head"].get("history_type", "post") == "pre":
            self.add_element_to_queue(self.vision_queue, image_x)
            image_x, _ = self.get_history(self.vision_queue, pad="first")
            image_x = torch.concatenate(image_x, dim=1)

            if gripper_x is not None:
                self.add_element_to_queue(self.vision_gripper_queue, gripper_x)
                gripper_x, _ = self.get_history(self.vision_gripper_queue, pad="first")
                gripper_x = (
                    torch.concatenate(gripper_x, dim=1).to(self.device).to(self.dtype)
                )

        if mode == "discrete":
            if "llava" in self.policy.configs:
                model_name = self.policy.configs["llava"]
            elif "qwen" in self.policy.configs:
                model_name = "qwen"
            else:
                # model_name = self.policy.configs['llm']['pretrained_model_name_or_path']
                model_name = self.policy.configs["model"]

            prompt_builder = get_prompt_builder(
                model_name, bos=self.tokenizer.bos_token, eos=self.tokenizer.eos_token
            )

            conversation = [
                {
                    "from": "human",
                    "value": (
                        f"What action should the robot take to {lang}?"
                        if self.act_step == 1
                        else f"What {self.act_step} step actions should the robot take to {lang}?"
                    ),
                },
                {"from": "gpt", "value": ""},
            ]

            input_ids = []
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = torch.tensor(
                list(
                    self.tokenizer(
                        prompt_builder.get_prompt(), add_special_tokens=True
                    ).input_ids
                )
            )
            if self.tokenizer.eos_token is not None:
                input_ids = input_ids[:-1]

            text_x = input_ids.unsqueeze(0)
            mask = torch.full((1, text_x.shape[-1]), True, dtype=torch.bool)
        else:
            text_x, mask = self.text_preprocess([lang])

        return (
            image_x.to(self.device).to(self.dtype),
            gripper_x,
            text_x.to(self.device),
            mask.to(self.device),
        )

    def step(self, obs, goal):
        """Step function."""
        input_dict = dict()
        image_x, gripper_x, text_x, mask = self.preprocess(obs, goal, self.action_space)

        input_dict["rgb"] = image_x
        input_dict["hand_rgb"] = gripper_x
        input_dict["text"] = text_x
        input_dict["text_mask"] = mask

        if self.action_space == "discrete":
            input_dict["instr_and_action_ids"] = text_x
            input_dict["instr_and_action_mask"] = mask

        with torch.no_grad():
            action = self.policy.inference_step(input_dict)["action"]
        if self.action_space != "discrete":
            # print(action)
            if action[0].ndim == action[1].ndim + 1:
                action = (action[0], action[1].unsqueeze(2))
            action = torch.cat(
                [action[0], (torch.nn.functional.sigmoid(action[1]) > 0.5).float()],
                dim=-1,
            )

        # action = action[0, 0, 0] # batch, seq_len, chunck_idx
        if isinstance(action, tuple):
            action = torch.cat([action[0], action[1]], dim=-1)

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        if action.ndim == 2:
            action = action.unsqueeze(1)

        if action.ndim == 3:
            action = action.unsqueeze(1)

        action = action.detach().cpu()

        if self.tcp_rel:
            robot_obs = (
                torch.from_numpy(obs["robot_obs"])
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, 1, self.fwd_pred_next_n, 1)
            )
            action = tcp_to_world_frame(action, robot_obs)

        # action = self.ensemble_action(action)
        if action.ndim == 4:
            action = action.squeeze().squeeze()

        # if isinstance(action, torch.Tensor):
        #     action = action.squeeze()
        #     if action.ndim == 2:
        #         action = action[0]
            # action = action.numpy()
        if self.configs.get("use_mu_law", False):
            from robovlms.data.data_utils import inverse_mu_law_companding

            action = inverse_mu_law_companding(
                action, self.configs.get("mu_val", 255), maintain_last=True
            )
        if self.configs.get("norm_action", False):
            from robovlms.data.data_utils import unnoramalize_action

            if isinstance(action, tuple):
                action = (
                    unnoramalize_action(
                        action[0], self.configs["norm_min"], self.configs["norm_max"]
                    ),
                    action[1],
                )
            else:
                action = unnoramalize_action(
                    action, self.configs["norm_min"], self.configs["norm_max"]
                )

        if self.action_space == "discrete":
            # action[-1] = 1 if action[-1] > 0 else -1
            pass
        else:
            if self.raw_calvin:
                if action.ndim == 2:
                    action[:,-1] = (action[:,-1] - 0.5) * 2
                else:
                    action[-1] = (action[-1] - 0.5) * 2
            else:
                state = obs["robot_obs"]  # (15,)
                xyz_state = state[:3]
                rpy_state = state[3:6]
                rotm_state = euler2rotm(rpy_state)
                rel_action = action.numpy()
                _c_rel_action = rel_action[:6]
                xyz_action = _c_rel_action[:3]
                rpy_action = _c_rel_action[3:6]
                gripper_action = rel_action[6]
                rotm_action = euler2rotm(rpy_action)
                xyz_next_state = xyz_state + rotm_state @ xyz_action
                rotm_next_state = rotm_state @ rotm_action
                rpy_next_state = rotm2euler(rotm_next_state)

                action = action.numpy()
                action[:3] = xyz_next_state - xyz_state
                action[3:6] = rpy_next_state - rpy_state
                action[:6] *= [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
                action[-1] = (gripper_action - 0.5) * 2
                action = torch.from_numpy(action)

        self.rollout_step_counter += 1
        if action.ndim >= 2:
            action[..., -1] = torch.where(action[..., -1] > 0, 1, -1)
        else:
            action[-1] = 1 if action[-1] > 0 else -1
        # print(f"step {self.rollout_step_counter} action {action}")
        return np.array(action)

    def reset(self):
        if hasattr(self.model.model, "lm_head"):
            self.model.model.lm_head.hidden_state = None
            self.model.model.lm_head.history_memory = []
        if hasattr(self.model.model, "act_head"):
            self.model.model.act_head.hidden_state = None
            self.model.model.act_head.history_memory = []

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

class EmuVLAModel_i_ia(EmuVLAModel):
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
        self.debug_image=kwargs.get("debug_image",False)
        self.debug_image=True
        self.count=0
        print("self.window_size:",self.window_size)
        print("self.predict_action_frames:",self.predict_action_frames)
        print("self.max_new_tokens:",self.max_new_tokens)
        print("self.use_mutil_maxnewtokens:",self.use_mutil_maxnewtokens)
        print("self.use_jacobi_generate:",self.use_jacobi_generate)


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
            print("final_inputs['input_ids']:", final_inputs['input_ids'])
            final_inputs['token_type_ids'] = concatenated_token_type_ids
            final_inputs['attention_mask'] = concatenated_attention_mask
        else:
            final_inputs = pos_inputs

        if self.use_fast: 
            last_token_id = self.tokenizer.pad_token_id - 1#150618->151642
            allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            allowed_action_token_ids=list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            allowed_image_token_ids = list(range(self.tokenizer.pad_token_id+1 , self.tokenizer.pad_token_id+1 + 32768))+\
            [self.boi_token_id,self.eoi_token_id,self.img_token_id,self.eof_token_id]+\
            [16, 9, 15,17, 20]

            action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            
            with torch.no_grad():
                if self.vllm_accelerator:
                    from vllm import LLM, SamplingParams, TokensPrompt
                    self.GENERATION_CONFIG = SamplingParams(
                        temperature=0.0,            # Set to 0 for deterministic behavior (do_sample=False equivalent)
                        top_p=1.0,                  # No probability filtering
                        top_k=-1,                   # No token count limiting
                        max_tokens=50+747,             # Maximum tokens to generate
                        stop_token_ids=[self.eoa_token_id],  # End generation at this token
                        skip_special_tokens=False,  # Keep special tokens
                        logits_processors=[action_id_processor],
                    )
                    input_ids_list = final_inputs.input_ids.detach().cpu().tolist()[0]
                    inputs = TokensPrompt(prompt_token_ids=input_ids_list)
                    outputs = self.model.generate(inputs,self.GENERATION_CONFIG)
                    token_ids_tuple = outputs[0].outputs[0].token_ids
                    token_ids_list = list(token_ids_tuple)
                    outputs = torch.tensor([token_ids_list], dtype=torch.long)
                elif self.use_jacobi_generate:
                    # print("not use vllm")
                    # print("final_inputs.input_ids:", final_inputs.input_ids)
                    # print("final_inputs.attention_mask:", final_inputs.attention_mask)
                    #记录时间 
                    start = time.perf_counter()
                    # print("self.use_mutil_maxnewtokens:",self.use_mutil_maxnewtokens)
                    if self.use_mutil_maxnewtokens:
                        max_new_tokens=[747,64]
                        max_iter=[747,64]
                        print("use_mutil_maxnewtokens")
                        outputs = self.model.generate_jacobi_kv_mutil_maxnewtokens(
                            final_inputs.input_ids.to(self.device),
                            max_new_tokens=max_new_tokens,
                            max_iter=max_new_tokens,
                            attention_mask=final_inputs.attention_mask.to(self.device),
                            eos_token=self.eoa_token_id,
                        )
                    else:
                        
                        # max_new_tokens=self.max_new_tokens
                        # max_iter=self.max_new_tokens
                        # # print("final_inputs.input_ids",final_inputs.input_ids)
                        # # print("max_new_tokens:",max_new_tokens)
                        # # print("max_iter:",max_iter)
                        # outputs = self.model.generate_jacobi_kv(
                        #     final_inputs.input_ids.to(self.device),
                        #     max_new_tokens=max_new_tokens,
                        #     max_iter=max_iter,
                        #     attention_mask=final_inputs.attention_mask.to(self.device),
                        #     eos_token=self.eoa_token_id,
                        # )
                        max_new_tokens=self.max_new_tokens
                        max_iter=self.max_new_tokens
                        outputs = self.model.generate_jacobi_kv(
                            final_inputs.input_ids.to(self.device),
                            max_new_tokens=max_new_tokens,
                            max_iter=max_iter,
                            max_new_seq_len=747,
                            attention_mask=final_inputs.attention_mask.to(self.device),
                            eos_token=self.eoa_token_id,
                            allowed_token_ids=allowed_image_token_ids,
                        )
                        past_kv=outputs["past_key_values"]
                        final_tokens=outputs["final_tokens"]
                        first_correct_token=torch.tensor(self.boa_token_id,dtype=final_inputs.input_ids.dtype).view(1,1)
                        # print("output_token_ids:",outputs["output_token_ids"])
                        outputs = self.model.generate_jacobi_kv(
                            input_ids=final_tokens.to(self.device),
                            max_new_tokens=max_new_tokens,
                            max_iter=max_iter,
                            max_new_seq_len=70,
                            attention_mask=final_inputs.attention_mask.to(self.device),
                            eos_token=self.eoa_token_id,
                            past_kv=past_kv,
                            first_correct_token=first_correct_token.to(self.device),
                            allowed_token_ids=allowed_action_token_ids,
                        )
                        print("output_token_ids:",outputs["output_token_ids"])
                    end = time.perf_counter()
                    elapsed = end - start
                    # print("converge_step:", outputs["converge_step"])

                    outputs = outputs["output_token_ids"]#
                    # print("outputs.shape:", outputs.shape)
                    speed = (outputs.shape[-1]+747)/elapsed
                    print("speed:", speed)#token/s
                    self.inference_time.append(speed)
                else:
                    start = time.perf_counter()
                    outputs = self.model.generate(
                        final_inputs.input_ids.to(self.device),
                        self.GENERATION_CONFIG,
                        max_new_tokens=70+747,
                        # logits_processor=[action_id_processor],
                        attention_mask=final_inputs.attention_mask.to(self.device),
                    )
                    end = time.perf_counter()
                    elapsed = end - start
                    speed = (outputs.shape[-1]-final_inputs.input_ids.shape[-1])/elapsed
                    self.inference_time.append(speed)
                    print("speed:", speed)
            # omit the eoa token
            if self.vllm_accelerator:
                orig_outputs = outputs
                outputs = outputs[:, :-1]
            elif self.use_jacobi_generate:
                eoa_token_index=torch.where(outputs==self.eoa_token_id)[1]
                if len(eoa_token_index) > 0:
                    orig_outputs = outputs[:, :eoa_token_index[0]+1].clone()
                    outputs = outputs[:, :eoa_token_index[0]+1]
                    # print("outputs:",outputs)
                else:
                    print("find no eoa")
                    orig_outputs = outputs.clone()
                    outputs = outputs
            else:
                # print("outputs:", outputs)
                orig_outputs = outputs[:, final_inputs.input_ids.shape[-1]:].clone()
                # print("orig_outputs:", orig_outputs)
                # outputs = outputs[:, final_inputs.input_ids.shape[-1]:-1]
                outputs = outputs[:, final_inputs.input_ids.shape[-1]:]
            if self.debug_image:
                img_index=torch.where(outputs==torch.tensor(self.img_token_id,dtype=outputs.dtype))
                eof_index=torch.where(outputs==torch.tensor(self.eof_token_id,dtype=outputs.dtype))
                print("img_index:",img_index)
                print("eof_index:",eof_index)
                image_1=outputs[:,img_index[1][0]+1:eof_index[1][0]]-self.bov_token_id
                # image_2=outputs[:,img_index[1][1]+1:eof_index[1][1]]-self.bov_token_id
                print("image_1:",image_1)
                # print("image_2:",image_2)
                image_1=image_1.view(-1, 25,25)
                recon = self.image_tokenizer.decode(image_1)
                # image_2_text = self.image_tokenizer.decode(image_2)
                # recon = recon.view(-1, 25,25)
                recon_images = self.image_processor.postprocess(recon)["pixel_values"]
                for idx, im in enumerate(recon_images):
                    im.save(f"./log/image/image_jacobi_{self.count:03d}.jpg")
                    self.count+=1
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

class EmuVLAModel_i_aia(EmuVLAModel):
    def __init__(self,emu_hub,vq_hub,vision_hub,device,raw_calvin=True,use_jacobi_generate=False):
        super().__init__(emu_hub,vq_hub,vision_hub,device,raw_calvin,use_jacobi_generate)
        self.window_size=1
        print("self.window_size:",self.window_size)
        # print("self.window_size:",self.window_size)
        torch.set_printoptions(threshold=2000)
        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)
        self.GENERATION_CONFIG = GenerationConfig(
                pad_token_id=self.model.config.pad_token_id,
                bos_token_id=self.model.config.bos_token_id,
                eos_token_id=self.model.config.eos_token_id,
                do_sample=False,
            )

    def step(self, obs, goal):
        """Step function."""
        # try:
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
                img_input_ids = history[i]['input_ids'][:,:-1]#去掉boa
                img_token_type_ids = history[i]['token_type_ids'][:,:-1]
                img_attention_mask = history[i]['attention_mask'][:,:-1]
                
                # 对应的动作 i-aia
                # if i < len(action_history):
                #     act_input_ids = action_history[i]
                    
                #     # 动作的token_type_ids和attention_mask分别填充为全0和全1
                #     act_token_type_ids = torch.zeros_like(act_input_ids)
                #     act_attention_mask = torch.ones_like(act_input_ids)
                    
                #     # 交替添加图像和动作数据
                #     all_input_ids.extend([img_input_ids, act_input_ids])
                #     all_token_type_ids.extend([img_token_type_ids, act_token_type_ids])
                #     all_attention_mask.extend([img_attention_mask, act_attention_mask])
                # else:
                #     # 若没有对应的动作，添加图像数据
                #     all_input_ids.append(img_input_ids)
                #     all_token_type_ids.append(img_token_type_ids)
                #     all_attention_mask.append(img_attention_mask)
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
            # print("final_inputs['input_ids']:", final_inputs['input_ids'])
            final_inputs['token_type_ids'] = concatenated_token_type_ids
            final_inputs['attention_mask'] = concatenated_attention_mask
        else:
            final_inputs = pos_inputs

        if self.use_fast: 
            last_token_id = self.tokenizer.pad_token_id - 1
            # allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            # action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            
            with torch.no_grad():
                if self.vllm_accelerator:
                    print("use vllm")
                    from vllm import LLM, SamplingParams, TokensPrompt
                    self.GENERATION_CONFIG = SamplingParams(
                        temperature=0.0,            # Set to 0 for deterministic behavior (do_sample=False equivalent)
                        top_p=1.0,                  # No probability filtering
                        top_k=-1,                   # No token count limiting
                        max_tokens=50+725,             # Maximum tokens to generate
                        stop_token_ids=[self.eoa_token_id],  # End generation at this token
                        skip_special_tokens=False,  # Keep special tokens
                        # logits_processors=[action_id_processor],
                    )
                    input_ids_list = final_inputs.input_ids.detach().cpu().tolist()[0]
                    inputs = TokensPrompt(prompt_token_ids=input_ids_list)
                    outputs = self.model.generate(inputs,self.GENERATION_CONFIG)
                    token_ids_tuple = outputs[0].outputs[0].token_ids
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
                        max_new_tokens=70+747+70,
                        max_iter=70+747,
                        attention_mask=final_inputs.attention_mask.to(self.device),
                        eos_token=184622,
                    )
                    end = time.perf_counter()
                    elapsed = end - start
                    # print("converge_step:", outputs["converge_step"])

                    outputs = outputs["output_token_ids"]#
                    speed = (outputs.shape[-1])/elapsed
                    self.inference_time.append(speed)
                else:
                    start = time.perf_counter()
                    outputs = self.model.generate(
                        final_inputs.input_ids.to(self.device),
                        self.GENERATION_CONFIG,
                        max_new_tokens=60+747+60,
                        # logits_processor=[action_id_processor],
                        attention_mask=final_inputs.attention_mask.to(self.device),
                    )
                    end = time.perf_counter()
                    elapsed = end - start
                    speed = (outputs.shape[-1]-final_inputs.input_ids.shape[-1])/elapsed
                    self.inference_time.append(speed)
                    # print("speed:", speed)
            # omit the eoa token
            if self.vllm_accelerator:
                orig_outputs = outputs
                outputs = outputs[:, :-1]
            else:
                orig_outputs = outputs[:, final_inputs.input_ids.shape[-1]:].clone()
                outputs = outputs[:, final_inputs.input_ids.shape[-1]:]
            # print("outputs:",outputs)
            boa_index=torch.where(outputs==151844)[1][:2]#"boa_token_id": 151844,
            eoa_index=torch.where(outputs==151845)[1][:2]#"eoa_token_id": 151847,
            # print("boa_index:", boa_index)
            # print("eoa_index:", eoa_index)
            outputs_tmp=[]
            for boa_ind,eoa_ind in zip(boa_index,eoa_index):
                outputs_tmp.append(outputs[:,boa_ind+1:eoa_ind+1])
            # print("outputs_tmp:",outputs_tmp)
            # outputs=torch.cat(outputs_tmp,dim=0)
            # print("outputs:",outputs)

            # print("processed_outputs:", processed_outputs)
            action_outputs=[]
            for outputs_tmp_i in outputs_tmp:
                last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
                processed_outputs = last_token_id_tensor - outputs_tmp_i
                action_outputs_i = self.action_tokenizer.decode(
                    processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
                )#[1,10,7]
                # print(type(action_outputs_i))
                action_outputs.append(action_outputs_i)
            # action_outputs = torch.cat(action_outputs,dim=0)#(2,10,7)
            action_outputs = np.concatenate(action_outputs, axis=0)
            # print("action_outputs:",action_outputs)
            # print("shape:", action_outputs.shape, "dtype:", action_outputs.dtype)
            # action = torch.from_numpy(action_outputs.reshape(1,-1,7)[0])
            action = torch.as_tensor(action_outputs, dtype=torch.float64).reshape(1, -1, 7)[0]
            if self.video_mode:
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
        # except Exception as e:
        #     print(f"Error in step: {e}")
        #     return None