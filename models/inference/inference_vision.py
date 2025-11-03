import argparse
import os
import torch
import pickle
import numpy as np
from time import time
from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import sys
from torch.nn.functional import cross_entropy
from random import shuffle
import random
from PIL import Image, ImageDraw, ImageFont
import math
import time

# Add vllm imports
try:
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.model_executor.layers.logits_processor import LogitsProcessor as VLLMLogitsProcessor
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vllm not available. Install with 'pip install vllm' to enable faster inference.")

sys.path.append("/share/project/yuqi.wang/UniVLA/reference/Emu3")
from emu3.mllm.processing_emu3 import Emu3Processor
from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM
from emu3.mllm import Emu3MoE
from transformers import LogitsProcessor

def save_pil_image_grid_with_title(image_list, filename, title="", nrow=4, padding=2, title_height=40, bg_color=(255, 255, 255), font_path=None, font_size=20):
    if not image_list:
        print("Empty image list.")
        return

    w, h = image_list[0].size
    n_images = len(image_list)
    ncol = math.ceil(n_images / nrow)

    grid_w = nrow * w + (nrow - 1) * padding
    grid_h = ncol * h + (ncol - 1) * padding + (title_height if title else 0)

    grid_img = Image.new('RGB', (grid_w, grid_h), color=bg_color)
    draw = ImageDraw.Draw(grid_img)

    if title:
        try:
            font = ImageFont.truetype(font_path if font_path else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (grid_w - text_w) // 2
        text_y = (title_height - text_h) // 2
        draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)

    for idx, img in enumerate(image_list):
        row = idx // nrow
        col = idx % nrow
        x = col * (w + padding)
        y = row * (h + padding) + (title_height if title else 0)
        grid_img.paste(img, (x, y))

    grid_img.save(filename)
    print(f"Saved image grid to {filename}")

class VisionIDConstraintLogitsProcessor(LogitsProcessor):
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

# VLLM version of the logits processor
class VLLMVisionIDConstraintLogitsProcessor(VLLMLogitsProcessor):
    def __init__(self, allowed_token_ids):
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

def load_torch_model(model_path):
    """Load model using standard PyTorch approach"""
    model = Emu3MoE.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    return model.to(device).eval()

def load_vllm_model(model_path):
    """Load model using VLLM for optimized inference"""
    if not VLLM_AVAILABLE:
        raise ImportError("vllm package is not available. Install with 'pip install vllm'")
    
    # Create a modified config.json that VLLM can understand
    import json
    import os
    import shutil
    import tempfile
    
    # Create a temporary directory to hold modified config
    temp_model_dir = tempfile.mkdtemp(prefix="vllm_emu3_")
    print(f"Created temporary directory for model: {temp_model_dir}")
    
    try:
        # Copy original model files to temp directory
        for item in os.listdir(model_path):
            src_path = os.path.join(model_path, item)
            dst_path = os.path.join(temp_model_dir, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        
        # Load and modify config.json
        config_path = os.path.join(temp_model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Save original model_type and architectures for reference
            if 'model_type' in config:
                config['original_model_type'] = config['model_type']
            if 'architectures' in config:
                config['original_architectures'] = config['architectures']
            
            # Change model_type to llama
            config['model_type'] = 'llama'
            
            # Change architectures to LlamaForCausalLM
            config['architectures'] = ['LlamaForCausalLM']
            
            # Write modified config back
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("Modified config.json to use model_type 'llama' and architecture 'LlamaForCausalLM'")
        else:
            print(f"Warning: config.json not found in {model_path}")
        
        # Load the model from the temporary directory
        model = LLM(
            model=temp_model_dir,
            dtype="bfloat16",
            trust_remote_code=True,
            skip_tokenizer_init=True,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
        )
        return model, temp_model_dir  # Return the temp dir so it can be cleaned up later
    
    except Exception as e:
        # Clean up temp directory if there's an error
        shutil.rmtree(temp_model_dir, ignore_errors=True)
        raise e

def parse_args():
    parser = argparse.ArgumentParser(description="Vision Language Model Inference")
    parser.add_argument("--predict_frames", type=int, default=5, help="Number of frames to predict")
    parser.add_argument("--context_frames", type=int, default=1, help="Number of context frames")
    parser.add_argument("--use_gripper", action="store_true", help="Use gripper images")
    parser.add_argument("--samples", type=int, default=16, help="Number of videos to generate")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--use_vllm", action="store_true", help="Use VLLM for faster inference")
    parser.add_argument("--output_dir", type=str, default="/share/project/yuqi.wang/UniVLA/logs/VLA_VIS_WM_c1f6", 
                        help="Output directory for visualizations")
    return parser.parse_args()

"""
    Stage: Post train
    Description: Inference for world model
"""

if __name__ == "__main__":
    args = parse_args()
    
    # Apply command line arguments
    predict_frame = args.predict_frames
    context_frames = args.context_frames
    gripper = args.use_gripper
    video_average_nums = args.samples
    debug = args.debug
    use_vllm = args.use_vllm and VLLM_AVAILABLE
    OUTPUT_DIR = args.output_dir
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_pickle = '/share/project/yuqi.wang/datasets/post_train_data/meta/world_model_post_train_v3.pkl'
    with open(train_pickle, 'rb') as f:
        train_meta = pickle.load(f)

    EMU_HUB = "/share/project/yuqi.wang/UniVLA/logs/ckpts/WORLD_MODEL_POSTTRAIN_VIDEO3"
    VQ_HUB = "/share/project/yuqi.wang/UniVLA/pretrain/Emu3-Base"
    VISION_HUB = "/share/project/yuqi.wang/UniVLA/pretrain/Emu3-VisionVQ"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model based on selected inference mode
    temp_dir_to_cleanup = None
   
    if use_vllm:
        print("Using VLLM for accelerated inference")
        model, temp_dir_to_cleanup = load_vllm_model(EMU_HUB)
    else:
        print("Using PyTorch for inference")
        model = load_torch_model(EMU_HUB)
    
    kwargs = dict(
        mode='INTERLEAVE',
        padding="longest",
    )
    
    GENERATION_CONFIG = GenerationConfig(
        pad_token_id=model.config.pad_token_id if not use_vllm else None,
        bos_token_id=model.config.bos_token_id if not use_vllm else None,
        eos_token_id=model.config.eos_token_id if not use_vllm else None,
        do_sample=True,
        top_k=1024,       
        temperature=0.7,
    )

    tokenizer = Emu3Tokenizer.from_pretrained(
            VQ_HUB,
            model_max_length=model.config.max_position_embeddings if not use_vllm else 4096,
            padding_side="right",
            use_fast=False,
        )
    image_processor = AutoImageProcessor.from_pretrained(VISION_HUB, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(VISION_HUB, trust_remote_code=True).to(device).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    image_processor.min_pixels = 128 * 128

    # Get model-specific configuration
    if use_vllm:
        bov_token_id = 151854
        eov_token_id = 184621
    else:
        bov_token_id = model.config.bov_token_id
        eov_token_id = model.config.eov_token_id

    allowed_token_ids = list(range(bov_token_id, eov_token_id + 1))
    
    saved_count = 0  
    i = 0
    while saved_count < video_average_nums:
        task_idx = random.randint(0, len(train_meta) - 1)
        task_data = train_meta[task_idx]
        text = task_data['text']
        print(f"Task {saved_count+1}/{video_average_nums}: {text}")
        image_list = task_data['image']
        rand_idx = random.randint(0, len(image_list) - predict_frame - 1)
        image = image_list[rand_idx]
        if gripper:
            gripper_list = task_data['gripper_image']
            gripper = gripper_list[rand_idx]
            gripper_code = np.load(gripper)
            gripper_code = torch.tensor(gripper_code).unsqueeze(0)
        else:
            gripper_code = None
        video_code = np.load(image)
        video_code = torch.tensor(video_code).unsqueeze(0)

        pos_inputs = processor.video_process(text=text, video_tokens=video_code, gripper_tokens=gripper_code ,\
            context_frames=context_frames, frames=predict_frame, return_tensors="pt", **kwargs)
        max_tokens = pos_inputs.video_size[0][0]*pos_inputs.video_size[0][1]

        frame_list = []
        # add the first frame
        with torch.no_grad():
            first_frame = image_tokenizer.decode(video_code.squeeze(0).to(device))
            first_image = image_processor.postprocess(first_frame)["pixel_values"][0]
            frame_list.append(first_image)

            # Initialize input_ids and attention_mask
            input_ids = pos_inputs.input_ids
            attention_mask = pos_inputs.attention_mask
            
            if use_vllm:
                # VLLM-based inference
                sampling_params = SamplingParams(
                    temperature=1.0,
                    top_k=2048,
                    max_tokens=max_tokens,
                    logits_processors=[VLLMVisionIDConstraintLogitsProcessor(allowed_token_ids)]
                )
                
                # Move input_ids to CPU once, reuse for each frame
                input_ids_cpu = input_ids[0].tolist()
                
                for frame_id in range(predict_frame):
                    start_time = time.time()  # Start timing
                    
                    # Use the pre-converted list for TokensPrompt
                    inputs_text = TokensPrompt(prompt_token_ids=input_ids_cpu)
                    
                    # Generate with vllm
                    outputs = model.generate(inputs_text, sampling_params)
                    token_ids_tuple = outputs[0].outputs[0].token_ids
                    token_ids_list = list(token_ids_tuple)
                    outputs_tensor = torch.tensor([token_ids_list], dtype=torch.long)
                    
                    video_tokens = outputs_tensor - bov_token_id
                    video_tokens = video_tokens.reshape(1, pos_inputs.video_size[0][0], pos_inputs.video_size[0][1]).to(device)
                    
                    # Decode video tokens into an image
                    decoded = image_tokenizer.decode(video_tokens)
                    recon_image = image_processor.postprocess(decoded)["pixel_values"][0]
                    frame_list.append(recon_image)
                    
                    end_time = time.time()
                    print(f"[Frame {frame_id+1}] VLLM Inference time: {end_time - start_time:.2f} seconds")
                    
                    # Update input text for next generation
                    prefix_ids = processor.add_prefix_template(1, pos_inputs.video_size[0][0], pos_inputs.video_size[0][1])
                    prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long).unsqueeze(0).to(outputs_tensor.device)
                    
                    # Convert outputs_tensor to a list and extend input_ids_cpu
                    input_ids_cpu.extend(outputs_tensor[0].tolist())
                    input_ids_cpu.extend(prefix_tensor[0].tolist())

                    
            else:
                # PyTorch-based inference
                vision_id_processor = VisionIDConstraintLogitsProcessor(allowed_token_ids)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                for frame_id in range(predict_frame):
                    start_time = time.time()  # Start timing
                    
                    # Autoregressively generate new tokens based on the current input
                    outputs = model.generate(
                        input_ids,
                        GENERATION_CONFIG,
                        max_new_tokens=max_tokens,
                        logits_processor=[vision_id_processor],
                        attention_mask=attention_mask,
                    )

                    final_ids = outputs  # Save the complete sequence including previous and newly generated tokens
                    new_tokens = final_ids[:, input_ids.shape[1]:]  # Extract only the newly generated tokens

                    # Reshape and adjust the video tokens
                    video_tokens = new_tokens.reshape(1, pos_inputs.video_size[0][0], pos_inputs.video_size[0][1])
                    video_tokens = video_tokens - model.config.bov_token_id  # Remove base offset

                    # Decode video tokens into an image
                    decoded = image_tokenizer.decode(video_tokens)
                    recon_image = image_processor.postprocess(decoded)["pixel_values"][0]
                    frame_list.append(recon_image)
                    
                    end_time = time.time()  # End timing
                    print(f"[Frame {frame_id+1}] PyTorch Inference time: {end_time - start_time:.2f} seconds")
                    
                    # Update input_ids and attention_mask for the next autoregressive step
                    prefix_ids = processor.add_prefix_template(1, pos_inputs.video_size[0][0], pos_inputs.video_size[0][1])
                    prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long).unsqueeze(0).to(final_ids.device)

                    input_ids = torch.cat([final_ids, prefix_tensor], dim=-1)
                    attention_mask = torch.ones_like(input_ids)

        # save the generated frames
        output_name = os.path.join(OUTPUT_DIR, f"video_{saved_count}.png")
        save_pil_image_grid_with_title(frame_list, output_name, title=text, nrow=args.predict_frames+args.context_frames, padding=2, title_height=40)
        saved_count += 1
        i += 1
    # Clean up temporary directory if it exists
    if temp_dir_to_cleanup and os.path.exists(temp_dir_to_cleanup):
        print(f"Cleaning up temporary directory: {temp_dir_to_cleanup}")
        import shutil
        shutil.rmtree(temp_dir_to_cleanup, ignore_errors=True)