import os
import os.path as osp
import pickle
from tqdm import tqdm
import numpy as np
import sys
# ====================== Config ======================

# Project-specific paths
PROJECT_ROOT = "/data/user/wsong890/user68/project"
sys.path.append(f"{PROJECT_ROOT}/UniVLA")

# Import normalization utility
from train.dataset.normalize_pi0 import RunningStats, save

# Input paths
DATASET_NAME = "calvin_abcd"  # Options: calvin, calvin_d, calvin_abc
dataset_path = f"/share/user/iperror/data/univla/processed_data"
language_dir = f"{dataset_path}/{DATASET_NAME}"
vq_dir = f"{dataset_path}/{DATASET_NAME}_codes"
gripper_vq_dir = f"{dataset_path}/{DATASET_NAME}_gripper_codes"

# Output paths
output_path = f"{dataset_path}/meta"
normalizer_path = f"{PROJECT_ROOT}/OmniSim/configs/normalizer_calvin"
output_pkl_file = osp.join(output_path, f"{DATASET_NAME}_norm.pkl")

# Settings
interval = 1           # Not currently used but may be useful
min_frame_count = 8    # Minimum frame count per scene
use_raw_images = False # If True, use raw RGB images instead of VQ codes

# ====================================================

# Ensure output dirs exist
os.makedirs(normalizer_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# Load and process dataset
result_file = []
for scene in tqdm(os.listdir(language_dir), desc="Processing scenes"):#遍历每个episode
    instr_file = osp.join(language_dir, scene, "instruction.txt")
    if not osp.exists(instr_file):
        print(f"Warning: Missing instruction file in {scene}")
        continue

    with open(instr_file, "r") as f:
        text = f.read().strip()

    # Load action data
    action_folder = osp.join(language_dir, scene, "actions")
    if not osp.exists(action_folder):
        print(f"Warning: Missing action folder in {scene}")
        continue

    action_files = sorted([
        osp.join(action_folder, fname)
        for fname in os.listdir(action_folder)
        if fname.endswith(".npz")
    ])

    try:
        actions = [np.load(f)["rel_actions"] for f in action_files]
    except Exception as e:
        print(f"Error loading actions for {scene}: {e}")
        continue

    # Load image paths
    if use_raw_images:
        img_dir = osp.join(language_dir, scene, "rgb_static")
        gripper_dir = osp.join(language_dir, scene, "rgb_gripper")
    else:
        img_dir = osp.join(vq_dir, scene)
        gripper_dir = osp.join(gripper_vq_dir, scene)
        # print(img_dir)
        # print(gripper_dir)

    try:
        img_files = sorted([
            osp.join(img_dir, fname) for fname in os.listdir(img_dir)
        ])
        gripper_img_files = sorted([
            osp.join(gripper_dir, fname) for fname in os.listdir(gripper_dir)
        ])
    except FileNotFoundError:
        print(f"Warning: Missing VQ images in {scene}")
        continue

    # Skip scenes with too few frames
    if len(img_files) < min_frame_count:
        continue

    result_file.append({
        "text": text,
        "image": img_files,#路径
        "action": actions,
        "gripper_image": gripper_img_files,
    })

print(f"\nTotal valid scenes: {len(result_file)}")

# Initialize and compute normalization statistics
if not result_file:
    raise RuntimeError("No valid scenes found. Exiting.")

normalizer = RunningStats()
action_data = np.concatenate([scene["action"] for scene in result_file])
normalizer.update(action_data)
stats = normalizer.get_statistics()
save(normalizer_path, {DATASET_NAME: stats})
# Print stats
print("Normalization statistics:")
print("  Mean:", stats.mean)
print("  Std:", stats.std)
print("  Q01:", stats.q01)
print("  Q99:", stats.q99)

# Save normalization parameters
save(normalizer_path, {DATASET_NAME: stats})

# Normalize all actions
for scene in result_file:
    action = scene["action"]
    normalized = 2 * (action - stats.q01) / (stats.q99 - stats.q01 + 1e-8) - 1
    scene["action"] = np.clip(normalized, -1, 1)

# Save result
with open(output_pkl_file, "wb") as f:
    pickle.dump(result_file, f)

print(f"\nProcessed dataset saved to: {output_pkl_file}")
