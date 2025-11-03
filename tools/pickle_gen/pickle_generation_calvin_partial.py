import os
import os.path as osp
import pickle
from tqdm import tqdm
import numpy as np
import sys

# specific to the project
sys.path.append("/share/project/yuqi.wang/UniVLA")
from train.dataset.normalizer import LinearNormalizer
from train.dataset.normalize_pi0 import RunningStats, save, load

dataset_path = "/share/project/yuqi.wang/datasets/processed_data"
output_path = "/share/project/yuqi.wang/datasets/processed_data/meta"

language_dir = f"{dataset_path}/calvin_partial"
vq_dir = f"{dataset_path}/calvin_partial_codes"
gripper_vq_dir = f"{dataset_path}/calvin_partial_gripper_codes"

normalizer_path = "/share/project/yuqi.wang/UniVLA/configs/normalizer_calvin_partial"
os.makedirs(normalizer_path, exist_ok=True)

interval = 1
frames = 8  # filter at least 8 frames

# Load scenes and process actions
result_file = []
for scene in tqdm(os.listdir(language_dir)):
    with open(f"{language_dir}/{scene}/instruction.txt", "r") as f:
        text = f.read()
    
    action_folder = f"{language_dir}/{scene}/actions"
    action_files = [osp.join(action_folder,file) for file in sorted(os.listdir(action_folder))]
    action = [np.load(a)['rel_actions'] for a in action_files]
    
    img_files = [osp.join(vq_dir, scene, file) for file in sorted(os.listdir(osp.join(vq_dir, scene)))]
    gripper_img_files = [osp.join(gripper_vq_dir, scene, file) for file in sorted(os.listdir(osp.join(gripper_vq_dir, scene)))]
    # Filter scenes with less than 8 frames
    if len(img_files) < frames:
        continue
    result_file.append({"text": text, "image": img_files, "action": action, "gripper_image": gripper_img_files})
print(f"Total number of scenes: {len(result_file)}")



# Initialize RunningStats
normalizer = RunningStats()

# Ensure we have action data
if not result_file:
    raise ValueError("No valid action data found in the dataset.")

# Aggregate action data
action_data = np.concatenate([scene["action"] for scene in result_file])

# Update statistics
normalizer.update(action_data)

# Get normalization statistics
norm_stats = normalizer.get_statistics()

# Print normalization parameters
print("Mean:", norm_stats.mean)
print("Standard Deviation:", norm_stats.std)
print("Q01 (1% quantile):", norm_stats.q01)
print("Q99 (99% quantile):", norm_stats.q99)

# Convert statistics to a JSON-compatible format
norm_stats_save = {
    "calvin_abcd_partial": norm_stats,
}

# Save normalizer parameters
save(normalizer_path, norm_stats_save)

# Normalize actions
for scene in result_file:
    action = scene["action"].copy()
    # Normalize and clip
    normalized = 2 * (action - norm_stats.q01) / (norm_stats.q99 - norm_stats.q01 + 1e-8) - 1
    scene["action"] = np.clip(normalized, -1, 1)

    # Decode check
    # action_decoded = 0.5 * (scene["action"] + 1) * (norm_stats.q99 - norm_stats.q01) + norm_stats.q01
# Save processed data
output_file = osp.join(output_path, "calvin_abcd_partial_norm.pkl")
with open(output_file, "wb") as f:
    pickle.dump(result_file, f)

print(f"Processed dataset saved to {output_file}")

