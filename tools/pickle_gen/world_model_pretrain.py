import os
import os.path as osp
import pickle
from tqdm import tqdm
import random
import sys

# Add project path
sys.path.append("/share/project/yuqi.wang/UniVLA")

# ========== Settings ==========
debug = False  # Set to True for debugging (only processes 10 samples per dataset)
max_videos = 100000  # Maximum number of videos to use per dataset (set None to disable)
minimum_frames = 6

# Paths
dataset_path = "/share/project/yuqi.wang/datasets/processed_data"
vq_dataset_path = "/share/project/yuqi.wang/datasets/post_train_data"
output_path = "/share/project/yuqi.wang/datasets/post_train_data/meta"
os.makedirs(output_path, exist_ok=True)

# Dataset to VQ map
DATASETS_VQMAP = {
    'rt1': 'rt1_codes_240_3hz',
    'droid_fast': [
        'droid/exterior_image_1_left_codes_256_15hz',
        'droid/exterior_image_2_left_codes_256_15hz',
        'droid/wrist_image_left_codes_256_15hz'
    ],
    'oxembodiment/bridge': 'bridge_orig_codes_256',
    'oxembodiment/toto': 'toto_codes_256',
    'oxembodiment/taco_play': 'taco_play_codes_256',
    'oxembodiment/fmb': 'fmb_codes_256',
    'oxembodiment/maniskill': 'maniskill_codes_256',
    'oxembodiment/kuka': 'kuka_codes_256',
    'oxembodiment/berkeley_autolab_ur5': 'berkeley_autolab_ur5_codes_256',
    'oxembodiment/cmu_play_fusion': 'cmu_play_fusion_codes_256',
    'oxembodiment/viola': 'viola_codes_256',
    'oxembodiment/utaustin_mutex': 'utaustin_mutex_codes_256',
}

# Additional simulator datasets
calvin_pickle_path = "/share/project/yuqi.wang/datasets/processed_data/meta/calvin_gripper.pkl"
libero_pickle_path = "/share/project/yuqi.wang/datasets/processed_data/meta/libero_all_norm_aug.pkl"
ssv2_pickle_path = "/share/project/yuqi.wang/datasets/post_train_data/meta/SSv2.pickle"

ALL_DATASETS = list(DATASETS_VQMAP.keys())
all_samples = []
dataset_stats = {}

# Determine required frame count per dataset
def get_required_frames(dataset_name):
    if dataset_name == 'rt1' or dataset_name == 'oxembodiment/kuka':
        return 3 * minimum_frames
    elif dataset_name == 'droid_fast' or dataset_name == 'oxembodiment/viola':
        return 15 * minimum_frames
    elif dataset_name == 'oxembodiment/maniskill' or dataset_name == 'oxembodiment/cmu_play_fusion' or dataset_name == 'oxembodiment/utaustin_mutex':
        return 10 * minimum_frames
    elif dataset_name == 'oxembodiment/toto':
        return 20 * minimum_frames
    else:
        return 5 * minimum_frames

for dataset in ALL_DATASETS:
    print(f"\nProcessing dataset: {dataset}")
    required_frames = get_required_frames(dataset)
    is_multi_vq = isinstance(DATASETS_VQMAP[dataset], list)

    valid_samples = []

    if is_multi_vq:
        language_dir = osp.join(dataset_path, dataset, "language")
        for vq_name in DATASETS_VQMAP[dataset]:
            vq_dir = osp.join(vq_dataset_path, vq_name)
            for scene_file in tqdm(os.listdir(language_dir), desc=f"Processing {vq_name}"):
                if debug and len(valid_samples) >= 10:
                    break

                scene_path = osp.join(language_dir, scene_file)
                instruction = open(scene_path, "r").read().strip().split(",")[0]
                if instruction == "":
                    continue

                scene_name = scene_file.replace(".txt", "")
                scene_vq_path = osp.join(vq_dir, scene_name)
                if not osp.exists(scene_vq_path):
                    continue

                image_files = [osp.join(scene_vq_path, f) for f in sorted(os.listdir(scene_vq_path))]
                if len(image_files) < required_frames:
                    continue

                valid_samples.append({
                    "text": instruction,
                    "image": image_files,
                    "dataset": dataset
                })
    else:
        language_dir = osp.join(dataset_path, dataset)
        vq_dir = osp.join(vq_dataset_path, DATASETS_VQMAP[dataset])

        for scene in tqdm(os.listdir(language_dir), desc=f"Processing {dataset}"):
            if debug and len(valid_samples) >= 10:
                break

            scene_path = osp.join(language_dir, scene, "instruction.txt")
            instruction = open(scene_path, "r").read().strip()
            if instruction == "":
                continue

            vq_scene_path = osp.join(vq_dir, scene)
            if not osp.exists(vq_scene_path):
                continue

            if osp.exists(osp.join(vq_scene_path, "images")):
                vq_scene_path = osp.join(vq_scene_path, "images")

            image_files = [osp.join(vq_scene_path, f) for f in sorted(os.listdir(vq_scene_path))]
            if len(image_files) < required_frames:
                continue

            valid_samples.append({
                "text": instruction,
                "image": image_files,
                "dataset": dataset
            })

    # Limit to max_videos if set
    if dataset != "droid_fast":
        if max_videos is not None and len(valid_samples) > max_videos:
            valid_samples = random.sample(valid_samples, max_videos)

    all_samples.extend(valid_samples)
    dataset_stats[dataset] = len(valid_samples)

# Load simulator data and append
with open(calvin_pickle_path, "rb") as f:
    calvin_data = pickle.load(f)
for sample in calvin_data:
    sample["dataset"] = "calvin"
print(f"\nAdded {len(calvin_data)} samples from Calvin.")

with open(libero_pickle_path, "rb") as f:
    libero_data = pickle.load(f)
for sample in libero_data:
    sample["dataset"] = "libero"
print(f"Added {len(libero_data)} samples from Libero.")

# Load SSv2 data
with open(ssv2_pickle_path, "rb") as f:
    ssv2_data = pickle.load(f)
for sample in ssv2_data:
    sample["dataset"] = "ssv2"
print(f"Added {len(ssv2_data)} samples from SSv2.")

all_samples.extend(ssv2_data)
all_samples.extend(calvin_data)
all_samples.extend(libero_data)

# Print dataset stats
print("\n=== Dataset Summary ===")
total = sum(dataset_stats.values()) + len(calvin_data) + len(libero_data) + len(ssv2_data)
for name, count in dataset_stats.items():
    print(f"{name:35s}: {count}")
print(f"\nTotal number of valid scenes after merging: {total}")

# Save to file
output_file = osp.join(output_path, "world_model_post_train_v3.pkl")
with open(output_file, "wb") as f:
    pickle.dump(all_samples, f)
print(f"\nSaved processed dataset to: {output_file}")
