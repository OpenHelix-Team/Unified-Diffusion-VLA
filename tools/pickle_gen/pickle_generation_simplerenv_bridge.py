import os
import os.path as osp
import pickle
from tqdm import tqdm
import numpy as np
import json
import tensorflow as tf
import enum
import sys

# Reasoning tag definitions
class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"

def get_cot_tags_list():
    return [tag.value for tag in CotTag if tag != CotTag.ACTION]

def get_cot_database_keys():
    return {
        CotTag.TASK.value: "task",
        CotTag.PLAN.value: "plan",
        CotTag.VISIBLE_OBJECTS.value: "bboxes",
        CotTag.SUBTASK_REASONING.value: "subtask_reason",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE_REASONING.value: "move_reason",
        CotTag.MOVE.value: "move",
        CotTag.GRIPPER_POSITION.value: "gripper",
    }

def extract_reasoning_list_from_episode(raw_episode):
    reasoning_list = []

    def reasoning_dict_to_str(d):
        tags = get_cot_tags_list()
        database_keys = get_cot_database_keys()
        reasoning_parts = [(tag, d.get(database_keys[tag], "")) for tag in tags]
        return "@".join(f"{tag}@{part}" for tag, part in reasoning_parts)

    if "reasoning" not in raw_episode:
        return reasoning_list

    trajectory_features = raw_episode.get("features", {})
    gripper_positions = trajectory_features.get("gripper_position")
    bboxes = trajectory_features.get("bboxes")

    for i in raw_episode["reasoning"]:
        key = str(i)
        reasoning_dict = raw_episode["reasoning"][i]
        reasoning_dict["gripper"] = ""
        reasoning_dict["bboxes"] = ""

        idx = int(i)
        # Add gripper positions
        if gripper_positions and 0 <= idx < len(gripper_positions):
            future_positions = []
            for j in range(5):
                jdx = idx + j
                if jdx < len(gripper_positions):
                    future_positions += gripper_positions[jdx]
                elif future_positions:
                    future_positions += future_positions[-2:]
            reasoning_dict["gripper"] = str(future_positions)

        # Add bounding boxes
        if bboxes and 0 <= idx < len(bboxes):
            box_list = bboxes[idx]
            if box_list:
                reasoning_dict["bboxes"] = ", ".join(
                    [f"{name} {box}" for prob, name, box in box_list]
                )

        reasoning_str = reasoning_dict_to_str(reasoning_dict)
        reasoning_list.append({
            "key": key,
            "reasoning": reasoning_str,
            "raw": reasoning_dict
        })

    return reasoning_list


# Project-specific imports
sys.path.append("/data/user/wsong890/user68/project/UniVLA")
from train.dataset.normalize_pi0 import RunningStats, save, load

# ======= Path and config =======
dataset_path = "/share/user/iperror/data/univla/processed_data"
output_path = "/share/user/iperror/data/univla/processed_data/meta"
normalizer_path = "/data/user/wsong890/user68/project/UniVLA/configs/normalizer_bridge_tmp"
os.makedirs(normalizer_path, exist_ok=True)

language_dir = f"{dataset_path}/bridge"
vq_dir = "/share/user/iperror/data/univla/processed_data/bridge_orig_codes_256"
interval = 1
frames = 10
with_cot = False

# ======= Processing all scenes =======
result_file = []
for scene in tqdm(sorted(os.listdir(language_dir))):
    try:
        with open(f"{language_dir}/{scene}/instruction.txt", "r") as f:
            text = f.read()

        action = np.load(f"{language_dir}/{scene}/actions/actions.npy")[::interval]

        img_files = [osp.join(vq_dir, scene, 'images', file) for file in sorted(os.listdir(osp.join(vq_dir, scene, 'images')))]
        img1_files = [osp.join(vq_dir, scene, 'images1', file) for file in sorted(os.listdir(osp.join(vq_dir, scene, 'images1')))]

        img_files = img_files[:-1]
        img1_files = img1_files[:-1]

        assert len(img_files) == len(action)

        # COT reasoning (optional)
        if with_cot:
            with open(f"{language_dir}/{scene}/cot.json", "r") as f:
                cot = json.load(f)
            reasoning_list = extract_reasoning_list_from_episode(cot)
            if len(reasoning_list) != len(action):
                reasoning_list = reasoning_list[1:1 + len(action)]
            assert len(reasoning_list) == len(action)
        else:
            reasoning_list = []

        if len(img_files) < frames:
            continue

        result_file.append({
            "text": text,
            "image": img_files,
            "gripper_image": img1_files,
            "action": action,
            "reasoning": reasoning_list
        })
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")
        continue

print(f"Total number of valid scenes: {len(result_file)}")

# ======= Normalize action =======
normalizer = RunningStats()
action_data = np.concatenate([scene["action"] for scene in result_file])
normalizer.update(action_data)
norm_stats = normalizer.get_statistics()

print("Mean:", norm_stats.mean)
print("Std:", norm_stats.std)
print("Q01:", norm_stats.q01)
print("Q99:", norm_stats.q99)

save(normalizer_path, {"bridge_robot": norm_stats})

for scene in result_file:
    action = scene["action"].copy()
    normalized = 2 * (action - norm_stats.q01) / (norm_stats.q99 - norm_stats.q01 + 1e-8) - 1
    scene["action"] = np.clip(normalized, -1, 1)

# ======= Save final data =======
output_file = osp.join(output_path, "simplerenv_bridge_trainval.pkl")
with open(output_file, "wb") as f:
    pickle.dump(result_file, f)

print(f"Processed dataset saved to {output_file}")
