import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
import sys

# 假设 utils.py 在项目根目录下
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import relabel_bridge_actions, binarize_gripper_actions

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用GPU，适合CPU环境

def process_bridge_dataset(dataset_dir, output_dir):
    # 加载TFDS数据集
    builder = tfds.builder_from_directory(dataset_dir)
    ds_all_dict = builder.as_dataset(split="train")
    # ds_all_dict = builder.as_dataset(split="validation")

    os.makedirs(output_dir, exist_ok=True)

    count = 1
    for episode in tqdm(ds_all_dict, desc="Processing episodes", unit="episode"):
        episode_name = str(count)
        episode_dir = os.path.join(output_dir, episode_name)
        os.makedirs(episode_dir, exist_ok=True)

        image_dir = os.path.join(episode_dir, 'images')
        image_dir1 = os.path.join(episode_dir, 'images1')
        action_dir = os.path.join(episode_dir, 'actions')

        for d in [image_dir, image_dir1, action_dir]:
            os.makedirs(d, exist_ok=True)

        languages = []
        actions = []
        states = []

        for i, step in tqdm(enumerate(episode["steps"]), desc=f"Episode {episode_name}", total=len(episode["steps"]), unit="step"):
            if i == 0:
                continue

            observation = step["observation"]
            action = step["action"]
            state = observation["state"]

            step_action = tf.concat(
                (
                    action[:6],  # First 6 dims of action
                    binarize_gripper_actions(action[6:]),  # Binarize gripper actions
                ),
                axis=-1
            )

            actions.append(step_action)
            states.append(state)

            # 保存两路图片
            image = Image.fromarray(observation["image_0"].numpy())
            image1 = Image.fromarray(observation["image_1"].numpy())

            image.save(os.path.join(image_dir, f"{i:04d}.jpg"))
            image1.save(os.path.join(image_dir1, f"{i:04d}.jpg"))

            if i == 1:
                text = step["language_instruction"].numpy().decode()
                languages.append(text)
                with open(os.path.join(episode_dir, "instruction.txt"), "w") as f:
                    f.write(text)

        traj = {
            "observation": {"state": tf.convert_to_tensor(states, dtype=tf.float32)},
            "action": tf.convert_to_tensor(actions, dtype=tf.float32)
        }

        traj = relabel_bridge_actions(traj)
        actions_np = traj["action"].numpy()
        np.save(os.path.join(action_dir, "actions.npy"), actions_np)

        count += 1


def main():
    parser = argparse.ArgumentParser(description="Process Bridge dataset into image/action episodes.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the extracted Bridge TFDS dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the processed episodes.")
    args = parser.parse_args()

    process_bridge_dataset(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
