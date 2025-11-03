import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
import sys

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import rt1_dataset_transform  # 确保utils.py里有这个函数

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用 GPU（可选）

def process_fractal_dataset(dataset_dir, output_dir):
    # Load dataset from TFDS format directory
    builder = tfds.builder_from_directory(dataset_dir)
    ds_all_dict = builder.as_dataset(split="train")

    os.makedirs(output_dir, exist_ok=True)

    episode_count = 0
    for episode in tqdm(ds_all_dict, desc="Processing episodes", unit="episode"):
        episode_name = str(episode_count)
        episode_dir = os.path.join(output_dir, episode_name)
        os.makedirs(episode_dir, exist_ok=True)

        image_dir = os.path.join(episode_dir, 'images')
        action_dir = os.path.join(episode_dir, 'actions')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(action_dir, exist_ok=True)

        actions = []

        for i, step in tqdm(enumerate(episode["steps"]), desc=f"Episode {episode_name}", total=len(episode["steps"]), unit="step"):
            observation = step["observation"]
            action = step["action"]

            # Extract action components
            gripper_action = action["gripper_closedness_action"]
            world_vector = action["world_vector"]
            rotation_delta = action["rotation_delta"]

            step_action = tf.concat([world_vector, rotation_delta, gripper_action], axis=-1)
            actions.append(step_action)

            # Save image
            image = Image.fromarray(observation["image"].numpy())
            image_path = os.path.join(image_dir, f"{i:04d}.jpg")
            image.save(image_path)

            # Save instruction at first step
            if i == 0:
                text = observation["natural_language_instruction"].numpy().decode()
                with open(os.path.join(episode_dir, "instruction.txt"), "w") as f:
                    f.write(text)

        # Save transformed actions
        actions_np = np.array(rt1_dataset_transform(actions))
        np.save(os.path.join(action_dir, "actions.npy"), actions_np)

        episode_count += 1


def main():
    parser = argparse.ArgumentParser(description="Process RT-1 style dataset from Fractal TFDS format.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the TFDS dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed episodes.")
    args = parser.parse_args()

    process_fractal_dataset(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
