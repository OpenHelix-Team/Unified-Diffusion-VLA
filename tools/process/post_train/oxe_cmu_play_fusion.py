import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
from tqdm import tqdm  # Import tqdm for progress bars
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import relabel_bridge_actions, binarize_gripper_actions, rel2abs_gripper_actions,invert_gripper_actions

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the dataset
dataset_dirs = "/share/project/lxh/datasets/tf_datasets/cmu_play_fusion"
builder = tfds.builder_from_directory(dataset_dirs)

ds_all_dict = builder.as_dataset(split="train")

# Base directory where all episodes will be saved
base_output_dir = "/share/project/yuqi.wang/datasets/processed_data/oxembodiment/cmu_play_fusion"
os.makedirs(base_output_dir, exist_ok=True)

# Process the dataset and save with tqdm progress bar for episodes
count = 1
for episode in tqdm(ds_all_dict, desc="Processing episodes", unit="episode"):
    name = str(count)  # Use episode index as folder name
    
    # Create a directory for each episode
    episode_dir = os.path.join(base_output_dir, name)
    os.makedirs(episode_dir, exist_ok=True)
    
    # Create a subdirectory for images
    image_dir = os.path.join(episode_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    # Create a subdirectory for actions
    action_dir = os.path.join(episode_dir, 'actions')
    os.makedirs(action_dir, exist_ok=True)

    # Prepare to store images and text
    images = []
    languages = []
    actions = []
    states = []
    
    # Process each step in the episode with tqdm for progress within the episode
    for i, step in tqdm(enumerate(episode["steps"]), desc=f"Processing episode {name}", total=len(episode["steps"]), unit="step"):
        observation = step["observation"]
        action = step["action"]

        step_action = tf.concat(
            (
                action[:3],
                action[-4:],
            ),
            axis=0,
        )

        actions.append(step_action)

        image = observation["image"]
        image = Image.fromarray(image.numpy())

        # Save images to the episode folder
        image_filename = os.path.join(image_dir, f"{i:04d}.jpg")
        image.save(image_filename)

        if i == 0:  # Save text from the first step (natural language instruction)
            text = step["language_instruction"].numpy().decode()
            languages.append(text)
            
            # Save the language instruction as a text file in the episode folder
            text_filename = os.path.join(episode_dir, f"instruction.txt")
            with open(text_filename, "w") as text_file:
                text_file.write(text)    
    # save actions
    actions = np.array(actions)
    action_filename = os.path.join(action_dir, f"actions.npy")
    np.save(action_filename, actions)

    # Clear variables to free memory
    images = None
    count += 1

