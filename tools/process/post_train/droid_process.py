import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
from tqdm import tqdm 
from utils import droid_dataset_transform

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Set the dataset name for saving, which affects the save path
SAVE_DATASET_NAME = "droid_fast"  # Can be changed to "droid" or another name

def save_gif(images, path="temp.gif"):
    # Convert a sequence of images to a gif (frame rate: 15Hz)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000/15), loop=0)

def save_images_in_batch(images, save_dir, scene_name):
    # Save image frames in batch
    for i, img in enumerate(images):
        img.save(f"{save_dir}/{scene_name}_{i}.jpg")

def process_episode(episode, save_dirs, scene):
    images = []
    images2 = []
    images3 = []
    language = []
    actions = []  # 7-dimensional action
    reward = []
    actions_dict = []  # detailed actions
    gripper_position = []

    # Get the language instruction from the first step
    lang = []
    for i, step in enumerate(episode["steps"]):
        if i == 0:  # Only keep the instruction from the first step
            if "language_instruction" in step and step["language_instruction"].numpy().decode() != "":
                lang.append(step["language_instruction"].numpy().decode())
            if "language_instruction_2" in step and step["language_instruction_2"].numpy().decode() != "":
                lang.append(step["language_instruction_2"].numpy().decode())
            if "language_instruction_3" in step and step["language_instruction_3"].numpy().decode() != "":
                lang.append(step["language_instruction_3"].numpy().decode())
        else:
            break

    # Save language whether it's empty or not
    language.append(",".join(lang) if lang else "")

    # Process image and action data
    for i, step in enumerate(episode["steps"]):
        images.append(Image.fromarray(step["observation"]["wrist_image_left"].numpy()))
        images2.append(Image.fromarray(step["observation"]["exterior_image_1_left"].numpy()))
        images3.append(Image.fromarray(step["observation"]["exterior_image_2_left"].numpy()))
        actions.append(step['action'])
        reward.append(step["reward"].numpy())
        gripper_position.append(step["action_dict"]['gripper_position'].numpy())
    
    # Create save directories for each scene
    scene_video_dir = os.path.join(save_dirs['video_dir'], scene)
    scene_video2_dir = os.path.join(save_dirs['video2_dir'], scene)
    scene_video3_dir = os.path.join(save_dirs['video3_dir'], scene)
    
    os.makedirs(scene_video_dir, exist_ok=True)
    os.makedirs(scene_video2_dir, exist_ok=True)
    os.makedirs(scene_video3_dir, exist_ok=True)

    actions = np.array(droid_dataset_transform(actions))

    # Save action data
    np.save(f"{save_dirs['action']}/{scene}.npy", actions)

    # Save GIF animations
    # save_gif(images, f"{save_dirs['gif']}/{SAVE_DATASET_NAME}_{scene}_wrist.gif")
    # save_gif(images2, f"{save_dirs['gif']}/{SAVE_DATASET_NAME}_{scene}_exterior_image_1.gif")
    # save_gif(images3, f"{save_dirs['gif']}/{SAVE_DATASET_NAME}_{scene}_exterior_image_2.gif")

    # Save image frames in batch
    # save_images_in_batch(images, scene_video_dir, scene)
    # save_images_in_batch(images2, scene_video2_dir, scene)
    # save_images_in_batch(images3, scene_video3_dir, scene)

    # Save language instructions to text file regardless of whether they exist
    # with open(f"{save_dirs['language']}/{scene}.txt", "w") as f:
    #     f.write("\n".join(language))

    print(f"Finished processing scene {scene}")
    return 1

def main():
    # Set the root path for the dataset
    root = "/share/project/yuqi.wang/datasets/robotics"
    ds = tfds.load("droid", data_dir=root, split="train")

    save_dir = f"/share/project/yuqi.wang/datasets/processed_data/{SAVE_DATASET_NAME}"
    os.makedirs(save_dir, exist_ok=True)

    save_dirs = {
        'gif': f"{save_dir}/gif",
        'video_dir': f"{save_dir}/wrist_image_left",
        'video2_dir': f"{save_dir}/exterior_image_1_left",
        'video3_dir': f"{save_dir}/exterior_image_2_left",
        'language': f"{save_dir}/language",
        'action': f"{save_dir}/action"
    }

    # Create all necessary subdirectories
    for dir_path in save_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    episode_count = 0

    # Process each episode with a progress bar
    for episode_index, episode in enumerate(tqdm(ds, desc="Processing episodes", unit="episode")):
        scene = episode["episode_metadata"]['recording_folderpath'].numpy().decode().split("/")[-3]
        # if os.path.exists(f"{save_dirs['video_dir']}/{scene}"):
        #     print(f"Scene {scene} already processed. Skipping...")
        #     continue
        result = process_episode(episode, save_dirs, scene)
        if result:
            episode_count += result

    print(f"Processed {episode_count} episodes.")

if __name__ == "__main__":
    main()
