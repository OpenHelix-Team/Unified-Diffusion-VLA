import os
import json
from tqdm import tqdm

anno_root = "/share/project/datasets/SSv2/annotations"
process_root = "/share/project/yuqi.wang/datasets/processed_data/SSv2"
file_list = os.listdir(process_root)

splits = ["train", "validation"]

for split in splits:
    json_path = os.path.join(anno_root, f"something-something-v2-{split}.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    for item in tqdm(data):
        video_id = item["id"]
        if video_id in file_list:
            video_path = os.path.join(process_root, video_id)
            text = item["label"]

            text_file = os.path.join(video_path, "instruction.txt")
            with open(text_file, "w") as f:
                f.write(text)
