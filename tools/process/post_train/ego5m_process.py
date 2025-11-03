import os
import pandas as pd
import re
import cv2
from tqdm import tqdm

# Path to the CSV file
file_path = 'datasets/general/egovid-text.csv'

# Root directory containing the video files
EGO_ROOT = 'datasets/EGO4D/full_scale'

# Directory to save the extracted results
OUTPUT_DIR = 'datasets/processed_data/ego5m'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the CSV file
df = pd.read_csv(file_path)

def process_video(row):
    video_id = row['video_id']
    video_name = video_id.split('_')[0] + '.mp4'
    video_path = os.path.join(EGO_ROOT, video_name)
    frame_idx = row['frame_idx']
    
    # Extract start and end frame indices using regex
    numbers = re.findall(r'\d+', frame_idx)
    if len(numbers) < 2:
        print(f"Invalid frame index format: {frame_idx}")
        return
    
    begin = int(numbers[0])
    end = int(numbers[1])
    
    language = row['llava_cap']
    
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        
        # Output directory for this video segment
        video_output_dir = os.path.join(OUTPUT_DIR, video_id.split('.')[0])
        images_dir = os.path.join(video_output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # Save language description to text file
        language_filename = os.path.join(video_output_dir, 'language.txt')
        with open(language_filename, 'w') as f:
            f.write(language)
        
        # Set the starting frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, begin)
        
        frame_count = begin
        while frame_count <= end:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_count} from {video_name}")
                break
            
            # Save the current frame to the images directory
            frame_filename = os.path.join(images_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
        
        cap.release()
        print(f"Frames {begin}-{end} from {video_name} saved to {images_dir}")
    else:
        print(f"Video not found: {video_path}")

# Iterate through each row in the CSV and process the corresponding video segment
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
    process_video(row)
