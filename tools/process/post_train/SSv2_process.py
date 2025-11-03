import os
import cv2

def extract_frames_from_webm(input_dir, output_dir, fps=5):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".webm"):
            video_id = os.path.splitext(filename)[0]  # e.g., '1234'
            video_path = os.path.join(input_dir, filename)
            
            output_subdir = os.path.join(output_dir, video_id, "images")
            os.makedirs(output_subdir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps == 0:
                print(f"Warning: can't read FPS of {video_path}")
                continue
            
            interval = int(video_fps / fps)
            if interval <= 0:
                interval = 1

            frame_idx = 0
            saved_idx = 1
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval == 0:
                    out_path = os.path.join(output_subdir, f"{saved_idx:04d}.jpg")
                    cv2.imwrite(out_path, frame)
                    saved_idx += 1
                frame_idx += 1
            cap.release()
            print(f"Finished processing {filename}, saved {saved_idx - 1} frames.")

# 用法示例
input_folder = "/share/project/datasets/SSv2/videos"
output_folder = "/share/project/yuqi.wang/datasets/processed_data/SSv2"
os.makedirs(output_folder, exist_ok=True)
extract_frames_from_webm(input_folder, output_folder, fps=5)
