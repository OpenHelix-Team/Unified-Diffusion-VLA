# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os, os.path as osp, json, subprocess
# from glob import glob
# from tqdm import tqdm
# import numpy as np
# import pyarrow.parquet as pq
# from PIL import Image

# # 读取 mp4 的方式：'opencv' 或 'ffmpeg'
# VIDEO_READER = 'ffmpeg'  # 改成 'ffmpeg' 来规避 AV1 问题；如需回退可设为 'opencv'

# # ======== 路径（按需修改） ========
# LEROBOT_ROOT = "/share/user/iperror/data/ur5e_datasets/origin/cube4_1"        # 含 meta/ videos/ data/
# SAVE_ROOT    = "/data/user/wsong890/user68/project/UniVLA/debug_real_data"    # 输出 processed_data 根

# # 相机 -> processed_data 目录名
# CAMERA_MAP = {
#     "observation.images.webcam": "rgb_static",
#     "observation.images.wrist_camera":  "rgb_gripper",
# }

# # 文件命名
# FRAME_PAD = 5
# IMG_EXT   = ".jpg"

# def ensure_dir(d):
#     os.makedirs(d, exist_ok=True); return d

# # ---------- OpenCV 路径（保留以备不时之需） ----------
# def save_rgb_jpg(np_rgb, out_path):
#     if np_rgb.dtype != np.uint8:
#         np_rgb = np.clip(np_rgb, 0, 255).astype(np.uint8)
#     Image.fromarray(np_rgb).save(out_path, quality=95)

# def extract_frames_opencv(video_path, max_frames=None):
#     import cv2
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {video_path}")
#     frames = []
#     while True:
#         ok, frame = cap.read()
#         if not ok: break
#         frames.append(frame[:, :, ::-1])  # BGR->RGB
#         if max_frames and len(frames) >= max_frames:
#             break
#     cap.release()
#     return frames

# # ---------- ffmpeg 解帧（推荐：支持 AV1 via libdav1d） ----------
# def ffmpeg_extract_to_dir(video_path, out_img_dir, frame_pad=7, max_frames=None):
#     """
#     用 ffmpeg 把 video_path 解出为 out_img_dir/%0{pad}d.jpg（从 1 开始编号）
#     依赖系统 ffmpeg，建议加载新一点的模块（例如 module load ffmpeg/7.0.2）
#     """
#     ensure_dir(out_img_dir)

#     # 先清理历史帧（避免残留）
#     for p in glob(osp.join(out_img_dir, "*.jpg")):
#         try: os.remove(p)
#         except: pass

#     pattern = osp.join(out_img_dir, f"%0{frame_pad}d.jpg")

#     cmd = [
#         "ffmpeg", "-y",
#         "-c:v", "libdav1d",      # 强制用 dav1d 软件解码 AV1；其他编解码也能走此路径
#         "-i", video_path,
#         "-pix_fmt", "yuv420p",   # 保险起见，统一 8-bit 4:2:0
#         "-q:v", "2",             # JPG 质量（2 高、31 低）
#     ]
#     if max_frames is not None and max_frames > 0:
#         cmd += ["-frames:v", str(max_frames)]
#     cmd += [pattern]

#     # 静默运行；如需调试把 stderr/stdout 去掉 DEVNULL
#     subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     # 数一数导出了多少帧
#     return len(glob(osp.join(out_img_dir, "*.jpg")))

# def find_episode_video(cam_dir, episode_index):
#     idx = int(episode_index)
#     fname = f"episode_{idx:06d}.mp4"
#     path = osp.join(cam_dir, fname)
#     return path if osp.exists(path) else None


# def read_episode_parquet_rows(data_chunk_dir, episode_index):
#     files = sorted(glob(osp.join(data_chunk_dir, "*.parquet")))
#     for fp in files:
#         table = pq.read_table(fp)

#         # 直接按 episode_index 过滤行
#         eidx = table["episode_index"].to_pylist()
#         rows = [i for i, v in enumerate(eidx) if int(v) == int(episode_index)]
#         if not rows:
#             continue

#         # 取固定列：frame_index, action, observation.state
#         fi  = [table["frame_index"][i].as_py() for i in rows]
#         act = [table["action"][i].as_py() for i in rows]
#         # observation.state 可能缺失；如果一定存在也可直接读取
#         try:
#             obs = [table["observation.state"][i].as_py() for i in rows]
#         except KeyError:
#             obs = None

#         out = {
#             "frame_index": np.array(fi, dtype=int),
#             "actions":     np.array(act, dtype=object),
#         }
#         if obs is not None:
#             out["robot_obs"] = np.array(obs, dtype=object)  # 映射为 robot_obs 键名
#         return out

#     return None


# def to_array_list(x):
#     if x is None: return None
#     if isinstance(x, np.ndarray): x = x.tolist()
#     arr = []
#     for v in x:
#         v = np.asarray(v).reshape(-1)
#         arr.append(v)
#     return arr

# def main():
#     meta_dir   = osp.join(LEROBOT_ROOT, "meta")
#     videos_dir = osp.join(LEROBOT_ROOT, "videos", "chunk-000")
#     data_dir   = osp.join(LEROBOT_ROOT, "data",   "chunk-000")  # 

#     # 读 episodes.jsonl（固定字段：episode_index、tasks、length）
#     episodes_path = osp.join(meta_dir, "episodes.jsonl")
#     assert osp.exists(episodes_path), f"not found: {episodes_path}"
#     episodes = []
#     with open(episodes_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             rec = json.loads(line)
#             episodes.append({
#                 "episode_index": int(rec["episode_index"]),
#                 "instruction":   (rec.get("tasks") or [""])[0],
#                 "num_frames":    int(rec["length"]),
#             })
#     print(f"[INFO] episodes: {len(episodes)}")

#     for epi in tqdm(episodes, desc="chunk-000 -> processed_data"):
#         eidx = epi["episode_index"]
#         instr = epi["instruction"]
#         nframes_meta = epi["num_frames"]

#         # 输出目录 + 指令
#         out_dir = ensure_dir(osp.join(SAVE_ROOT, f"video_{eidx}"))
#         with open(osp.join(out_dir, "instruction.txt"), "w", encoding="utf-8") as fw:
#             fw.write(instr)

#         # 逐相机解帧
#         cam_frame_counts = {}
#         for cam_key, dst_name in CAMERA_MAP.items():
#             cam_dir = osp.join(videos_dir, cam_key)
#             if not osp.isdir(cam_dir):
#                 print(f"[WARN] missing dir: {cam_dir}")
#                 cam_frame_counts[cam_key] = 0
#                 continue

#             vpath = find_episode_video(cam_dir, eidx)
#             if not vpath:
#                 print(f"[WARN] no video for ep {eidx} / {cam_key}")
#                 cam_frame_counts[cam_key] = 0
#                 continue

#             out_img_dir = ensure_dir(osp.join(out_dir, dst_name))
#             if VIDEO_READER == 'ffmpeg':
#                 n = ffmpeg_extract_to_dir(vpath, out_img_dir, frame_pad=FRAME_PAD, max_frames=nframes_meta)
#                 cam_frame_counts[cam_key] = n
#             else:
#                 frames = extract_frames_opencv(vpath, max_frames=nframes_meta)
#                 for t, frame in enumerate(frames):
#                     save_rgb_jpg(frame, osp.join(out_img_dir, f"{t+1:0{FRAME_PAD}d}{IMG_EXT}"))
#                 cam_frame_counts[cam_key] = len(frames)

#         # 读取并整理 Parquet 动作/观测（固定列：frame_index, action, observation.state, episode_index）
#         pack = read_episode_parquet_rows(data_dir, eidx) if osp.isdir(data_dir) else None
#         act_list = rob_list = None
#         if pack:
#             fi = pack["frame_index"]

#             def to_vec_list(x):
#                 if x is None:
#                     return None
#                 if isinstance(x, np.ndarray):
#                     x = x.tolist()
#                 return [np.asarray(v).reshape(-1) for v in x]

#             act_list = to_vec_list(pack["actions"])
#             rob_list = to_vec_list(pack.get("robot_obs"))  # 由 observation.state 映射来

#             # 按 frame_index 排序
#             order = np.argsort(fi)
#             if act_list is not None:
#                 act_list = [act_list[i] for i in order]
#             if rob_list is not None:
#                 rob_list = [rob_list[i] for i in order]

#         # 对齐长度（相机帧、meta、动作/观测）
#         counts = [c for c in cam_frame_counts.values() if c > 0]
#         counts.append(nframes_meta)
#         if act_list is not None:
#             counts.append(len(act_list))
#         if rob_list is not None:
#             counts.append(len(rob_list))
#         T = min(counts) if counts else 0

#         # 保存 actions/*.npz（仅写存在的键）
#         if T > 0 and (act_list is not None or rob_list is not None):
#             act_dir = ensure_dir(osp.join(out_dir, "actions"))
#             for t in range(T):
#                 rec = {}
#                 if act_list is not None:
#                     rec["actions"]   = np.asarray(act_list[t])
#                 if rob_list is not None:
#                     rec["robot_obs"] = np.asarray(rob_list[t])
#                 np.savez(osp.join(act_dir, f"{t+1:0{FRAME_PAD}d}.npz"), **rec)

#     print(f"[DONE] processed_data saved to: {SAVE_ROOT}")
# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, os.path as osp, json, subprocess
from glob import glob
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq

# 固定：只用 ffmpeg
VIDEO_READER = 'ffmpeg'

# ======== 路径（按需修改） ========
LEROBOT_ROOT = "/share/user/iperror/data/ur5e_datasets/origin/cube4_1"
task_name=LEROBOT_ROOT.split("/")[-1]
SAVE_ROOT    = f"/data/user/wsong890/user68/project/UniVLA/debug_real_data"

# 固定相机与输出子目录
CAMERA_MAP = {
    "observation.images.webcam":        "rgb_static",
    "observation.images.wrist_camera":  "rgb_gripper",
}

FRAME_PAD = 5
IMG_EXT   = ".jpg"

def ensure_dir(d):
    os.makedirs(d, exist_ok=True); return d

def ffmpeg_extract_to_dir(video_path, out_img_dir, frame_pad=5, max_frames=None):
    ensure_dir(out_img_dir)
    # 清空旧帧
    for p in glob(osp.join(out_img_dir, "*.jpg")): 
        try: os.remove(p)
        except: pass
    pattern = osp.join(out_img_dir, f"%0{frame_pad}d.jpg")
    cmd = ["ffmpeg", "-y", "-c:v", "libdav1d", "-i", video_path, "-pix_fmt", "yuv420p", "-q:v", "2"]
    if max_frames and max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [pattern]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return len(glob(osp.join(out_img_dir, "*.jpg")))

def find_episode_video(cam_dir, episode_index):
    return osp.join(cam_dir, f"episode_{int(episode_index):06d}.mp4")

def read_episode_parquet_rows(data_chunk_dir, episode_index):
    # fp = sorted(glob(osp.join(data_chunk_dir, "*.parquet")))[0]  # 固定一个文件即可
    fp=osp.join(data_chunk_dir, f"episode_{int(episode_index):06d}.parquet")
    print("fp:",fp)
    table = pq.read_table(fp)
    # print("table.keys:",table.schema.names)
    eidx = table["episode_index"].to_pylist()
    rows = [i for i, v in enumerate(eidx) if int(v) == int(episode_index)]

    fi  = [table["frame_index"][i].as_py() for i in rows]
    act = [table["action"][i].as_py() for i in rows]
    print("len_act:",len(act))
    # print("act:",act)
    obs = [table["observation.state"][i].as_py() for i in rows]

    return {
        "frame_index": np.array(fi,  dtype=int),
        "actions":     np.array(act, dtype=object),
        "robot_obs":   np.array(obs, dtype=object),
    }

def to_vec_list(x):
    # 把每帧数据变成一维向量
    if isinstance(x, np.ndarray): x = x.tolist()
    return [np.asarray(v).reshape(-1) for v in x]

def main():
    meta_dir   = osp.join(LEROBOT_ROOT, "meta")
    videos_dir = osp.join(LEROBOT_ROOT, "videos", "chunk-000")
    data_dir   = osp.join(LEROBOT_ROOT, "data",   "chunk-000")

    # 固定格式读取 episodes.jsonl
    episodes_path = osp.join(meta_dir, "episodes.jsonl")
    episodes = []
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            episodes.append({
                "episode_index": int(rec["episode_index"]),
                "instruction":   (rec.get("tasks") or [""])[0],
                "num_frames":    int(rec["length"]),
            })
    print(f"[INFO] episodes: {len(episodes)}")

    for epi in tqdm(episodes, desc="chunk-000 -> processed_data"):
        eidx = epi["episode_index"]
        instr = epi["instruction"]
        nframes_meta = epi["num_frames"]

        # 输出目录与指令
        out_dir = ensure_dir(osp.join(SAVE_ROOT, f"{task_name}_video_{eidx}"))
        with open(osp.join(out_dir, "instruction.txt"), "w", encoding="utf-8") as fw:
            fw.write(instr)

        # 两路相机解帧（固定存在）
        cam_counts = []
        for cam_key, dst_name in CAMERA_MAP.items():
            cam_dir = osp.join(videos_dir, cam_key)
            vpath = find_episode_video(cam_dir, eidx)
            out_img_dir = ensure_dir(osp.join(out_dir, dst_name))
            n = ffmpeg_extract_to_dir(vpath, out_img_dir, frame_pad=FRAME_PAD, max_frames=nframes_meta)
            cam_counts.append(n)

        # 读取 parquet（固定列）
        pack = read_episode_parquet_rows(data_dir, eidx)
        fi = pack["frame_index"]
        act_list = to_vec_list(pack["actions"])
        # print("act_list:",act_list)
        rob_list = to_vec_list(pack["robot_obs"])

        # 按帧号排序
        order = np.argsort(fi)
        act_list = [act_list[i] for i in order]
        # rob_list = [rob_list[i] for i in order]

        # 取对齐长度（最小值）
        T = min(min(cam_counts), nframes_meta, len(act_list), len(rob_list))
        print("len(act_list):",len(act_list))
        # T = min(min(cam_counts), nframes_meta, len(act_list))
        # 写 actions/*.npz
        act_dir = ensure_dir(osp.join(out_dir, "actions"))
        for t in range(T):
            np.savez(
                osp.join(act_dir, f"{t+1:0{FRAME_PAD}d}.npz"),
                actions   = np.asarray(act_list[t]),
                robot_obs = np.asarray(rob_list[t]),
            )

    print(f"[DONE] processed_data saved to: {SAVE_ROOT}")

if __name__ == "__main__":
    main()
