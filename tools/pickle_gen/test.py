import numpy as np
from pathlib import Path
import sys
PROJECT_ROOT = "/data/user/wsong890/user68/project"
sys.path.append(f"{PROJECT_ROOT}/UniVLA")
# ====== 引入你的 RunningStats 和 save 方法 ======
from train.dataset.normalize_pi0 import RunningStats, save

# ====== 模拟数据 ======
# 假设每个 scene 有 shape (N, action_dim) 的动作数据
result_file = [
    {"action": np.random.randn(10, 4)},  # 10 帧，每帧4维动作
    {"action": np.random.randn(8, 4)},
    {"action": np.random.randn(12, 4)},
]

DATASET_NAME = "test_dataset"
normalizer_path = "./normalizer_test"

# ====== 测试流程 ======
normalizer = RunningStats()
# 把所有 scene 的动作拼成一个大矩阵
action_data = np.concatenate([scene["action"] for scene in result_file])
# 更新统计量
normalizer.update(action_data)
# 获取统计结果
stats = normalizer.get_statistics()

# 打印结果
print("Normalization statistics:")
print("  Mean:", stats.mean)
print("  Std:", stats.std)
print("  Q01:", stats.q01)
print("  Q99:", stats.q99)

# 保存
Path(normalizer_path).mkdir(parents=True, exist_ok=True)
save(normalizer_path, {DATASET_NAME: stats})

print(f"Saved stats to {normalizer_path}")
