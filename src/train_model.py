# train_model.py
# -----------------------------
# 训练 YOLOv8 自定义模型
# 返回训练完成的 best.pt 路径
# -----------------------------

import os
import glob
from ultralytics import YOLO

def ensure_dir(path):
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)

def find_best_pt(start_dir):
    """递归搜索 start_dir 下最新的 best.pt"""
    pattern = os.path.join(start_dir, "**", "best.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"❌ 在 {start_dir} 及其子目录中未找到 best.pt")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def train_model(data_yaml, train_cfg):
    """
    参数：
        data_yaml -> 数据集配置路径
        train_cfg -> 训练参数字典
    返回：
        best_model_path -> 训练完成的模型权重路径
    """

    print("\n🚀 开始训练 YOLOv8 自定义模型...")

    # 初始化YOLO模型
    model = YOLO(train_cfg.get('model_type', 'yolov8n.pt'))

    # 执行训练
    model.train(
        data=str(data_yaml),
        epochs=train_cfg.get('epochs', 100),
        imgsz=train_cfg.get('imgsz', 640),
        batch=train_cfg.get('batch', 16),
        name="custom_dataset",
        project=train_cfg.get('save_dir', './models'),
        workers=train_cfg.get('workers', 4),
        device=train_cfg.get('device'),
        patience=train_cfg.get('patience', 20),
        optimizer=train_cfg.get('optimizer', 'SGD'),
        pretrained=train_cfg.get('pretrained', True)
    )

    # 搜索训练好的 best.pt
    search_dir = train_cfg.get('save_dir', './models')
    best_model_path = find_best_pt(search_dir)
    print(f"✅ 训练完成，找到最新 best.pt：{best_model_path}")

    return best_model_path
