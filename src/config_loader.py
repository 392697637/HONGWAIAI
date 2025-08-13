# config_loader.py
# -----------------------------
# 读取 dataset/data.yaml 配置
# 自动检测 CPU/GPU
# -----------------------------

import torch
from pathlib import Path
import yaml

def load_config(base_dir):
    """
    返回：
        data_yaml       -> 配置文件路径
        names           -> 类别ID -> 名称映射
        categories      -> 分类规则（human/animal/landscape）
        human_classes   -> 人类类别集合
        animal_classes  -> 动物类别集合
        landscape_classes -> 风景/无目标类别集合
        train_cfg       -> 训练参数字典
        infer_cfg       -> 推理参数字典
        device          -> 训练/推理设备（GPU/CPU）
    """

    # -----------------------------
    # 1. 配置文件路径
    # -----------------------------
    data_yaml = base_dir / "dataset" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"❌ 找不到配置文件: {data_yaml}")

    # -----------------------------
    # 2. 读取配置文件
    # -----------------------------
    with open(data_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    names = cfg.get('names', {})             # 类别ID -> 名称映射
    categories = cfg.get('categories', {})   # 分类规则

    human_classes = set(categories.get('human', []))
    animal_classes = set(categories.get('animal', []))
    landscape_classes = set(categories.get('landscape', []))

    train_cfg = cfg.get('train_params', {})
    infer_cfg = cfg.get('infer_params', {})

    # -----------------------------
    # 3. 自动选择设备
    # -----------------------------
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = '0'  # 使用第一块GPU
        print("✅ 检测到GPU，使用设备:", device)
    else:
        device = 'cpu'
        print("⚠️ 未检测到GPU，使用CPU进行训练/推理")

    train_cfg['device'] = device

    return data_yaml, names, categories, human_classes, animal_classes, landscape_classes, train_cfg, infer_cfg, device
