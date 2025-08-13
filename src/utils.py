# utils.py  utils.py（工具函数）
# src/utils.py
"""
工具函数模块
- 创建目录
- 保存 YOLO 检测结果
- 搜索最新 best.pt
"""

import os
import glob
from pathlib import Path

def ensure_dir(path):
    """确保目录存在，不存在则自动创建"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_results_txt(results, save_txt_path):
    """保存 YOLO 检测结果到 txt 文件"""
    with open(save_txt_path, 'w', encoding='utf-8') as f:
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            x_center = float(box.xywhn[0][0].cpu().numpy())
            y_center = float(box.xywhn[0][1].cpu().numpy())
            w = float(box.xywhn[0][2].cpu().numpy())
            h = float(box.xywhn[0][3].cpu().numpy())
            f.write(f"{cls_id} {conf:.4f} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def find_best_pt(start_dir):
    """递归搜索 start_dir 下最新的 best.pt 文件"""
    pattern = os.path.join(start_dir, "**", "best.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"❌ 在 {start_dir} 及其子目录中未找到 best.pt")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]
