import os
作用：提供操作系统相关功能

用途：
路径操作（拼接、检查是否存在）
文件和目录管理（创建、删除、遍历文件夹）
获取环境变量等

import cv2
作用：OpenCV 库，用于图像和视频处理

用途：
读取、显示和保存图片 (cv2.imread, cv2.imwrite)
视频处理 (cv2.VideoCapture, cv2.VideoWriter)
图像绘制（在图片上画框、文字、标注）
图像格式转换（RGB↔BGR）

import yaml
作用：解析 YAML 配置文件

用途：
读取 dataset/data.yaml 等配置
将 YAML 数据转换成 Python 字典
支持方便地配置训练参数、类别信息、推理参数

import torch
作用：PyTorch 深度学习框架
用途：
检查 GPU/CPU 设备
加载模型权重、执行张量运算
YOLOv8 内部依赖 PyTorch 进行训练和推理

import glob
作用：文件模式匹配工具

用途：
搜索符合特定模式的文件（如 **/best.pt）
支持递归查找子目录
自动找到训练生成的最新模型文件

from pathlib import Path
作用：面向对象的路径操作库（Python 内置）

用途：
拼接路径 (base_dir / "dataset" / "data.yaml")
检查路径是否存在 (Path.exists())
创建目录 (Path.mkdir())
更直观、跨平台比 os.path 更易用

from ultralytics import YOLO
作用：导入 YOLOv8 模型接口

用途：
初始化模型 (YOLO("yolov8n.pt"))
训练模型 (model.train(...))
推理预测 (model.predict(...))
保存和加载权重 (model.load() / model.save())

from tqdm import tqdm
作用：可视化进度条工具

用途：
在循环中显示处理进度（如批量图片推理）
for img in tqdm(image_files) 会显示已完成/总数/估计剩余时间
提高用户体验，尤其在处理大量数据时

总结一下，这些库配合使用可以实现 读取配置 → 管理路径 → 训练 YOLOv8 模型 → 批量图片/视频推理 → 保存结果并显示进度条 的完整流程。