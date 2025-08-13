import os
import cv2
import yaml
import torch
import glob
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ========== 1. 读取 dataset/data.yaml 配置 ==========
base_dir = Path(__file__).resolve().parent  # 获取当前脚本所在目录
print("base_dir", base_dir)

data_yaml = base_dir / "dataset" / "data.yaml"  # 数据集配置文件路径

# 检查配置文件是否存在
if not data_yaml.exists():
    raise FileNotFoundError(f"❌ 找不到配置文件: {data_yaml}")

# 读取配置文件
with open(data_yaml, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# 解析配置文件内容
names = cfg.get('names', {})                     # 类别ID -> 名称映射
categories = cfg.get('categories', {})           # 分类规则（人为分组）

# 将类别分成不同集合，方便推理后分类存储
human_classes = set(categories.get('human', []))
animal_classes = set(categories.get('animal', []))
landscape_classes = set(categories.get('landscape', []))

# 获取训练参数与推理参数
train_cfg = cfg.get('train_params', {})
infer_cfg = cfg.get('infer_params', {})

# 打印配置信息
print(f"📌 类别: {names}")
print(f"📌 分类规则: {categories}")
print(f"📌 训练参数: {train_cfg}")
print(f"📌 推理参数: {infer_cfg}")

# ========== 设备自动检测 ==========
if torch.cuda.is_available():
    device = '0'  # 使用第一块GPU
    print("检测到GPU，使用设备:", device)
else:
    device = 'cpu'  # 使用CPU
    print("未检测到GPU，使用设备:", device)

# 将设备信息写入训练配置
train_cfg['device'] = device

# ========== 工具函数 ==========
def ensure_dir(path):
    """确保目录存在，不存在则创建"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_results_txt(results, save_txt_path):
    """保存检测结果到txt文件（YOLO格式 + 置信度）"""
    with open(save_txt_path, 'w', encoding='utf-8') as f:
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])           # 类别ID
            conf = float(box.conf.cpu().numpy()[0])          # 置信度
            x_center = float(box.xywhn[0][0].cpu().numpy())  # 归一化中心X
            y_center = float(box.xywhn[0][1].cpu().numpy())  # 归一化中心Y
            w = float(box.xywhn[0][2].cpu().numpy())         # 归一化宽
            h = float(box.xywhn[0][3].cpu().numpy())         # 归一化高
            f.write(f"{cls_id} {conf:.4f} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def find_best_pt(start_dir):
    """递归搜索 start_dir 下最新的 best.pt 模型文件"""
    pattern = os.path.join(start_dir, "**", "best.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"❌ 在 {start_dir} 及其子目录中未找到 best.pt")
    candidates.sort(key=os.path.getmtime, reverse=True)  # 按修改时间排序，最新的在前
    return candidates[0]

# ========== 2. 训练模型 ==========
print("\n🚀 开始训练 YOLOv8 自定义模型...")

# 初始化YOLO模型（可指定预训练模型类型）
model = YOLO(train_cfg.get('model_type', 'yolov8n.pt'))

# 执行训练
model.train(
    data=str(data_yaml),                           # 数据配置文件
    epochs=train_cfg.get('epochs', 100),           # 训练轮数
    imgsz=train_cfg.get('imgsz', 640),             # 输入图片大小
    batch=train_cfg.get('batch', 16),              # batch大小
    name="custom_dataset",                         # 训练任务名称
    project=train_cfg.get('save_dir', './models'), # 训练结果保存路径
    workers=4,                                     # 数据加载线程数
    device=train_cfg.get('device'),                # 训练设备
    patience=train_cfg.get('patience', 20),        # 提前停止容忍度
    optimizer=train_cfg.get('optimizer', 'SGD'),   # 优化器
    pretrained=train_cfg.get('pretrained', True)   # 是否使用预训练权重
)

# 自动搜索训练好的best.pt
search_dir = train_cfg.get('save_dir', './models')
best_model_path = find_best_pt(search_dir)
print(f"✅ 训练完成，找到最新 best.pt：{best_model_path}")

# ========== 3. 批量推理 ==========
print("\n🔍 开始批量检测图片...")

# 读取推理输入输出配置
input_dir = Path(infer_cfg.get('input_dir', './images'))
output_dir = Path(infer_cfg.get('output_dir', './results'))
conf_threshold = infer_cfg.get('conf_threshold', 0.25)  # 置信度阈值
draw_boxes = infer_cfg.get('draw_boxes', True)          # 是否绘制检测框

# 创建输出分类目录
ensure_dir(output_dir)
human_dir = output_dir / "human"
animal_dir = output_dir / "animal"
landscape_dir = output_dir / "landscape"
for d in [human_dir, animal_dir, landscape_dir]:
    ensure_dir(d)

# 加载训练好的模型
model = YOLO(best_model_path)

# 获取所有待检测图片
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
print(f"检测到 {len(image_files)} 张图片")

# 识别结果汇总文件
summary_path = output_dir / "识别结果.txt"
with open(summary_path, 'w', encoding='utf-8') as summary_file:
    for img_name in tqdm(image_files, desc="批量检测中"):
        img_path = input_dir / img_name

        # 执行推理
        results = model.predict(str(img_path), conf=conf_threshold, device=device, verbose=False)
        r = results[0]
        r.names = names  # 设置类别名称映射

        # 记录每类最高置信度 & 检测到的类别
        confidences = {}
        detected_cls = set()
        for box in r.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            detected_cls.add(cls_id)
            if cls_id not in confidences or conf > confidences[cls_id]:
                confidences[cls_id] = conf

        # 按类别分到不同目录
        if len(detected_cls) == 0:
            save_dir_target = landscape_dir
        elif detected_cls & human_classes:
            save_dir_target = human_dir
        elif detected_cls & animal_classes:
            save_dir_target = animal_dir
        else:
            save_dir_target = landscape_dir

        # 输出图片路径 & txt路径
        save_img_path = save_dir_target / img_name
        save_txt_path = save_img_path.with_suffix(".txt")

        # 保存绘制框后的图片
        if draw_boxes:
            annotated_img = r.plot()  # 绘制预测结果
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        else:
            annotated_img = cv2.imread(str(img_path))
        cv2.imwrite(str(save_img_path), annotated_img)

        # 保存检测框信息
        save_results_txt(r, save_txt_path)

        # 生成物种识别描述
        species_list = [f"{names[cid]}: 置信度{int(confidences[cid]*100)}%" for cid in confidences]
        species_str = "； ".join(species_list) if species_list else "无检测目标"

        # 记录到识别结果汇总文件
        rel_save_path = os.path.relpath(save_img_path, output_dir).replace("\\", "/")
        summary_file.write(f"原照片: {img_name}, 识别照片: {rel_save_path}, 识别物种: {species_str}\n")

print(f"\n✅ 推理完成，结果保存在：{output_dir}")
