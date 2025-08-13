# infer_batch.py
# -----------------------------
# 批量推理模块：使用训练好的 YOLOv8 模型对图片进行检测
# 并按类别（human/animal/landscape）保存结果
# -----------------------------

import os
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def ensure_dir(path):
    """确保目录存在，不存在则创建"""
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

def batch_infer(model_path, names, human_classes, animal_classes, landscape_classes, infer_cfg, device):
    """
    批量推理入口
    参数：
        model_path      -> 训练好的 best.pt 模型路径
        names           -> 类别ID -> 名称映射字典
        human_classes   -> 人类类别ID集合
        animal_classes  -> 动物类别ID集合
        landscape_classes -> 风景/无目标类别ID集合
        infer_cfg       -> 推理配置字典，包括输入/输出路径、置信度阈值等
        device          -> 推理设备，GPU或CPU
    """

    print("\n🔍 开始批量检测图片...")

    # 推理参数
    input_dir = Path(infer_cfg.get('input_dir', './images'))
    output_dir = Path(infer_cfg.get('output_dir', './results'))
    conf_threshold = infer_cfg.get('conf_threshold', 0.25)
    draw_boxes = infer_cfg.get('draw_boxes', True)

    # 创建输出目录和子目录
    ensure_dir(output_dir)
    human_dir = output_dir / "human"
    animal_dir = output_dir / "animal"
    landscape_dir = output_dir / "landscape"
    for d in [human_dir, animal_dir, landscape_dir]:
        ensure_dir(d)

    # 加载模型
    model = YOLO(model_path)

    # 获取图片列表
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"检测到 {len(image_files)} 张图片")

    # 识别结果汇总
    summary_path = output_dir / "识别结果.txt"
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        for img_name in tqdm(image_files, desc="批量检测中"):
            img_path = input_dir / img_name

            # YOLO推理
            results = model.predict(str(img_path), conf=conf_threshold, device=device, verbose=False)
            r = results[0]
            r.names = names

            # 统计每类最高置信度 & 检测到的类别
            confidences = {}
            detected_cls = set()
            for box in r.boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                detected_cls.add(cls_id)
                if cls_id not in confidences or conf > confidences[cls_id]:
                    confidences[cls_id] = conf

            # 根据类别分目录
            if len(detected_cls) == 0:
                save_dir_target = landscape_dir
            elif detected_cls & human_classes:
                save_dir_target = human_dir
            elif detected_cls & animal_classes:
                save_dir_target = animal_dir
            else:
                save_dir_target = landscape_dir

            # 输出图片路径和 txt 路径
            save_img_path = save_dir_target / img_name
            save_txt_path = save_img_path.with_suffix(".txt")

            # 保存带检测框的图片
            if draw_boxes:
                annotated_img = r.plot()
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            else:
                annotated_img = cv2.imread(str(img_path))
            cv2.imwrite(str(save_img_path), annotated_img)

            # 保存检测框信息
            save_results_txt(r, save_txt_path)

            # 生成物种识别描述
            species_list = [f"{names[cid]}: 置信度{int(confidences[cid]*100)}%" for cid in confidences]
            species_str = "； ".join(species_list) if species_list else "无检测目标"

            # 记录汇总文件
            rel_save_path = os.path.relpath(save_img_path, output_dir).replace("\\", "/")
            summary_file.write(f"原照片: {img_name}, 识别照片: {rel_save_path}, 识别物种: {species_str}\n")

    print(f"\n✅ 批量推理完成，结果保存在：{output_dir}")
