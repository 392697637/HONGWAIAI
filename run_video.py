# run_video.py
from pathlib import Path
from src.utils import ensure_dir, find_best_pt
from ultralytics import YOLO
import cv2

# ---------------- 配置 ----------------
input_video_path = Path("./dataset/videos/sample.mp4")  # 输入视频
output_dir = Path("./results/videos")                  # 输出目录
conf_threshold = 0.25                                  # 置信度阈值
draw_boxes = True                                      # 是否绘制框

# 确保输出目录存在
ensure_dir(output_dir)

# 自动找到最新 best.pt
best_model_path = find_best_pt("./models")
print(f"✅ 使用模型: {best_model_path}")

# 加载模型
model = YOLO(best_model_path)

# 打开视频
cap = cv2.VideoCapture(str(input_video_path))
if not cap.isOpened():
    raise FileNotFoundError(f"❌ 无法打开视频: {input_video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_video_path = output_dir / f"{input_video_path.stem}_out.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

print("🚀 开始视频推理...")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]

    if draw_boxes:
        annotated_frame = results.plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    else:
        annotated_frame = frame

    out.write(annotated_frame)

cap.release()
out.release()
print(f"✅ 视频推理完成，结果保存在: {output_video_path}, 总帧数: {frame_count}")
