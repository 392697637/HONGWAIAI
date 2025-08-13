#infer_video.py         # 视频推理模块
from ultralytics import YOLO
import cv2
from utils import ensure_dir, save_results_txt

model = YOLO("models/custom_dataset/weights/best.pt")  # 加载训练好的模型
input_video = "dataset/videos/sample.mp4"
output_video = "results/videos/sample_out.mp4"

cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, conf=0.25, verbose=False)[0]
    annotated_frame = results.plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    out.write(annotated_frame)

cap.release()
out.release()
print("✅ 视频推理完成，保存到：", output_video)
