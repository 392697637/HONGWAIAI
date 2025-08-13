# web_app/app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import cv2
from ultralytics import YOLO
from src.utils import ensure_dir, find_best_pt
import asyncio

app = FastAPI(title="YOLOv8 Web API")

# 文件夹配置
upload_dir = Path("./web_app/uploads")
output_dir = Path("./web_app/results")
templates_dir = Path("./web_app/templates")
ensure_dir(upload_dir)
ensure_dir(output_dir)

# 静态文件挂载
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# 加载模型
best_model_path = find_best_pt("./models")
model = YOLO(best_model_path)

# ---------------- 页面 ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    with open(templates_dir / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ---------------- 图片检测 ----------------
@app.post("/detect_image")
async def detect_image(file: UploadFile = File(...)):
    temp_path = upload_dir / file.filename
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model.predict(str(temp_path), conf=0.25, verbose=False)[0]
    annotated_img = results.plot()
    out_path = output_dir / f"web_{file.filename}"
    cv2.imwrite(str(out_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

    return {"filename": file.filename, "result_path": str(out_path)}

# ---------------- 视频检测 ----------------
@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    temp_path = upload_dir / file.filename
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        return {"error": "无法打开视频"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = output_dir / f"web_{file.filename}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.25, verbose=False)[0]
        annotated_frame = results.plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(annotated_frame)

    cap.release()
    out.release()
    return {"filename": file.filename, "result_path": str(out_path)}

# ---------------- 实时摄像头流 ----------------
@app.get("/video_feed")
def video_feed():
    # 使用摄像头 0
    cap = cv2.VideoCapture(0)

    def gen():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=0.25, verbose=False)[0]
            annotated_frame = results.plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')
