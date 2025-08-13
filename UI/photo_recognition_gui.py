# photo_recognition_gui.py
# -----------------------------
# 本地照片识别界面示例
# 使用 PyQt5 + YOLOv8
# -----------------------------

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QListWidget, QListWidgetItem, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
from ultralytics import YOLO

# -----------------------------
# 配置部分
# -----------------------------
MODEL_PATH = "models/custom_dataset/weights/best.pt"  # YOLOv8 模型路径
DEVICE = "0"  # GPU: "0", CPU: "cpu"

# -----------------------------
# 主窗口类
# -----------------------------
class PhotoRecognizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("本地照片识别")
        self.setGeometry(100, 100, 1000, 600)

        # 初始化模型
        self.model = YOLO(MODEL_PATH)

        # 存储图片路径和标注结果
        self.image_paths = []
        self.annotated_images = []

        # UI布局
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 上传按钮
        self.upload_btn = QPushButton("上传图片")
        self.upload_btn.clicked.connect(self.upload_images)
        layout.addWidget(self.upload_btn)

        # 开始识别按钮
        self.infer_btn = QPushButton("开始识别")
        self.infer_btn.clicked.connect(self.start_infer)
        layout.addWidget(self.infer_btn)

        # 分类列表
        self.category_list = QListWidget()
        self.category_list.setMaximumHeight(120)
        layout.addWidget(self.category_list)

        # 图片显示区域
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QHBoxLayout()
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    # -----------------------------
    # 上传图片
    # -----------------------------
    def upload_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if files:
            self.image_paths = files
            self.category_list.clear()
            self.scroll_layout.setParent(None)  # 清空原来的图片
            self.scroll_layout = QHBoxLayout()
            self.scroll_widget.setLayout(self.scroll_layout)

            for f in self.image_paths:
                item = QListWidgetItem(Path(f).name)
                self.category_list.addItem(item)

    # -----------------------------
    # 开始识别
    # -----------------------------
    def start_infer(self):
        if not self.image_paths:
            return

        self.annotated_images = []

        for img_path in self.image_paths:
            results = self.model.predict(
                source=img_path,
                device=DEVICE,
                conf=0.25,
                verbose=False
            )
            r = results[0]
            # 获取标注图片
            annotated = r.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            self.annotated_images.append(annotated)

        self.display_results()

    # -----------------------------
    # 显示标注图片
    # -----------------------------
    def display_results(self):
        # 清空旧图片
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for img in self.annotated_images:
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            label = QLabel()
            label.setPixmap(pixmap.scaled(320, 240, Qt.KeepAspectRatio))
            self.scroll_layout.addWidget(label)

# -----------------------------
# 运行应用
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoRecognizer()
    window.show()
    sys.exit(app.exec_())
