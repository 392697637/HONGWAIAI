# photo_recognition_gui_v2.py
# -----------------------------
# 本地照片识别界面 + 分类显示
# 使用 PyQt5 + YOLOv8
# -----------------------------

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QListWidget, QScrollArea, QTabWidget
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

# 分类规则（与 data.yaml 中一致）
human_classes = {0}        # 示例类别ID，可根据你的data.yaml调整
animal_classes = {1, 2, 3} # 示例类别ID
landscape_classes = {4,5,6}

# 类别名称映射
names = {0:"person", 1:"dog", 2:"cat", 3:"elephant", 4:"tree", 5:"mountain", 6:"river"}

# -----------------------------
# 主窗口类
# -----------------------------
class PhotoRecognizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("本地照片识别")
        self.setGeometry(100, 100, 1200, 700)

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

        # 分类Tab
        self.tab_widget = QTabWidget()
        self.human_tab = QWidget()
        self.animal_tab = QWidget()
        self.landscape_tab = QWidget()

        # 滚动显示图片
        self.human_layout = QHBoxLayout()
        self.human_tab.setLayout(self.human_layout)
        self.animal_layout = QHBoxLayout()
        self.animal_tab.setLayout(self.animal_layout)
        self.landscape_layout = QHBoxLayout()
        self.landscape_tab.setLayout(self.landscape_layout)

        self.tab_widget.addTab(self.human_tab, "Human")
        self.tab_widget.addTab(self.animal_tab, "Animal")
        self.tab_widget.addTab(self.landscape_tab, "Landscape")

        layout.addWidget(self.tab_widget)

        # 分类信息显示
        self.info_list = QListWidget()
        layout.addWidget(self.info_list)

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
            self.info_list.clear()
            # 清空旧布局
            for layout in [self.human_layout, self.animal_layout, self.landscape_layout]:
                for i in reversed(range(layout.count())):
                    widget = layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)

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

            # 分类判断
            detected_cls = set(int(box.cls.cpu().numpy()[0]) for box in r.boxes)
            if detected_cls & human_classes:
                layout_target = self.human_layout
            elif detected_cls & animal_classes:
                layout_target = self.animal_layout
            else:
                layout_target = self.landscape_layout

            # 显示图片
            height, width, channel = annotated.shape
            bytes_per_line = 3 * width
            q_img = QImage(annotated.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            label = QLabel()
            label.setPixmap(pixmap.scaled(320, 240, Qt.KeepAspectRatio))
            layout_target.addWidget(label)

            # 保存信息
            species_list = [f"{names[int(box.cls.cpu().numpy()[0])]}: {float(box.conf.cpu().numpy()[0])*100:.1f}%"
                            for box in r.boxes]
            species_str = "; ".join(species_list) if species_list else "无检测目标"
            self.info_list.addItem(f"{Path(img_path).name}: {species_str}")

# -----------------------------
# 运行应用
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoRecognizer()
    window.show()
    sys.exit(app.exec_())
