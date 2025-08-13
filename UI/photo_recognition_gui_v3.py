# photo_recognition_gui_v3.py
# -----------------------------
# 本地照片识别 + 分类显示 + 批量导出TXT和保存图片
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
import os

# -----------------------------
# 配置部分
# -----------------------------
MODEL_PATH = "models/custom_dataset/weights/best.pt"  # YOLOv8 模型路径
DEVICE = "0"  # GPU: "0", CPU: "cpu"

# 分类规则（根据 data.yaml 配置调整）
human_classes = {0}        # 示例类别ID
animal_classes = {1, 2, 3} # 示例类别ID
landscape_classes = {4,5,6}

# 类别名称映射
names = {0:"person", 1:"dog", 2:"cat", 3:"elephant", 4:"tree", 5:"mountain", 6:"river"}

# -----------------------------
# 工具函数
# -----------------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_results_txt(results, save_txt_path):
    """保存检测结果到txt文件"""
    with open(save_txt_path, 'w', encoding='utf-8') as f:
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            x_center = float(box.xywhn[0][0].cpu().numpy())
            y_center = float(box.xywhn[0][1].cpu().numpy())
            w = float(box.xywhn[0][2].cpu().numpy())
            h = float(box.xywhn[0][3].cpu().numpy())
            f.write(f"{cls_id} {conf:.4f} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

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

        # 选择输出目录
        save_root = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not save_root:
            return
        save_root = Path(save_root)

        # 创建分类目录
        human_dir = save_root / "human"
        animal_dir = save_root / "animal"
        landscape_dir = save_root / "landscape"
        for d in [human_dir, animal_dir, landscape_dir]:
            ensure_dir(d)

        # 清空旧布局
        for layout in [self.human_layout, self.animal_layout, self.landscape_layout]:
            for i in reversed(range(layout.count())):
                widget = layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

        self.info_list.clear()

        # 批量识别
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
                save_dir_target = human_dir
            elif detected_cls & animal_classes:
                layout_target = self.animal_layout
                save_dir_target = animal_dir
            else:
                layout_target = self.landscape_layout
                save_dir_target = landscape_dir

            # 显示图片
            height, width, channel = annotated.shape
            bytes_per_line = 3 * width
            q_img = QImage(annotated.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            label = QLabel()
            label.setPixmap(pixmap.scaled(320, 240, Qt.KeepAspectRatio))
            layout_target.addWidget(label)

            # 保存标注图片
            save_img_path = save_dir_target / Path(img_path).name
            cv2.imwrite(str(save_img_path), annotated)

            # 保存TXT文件
            save_txt_path = save_img_path.with_suffix(".txt")
            save_results_txt(r, save_txt_path)

            # 生成识别信息
            species_list = [f"{names[int(box.cls.cpu().numpy()[0])]}: {float(box.conf.cpu().numpy()[0])*100:.1f}%"
                            for box in r.boxes]
            species_str = "; ".join(species_list) if species_list else "无检测目标"
            self.info_list.addItem(f"{Path(img_path).name}: {species_str}")

        self.info_list.addItem(f"\n✅ 所有图片已分类保存到 {save_root}")

# -----------------------------
# 运行应用
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoRecognizer()
    window.show()
    sys.exit(app.exec_())
