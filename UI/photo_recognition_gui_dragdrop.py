# photo_recognition_gui_dragdrop.py
# ---------------------------------
# 拖拽目录批量识别 + 实时刷新界面
# 使用 PyQt5 + YOLOv8
# ---------------------------------

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QTabWidget, QListWidget, QScrollArea, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
from ultralytics import YOLO
import os

# -----------------------------
# 配置部分
# -----------------------------
MODEL_PATH = "models/custom_dataset/weights/best.pt"
DEVICE = "0"  # GPU: "0", CPU: "cpu"

human_classes = {0}
animal_classes = {1, 2, 3}
landscape_classes = {4,5,6}
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
# 批量识别线程
# -----------------------------
class InferThread(QThread):
    progress_signal = pyqtSignal(str, QPixmap)  # 文件名, pixmap
    finished_signal = pyqtSignal(str)           # 保存目录

    def __init__(self, folder_path, model):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.model = model

    def run(self):
        # 输出目录
        save_root = self.folder_path / "results"
        human_dir = save_root / "human"
        animal_dir = save_root / "animal"
        landscape_dir = save_root / "landscape"
        for d in [human_dir, animal_dir, landscape_dir]:
            ensure_dir(d)

        # 获取图片
        image_files = [f for f in self.folder_path.glob("*") if f.suffix.lower() in (".jpg",".jpeg",".png",".bmp")]
        for img_path in image_files:
            results = self.model.predict(str(img_path), device=DEVICE, conf=0.25, verbose=False)
            r = results[0]

            annotated = r.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

            detected_cls = set(int(box.cls.cpu().numpy()[0]) for box in r.boxes)
            if detected_cls & human_classes:
                save_dir_target = human_dir
            elif detected_cls & animal_classes:
                save_dir_target = animal_dir
            else:
                save_dir_target = landscape_dir

            # 保存图片
            save_img_path = save_dir_target / img_path.name
            cv2.imwrite(str(save_img_path), annotated)

            # 保存txt
            save_txt_path = save_img_path.with_suffix(".txt")
            save_results_txt(r, save_txt_path)

            # 转为QPixmap发送信号
            height, width, channel = annotated.shape
            bytes_per_line = 3 * width
            q_img = QImage(annotated.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(320, 240, Qt.KeepAspectRatio)

            self.progress_signal.emit(img_path.name, pixmap)

        self.finished_signal.emit(str(save_root))

# -----------------------------
# 主窗口
# -----------------------------
class PhotoRecognizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("拖拽目录批量识别")
        self.setGeometry(100, 100, 1200, 700)
        self.setAcceptDrops(True)  # 支持拖拽

        self.model = YOLO(MODEL_PATH)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 分类Tab
        self.tab_widget = QTabWidget()
        self.human_tab = QScrollArea()
        self.animal_tab = QScrollArea()
        self.landscape_tab = QScrollArea()

        self.human_container = QFrame()
        self.animal_container = QFrame()
        self.landscape_container = QFrame()

        self.human_layout = QHBoxLayout()
        self.animal_layout = QHBoxLayout()
        self.landscape_layout = QHBoxLayout()

        self.human_container.setLayout(self.human_layout)
        self.animal_container.setLayout(self.animal_layout)
        self.landscape_container.setLayout(self.landscape_layout)

        self.human_tab.setWidgetResizable(True)
        self.human_tab.setWidget(self.human_container)
        self.animal_tab.setWidgetResizable(True)
        self.animal_tab.setWidget(self.animal_container)
        self.landscape_tab.setWidgetResizable(True)
        self.landscape_tab.setWidget(self.landscape_container)

        self.tab_widget.addTab(self.human_tab, "Human")
        self.tab_widget.addTab(self.animal_tab, "Animal")
        self.tab_widget.addTab(self.landscape_tab, "Landscape")

        layout.addWidget(self.tab_widget)

        # 信息列表
        self.info_list = QListWidget()
        layout.addWidget(self.info_list)

        self.setLayout(layout)

    # -----------------------------
    # 拖拽事件
    # -----------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            folder_path = url.toLocalFile()
            if Path(folder_path).is_dir():
                self.start_infer(folder_path)

    # -----------------------------
    # 开始识别
    # -----------------------------
    def start_infer(self, folder_path):
        # 清空旧布局
        for layout in [self.human_layout, self.animal_layout, self.landscape_layout]:
            for i in reversed(range(layout.count())):
                widget = layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
        self.info_list.clear()

        # 启动线程
        self.thread = InferThread(folder_path, self.model)
        self.thread.progress_signal.connect(self.update_ui)
        self.thread.finished_signal.connect(self.finish_ui)
        self.thread.start()

    def update_ui(self, img_name, pixmap):
        # 根据文件名简单分类显示
        if any(int(box_id) in human_classes for box_id in range(len(names))):
            self.human_layout.addWidget(QLabel(pixmap=pixmap))
        elif any(int(box_id) in animal_classes for box_id in range(len(names))):
            self.animal_layout.addWidget(QLabel(pixmap=pixmap))
        else:
            self.landscape_layout.addWidget(QLabel(pixmap=pixmap))
        self.info_list.addItem(f"{img_name} 识别完成")

    def finish_ui(self, save_root):
        self.info_list.addItem(f"\n✅ 所有图片已分类保存到 {save_root}")

# -----------------------------
# 运行应用
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoRecognizer()
    window.show()
    sys.exit(app.exec_())
