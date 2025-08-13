# run_train.py
# 🚀 训练入口脚本
# 从 src/train_model.py 调用训练流程

from src.train_model import train_custom_yolov8_model

if __name__ == "__main__":
    print("\n=== YOLOv8 自定义模型训练 ===")
    best_model_path = train_custom_yolov8_model()
    print(f"\n✅ 训练完成，模型路径：{best_model_path}")
