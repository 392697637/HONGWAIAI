# run_infer.py
# 🔍 推理入口脚本
# 从 src/infer_batch.py 调用批量推理

from src.infer_batch import batch_inference

if __name__ == "__main__":
    print("\n=== YOLOv8 批量推理 ===")
    batch_inference()
    print("\n✅ 批量推理完成！结果已保存到 results 目录。")
