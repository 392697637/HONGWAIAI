# main.py
# -----------------------------
# 入口脚本：一键执行训练与批量推理流程
# -----------------------------

from pathlib import Path

# 导入自定义模块
from src.config_loader import load_config    # 读取 dataset/data.yaml 配置
from src.train_model import train_model      # 训练模型
from src.infer_batch import batch_infer      # 批量推理

if __name__ == "__main__":
    # -----------------------------
    # 1. 获取当前项目根目录
    # -----------------------------
    base_dir = Path(__file__).resolve().parent

    # -----------------------------
    # 2. 读取配置文件
    # 返回：
    # data_yaml       -> 配置文件路径
    # names           -> 类别ID -> 名称映射
    # categories      -> 分类规则（human/animal/landscape）
    # human_classes   -> 人类类别集合
    # animal_classes  -> 动物类别集合
    # landscape_classes -> 风景/无目标类别集合
    # train_cfg       -> 训练参数字典
    # infer_cfg       -> 推理参数字典
    # device          -> 训练/推理设备（GPU/CPU）
    # -----------------------------
    data_yaml, names, categories, human_classes, animal_classes, landscape_classes, train_cfg, infer_cfg, device = load_config(base_dir)

    print("训练/推理设备:", device)

    # -----------------------------
    # 3. 训练模型
    # 返回训练完成的 best.pt 路径
    # -----------------------------
    best_model_path = train_model(data_yaml, train_cfg)

    # -----------------------------
    # 4. 批量推理
    # 使用训练好的 best.pt 对图片进行检测
    # 并按类别保存结果
    # -----------------------------
    batch_infer(
        best_model_path,       # 训练完成的模型权重
        names,                 # 类别名称映射
        human_classes,         # 人类类别集合
        animal_classes,        # 动物类别集合
        landscape_classes,     # 风景类别集合
        infer_cfg,             # 推理参数（输入/输出路径、置信度阈值等）
        device                 # 设备（GPU/CPU）
    )

    print("\n✅ 全流程完成：训练 + 批量推理")
