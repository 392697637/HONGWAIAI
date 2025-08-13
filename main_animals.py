from pathlib import Path  # 导入 Path，用于处理文件路径

# 导入自定义模块
from src.config_loader import load_config          # 读取 dataset/data.yaml 配置文件
from src.train_animals import train_two_stages    # 两阶段训练函数：第一阶段（人+风景），第二阶段（动物）
from src.infer_batch import batch_infer           # 批量推理函数，用训练好的模型检测图片

if __name__ == "__main__":
    # -----------------------------
    # 1️⃣ 获取项目根目录
    # -----------------------------
    base_dir = Path(__file__).resolve().parent   # 获取当前脚本所在目录，用于构建数据集、模型、结果路径

    # -----------------------------
    # 2️⃣ 读取配置文件
    # 返回：
    # data_yaml          -> 配置文件路径
    # names              -> 类别ID -> 类别名称映射
    # categories         -> 分类规则（human/animal/landscape）
    # human_classes      -> 人类类别ID集合
    # animal_classes     -> 动物类别ID集合
    # landscape_classes  -> 风景/无目标类别ID集合
    # train_cfg          -> 训练参数字典
    # infer_cfg          -> 推理参数字典
    # device             -> 训练/推理设备（GPU/CPU）
    # -----------------------------
    data_yaml, names, categories, human_classes, animal_classes, landscape_classes, train_cfg, infer_cfg, device = load_config(base_dir)
    print("训练/推理设备:", device)  # 打印当前使用的设备（CPU/GPU）

    # -----------------------------
    # 3️⃣ 执行两阶段训练
    # 第一阶段：训练人 + 风景类别
    # 第二阶段：训练动物类别，使用第一阶段权重继续训练
    # 返回：
    # best_model_hl      -> 第一阶段训练完成的模型权重
    # best_model_animal  -> 第二阶段训练完成的模型权重
    # -----------------------------
    best_model_hl, best_model_animal = train_two_stages(base_dir, device=device)

    # -----------------------------
    # 4️⃣ 使用第二阶段训练模型（动物模型）进行批量推理
    # 将输入目录中的图片进行检测，并按类别保存结果到输出目录
    # -----------------------------
    batch_infer(
        best_model_animal,      # 使用第二阶段训练好的动物模型权重
        names,                  # 类别名称映射，用于生成可读的识别结果
        human_classes,          # 人类类别ID集合
        animal_classes,         # 动物类别ID集合
        landscape_classes,      # 风景/无目标类别ID集合
        infer_cfg,              # 推理参数（输入/输出路径、置信度阈值、是否绘制检测框等）
        device                  # 设备（CPU/GPU）
    )

    # -----------------------------
    # 5️⃣ 输出完成提示
    # -----------------------------
    print("\n✅ 全流程完成：两阶段训练 + 批量推理")
