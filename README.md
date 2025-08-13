YOLOv8_Custom_Project/
│
├── dataset/                 # 数据集与配置
│   ├── data.yaml             # 类别、训练参数、推理参数等配置
│   ├── images/               # 原始图片
│   │   ├── train/            # 训练图片
│   │   └── val/              # 验证图片
│   ├── labels/               # YOLO 格式标签
│   │   ├── train/               # YOLO 标签
│   │   └── val/
│   └── ...                   
│
├── models/                   # 模型保存目录
│   └── custom_dataset/       # 训练输出（权重、日志）
│           └── best.pt       # 训练得到的模型
│
├── results/                  # 推理结果
│   ├── human/                # 人类类别图片
│   ├── animal/               # 动物类别图片
│   ├── landscape/            # 风景/无目标
│   └── 识别结果.txt           # 推理结果汇总
│
├── src/                      # 源码目录
│   ├── config_loader.py      # 负责读取并解析 dataset/data.yaml
│   ├── utils.py              # 工具函数（目录、保存txt、找best.pt等）
│   ├── train_model.py        # 训练模型
│   └── infer_batch.py        # 批量推理
│   └── infer_video.py        # 视频推理
│
├── web_app/                # Web部署模块
│   ├── app.py
│   ├── static/
│   └── templates/
│
├── run_train.py               # 训练入口脚本
├── run_infer.py               # 推理入口脚本
├── run_video.py               # 视频推理入口
└── requirements.txt           # 依赖包


--------------------------------------------------------------------------------------------------------------------------
src/config_loader.py
负责读取并解析 dataset/data.yaml
返回类别映射、分类规则、训练参数、推理参数、项目根路径

src/utils.py
ensure_dir(path)：保证目录存在
save_results_txt(results, save_txt_path)：保存检测框结果到 TXT
find_best_pt(start_dir)：递归搜索最新 best.pt

src/train_model.py
从配置文件读取训练参数
自动检测 GPU / CPU
训练 YOLOv8 模型
自动找到最新 best.pt 并返回路径

src/infer_batch.py
读取配置和 best.pt
批量推理图片
按类别保存图片（human / animal / landscape）
保存检测框 TXT 和推理结果汇总文件
 
------------------------------------------------------------------------------------------------------------------------
视频推理
python run_video.py

Web API
uvicorn web_app.app:app --reload --host 0.0.0.0 --port 8000
上传图片到 POST /detect_image
获取结果图片 GET /get_result/{filename}

