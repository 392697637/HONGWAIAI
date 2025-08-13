# train_animals.py
from pathlib import Path
from src.train_model import train_model

def train_two_stages(base_dir: Path, device='cpu'):
    """
    两阶段训练：
    1️⃣ 人 + 风景
    2️⃣ 动物
    返回：
        best_model_hl   -> 第一阶段权重路径
        best_model_animal -> 第二阶段权重路径
    """
    # -----------------------------
    # 1️⃣ 第一阶段训练：人 + 风景
    # -----------------------------
    data_yaml_hl = base_dir / 'dataset' / 'data_human_landscape.yaml'
    train_cfg_step1 = {
        'model_type': 'yolov8n.pt',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'save_dir': str(base_dir / 'models'),
        'workers': 4,
        'patience': 20,
        'optimizer': 'SGD',
        'pretrained': True
    }
    best_model_hl = train_model(data_yaml_hl, train_cfg_step1)

    # -----------------------------
    # 2️⃣ 第二阶段训练：动物
    # -----------------------------
    data_yaml_animal = base_dir / 'dataset' / 'data_animal.yaml'
    train_cfg_step2 = train_cfg_step1.copy()
    train_cfg_step2['pretrained'] = best_model_hl  # 使用第一阶段权重
    train_cfg_step2['name'] = 'animal_dataset'

    best_model_animal = train_model(data_yaml_animal, train_cfg_step2)

    return best_model_hl, best_model_animal

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    train_two_stages(base_dir, device='cpu')
