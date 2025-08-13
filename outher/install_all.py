# install_all.py
# -----------------------------
# 一键创建虚拟环境 + 安装依赖 + 自动检测 GPU
# -----------------------------

import os
import subprocess
import sys
import platform

def run_cmd(cmd):
    """执行 shell 命令"""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("❌ 命令执行失败，请检查错误信息。")
        sys.exit(1)

def main():
    venv_dir = "venv"

    # -----------------------------
    # 1. 创建虚拟环境
    # -----------------------------
    if not os.path.exists(venv_dir):
        print("🔹 创建虚拟环境...")
        run_cmd(f"{sys.executable} -m venv {venv_dir}")
    else:
        print("⚠️ 虚拟环境已存在，跳过创建。")

    # -----------------------------
    # 2. 激活虚拟环境
    # -----------------------------
    system = platform.system()
    if system == "Windows":
        activate_cmd = f"{venv_dir}\\Scripts\\activate"
        pip_exe = f"{venv_dir}\\Scripts\\pip"
    else:  # Linux / Mac
        activate_cmd = f"source {venv_dir}/bin/activate"
        pip_exe = f"{venv_dir}/bin/pip"

    print(f"🔹 使用虚拟环境 pip: {pip_exe}")

    # -----------------------------
    # 3. 升级 pip
    # -----------------------------
    run_cmd(f"{pip_exe} install --upgrade pip")

    # -----------------------------
    # 4. 安装 PyTorch 自动 CPU/GPU
    # -----------------------------
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    if gpu_available:
        print("✅ 检测到 GPU，可安装 CUDA 版本 PyTorch")
        torch_cmd = f"{pip_exe} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        print("⚠️ 未检测到 GPU，将安装 CPU 版本 PyTorch")
        torch_cmd = f"{pip_exe} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    run_cmd(torch_cmd)

    # -----------------------------
    # 5. 安装 YOLOv8 与其他依赖
    # -----------------------------
    deps = [
        "ultralytics>=8.0.0",
        "opencv-python>=4.7.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0"
    ]
    run_cmd(f"{pip_exe} install {' '.join(deps)}")

    # -----------------------------
    # 6. 完成提示
    # -----------------------------
    print("\n✅ 安装完成！")
    print(f"💡 激活虚拟环境：")
    if system == "Windows":
        print(f"{venv_dir}\\Scripts\\activate")
    else:
        print(f"source {venv_dir}/bin/activate")
    print("💡 测试 GPU 支持：")
    print("python -c \"import torch; print(torch.cuda.is_available())\"")

if __name__ == "__main__":
    main()
