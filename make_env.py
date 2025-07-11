import subprocess
import os


LITE = False  # LITEモードを有効にする場合はTrueに設定


if LITE:
    env_name = ".env_lite"
    requirements_file = "requirements-lite.txt"
else:
    env_name = ".env"
    requirements_file = "requirements.txt"

# 仮想環境を作成
subprocess.run(["python", "-m", "venv", env_name], encoding="utf-8")

# Windowsでの仮想環境のPythonとpipのパス
if os.name == 'nt':  # Windows
    python_exe = os.path.join(env_name, "Scripts", "python.exe")
    pip_exe = os.path.join(env_name, "Scripts", "pip.exe")
else:  # Unix/Linux/macOS
    python_exe = os.path.join(env_name, "bin", "python")
    pip_exe = os.path.join(env_name, "bin", "pip")

# 仮想環境のpipを使ってパッケージをインストール
subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], encoding="utf-8")
subprocess.run([pip_exe, "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu126"], encoding="utf-8")
subprocess.run([pip_exe, "install", "-r", requirements_file], encoding="utf-8")
