#!/bin/bash
# download_model.sh — 下载模型到 WSL2 本地目录
set -e

source ~/vllm-venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com

export MODEL_NAME="Qwen/Qwen2.5-3B-Instruct-AWQ"
export LOCAL_DIR="$HOME/models/Qwen2.5-3B-Instruct-AWQ"

echo ">>> 下载模型: $MODEL_NAME"
echo ">>> 目标目录: $LOCAL_DIR"
echo ">>> 镜像源:   $HF_ENDPOINT"
echo ""

python3 << 'PYEOF'
import os
from huggingface_hub import snapshot_download

model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
local_dir = os.environ.get("LOCAL_DIR", os.path.expanduser("~/models/Qwen2.5-3B-Instruct"))

print(f"开始下载 {model_name} ...")
path = snapshot_download(model_name, local_dir=local_dir)
print(f"下载完成: {path}")
PYEOF

echo ""
echo "========================================="
echo " 模型下载完成！"
echo " 路径: $LOCAL_DIR"
echo "========================================="
