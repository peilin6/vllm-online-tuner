#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# =============================================================================
# bootstrap_a6000.sh — 在 A6000 服务器上初始化本项目环境
# 用法: bash scripts/remote/bootstrap_a6000.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VLLM_VENV:-${HOME}/vllm-venv-a6000}"
MODEL_DIR="${MODEL_DIR:-${HOME}/models/Qwen3-8B}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

export no_proxy="*"
export NO_PROXY="*"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "============================================================"
echo " A6000 项目环境初始化"
echo " 项目目录: ${PROJECT_DIR}"
echo " 虚拟环境: ${VENV}"
echo " 模型目录: ${MODEL_DIR}"
echo "============================================================"

echo "[Step 1/6] 检查 GPU"
nvidia-smi

echo "[Step 2/6] 创建 Python 虚拟环境"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi
if [ ! -d "${VENV}" ]; then
    "${PYTHON_BIN}" -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel

echo "[Step 3/6] 安装 vLLM / PyTorch / Transformers"
python3 -m pip install \
    "vllm==0.6.6.post1" \
    "torch==2.5.1" \
    "transformers==4.49.0"

echo "[Step 4/6] 安装项目依赖"
python3 -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo "[Step 5/6] 下载或校验 Qwen3-8B 模型"
mkdir -p "$(dirname "${MODEL_DIR}")"
if [ -f "${MODEL_DIR}/config.json" ]; then
    echo "[INFO] 模型已存在: ${MODEL_DIR}"
else
    python3 -m pip install "huggingface_hub[cli]"
    huggingface-cli download Qwen/Qwen3-8B --local-dir "${MODEL_DIR}"
fi

echo "[Step 6/6] 环境自检"
python3 - <<'PY'
import os
import torch
import vllm
from transformers import AutoConfig

model_dir = os.path.expanduser("~/models/Qwen3-8B")
print("CUDA 可用:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("vLLM:", vllm.__version__)
c = AutoConfig.from_pretrained(model_dir)
print("模型:", c.model_type, getattr(c, "hidden_size", "N/A"), getattr(c, "num_hidden_layers", "N/A"))
PY

echo "============================================================"
echo " 初始化完成"
echo " 下一步: bash scripts/experiment/run_a6000_baseline.sh"
echo "============================================================"