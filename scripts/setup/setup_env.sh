#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — 一键初始化 vLLM 实验环境
# 用法: bash scripts/setup/setup_env.sh
# 前置: 已安装 conda 或 python3.10、CUDA 12.1+、nvidia-smi 可用
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# 虚拟环境放在 WSL2 本地文件系统，避免跨文件系统 I/O 慢
VENV_DIR="${HOME}/vllm-venv"
PYTHON_VERSION="3.10"

echo "============================================"
echo " vLLM 实验环境安装"
echo " 项目目录: ${PROJECT_DIR}"
echo "============================================"

# ---------- Step 1: 创建虚拟环境 ----------
if [ -d "${VENV_DIR}" ]; then
    echo "[INFO] 虚拟环境已存在: ${VENV_DIR}，跳过创建"
else
    echo "[Step 1] 创建 Python ${PYTHON_VERSION} 虚拟环境..."
    python3 -m venv "${VENV_DIR}"
    echo "[OK] 虚拟环境创建完成"
fi

# 激活虚拟环境
source "${VENV_DIR}/bin/activate"
echo "[INFO] 当前 Python: $(python --version) @ $(which python)"

# ---------- Step 2: 升级 pip ----------
echo "[Step 2] 升级 pip..."
pip install --upgrade pip

# ---------- Step 3: 安装核心依赖 ----------
echo "[Step 3] 安装核心依赖..."
pip install -r "${PROJECT_DIR}/requirements.txt"

# ---------- Step 4: 验证安装 ----------
echo "[Step 4] 验证关键包版本..."
python -c "
import torch
import vllm
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA ver : {torch.version.cuda}')
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
print(f'  vLLM     : {vllm.__version__}')
"

echo ""
echo "============================================"
echo " 环境安装完成！"
echo " 激活方式: source ~/vllm-venv/bin/activate"
echo "============================================"
