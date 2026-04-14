#!/bin/bash
# install_all.sh — 一键在 WSL2 中创建环境并安装 vLLM
set -e

VENV=~/vllm-venv

echo ">>> Step 1: 创建虚拟环境"
if [ -d "$VENV" ]; then
    echo "    虚拟环境已存在，跳过创建"
else
    python3 -m venv "$VENV"
    echo "    虚拟环境已创建: $VENV"
fi

echo ">>> Step 2: 激活虚拟环境"
source "$VENV/bin/activate"
echo "    Python: $(python3 --version)"
echo "    pip: $(pip --version)"

echo ">>> Step 3: 升级 pip"
pip install --upgrade pip

echo ">>> Step 4: 安装 vLLM 及依赖"
pip install 'vllm>=0.6.0,<0.7.0' aiohttp requests numpy pandas pynvml tqdm

echo ">>> Step 5: 验证安装"
python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python3 -c "import vllm; print('vLLM:', vllm.__version__)"

echo ""
echo "========================================="
echo " 安装完成！"
echo " 激活: source ~/vllm-venv/bin/activate"
echo "========================================="
