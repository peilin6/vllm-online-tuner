#!/bin/bash
# =============================================================================
# launch_server.sh — 在 WSL2 中启动 vLLM 服务（使用本地模型）
# 用法: 在 WSL2 中执行 bash /mnt/d/vlllm/scripts/server/launch_server.sh
# =============================================================================
set -euo pipefail

source ~/vllm-venv/bin/activate

MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct-AWQ"
HOST="0.0.0.0"
PORT=8000
GPU_MEM_UTIL=0.90
MAX_MODEL_LEN=2048
LOG_FILE="/mnt/d/vlllm/logs/vllm_server_$(date +%Y%m%d_%H%M%S).log"

mkdir -p /mnt/d/vlllm/logs

echo "============================================"
echo " vLLM 服务启动"
echo " 模型: ${MODEL_PATH}"
echo " 地址: ${HOST}:${PORT}"
echo " 显存: ${GPU_MEM_UTIL}"
echo " 日志: ${LOG_FILE}"
echo "============================================"

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "Qwen/Qwen2.5-3B-Instruct-AWQ" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype auto \
    --quantization awq \
    --trust-remote-code \
    --swap-space 1 \
    2>&1 | tee "${LOG_FILE}"
