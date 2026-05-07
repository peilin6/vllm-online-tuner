#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# =============================================================================
# run_a6000_baseline.sh — A6000 + Qwen3-8B 一键 baseline 实验
# 用法: bash scripts/experiment/run_a6000_baseline.sh [config_path] [workload_path]
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${1:-${PROJECT_DIR}/configs/experiments/baseline_a6000_0.json}"
WORKLOAD="${2:-${PROJECT_DIR}/configs/workloads/workload_baseline.json}"
VENV="${VLLM_VENV:-${HOME}/vllm-venv-a6000}"
PORT="$(python3 -c "import json; c=json.load(open('${CONFIG}', encoding='utf-8')); print(c['server']['port'])")"

export no_proxy="*"
export NO_PROXY="*"

echo "============================================================"
echo " A6000 + Qwen3-8B Baseline 实验"
echo " 项目目录: ${PROJECT_DIR}"
echo " 配置文件: ${CONFIG}"
echo " Workload: ${WORKLOAD}"
echo " 虚拟环境: ${VENV}"
echo "============================================================"

if [ ! -d "${VENV}" ]; then
    echo "[ERROR] 未找到虚拟环境: ${VENV}"
    echo "[HINT] 先按 Week 5 Task 5.1 创建 ~/vllm-venv-a6000 并安装依赖。"
    exit 1
fi

source "${VENV}/bin/activate"

echo "[Step 1/5] 检查 GPU 与 Python 环境"
nvidia-smi
python3 - <<'PY'
import torch
try:
    import vllm
    vllm_version = vllm.__version__
except Exception as exc:
    vllm_version = f"导入失败: {exc}"
print("Python 环境检查:")
print("  CUDA 可用:", torch.cuda.is_available())
print("  GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("  vLLM:", vllm_version)
PY

echo "[Step 2/5] 若已有 vLLM 服务则停止"
if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    bash "${PROJECT_DIR}/scripts/server/stop_server.sh" || true
fi

echo "[Step 3/5] 启动 vLLM 服务"
bash "${PROJECT_DIR}/scripts/server/start_server.sh" "${CONFIG}"

echo "[Step 4/5] 验证服务"
python3 "${PROJECT_DIR}/scripts/verify/verify_server.py" --port "${PORT}"

echo "[Step 5/5] 运行 baseline 压测"
python3 "${PROJECT_DIR}/benchmarks/run_benchmark.py" \
    --config "${CONFIG}" \
    --workload "${WORKLOAD}" \
    --host 127.0.0.1 \
    --experiment-id baseline_a6000_0 \
    --output-dir "${PROJECT_DIR}/results"

echo "============================================================"
echo " A6000 baseline 完成"
echo " 摘要: ${PROJECT_DIR}/results/baseline_a6000_0/summary.txt"
echo " 停止服务: bash ${PROJECT_DIR}/scripts/server/stop_server.sh"
echo "============================================================"