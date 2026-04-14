#!/usr/bin/env bash
# =============================================================================
# start_server.sh — 启动 vLLM OpenAI-compatible API 服务
# 用法: bash scripts/server/start_server.sh [config_path]
# 默认配置: configs/experiments/baseline_0.json
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${1:-${PROJECT_DIR}/configs/experiments/baseline_0.json}"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/vllm_server_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

# ---------- 从 JSON 配置中解析参数 ----------
MODEL=$(python3 -c "import json; c=json.load(open('${CONFIG}')); print(c['model']['name'])")
HOST=$(python3 -c "import json; c=json.load(open('${CONFIG}')); print(c['server']['host'])")
PORT=$(python3 -c "import json; c=json.load(open('${CONFIG}')); print(c['server']['port'])")
TP=$(python3 -c "import json; c=json.load(open('${CONFIG}')); print(c['server']['tensor_parallel_size'])")
GPU_UTIL=$(python3 -c "import json; c=json.load(open('${CONFIG}')); print(c['server']['gpu_memory_utilization'])")
MAX_MODEL_LEN=$(python3 -c "import json; c=json.load(open('${CONFIG}')); print(c['model']['max_model_len'])")
DTYPE=$(python3 -c "import json; c=json.load(open('${CONFIG}')); print(c['model']['dtype'])")
TRUST_REMOTE=$(python3 -c "import json; c=json.load(open('${CONFIG}')); print('--trust-remote-code' if c['model'].get('trust_remote_code') else '')")

# ---------- 本地模型优先：本地有则用本地，没有则在线下载 ----------
if [[ "${MODEL}" == */* ]]; then
    LOCAL_NAME=$(basename "${MODEL}")
    LOCAL_PATH="${HOME}/models/${LOCAL_NAME}"
    if [ -d "${LOCAL_PATH}" ] && [ -f "${LOCAL_PATH}/config.json" ]; then
        echo "[INFO] 发现本地模型: ${LOCAL_PATH}，使用离线模式"
        MODEL="${LOCAL_PATH}"
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
    else
        echo "[INFO] 本地未找到模型，将从 HuggingFace 在线下载: ${MODEL}"
    fi
fi

echo "============================================"
echo " vLLM 服务启动"
echo " 模型: ${MODEL}"
echo " 地址: ${HOST}:${PORT}"
echo " TP:   ${TP}"
echo " GPU利用: ${GPU_UTIL}"
echo " 日志: ${LOG_FILE}"
echo "============================================"

# ---------- 启动服务 (后台运行，nohup 确保脚本退出后服务存活) ----------
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype "${DTYPE}" \
    ${TRUST_REMOTE} \
    > "${LOG_FILE}" 2>&1 &
disown

SERVER_PID=$!
echo "[INFO] vLLM 服务 PID: ${SERVER_PID}"
echo "${SERVER_PID}" > "${LOG_DIR}/server.pid"

# ---------- 等待服务就绪 ----------
echo "[INFO] 等待服务就绪 (最多 300 秒)..."
MAX_WAIT=300
WAITED=0
while [ ${WAITED} -lt ${MAX_WAIT} ]; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[OK] 服务已就绪! (耗时 ${WAITED}s)"
        echo ""
        echo "  API 地址: http://localhost:${PORT}/v1"
        echo "  健康检查: http://localhost:${PORT}/health"
        echo "  停止服务: kill \$(cat ${LOG_DIR}/server.pid)"
        exit 0
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  ... 已等待 ${WAITED}s"
done

echo "[ERROR] 服务启动超时 (${MAX_WAIT}s)，请检查日志: ${LOG_FILE}"
exit 1
