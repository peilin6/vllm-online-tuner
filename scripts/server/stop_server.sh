#!/usr/bin/env bash
# =============================================================================
# stop_server.sh — 停止 vLLM 服务
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PID_FILE="${PROJECT_DIR}/logs/server.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "[WARN] PID 文件不存在: ${PID_FILE}"
    echo "  尝试手动查找: ps aux | grep vllm"
    exit 1
fi

PID=$(cat "${PID_FILE}")
if kill -0 "${PID}" 2>/dev/null; then
    echo "[INFO] 正在停止 vLLM 服务 (PID: ${PID})..."
    kill "${PID}"
    sleep 2
    if kill -0 "${PID}" 2>/dev/null; then
        echo "[WARN] 进程未退出，发送 SIGKILL..."
        kill -9 "${PID}"
    fi
    echo "[OK] 服务已停止"
else
    echo "[INFO] 进程 ${PID} 已不存在"
fi

rm -f "${PID_FILE}"
