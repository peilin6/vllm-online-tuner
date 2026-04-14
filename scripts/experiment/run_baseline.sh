#!/usr/bin/env bash
# =============================================================================
# run_baseline.sh — Baseline 0 一键执行流水线
#
# 顺序: 采集系统信息 → 启动服务 → 等待就绪 → 验证服务 → 运行压测 → 汇总结果
#
# 用法: bash scripts/experiment/run_baseline.sh [config_path]
# 默认配置: configs/experiments/baseline_0.json
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${1:-${PROJECT_DIR}/configs/experiments/baseline_0.json}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs"
RESULT_DIR="${PROJECT_DIR}/results"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

echo "============================================================"
echo " Baseline 0 执行流水线"
echo " 配置: ${CONFIG}"
echo " 时间: ${TIMESTAMP}"
echo "============================================================"

# ---------- Step 1: 采集系统信息 ----------
echo ""
echo "[Step 1/5] 采集系统环境信息..."
python3 "${PROJECT_DIR}/scripts/tools/collect_sysinfo.py" \
    --output "${RESULT_DIR}/sysinfo_${TIMESTAMP}.json"
echo "[OK] 系统信息已保存"

# ---------- Step 2: 启动 vLLM 服务 ----------
echo ""
echo "[Step 2/5] 启动 vLLM 服务..."

PORT=$(python3 -c "import json; c=json.load(open('${CONFIG}')); print(c['server']['port'])")

# 检查是否已有服务在运行
if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "[INFO] 服务已在 localhost:${PORT} 运行，跳过启动"
else
    bash "${PROJECT_DIR}/scripts/server/start_server.sh" "${CONFIG}"
fi

# ---------- Step 3: 验证服务可用 ----------
echo ""
echo "[Step 3/5] 验证服务可用性..."
python3 "${PROJECT_DIR}/scripts/verify/verify_server.py" --port "${PORT}"
echo "[OK] 服务验证通过"

# ---------- Step 4: 运行压测 ----------
echo ""
echo "[Step 4/5] 执行 Baseline 0 压测..."
python3 "${PROJECT_DIR}/benchmarks/run_benchmark.py" \
    --config "${CONFIG}" \
    --prompts "${PROJECT_DIR}/benchmarks/prompts.json" \
    --output-dir "${RESULT_DIR}"
echo "[OK] 压测完成"

# ---------- Step 5: 汇总 ----------
echo ""
echo "[Step 5/5] 汇总结果..."
echo ""
echo "============================================================"
echo " Baseline 0 执行完毕"
echo "============================================================"
echo " 系统信息 : ${RESULT_DIR}/sysinfo_${TIMESTAMP}.json"
echo " 压测结果 : ${RESULT_DIR}/benchmark_*.json"
echo " 结果摘要 : ${RESULT_DIR}/benchmark_*.txt"
echo " 服务日志 : ${LOG_DIR}/vllm_server_*.log"
echo ""
echo " 停止服务: kill \$(cat ${LOG_DIR}/server.pid)"
echo "============================================================"
