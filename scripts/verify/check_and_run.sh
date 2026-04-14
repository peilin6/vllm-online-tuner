#!/bin/bash
# check_and_run.sh — 检查服务状态、启动服务（如需）、运行全套测试
set -euo pipefail

source ~/vllm-venv/bin/activate
cd /mnt/d/vlllm

export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""

echo "=========================================="
echo " [1/6] 检查 vLLM 服务状态"
echo "=========================================="

set +e
python3 -c "
import requests
try:
    r = requests.get('http://127.0.0.1:8000/health', timeout=3)
    print('SERVICE_STATUS: RUNNING (HTTP ' + str(r.status_code) + ')')
except Exception as e:
    print('SERVICE_STATUS: NOT_RUNNING (' + str(e)[:60] + ')')
    exit(1)
"
SERVICE_UP=$?
set -e

if [ $SERVICE_UP -ne 0 ]; then
    echo ""
    echo "[INFO] 服务未运行，正在启动..."
    echo "[INFO] 请在另一个终端手动执行: bash /mnt/d/vlllm/scripts/server/launch_server.sh"
    echo "[INFO] 或者使用后台启动方式..."

    # 后台启动服务
    MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct-AWQ"
    LOG_FILE="/mnt/d/vlllm/logs/vllm_server_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p /mnt/d/vlllm/logs

    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "Qwen/Qwen2.5-3B-Instruct-AWQ" \
        --host "0.0.0.0" \
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 2048 \
        --dtype auto \
        --quantization awq \
        --trust-remote-code \
        --swap-space 1 \
        > "${LOG_FILE}" 2>&1 &

    SERVER_PID=$!
    echo "[INFO] 服务 PID: ${SERVER_PID}"
    echo "${SERVER_PID}" > /mnt/d/vlllm/logs/server.pid
    echo "[INFO] 日志: ${LOG_FILE}"

    # 等待服务就绪
    echo "[INFO] 等待服务就绪 (最长 180 秒)..."
    READY=0
    for i in $(seq 1 36); do
        sleep 5
        set +e
        python3 -c "
import requests
try:
    r = requests.get('http://127.0.0.1:8000/health', timeout=3)
    if r.status_code == 200:
        exit(0)
except:
    pass
exit(1)
"
        if [ $? -eq 0 ]; then
            echo "  服务已就绪! (等待 $((i*5)) 秒)"
            READY=1
            set -e
            break
        fi
        set -e
        echo "  ... 等待中 ($((i*5))s)"
    done

    if [ $READY -ne 1 ]; then
        echo "[FATAL] 服务启动超时，请检查日志: ${LOG_FILE}"
        tail -50 "${LOG_FILE}"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo " [2/6] 验证服务可用性"
echo "=========================================="
python3 scripts/verify/verify_server.py --host 127.0.0.1 --port 8000

echo ""
echo "=========================================="
echo " [3/6] 采集系统信息"
echo "=========================================="
python3 scripts/tools/collect_sysinfo.py --output results/sysinfo.json

echo ""
echo "=========================================="
echo " [4/6] 运行基准压测 (50 请求, 并发 1)"
echo "=========================================="
python3 benchmarks/run_benchmark.py \
    --config configs/baseline_0.json \
    --host 127.0.0.1 \
    --port 8000 \
    --output-dir results

echo ""
echo "=========================================="
echo " [5/6] 可视化输出速度 (对比模式)"
echo "=========================================="
python3 benchmarks/visualize_speed.py \
    --host 127.0.0.1 \
    --port 8000 \
    --compare \
    2>&1 | tee /mnt/d/vlllm/logs/visualize_speed.log

echo ""
echo "=========================================="
echo " [6/6] 检查结果完整性"
echo "=========================================="
echo "--- results/ 目录内容 ---"
ls -lh results/
echo ""
echo "--- logs/ 目录内容 ---"
ls -lh logs/
echo ""
echo "=========================================="
echo " 全部完成!"
echo "=========================================="
