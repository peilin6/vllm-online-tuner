#!/bin/bash
# run_all_tests.sh — 运行全套测试（绕过代理问题）
set -euo pipefail

source ~/vllm-venv/bin/activate
cd /mnt/d/vlllm

# 彻底清除代理设置
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY 2>/dev/null || true
export no_proxy="*"
export NO_PROXY="*"

{
echo "=========================================="
echo " [1/5] 验证服务可用性"
echo "=========================================="
python3 scripts/verify/verify_server.py --host 127.0.0.1 --port 8000

echo ""
echo "=========================================="
echo " [2/5] 采集系统信息"
echo "=========================================="
python3 scripts/tools/collect_sysinfo.py --output results/sysinfo.json

echo ""
echo "=========================================="
echo " [3/5] 运行基准压测 (50 请求, 并发 1)"
echo "=========================================="
python3 benchmarks/run_benchmark.py \
    --config configs/experiments/baseline_0.json \
    --host 127.0.0.1 \
    --port 8000 \
    --output-dir results

echo ""
echo "=========================================="
echo " [4/5] 可视化输出速度 (对比模式)"
echo "=========================================="
python3 benchmarks/visualize_speed.py \
    --host 127.0.0.1 \
    --port 8000 \
    --compare 2>&1 | tee logs/visualize_speed.log

echo ""
echo "=========================================="
echo " [5/5] 检查结果完整性"
echo "=========================================="
echo "--- results/ ---"
ls -lh results/
echo ""
echo "--- logs/ ---"
ls -lh logs/*.log 2>/dev/null || echo "(无日志)"
echo ""
echo "=========================================="
echo " 全部测试完成!"
echo "=========================================="
} 2>&1 | tee /mnt/d/vlllm/logs/all_tests.log
