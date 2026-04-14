#!/bin/bash
# -*- coding: utf-8 -*-
# run_initial_experiments.sh — 运行 3 个初始实验以获取 few-shot 数据
#
# 使用方法: bash scripts/experiment/run_initial_experiments.sh
# 前提: vLLM 服务已在 8000 端口运行

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

source /home/lpl/vllm-venv/bin/activate
export no_proxy="*"

echo "=============================================="
echo " 初始实验集 (Task 4.6)"
echo " 目标: 获取 few-shot 示例数据"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# 检查 vLLM 服务可用性
echo "检查 vLLM 服务..."
if ! curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health | grep -q "200"; then
    echo "错误: vLLM 服务未在 127.0.0.1:8000 运行"
    echo "请先执行: bash scripts/server/start_server.sh"
    exit 1
fi
echo "vLLM 服务正常 ✓"

# 实验 1: baseline burst 混合长度
echo ""
echo "========== 实验 1/3: baseline burst 混合 =========="
python3 -m benchmarks.run_benchmark \
    --config configs/experiments/baseline_0.json \
    --workload configs/workloads/workload_baseline.json \
    --experiment-id "initial_burst_mixed" \
    --concurrency 4
echo "✓ 实验 1 完成"

# 实验 2: constant rate 2 req/s
echo ""
echo "========== 实验 2/3: constant rate 2 =========="
python3 -m benchmarks.run_benchmark \
    --config configs/experiments/baseline_0.json \
    --workload configs/workloads/workload_rate2.json \
    --experiment-id "initial_rate2" \
    --concurrency 4
echo "✓ 实验 2 完成"

# 实验 3: long only
echo ""
echo "========== 实验 3/3: long only =========="
python3 -m benchmarks.run_benchmark \
    --config configs/experiments/baseline_0.json \
    --workload configs/workloads/workload_long_only.json \
    --experiment-id "initial_long_only" \
    --concurrency 4
echo "✓ 实验 3 完成"

echo ""
echo "=============================================="
echo " 全部初始实验完成"
echo " 结果目录:"
echo "   results/initial_burst_mixed/"
echo "   results/initial_rate2/"
echo "   results/initial_long_only/"
echo ""
echo " 请使用实际数据更新:"
echo "   configs/llm_prompts/few_shot_examples.json"
echo "=============================================="
