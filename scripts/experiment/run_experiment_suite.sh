#!/bin/bash
# -*- coding: utf-8 -*-
# run_experiment_suite.sh — 批量运行实验套件
#
# 使用方法: bash scripts/experiment/run_experiment_suite.sh [axis]
# axis: arrival | length | prefix | all (默认: all)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

source /home/lpl/vllm-venv/bin/activate
export no_proxy="*"

AXIS="${1:-all}"

echo "=============================================="
echo " 实验套件批量运行"
echo " 轴: $AXIS"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

run_experiment() {
    local workload_file="$1"
    local experiment_id="$2"

    echo ""
    echo "----------------------------------------------"
    echo " 运行实验: $experiment_id"
    echo " 配置: $workload_file"
    echo "----------------------------------------------"

    python3 -m benchmarks.run_benchmark \
        --config configs/experiments/baseline_0.json \
        --workload "$workload_file" \
        --experiment-id "$experiment_id" \
        --concurrency 4

    echo " ✓ 实验 $experiment_id 完成"
    echo ""
}

# 到达率轴
run_arrival() {
    echo ""
    echo "========== 到达率轴实验 =========="
    run_experiment "configs/workloads/workload_rate2.json"    "arrival_rate2"
    run_experiment "configs/workloads/workload_rate4.json"    "arrival_rate4"
    run_experiment "configs/workloads/workload_poisson2.json" "arrival_poisson2"
}

# 长度分布轴
run_length() {
    echo ""
    echo "========== 长度分布轴实验 =========="
    run_experiment "configs/workloads/workload_short_only.json" "length_short_only"
    run_experiment "configs/workloads/workload_long_only.json"  "length_long_only"
    run_experiment "configs/workloads/workload_mixed.json"      "length_mixed"
}

# 共享前缀轴
run_prefix() {
    echo ""
    echo "========== 共享前缀轴实验 =========="
    run_experiment "configs/workloads/workload_prefix_0.json"  "prefix_ratio_0"
    run_experiment "configs/workloads/workload_prefix_50.json" "prefix_ratio_50"
    run_experiment "configs/workloads/workload_prefix_90.json" "prefix_ratio_90"
}

case "$AXIS" in
    arrival)
        run_arrival
        ;;
    length)
        run_length
        ;;
    prefix)
        run_prefix
        ;;
    all)
        run_arrival
        run_length
        run_prefix
        ;;
    *)
        echo "错误: 未知轴 '$AXIS'"
        echo "用法: bash scripts/experiment/run_experiment_suite.sh [arrival|length|prefix|all]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo " 全部实验完成！"
echo " 结果目录: results/"
echo "=============================================="
ls -la results/
