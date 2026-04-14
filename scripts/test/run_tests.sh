#!/bin/bash
# -*- coding: utf-8 -*-
# run_tests.sh — 运行所有单元测试

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

source /home/lpl/vllm-venv/bin/activate
export no_proxy="*"

echo "=============================================="
echo " 单元测试"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

python3 -m pytest tests/ -v --tb=short 2>/dev/null || python3 -m unittest discover -s tests -p "test_*.py" -v
