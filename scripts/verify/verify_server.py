#!/usr/bin/env python3
"""
verify_server.py — 验证 vLLM 服务可用性

用法: python scripts/verify/verify_server.py [--host HOST] [--port PORT]

验证项:
  1. /health 端点可达
  2. /v1/models 返回已加载模型
  3. /v1/chat/completions 能正常生成文本
"""
import argparse
import json
import sys
import time

import requests


def check_health(base_url: str) -> bool:
    """检查服务健康端点"""
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def check_models(base_url: str) -> str | None:
    """获取已加载模型名称"""
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        data = r.json()
        models = data.get("data", [])
        if models:
            return models[0]["id"]
        return None
    except Exception as e:
        print(f"  [WARN] 获取模型列表失败: {e}")
        return None


def check_chat(base_url: str, model_name: str) -> bool:
    """发送一次最小 chat 请求，验证生成能力"""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "请用一句话介绍 vLLM。"}
        ],
        "max_tokens": 64,
        "temperature": 0.7,
    }
    try:
        t0 = time.time()
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        elapsed = time.time() - t0
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        print(f"  模型回复: {content[:120]}...")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  tokens: prompt={usage.get('prompt_tokens', '?')}, "
              f"completion={usage.get('completion_tokens', '?')}")
        return True
    except Exception as e:
        print(f"  [ERROR] Chat 请求失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="验证 vLLM 服务可用性")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"目标服务: {base_url}")
    print("=" * 50)

    # 1. 健康检查
    print("[1/3] 健康检查...")
    if not check_health(base_url):
        print("  [FAIL] 服务不可达，请确认服务已启动")
        sys.exit(1)
    print("  [PASS]")

    # 2. 模型列表
    print("[2/3] 检查已加载模型...")
    model_name = check_models(base_url)
    if not model_name:
        print("  [FAIL] 未发现已加载模型")
        sys.exit(1)
    print(f"  [PASS] 模型: {model_name}")

    # 3. Chat 验证
    print("[3/3] 发送测试请求...")
    if not check_chat(base_url, model_name):
        print("  [FAIL] Chat 请求失败")
        sys.exit(1)
    print("  [PASS]")

    print("=" * 50)
    print("所有验证通过！服务可正常使用。")


if __name__ == "__main__":
    main()
