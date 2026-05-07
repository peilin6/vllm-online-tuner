#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm_advisor.config — 从环境变量 / .env 文件构建 LlmClient。

用法:
    from llm_advisor.config import build_llm_client
    client = build_llm_client()    # 读 .env / 环境变量

环境变量优先级: 进程 env > .env 文件
- VTA_LLM_BASE_URL  默认 https://api.deepseek.com/v1
- VTA_LLM_API_KEY   必填
- VTA_LLM_MODEL     默认 deepseek-chat
"""
from __future__ import annotations

import os
from pathlib import Path

from .llm_client import LlmClient, LlmClientConfig


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """轻量 .env 解析器：KEY=VALUE，跳过注释/空行；不覆盖已设置的环境变量。"""
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        loaded[k] = v
        os.environ.setdefault(k, v)
    return loaded


def build_llm_client(
    *,
    dotenv_path: str | Path = ".env",
    rate_per_s: float = 2.0,
    timeout_s: float = 60.0,
    max_retries: int = 3,
) -> LlmClient:
    """从 env 构造 LlmClient；缺 key 时抛 RuntimeError。"""
    load_dotenv(dotenv_path)
    api_key = os.environ.get("VTA_LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "缺少 VTA_LLM_API_KEY；请在 .env 中设置或 export VTA_LLM_API_KEY=..."
        )
    cfg = LlmClientConfig(
        base_url=os.environ.get("VTA_LLM_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=api_key,
        model=os.environ.get("VTA_LLM_MODEL", "deepseek-chat"),
        timeout_s=timeout_s,
        max_retries=max_retries,
        rate_per_s=rate_per_s,
    )
    return LlmClient(cfg)
