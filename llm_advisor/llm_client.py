#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_client.py — OpenAI 兼容 LLM 客户端（function-calling + 重试 + 缓存 + 限速）

Task 6.8。封装一次 chat 请求 + 多轮 tool-call loop 的最小可用实现，依赖仅 stdlib
（urllib + json + threading）以便在无 GPU 开发机也能用 mock 服务器单元测试。

支持：
1. chat(messages, tools=None, tool_choice="auto", **gen_args) → dict（OpenAI 响应原文）
2. chat_with_tools(messages, registry, max_rounds=4, **gen_args)
   多轮 function-calling：模型返回 tool_calls → 本地 dispatch → 把结果以 role="tool"
   消息回投，循环直到 finish_reason!="tool_calls" 或达到 max_rounds
3. 指数退避重试（network / 5xx）
4. 内存 LRU 缓存（按 messages+tools+gen_args 哈希）
5. 简单令牌桶限速（每秒 N 次）
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =====================================================================
# 工具：令牌桶限速器
# =====================================================================
class _RateLimiter:
    def __init__(self, rate_per_s: float):
        self.rate = max(0.0, float(rate_per_s))
        self._lock = threading.Lock()
        self._next_available = 0.0

    def acquire(self, time_fn: Callable[[], float] = time.perf_counter) -> float:
        if self.rate <= 0:
            return 0.0
        with self._lock:
            now = time_fn()
            wait = max(0.0, self._next_available - now)
            issue_at = now + wait
            self._next_available = issue_at + 1.0 / self.rate
            return wait


# =====================================================================
# LRU 缓存
# =====================================================================
class _LruCache:
    def __init__(self, capacity: int = 128):
        self.capacity = max(0, int(capacity))
        self._d: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key: str, value: Any):
        if self.capacity == 0:
            return
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)


# =====================================================================
# LlmClient
# =====================================================================
@dataclass
class LlmClientConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout_s: float = 60.0
    max_retries: int = 3
    backoff_base_s: float = 0.5
    cache_capacity: int = 128
    rate_per_s: float = 2.0
    extra_headers: dict = field(default_factory=dict)


# 抽象 transport：单元测试用 fake 替换，避免真 HTTP
Transport = Callable[[str, dict, dict, float], dict]
"""(url, headers, body_json, timeout) → response_json"""


def _default_transport(url: str, headers: dict, body: dict, timeout: float) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


class LlmClient:
    def __init__(
        self,
        config: LlmClientConfig,
        *,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cfg = config
        self._transport = transport or _default_transport
        self._clock = clock
        self._sleep = sleep
        self._cache = _LruCache(config.cache_capacity)
        self._rate = _RateLimiter(config.rate_per_s)

    # ---------- 单轮 chat ----------
    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        use_cache: bool = True,
        **extra,
    ) -> dict:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        body.update(extra)

        cache_key = self._cache_key(body) if use_cache else None
        if cache_key:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        wait = self._rate.acquire(self._clock)
        if wait > 0:
            self._sleep(wait)

        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }
        headers.update(self.cfg.extra_headers)

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._transport(url, headers, body, self.cfg.timeout_s)
                if cache_key:
                    self._cache.put(cache_key, resp)
                return resp
            except urllib.error.HTTPError as e:
                last_err = e
                if 400 <= e.code < 500 and e.code not in (408, 429):
                    raise   # 客户端错误，立即抛出
                logger.warning("LLM HTTPError code=%s attempt=%d", e.code, attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                logger.warning("LLM transport error %r attempt=%d", e, attempt)
            if attempt < self.cfg.max_retries:
                self._sleep(self.cfg.backoff_base_s * (2 ** attempt))
        raise RuntimeError(f"LLM 调用失败 ({self.cfg.max_retries+1} 次): {last_err}")

    # ---------- 多轮 tool-call ----------
    def chat_with_tools(
        self,
        messages: list[dict],
        registry,                     # ToolRegistry duck-typed: openai_tools_schema()/dispatch()
        *,
        max_rounds: int = 4,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> dict:
        """多轮 function-calling 主循环。

        Returns:
            {"messages": 最终对话历史, "tool_trace": [...], "final": 最后一条 assistant 消息}
        """
        msgs = list(messages)
        trace: list[dict] = []
        last_resp = None
        for round_idx in range(max_rounds):
            resp = self.chat(
                msgs,
                tools=registry.openai_tools_schema(),
                temperature=temperature,
                max_tokens=max_tokens,
                use_cache=False,
            )
            last_resp = resp
            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            msgs.append(msg)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls or choice.get("finish_reason") != "tool_calls":
                break
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = registry.dispatch(name, args)
                trace.append({"round": round_idx, "name": name,
                              "arguments": args, "result": result})
                msgs.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        return {"messages": msgs, "tool_trace": trace,
                "final": msgs[-1] if msgs else None,
                "raw_last_response": last_resp}

    # ---------- 内部 ----------
    @staticmethod
    def _cache_key(body: dict) -> str:
        s = json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
