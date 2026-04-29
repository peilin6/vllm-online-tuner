#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_llm_client.py — Task 6.8 LlmClient 单元测试。

全部用 fake transport 替换网络层，不发真请求。
"""
from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_advisor.llm_client import (
    LlmClient, LlmClientConfig, _LruCache, _RateLimiter,
)


# ---------- fake clock & transport ----------

class _FakeClock:
    def __init__(self, t0=0.0):
        self.t = t0
    def __call__(self):
        return self.t


class _FakeSleep:
    def __init__(self):
        self.sleeps: list[float] = []
    def __call__(self, s):
        self.sleeps.append(s)


def _make_response(content="hello", tool_calls=None, finish_reason="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
        msg["content"] = None
    return {"choices": [{"message": msg, "finish_reason": finish_reason}]}


# ---------- _RateLimiter ----------

def test_rate_limiter_zero_no_wait():
    r = _RateLimiter(0.0)
    clock = _FakeClock()
    assert r.acquire(clock) == 0.0
    assert r.acquire(clock) == 0.0


def test_rate_limiter_enforces_interval():
    r = _RateLimiter(2.0)  # 0.5s 间隔
    clock = _FakeClock(0.0)
    assert r.acquire(clock) == 0.0
    clock.t = 0.1
    w = r.acquire(clock)
    assert w == pytest.approx(0.4, abs=1e-6)


# ---------- _LruCache ----------

def test_lru_basic():
    c = _LruCache(2)
    c.put("a", 1); c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)  # 应淘汰 b（a 刚被访问）
    assert c.get("b") is None
    assert c.get("c") == 3


def test_lru_capacity_zero_disabled():
    c = _LruCache(0)
    c.put("a", 1)
    assert c.get("a") is None


# ---------- chat: 基础 ----------

def _client(transport, **cfg_overrides):
    defaults = dict(api_key="test", rate_per_s=0.0, max_retries=2, backoff_base_s=0.0)
    defaults.update(cfg_overrides)
    cfg = LlmClientConfig(**defaults)
    sleep = _FakeSleep()
    cli = LlmClient(cfg, transport=transport, clock=_FakeClock(), sleep=sleep)
    return cli, sleep


def test_chat_calls_transport_and_returns_response():
    seen = {}
    def fake(url, headers, body, timeout):
        seen["url"] = url
        seen["body"] = body
        return _make_response("ok")
    cli, _ = _client(fake)
    out = cli.chat([{"role": "user", "content": "hi"}], use_cache=False)
    assert out["choices"][0]["message"]["content"] == "ok"
    assert seen["url"].endswith("/chat/completions")
    assert seen["body"]["model"] == "gpt-4o-mini"
    assert "tools" not in seen["body"]


def test_chat_includes_tools_when_provided():
    seen = {}
    def fake(url, headers, body, timeout):
        seen["body"] = body
        return _make_response()
    cli, _ = _client(fake)
    cli.chat([{"role": "user", "content": "x"}],
             tools=[{"type": "function", "function": {"name": "f"}}],
             use_cache=False)
    assert "tools" in seen["body"]
    assert seen["body"]["tool_choice"] == "auto"


# ---------- 缓存 ----------

def test_chat_cache_hit_skips_transport():
    calls = {"n": 0}
    def fake(url, headers, body, timeout):
        calls["n"] += 1
        return _make_response("once")
    cli, _ = _client(fake)
    msgs = [{"role": "user", "content": "ping"}]
    a = cli.chat(msgs)
    b = cli.chat(msgs)
    assert a == b
    assert calls["n"] == 1  # 第二次命中缓存


def test_chat_use_cache_false_bypasses():
    calls = {"n": 0}
    def fake(url, headers, body, timeout):
        calls["n"] += 1
        return _make_response()
    cli, _ = _client(fake)
    cli.chat([{"role": "user", "content": "x"}], use_cache=False)
    cli.chat([{"role": "user", "content": "x"}], use_cache=False)
    assert calls["n"] == 2


# ---------- 重试 ----------

def test_chat_retries_on_transport_error_then_succeeds():
    state = {"calls": 0}
    def fake(url, headers, body, timeout):
        state["calls"] += 1
        if state["calls"] < 3:
            raise urllib.error.URLError("boom")
        return _make_response("ok")
    cli, sleeper = _client(fake, max_retries=3)
    out = cli.chat([{"role": "user", "content": "x"}], use_cache=False)
    assert out["choices"][0]["message"]["content"] == "ok"
    assert state["calls"] == 3
    assert len(sleeper.sleeps) >= 2  # 重试间至少 sleep 两次


def test_chat_retries_exhausted_raises():
    def fake(url, headers, body, timeout):
        raise urllib.error.URLError("always fail")
    cli, _ = _client(fake, max_retries=2)
    with pytest.raises(RuntimeError):
        cli.chat([{"role": "user", "content": "x"}], use_cache=False)


def test_chat_4xx_not_retried():
    state = {"calls": 0}
    def fake(url, headers, body, timeout):
        state["calls"] += 1
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)
    cli, _ = _client(fake, max_retries=5)
    with pytest.raises(urllib.error.HTTPError):
        cli.chat([{"role": "user", "content": "x"}], use_cache=False)
    assert state["calls"] == 1


def test_chat_5xx_retried():
    state = {"calls": 0}
    def fake(url, headers, body, timeout):
        state["calls"] += 1
        if state["calls"] < 2:
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)
        return _make_response()
    cli, _ = _client(fake, max_retries=3)
    cli.chat([{"role": "user", "content": "x"}], use_cache=False)
    assert state["calls"] == 2


# ---------- 限速 ----------

def test_chat_invokes_rate_limiter_sleep():
    def fake(url, headers, body, timeout):
        return _make_response()
    cfg = LlmClientConfig(api_key="t", rate_per_s=2.0, max_retries=0)
    sleeper = _FakeSleep()
    clock = _FakeClock()
    cli = LlmClient(cfg, transport=fake, clock=clock, sleep=sleeper)
    cli.chat([{"role": "user", "content": "1"}], use_cache=False)
    cli.chat([{"role": "user", "content": "2"}], use_cache=False)
    # 第二次应该触发限速 sleep
    assert any(s > 0 for s in sleeper.sleeps)


# ---------- chat_with_tools 多轮 ----------

class _FakeRegistry:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
    def openai_tools_schema(self):
        return [{"type": "function", "function": {"name": "echo",
                 "description": "", "parameters": {"type": "object", "properties": {}}}}]
    def dispatch(self, name, arguments):
        self.calls.append((name, arguments))
        return {"ok": True, "echoed": arguments}


def test_chat_with_tools_loops_then_finalizes():
    """模型先要求一次 tool_call，再返回普通回答。"""
    responses = iter([
        _make_response(
            tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "echo", "arguments": json.dumps({"x": 1})},
            }],
            finish_reason="tool_calls",
        ),
        _make_response("done", finish_reason="stop"),
    ])
    def fake(url, headers, body, timeout):
        return next(responses)
    cli, _ = _client(fake)
    reg = _FakeRegistry()
    out = cli.chat_with_tools([{"role": "user", "content": "go"}], reg, max_rounds=3)
    assert out["final"]["content"] == "done"
    assert len(out["tool_trace"]) == 1
    assert out["tool_trace"][0]["name"] == "echo"
    assert out["tool_trace"][0]["arguments"] == {"x": 1}
    # tool 角色消息应已注入
    roles = [m.get("role") for m in out["messages"]]
    assert "tool" in roles


def test_chat_with_tools_respects_max_rounds():
    """模型一直要求 tool_call → 应在 max_rounds 后停止。"""
    def fake(url, headers, body, timeout):
        return _make_response(
            tool_calls=[{"id": "x", "type": "function",
                         "function": {"name": "echo", "arguments": "{}"}}],
            finish_reason="tool_calls",
        )
    cli, _ = _client(fake)
    reg = _FakeRegistry()
    out = cli.chat_with_tools([{"role": "user", "content": "go"}], reg, max_rounds=2)
    # 2 轮 → 调用 2 次工具
    assert len(out["tool_trace"]) == 2
    assert len(reg.calls) == 2


def test_chat_with_tools_handles_invalid_arguments_json():
    responses = iter([
        _make_response(
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "echo", "arguments": "not-json"}}],
            finish_reason="tool_calls",
        ),
        _make_response("ok", finish_reason="stop"),
    ])
    def fake(url, headers, body, timeout):
        return next(responses)
    cli, _ = _client(fake)
    reg = _FakeRegistry()
    out = cli.chat_with_tools([{"role": "user", "content": "go"}], reg, max_rounds=2)
    assert out["tool_trace"][0]["arguments"] == {}
    assert reg.calls[0] == ("echo", {})
