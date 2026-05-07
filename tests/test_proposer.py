#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_proposer.py — Task 7.2 P-LLM Proposer 单元测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_advisor.proposer import propose, _fallback_propose, FALLBACK_PRIORITY
from llm_advisor.schemas import ConfigDelta, DiagnosisResult, StopSignal
from tuner.memory import ExperienceMemory, TrialRecord
from tuner.metrics_parser import TrialMetrics


def _metrics(tput=1000.0):
    return TrialMetrics(
        throughput_tok_per_s=tput, ttft_p95_ms=200.0, tpot_p95_ms=20.0,
        latency_p95_ms=2000.0, kv_cache_usage_p95_pct=0.5,
        success=True, early_killed=False,
    )


def _mem(*configs) -> ExperienceMemory:
    mem = ExperienceMemory()
    for i, cfg in enumerate(configs):
        mem.add(TrialRecord(
            trial_id=f"t{i}", config=dict(cfg),
            metrics=_metrics(1000.0 + i * 10),
            source="baseline" if i == 0 else "agent",
        ))
    return mem


def _diag(should_stop=False) -> DiagnosisResult:
    return DiagnosisResult(
        bottleneck="underutilized", confidence=0.5,
        evidence="", hypothesis="", slo_pressure="low",
        should_stop=should_stop, source="llm",
    )


# -------------------- fallback --------------------
def test_fallback_picks_highest_priority_untried():
    mem = _mem({"max_num_seqs": 32})  # 只有 max_num_seqs=32 试过
    res = _fallback_propose(mem, current_config={"max_num_seqs": 32})
    assert isinstance(res, ConfigDelta)
    # max_num_seqs 是优先级第一，应该选未试过的另一个
    assert res.param == "max_num_seqs"
    assert res.new_value != 32
    assert res.source == "fallback"


def test_fallback_skips_when_all_tried_and_moves_next():
    """把 max_num_seqs 全部 candidate 用完，应跳到下一个参数。"""
    from tuner import param_registry
    mns_cands = param_registry.get_spec("max_num_seqs").candidates
    configs = [{"max_num_seqs": v} for v in mns_cands]
    mem = _mem(*configs)
    res = _fallback_propose(mem, current_config=configs[-1])
    assert isinstance(res, ConfigDelta)
    assert res.param == "max_num_batched_tokens"


def test_fallback_stops_when_all_priority_exhausted():
    from tuner import param_registry
    # 把所有优先参数都耗尽
    records = []
    for name in FALLBACK_PRIORITY:
        spec = param_registry.get_spec(name)
        for v in spec.candidates:
            records.append({name: v})
    mem = _mem(*records)
    res = _fallback_propose(mem, current_config=records[-1])
    assert isinstance(res, StopSignal)


def test_diagnosis_should_stop_returns_stop():
    mem = _mem({"max_num_seqs": 32})
    res = propose(client=None, tools=None, memory=mem,
                  diagnosis=_diag(should_stop=True),
                  current_config={"max_num_seqs": 32})
    assert isinstance(res, StopSignal)


def test_no_client_uses_fallback():
    mem = _mem({"max_num_seqs": 32})
    res = propose(client=None, tools=None, memory=mem,
                  diagnosis=_diag(), current_config={"max_num_seqs": 32})
    assert isinstance(res, ConfigDelta)
    assert res.source == "fallback"


# -------------------- LLM 路径 --------------------
class FakeTools:
    def openai_tools_schema(self):
        return [{"type": "function",
                 "function": {"name": "list_params", "description": "x",
                              "parameters": {"type": "object", "properties": {}}}}]
    def dispatch(self, name, args):
        return {"ok": True}


class FakeClient:
    def __init__(self, content, tool_trace=None):
        self.content = content
        self.tool_trace = tool_trace or []
        self.calls = 0

    def chat_with_tools(self, messages, registry, **kw):
        self.calls += 1
        return {
            "messages": messages,
            "tool_trace": self.tool_trace,
            "final": {"role": "assistant", "content": self.content},
            "raw_last_response": None,
        }


def test_llm_happy_path_returns_config_delta():
    raw = {
        "param": "max_num_seqs", "old_value": 32, "new_value": 64,
        "hypothesis_ref": "h1", "tools_used": [],
        "reason": "提升并发", "expected_effect": {"throughput": "+10%"},
        "rollback_if": "preempt > 5",
    }
    client = FakeClient(json.dumps(raw),
                        tool_trace=[{"name": "list_params", "arguments": {}, "result": {"ok": True}}])
    mem = _mem({"max_num_seqs": 32})
    res = propose(client, FakeTools(), mem, _diag(),
                  current_config={"max_num_seqs": 32})
    assert isinstance(res, ConfigDelta)
    assert res.param == "max_num_seqs"
    assert res.new_value == 64
    # tools_used 应自动注入
    assert "list_params" in res.tools_used


def test_llm_returns_stop_action():
    raw = {"action": "stop", "reason": "已收敛"}
    client = FakeClient(json.dumps(raw))
    mem = _mem({"max_num_seqs": 32})
    res = propose(client, FakeTools(), mem, _diag(),
                  current_config={"max_num_seqs": 32})
    assert isinstance(res, StopSignal)
    assert "收敛" in res.reason


def test_llm_malformed_json_falls_back():
    client = FakeClient("not json {")
    mem = _mem({"max_num_seqs": 32})
    res = propose(client, FakeTools(), mem, _diag(),
                  current_config={"max_num_seqs": 32})
    assert isinstance(res, ConfigDelta)
    assert res.source == "fallback"


def test_llm_strips_markdown_fence():
    # underutilized -> max_num_seqs direction=up, 32 -> 64 是合法的
    raw = {"param": "max_num_seqs", "new_value": 64, "old_value": 32,
           "reason": "x", "expected_effect": {}}
    content = f"```json\n{json.dumps(raw)}\n```"
    client = FakeClient(content)
    mem = _mem({"max_num_seqs": 32})
    res = propose(client, FakeTools(), mem, _diag(),
                  current_config={"max_num_seqs": 32})
    assert isinstance(res, ConfigDelta)
    assert res.new_value == 64


def test_llm_runtime_error_falls_back():
    class BadClient:
        def chat_with_tools(self, *a, **k):
            raise RuntimeError("api down")
    mem = _mem({"max_num_seqs": 32})
    res = propose(BadClient(), FakeTools(), mem, _diag(),
                  current_config={"max_num_seqs": 32})
    assert isinstance(res, ConfigDelta)
    assert res.source == "fallback"
