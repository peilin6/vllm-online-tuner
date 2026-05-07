#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_proposer_playbook.py — Proposer × Playbook 合规性测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_advisor.proposer import (
    propose, _fallback_propose, _validate_against_playbook,
)
from llm_advisor.playbook import get_entry
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


def _diag(bottleneck="kv_cache_pressure", confidence=0.7) -> DiagnosisResult:
    return DiagnosisResult(
        bottleneck=bottleneck, confidence=confidence,
        evidence="", hypothesis="", slo_pressure="medium",
        should_stop=False, source="llm",
    )


# -------------------- fallback 按 playbook 选参数 --------------------
def test_fallback_for_kv_pressure_picks_gpu_mem_first_with_up():
    cur = {"gpu_memory_utilization": 0.85, "max_num_seqs": 64}
    mem = _mem(cur)
    res = _fallback_propose(mem, cur, diagnosis=_diag("kv_cache_pressure"))
    assert isinstance(res, ConfigDelta)
    # playbook[kv_cache_pressure] 第一个允许参数是 gpu_memory_utilization, direction=up
    assert res.param == "gpu_memory_utilization"
    assert res.new_value > 0.85


def test_fallback_for_decode_bound_picks_batched_tokens_down():
    cur = {"max_num_batched_tokens": 8192, "max_num_seqs": 128}
    mem = _mem(cur)
    res = _fallback_propose(mem, cur, diagnosis=_diag("decode_bound"))
    assert isinstance(res, ConfigDelta)
    assert res.param == "max_num_batched_tokens"
    assert res.new_value < 8192


def test_fallback_for_underutilized_picks_batched_tokens_up():
    cur = {"max_num_batched_tokens": 4096, "max_num_seqs": 64}
    mem = _mem(cur)
    res = _fallback_propose(mem, cur, diagnosis=_diag("underutilized"))
    assert isinstance(res, ConfigDelta)
    assert res.param == "max_num_batched_tokens"
    assert res.new_value > 4096


def test_fallback_converged_returns_stop():
    cur = {"max_num_seqs": 64}
    mem = _mem(cur)
    res = _fallback_propose(mem, cur, diagnosis=_diag("converged"))
    assert isinstance(res, StopSignal)


def test_fallback_avoids_recent_rejected():
    cur = {"gpu_memory_utilization": 0.85}
    mem = _mem(cur)
    # 把 0.90/0.92/0.95 都标为 rejected，应该挑不到 up 候选 -> 转下一个参数
    for v in (0.90, 0.92, 0.95):
        mem.record_rejected({"gpu_memory_utilization": v}, "test")
    res = _fallback_propose(mem, cur, diagnosis=_diag("kv_cache_pressure"))
    assert not (isinstance(res, ConfigDelta)
                and res.param == "gpu_memory_utilization"
                and res.new_value in (0.90, 0.92, 0.95))


# -------------------- _validate_against_playbook --------------------
def test_validate_rejects_param_not_in_whitelist():
    entry = get_entry("kv_cache_pressure")
    # block_size 在白名单里，先用一个不在白名单的: enable_prefix_caching
    delta = ConfigDelta(param="enable_prefix_caching", old_value=False, new_value=True)
    err = _validate_against_playbook(delta, entry, {"enable_prefix_caching": False},
                                     ExperienceMemory())
    assert err is not None and "allowed_params" in err


def test_validate_rejects_wrong_direction():
    entry = get_entry("kv_cache_pressure")  # max_num_seqs 应 down
    delta = ConfigDelta(param="max_num_seqs", old_value=64, new_value=128)
    err = _validate_against_playbook(delta, entry, {"max_num_seqs": 64},
                                     ExperienceMemory())
    assert err is not None and "direction" in err


def test_validate_accepts_legal_proposal():
    entry = get_entry("kv_cache_pressure")
    delta = ConfigDelta(param="max_num_seqs", old_value=128, new_value=64)
    err = _validate_against_playbook(delta, entry, {"max_num_seqs": 128},
                                     ExperienceMemory())
    assert err is None


def test_validate_rejects_value_out_of_range():
    entry = get_entry("kv_cache_pressure")
    # max_num_seqs candidates=[8..256], range=(1,512)；9999 同时超出两者
    delta = ConfigDelta(param="max_num_seqs", old_value=128, new_value=9999)
    err = _validate_against_playbook(delta, entry, {"max_num_seqs": 128},
                                     ExperienceMemory())
    assert err is not None and ("candidates" in err or "range" in err)


# -------------------- LLM 路径：违反 playbook 时降级 --------------------
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

    def chat_with_tools(self, messages, registry, **kw):
        return {"messages": messages, "tool_trace": self.tool_trace,
                "final": {"role": "assistant", "content": self.content},
                "raw_last_response": None}


def test_llm_violation_param_not_in_whitelist_falls_back_and_records():
    # diagnosis=kv_cache_pressure, LLM 试图改 enable_prefix_caching（不在白名单）
    raw = {"param": "enable_prefix_caching", "old_value": False, "new_value": True,
           "reason": "尝试", "expected_effect": {"ttft": "-10%"}}
    client = FakeClient(json.dumps(raw))
    cur = {"enable_prefix_caching": False, "gpu_memory_utilization": 0.85,
           "max_num_seqs": 128}
    mem = _mem(cur)
    res = propose(client, FakeTools(), mem, _diag("kv_cache_pressure"), cur)
    assert isinstance(res, ConfigDelta)
    # 应已降级
    assert res.source == "fallback"
    # 越权应被记录
    rejected = [r for r in mem.rejected_proposals
                if "playbook_violation" in r[1]]
    assert len(rejected) >= 1


def test_llm_violation_wrong_direction_falls_back():
    # kv_cache_pressure: max_num_seqs 应 down，LLM 想 up
    raw = {"param": "max_num_seqs", "old_value": 64, "new_value": 256,
           "reason": "x", "expected_effect": {}}
    client = FakeClient(json.dumps(raw))
    cur = {"max_num_seqs": 64, "gpu_memory_utilization": 0.85}
    mem = _mem(cur)
    res = propose(client, FakeTools(), mem, _diag("kv_cache_pressure"), cur)
    assert isinstance(res, ConfigDelta)
    assert res.source == "fallback"


def test_llm_legal_proposal_passes():
    # kv_cache_pressure: gpu_memory_utilization up 是合法的
    raw = {"param": "gpu_memory_utilization", "old_value": 0.85, "new_value": 0.92,
           "reason": "release more KV", "expected_effect": {"kv_p95": "down"},
           "tools_used": []}
    client = FakeClient(json.dumps(raw),
                        tool_trace=[{"name": "query_param_docs",
                                     "arguments": {"name": "gpu_memory_utilization"},
                                     "result": {}}])
    cur = {"gpu_memory_utilization": 0.85, "max_num_seqs": 128}
    mem = _mem(cur)
    res = propose(client, FakeTools(), mem, _diag("kv_cache_pressure"), cur)
    assert isinstance(res, ConfigDelta)
    assert res.source == "llm"
    assert res.param == "gpu_memory_utilization"
    assert res.new_value == 0.92


def test_converged_diagnosis_short_circuits_to_stop():
    cur = {"max_num_seqs": 64}
    mem = _mem(cur)
    # converged 时不应进入 LLM，直接 stop
    res = propose(client=FakeClient("{}"), tools=FakeTools(),
                  memory=mem, diagnosis=_diag("converged", 0.7),
                  current_config=cur)
    assert isinstance(res, StopSignal)
