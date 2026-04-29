#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_diagnoser.py — Task 7.1 A-LLM Diagnoser 单元测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_advisor.diagnoser import diagnose, _fallback_diagnose
from llm_advisor.schemas import DiagnosisResult, ParseError
from tuner.memory import ExperienceMemory, TrialRecord
from tuner.metrics_parser import TrialMetrics


# -------------------- helpers --------------------
def _metrics(**kw) -> TrialMetrics:
    base = dict(
        throughput_req_per_s=1.0, throughput_tok_per_s=1000.0,
        ttft_p95_ms=200.0, tpot_p95_ms=20.0, latency_p95_ms=2000.0,
        preemptions_total=0, preemption_rate_per_min=0.0,
        kv_cache_usage_p95_pct=0.5, queue_time_p95_ms=10.0,
        success=True, early_killed=False, wall_time_s=60.0,
    )
    base.update(kw)
    return TrialMetrics(**base)


def _record(trial_id, **mkw) -> TrialRecord:
    return TrialRecord(trial_id=trial_id, config={"max_num_seqs": 32},
                       metrics=_metrics(**mkw), source="agent")


def _mem_with_baseline(base_tput=1000.0) -> ExperienceMemory:
    mem = ExperienceMemory()
    mem.add(TrialRecord(
        trial_id="baseline_0", config={"max_num_seqs": 32},
        metrics=_metrics(throughput_tok_per_s=base_tput), source="baseline",
    ))
    return mem


class FakeClient:
    """伪 LlmClient：chat() 返回 OpenAI 格式 dict。"""
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append({"messages": messages, "kw": kw})
        return {"choices": [{"message": {"content": self.content}}]}


# -------------------- fallback 规则 --------------------
def test_fallback_preempt_storm():
    rec = _record("t1", preemption_rate_per_min=10.0)
    res = _fallback_diagnose(rec, baseline=_metrics())
    assert res.bottleneck == "preempt_storm"
    assert res.source == "fallback"
    assert 0.0 <= res.confidence <= 1.0


def test_fallback_kv_pressure():
    rec = _record("t1", kv_cache_usage_p95_pct=0.97)
    res = _fallback_diagnose(rec, baseline=_metrics())
    assert res.bottleneck == "kv_cache_pressure"


def test_fallback_slo_margin_low():
    rec = _record("t1", throughput_tok_per_s=800.0)  # baseline 1000 * 0.95 = 950 > 800
    res = _fallback_diagnose(rec, baseline=_metrics(throughput_tok_per_s=1000.0))
    assert res.bottleneck == "slo_margin_low"


def test_fallback_underutilized_default():
    rec = _record("t1")  # 健康
    res = _fallback_diagnose(rec, baseline=_metrics())
    assert res.bottleneck == "underutilized"


# -------------------- LLM 路径 --------------------
def test_diagnose_llm_happy_path():
    raw = {
        "bottleneck": "kv_cache_pressure",
        "confidence": 0.8, "evidence": "kv 高",
        "hypothesis": "降 max_num_seqs",
        "slo_pressure": "high", "should_stop": False,
    }
    client = FakeClient(json.dumps(raw))
    mem = _mem_with_baseline()
    rec = _record("t1", kv_cache_usage_p95_pct=0.96)
    mem.add(rec)
    res = diagnose(client, mem, rec)
    assert res.source == "llm"
    assert res.bottleneck == "kv_cache_pressure"
    assert res.confidence == pytest.approx(0.8)
    # 自动注入了 response_format
    assert client.calls[0]["kw"].get("response_format") == {"type": "json_object"}


def test_diagnose_llm_strips_markdown_fence():
    raw = {"bottleneck": "underutilized", "confidence": 0.4}
    content = f"```json\n{json.dumps(raw)}\n```"
    client = FakeClient(content)
    mem = _mem_with_baseline()
    rec = _record("t1")
    mem.add(rec)
    res = diagnose(client, mem, rec)
    assert res.bottleneck == "underutilized"


def test_diagnose_llm_malformed_falls_back():
    client = FakeClient("not a json at all {")
    mem = _mem_with_baseline()
    rec = _record("t1", preemption_rate_per_min=10.0)
    mem.add(rec)
    res = diagnose(client, mem, rec)
    assert res.source == "fallback"
    assert res.bottleneck == "preempt_storm"


def test_diagnose_llm_invalid_enum_falls_back():
    client = FakeClient(json.dumps({"bottleneck": "WRONG_VALUE", "confidence": 0.5}))
    mem = _mem_with_baseline()
    rec = _record("t1")
    mem.add(rec)
    res = diagnose(client, mem, rec)
    assert res.source == "fallback"


def test_diagnose_no_client_uses_fallback():
    mem = _mem_with_baseline()
    rec = _record("t1", kv_cache_usage_p95_pct=0.97)
    mem.add(rec)
    res = diagnose(None, mem, rec)
    assert res.source == "fallback"
    assert res.bottleneck == "kv_cache_pressure"


def test_diagnose_finds_baseline_from_memory():
    """不显式传 baseline，应自动从 memory 中 source='baseline' 找。"""
    mem = _mem_with_baseline(base_tput=1000.0)
    rec = _record("t1", throughput_tok_per_s=800.0)
    mem.add(rec)
    res = diagnose(None, mem, rec)  # fallback 路径需 baseline 才能判 slo_margin_low
    assert res.bottleneck == "slo_margin_low"


def test_schema_clamps_confidence():
    r = DiagnosisResult.from_dict({"bottleneck": "underutilized", "confidence": 1.5})
    assert r.confidence == 1.0
    r2 = DiagnosisResult.from_dict({"bottleneck": "underutilized", "confidence": -0.2})
    assert r2.confidence == 0.0


def test_schema_invalid_bottleneck_raises():
    with pytest.raises(ParseError):
        DiagnosisResult.from_dict({"bottleneck": "foo", "confidence": 0.5})
