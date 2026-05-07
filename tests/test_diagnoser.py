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
    # R4: ttft_headroom < 0.10 即触发（不再依赖 throughput）
    rec = _record("t1", ttft_p95_ms=290.0)   # ttft_slo=300 -> headroom 3.3%
    res = _fallback_diagnose(rec, baseline=_metrics(throughput_tok_per_s=1000.0))
    assert res.bottleneck == "slo_margin_low"
    assert "(R4)" in res.evidence


def test_fallback_underutilized_default():
    rec = _record("t1")  # 健康
    res = _fallback_diagnose(rec, baseline=_metrics())
    assert res.bottleneck == "underutilized"
    assert "(R8)" in res.evidence


# -------------------- 新增 R3/R5/R6/R7 fallback 覆盖 --------------------
def test_fallback_queue_backlog_R3():
    # queue_time / latency = 1500/2000 = 75% > 30%
    rec = _record("t1", queue_time_p95_ms=1500.0, latency_p95_ms=2000.0)
    res = _fallback_diagnose(rec, baseline=_metrics())
    assert res.bottleneck == "queue_backlog"
    assert "(R3)" in res.evidence


def test_fallback_prefill_bound_R5():
    # baseline ttft=200/tpot=20，新 ttft=260(+30%)，tpot=21(+5%)
    rec = _record("t1", ttft_p95_ms=260.0, tpot_p95_ms=21.0)
    res = _fallback_diagnose(rec, baseline=_metrics(ttft_p95_ms=200.0, tpot_p95_ms=20.0))
    assert res.bottleneck == "prefill_bound"
    assert "(R5)" in res.evidence


def test_fallback_decode_bound_R6():
    # baseline ttft=200/tpot=20，新 tpot=26(+30%)，ttft=210(+5%)
    rec = _record("t1", ttft_p95_ms=210.0, tpot_p95_ms=26.0)
    res = _fallback_diagnose(rec, baseline=_metrics(ttft_p95_ms=200.0, tpot_p95_ms=20.0))
    assert res.bottleneck == "decode_bound"
    assert "(R6)" in res.evidence


def test_fallback_converged_R7():
    # 最近 3 步吞吐极差 <2%、SLO 余量充足
    mem = _mem_with_baseline()
    for i, t in enumerate([1000.0, 1005.0, 1003.0]):
        mem.add(TrialRecord(trial_id=f"t{i}", config={"max_num_seqs": 32},
                            metrics=_metrics(throughput_tok_per_s=t), source="agent"))
    rec = _record("latest")  # 健康
    res = _fallback_diagnose(rec, baseline=_metrics(), memory=mem)
    assert res.bottleneck == "converged"
    assert res.should_stop is True
    assert "(R7)" in res.evidence


def test_fallback_priority_preempt_over_kv():
    # preempt 与 kv 同时高，R1 优先
    rec = _record("t1", preemption_rate_per_min=10.0, kv_cache_usage_p95_pct=0.97)
    res = _fallback_diagnose(rec, baseline=_metrics())
    assert res.bottleneck == "preempt_storm"


def test_fallback_priority_kv_over_queue():
    # kv 与 queue 都高，R2 优先
    rec = _record("t1", kv_cache_usage_p95_pct=0.97,
                  queue_time_p95_ms=1500.0, latency_p95_ms=2000.0)
    res = _fallback_diagnose(rec, baseline=_metrics())
    assert res.bottleneck == "kv_cache_pressure"


def test_fallback_missing_baseline_skips_R5_R6():
    # 没有 baseline 情况下，TPOT 高也不应误判 decode_bound
    rec = _record("t1", tpot_p95_ms=100.0)
    res = _fallback_diagnose(rec, baseline=None)
    assert res.bottleneck == "underutilized"


# -------------------- prompt 填充测试 --------------------
def test_user_prompt_includes_pct_and_swing():
    from llm_advisor.diagnoser import _build_user_prompt
    mem = _mem_with_baseline()
    for i, t in enumerate([900.0, 950.0]):
        mem.add(TrialRecord(trial_id=f"t{i}", config={"max_num_seqs": 32},
                            metrics=_metrics(throughput_tok_per_s=t), source="agent"))
    rec = _record("latest", ttft_p95_ms=300.0, tpot_p95_ms=30.0)
    user = _build_user_prompt(
        latest=rec, baseline=_metrics(throughput_tok_per_s=1000.0,
                                      ttft_p95_ms=200.0, tpot_p95_ms=20.0),
        memory=mem, ttft_slo_ms=300.0, lat_slo_ms=5000.0,
    )
    # TTFT/TPOT 相对 baseline 百分比都注入了
    assert "ttft_pct" not in user                     # 占位符全部已被 format 替换
    assert "+50.0%" in user                           # tpot +50%
    # 缺失字段段落存在（这里没有缺失，应输出"无"）
    assert "缺失指标" in user



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
    # R4 触发：ttft 接近 SLO 上限
    rec = _record("t1", ttft_p95_ms=290.0)
    mem.add(rec)
    res = diagnose(None, mem, rec)
    assert res.bottleneck == "slo_margin_low"


def test_schema_clamps_confidence():
    r = DiagnosisResult.from_dict({"bottleneck": "underutilized", "confidence": 1.5})
    assert r.confidence == 1.0
    r2 = DiagnosisResult.from_dict({"bottleneck": "underutilized", "confidence": -0.2})
    assert r2.confidence == 0.0


def test_schema_invalid_bottleneck_raises():
    with pytest.raises(ParseError):
        DiagnosisResult.from_dict({"bottleneck": "foo", "confidence": 0.5})
