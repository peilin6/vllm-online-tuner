#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_reflector.py — Task 7.3 R-LLM Reflector 单元测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_advisor.reflector import reflect, _fallback_reflect
from llm_advisor.schemas import ConfigDelta, ReflectionResult, ParseError
from tuner.memory import TrialRecord
from tuner.metrics_parser import TrialMetrics


def _rec(tid, tput, **kw):
    base = dict(throughput_tok_per_s=tput, ttft_p95_ms=200.0, tpot_p95_ms=20.0,
                latency_p95_ms=2000.0, kv_cache_usage_p95_pct=0.5,
                success=True, early_killed=False)
    base.update(kw)
    return TrialRecord(trial_id=tid, config={"max_num_seqs": 32},
                       metrics=TrialMetrics(**base), source="agent")


def _delta():
    return ConfigDelta(param="max_num_seqs", old_value=32, new_value=64)


# -------------------- fallback --------------------
def test_fallback_accept_on_improvement():
    prev, new = _rec("p", 1000.0), _rec("n", 1100.0)  # +10%
    res = _fallback_reflect(_delta(), prev, new, {"pass": True, "violations": []})
    assert res.verdict == "accept"
    assert res.next_move_hint == "double_down"
    assert res.source == "fallback"


def test_fallback_reject_on_constraint_violation():
    prev, new = _rec("p", 1000.0), _rec("n", 1500.0)
    res = _fallback_reflect(_delta(), prev, new,
                            {"pass": False, "violations": ["ttft 超标"]})
    assert res.verdict == "reject"
    assert res.next_move_hint == "rollback"


def test_fallback_reject_on_regression():
    prev, new = _rec("p", 1000.0), _rec("n", 950.0)  # -5%
    res = _fallback_reflect(_delta(), prev, new, {"pass": True, "violations": []})
    assert res.verdict == "reject"


def test_fallback_partial_when_change_small():
    prev, new = _rec("p", 1000.0), _rec("n", 1010.0)  # +1%
    res = _fallback_reflect(_delta(), prev, new, {"pass": True, "violations": []})
    assert res.verdict == "partial"


def test_fallback_reject_on_early_killed():
    prev, new = _rec("p", 1000.0), _rec("n", 0.0, early_killed=True, success=False)
    res = _fallback_reflect(_delta(), prev, new, {"pass": True, "violations": []})
    assert res.verdict == "reject"


# -------------------- LLM 路径 --------------------
class FakeClient:
    def __init__(self, content):
        self.content = content
    def chat(self, messages, **kw):
        return {"choices": [{"message": {"content": self.content}}]}


def test_llm_happy_path():
    raw = {"verdict": "accept", "reason": "+10% 吞吐",
           "new_note": "max_num_seqs ↑ 在该 workload 上有效",
           "next_move_hint": "double_down"}
    client = FakeClient(json.dumps(raw))
    prev, new = _rec("p", 1000.0), _rec("n", 1100.0)
    res = reflect(client, _delta(), prev, new, {"pass": True}, notes=[])
    assert res.source == "llm"
    assert res.verdict == "accept"


def test_llm_malformed_falls_back():
    client = FakeClient("garbage")
    prev, new = _rec("p", 1000.0), _rec("n", 1100.0)
    res = reflect(client, _delta(), prev, new, {"pass": True}, notes=[])
    assert res.source == "fallback"


def test_no_client_falls_back():
    prev, new = _rec("p", 1000.0), _rec("n", 950.0)
    res = reflect(None, _delta(), prev, new, {"pass": True}, notes=[])
    assert res.source == "fallback"


def test_schema_invalid_verdict_raises():
    with pytest.raises(ParseError):
        ReflectionResult.from_dict({"verdict": "weird"})
