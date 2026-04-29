#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_judge.py — Task 7.4 Judge 单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_advisor.schemas import ConfigDelta
from tuner.judge import Judge, JudgeVerdict
from tuner.memory import ExperienceMemory, TrialRecord
from tuner.metrics_parser import TrialMetrics


def _metrics(tput=1000.0, **kw):
    base = dict(throughput_tok_per_s=tput, ttft_p95_ms=200.0, tpot_p95_ms=20.0,
                latency_p95_ms=2000.0, preemption_rate_per_min=0.0,
                kv_cache_usage_p95_pct=0.5, success=True, early_killed=False)
    base.update(kw)
    return TrialMetrics(**base)


def _add(mem, tid, cfg, tput=1000.0, src="agent", **mkw):
    mem.add(TrialRecord(trial_id=tid, config=dict(cfg),
                        metrics=_metrics(tput, **mkw), source=src))


# -------------------- check_delta --------------------
def test_check_delta_pass():
    mem = ExperienceMemory()
    _add(mem, "t1", {"max_num_seqs": 32}, src="baseline")
    judge = Judge(mem)
    v = judge.check_delta(ConfigDelta(param="max_num_seqs",
                                       old_value=32, new_value=64))
    assert v.pass_ is True


def test_check_delta_unknown_param():
    judge = Judge(ExperienceMemory())
    v = judge.check_delta(ConfigDelta(param="bogus_param",
                                       old_value=1, new_value=2))
    assert v.pass_ is False
    assert "未登记" in v.reason


def test_check_delta_value_out_of_range():
    judge = Judge(ExperienceMemory())
    v = judge.check_delta(ConfigDelta(param="max_num_seqs",
                                       old_value=32, new_value=99999))
    assert v.pass_ is False


def test_check_delta_recently_rejected():
    mem = ExperienceMemory()
    mem.record_rejected({"max_num_seqs": 64}, "上次失败")
    judge = Judge(mem)
    v = judge.check_delta(ConfigDelta(param="max_num_seqs",
                                       old_value=32, new_value=64))
    assert v.pass_ is False
    assert "拒绝" in v.reason


def test_check_delta_no_change_rejected():
    mem = ExperienceMemory()
    _add(mem, "t1", {"max_num_seqs": 64}, tput=1000.0, src="baseline")
    judge = Judge(mem)
    # current_config 也是 64，且 best 是 64 → 等于没改
    v = judge.check_delta(
        ConfigDelta(param="max_num_seqs", old_value=64, new_value=64),
        current_config={"max_num_seqs": 64},
    )
    assert v.pass_ is False


# -------------------- check_trial_constraints --------------------
def test_constraint_pass():
    judge = Judge(ExperienceMemory())
    base = _metrics(1000.0)
    cur = _metrics(1100.0)
    cc = judge.check_trial_constraints(cur, base)
    assert cc.pass_ is True


def test_constraint_ttft_violation():
    judge = Judge(ExperienceMemory(), slo_ttft_mult=1.2)
    base = _metrics(ttft_p95_ms=200.0)
    cur = _metrics(ttft_p95_ms=300.0)  # 1.5x
    cc = judge.check_trial_constraints(cur, base)
    assert cc.pass_ is False
    assert any("ttft" in v for v in cc.violations)


def test_constraint_preempt_violation():
    judge = Judge(ExperienceMemory(), slo_preempt_per_min=5.0)
    cur = _metrics(preemption_rate_per_min=10.0)
    cc = judge.check_trial_constraints(cur, _metrics())
    assert cc.pass_ is False


def test_constraint_early_killed():
    judge = Judge(ExperienceMemory())
    cur = _metrics(early_killed=True, success=False)
    cc = judge.check_trial_constraints(cur, _metrics())
    assert cc.pass_ is False


# -------------------- should_terminate --------------------
def test_terminate_max_steps():
    judge = Judge(ExperienceMemory(), max_steps=5)
    done, why = judge.should_terminate(ExperienceMemory(), 5)
    assert done is True
    assert "max_steps" in why


def test_terminate_convergence():
    mem = ExperienceMemory()
    # 三次 trial 吞吐变化 <2%（1000, 1005, 1010 → range 1% < 2%）
    _add(mem, "t1", {}, tput=1000.0)
    _add(mem, "t2", {}, tput=1005.0)
    _add(mem, "t3", {}, tput=1010.0)
    judge = Judge(mem, max_steps=20, converge_window=3, converge_score_pct=0.02)
    done, why = judge.should_terminate(mem, 3)
    assert done is True
    assert "变化" in why or "<" in why


def test_no_terminate_when_diverging():
    mem = ExperienceMemory()
    _add(mem, "t1", {}, tput=800.0)
    _add(mem, "t2", {}, tput=1000.0)
    _add(mem, "t3", {}, tput=1200.0)
    judge = Judge(mem, max_steps=20, converge_window=3, converge_score_pct=0.02)
    done, _ = judge.should_terminate(mem, 3)
    assert done is False


def test_should_early_stop_trial():
    judge = Judge(ExperienceMemory())
    stop, _ = judge.should_early_stop_trial({"preempt_rate_per_s": 3.0})
    assert stop is True
    stop2, _ = judge.should_early_stop_trial({"kv_pct": 0.99})
    assert stop2 is True
    stop3, _ = judge.should_early_stop_trial({"preempt_rate_per_s": 0.5, "kv_pct": 0.5})
    assert stop3 is False
