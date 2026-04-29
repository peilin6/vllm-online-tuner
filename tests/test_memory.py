#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_memory.py — Task 6.4 ExperienceMemory 单元测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.memory import ExperienceMemory, TrialRecord
from tuner.metrics_parser import TrialMetrics


def _mk(trial_id, tput, success=True, early=False, src="agent", **cfg):
    m = TrialMetrics(
        throughput_tok_per_s=tput,
        ttft_p95_ms=100.0, tpot_p95_ms=20.0,
        preemptions_total=0, kv_cache_usage_p95_pct=0.5,
        success=success, early_killed=early,
    )
    return TrialRecord(trial_id=trial_id, config=dict(cfg), metrics=m, source=src)


# ---------- TrialRecord 序列化 ----------

def test_record_roundtrip():
    r = _mk("t1", 100.0, max_num_seqs=64)
    d = r.to_dict()
    r2 = TrialRecord.from_dict(d)
    assert r2.trial_id == "t1"
    assert r2.config == {"max_num_seqs": 64}
    assert r2.metrics.throughput_tok_per_s == 100.0
    assert isinstance(r2.metrics, TrialMetrics)


def test_record_from_dict_ignores_unknown_metric_fields():
    d = {
        "trial_id": "t1",
        "config": {},
        "metrics": {"throughput_tok_per_s": 50.0, "future_field": "ignored"},
    }
    r = TrialRecord.from_dict(d)
    assert r.metrics.throughput_tok_per_s == 50.0


# ---------- 排序与查询 ----------

def test_top_k_orders_by_throughput_desc():
    mem = ExperienceMemory()
    mem.add(_mk("a", 100.0))
    mem.add(_mk("b", 200.0))
    mem.add(_mk("c", 150.0))
    ids = [r.trial_id for r in mem.top_k(2)]
    assert ids == ["b", "c"]


def test_top_k_failed_trials_ranked_last():
    mem = ExperienceMemory()
    mem.add(_mk("ok_low", 50.0, success=True))
    mem.add(_mk("killed", 999.0, success=False, early=True))
    mem.add(_mk("failed_low", 80.0, success=False))
    top = mem.top_k(3)
    assert top[0].trial_id == "ok_low"
    assert {top[1].trial_id, top[2].trial_id} == {"killed", "failed_low"}


def test_recent_n_returns_last_in_order():
    mem = ExperienceMemory()
    for i in range(5):
        mem.add(_mk(f"t{i}", 10.0 + i))
    rec = mem.recent_n(2)
    assert [r.trial_id for r in rec] == ["t3", "t4"]


def test_best_returns_none_when_empty_or_all_failed():
    mem = ExperienceMemory()
    assert mem.best() is None
    mem.add(_mk("x", 100.0, success=False))
    assert mem.best() is None


def test_best_picks_max_throughput():
    mem = ExperienceMemory()
    mem.add(_mk("a", 100.0))
    mem.add(_mk("b", 250.0))
    mem.add(_mk("c", 200.0))
    assert mem.best().trial_id == "b"


# ---------- summarize ----------

def test_summarize_compact_view():
    mem = ExperienceMemory()
    mem.add(_mk("a", 100.0, max_num_seqs=64))
    mem.add(_mk("b", 250.0, max_num_seqs=128))
    s = mem.summarize(top_k=2, recent_n=1)
    assert s["n_trials"] == 2
    assert s["best"]["trial_id"] == "b"
    assert s["best"]["throughput_tok_per_s"] == 250.0
    assert len(s["top_k"]) == 2
    assert len(s["recent"]) == 1
    assert s["recent"][0]["trial_id"] == "b"
    # 紧凑视图字段
    keys = set(s["best"].keys())
    assert keys >= {"throughput_tok_per_s", "ttft_p95_ms", "config", "success",
                    "early_killed"}


def test_summarize_empty_memory():
    s = ExperienceMemory().summarize()
    assert s["n_trials"] == 0
    assert s["best"] is None
    assert s["top_k"] == [] and s["recent"] == []


# ---------- 持久化 ----------

def test_jsonl_append_on_add(tmp_path):
    p = tmp_path / "mem.jsonl"
    mem = ExperienceMemory(p)
    mem.add(_mk("a", 100.0))
    mem.add(_mk("b", 200.0))
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["trial_id"] == "b"


def test_load_from_jsonl(tmp_path):
    p = tmp_path / "mem.jsonl"
    mem1 = ExperienceMemory(p)
    mem1.add(_mk("a", 100.0, foo=1))
    mem1.add(_mk("b", 200.0, foo=2))
    # 重新打开
    mem2 = ExperienceMemory(p)
    assert len(mem2) == 2
    assert mem2.best().trial_id == "b"
    assert mem2.all()[0].config == {"foo": 1}


def test_save_to_explicit_path(tmp_path):
    mem = ExperienceMemory()
    mem.add(_mk("only", 42.0))
    out = tmp_path / "nested" / "mem.jsonl"
    mem.save(out)
    assert out.exists()
    reload = ExperienceMemory(out)
    assert len(reload) == 1
    assert reload.all()[0].metrics.throughput_tok_per_s == 42.0


def test_save_without_path_raises():
    with pytest.raises(ValueError):
        ExperienceMemory().save()


def test_load_skips_blank_lines(tmp_path):
    p = tmp_path / "mem.jsonl"
    rec = _mk("x", 10.0).to_dict()
    p.write_text(json.dumps(rec) + "\n\n   \n", encoding="utf-8")
    mem = ExperienceMemory(p)
    assert len(mem) == 1
