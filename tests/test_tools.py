#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_tools.py — Task 6.6 ToolRegistry 单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.memory import ExperienceMemory, TrialRecord
from tuner.metrics_parser import TrialMetrics
from tuner.tools import FORBIDDEN_TOOLS, ToolRegistry, ToolSpec


def _mk(tid, tput, source="agent", **cfg):
    m = TrialMetrics(
        throughput_tok_per_s=tput, throughput_req_per_s=tput / 100.0,
        ttft_p95_ms=100.0, tpot_p95_ms=20.0, latency_p95_ms=2000.0,
        kv_cache_usage_p95_pct=0.5, success=True,
    )
    return TrialRecord(trial_id=tid, config=dict(cfg), metrics=m, source=source)


@pytest.fixture
def mem_with_baseline():
    mem = ExperienceMemory()
    mem.add(_mk("baseline_0", 75.0, source="baseline", max_num_seqs=32))
    mem.add(_mk("trial_1", 120.0, source="agent", max_num_seqs=64))
    mem.add(_mk("trial_2", 200.0, source="agent", max_num_seqs=128))
    return mem


# ---------- 基础 ----------

def test_registry_has_six_a_tools(mem_with_baseline):
    reg = ToolRegistry(mem_with_baseline)
    assert set(reg.names()) == {
        "read_metrics", "query_param_docs", "list_params",
        "get_baseline", "get_history_summary", "compare_trials",
    }


def test_openai_schema_shape(mem_with_baseline):
    reg = ToolRegistry(mem_with_baseline)
    schema = reg.openai_tools_schema()
    assert len(schema) == 6
    for entry in schema:
        assert entry["type"] == "function"
        f = entry["function"]
        assert "name" in f and "description" in f and "parameters" in f
        assert f["parameters"]["type"] == "object"


# ---------- 拒绝写工具 ----------

def test_dispatch_apply_config_forbidden(mem_with_baseline):
    reg = ToolRegistry(mem_with_baseline)
    out = reg.dispatch("apply_config", {"max_num_seqs": 64})
    assert out["ok"] is False
    assert "forbidden" in out["error"].lower()


def test_dispatch_unknown_tool(mem_with_baseline):
    reg = ToolRegistry(mem_with_baseline)
    out = reg.dispatch("does_not_exist")
    assert out["ok"] is False


def test_register_extra_writer_tool_rejected(mem_with_baseline):
    bad = ToolSpec(name="apply_config", description="bad",
                   parameters={"type": "object", "properties": {}}, handler=lambda m: {})
    with pytest.raises(ValueError):
        ToolRegistry(mem_with_baseline, extra_tools=[bad])


def test_forbidden_set_contents():
    assert "apply_config" in FORBIDDEN_TOOLS


# ---------- read_metrics ----------

def test_read_metrics_latest(mem_with_baseline):
    reg = ToolRegistry(mem_with_baseline)
    out = reg.dispatch("read_metrics")
    assert out["ok"] is True
    assert out["metrics"]["trial_id"] == "trial_2"
    assert out["metrics"]["throughput_tok_per_s"] == 200.0


def test_read_metrics_specific_id(mem_with_baseline):
    reg = ToolRegistry(mem_with_baseline)
    out = reg.dispatch("read_metrics", {"trial_id": "trial_1"})
    assert out["ok"] is True
    assert out["metrics"]["trial_id"] == "trial_1"


def test_read_metrics_missing_id(mem_with_baseline):
    reg = ToolRegistry(mem_with_baseline)
    out = reg.dispatch("read_metrics", {"trial_id": "ghost"})
    assert out["ok"] is False


def test_read_metrics_empty_memory():
    reg = ToolRegistry(ExperienceMemory())
    out = reg.dispatch("read_metrics")
    assert out["ok"] is False


# ---------- query_param_docs / list_params ----------

def test_query_param_docs_known(mem_with_baseline):
    out = ToolRegistry(mem_with_baseline).dispatch(
        "query_param_docs", {"name": "max_num_seqs"})
    assert out["ok"] is True
    assert out["candidates"]
    assert out["requires_restart"] is True
    assert "throughput" in out["affects"]


def test_query_param_docs_unknown(mem_with_baseline):
    out = ToolRegistry(mem_with_baseline).dispatch(
        "query_param_docs", {"name": "fictitious"})
    assert out["ok"] is False


def test_list_params(mem_with_baseline):
    out = ToolRegistry(mem_with_baseline).dispatch("list_params")
    assert out["ok"] is True
    assert len(out["params"]) == 9


# ---------- get_baseline / history / compare ----------

def test_get_baseline_present(mem_with_baseline):
    out = ToolRegistry(mem_with_baseline).dispatch("get_baseline")
    assert out["ok"] is True
    assert out["baseline"]["trial_id"] == "baseline_0"


def test_get_baseline_absent():
    mem = ExperienceMemory()
    mem.add(_mk("t1", 10.0, source="agent"))
    out = ToolRegistry(mem).dispatch("get_baseline")
    assert out["ok"] is False


def test_get_history_summary(mem_with_baseline):
    out = ToolRegistry(mem_with_baseline).dispatch(
        "get_history_summary", {"top_k": 2, "recent_n": 1})
    assert out["ok"] is True
    assert out["n_trials"] == 3
    assert len(out["top_k"]) == 2
    assert out["best"]["trial_id"] == "trial_2"


def test_compare_trials(mem_with_baseline):
    out = ToolRegistry(mem_with_baseline).dispatch(
        "compare_trials", {"trial_id_a": "baseline_0", "trial_id_b": "trial_2"})
    assert out["ok"] is True
    assert out["metric_deltas"]["throughput_tok_per_s"] == pytest.approx(200.0 - 75.0)
    assert out["config_diff"]["max_num_seqs"] == {"a": 32, "b": 128}


def test_compare_trials_missing(mem_with_baseline):
    out = ToolRegistry(mem_with_baseline).dispatch(
        "compare_trials", {"trial_id_a": "ghost", "trial_id_b": "trial_2"})
    assert out["ok"] is False


# ---------- 错误参数 ----------

def test_dispatch_bad_arguments(mem_with_baseline):
    out = ToolRegistry(mem_with_baseline).dispatch(
        "query_param_docs", {})  # 缺 required name
    assert out["ok"] is False
    assert "参数" in out["error"] or "missing" in out["error"].lower() or "argument" in out["error"].lower()
