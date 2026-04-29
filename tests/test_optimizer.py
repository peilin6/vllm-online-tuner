#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_optimizer.py — Task 6.7 B 类算法工具单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.memory import ExperienceMemory, TrialRecord
from tuner.metrics_parser import TrialMetrics
from tuner.optimizer import (
    B_TOOL_SPECS,
    _tool_bo_suggest, _tool_param_sensitivity, _tool_pareto_front,
    _tool_local_grid, _tool_cluster_workload_phases, _spearman,
)
from tuner.tools import ToolRegistry


def _mk(tid, tput, ttft=100.0, lat=2000.0, success=True, early=False, **cfg):
    m = TrialMetrics(
        throughput_tok_per_s=tput, ttft_p95_ms=ttft, latency_p95_ms=lat,
        tpot_p95_ms=20.0, kv_cache_usage_p95_pct=0.5,
        success=success, early_killed=early,
    )
    return TrialRecord(trial_id=tid, config=dict(cfg), metrics=m)


# ---------- _spearman ----------

def test_spearman_perfect_pos():
    assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_perfect_neg():
    assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_constant_returns_zero():
    assert _spearman([1, 2, 3], [5, 5, 5]) == 0.0


def test_spearman_too_few_returns_zero():
    assert _spearman([1], [1]) == 0.0


# ---------- bo_suggest ----------

def test_bo_suggest_cold_start_returns_medians():
    mem = ExperienceMemory()
    out = _tool_bo_suggest(mem, n_warmup=3)
    assert out["ok"] is True
    assert out["mode"] == "cold_start"
    s = out["suggestion"]
    assert "max_num_seqs" in s and "max_num_batched_tokens" in s
    # 中位数应该是合法候选
    assert s["max_num_seqs"] in [8, 16, 32, 64, 96, 128, 192, 256]


def test_bo_suggest_with_observations_uses_tpe():
    mem = ExperienceMemory()
    mem.add(_mk("t1", 100.0, max_num_seqs=32,
                max_num_batched_tokens=2048, gpu_memory_utilization=0.85))
    mem.add(_mk("t2", 200.0, max_num_seqs=64,
                max_num_batched_tokens=4096, gpu_memory_utilization=0.90))
    mem.add(_mk("t3", 150.0, max_num_seqs=128,
                max_num_batched_tokens=8192, gpu_memory_utilization=0.92))
    out = _tool_bo_suggest(mem, n_warmup=3)
    assert out["ok"] is True
    assert out["mode"] == "tpe"
    s = out["suggestion"]
    assert s["max_num_seqs"] in [8, 16, 32, 64, 96, 128, 192, 256]
    spec_range_gmu = (0.5, 0.97)
    assert spec_range_gmu[0] <= s["gpu_memory_utilization"] <= spec_range_gmu[1]


# ---------- param_sensitivity ----------

def test_param_sensitivity_too_few_trials():
    mem = ExperienceMemory()
    mem.add(_mk("t1", 100.0, max_num_seqs=32))
    mem.add(_mk("t2", 120.0, max_num_seqs=64))
    out = _tool_param_sensitivity(mem)
    assert out["ok"] is False


def test_param_sensitivity_ranks_dominant_param():
    mem = ExperienceMemory()
    # max_num_seqs 与 throughput 完全单调；filler 与 throughput 无关
    for i, (mns, fil, tput) in enumerate([
        (16, 100, 50.0), (32, 100, 100.0), (64, 100, 150.0),
        (96, 200, 200.0), (128, 50, 230.0),
    ]):
        mem.add(_mk(f"t{i}", tput, max_num_seqs=mns, filler=fil))
    out = _tool_param_sensitivity(mem)
    assert out["ok"] is True
    assert out["ranking"][0] == "max_num_seqs"
    assert abs(out["sensitivity"]["max_num_seqs"]["spearman"]) > 0.9


# ---------- pareto_front ----------

def test_pareto_front_basic():
    mem = ExperienceMemory()
    # A: 高吞吐高延迟; B: 低吞吐低延迟; C: 被支配 (低吞吐高延迟)
    mem.add(_mk("A", 200.0, lat=2000.0))
    mem.add(_mk("B", 100.0, lat=500.0))
    mem.add(_mk("C", 80.0, lat=2500.0))
    out = _tool_pareto_front(mem)
    assert out["ok"] is True
    ids = {p["trial_id"] for p in out["front"]}
    assert ids == {"A", "B"}


def test_pareto_front_skips_negative_sentinels():
    mem = ExperienceMemory()
    m_bad = TrialMetrics(throughput_tok_per_s=-1.0, latency_p95_ms=-1.0, success=True)
    mem.add(TrialRecord(trial_id="bad", config={}, metrics=m_bad))
    mem.add(_mk("ok", 100.0, lat=1000.0))
    out = _tool_pareto_front(mem)
    assert {p["trial_id"] for p in out["front"]} == {"ok"}


# ---------- local_grid ----------

def test_local_grid_around_center():
    out = _tool_local_grid(
        ExperienceMemory(),
        around={"max_num_seqs": 64, "gpu_memory_utilization": 0.90},
        radius=1,
    )
    assert out["ok"] is True
    # max_num_seqs 候选 [...,32,64,96,...] → 邻域 {32,64,96}；gmu [...,0.88,0.90,0.92] 邻域 3 个
    # 笛卡尔 = 3×3 - 中心点 = 8
    assert out["n"] == 8
    for g in out["grid"]:
        assert g["max_num_seqs"] in [32, 64, 96]
        assert g["gpu_memory_utilization"] in [0.88, 0.90, 0.92]
        assert g != {"max_num_seqs": 64, "gpu_memory_utilization": 0.90}


def test_local_grid_unknown_center_value_keeps_singleton():
    out = _tool_local_grid(
        ExperienceMemory(),
        around={"max_num_seqs": 999},  # 不在 candidates
        radius=1,
    )
    assert out["ok"] is True
    # 中心点不在候选 → options 仅 [999]，去掉中心后笛卡尔积为空
    assert out["n"] == 0


# ---------- cluster_workload_phases ----------

def test_cluster_workload_phases_basic():
    mem = ExperienceMemory()
    # 两簇明显分离：高吞吐组 vs 低吞吐组
    for i, (tput, ttft) in enumerate([
        (200.0, 80.0), (210.0, 85.0), (195.0, 78.0),
        (50.0, 300.0), (55.0, 320.0), (48.0, 290.0),
    ]):
        mem.add(_mk(f"t{i}", tput, ttft=ttft))
    out = _tool_cluster_workload_phases(mem, k=2)
    assert out["ok"] is True
    sizes = sorted(len(v) for v in out["clusters"].values())
    assert sizes == [3, 3]


def test_cluster_workload_phases_too_few():
    mem = ExperienceMemory()
    mem.add(_mk("t1", 100.0))
    out = _tool_cluster_workload_phases(mem, k=3)
    assert out["ok"] is False


# ---------- 集成进 ToolRegistry ----------

def test_b_tools_register_into_tool_registry():
    mem = ExperienceMemory()
    mem.add(_mk("t1", 100.0, max_num_seqs=32))
    reg = ToolRegistry(mem, extra_tools=B_TOOL_SPECS)
    names = set(reg.names())
    assert {"bo_suggest", "param_sensitivity", "pareto_front",
            "local_grid", "cluster_workload_phases"} <= names
    schema = reg.openai_tools_schema()
    assert len(schema) == 6 + 5


def test_b_tool_dispatch_via_registry():
    mem = ExperienceMemory()
    reg = ToolRegistry(mem, extra_tools=B_TOOL_SPECS)
    out = reg.dispatch("bo_suggest", {"n_warmup": 3})
    assert out["ok"] is True
    assert "suggestion" in out
