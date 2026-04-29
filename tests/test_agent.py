#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_agent.py — Task 7.5 VtaAgent 主循环单元测试。

完全用 mock，不依赖 vLLM 进程：
- runner_fn: 给一个确定性指标合成器
- client=None：所有 LLM 调用走 fallback
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.agent import VtaAgent, AgentReport, apply_delta
from tuner.judge import Judge
from tuner.memory import ExperienceMemory
from tuner.metrics_parser import TrialMetrics
from tuner.optimizer import B_TOOL_SPECS
from tuner.tools import ToolRegistry
from llm_advisor.schemas import ConfigDelta


def _baseline_metrics() -> TrialMetrics:
    return TrialMetrics(
        throughput_tok_per_s=1000.0, ttft_p95_ms=200.0, tpot_p95_ms=20.0,
        latency_p95_ms=2000.0, preemption_rate_per_min=0.0,
        kv_cache_usage_p95_pct=0.5, success=True, early_killed=False,
    )


def _runner_factory(profile: dict[int, float]):
    """根据 max_num_seqs 决定吞吐；其他指标固定。"""
    def runner_fn(cfg, *, baseline_throughput_tok_per_s):
        mns = cfg.get("max_num_seqs", 32)
        tput = profile.get(mns, 800.0)
        return TrialMetrics(
            throughput_tok_per_s=tput, ttft_p95_ms=200.0, tpot_p95_ms=20.0,
            latency_p95_ms=2000.0, preemption_rate_per_min=0.0,
            kv_cache_usage_p95_pct=0.5, success=True, early_killed=False,
            wall_time_s=10.0,
        )
    return runner_fn


def test_apply_delta_returns_new_dict():
    cur = {"max_num_seqs": 32, "block_size": 16}
    new = apply_delta(cur, ConfigDelta(param="max_num_seqs",
                                        old_value=32, new_value=64))
    assert new["max_num_seqs"] == 64
    assert new["block_size"] == 16
    assert cur["max_num_seqs"] == 32   # 原 dict 不变


def test_agent_runs_to_max_steps_with_fallback():
    mem = ExperienceMemory()
    tools = ToolRegistry(mem, extra_tools=B_TOOL_SPECS)
    judge = Judge(mem, max_steps=5, converge_score_pct=0.0)  # 关掉收敛终止
    runner = _runner_factory({32: 1000.0, 64: 1100.0, 16: 900.0,
                              96: 1200.0, 128: 1150.0, 192: 1080.0, 256: 1050.0,
                              8: 800.0})
    agent = VtaAgent(mem, tools, judge, runner, client=None,
                     run_id="unit_max", use_llm=False)

    report = agent.run(_baseline_metrics(), {"max_num_seqs": 32}, max_steps=5)
    assert isinstance(report, AgentReport)
    assert report.n_steps >= 1
    # baseline + 至少 1 个 trial 进 memory
    assert len(mem) >= 2
    assert report.best_trial is not None
    # 没调 LLM
    assert all(v == 0 for v in report.llm_call_counts.values())


def test_agent_baseline_only_added_once():
    mem = ExperienceMemory()
    tools = ToolRegistry(mem, extra_tools=B_TOOL_SPECS)
    judge = Judge(mem, max_steps=2, converge_score_pct=0.0)
    runner = _runner_factory({})
    agent = VtaAgent(mem, tools, judge, runner, client=None, use_llm=False)

    agent.run(_baseline_metrics(), {"max_num_seqs": 32}, max_steps=2)
    n_baseline = sum(1 for r in mem.all() if r.source == "baseline")
    assert n_baseline == 1


def test_agent_terminates_on_judge_max_steps():
    mem = ExperienceMemory()
    tools = ToolRegistry(mem, extra_tools=B_TOOL_SPECS)
    judge = Judge(mem, max_steps=2, converge_score_pct=0.0)
    runner = _runner_factory({})
    agent = VtaAgent(mem, tools, judge, runner, client=None, use_llm=False)

    report = agent.run(_baseline_metrics(), {"max_num_seqs": 32}, max_steps=20)
    assert report.n_steps <= 2
    assert "max_steps" in report.stop_reason or "loop_end" in report.stop_reason


def test_agent_records_rejected_when_runner_raises():
    mem = ExperienceMemory()
    tools = ToolRegistry(mem, extra_tools=B_TOOL_SPECS)
    judge = Judge(mem, max_steps=2, converge_score_pct=0.0)

    def bad_runner(cfg, *, baseline_throughput_tok_per_s):
        raise RuntimeError("vLLM 启动失败")
    agent = VtaAgent(mem, tools, judge, bad_runner, client=None, use_llm=False)
    agent.run(_baseline_metrics(), {"max_num_seqs": 32}, max_steps=2)
    assert len(mem.rejected_proposals) >= 1


def test_agent_persists_memory_jsonl(tmp_path):
    mem_path = tmp_path / "mem.jsonl"
    mem = ExperienceMemory(mem_path)
    tools = ToolRegistry(mem, extra_tools=B_TOOL_SPECS)
    judge = Judge(mem, max_steps=3, converge_score_pct=0.0)
    runner = _runner_factory({32: 1000.0, 64: 1100.0, 96: 1200.0, 128: 1150.0})
    agent = VtaAgent(mem, tools, judge, runner, client=None, use_llm=False)
    agent.run(_baseline_metrics(), {"max_num_seqs": 32}, max_steps=3)
    assert mem_path.exists()
    lines = [l for l in mem_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) >= 2  # baseline + ≥1 trial
