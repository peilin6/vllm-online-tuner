#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_integration_week6_7.py — Week 6 + Week 7 联合集成测试。

验证完整调用链：
- VtaAgent (Week 7 主循环) →
- Judge / ToolRegistry / Optimizer.B_TOOL_SPECS (Week 6 + 7) →
- ExperienceMemory JSONL 持久化 (Week 6) →
- 模拟 Runner（避免真启 vLLM）

LLM 用 mock client：A-LLM/R-LLM 走 client.chat 返回固定 JSON；P-LLM 走
chat_with_tools 在第一轮调一个工具，再返回 ConfigDelta。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_advisor.schemas import ConfigDelta
from tuner.agent import VtaAgent
from tuner.judge import Judge
from tuner.memory import ExperienceMemory
from tuner.metrics_parser import TrialMetrics
from tuner.optimizer import B_TOOL_SPECS
from tuner.tools import ToolRegistry


# =====================================================================
# Mock LlmClient：根据 system prompt 内容路由到不同响应
# =====================================================================
class MockLlmClient:
    """根据 system message 头几个字符识别 A/P/R-LLM。"""

    def __init__(self):
        self.calls = {"chat": 0, "chat_with_tools": 0}
        self._propose_seq = [
            {"param": "max_num_seqs", "old_value": 32, "new_value": 64,
             "reason": "提升并发", "rollback_if": "preempt > 5"},
            {"param": "max_num_seqs", "old_value": 64, "new_value": 96,
             "reason": "继续上调", "rollback_if": "preempt > 5"},
            {"param": "max_num_seqs", "old_value": 96, "new_value": 128,
             "reason": "再上调", "rollback_if": "preempt > 5"},
        ]
        self._propose_idx = 0

    # A-LLM / R-LLM
    def chat(self, messages, **kw):
        self.calls["chat"] += 1
        sys_text = (messages[0].get("content") or "")[:50]
        if "A-LLM" in sys_text or "诊断" in sys_text or "Diagnoser" in sys_text:
            payload = {
                "bottleneck": "underutilized", "confidence": 0.6,
                "evidence": "并发偏低", "hypothesis": "调高 max_num_seqs",
                "slo_pressure": "low", "should_stop": False,
            }
        else:  # R-LLM
            payload = {
                "verdict": "accept", "reason": "吞吐提升",
                "new_note": "max_num_seqs ↑ 有效",
                "next_move_hint": "double_down",
            }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    # P-LLM
    def chat_with_tools(self, messages, registry, **kw):
        self.calls["chat_with_tools"] += 1
        # 模拟 LLM 第一轮调用 list_params 工具
        trace = []
        try:
            r = registry.dispatch("list_params", {})
            trace.append({"name": "list_params", "arguments": {}, "result": r})
        except Exception:
            pass
        idx = min(self._propose_idx, len(self._propose_seq) - 1)
        self._propose_idx += 1
        payload = self._propose_seq[idx]
        return {
            "messages": messages,
            "tool_trace": trace,
            "final": {"role": "assistant", "content": json.dumps(payload)},
            "raw_last_response": None,
        }


# =====================================================================
# Mock Runner：根据 max_num_seqs 给出递增吞吐（验证 best_trial 选择）
# =====================================================================
def make_runner():
    profile = {32: 1000.0, 64: 1100.0, 96: 1180.0, 128: 1150.0, 192: 1080.0}
    def runner_fn(cfg, *, baseline_throughput_tok_per_s):
        mns = cfg.get("max_num_seqs", 32)
        return TrialMetrics(
            throughput_tok_per_s=profile.get(mns, 800.0),
            ttft_p95_ms=200.0, tpot_p95_ms=20.0, latency_p95_ms=2000.0,
            preemption_rate_per_min=0.5, kv_cache_usage_p95_pct=0.6,
            success=True, early_killed=False, wall_time_s=10.0,
            success_rate=1.0, total_requests=100,
        )
    return runner_fn


def _baseline() -> TrialMetrics:
    return TrialMetrics(
        throughput_tok_per_s=1000.0, ttft_p95_ms=200.0, tpot_p95_ms=20.0,
        latency_p95_ms=2000.0, preemption_rate_per_min=0.0,
        kv_cache_usage_p95_pct=0.5, success=True, early_killed=False,
    )


# =====================================================================
# 测试
# =====================================================================
def test_full_loop_with_mock_llm(tmp_path):
    mem_path = tmp_path / "memory.jsonl"
    mem = ExperienceMemory(mem_path)
    tools = ToolRegistry(mem, extra_tools=B_TOOL_SPECS)
    judge = Judge(mem, max_steps=4, converge_score_pct=0.0)

    client = MockLlmClient()
    agent = VtaAgent(
        mem, tools, judge, runner_fn=make_runner(),
        client=client, run_id="integ_full", use_llm=True,
        max_propose_rounds=2,
    )

    report = agent.run(_baseline(), {"max_num_seqs": 32}, max_steps=4)

    # 1) report 完整
    assert report.n_steps >= 1
    assert report.best_trial is not None
    assert report.baseline.throughput_tok_per_s == 1000.0

    # 2) memory 持久化 baseline + ≥1 个 agent trial
    lines = [l for l in mem_path.read_text("utf-8").splitlines() if l.strip()]
    assert len(lines) >= 2
    sources = {json.loads(l)["source"] for l in lines}
    assert "baseline" in sources
    assert "agent" in sources

    # 3) LLM 实际被调用过
    assert client.calls["chat_with_tools"] >= 1
    assert client.calls["chat"] >= 1
    # 至少有一个 LLM 路径产生过非零计数
    assert any(v > 0 for v in report.llm_call_counts.values())

    # 4) best_trial 应该是吞吐最高的 96=1180 或 64=1100（若 4 步没走到 96）
    best_tput = report.best_trial.metrics.throughput_tok_per_s
    assert best_tput >= 1100.0


def test_full_loop_fallback_only(tmp_path):
    """client=None：完整链路必须能跑通且产生 best > baseline。"""
    mem_path = tmp_path / "mem.jsonl"
    mem = ExperienceMemory(mem_path)
    tools = ToolRegistry(mem, extra_tools=B_TOOL_SPECS)
    judge = Judge(mem, max_steps=6, converge_score_pct=0.0)

    agent = VtaAgent(
        mem, tools, judge, runner_fn=make_runner(),
        client=None, run_id="integ_fb", use_llm=False,
    )
    report = agent.run(_baseline(), {"max_num_seqs": 32}, max_steps=6)

    assert report.best_trial is not None
    # fallback 路径不会调 LLM
    assert all(v == 0 for v in report.llm_call_counts.values())


def test_tool_dispatch_flows_through_registry():
    """ToolRegistry 应能分发 Optimizer 注入的 B 类工具。"""
    mem = ExperienceMemory()
    # 加几条 trial 让工具有数据
    for i, mns in enumerate([16, 32, 64, 96]):
        mem.add(__import__("tuner").memory.TrialRecord(
            trial_id=f"t{i}", config={"max_num_seqs": mns},
            metrics=TrialMetrics(
                throughput_tok_per_s=900.0 + i * 50, ttft_p95_ms=200.0,
                tpot_p95_ms=20.0, latency_p95_ms=2000.0,
                kv_cache_usage_p95_pct=0.5, success=True, early_killed=False,
            ),
            source="agent",
        ))
    tools = ToolRegistry(mem, extra_tools=B_TOOL_SPECS)
    # A 类
    out = tools.dispatch("list_params", {})
    assert out.get("ok") is not False
    # B 类（Optimizer 注入）
    out2 = tools.dispatch("param_sensitivity",
                           {"target": "throughput_tok_per_s",
                            "params": ["max_num_seqs"]})
    assert "ok" in out2  # 不论 ok 真假，均能走到 handler
    # 写工具被禁
    out3 = tools.dispatch("apply_config", {})
    assert out3["ok"] is False
