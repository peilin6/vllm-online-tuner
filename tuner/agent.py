#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent.py — VtaAgent 主循环

Task 7.5。把 Diagnoser/Proposer/Reflector/Judge/Tools/Memory/Runner 串成
Observe → Diagnose → Propose → Safety → Act → Record → Reflect → Loop。

LLM 客户端通过依赖注入传入；client=None 时所有 LLM 调用退化为规则版 fallback，
用于离线/无 API 时的端到端测试。

设计要点
--------
- VtaAgent 不直接 import VllmLauncher / Runner，而是接受一个 `runner_fn`
  callable: (config_overrides, *, baseline_throughput_tok_per_s) -> TrialMetrics。
  这样既能挂上 tuner.runner.run_trial，也能在测试中传入 mock。
- ConfigDelta → next_config 的合并放在 apply_delta()，不可变（返回新 dict）。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from llm_advisor.diagnoser import diagnose
from llm_advisor.proposer import propose
from llm_advisor.reflector import reflect
from llm_advisor.schemas import ConfigDelta, StopSignal

from .judge import Judge
from .memory import ExperienceMemory, TrialRecord
from .metrics_parser import TrialMetrics
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


# ====================================================================
# helpers
# ====================================================================
def apply_delta(current_cfg: dict, delta: ConfigDelta) -> dict:
    """返回应用 delta 后的新配置（不修改 current_cfg）。"""
    new_cfg = dict(current_cfg)
    new_cfg[delta.param] = delta.new_value
    return new_cfg


# ====================================================================
# 报告
# ====================================================================
@dataclass
class AgentReport:
    run_id: str
    n_steps: int
    best_trial: TrialRecord | None
    baseline: TrialMetrics
    total_wall_time_s: float
    llm_call_counts: dict = field(default_factory=dict)
    memory_path: str = ""
    stop_reason: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        d["best_trial"] = (
            self.best_trial.to_dict() if self.best_trial is not None else None
        )
        d["baseline"] = self.baseline.to_dict() if self.baseline else None
        return d


# ====================================================================
# 主循环
# ====================================================================
class VtaAgent:
    """LLM 编排 + 算法工具混合的 vLLM 调参 agent。"""

    def __init__(
        self,
        memory: ExperienceMemory,
        tools: ToolRegistry,
        judge: Judge,
        runner_fn: Callable[..., TrialMetrics],
        *,
        client: Any = None,                  # LlmClient 或 None
        run_id: str | None = None,
        ttft_slo_ms: float = 300.0,
        lat_slo_ms: float = 5000.0,
        max_propose_rounds: int = 3,
        use_llm: bool = True,
    ):
        self.memory = memory
        self.tools = tools
        self.judge = judge
        self.runner_fn = runner_fn
        self.client = client
        self.run_id = run_id or f"agent_{int(time.time())}"
        self.ttft_slo_ms = float(ttft_slo_ms)
        self.lat_slo_ms = float(lat_slo_ms)
        self.max_propose_rounds = int(max_propose_rounds)
        self.use_llm = bool(use_llm)
        self._llm_calls = {"diagnose": 0, "propose": 0, "reflect": 0}

    # ------------------------------------------------------------------
    def run(
        self,
        baseline_metrics: TrialMetrics,
        baseline_config: dict,
        *,
        max_steps: int = 20,
    ) -> AgentReport:
        t0 = time.perf_counter()

        # 1) 把 baseline 落入 memory（若尚未存在）
        if not any(r.source == "baseline" for r in self.memory.all()):
            self.memory.append(TrialRecord(
                trial_id=f"{self.run_id}_baseline",
                config=dict(baseline_config),
                metrics=baseline_metrics,
                source="baseline",
            ))

        current_cfg = dict(baseline_config)
        prev_record = self.memory.recent(1)[0]
        stop_reason = ""
        step = 0

        while step < max_steps:
            # ---- 终止条件 ----
            done, why = self.judge.should_terminate(self.memory, step)
            if done:
                stop_reason = why
                logger.info("[%s] 终止: %s", self.run_id, why)
                break

            # ---- Observe + Diagnose ----
            latest = self.memory.recent(1)[0]
            diagnosis = diagnose(
                self.client, self.memory, latest,
                baseline=baseline_metrics,
                ttft_slo_ms=self.ttft_slo_ms, lat_slo_ms=self.lat_slo_ms,
                use_llm=self.use_llm,
            )
            if diagnosis.source == "llm":
                self._llm_calls["diagnose"] += 1
            logger.info("[%s] step=%d diagnose: %s (conf=%.2f, src=%s)",
                        self.run_id, step, diagnosis.bottleneck,
                        diagnosis.confidence, diagnosis.source)

            # ---- Propose ----
            proposal = propose(
                self.client, self.tools, self.memory, diagnosis,
                current_cfg, max_rounds=self.max_propose_rounds,
                use_llm=self.use_llm,
            )
            if isinstance(proposal, StopSignal):
                stop_reason = f"P-LLM stop: {proposal.reason}"
                logger.info("[%s] P-LLM 主动终止: %s", self.run_id, proposal.reason)
                break
            if proposal.source == "llm":
                self._llm_calls["propose"] += 1

            # ---- Safety: Judge.check_delta ----
            verdict = self.judge.check_delta(proposal, current_config=current_cfg)
            if not verdict.pass_:
                self.memory.record_rejected(
                    {proposal.param: proposal.new_value}, verdict.reason,
                )
                logger.info("[%s] step=%d Judge 拒绝: %s", self.run_id, step, verdict.reason)
                step += 1
                continue

            # ---- Act: 跑 trial ----
            next_cfg = apply_delta(current_cfg, proposal)
            try:
                metrics = self.runner_fn(
                    next_cfg,
                    baseline_throughput_tok_per_s=baseline_metrics.throughput_tok_per_s,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] runner 异常: %s", self.run_id, e)
                self.memory.record_rejected(next_cfg, f"runner 异常: {e}")
                step += 1
                continue

            new_record = TrialRecord(
                trial_id=f"{self.run_id}_step_{step:03d}",
                config=next_cfg, metrics=metrics, source="agent",
                notes=[proposal.reason] if proposal.reason else [],
            )
            self.memory.append(new_record)

            # ---- 约束检查 ----
            cc = self.judge.check_trial_constraints(metrics, baseline_metrics)

            # ---- Reflect ----
            reflection = reflect(
                self.client, proposal, prev_record, new_record,
                cc.to_dict(), self.memory.notes, use_llm=self.use_llm,
            )
            if reflection.source == "llm":
                self._llm_calls["reflect"] += 1
            if reflection.new_note:
                self.memory.add_note(reflection.new_note)
            logger.info("[%s] step=%d reflect: %s (hint=%s, src=%s)",
                        self.run_id, step, reflection.verdict,
                        reflection.next_move_hint, reflection.source)

            # ---- accept / rollback ----
            if reflection.verdict == "accept" and cc.pass_:
                current_cfg = next_cfg
                prev_record = new_record
            else:
                self.memory.record_rejected(
                    {proposal.param: proposal.new_value}, reflection.reason,
                )
                # current_cfg 不变，prev_record 也保持上一个 accepted 的

            if reflection.next_move_hint == "stop":
                stop_reason = f"R-LLM 建议停止: {reflection.reason}"
                break

            step += 1
        else:
            stop_reason = stop_reason or f"达到 max_steps={max_steps}"

        wall = round(time.perf_counter() - t0, 3)
        report = AgentReport(
            run_id=self.run_id,
            n_steps=step,
            best_trial=self.memory.best(),
            baseline=baseline_metrics,
            total_wall_time_s=wall,
            llm_call_counts=dict(self._llm_calls),
            memory_path=str(self.memory.path) if self.memory.path else "",
            stop_reason=stop_reason or "loop_end",
        )
        return report
