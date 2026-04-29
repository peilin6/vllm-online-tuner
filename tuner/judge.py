#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""judge.py — 配置安全闸 + trial 后约束检查 + 终止条件

Task 7.4。Judge 完全基于 ParamRegistry + ExperienceMemory 做规则判定，不调 LLM。

三类入口：
1. check_delta(ConfigDelta) -> JudgeVerdict      # P-LLM 提议落地前的安全闸
2. check_trial_constraints(metrics, baseline)    # trial 跑完后的 SLO 检查
3. should_terminate(memory, n_steps)             # 主循环终止条件
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import param_registry
from .memory import ExperienceMemory, TrialRecord
from .metrics_parser import TrialMetrics


@dataclass
class JudgeVerdict:
    """check_delta 的结果。"""
    pass_: bool                     # `pass` 是关键字，故加下划线
    reason: str = ""
    suggestion_for_retry: str = ""

    def to_dict(self) -> dict:
        return {"pass": self.pass_, "reason": self.reason,
                "suggestion_for_retry": self.suggestion_for_retry}


@dataclass
class ConstraintCheck:
    """check_trial_constraints 的结果。"""
    pass_: bool
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"pass": self.pass_, "violations": list(self.violations)}


class Judge:
    """规则裁判，不调用 LLM。"""

    def __init__(
        self,
        memory: ExperienceMemory,
        *,
        max_steps: int = 25,
        slo_ttft_mult: float = 1.2,
        slo_lat_mult: float = 1.2,
        slo_preempt_per_min: float = 5.0,
        recent_rejected_window: int = 3,
        converge_window: int = 3,
        converge_score_pct: float = 0.02,
        slo_headroom_pct: float = 0.05,
    ):
        self.memory = memory
        self.max_steps = int(max_steps)
        self.slo_ttft_mult = float(slo_ttft_mult)
        self.slo_lat_mult = float(slo_lat_mult)
        self.slo_preempt_per_min = float(slo_preempt_per_min)
        self.recent_rejected_window = int(recent_rejected_window)
        self.converge_window = int(converge_window)
        self.converge_score_pct = float(converge_score_pct)
        self.slo_headroom_pct = float(slo_headroom_pct)

    # ------------------------------------------------------------------
    # 1) ConfigDelta 安全闸
    # ------------------------------------------------------------------
    def check_delta(self, delta: Any, current_config: dict | None = None) -> JudgeVerdict:
        """ConfigDelta（duck-typed: 至少有 param/new_value）。"""
        param = getattr(delta, "param", None)
        new_value = getattr(delta, "new_value", None)
        if not param:
            return JudgeVerdict(False, "delta.param 为空")

        # a) 必须是登记过的 RESTART 参数
        try:
            spec = param_registry.get_spec(param)
        except KeyError:
            return JudgeVerdict(
                False, f"参数 {param} 未登记",
                suggestion_for_retry="改用 list_params 工具列出的参数",
            )

        # b) new_value 必须在 candidates / range 内
        if not spec.in_range(new_value):
            return JudgeVerdict(
                False,
                f"{param}={new_value!r} 越界 (candidates={spec.candidates}, range={spec.range})",
                suggestion_for_retry=f"改在 candidates={spec.candidates} 中选取",
            )

        # c) 不能在最近 N 次 rejected_proposals 中出现完全相同 (param,new_value)
        recent_rej = self.memory.rejected_proposals[-self.recent_rejected_window:]
        for cfg, _reason in recent_rej:
            if cfg.get(param) == new_value:
                return JudgeVerdict(
                    False, f"{param}={new_value!r} 最近被拒绝过",
                    suggestion_for_retry="换参数或换值",
                )

        # d) 不能与当前 best 完全重复（即提议没变化）
        best = self.memory.best()
        if best is not None and best.config.get(param) == new_value:
            # 同时和 current 也相同 → 等于没改
            cur_val = (current_config or {}).get(param,
                       best.config.get(param))
            if cur_val == new_value:
                return JudgeVerdict(
                    False, f"{param} 已经是 {new_value!r}，提议无变化",
                    suggestion_for_retry="尝试相邻 candidate",
                )

        return JudgeVerdict(True, "ok")

    # ------------------------------------------------------------------
    # 2) trial 后 SLO 检查
    # ------------------------------------------------------------------
    def check_trial_constraints(
        self,
        metrics: TrialMetrics,
        baseline: TrialMetrics | None,
    ) -> ConstraintCheck:
        violations: list[str] = []

        if not metrics.success or metrics.early_killed:
            violations.append(
                f"trial 失败/早停 (success={metrics.success}, early_killed={metrics.early_killed})"
            )

        if baseline is not None and baseline.ttft_p95_ms > 0 and metrics.ttft_p95_ms > 0:
            if metrics.ttft_p95_ms > baseline.ttft_p95_ms * self.slo_ttft_mult:
                violations.append(
                    f"ttft_p95 {metrics.ttft_p95_ms:.0f}ms > baseline×{self.slo_ttft_mult} "
                    f"({baseline.ttft_p95_ms * self.slo_ttft_mult:.0f}ms)"
                )

        if baseline is not None and baseline.latency_p95_ms > 0 and metrics.latency_p95_ms > 0:
            if metrics.latency_p95_ms > baseline.latency_p95_ms * self.slo_lat_mult:
                violations.append(
                    f"latency_p95 {metrics.latency_p95_ms:.0f}ms > baseline×{self.slo_lat_mult}"
                )

        if metrics.preemption_rate_per_min > self.slo_preempt_per_min:
            violations.append(
                f"preempt_rate {metrics.preemption_rate_per_min:.2f}/min > "
                f"{self.slo_preempt_per_min}/min"
            )

        return ConstraintCheck(pass_=len(violations) == 0, violations=violations)

    # ------------------------------------------------------------------
    # 3) 主循环终止条件
    # ------------------------------------------------------------------
    def should_terminate(
        self, memory: ExperienceMemory, n_steps: int,
    ) -> tuple[bool, str]:
        if n_steps >= self.max_steps:
            return True, f"达到 max_steps={self.max_steps}"

        recent = memory.recent_n(self.converge_window)
        if len(recent) >= self.converge_window:
            scores = [r.metrics.throughput_tok_per_s for r in recent
                      if r.metrics.success and not r.metrics.early_killed]
            if len(scores) >= 2:
                lo, hi = min(scores), max(scores)
                if lo > 0 and (hi - lo) / lo < self.converge_score_pct:
                    return True, (
                        f"最近 {self.converge_window} 次吞吐变化 "
                        f"<{self.converge_score_pct*100:.1f}%"
                    )

        return False, ""

    # ------------------------------------------------------------------
    # 4) trial 进行中早停辅助（可选，给 Runner 用）
    # ------------------------------------------------------------------
    def should_early_stop_trial(self, intermediate: dict) -> tuple[bool, str]:
        """intermediate: {'preempt_rate_per_s': ..., 'kv_pct': ...}"""
        pr = float(intermediate.get("preempt_rate_per_s", 0.0))
        kv = float(intermediate.get("kv_pct", 0.0))
        if pr > 2.0:
            return True, f"preempt_rate {pr:.2f}/s > 2/s"
        if kv > 0.98:
            return True, f"kv_usage {kv:.2%} > 98%"
        return False, ""
