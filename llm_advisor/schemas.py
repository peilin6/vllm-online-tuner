#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""schemas.py — A/P/R-LLM 输出结构 & 共享异常。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ParseError(ValueError):
    """LLM 输出无法解析为预期结构。"""


# ============================================================
# A-LLM Diagnoser
# ============================================================
BOTTLENECK_ENUM = {
    "prefill_bound", "decode_bound", "kv_cache_pressure",
    "preempt_storm", "queue_backlog", "underutilized",
    "slo_margin_low", "converged",
}

SLO_PRESSURE_ENUM = {"low", "medium", "high"}


@dataclass
class DiagnosisResult:
    bottleneck: str
    confidence: float
    evidence: str = ""
    hypothesis: str = ""
    slo_pressure: str = "low"
    should_stop: bool = False
    source: str = "llm"   # llm / fallback

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnosisResult":
        b = d.get("bottleneck")
        if b not in BOTTLENECK_ENUM:
            raise ParseError(f"非法 bottleneck: {b}")
        sp = d.get("slo_pressure", "low")
        if sp not in SLO_PRESSURE_ENUM:
            sp = "low"
        try:
            conf = float(d.get("confidence", 0.0))
        except (TypeError, ValueError):
            raise ParseError(f"confidence 不可解析: {d.get('confidence')}")
        return cls(
            bottleneck=b,
            confidence=max(0.0, min(1.0, conf)),
            evidence=str(d.get("evidence", "")),
            hypothesis=str(d.get("hypothesis", "")),
            slo_pressure=sp,
            should_stop=bool(d.get("should_stop", False)),
            source=str(d.get("source", "llm")),
        )


# ============================================================
# P-LLM Proposer
# ============================================================
@dataclass
class ConfigDelta:
    param: str
    old_value: Any
    new_value: Any
    hypothesis_ref: str = ""
    tools_used: list[str] = field(default_factory=list)
    reason: str = ""
    expected_effect: dict = field(default_factory=dict)
    rollback_if: str = ""
    source: str = "llm"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StopSignal:
    reason: str
    source: str = "llm"

    def to_dict(self) -> dict:
        return {"action": "stop", "reason": self.reason, "source": self.source}


def parse_proposal(raw: dict) -> "ConfigDelta | StopSignal":
    action = raw.get("action", "change_param")
    if action == "stop":
        return StopSignal(reason=str(raw.get("reason", "")),
                          source=str(raw.get("source", "llm")))
    if "param" not in raw or "new_value" not in raw:
        raise ParseError("ConfigDelta 缺少 param / new_value")
    return ConfigDelta(
        param=str(raw["param"]),
        old_value=raw.get("old_value"),
        new_value=raw["new_value"],
        hypothesis_ref=str(raw.get("hypothesis_ref", "")),
        tools_used=list(raw.get("tools_used", []) or []),
        reason=str(raw.get("reason", "")),
        expected_effect=dict(raw.get("expected_effect") or {}),
        rollback_if=str(raw.get("rollback_if", "")),
        source=str(raw.get("source", "llm")),
    )


# ============================================================
# R-LLM Reflector
# ============================================================
VERDICT_ENUM = {"accept", "partial", "reject"}
HINT_ENUM = {"double_down", "explore_other", "rollback", "stop"}


@dataclass
class ReflectionResult:
    verdict: str
    reason: str = ""
    new_note: str = ""
    next_move_hint: str = "explore_other"
    hint_detail: str = ""
    source: str = "llm"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReflectionResult":
        v = d.get("verdict")
        if v not in VERDICT_ENUM:
            raise ParseError(f"非法 verdict: {v}")
        h = d.get("next_move_hint", "explore_other")
        if h not in HINT_ENUM:
            h = "explore_other"
        return cls(
            verdict=v,
            reason=str(d.get("reason", "")),
            new_note=str(d.get("new_note", "")),
            next_move_hint=h,
            hint_detail=str(d.get("hint_detail", "")),
            source=str(d.get("source", "llm")),
        )
