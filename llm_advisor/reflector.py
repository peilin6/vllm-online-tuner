#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reflector.py — R-LLM Reflector

Task 7.3。给定 ConfigDelta + 改动前后 trial + 约束检查结果，输出 verdict + new_note +
next_move_hint。LLM 失败时降级为规则版（基于吞吐增幅 + SLO 违反 决定 accept/reject）。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from tuner.memory import TrialRecord

from .prompts import R_LLM_SYSTEM, R_LLM_USER_TMPL
from .schemas import ParseError, ReflectionResult

logger = logging.getLogger(__name__)


ACCEPT_DELTA_PCT = 0.03   # 提升 ≥3% 视为 accept
REJECT_DELTA_PCT = -0.01  # 退步 >1% 视为 reject


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _record_to_view(rec: TrialRecord) -> dict:
    m = rec.metrics
    return {
        "trial_id": rec.trial_id,
        "config": rec.config,
        "throughput_tok_per_s": round(m.throughput_tok_per_s, 2),
        "ttft_p95_ms": round(m.ttft_p95_ms, 1),
        "tpot_p95_ms": round(m.tpot_p95_ms, 1),
        "latency_p95_ms": round(m.latency_p95_ms, 1),
        "preempt_rate_per_min": round(m.preemption_rate_per_min, 2),
        "kv_p95_pct": round(m.kv_cache_usage_p95_pct, 3),
        "success": bool(m.success),
        "early_killed": bool(m.early_killed),
    }


def _fallback_reflect(
    proposal,                                   # ConfigDelta
    prev: TrialRecord, new: TrialRecord,
    constraint_check: dict,
) -> ReflectionResult:
    prev_t = prev.metrics.throughput_tok_per_s
    new_t = new.metrics.throughput_tok_per_s
    pass_constraint = bool(constraint_check.get("pass", True))
    delta_pct = ((new_t - prev_t) / prev_t) if prev_t > 0 else 0.0

    if not pass_constraint:
        verdict = "reject"
        reason = f"违反 SLO：{constraint_check.get('violations', [])}"
        hint = "rollback"
    elif new.metrics.early_killed:
        verdict = "reject"
        reason = "trial 被早停"
        hint = "explore_other"
    elif delta_pct >= ACCEPT_DELTA_PCT:
        verdict = "accept"
        reason = f"吞吐 +{delta_pct*100:.1f}% 且无 SLO 违反"
        hint = "double_down"
    elif delta_pct <= REJECT_DELTA_PCT:
        verdict = "reject"
        reason = f"吞吐 {delta_pct*100:+.1f}% 退步"
        hint = "rollback"
    else:
        verdict = "partial"
        reason = f"吞吐变化 {delta_pct*100:+.1f}% 不显著"
        hint = "explore_other"

    note = (
        f"{getattr(proposal, 'param', 'unknown')} "
        f"{getattr(proposal, 'old_value', '?')}→{getattr(proposal, 'new_value', '?')} "
        f"在该 workload 上 verdict={verdict}"
    )
    return ReflectionResult(
        verdict=verdict, reason=reason, new_note=note,
        next_move_hint=hint, hint_detail=reason, source="fallback",
    )


def _extract_content(resp: dict) -> str:
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ParseError(f"LLM 响应结构异常: {e}")


def reflect(
    client: Any,
    proposal,                                   # ConfigDelta
    prev: TrialRecord, new: TrialRecord,
    constraint_check: dict,
    notes: list[str],
    *,
    use_llm: bool = True,
) -> ReflectionResult:
    if client is None or not use_llm:
        return _fallback_reflect(proposal, prev, new, constraint_check)

    user = R_LLM_USER_TMPL.format(
        proposal_json=json.dumps(
            proposal.to_dict() if hasattr(proposal, "to_dict") else dict(proposal),
            ensure_ascii=False),
        prev_trial_json=json.dumps(_record_to_view(prev), ensure_ascii=False),
        new_trial_json=json.dumps(_record_to_view(new), ensure_ascii=False),
        constraint_check_json=json.dumps(constraint_check, ensure_ascii=False),
        notes_list=json.dumps(list(notes[-10:]), ensure_ascii=False),
    )
    messages = [
        {"role": "system", "content": R_LLM_SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        resp = client.chat(
            messages, response_format={"type": "json_object"}, use_cache=False,
        )
        content = _strip_fence(_extract_content(resp))
        raw = json.loads(content)
        return ReflectionResult.from_dict(raw)
    except (ParseError, json.JSONDecodeError, RuntimeError) as e:
        logger.warning("R-LLM 失败，走 fallback: %s | raw=%r", e, locals().get("content"))
        return _fallback_reflect(proposal, prev, new, constraint_check)
