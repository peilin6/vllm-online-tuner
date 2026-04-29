#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diagnoser.py — A-LLM Diagnoser

Task 7.1。输入 latest TrialRecord + memory，输出 DiagnosisResult。
LLM 调用失败时降级为规则版（preempt_storm / kv_cache_pressure / underutilized）。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from tuner.memory import ExperienceMemory, TrialRecord
from tuner.metrics_parser import TrialMetrics

from .prompts import A_LLM_SYSTEM, A_LLM_USER_TMPL
from .schemas import DiagnosisResult, ParseError

logger = logging.getLogger(__name__)


# 默认 SLO 上限（可由 agent.config 覆盖）
DEFAULT_TTFT_SLO_MS = 300.0
DEFAULT_LAT_SLO_MS = 5000.0


def _headroom(value: float, limit: float) -> float:
    if value < 0 or limit <= 0:
        return -1.0
    return max(0.0, (limit - value) / limit)


def _pct_change(cur: float, base: float) -> str:
    if base <= 0 or cur < 0:
        return "n/a"
    return f"{(cur - base) / base * 100:+.1f}%"


def _build_user_prompt(
    *, latest: TrialRecord, baseline: TrialMetrics | None,
    memory: ExperienceMemory, ttft_slo_ms: float, lat_slo_ms: float,
) -> str:
    base_tput = baseline.throughput_tok_per_s if baseline else -1.0
    m = latest.metrics
    return A_LLM_USER_TMPL.format(
        baseline_metrics_json=json.dumps(
            baseline.to_dict() if baseline else {}, ensure_ascii=False),
        memory_summary=json.dumps(memory.summarize(top_k=3, recent_n=5),
                                  ensure_ascii=False),
        trial_id=latest.trial_id,
        config_json=json.dumps(latest.config, ensure_ascii=False),
        tput=round(m.throughput_tok_per_s, 2),
        base_tput=round(base_tput, 2),
        tput_pct=_pct_change(m.throughput_tok_per_s, base_tput),
        ttft=round(m.ttft_p95_ms, 1),
        lat=round(m.latency_p95_ms, 1),
        tpot=round(m.tpot_p95_ms, 1),
        preempt=round(m.preemption_rate_per_min, 2),
        kv=round(m.kv_cache_usage_p95_pct, 3),
        qt=round(m.queue_time_p95_ms, 1),
        early_killed=m.early_killed,
        ttft_headroom=f"{_headroom(m.ttft_p95_ms, ttft_slo_ms):.1%}",
        lat_headroom=f"{_headroom(m.latency_p95_ms, lat_slo_ms):.1%}",
    )


def _extract_content(resp: dict) -> str:
    """从 OpenAI 格式响应中拿 content。"""
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ParseError(f"LLM 响应结构异常: {e}")


def _strip_markdown_fence(s: str) -> str:
    """LLM 偶尔会用 ```json ... ``` 包裹，这里兜底剥掉。"""
    s = s.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _fallback_diagnose(latest: TrialRecord, baseline: TrialMetrics | None) -> DiagnosisResult:
    """规则版：纯数值判断，给出可解释的 bottleneck。"""
    m = latest.metrics
    if m.preemption_rate_per_min > 5:
        return DiagnosisResult(
            bottleneck="preempt_storm", confidence=0.7,
            evidence=f"preempt_rate={m.preemption_rate_per_min:.2f}/min > 5",
            hypothesis="降低 max_num_seqs 或开启 enable_chunked_prefill 降抢占",
            slo_pressure="high", should_stop=False, source="fallback",
        )
    if m.kv_cache_usage_p95_pct >= 0 and m.kv_cache_usage_p95_pct > 0.95:
        return DiagnosisResult(
            bottleneck="kv_cache_pressure", confidence=0.7,
            evidence=f"kv_p95={m.kv_cache_usage_p95_pct:.2f} > 0.95",
            hypothesis="降低 max_num_seqs 或 max_num_batched_tokens 释放 KV",
            slo_pressure="medium", should_stop=False, source="fallback",
        )
    base_tput = baseline.throughput_tok_per_s if baseline else 0.0
    if base_tput > 0 and m.throughput_tok_per_s < base_tput * 0.95:
        return DiagnosisResult(
            bottleneck="slo_margin_low", confidence=0.5,
            evidence=f"throughput {m.throughput_tok_per_s:.1f} < baseline {base_tput:.1f}",
            hypothesis="回退到 baseline 配置或在邻域细调",
            slo_pressure="medium", should_stop=False, source="fallback",
        )
    return DiagnosisResult(
        bottleneck="underutilized", confidence=0.4,
        evidence="无明显瓶颈信号",
        hypothesis="尝试增大 max_num_seqs 提升并发",
        slo_pressure="low", should_stop=False, source="fallback",
    )


def diagnose(
    client: Any,                                   # LlmClient duck-typed
    memory: ExperienceMemory,
    latest: TrialRecord,
    *,
    baseline: TrialMetrics | None = None,
    ttft_slo_ms: float = DEFAULT_TTFT_SLO_MS,
    lat_slo_ms: float = DEFAULT_LAT_SLO_MS,
    use_llm: bool = True,
) -> DiagnosisResult:
    """主入口。LLM 失败 / 解析失败时自动降级到 fallback。

    Args:
        client: 任意提供 .chat(messages, ...) -> OpenAI dict 的对象；可为 None 强制 fallback。
        memory: 当前记忆。
        latest: 最近一次 TrialRecord。
        baseline: 用于打 baseline 对比；缺省时取 memory 中 source='baseline' 的记录。
    """
    if baseline is None:
        for r in memory.all():
            if r.source == "baseline":
                baseline = r.metrics
                break
    if client is None or not use_llm:
        return _fallback_diagnose(latest, baseline)
    user = _build_user_prompt(
        latest=latest, baseline=baseline, memory=memory,
        ttft_slo_ms=ttft_slo_ms, lat_slo_ms=lat_slo_ms,
    )
    messages = [
        {"role": "system", "content": A_LLM_SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        resp = client.chat(
            messages,
            response_format={"type": "json_object"},
            use_cache=False,
        )
        content = _strip_markdown_fence(_extract_content(resp))
        raw = json.loads(content)
        return DiagnosisResult.from_dict(raw)
    except (ParseError, json.JSONDecodeError, RuntimeError) as e:
        logger.warning("A-LLM 失败，走 fallback: %s", e)
        return _fallback_diagnose(latest, baseline)
