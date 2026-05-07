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


def _missing_fields(m: TrialMetrics) -> list[str]:
    """返回值=-1 的字段名清单，提示 LLM 不得据此推断。"""
    out = []
    if m.throughput_tok_per_s < 0: out.append("throughput_tok_per_s")
    if m.ttft_p95_ms < 0:          out.append("ttft_p95_ms")
    if m.tpot_p95_ms < 0:          out.append("tpot_p95_ms")
    if m.latency_p95_ms < 0:       out.append("latency_p95_ms")
    if m.kv_cache_usage_p95_pct < 0: out.append("kv_cache_usage_p95_pct")
    if m.queue_time_p95_ms < 0:    out.append("queue_time_p95_ms")
    return out


def _recent_score_swing(memory: ExperienceMemory, n: int = 3) -> float:
    """最近 n 个 trial 的 throughput 极差 / 均值；用于 R7 收敛判断。"""
    recs = memory.recent_n(n)
    vals = [r.metrics.throughput_tok_per_s for r in recs
            if r.metrics.throughput_tok_per_s > 0]
    if len(vals) < n:
        return -1.0
    avg = sum(vals) / len(vals)
    if avg <= 0:
        return -1.0
    return (max(vals) - min(vals)) / avg


def _build_user_prompt(
    *, latest: TrialRecord, baseline: TrialMetrics | None,
    memory: ExperienceMemory, ttft_slo_ms: float, lat_slo_ms: float,
) -> str:
    base_tput = baseline.throughput_tok_per_s if baseline else -1.0
    base_ttft = baseline.ttft_p95_ms if baseline else -1.0
    base_tpot = baseline.tpot_p95_ms if baseline else -1.0
    m = latest.metrics
    qt_ratio = "n/a"
    if m.queue_time_p95_ms >= 0 and m.latency_p95_ms > 0:
        qt_ratio = f"{m.queue_time_p95_ms / m.latency_p95_ms:.0%}"
    swing = _recent_score_swing(memory, n=3)
    swing_str = "n/a" if swing < 0 else f"{swing:.2%}"
    missing = _missing_fields(m)
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
        base_ttft=round(base_ttft, 1),
        ttft_pct=_pct_change(m.ttft_p95_ms, base_ttft),
        tpot=round(m.tpot_p95_ms, 1),
        base_tpot=round(base_tpot, 1),
        tpot_pct=_pct_change(m.tpot_p95_ms, base_tpot),
        lat=round(m.latency_p95_ms, 1),
        preempt=round(m.preemption_rate_per_min, 2),
        kv=round(m.kv_cache_usage_p95_pct, 3),
        qt=round(m.queue_time_p95_ms, 1),
        qt_ratio=qt_ratio,
        early_killed=m.early_killed,
        ttft_headroom=f"{_headroom(m.ttft_p95_ms, ttft_slo_ms):.1%}",
        lat_headroom=f"{_headroom(m.latency_p95_ms, lat_slo_ms):.1%}",
        score_swing=swing_str,
        missing_fields=(missing or "无"),
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
    """规则版：与 prompt 中的"证据→瓶颈"表保持一致，按优先级匹配。"""
def _fallback_diagnose(
    latest: TrialRecord, baseline: TrialMetrics | None,
    *,
    memory: ExperienceMemory | None = None,
    ttft_slo_ms: float = DEFAULT_TTFT_SLO_MS,
    lat_slo_ms: float = DEFAULT_LAT_SLO_MS,
) -> DiagnosisResult:
    """规则版：与 prompt 中的 R1→R8 完全一致，按优先级匹配。"""
    m = latest.metrics
    base_tput = baseline.throughput_tok_per_s if baseline else 0.0
    base_ttft = baseline.ttft_p95_ms if baseline else 0.0
    base_tpot = baseline.tpot_p95_ms if baseline else 0.0
    ttft_hr = _headroom(m.ttft_p95_ms, ttft_slo_ms)
    lat_hr = _headroom(m.latency_p95_ms, lat_slo_ms)

    # R1. preempt_storm
    if m.preemption_rate_per_min > 5:
        conf = 0.9 if (lat_hr >= 0 and lat_hr < 0.10) else 0.8
        return DiagnosisResult(
            bottleneck="preempt_storm", confidence=conf,
            evidence=f"preempt_rate={m.preemption_rate_per_min:.2f}/min > 5 (R1)",
            hypothesis="↑gpu_memory_utilization 给 KV 腾空间，或 ↓max_num_seqs/↓max_num_batched_tokens 降并发",
            slo_pressure="high", should_stop=False, source="fallback",
        )
    # R2. kv_cache_pressure
    if m.kv_cache_usage_p95_pct >= 0 and m.kv_cache_usage_p95_pct > 0.95:
        return DiagnosisResult(
            bottleneck="kv_cache_pressure", confidence=0.8,
            evidence=f"kv_p95={m.kv_cache_usage_p95_pct:.2f} > 0.95 (R2)",
            hypothesis="优先 ↑gpu_memory_utilization，必要时 ↓max_num_seqs / ↓max_num_batched_tokens",
            slo_pressure="medium", should_stop=False, source="fallback",
        )
    # R3. queue_backlog
    if m.queue_time_p95_ms > 0 and m.latency_p95_ms > 0 \
            and m.queue_time_p95_ms > m.latency_p95_ms * 0.3:
        ratio = m.queue_time_p95_ms / m.latency_p95_ms
        return DiagnosisResult(
            bottleneck="queue_backlog", confidence=0.6,
            evidence=f"queue_time_p95={m.queue_time_p95_ms:.0f}ms / latency={m.latency_p95_ms:.0f}ms = "
                     f"{ratio:.0%} > 30% (R3)",
            hypothesis="↑max_num_batched_tokens / ↑max_num_seqs 提升并发吞吐",
            slo_pressure="medium", should_stop=False, source="fallback",
        )
    # R4. slo_margin_low（headroom 已贴近 SLO 上限）
    if (ttft_hr >= 0 and ttft_hr < 0.10) or (lat_hr >= 0 and lat_hr < 0.10):
        # confidence: headroom 越低越高
        worst = min(x for x in [ttft_hr, lat_hr] if x >= 0)
        conf = 0.7 if worst < 0.05 else 0.6
        return DiagnosisResult(
            bottleneck="slo_margin_low", confidence=conf,
            evidence=f"ttft_headroom={ttft_hr:.1%} / lat_headroom={lat_hr:.1%} 任一<10% (R4)",
            hypothesis="↓max_num_batched_tokens 保护 P99，或回退到 baseline 邻域细调",
            slo_pressure="high", should_stop=False, source="fallback",
        )
    # R5/R6. prefill_bound / decode_bound（需要 baseline）
    if base_ttft > 0 and base_tpot > 0 and m.ttft_p95_ms >= 0 and m.tpot_p95_ms >= 0:
        ttft_up = (m.ttft_p95_ms - base_ttft) / base_ttft
        tpot_up = (m.tpot_p95_ms - base_tpot) / base_tpot
        if ttft_up >= 0.20 and tpot_up < 0.10:
            return DiagnosisResult(
                bottleneck="prefill_bound", confidence=0.6,
                evidence=f"ttft {ttft_up:+.0%} vs baseline ≥20%, tpot {tpot_up:+.0%} <10% (R5)",
                hypothesis="↑max_num_batched_tokens 并开启 chunked prefill / prefix caching",
                slo_pressure="medium", should_stop=False, source="fallback",
            )
        if tpot_up >= 0.20 and ttft_up < 0.10:
            return DiagnosisResult(
                bottleneck="decode_bound", confidence=0.6,
                evidence=f"tpot {tpot_up:+.0%} vs baseline ≥20%, ttft {ttft_up:+.0%} <10% (R6)",
                hypothesis="↓max_num_batched_tokens 减少 prefill 对 decode 干扰",
                slo_pressure="medium", should_stop=False, source="fallback",
            )
    # R7. converged: 最近 3 步极差 <2% 且 SLO 余量充足
    if memory is not None:
        swing = _recent_score_swing(memory, n=3)
        slo_ok = (ttft_hr < 0 or ttft_hr >= 0.10) and (lat_hr < 0 or lat_hr >= 0.10)
        if 0 <= swing < 0.02 and slo_ok:
            return DiagnosisResult(
                bottleneck="converged", confidence=0.7,
                evidence=f"recent3 swing={swing:.2%} <2% & SLO 余量充足 (R7)",
                hypothesis="已收敛，建议停止",
                slo_pressure="low", should_stop=True, source="fallback",
            )
    # R8. underutilized 兜底
    return DiagnosisResult(
        bottleneck="underutilized", confidence=0.4,
        evidence="preempt/kv/queue/SLO 均健康，无显著 prefill/decode 信号 (R8)",
        hypothesis="↑max_num_batched_tokens / ↑max_num_seqs 挖掘吞吐",
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
        return _fallback_diagnose(
            latest, baseline, memory=memory,
            ttft_slo_ms=ttft_slo_ms, lat_slo_ms=lat_slo_ms,
        )
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
        return _fallback_diagnose(
            latest, baseline, memory=memory,
            ttft_slo_ms=ttft_slo_ms, lat_slo_ms=lat_slo_ms,
        )
