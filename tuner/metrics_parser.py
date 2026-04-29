#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_parser.py — 把 results/<exp_id>/ 目录解析成 TrialMetrics

Task 6.3。Runner 跑完一个 trial 后调 parse_trial(exp_dir) 即可得到 TrialMetrics；
不重新计算压测，只搬运 summary.json + request_trace.jsonl 的字段。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrialMetrics:
    """一次 trial 的全部 trial-level 标量指标。供 agent 决策与 memory 存储。"""
    throughput_req_per_s: float = -1.0
    throughput_tok_per_s: float = -1.0
    ttft_p95_ms: float = -1.0
    tpot_p95_ms: float = -1.0
    latency_p95_ms: float = -1.0
    preemptions_total: int = 0
    preemption_rate_per_min: float = 0.0
    kv_cache_usage_p95_pct: float = -1.0
    queue_time_p95_ms: float = -1.0
    success: bool = False
    early_killed: bool = False
    wall_time_s: float = 0.0
    # 附加诊断字段（不强制）
    success_rate: float = 0.0
    total_requests: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_get(d: dict, path: str, default=None):
    """以点路径安全取值。"""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_trial(
    exp_dir: str | Path,
    *,
    early_killed: bool = False,
    wall_time_s: float | None = None,
) -> TrialMetrics:
    """从 results/<exp_id>/ 目录解析 TrialMetrics。

    Args:
        exp_dir: 形如 results/baseline_a6000_0/ 的目录，必须含 summary.json。
        early_killed: 由 Runner 传入；本函数不会自己判断早停。
        wall_time_s: 由 Runner 传入；为 None 时回退到 summary.wall_time_s。

    Raises:
        FileNotFoundError: 目录或 summary.json 不存在。
    """
    exp_dir = Path(exp_dir)
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json 不存在: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    metrics = TrialMetrics(
        throughput_req_per_s=float(summary.get("throughput_rps", -1.0)),
        throughput_tok_per_s=float(summary.get("token_throughput_tps", -1.0)),
        ttft_p95_ms=float(_safe_get(summary, "ttft_ms.p95", -1.0)),
        tpot_p95_ms=float(_safe_get(summary, "tpot_ms.p95", -1.0)),
        latency_p95_ms=float(_safe_get(summary, "latency_ms.p95", -1.0)),
        preemptions_total=int(_safe_get(summary, "vllm_aggregates.preemptions_total", 0) or 0),
        preemption_rate_per_min=float(
            _safe_get(summary, "vllm_aggregates.preemption_rate_per_min", 0.0) or 0.0
        ),
        kv_cache_usage_p95_pct=float(
            _safe_get(summary, "vllm_aggregates.kv_cache_usage_p95_pct", -1.0)
        ),
        queue_time_p95_ms=_derive_queue_time_p95_ms(summary, exp_dir),
        success=not early_killed and float(summary.get("success_rate", 0.0)) >= 0.95,
        early_killed=bool(early_killed),
        wall_time_s=float(wall_time_s if wall_time_s is not None
                          else summary.get("wall_time_s", 0.0)),
        success_rate=float(summary.get("success_rate", 0.0)),
        total_requests=int(summary.get("total_requests", 0)),
    )
    return metrics


def _derive_queue_time_p95_ms(summary: dict, exp_dir: Path) -> float:
    """vLLM 的 time_in_queue 是 histogram，目前 collector 只采到 _sum；
    退化方案: 把 queue_time_delta_s 平均到成功请求数转成毫秒，作为 P95 的粗估。
    若 request_trace.jsonl 含每请求 queue_time 则优先取真 P95（未来扩展）。"""
    delta_s = _safe_get(summary, "vllm_aggregates.queue_time_delta_s", -1.0)
    if delta_s is None or delta_s < 0:
        return -1.0
    n = int(summary.get("successful", 0)) or int(summary.get("total_requests", 0))
    if n <= 0:
        return -1.0
    avg_ms = delta_s / n * 1000.0
    # 粗估：P95 ≈ 平均 × 2（典型长尾经验系数）；待真实 histogram 接入后替换
    return round(avg_ms * 2.0, 2)
