#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_metrics_aggregation.py — Task 6.2 单元测试

覆盖:
- vllm_metrics_collector 解析新增字段 queue_time_seconds_sum
- run_benchmark._aggregate_vllm_samples 把时序聚合为 trial-level 标量
- compute_stats 在 vllm_samples=None 时仍能返回完整 stats（向后兼容）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitors.vllm_metrics_collector import VllmMetricsCollector
from benchmarks.run_benchmark import compute_stats, _aggregate_vllm_samples


# ============================================================
# Collector: 新字段 queue_time_seconds_sum 解析
# ============================================================

_PROM_FIXTURE = """\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen3-8B"} 4.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="Qwen3-8B"} 2.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="Qwen3-8B"} 0.62
# HELP vllm:cpu_cache_usage_perc CPU KV-cache usage.
# TYPE vllm:cpu_cache_usage_perc gauge
vllm:cpu_cache_usage_perc{model_name="Qwen3-8B"} 0.0
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="Qwen3-8B"} 12.0
# HELP vllm:time_in_queue_requests_seconds Histogram of queue wait time per request.
# TYPE vllm:time_in_queue_requests_seconds histogram
vllm:time_in_queue_requests_seconds_count{model_name="Qwen3-8B"} 50.0
vllm:time_in_queue_requests_seconds_sum{model_name="Qwen3-8B"} 12.345
"""


def test_parse_new_queue_time_field():
    sample = VllmMetricsCollector._parse_prometheus(_PROM_FIXTURE, timestamp_s=1.0)
    assert sample["source"] == "vllm"
    assert sample["num_requests_running"] == 4
    assert sample["num_preemptions_total"] == 12
    assert sample["gpu_cache_usage_pct"] == 0.62
    assert sample["queue_time_seconds_sum"] == 12.345


def test_parse_missing_queue_time_marked_negative():
    minimal = "vllm:num_requests_running 1.0\n"
    sample = VllmMetricsCollector._parse_prometheus(minimal, timestamp_s=0.0)
    # queue_time_seconds_sum 不在 _METRIC_PATTERNS 的 "num_" 系列里 → 缺失填 -1.0
    assert sample.get("queue_time_seconds_sum") == -1.0


def test_parse_scientific_notation_in_queue_time():
    text = "vllm:time_in_queue_requests_seconds_sum 1.5e+02\n"
    sample = VllmMetricsCollector._parse_prometheus(text, timestamp_s=0.0)
    assert sample["queue_time_seconds_sum"] == 150.0


# ============================================================
# _aggregate_vllm_samples
# ============================================================

def _vllm_sample(ts, preempts, kv, queue_sum, source="vllm"):
    return {
        "timestamp_s": ts,
        "source": source,
        "num_requests_running": 1,
        "num_requests_waiting": 0,
        "gpu_cache_usage_pct": kv,
        "cpu_cache_usage_pct": 0.0,
        "num_preemptions_total": preempts,
        "queue_time_seconds_sum": queue_sum,
    }


def test_aggregate_basic_counter_diff():
    samples = [
        _vllm_sample(0.0, 0, 0.10, 1.0),
        _vllm_sample(1.0, 2, 0.30, 3.0),
        _vllm_sample(2.0, 5, 0.55, 6.5),
        _vllm_sample(3.0, 8, 0.40, 9.0),
    ]
    out = _aggregate_vllm_samples(samples, wall_time_s=60.0)
    assert out["vllm_metrics_available"] is True
    assert out["preemptions_total"] == 8
    assert out["preemption_rate_per_min"] == pytest.approx(8.0)
    assert 0.0 <= out["kv_cache_usage_p95_pct"] <= 1.0
    assert out["kv_cache_usage_p95_pct"] == 0.55  # 4 个值，P95 取 sorted[3]
    assert out["queue_time_delta_s"] == 8.0


def test_aggregate_empty_returns_unavailable():
    out = _aggregate_vllm_samples([], wall_time_s=0.0)
    assert out["vllm_metrics_available"] is False
    assert out["preemptions_total"] == 0


def test_aggregate_only_unavailable_samples():
    samples = [
        {"source": "unavailable", "num_preemptions_total": -1,
         "gpu_cache_usage_pct": -1.0, "queue_time_seconds_sum": -1.0},
    ]
    out = _aggregate_vllm_samples(samples, wall_time_s=10.0)
    assert out["vllm_metrics_available"] is False


def test_aggregate_skips_negative_sentinel_values():
    samples = [
        _vllm_sample(0.0, -1, -1.0, -1.0),  # 全 sentinel，被过滤
        _vllm_sample(1.0, 0, 0.10, 1.0),
        _vllm_sample(2.0, 3, 0.20, 2.0),
    ]
    out = _aggregate_vllm_samples(samples, wall_time_s=120.0)
    assert out["vllm_metrics_available"] is True
    assert out["preemptions_total"] == 3       # max(0,3)-min(0,3)=3
    assert out["queue_time_delta_s"] == 1.0


def test_aggregate_zero_walltime_keeps_zero_rate():
    samples = [_vllm_sample(0.0, 0, 0.1, 1.0), _vllm_sample(1.0, 5, 0.2, 2.0)]
    out = _aggregate_vllm_samples(samples, wall_time_s=0.0)
    assert out["preemption_rate_per_min"] == 0.0


# ============================================================
# compute_stats 集成
# ============================================================

def _success_request(latency_ms=100.0, ttft_ms=20.0, tpot_ms=10.0, tokens=50):
    return {
        "success": True,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tpot_p95_ms": tpot_ms,
        "output_tokens": tokens,
    }


def test_compute_stats_includes_vllm_aggregates_field():
    results = [_success_request() for _ in range(5)]
    samples = [_vllm_sample(i, i, 0.1 + 0.05 * i, float(i))
               for i in range(4)]
    stats = compute_stats(results, wall_time_s=10.0, vllm_samples=samples)
    assert "vllm_aggregates" in stats
    va = stats["vllm_aggregates"]
    assert va["vllm_metrics_available"] is True
    assert va["preemptions_total"] == 3
    assert va["queue_time_delta_s"] == 3.0


def test_compute_stats_no_vllm_samples_back_compat():
    results = [_success_request() for _ in range(3)]
    stats = compute_stats(results, wall_time_s=5.0)  # 不传 vllm_samples
    assert stats["successful"] == 3
    assert stats["vllm_aggregates"]["vllm_metrics_available"] is False
