#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_metrics_parser_and_runner.py — Task 6.3 单元测试

- TrialMetrics 默认值 + asdict
- parse_trial 在合成 results/<exp_id>/summary.json 上字段映射正确
- parse_trial 处理缺字段 / 缺目录的兜底
- _derive_queue_time_p95_ms 的退化估算
- _EarlyStopMonitor 的 preempt / kv 触发逻辑（用 mock fetch）
- run_trial 在 launcher 失败时短路返回 early_killed=False, success=False
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.metrics_parser import TrialMetrics, parse_trial, _derive_queue_time_p95_ms
from tuner.runner import _EarlyStopMonitor, run_trial


# ============================================================
# TrialMetrics
# ============================================================

def test_trialmetrics_defaults_and_to_dict():
    m = TrialMetrics()
    d = m.to_dict()
    assert d["throughput_tok_per_s"] == -1.0
    assert d["preemptions_total"] == 0
    assert d["success"] is False
    assert d["notes"] == []


# ============================================================
# parse_trial
# ============================================================

def _write_summary(tmp_path: Path, summary: dict, exp_name: str = "exp1") -> Path:
    exp = tmp_path / exp_name
    exp.mkdir()
    (exp / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return exp


def test_parse_trial_full_fields(tmp_path):
    summary = {
        "throughput_rps": 1.2,
        "token_throughput_tps": 220.5,
        "ttft_ms": {"p95": 90.0},
        "tpot_ms": {"p95": 14.0},
        "latency_ms": {"p95": 1800.0},
        "vllm_aggregates": {
            "vllm_metrics_available": True,
            "preemptions_total": 3,
            "preemption_rate_per_min": 1.5,
            "kv_cache_usage_p95_pct": 0.71,
            "queue_time_delta_s": 1.0,
        },
        "successful": 50,
        "total_requests": 50,
        "success_rate": 1.0,
        "wall_time_s": 120.0,
    }
    exp = _write_summary(tmp_path, summary)
    m = parse_trial(exp)
    assert m.throughput_tok_per_s == 220.5
    assert m.ttft_p95_ms == 90.0
    assert m.preemptions_total == 3
    assert m.kv_cache_usage_p95_pct == 0.71
    assert m.queue_time_p95_ms == pytest.approx(1.0 / 50 * 1000 * 2.0)
    assert m.success is True
    assert m.early_killed is False
    assert m.wall_time_s == 120.0


def test_parse_trial_missing_fields_safe(tmp_path):
    exp = _write_summary(tmp_path, {"successful": 0, "total_requests": 5})
    m = parse_trial(exp)
    # 全部填默认 -1 / 0
    assert m.throughput_tok_per_s == -1.0
    assert m.ttft_p95_ms == -1.0
    assert m.queue_time_p95_ms == -1.0
    assert m.success is False


def test_parse_trial_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_trial(tmp_path / "nonexistent")


def test_parse_trial_early_killed_overrides_success(tmp_path):
    # 即使 success_rate=1.0，early_killed=True 也要 success=False
    exp = _write_summary(tmp_path, {"success_rate": 1.0, "successful": 50})
    m = parse_trial(exp, early_killed=True, wall_time_s=10.0)
    assert m.early_killed is True
    assert m.success is False
    assert m.wall_time_s == 10.0


def test_derive_queue_time_p95_ms_negative_input():
    s = {"vllm_aggregates": {"queue_time_delta_s": -1.0}}
    assert _derive_queue_time_p95_ms(s, Path(".")) == -1.0


def test_derive_queue_time_p95_ms_no_requests():
    s = {"vllm_aggregates": {"queue_time_delta_s": 5.0},
         "successful": 0, "total_requests": 0}
    assert _derive_queue_time_p95_ms(s, Path(".")) == -1.0


# ============================================================
# _EarlyStopMonitor
# ============================================================

def test_early_stop_preempt_burst():
    m = _EarlyStopMonitor("http://x", {"warmup_s": 0.0, "preempt_rate_per_s": 1.0}, None)
    samples = iter([
        {"preempt": 0, "kv": 0.1},
        {"preempt": 10, "kv": 0.1},  # 10 次 preempt 在 ~0s 内 → 远超 1/s
    ])
    with mock.patch.object(m, "_fetch", side_effect=lambda: next(samples)):
        m._t0 = time.perf_counter() - 100  # 越过 warmup
        m._check_once()
        time.sleep(0.05)
        m._check_once()
    assert m.should_stop is True
    assert "preempt_rate" in (m.stop_reason or "")


def test_early_stop_kv_consecutive():
    m = _EarlyStopMonitor("http://x",
                          {"warmup_s": 0.0, "kv_usage_pct": 0.9, "kv_consecutive": 2},
                          None)
    samples = iter([
        {"preempt": 0, "kv": 0.95},
        {"preempt": 0, "kv": 0.96},  # 第二次连续超阈值
    ])
    with mock.patch.object(m, "_fetch", side_effect=lambda: next(samples)):
        m._t0 = time.perf_counter() - 100
        m._check_once()
        m._check_once()
    assert m.should_stop is True
    assert "kv_usage" in (m.stop_reason or "")


def test_early_stop_kv_resets_streak_when_dips():
    m = _EarlyStopMonitor("http://x",
                          {"warmup_s": 0.0, "kv_usage_pct": 0.9, "kv_consecutive": 2}, None)
    samples = iter([
        {"preempt": 0, "kv": 0.95},   # streak 1
        {"preempt": 0, "kv": 0.50},   # streak 重置
        {"preempt": 0, "kv": 0.95},   # streak 1
    ])
    with mock.patch.object(m, "_fetch", side_effect=lambda: next(samples)):
        m._t0 = time.perf_counter() - 100
        for _ in range(3):
            m._check_once()
    assert m.should_stop is False


def test_early_stop_within_warmup_no_trigger():
    m = _EarlyStopMonitor("http://x", {"warmup_s": 60.0, "preempt_rate_per_s": 0.1}, None)
    with mock.patch.object(m, "_fetch", return_value={"preempt": 99999, "kv": 0.99}):
        m._t0 = time.perf_counter()  # 刚开始
        m._check_once()
    assert m.should_stop is False


def test_early_stop_throughput_floor_triggers():
    m = _EarlyStopMonitor("http://x", {"warmup_s": 0.0, "throughput_floor_ratio": 0.5}, 200.0)
    triggered = m.trigger_throughput_check(80.0)  # 远低于 100
    assert triggered is True
    assert m.should_stop is True


def test_early_stop_throughput_no_baseline_skipped():
    m = _EarlyStopMonitor("http://x", {"warmup_s": 0.0}, None)
    assert m.trigger_throughput_check(10.0) is False
    assert m.should_stop is False


def test_early_stop_fetch_unreachable_returns_none():
    m = _EarlyStopMonitor("http://127.0.0.1:1", None, None)  # 端口 1 一定连不上
    assert m._fetch() is None


# ============================================================
# run_trial: launcher 失败短路
# ============================================================

class _FakeLauncher:
    """用于 run_trial 单元测试的 fake launcher。"""
    def __init__(self, success: bool, host="127.0.0.1", port=8000):
        self._success = success
        self.host = host
        self.port = port
        self.stop_called = False

    def restart(self, cfg, enforce_eager=False):
        from tuner.launcher import LaunchResult
        return LaunchResult(
            success=self._success,
            pid=123 if self._success else None,
            restart_wall_time_s=0.1,
            error=None if self._success else "fake-failure",
        )

    def stop(self, grace_s=10):
        self.stop_called = True
        return True


def test_run_trial_launcher_failure_short_circuit(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"server": {"max_num_seqs": 32}}))
    workload = tmp_path / "wl.json"
    workload.write_text("{}")

    fake = _FakeLauncher(success=False)
    m = run_trial(
        {"max_num_seqs": 64},
        base_config_path=base,
        workload_path=workload,
        launcher=fake,
        results_dir=tmp_path / "results",
    )
    assert m.success is False
    assert m.early_killed is False
    assert any("launcher 失败" in n for n in m.notes)
