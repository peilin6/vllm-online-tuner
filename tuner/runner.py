#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runner.py — 单 trial 闭环执行器

Task 6.3。一次 run_trial(...) 调用 = 一个 trial：
1. render_experiment_config + write_temp_config
2. launcher.restart(config_path, enforce_eager)
3. subprocess 调 benchmarks/run_benchmark.py 跑压测（复用 Week 4 已验证路径）
4. 期间另起轮询线程，每 5s 抓一次 /metrics 检查早停（preempt/吞吐/KV）
5. parse_trial(results/<exp_id>) → TrialMetrics
6. launcher.stop()
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config_generator import render_experiment_config, write_temp_config
from .launcher import VllmLauncher
from .metrics_parser import TrialMetrics, parse_trial

logger = logging.getLogger(__name__)


# ====================================================================
# 早停轮询线程
# ====================================================================
class _EarlyStopMonitor:
    """5s 一次轮询 /metrics，命中规则就置 should_stop=True。Runner 主流程检查标志位，
    自行 SIGTERM 子进程。"""

    def __init__(
        self,
        base_url: str,
        cfg: dict | None,
        baseline_throughput_tok_per_s: float | None,
    ):
        self.url = f"{base_url.rstrip('/')}/metrics"
        cfg = cfg or {}
        self.warmup_s: float = float(cfg.get("warmup_s", 20.0))
        self.preempt_rate_threshold = float(cfg.get("preempt_rate_per_s", 2.0))
        self.kv_threshold = float(cfg.get("kv_usage_pct", 0.98))
        self.kv_consecutive = int(cfg.get("kv_consecutive", 3))
        self.throughput_floor_ratio = float(cfg.get("throughput_floor_ratio", 0.5))
        self.baseline_tput = baseline_throughput_tok_per_s
        self.poll_interval_s = float(cfg.get("poll_interval_s", 5.0))

        self.should_stop = False
        self.stop_reason: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float = 0.0
        self._kv_streak = 0
        self._last_preempt: int | None = None
        self._last_preempt_t: float | None = None

    def start(self):
        self._t0 = time.perf_counter()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while not self._stop_event.is_set():
            self._check_once()
            if self.should_stop:
                return
            self._stop_event.wait(self.poll_interval_s)

    def _check_once(self):
        elapsed = time.perf_counter() - self._t0
        if elapsed < self.warmup_s:
            return
        sample = self._fetch()
        if sample is None:
            return
        # 1) preempt 速率
        cur_p = sample.get("preempt", 0)
        now = time.perf_counter()
        if self._last_preempt is not None and self._last_preempt_t is not None:
            dt = max(now - self._last_preempt_t, 1e-6)
            rate = (cur_p - self._last_preempt) / dt
            if rate > self.preempt_rate_threshold:
                self._trigger(f"preempt_rate={rate:.2f}/s 超阈值")
                return
        self._last_preempt = cur_p
        self._last_preempt_t = now
        # 2) KV 连续高位
        kv = sample.get("kv", -1.0)
        if 0 <= kv and kv > self.kv_threshold:
            self._kv_streak += 1
            if self._kv_streak >= self.kv_consecutive:
                self._trigger(f"kv_usage={kv:.2f} 连续 {self._kv_streak} 次 > {self.kv_threshold}")
                return
        else:
            self._kv_streak = 0

    def trigger_throughput_check(self, observed_tput_tok_per_s: float | None) -> bool:
        """由 Runner 在压测中段调用：如果观察到吞吐低于 baseline*ratio 立即触发。"""
        if (observed_tput_tok_per_s is None or self.baseline_tput is None
                or self.baseline_tput <= 0):
            return False
        if observed_tput_tok_per_s < self.baseline_tput * self.throughput_floor_ratio:
            self._trigger(
                f"throughput={observed_tput_tok_per_s:.1f} < baseline×{self.throughput_floor_ratio} "
                f"({self.baseline_tput * self.throughput_floor_ratio:.1f})"
            )
            return True
        return False

    def _trigger(self, reason: str):
        self.should_stop = True
        self.stop_reason = reason
        logger.warning("早停触发: %s", reason)

    def _fetch(self) -> dict | None:
        try:
            with urllib.request.urlopen(self.url, timeout=2) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError):
            return None
        m_p = re.search(r"^vllm:num_preemptions_total\b.*?\s+([\d.]+)", body, re.MULTILINE)
        m_k = re.search(r"^vllm:gpu_cache_usage_perc\b.*?\s+([\d.]+)", body, re.MULTILINE)
        return {
            "preempt": int(float(m_p.group(1))) if m_p else 0,
            "kv": float(m_k.group(1)) if m_k else -1.0,
        }


# ====================================================================
# run_trial
# ====================================================================
def run_trial(
    config_overrides: dict,
    *,
    base_config_path: str | Path,
    workload_path: str | Path,
    enforce_eager: bool = False,
    bench_timeout_s: int = 600,
    early_stop_cfg: dict | None = None,
    launcher: VllmLauncher | None = None,
    results_dir: str | Path = "results",
    experiment_id: str | None = None,
    baseline_throughput_tok_per_s: float | None = None,
    bench_script: str | Path | None = None,
) -> TrialMetrics:
    """执行单 trial 闭环。"""
    base_cfg = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    new_cfg = render_experiment_config(base_cfg, config_overrides)
    cfg_path = write_temp_config(new_cfg)

    if launcher is None:
        launcher = VllmLauncher()

    t_trial0 = time.perf_counter()
    launch_res = launcher.restart(cfg_path, enforce_eager=enforce_eager)
    if not launch_res.success:
        return TrialMetrics(
            success=False, early_killed=False,
            wall_time_s=round(time.perf_counter() - t_trial0, 3),
            notes=[f"launcher 失败: {launch_res.error}"],
        )

    base_url = f"http://{launcher.host}:{launcher.port}"
    monitor = _EarlyStopMonitor(base_url, early_stop_cfg, baseline_throughput_tok_per_s)
    monitor.start()

    if experiment_id is None:
        experiment_id = f"trial_{int(time.time())}"
    bench = bench_script or (
        Path(__file__).resolve().parent.parent / "benchmarks" / "run_benchmark.py"
    )
    cmd = [
        sys.executable, str(bench),
        "--config", str(cfg_path),
        "--workload", str(workload_path),
        "--host", launcher.host,
        "--output-dir", str(results_dir),
    ]

    early_killed = False
    proc = subprocess.Popen(
        cmd, env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    deadline = time.time() + bench_timeout_s
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if monitor.should_stop:
                early_killed = True
                _terminate(proc)
                break
            if time.time() > deadline:
                early_killed = True
                monitor._trigger("bench_timeout_s 超时")
                _terminate(proc)
                break
            time.sleep(0.5)
    finally:
        monitor.stop()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        launcher.stop()

    wall = round(time.perf_counter() - t_trial0, 3)
    exp_dir = _resolve_exp_dir(results_dir, experiment_id)
    if exp_dir is None or not (exp_dir / "summary.json").exists():
        return TrialMetrics(
            success=False, early_killed=early_killed, wall_time_s=wall,
            notes=[f"未找到 summary.json (early_killed={early_killed})",
                   f"stop_reason={monitor.stop_reason}" if monitor.stop_reason else ""],
        )
    metrics = parse_trial(exp_dir, early_killed=early_killed, wall_time_s=wall)
    if early_killed:
        metrics.success = False
        if monitor.stop_reason:
            metrics.notes.append(f"early_stop: {monitor.stop_reason}")
    return metrics


def _terminate(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _resolve_exp_dir(results_dir: str | Path, experiment_id: str) -> Path | None:
    """run_benchmark.py 用 experiment_id 作为子目录名（带时间戳），所以这里取最新 mtime。"""
    base = Path(results_dir)
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
