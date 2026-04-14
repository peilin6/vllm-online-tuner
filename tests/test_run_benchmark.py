#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_run_benchmark.py — run_benchmark 核心函数单元测试

覆盖:
- compute_stats() 统计计算
- format_summary() 输出格式
- _build_request_trace() trace 构建
- _build_metrics_timeseries() 时序合并
- TPOT 统计聚合

注意: send_request / run_benchmark / run_benchmark_workload 涉及 HTTP
调用，仅在集成测试中验证（需要运行 vLLM 服务）。
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_benchmark import (
    compute_stats,
    format_summary,
    _build_request_trace,
    _build_metrics_timeseries,
)


def _make_result(success=True, latency=1000.0, ttft=50.0, tpot=12.0,
                 tpot_p95=18.0, output_tokens=200, **kwargs):
    """构造一个模拟请求结果"""
    r = {
        "success": success,
        "ttft_ms": ttft if success else None,
        "latency_ms": latency if success else None,
        "output_tokens": output_tokens if success else 0,
        "output_tokens_source": "usage",
        "tpot_ms": tpot if success else None,
        "tpot_p95_ms": tpot_p95 if success else None,
        "token_timestamps_ms": [50 + i * 12 for i in range(output_tokens)] if success else [],
        "error": None if success else "timeout",
        "request_id": kwargs.get("request_id", "req_0000"),
        "prompt_length_bucket": kwargs.get("prompt_length_bucket", "medium"),
        "target_max_tokens": kwargs.get("target_max_tokens", 256),
        "shared_prefix_group": kwargs.get("shared_prefix_group"),
        "phase_name": kwargs.get("phase_name", "default"),
        "is_warmup": kwargs.get("is_warmup", False),
        "is_cooldown": kwargs.get("is_cooldown", False),
        "scheduled_time_s": kwargs.get("scheduled_time_s"),
    }
    return r


class TestComputeStats(unittest.TestCase):
    """compute_stats 统计计算"""

    def test_basic_stats(self):
        """基本统计指标"""
        results = [_make_result(latency=1000 + i * 100, output_tokens=200)
                   for i in range(10)]
        stats = compute_stats(results, wall_time_s=10.0)

        self.assertEqual(stats["total_requests"], 10)
        self.assertEqual(stats["successful"], 10)
        self.assertEqual(stats["failed"], 0)
        self.assertAlmostEqual(stats["success_rate"], 1.0)

    def test_throughput_calculation(self):
        """吞吐量 = 成功数 / 墙钟时间"""
        results = [_make_result() for _ in range(10)]
        stats = compute_stats(results, wall_time_s=5.0)
        self.assertAlmostEqual(stats["throughput_rps"], 2.0)

    def test_token_throughput(self):
        """token 吞吐 = 总 tokens / 墙钟时间"""
        results = [_make_result(output_tokens=100) for _ in range(10)]
        stats = compute_stats(results, wall_time_s=10.0)
        self.assertAlmostEqual(stats["token_throughput_tps"], 100.0)

    def test_latency_percentiles(self):
        """延迟百分位数"""
        results = [_make_result(latency=i * 100.0) for i in range(1, 101)]
        stats = compute_stats(results, wall_time_s=100.0)

        self.assertIn("latency_ms", stats)
        self.assertEqual(stats["latency_ms"]["min"], 100.0)
        self.assertEqual(stats["latency_ms"]["max"], 10000.0)

    def test_tpot_stats(self):
        """TPOT 聚合"""
        results = [_make_result(tpot=10 + i * 0.5) for i in range(20)]
        stats = compute_stats(results, wall_time_s=20.0)

        self.assertIn("tpot_ms", stats)
        self.assertIn("mean", stats["tpot_ms"])
        self.assertIn("p95", stats["tpot_ms"])
        self.assertIn("p99", stats["tpot_ms"])

    def test_with_failures(self):
        """含失败请求"""
        results = [_make_result() for _ in range(8)]
        results.append(_make_result(success=False))
        results.append(_make_result(success=False))
        stats = compute_stats(results, wall_time_s=10.0)

        self.assertEqual(stats["successful"], 8)
        self.assertEqual(stats["failed"], 2)
        self.assertAlmostEqual(stats["success_rate"], 0.8)
        self.assertIn("errors", stats)

    def test_empty_results(self):
        """空结果不崩溃"""
        stats = compute_stats([], wall_time_s=1.0)
        self.assertEqual(stats["total_requests"], 0)
        self.assertNotIn("latency_ms", stats)
        self.assertNotIn("tpot_ms", stats)

    def test_all_failures(self):
        """全部失败"""
        results = [_make_result(success=False) for _ in range(5)]
        stats = compute_stats(results, wall_time_s=1.0)
        self.assertEqual(stats["successful"], 0)
        self.assertEqual(stats["failed"], 5)
        self.assertNotIn("latency_ms", stats)


class TestFormatSummary(unittest.TestCase):
    """format_summary 输出格式"""

    def test_contains_key_sections(self):
        """摘要包含关键段落"""
        results = [_make_result() for _ in range(10)]
        stats = compute_stats(results, wall_time_s=10.0)
        config_info = {
            "timestamp": "20260413_120000",
            "model": "test-model",
            "concurrency": 4,
        }
        summary = format_summary(stats, config_info)

        self.assertIn("压测结果摘要", summary)
        self.assertIn("test-model", summary)
        self.assertIn("成功率", summary)
        self.assertIn("首Token时延 (TTFT)", summary)
        self.assertIn("输出Token间隔 (TPOT)", summary)
        self.assertIn("端到端时延", summary)

    def test_no_crash_on_minimal_stats(self):
        """最小 stats 不崩溃"""
        stats = {"total_requests": 0, "successful": 0, "failed": 0,
                 "success_rate": 0}
        config_info = {}
        summary = format_summary(stats, config_info)
        self.assertIsInstance(summary, str)


class TestBuildRequestTrace(unittest.TestCase):
    """_build_request_trace"""

    def test_trace_fields(self):
        """trace 包含必须字段"""
        results = [_make_result(request_id="req_0001")]
        traces = _build_request_trace(results, save_timestamps=False)
        self.assertEqual(len(traces), 1)
        t = traces[0]
        required = [
            "request_id", "success", "ttft_ms", "tpot_ms",
            "latency_ms", "output_tokens", "phase_name",
            "is_warmup", "is_cooldown", "error",
        ]
        for f in required:
            self.assertIn(f, t)

    def test_no_timestamps_by_default(self):
        """save_timestamps=False 时不含 token_timestamps_ms"""
        results = [_make_result()]
        traces = _build_request_trace(results, save_timestamps=False)
        self.assertNotIn("token_timestamps_ms", traces[0])

    def test_with_timestamps(self):
        """save_timestamps=True 时包含 token_timestamps_ms"""
        results = [_make_result()]
        traces = _build_request_trace(results, save_timestamps=True)
        self.assertIn("token_timestamps_ms", traces[0])


class TestBuildMetricsTimeseries(unittest.TestCase):
    """_build_metrics_timeseries"""

    def test_merge_and_sort(self):
        """合并 GPU 和 vLLM 时序数据并按时间排序"""
        gpu = [
            {"timestamp_s": 0.5, "gpu_util_pct": 80.0},
            {"timestamp_s": 1.5, "gpu_util_pct": 85.0},
        ]
        vllm = [
            {"timestamp_s": 1.0, "source": "vllm", "num_requests_running": 2},
            {"timestamp_s": 2.0, "source": "vllm", "num_requests_running": 0},
        ]
        ts = _build_metrics_timeseries(gpu, vllm)

        self.assertEqual(len(ts), 4)
        # 时间排序
        timestamps = [t["timestamp_s"] for t in ts]
        self.assertEqual(timestamps, sorted(timestamps))
        # source 标记
        self.assertEqual(ts[0]["source"], "gpu")
        self.assertEqual(ts[1]["source"], "vllm")

    def test_empty_inputs(self):
        """空输入不崩溃"""
        ts = _build_metrics_timeseries([], [])
        self.assertEqual(len(ts), 0)

    def test_gpu_only(self):
        """只有 GPU 数据"""
        gpu = [{"timestamp_s": 0.5, "gpu_util_pct": 80.0}]
        ts = _build_metrics_timeseries(gpu, [])
        self.assertEqual(len(ts), 1)
        self.assertEqual(ts[0]["source"], "gpu")


if __name__ == "__main__":
    unittest.main()
