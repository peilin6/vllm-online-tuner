#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_vllm_metrics_collector.py — VllmMetricsCollector 单元测试

覆盖:
- 启动/停止生命周期
- Prometheus 文本解析
- 端点不可用时的降级处理
"""
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from monitors.vllm_metrics_collector import VllmMetricsCollector, _METRIC_PATTERNS


SAMPLE_PROMETHEUS_TEXT = """
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen2.5-3B-Instruct-AWQ"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-3B-Instruct-AWQ"} 5.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.4523
# HELP vllm:cpu_cache_usage_perc CPU KV-cache usage.
# TYPE vllm:cpu_cache_usage_perc gauge
vllm:cpu_cache_usage_perc 0.0
# HELP vllm:num_preemptions_total Cumulative number of preemptions.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total 2.0
"""


class TestPrometheusParser(unittest.TestCase):
    """Prometheus 文本解析"""

    def test_parse_all_metrics(self):
        """解析所有关键指标"""
        result = VllmMetricsCollector._parse_prometheus(
            SAMPLE_PROMETHEUS_TEXT, timestamp_s=1.0
        )
        self.assertEqual(result["source"], "vllm")
        self.assertEqual(result["timestamp_s"], 1.0)
        self.assertEqual(result["num_requests_running"], 3)
        self.assertEqual(result["num_requests_waiting"], 5)
        self.assertAlmostEqual(result["gpu_cache_usage_pct"], 0.4523, places=4)
        self.assertAlmostEqual(result["cpu_cache_usage_pct"], 0.0, places=4)
        self.assertEqual(result["num_preemptions_total"], 2)

    def test_parse_empty_text(self):
        """空文本返回 -1 占位"""
        result = VllmMetricsCollector._parse_prometheus("", timestamp_s=0.5)
        self.assertEqual(result["source"], "vllm")
        self.assertEqual(result["num_requests_running"], -1)
        self.assertEqual(result["gpu_cache_usage_pct"], -1.0)

    def test_parse_partial_metrics(self):
        """只有部分指标时，缺失的用 -1 填充"""
        partial = "vllm:num_requests_running{} 7.0\n"
        result = VllmMetricsCollector._parse_prometheus(partial, timestamp_s=2.0)
        self.assertEqual(result["num_requests_running"], 7)
        self.assertEqual(result["num_requests_waiting"], -1)


class TestCollectorLifecycle(unittest.TestCase):
    """VllmMetricsCollector 生命周期"""

    def test_start_stop_no_crash(self):
        """启动和停止（目标不可达）不崩溃"""
        c = VllmMetricsCollector(
            base_url="http://127.0.0.1:19999",  # 不存在的端口
            interval_ms=200,
        )
        c.start()
        time.sleep(0.5)
        samples = c.stop()
        self.assertIsInstance(samples, list)

    def test_stop_without_start(self):
        """未启动就 stop 不崩溃"""
        c = VllmMetricsCollector()
        samples = c.stop()
        self.assertEqual(len(samples), 0)

    def test_unavailable_endpoint_returns_degraded(self):
        """端点不可用时返回 source=unavailable"""
        c = VllmMetricsCollector(
            base_url="http://127.0.0.1:19999",
            interval_ms=200,
        )
        c.start()
        time.sleep(0.8)
        samples = c.stop()
        if len(samples) > 0:
            self.assertEqual(samples[0]["source"], "unavailable")
            self.assertIn("error", samples[0])

    def test_daemon_thread(self):
        """采集线程是 daemon"""
        c = VllmMetricsCollector(
            base_url="http://127.0.0.1:19999",
            interval_ms=200,
        )
        c.start()
        if c._thread is not None:
            self.assertTrue(c._thread.daemon)
        c.stop()


class TestMetricPatterns(unittest.TestCase):
    """正则模式匹配"""

    def test_pattern_count(self):
        """_METRIC_PATTERNS 有 5 个指标"""
        self.assertEqual(len(_METRIC_PATTERNS), 5)

    def test_patterns_match_sample(self):
        """所有模式都能在示例文本中匹配"""
        for name, pattern in _METRIC_PATTERNS.items():
            match = pattern.search(SAMPLE_PROMETHEUS_TEXT)
            self.assertIsNotNone(match, f"模式 {name} 未匹配到示例文本")


if __name__ == "__main__":
    unittest.main()
