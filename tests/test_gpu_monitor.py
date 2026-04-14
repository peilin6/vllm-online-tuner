#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_gpu_monitor.py — GpuMonitor 单元测试

覆盖:
- 启动/停止生命周期
- daemon 线程属性
- 采样数据结构
- 优雅降级（无 GPU 环境）
"""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from monitors.gpu_monitor import GpuMonitor


class TestGpuMonitorLifecycle(unittest.TestCase):
    """GpuMonitor 生命周期"""

    def test_start_stop_no_crash(self):
        """启动和停止不崩溃"""
        m = GpuMonitor(interval_ms=200)
        m.start()
        time.sleep(0.5)
        samples = m.stop()
        self.assertIsInstance(samples, list)

    def test_daemon_thread(self):
        """采样线程是 daemon 线程"""
        m = GpuMonitor(interval_ms=200)
        m.start()
        if m._thread is not None:
            self.assertTrue(m._thread.daemon)
        m.stop()

    def test_stop_without_start(self):
        """未启动就 stop 不崩溃"""
        m = GpuMonitor(interval_ms=200)
        samples = m.stop()
        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), 0)

    def test_double_stop(self):
        """重复 stop 不崩溃"""
        m = GpuMonitor(interval_ms=200)
        m.start()
        time.sleep(0.3)
        m.stop()
        samples = m.stop()
        self.assertIsInstance(samples, list)


class TestGpuMonitorSampleFormat(unittest.TestCase):
    """采样数据格式（需要 GPU 或 mock）"""

    def test_sample_keys(self):
        """如果有采样数据，检查字段完整"""
        m = GpuMonitor(interval_ms=200)
        m.start()
        time.sleep(1)
        samples = m.stop()

        if len(samples) > 0:
            s = samples[0]
            expected_keys = [
                "timestamp_s", "gpu_util_pct", "mem_used_mib",
                "mem_total_mib", "mem_util_pct", "temperature_c", "power_w",
            ]
            for k in expected_keys:
                self.assertIn(k, s, f"采样缺少字段: {k}")

    def test_timestamp_monotonic(self):
        """时间戳单调递增"""
        m = GpuMonitor(interval_ms=200)
        m.start()
        time.sleep(1)
        samples = m.stop()

        if len(samples) >= 2:
            for i in range(1, len(samples)):
                self.assertGreater(
                    samples[i]["timestamp_s"],
                    samples[i - 1]["timestamp_s"],
                )

    def test_sampling_interval(self):
        """3 秒内每 500ms 采样一次，应有 ≥5 个样本（如果 GPU 可用）"""
        m = GpuMonitor(interval_ms=500)
        m.start()
        time.sleep(3)
        samples = m.stop()
        # 如果 pynvml 可用，至少应有 5 个样本
        if m._available:
            self.assertGreaterEqual(len(samples), 5)


class TestGpuMonitorGracefulDegradation(unittest.TestCase):
    """优雅降级"""

    @patch("monitors.gpu_monitor._ensure_pynvml", return_value=False)
    def test_no_gpu_returns_empty(self, mock_pynvml):
        """无 GPU 时，返回空列表"""
        m = GpuMonitor(interval_ms=200)
        m.start()
        time.sleep(0.5)
        samples = m.stop()
        self.assertEqual(len(samples), 0)


if __name__ == "__main__":
    unittest.main()
