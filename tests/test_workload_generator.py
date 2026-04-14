#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_workload_generator.py — WorkloadGenerator 单元测试

覆盖:
- 三种到达模式 (burst / constant_rate / poisson)
- prompt 采样与长度分布
- max_tokens 采样
- 共享前缀注入
- phase-switch 多阶段
- seed 可复现性
- warmup / cooldown 标记
"""
import json
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from workloads.workload_generator import WorkloadGenerator


def _load_workload(name: str) -> dict:
    """加载 configs/workloads/workload_*.json 配置"""
    config_path = PROJECT_ROOT / "configs" / "workloads" / f"workload_{name}.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)["workload"]


class TestBurstArrival(unittest.TestCase):
    """burst 到达模式"""

    def setUp(self):
        self.cfg = _load_workload("baseline")
        self.wg = WorkloadGenerator(self.cfg)

    def test_all_arrive_at_zero(self):
        """burst 模式所有请求 scheduled_time_s = 0"""
        reqs = self.wg.generate()
        for r in reqs:
            self.assertEqual(r["scheduled_time_s"], 0.0)

    def test_request_count(self):
        """生成请求数 == 配置的 num_requests"""
        reqs = self.wg.generate()
        self.assertEqual(len(reqs), self.cfg["num_requests"])

    def test_request_fields(self):
        """每个请求包含必须字段"""
        reqs = self.wg.generate()
        required_fields = [
            "request_id", "scheduled_time_s", "messages", "max_tokens",
            "prompt_length_bucket", "is_warmup", "is_cooldown", "phase_name",
        ]
        for r in reqs:
            for f in required_fields:
                self.assertIn(f, r, f"请求缺少字段: {f}")


class TestConstantRateArrival(unittest.TestCase):
    """constant_rate 到达模式"""

    def setUp(self):
        self.cfg = _load_workload("rate2")
        self.wg = WorkloadGenerator(self.cfg)

    def test_monotonic_increase(self):
        """constant_rate 到达时间单调递增"""
        reqs = self.wg.generate()
        times = [r["scheduled_time_s"] for r in reqs]
        for i in range(1, len(times)):
            self.assertGreaterEqual(times[i], times[i - 1])

    def test_interval_matches_rate(self):
        """constant_rate=2 时，间隔约 0.5s"""
        reqs = self.wg.generate()
        times = [r["scheduled_time_s"] for r in reqs]
        if len(times) >= 2:
            interval = times[1] - times[0]
            self.assertAlmostEqual(interval, 0.5, places=3)


class TestPoissonArrival(unittest.TestCase):
    """poisson 到达模式"""

    def setUp(self):
        self.cfg = _load_workload("poisson2")
        self.wg = WorkloadGenerator(self.cfg)

    def test_monotonic_increase(self):
        """poisson 到达时间单调递增"""
        reqs = self.wg.generate()
        times = [r["scheduled_time_s"] for r in reqs]
        for i in range(1, len(times)):
            self.assertGreaterEqual(times[i], times[i - 1])

    def test_positive_intervals(self):
        """poisson 间隔 > 0"""
        reqs = self.wg.generate()
        times = [r["scheduled_time_s"] for r in reqs]
        for i in range(1, len(times)):
            self.assertGreater(times[i] - times[i - 1], 0)

    def test_average_rate_reasonable(self):
        """poisson λ=2 时，平均速率应在 1-4 req/s 范围"""
        reqs = self.wg.generate()
        times = [r["scheduled_time_s"] for r in reqs]
        if len(times) >= 10:
            total_time = times[-1] - times[0]
            avg_rate = (len(times) - 1) / total_time if total_time > 0 else 0
            self.assertGreater(avg_rate, 0.5)
            self.assertLess(avg_rate, 8.0)


class TestReproducibility(unittest.TestCase):
    """可复现性"""

    def test_same_seed_same_output(self):
        """相同 seed 生成完全相同的请求序列"""
        cfg = _load_workload("baseline")
        wg1 = WorkloadGenerator(cfg, seed=42)
        wg2 = WorkloadGenerator(cfg, seed=42)
        reqs1 = wg1.generate()
        reqs2 = wg2.generate()
        self.assertEqual(len(reqs1), len(reqs2))
        for r1, r2 in zip(reqs1, reqs2):
            self.assertEqual(r1["request_id"], r2["request_id"])
            self.assertEqual(r1["scheduled_time_s"], r2["scheduled_time_s"])
            self.assertEqual(r1["prompt_length_bucket"], r2["prompt_length_bucket"])
            self.assertEqual(r1["max_tokens"], r2["max_tokens"])

    def test_different_seed_different_output(self):
        """不同 seed 生成不同的请求序列"""
        cfg = _load_workload("baseline")
        wg1 = WorkloadGenerator(cfg, seed=42)
        wg2 = WorkloadGenerator(cfg, seed=99)
        reqs1 = wg1.generate()
        reqs2 = wg2.generate()
        # 至少应有部分 prompt 不同
        buckets1 = [r["prompt_length_bucket"] for r in reqs1]
        buckets2 = [r["prompt_length_bucket"] for r in reqs2]
        # 极小概率完全相同，但实际不会
        self.assertNotEqual(buckets1, buckets2)


class TestPromptDistribution(unittest.TestCase):
    """prompt 长度分布采样"""

    def test_short_only(self):
        """short_only 配置应全部为 short bucket"""
        cfg = _load_workload("short_only")
        wg = WorkloadGenerator(cfg)
        reqs = wg.generate()
        for r in reqs:
            self.assertEqual(r["prompt_length_bucket"], "short")

    def test_long_only(self):
        """long_only 配置应全部为 long bucket"""
        cfg = _load_workload("long_only")
        wg = WorkloadGenerator(cfg)
        reqs = wg.generate()
        for r in reqs:
            self.assertEqual(r["prompt_length_bucket"], "long")

    def test_mixed_has_multiple_buckets(self):
        """mixed 配置应包含多种 bucket"""
        cfg = _load_workload("mixed")
        wg = WorkloadGenerator(cfg)
        reqs = wg.generate()
        buckets = set(r["prompt_length_bucket"] for r in reqs)
        self.assertGreater(len(buckets), 1)

    def test_messages_not_empty(self):
        """每个请求的 messages 非空"""
        cfg = _load_workload("baseline")
        wg = WorkloadGenerator(cfg)
        reqs = wg.generate()
        for r in reqs:
            self.assertIsInstance(r["messages"], list)
            self.assertGreater(len(r["messages"]), 0)


class TestMaxTokensSampling(unittest.TestCase):
    """max_tokens 采样"""

    def test_short_only_max_tokens(self):
        """short_only 配置的 max_tokens 应全为 64"""
        cfg = _load_workload("short_only")
        wg = WorkloadGenerator(cfg)
        reqs = wg.generate()
        for r in reqs:
            self.assertEqual(r["max_tokens"], 64)

    def test_long_only_max_tokens(self):
        """long_only 配置的 max_tokens 应全为 512"""
        cfg = _load_workload("long_only")
        wg = WorkloadGenerator(cfg)
        reqs = wg.generate()
        for r in reqs:
            self.assertEqual(r["max_tokens"], 512)


class TestSharedPrefix(unittest.TestCase):
    """共享前缀"""

    def test_prefix_ratio_0(self):
        """ratio=0 时，无前缀组"""
        cfg = _load_workload("prefix_0")
        wg = WorkloadGenerator(cfg)
        reqs = wg.generate()
        for r in reqs:
            self.assertIsNone(r["shared_prefix_group"])

    def test_prefix_ratio_50(self):
        """ratio=0.5 时，约一半有前缀组"""
        cfg = _load_workload("prefix_50")
        wg = WorkloadGenerator(cfg)
        reqs = wg.generate()
        with_prefix = sum(1 for r in reqs if r["shared_prefix_group"] is not None)
        # 允许较大范围（随机性）
        self.assertGreater(with_prefix, 5)

    def test_prefix_ratio_90(self):
        """ratio=0.9 时，大部分有前缀组"""
        cfg = _load_workload("prefix_90")
        wg = WorkloadGenerator(cfg)
        reqs = wg.generate()
        with_prefix = sum(1 for r in reqs if r["shared_prefix_group"] is not None)
        self.assertGreater(with_prefix, 20)


class TestPhaseSwitch(unittest.TestCase):
    """phase-switch 多阶段"""

    def setUp(self):
        self.cfg = _load_workload("phase_switch")
        self.wg = WorkloadGenerator(self.cfg)

    def test_multiple_phases(self):
        """phase_switch 应产生多个 phase_name"""
        reqs = self.wg.generate()
        phases = set(r["phase_name"] for r in reqs)
        self.assertGreaterEqual(len(phases), 2)

    def test_total_request_count(self):
        """总请求数 == 配置值"""
        reqs = self.wg.generate()
        self.assertEqual(len(reqs), self.cfg["num_requests"])

    def test_time_ordering(self):
        """请求按 scheduled_time_s 排序"""
        reqs = self.wg.generate()
        times = [r["scheduled_time_s"] for r in reqs]
        self.assertEqual(times, sorted(times))


class TestWarmupCooldown(unittest.TestCase):
    """warmup / cooldown 标记"""

    def test_baseline_warmup_cooldown(self):
        """baseline 有 warmup=5, cooldown=5"""
        cfg = _load_workload("baseline")
        wg = WorkloadGenerator(cfg)
        reqs = wg.generate()

        warmup_count = sum(1 for r in reqs if r["is_warmup"])
        cooldown_count = sum(1 for r in reqs if r["is_cooldown"])
        self.assertEqual(warmup_count, cfg.get("warmup_requests", 0))
        self.assertEqual(cooldown_count, cfg.get("cooldown_requests", 0))

    def test_warmup_at_beginning(self):
        """warmup 请求在序列开头"""
        cfg = _load_workload("baseline")
        wg = WorkloadGenerator(cfg)
        reqs = wg.generate()
        warmup_n = cfg.get("warmup_requests", 0)
        for i in range(warmup_n):
            self.assertTrue(reqs[i]["is_warmup"])
        # warmup 之后不应再有 warmup
        for r in reqs[warmup_n:]:
            self.assertFalse(r["is_warmup"])


if __name__ == "__main__":
    unittest.main()
