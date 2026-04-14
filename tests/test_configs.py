#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_configs.py — 配置文件完整性测试

覆盖:
- 所有 workload 配置文件可解析
- 必须字段存在
- 数据文件（prompts_pool, prefix_pool）格式正确
- few_shot_examples 格式正确
"""
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestWorkloadConfigs(unittest.TestCase):
    """所有 workload_*.json 配置文件"""

    CONFIG_DIR = PROJECT_ROOT / "configs" / "workloads"
    REQUIRED_WORKLOAD_FIELDS = [
        "name", "seed", "num_requests", "arrival", "prompt", "output",
    ]

    def _get_workload_files(self):
        """获取所有 workload 配置文件"""
        return sorted(self.CONFIG_DIR.glob("workload_*.json"))

    def test_at_least_5_configs(self):
        """至少 5 个 workload 配置文件"""
        files = self._get_workload_files()
        # 排除 schema
        configs = [f for f in files if "schema" not in f.name]
        self.assertGreaterEqual(len(configs), 5)

    def test_all_parseable(self):
        """所有配置文件可解析"""
        for f in self._get_workload_files():
            if "schema" in f.name:
                continue
            with open(f, "r", encoding="utf-8") as fp:
                try:
                    data = json.load(fp)
                except json.JSONDecodeError as e:
                    self.fail(f"{f.name} JSON 解析失败: {e}")
                self.assertIn("workload", data, f"{f.name} 缺少 'workload' 顶层键")

    def test_required_fields(self):
        """所有配置包含必须字段"""
        for f in self._get_workload_files():
            if "schema" in f.name:
                continue
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            wl = data["workload"]
            for field in self.REQUIRED_WORKLOAD_FIELDS:
                self.assertIn(field, wl,
                              f"{f.name} 缺少字段: {field}")

    def test_seed_is_42(self):
        """所有实验配置 seed=42"""
        for f in self._get_workload_files():
            if "schema" in f.name:
                continue
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            self.assertEqual(data["workload"]["seed"], 42,
                            f"{f.name} seed 不为 42")

    def test_num_requests_50(self):
        """所有实验配置 num_requests=50（除 phase_switch 可能不同）"""
        for f in self._get_workload_files():
            if "schema" in f.name:
                continue
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            wl = data["workload"]
            # phase_switch 可能有 80 个请求
            if "phase_switch" in f.name:
                self.assertGreaterEqual(wl["num_requests"], 50)
            else:
                self.assertEqual(wl["num_requests"], 50,
                                f"{f.name} num_requests 不为 50")

    def test_arrival_patterns(self):
        """配置集中包含 burst/constant_rate/poisson 三种到达模式"""
        patterns = set()
        for f in self._get_workload_files():
            if "schema" in f.name:
                continue
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            patterns.add(data["workload"]["arrival"]["pattern"])
        self.assertIn("burst", patterns)
        self.assertIn("constant_rate", patterns)
        self.assertIn("poisson", patterns)


class TestPromptsPool(unittest.TestCase):
    """workloads/prompts_pool.json"""

    def setUp(self):
        pool_path = PROJECT_ROOT / "workloads" / "prompts_pool.json"
        with open(pool_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_at_least_30_prompts(self):
        """至少 30 条 prompt"""
        self.assertGreaterEqual(len(self.data["prompts"]), 30)

    def test_three_categories(self):
        """包含 short/medium/long 三类"""
        cats = set(p["category"] for p in self.data["prompts"])
        self.assertIn("short", cats)
        self.assertIn("medium", cats)
        self.assertIn("long", cats)

    def test_prompt_has_messages(self):
        """每条 prompt 有 messages 字段"""
        for p in self.data["prompts"]:
            self.assertIn("messages", p)
            self.assertIsInstance(p["messages"], list)
            self.assertGreater(len(p["messages"]), 0)

    def test_prompt_has_tokens_estimate(self):
        """每条 prompt 有 estimated_tokens"""
        for p in self.data["prompts"]:
            self.assertIn("estimated_tokens", p)
            self.assertGreater(p["estimated_tokens"], 0)


class TestPrefixPool(unittest.TestCase):
    """workloads/prefix_pool.json"""

    def setUp(self):
        pool_path = PROJECT_ROOT / "workloads" / "prefix_pool.json"
        with open(pool_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_at_least_5_groups(self):
        """至少 5 个前缀组"""
        self.assertGreaterEqual(len(self.data["prefixes"]), 5)

    def test_group_has_suffix_pool(self):
        """每组有 suffix_pool 且非空"""
        for g in self.data["prefixes"]:
            self.assertIn("suffix_pool", g)
            self.assertGreater(len(g["suffix_pool"]), 0)

    def test_group_has_id(self):
        """每组有唯一 id"""
        ids = [g["id"] for g in self.data["prefixes"]]
        self.assertEqual(len(ids), len(set(ids)))


class TestFewShotExamples(unittest.TestCase):
    """configs/llm_prompts/few_shot_examples.json"""

    def setUp(self):
        path = PROJECT_ROOT / "configs" / "llm_prompts" / "few_shot_examples.json"
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_at_least_3_examples(self):
        """至少 3 个示例"""
        self.assertGreaterEqual(len(self.data["examples"]), 3)

    def test_example_has_scenario(self):
        """每个示例有 scenario 和 metrics_snapshot"""
        for ex in self.data["examples"]:
            self.assertIn("scenario", ex)
            self.assertIn("metrics_snapshot", ex)
            self.assertIn("analysis", ex)

    def test_metrics_has_key_fields(self):
        """metrics_snapshot 包含关键指标"""
        for ex in self.data["examples"]:
            ms = ex["metrics_snapshot"]
            self.assertIn("throughput_rps", ms)
            self.assertIn("ttft_ms", ms)
            self.assertIn("latency_ms", ms)
            self.assertIn("success_rate", ms)


class TestBaselineConfig(unittest.TestCase):
    """configs/experiments/baseline_0.json"""

    def setUp(self):
        path = PROJECT_ROOT / "configs" / "experiments" / "baseline_0.json"
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_has_required_sections(self):
        """包含 model/server/sampling/benchmark 段"""
        for section in ["model", "server", "sampling", "benchmark"]:
            self.assertIn(section, self.data)

    def test_model_is_awq(self):
        """模型为 AWQ 量化"""
        self.assertIn("AWQ", self.data["model"]["name"])

    def test_port_8000(self):
        """服务端口为 8000"""
        self.assertEqual(self.data["server"]["port"], 8000)


if __name__ == "__main__":
    unittest.main()
