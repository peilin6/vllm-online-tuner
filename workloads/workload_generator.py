#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workload_generator.py — 根据 workload 配置生成请求序列及其调度时间

支持三种到达模式: burst / constant_rate / poisson
支持 phase-switch: 实验中途按时间表切换负载场景
支持共享前缀: 按比例将部分请求替换为前缀模板+随机后缀
"""
import copy
import json
import logging
import math
import random
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkloadGenerator:
    """根据 workload 配置生成请求序列及其调度时间"""

    def __init__(self, config: dict, seed: int = None):
        """
        config: workload 配置字典（从 JSON 加载后的 "workload" 部分）
        seed: 显式指定种子，覆盖配置中的 seed
        """
        self._config = copy.deepcopy(config)
        self._seed = seed if seed is not None else config.get("seed", 42)
        self._rng = random.Random(self._seed)

        # 加载 prompt 语料池
        pool_file = config["prompt"]["pool_file"]
        self._prompt_pool = self._load_json(pool_file)
        self._prompts_by_category = self._index_prompts()

        # 加载共享前缀池（如启用）
        self._prefix_pool = None
        sp = config.get("shared_prefix", {})
        if sp.get("enabled", False) and sp.get("prefix_pool_file"):
            self._prefix_pool = self._load_json(sp["prefix_pool_file"])

    @staticmethod
    def _load_json(path: str) -> dict:
        """加载 JSON 文件"""
        p = Path(path)
        if not p.is_absolute():
            # 尝试从工作目录加载
            candidates = [p, Path("/mnt/d/vlllm") / p]
            for c in candidates:
                if c.exists():
                    with open(c, "r", encoding="utf-8") as f:
                        return json.load(f)
            raise FileNotFoundError(f"无法找到文件: {path}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _index_prompts(self) -> dict[str, list[dict]]:
        """按 category 索引 prompt"""
        index = {}
        for p in self._prompt_pool["prompts"]:
            cat = p["category"]
            index.setdefault(cat, []).append(p)
        return index

    def generate(self) -> list[dict]:
        """
        生成完整的请求序列。
        每个请求包含:
        - request_id: str
        - scheduled_time_s: float  (相对于实验开始的调度时间)
        - messages: list[dict]
        - max_tokens: int
        - prompt_length_bucket: str  ("short"/"medium"/"long")
        - actual_prompt_tokens: int  (估算的 prompt token 数)
        - target_max_tokens: int
        - shared_prefix_group: str | None
        - is_warmup: bool
        - is_cooldown: bool
        - phase_name: str
        - metadata: dict
        """
        cfg = self._config
        num_requests = cfg["num_requests"]
        warmup = cfg.get("warmup_requests", 0)
        cooldown = cfg.get("cooldown_requests", 0)

        phase_switch = cfg.get("phase_switch", {})
        use_phase_switch = phase_switch.get("enabled", False) and phase_switch.get("phases")

        if use_phase_switch:
            return self._generate_phase_switch(num_requests, warmup, cooldown, phase_switch)
        else:
            return self._generate_single_phase(num_requests, warmup, cooldown, cfg)

    def _generate_single_phase(self, num_requests: int, warmup: int,
                               cooldown: int, phase_cfg: dict,
                               phase_name: str = "default",
                               time_offset: float = 0.0) -> list[dict]:
        """生成单一阶段的请求序列"""
        arrival_cfg = phase_cfg.get("arrival", self._config["arrival"])
        prompt_cfg = phase_cfg.get("prompt", self._config["prompt"])
        output_cfg = phase_cfg.get("output", self._config["output"])
        sp_cfg = phase_cfg.get("shared_prefix", self._config.get("shared_prefix", {}))

        # 生成到达时间
        arrival_times = self._sample_arrival_times(num_requests, arrival_cfg)

        requests = []
        for i in range(num_requests):
            # 采样 prompt
            messages, bucket, est_tokens = self._sample_prompt(prompt_cfg)

            # 采样 max_tokens
            max_tokens = self._sample_max_tokens(output_cfg)

            # 应用共享前缀
            prefix_group = None
            if sp_cfg.get("enabled", False) and self._prefix_pool:
                messages, prefix_group = self._apply_shared_prefix(
                    messages, sp_cfg.get("ratio", 0.0))

            # 标记 warmup / cooldown
            is_warmup = i < warmup
            is_cooldown = i >= (num_requests - cooldown)

            req = {
                "request_id": f"req_{i:04d}",
                "scheduled_time_s": arrival_times[i] + time_offset,
                "messages": messages,
                "max_tokens": max_tokens,
                "prompt_length_bucket": bucket,
                "actual_prompt_tokens": est_tokens,
                "target_max_tokens": max_tokens,
                "shared_prefix_group": prefix_group,
                "is_warmup": is_warmup,
                "is_cooldown": is_cooldown,
                "phase_name": phase_name,
                "metadata": {
                    "seed": self._seed,
                    "index_in_phase": i,
                    "arrival_pattern": arrival_cfg.get("pattern", "burst"),
                },
            }
            requests.append(req)

        return requests

    def _generate_phase_switch(self, total_requests: int, warmup: int,
                               cooldown: int, phase_switch_cfg: dict) -> list[dict]:
        """生成带 phase-switch 的请求序列"""
        phases = sorted(phase_switch_cfg["phases"], key=lambda p: p["start_time_s"])
        all_requests = []
        remaining = total_requests
        req_counter = 0

        for idx, phase in enumerate(phases):
            phase_name = phase["name"]
            start_time = phase["start_time_s"]

            # 计算本 phase 的持续时间
            if idx + 1 < len(phases):
                duration = phases[idx + 1]["start_time_s"] - start_time
            else:
                duration = None  # 最后一个 phase 消耗剩余所有请求

            # 构建本 phase 的配置（继承全局默认 + phase 覆盖）
            phase_cfg = copy.deepcopy(self._config)
            for key in ("arrival", "prompt", "output", "shared_prefix"):
                if key in phase:
                    if isinstance(phase[key], dict):
                        if key in phase_cfg:
                            phase_cfg[key].update(phase[key])
                        else:
                            phase_cfg[key] = phase[key]

            # 估算本 phase 的请求数
            arrival_cfg = phase_cfg.get("arrival", self._config["arrival"])
            pattern = arrival_cfg.get("pattern", "burst")
            rate = arrival_cfg.get("rate")

            if duration is not None and pattern != "burst" and rate:
                # 按速率 × 持续时间估算请求数
                phase_count = max(1, int(duration * rate))
            elif duration is not None and pattern == "burst":
                # burst 模式在 phase 时间内一次性发完，按总请求比例分配
                total_duration = phases[-1]["start_time_s"] + 30  # 最后 phase 默认 30s
                phase_count = max(1, int(remaining * duration / max(1, total_duration - start_time)))
            else:
                # 最后一个 phase，消耗剩余
                phase_count = remaining

            phase_count = min(phase_count, remaining)
            if phase_count <= 0:
                continue

            # 生成本 phase 的请求
            phase_requests = self._generate_single_phase(
                num_requests=phase_count,
                warmup=warmup if idx == 0 else 0,
                cooldown=cooldown if idx == len(phases) - 1 else 0,
                phase_cfg=phase_cfg,
                phase_name=phase_name,
                time_offset=start_time,
            )

            # 重新编号 request_id
            for req in phase_requests:
                req["request_id"] = f"req_{req_counter:04d}"
                req["metadata"]["global_index"] = req_counter
                req_counter += 1

            all_requests.extend(phase_requests)
            remaining -= phase_count

        # 按调度时间排序
        all_requests.sort(key=lambda r: r["scheduled_time_s"])
        return all_requests

    def _sample_arrival_times(self, n: int, arrival_cfg: dict = None) -> list[float]:
        """根据 arrival pattern 生成 n 个到达时间"""
        if arrival_cfg is None:
            arrival_cfg = self._config["arrival"]
        pattern = arrival_cfg.get("pattern", "burst")
        rate = arrival_cfg.get("rate")

        if pattern == "burst":
            # 所有请求同时到达
            return [0.0] * n

        elif pattern == "constant_rate":
            if not rate or rate <= 0:
                warnings.warn("constant_rate 模式未指定 rate，回退为 burst")
                return [0.0] * n
            return [i / rate for i in range(n)]

        elif pattern == "poisson":
            if not rate or rate <= 0:
                warnings.warn("poisson 模式未指定 rate，回退为 burst")
                return [0.0] * n
            times = []
            t = 0.0
            for _ in range(n):
                interval = self._rng.expovariate(rate)
                t += interval
                times.append(t)
            return times

        else:
            warnings.warn(f"未知到达模式 '{pattern}'，回退为 burst")
            return [0.0] * n

    def _sample_prompt(self, prompt_cfg: dict = None) -> tuple:
        """
        按长度分布从 pool 采样一条 prompt。
        返回 (messages, bucket, est_tokens)
        """
        if prompt_cfg is None:
            prompt_cfg = self._config["prompt"]

        length_dist = prompt_cfg.get("length_distribution", {})
        if not length_dist:
            # 无分布信息时，从所有 prompt 中随机选
            all_prompts = self._prompt_pool["prompts"]
            p = self._rng.choice(all_prompts)
            return (
                copy.deepcopy(p["messages"]),
                p["category"],
                p.get("estimated_tokens", 50),
            )

        # 加权随机选择 bucket
        buckets = []
        weights = []
        for bucket_name, info in length_dist.items():
            buckets.append(bucket_name)
            weights.append(info.get("weight", 0))

        chosen_bucket = self._rng.choices(buckets, weights=weights, k=1)[0]

        # 从对应 bucket 的 prompt 中选取
        pool = self._prompts_by_category.get(chosen_bucket, [])
        if not pool:
            # 当前 bucket 无 prompt，从最近的 bucket 借用
            logger.warning(f"bucket '{chosen_bucket}' 无可用 prompt，从其他 bucket 借用")
            for fallback in ["medium", "short", "long"]:
                pool = self._prompts_by_category.get(fallback, [])
                if pool:
                    break

        if not pool:
            raise ValueError("prompt 语料池为空，无法采样")

        p = self._rng.choice(pool)
        return (
            copy.deepcopy(p["messages"]),
            chosen_bucket,
            p.get("estimated_tokens", 50),
        )

    def _sample_max_tokens(self, output_cfg: dict = None) -> int:
        """按 max_tokens_distribution 加权采样"""
        if output_cfg is None:
            output_cfg = self._config["output"]

        dist = output_cfg.get("max_tokens_distribution", {})
        if not dist:
            return 256  # 默认值

        values = []
        weights = []
        for _, info in dist.items():
            values.append(info["value"])
            weights.append(info.get("weight", 1.0))

        return self._rng.choices(values, weights=weights, k=1)[0]

    def _apply_shared_prefix(self, messages: list[dict],
                             ratio: float) -> tuple:
        """按共享前缀比例决定是否替换为前缀模板+随机后缀"""
        if not self._prefix_pool or self._rng.random() >= ratio:
            return messages, None

        prefixes = self._prefix_pool.get("prefixes", [])
        if not prefixes:
            return messages, None

        prefix = self._rng.choice(prefixes)
        suffix = self._rng.choice(prefix["suffix_pool"])

        new_messages = []
        if prefix.get("system_message"):
            new_messages.append({"role": "system", "content": prefix["system_message"]})
        user_content = prefix.get("user_prefix", "") + suffix
        new_messages.append({"role": "user", "content": user_content})

        return new_messages, prefix["id"]


if __name__ == "__main__":
    import sys

    # 简单自测
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/workloads/workload_baseline.json"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    wg = WorkloadGenerator(cfg["workload"])
    reqs = wg.generate()

    print(f"生成 {len(reqs)} 个请求")
    print(f"首个请求: id={reqs[0]['request_id']}, "
          f"time={reqs[0]['scheduled_time_s']:.3f}s, "
          f"bucket={reqs[0]['prompt_length_bucket']}, "
          f"max_tokens={reqs[0]['max_tokens']}")
    if len(reqs) > 1:
        print(f"末个请求: id={reqs[-1]['request_id']}, "
              f"time={reqs[-1]['scheduled_time_s']:.3f}s")

    # 可复现性验证
    wg2 = WorkloadGenerator(cfg["workload"])
    reqs2 = wg2.generate()
    for i in range(len(reqs)):
        assert reqs[i]["request_id"] == reqs2[i]["request_id"], f"请求 {i} ID 不一致"
        assert reqs[i]["scheduled_time_s"] == reqs2[i]["scheduled_time_s"], f"请求 {i} 调度时间不一致"
        assert reqs[i]["prompt_length_bucket"] == reqs2[i]["prompt_length_bucket"], f"请求 {i} bucket 不一致"
    print("可复现性验证: 通过 ✓")

    # phase 分布
    from collections import Counter
    phases = Counter(r["phase_name"] for r in reqs)
    print(f"Phase 分布: {dict(phases)}")
