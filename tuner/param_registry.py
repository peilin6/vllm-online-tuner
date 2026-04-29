#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
param_registry.py — RESTART 类参数登记表

Task 6.5。所有需要重启 vLLM 才能生效的可调参数（共 9 条）的元信息：
- candidates: 推荐取值集合（供 BO/网格/LLM 提议时离散化使用）
- range: 数值型参数的 [low, high] 边界（用于安全裁剪与连续 BO）
- affects: 该参数主要影响的指标维度（吞吐 / 显存 / 延迟 / 调度等）
- notes: 调参经验性说明（供 LLM Tool query_param_docs 直接读出）

只读且纯数据。Judge / Optimizer / Tools 共享同一份事实来源。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParamSpec:
    name: str
    dotted_path: str                 # 在 config.json 中的写入路径，例如 "server.max_num_seqs"
    type: str                        # "int" / "float" / "bool" / "str"
    default: Any
    candidates: list[Any] = field(default_factory=list)
    range: tuple[float, float] | None = None
    affects: list[str] = field(default_factory=list)
    requires_restart: bool = True
    notes: str = ""

    def in_range(self, value: Any) -> bool:
        if self.candidates:
            if value in self.candidates:
                return True
            # 离散候选模式下：仅对数值型允许 range 兜底；str/bool 必须命中候选
            if self.range is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
                lo, hi = self.range
                return lo <= float(value) <= hi
            return False
        if self.range is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            lo, hi = self.range
            return lo <= float(value) <= hi
        # 无 range/candidates 限制时（如 bool/str）只做类型粗检
        if self.type == "bool":
            return isinstance(value, bool)
        if self.type == "str":
            return isinstance(value, str)
        return True

    def clamp(self, value: Any) -> Any:
        """把越界值裁回合法范围；离散候选集走最近邻；其他类型原样返回。"""
        if self.candidates:
            if value in self.candidates:
                return value
            # 数值型最近邻
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                num_cands = [c for c in self.candidates if isinstance(c, (int, float))]
                if num_cands:
                    return min(num_cands, key=lambda c: abs(c - value))
            return self.default
        if self.range is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            lo, hi = self.range
            v = max(lo, min(hi, float(value)))
            return int(v) if self.type == "int" else v
        return value


# ---------------------------------------------------------------------
# 9 条 RESTART 参数（vLLM 0.6.x；Qwen2.5-3B-AWQ on RTX 4060 8GB / Qwen3-8B on A6000）
# ---------------------------------------------------------------------
_SPECS: list[ParamSpec] = [
    ParamSpec(
        name="max_num_seqs",
        dotted_path="server.max_num_seqs",
        type="int", default=32,
        candidates=[8, 16, 32, 64, 96, 128, 192, 256],
        range=(1, 512),
        affects=["throughput", "kv_cache", "ttft"],
        notes="并发 batch 上限。↑ 提升吞吐但加 KV 压力；过大→preempt 飙升。",
    ),
    ParamSpec(
        name="max_num_batched_tokens",
        dotted_path="server.max_num_batched_tokens",
        type="int", default=2048,
        candidates=[1024, 2048, 4096, 8192, 16384],
        range=(256, 32768),
        affects=["throughput", "ttft", "tpot"],
        notes="单 step 处理 token 上限。chunked-prefill 开启时直接决定 prefill 块大小。",
    ),
    ParamSpec(
        name="gpu_memory_utilization",
        dotted_path="server.gpu_memory_utilization",
        type="float", default=0.90,
        candidates=[0.80, 0.85, 0.88, 0.90, 0.92, 0.95],
        range=(0.50, 0.97),
        affects=["kv_cache", "throughput", "stability"],
        notes="vLLM 占用 GPU 显存比例。↑ 给 KV 更多空间但留给图编译/激活更少；>0.95 风险 OOM。",
    ),
    ParamSpec(
        name="block_size",
        dotted_path="server.block_size",
        type="int", default=16,
        candidates=[8, 16, 32],
        range=(8, 32),
        affects=["kv_cache", "throughput"],
        notes="KV 分块大小。小 block 内存利用率高但元数据多；大 block 反之。",
    ),
    ParamSpec(
        name="enable_chunked_prefill",
        dotted_path="server.enable_chunked_prefill",
        type="bool", default=True,
        candidates=[True, False],
        affects=["ttft", "tpot", "fairness"],
        notes="开启后 prefill 与 decode 同 step 调度，显著降低 decode TPOT 抖动。",
    ),
    ParamSpec(
        name="enable_prefix_caching",
        dotted_path="server.enable_prefix_caching",
        type="bool", default=False,
        candidates=[True, False],
        affects=["ttft", "kv_cache"],
        notes="开启后共享前缀复用 KV，长上下文同前缀场景 TTFT 显著下降。",
    ),
    ParamSpec(
        name="swap_space",
        dotted_path="server.swap_space",
        type="float", default=4.0,
        candidates=[0, 2, 4, 8, 16],
        range=(0, 64),
        affects=["preempt_recovery", "throughput"],
        notes="CPU 交换空间(GB)。preempt 频繁时 ↑ 可减少重算成本，但 swap-in 慢。",
    ),
    ParamSpec(
        name="cuda_graph_sizes",
        dotted_path="server.cuda_graph_sizes",
        type="str", default="default",
        candidates=["default", "small", "wide", "off"],
        affects=["startup_time", "tpot"],
        notes="CUDA Graph 捕获 batch 形状集合。off=enforce_eager；wide 启动慢但 decode 快。",
    ),
    ParamSpec(
        name="tensor_parallel_size",
        dotted_path="server.tensor_parallel_size",
        type="int", default=1,
        candidates=[1, 2, 4, 8],
        range=(1, 8),
        affects=["throughput", "memory", "latency"],
        notes="张量并行卡数。单卡保持 1；多卡 ↑ 时显存压力下降但通信开销上升。",
    ),
]

_BY_NAME: dict[str, ParamSpec] = {s.name: s for s in _SPECS}


def all_specs() -> list[ParamSpec]:
    """返回全部 RESTART 参数的列表（拷贝防外部 mutation）。"""
    return list(_SPECS)


def get_spec(name: str) -> ParamSpec:
    if name not in _BY_NAME:
        raise KeyError(f"未登记的 RESTART 参数: {name}")
    return _BY_NAME[name]


def names() -> list[str]:
    return [s.name for s in _SPECS]


def validate_overrides(overrides: dict) -> tuple[bool, list[str]]:
    """检查 overrides 中所有键是否登记 + 值是否在合法集合/范围内。

    Returns:
        (ok, errors): 全部通过则 ok=True, errors=[]
    """
    errs: list[str] = []
    for k, v in overrides.items():
        if k not in _BY_NAME:
            errs.append(f"未登记参数: {k}")
            continue
        spec = _BY_NAME[k]
        if not spec.in_range(v):
            errs.append(f"{k}={v} 超出合法范围 (candidates={spec.candidates}, range={spec.range})")
    return (len(errs) == 0, errs)
