#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools.py — A 类只读工具 + OpenAI function-calling schema 导出 + dispatch

Task 6.6。LLM 通过 function-calling 调用这些只读工具理解当前态势，**绝不允许**
直接调用 `apply_config` 或任何写工具——写入 vLLM 配置只能由 Runner 触发。

A 类工具（6 个）：
1. read_metrics(trial_id?)            最新 / 指定 trial 的标量指标
2. query_param_docs(name)             单个 RESTART 参数的元信息（candidates/range/notes）
3. list_params()                      9 条 RESTART 参数列表
4. get_baseline()                     baseline TrialRecord 紧凑视图
5. get_history_summary(top_k, recent_n)  memory.summarize 的对外暴露
6. compare_trials(trial_id_a, trial_id_b)  字段差异
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import param_registry
from .memory import ExperienceMemory, TrialRecord


FORBIDDEN_TOOLS = frozenset({"apply_config", "write_config", "set_param", "restart_server"})


# =====================================================================
# 单个工具实现（纯函数；签名固定 (memory, **kwargs) -> dict）
# =====================================================================

def _record_view(rec: TrialRecord) -> dict:
    m = rec.metrics
    return {
        "trial_id": rec.trial_id,
        "source": rec.source,
        "config": rec.config,
        "throughput_tok_per_s": round(m.throughput_tok_per_s, 2),
        "throughput_req_per_s": round(m.throughput_req_per_s, 3),
        "ttft_p95_ms": round(m.ttft_p95_ms, 1),
        "tpot_p95_ms": round(m.tpot_p95_ms, 1),
        "latency_p95_ms": round(m.latency_p95_ms, 1),
        "preemptions_total": int(m.preemptions_total),
        "preemption_rate_per_min": round(m.preemption_rate_per_min, 3),
        "kv_p95_pct": round(m.kv_cache_usage_p95_pct, 3),
        "queue_time_p95_ms": round(m.queue_time_p95_ms, 1),
        "success": bool(m.success),
        "early_killed": bool(m.early_killed),
        "wall_time_s": round(m.wall_time_s, 1),
    }


def _tool_read_metrics(memory: ExperienceMemory, *, trial_id: str | None = None) -> dict:
    if trial_id is None:
        recs = memory.recent_n(1)
        if not recs:
            return {"ok": False, "error": "memory 为空"}
        return {"ok": True, "metrics": _record_view(recs[0])}
    for r in memory.all():
        if r.trial_id == trial_id:
            return {"ok": True, "metrics": _record_view(r)}
    return {"ok": False, "error": f"未找到 trial_id={trial_id}"}


def _tool_query_param_docs(memory: ExperienceMemory, *, name: str) -> dict:
    try:
        spec = param_registry.get_spec(name)
    except KeyError as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "name": spec.name,
        "type": spec.type,
        "default": spec.default,
        "candidates": list(spec.candidates),
        "range": list(spec.range) if spec.range is not None else None,
        "affects": list(spec.affects),
        "requires_restart": spec.requires_restart,
        "notes": spec.notes,
    }


def _tool_list_params(memory: ExperienceMemory) -> dict:
    return {
        "ok": True,
        "params": [
            {"name": s.name, "type": s.type, "default": s.default,
             "affects": list(s.affects)}
            for s in param_registry.all_specs()
        ],
    }


def _tool_get_baseline(memory: ExperienceMemory) -> dict:
    for r in memory.all():
        if r.source == "baseline":
            return {"ok": True, "baseline": _record_view(r)}
    return {"ok": False, "error": "memory 中没有 source='baseline' 的记录"}


def _tool_get_history_summary(
    memory: ExperienceMemory, *, top_k: int = 3, recent_n: int = 3
) -> dict:
    return {"ok": True, **memory.summarize(top_k=top_k, recent_n=recent_n)}


def _tool_compare_trials(
    memory: ExperienceMemory, *, trial_id_a: str, trial_id_b: str
) -> dict:
    a = next((r for r in memory.all() if r.trial_id == trial_id_a), None)
    b = next((r for r in memory.all() if r.trial_id == trial_id_b), None)
    if a is None or b is None:
        return {"ok": False, "error": f"未找到 trial(s): a={a is not None} b={b is not None}"}
    va, vb = _record_view(a), _record_view(b)
    metric_keys = [k for k in va if k not in {"trial_id", "source", "config"}]
    deltas = {}
    for k in metric_keys:
        try:
            deltas[k] = round(float(vb[k]) - float(va[k]), 3)
        except (TypeError, ValueError):
            deltas[k] = None
    cfg_diff = {}
    keys = set(a.config) | set(b.config)
    for k in keys:
        if a.config.get(k) != b.config.get(k):
            cfg_diff[k] = {"a": a.config.get(k), "b": b.config.get(k)}
    return {
        "ok": True,
        "a": va, "b": vb,
        "metric_deltas": deltas,
        "config_diff": cfg_diff,
    }


# =====================================================================
# 工具元信息 + Registry
# =====================================================================
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict           # JSON-Schema (OpenAI tools 格式)
    handler: Callable[..., dict]


_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="read_metrics",
        description="读取最近一次或指定 trial 的标量指标（吞吐 / TTFT / TPOT / 抢占 / KV / 早停标志）。只读。",
        parameters={
            "type": "object",
            "properties": {
                "trial_id": {"type": "string",
                             "description": "可选：指定 trial_id；省略则取最近一次。"},
            },
            "required": [],
        },
        handler=_tool_read_metrics,
    ),
    ToolSpec(
        name="query_param_docs",
        description="查询单个 RESTART 参数的元信息（候选集 / 取值范围 / 影响维度 / 调参经验）。",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "参数名，例如 max_num_seqs。"},
            },
            "required": ["name"],
        },
        handler=_tool_query_param_docs,
    ),
    ToolSpec(
        name="list_params",
        description="列出 9 条可调 RESTART 参数及其影响维度。",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_tool_list_params,
    ),
    ToolSpec(
        name="get_baseline",
        description="返回 baseline trial 的紧凑指标视图，作为优化目标的参考起点。",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_tool_get_baseline,
    ),
    ToolSpec(
        name="get_history_summary",
        description="返回历史 trial 的紧凑摘要：best / top_k / 最近 n 个。",
        parameters={
            "type": "object",
            "properties": {
                "top_k": {"type": "integer", "minimum": 1, "default": 3},
                "recent_n": {"type": "integer", "minimum": 1, "default": 3},
            },
            "required": [],
        },
        handler=_tool_get_history_summary,
    ),
    ToolSpec(
        name="compare_trials",
        description="对比两个 trial 的指标差异与配置差异。",
        parameters={
            "type": "object",
            "properties": {
                "trial_id_a": {"type": "string"},
                "trial_id_b": {"type": "string"},
            },
            "required": ["trial_id_a", "trial_id_b"],
        },
        handler=_tool_compare_trials,
    ),
]


class ToolRegistry:
    """A 类只读工具注册中心。`apply_config` 等写工具会被永久拒绝。"""

    def __init__(self, memory: ExperienceMemory, extra_tools: list[ToolSpec] | None = None):
        self._memory = memory
        self._tools: dict[str, ToolSpec] = {t.name: t for t in _TOOLS}
        # 允许 Optimizer (Task 6.7) 注入 B 类纯算法工具
        for t in (extra_tools or []):
            if t.name in FORBIDDEN_TOOLS:
                raise ValueError(f"拒绝注册写工具: {t.name}")
            self._tools[t.name] = t

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"未注册工具: {name}")
        return self._tools[name]

    def openai_tools_schema(self) -> list[dict]:
        """生成 OpenAI tools 字段（function calling）。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def dispatch(self, name: str, arguments: dict | None = None) -> dict:
        """根据 LLM tool_call 名 + 参数，执行对应工具并返回 JSON-able dict。"""
        if name in FORBIDDEN_TOOLS:
            return {
                "ok": False,
                "error": f"forbidden tool: {name}（写入配置必须由 Runner 触发，禁止 LLM 直接调用）",
            }
        if name not in self._tools:
            return {"ok": False, "error": f"未知工具: {name}"}
        spec = self._tools[name]
        try:
            return spec.handler(self._memory, **(arguments or {}))
        except TypeError as e:
            return {"ok": False, "error": f"工具参数不匹配: {e}"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"工具执行异常: {type(e).__name__}: {e}"}
