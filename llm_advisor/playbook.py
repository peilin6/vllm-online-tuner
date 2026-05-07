#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""playbook.py — 瓶颈 → 可调参数白名单 + 工具路由 + 方向提示。

目的：让 P-LLM 不再"自由发挥"，而是收到一份针对当前 bottleneck 的"作战手册"。
- allowed_params: 仅允许从中挑 1 个参数；其它参数被视为越权。
- direction:      该 bottleneck 下推荐的调整方向（"up"/"down"/"toggle"）。
- preferred_tool: 命中此 bottleneck 时优先调用的 B 类算法工具（可空）。
- required_reads: 必须调用的 A 类只读工具（默认至少 query_param_docs）。
- rationale:      给 LLM 看的简短解释，写进 prompt。

修改本文件 = 同时修改 LLM 行为与 fallback 行为，是单一事实来源。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlaybookEntry:
    bottleneck: str
    allowed_params: tuple[str, ...]
    direction: dict[str, str] = field(default_factory=dict)   # param -> "up"/"down"/"toggle"
    preferred_tool: str | None = None
    required_reads: tuple[str, ...] = ("query_param_docs",)
    rationale: str = ""


# 瓶颈枚举（与 schemas.BOTTLENECK_ENUM 对齐）
BOTTLENECK_PLAYBOOK: dict[str, PlaybookEntry] = {
    "prefill_bound": PlaybookEntry(
        bottleneck="prefill_bound",
        allowed_params=("max_num_batched_tokens", "enable_chunked_prefill",
                        "enable_prefix_caching", "max_num_seqs"),
        direction={"max_num_batched_tokens": "up",
                   "enable_chunked_prefill": "toggle",
                   "enable_prefix_caching": "toggle",
                   "max_num_seqs": "down"},
        preferred_tool="param_sensitivity",
        rationale="prefill 阶段算力受限（TTFT 高、ITL 正常）：vLLM 官方建议优先增大 "
                  "max_num_batched_tokens（>=8192）并开启 chunked prefill；若大量请求"
                  "共享系统 prompt，开启 prefix caching；只有当 KV 紧张时才降并发。",
    ),
    "decode_bound": PlaybookEntry(
        bottleneck="decode_bound",
        allowed_params=("max_num_batched_tokens", "max_num_seqs",
                        "enable_prefix_caching"),
        direction={"max_num_batched_tokens": "down",
                   "max_num_seqs": "down",
                   "enable_prefix_caching": "toggle"},
        preferred_tool="bo_suggest",
        rationale="decode 阶段 memory-bandwidth bound（ITL/TPOT 高、TTFT 正常）：vLLM "
                  "官方建议降低 max_num_batched_tokens（如 8192→4096→2048）减少 prefill "
                  "对 decode 的干扰，必要时也降 max_num_seqs。",
    ),
    "kv_cache_pressure": PlaybookEntry(
        bottleneck="kv_cache_pressure",
        allowed_params=("gpu_memory_utilization", "max_num_seqs",
                        "max_num_batched_tokens", "block_size"),
        direction={"gpu_memory_utilization": "up",
                   "max_num_seqs": "down",
                   "max_num_batched_tokens": "down",
                   "block_size": "toggle"},
        preferred_tool="local_grid",
        rationale="KV 显存吃紧（kv_p95>0.95）：vLLM 官方建议优先 ↑gpu_memory_utilization "
                  "（如 0.90→0.92→0.95），其次 ↓max_num_seqs / ↓max_num_batched_tokens 缓解 KV 压力。",
    ),
    "preempt_storm": PlaybookEntry(
        bottleneck="preempt_storm",
        allowed_params=("gpu_memory_utilization", "max_num_seqs",
                        "max_num_batched_tokens", "swap_space",
                        "enable_chunked_prefill"),
        direction={"gpu_memory_utilization": "up",
                   "max_num_seqs": "down",
                   "max_num_batched_tokens": "down",
                   "swap_space": "up",
                   "enable_chunked_prefill": "toggle"},
        preferred_tool="param_sensitivity",
        rationale="抢占风暴（preempt>5/min）：vLLM 官方建议先 ↑gpu_memory_utilization 给"
                  "KV 腾空间，再 ↓max_num_seqs / ↓max_num_batched_tokens 降批并发；"
                  "swap_space 仅在重算成本高时考虑。",
    ),
    "queue_backlog": PlaybookEntry(
        bottleneck="queue_backlog",
        allowed_params=("max_num_seqs", "max_num_batched_tokens",
                        "enable_chunked_prefill"),
        direction={"max_num_seqs": "up",
                   "max_num_batched_tokens": "up",
                   "enable_chunked_prefill": "toggle"},
        preferred_tool="bo_suggest",
        rationale="队列积压（queue_time 占比高）：GPU 未饱和但请求等待长，提高并发与"
                  "批 token 上限以挖掘吞吐。",
    ),
    "underutilized": PlaybookEntry(
        bottleneck="underutilized",
        allowed_params=("max_num_batched_tokens", "max_num_seqs",
                        "gpu_memory_utilization"),
        direction={"max_num_batched_tokens": "up",
                   "max_num_seqs": "up",
                   "gpu_memory_utilization": "up"},
        preferred_tool="bo_suggest",
        rationale="资源闲置（吞吐低且 GPU/KV 都不紧张）：vLLM 官方吞吐优先建议 "
                  "max_num_batched_tokens>=8192，可同时 ↑max_num_seqs。",
    ),
    "slo_margin_low": PlaybookEntry(
        bottleneck="slo_margin_low",
        allowed_params=("max_num_batched_tokens", "max_num_seqs",
                        "enable_chunked_prefill"),
        direction={"max_num_batched_tokens": "down",
                   "max_num_seqs": "down",
                   "enable_chunked_prefill": "toggle"},
        preferred_tool="pareto_front",
        rationale="SLO 余量不足（headroom<10%）：保守回退保护 P99——↓max_num_batched_tokens "
                  "优先保护 ITL，必要时再 ↓max_num_seqs；用 Pareto 前沿权衡吞吐与延迟。",
    ),
    "converged": PlaybookEntry(
        bottleneck="converged",
        allowed_params=(),                   # 不允许再改
        direction={},
        preferred_tool=None,
        required_reads=(),
        rationale="已收敛：不再调参，应输出 stop。",
    ),
}


def get_entry(bottleneck: str) -> PlaybookEntry:
    """瓶颈 -> playbook 条目；未识别时回退到 underutilized（最保守的探索方向）。"""
    return BOTTLENECK_PLAYBOOK.get(bottleneck, BOTTLENECK_PLAYBOOK["underutilized"])


def render_for_prompt(bottleneck: str) -> str:
    """把 entry 渲染成简短 markdown 片段，供 P-LLM user prompt 使用。"""
    e = get_entry(bottleneck)
    if not e.allowed_params:
        return (f"### Playbook（bottleneck={bottleneck}）\n"
                f"- 状态：已收敛，禁止再调参，请输出 stop。\n")
    lines = [f"### Playbook（bottleneck={bottleneck}）",
             f"- 允许调整的参数白名单（必须从中选 1 个）: {list(e.allowed_params)}",
             "- 方向提示:"]
    for p, d in e.direction.items():
        arrow = {"up": "↑（增大）", "down": "↓（减小）", "toggle": "切换/枚举"}.get(d, d)
        lines.append(f"    - {p}: {arrow}")
    if e.preferred_tool:
        lines.append(f"- 优先调用的 B 类算法工具: {e.preferred_tool}")
    if e.required_reads:
        lines.append(f"- 必须先调用的 A 类只读工具: {list(e.required_reads)}")
    lines.append(f"- 经验依据: {e.rationale}")
    return "\n".join(lines) + "\n"
