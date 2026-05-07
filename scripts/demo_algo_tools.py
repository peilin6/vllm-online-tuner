#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/demo_algo_tools.py — 演示 P-LLM 如何调用 B 类算法工具完成参数寻优。

与 demo_fake_metrics.py 的差异:
  1. 把 tuner.optimizer.B_TOOL_SPECS 注入 ToolRegistry，让 LLM 看得到
     bo_suggest / param_sensitivity / pareto_front / local_grid /
     cluster_workload_phases 这 5 个算法工具
  2. 预填 6 条互相不同的 trial 历史，使 BO/灵敏度/Pareto 有足够数据可算
  3. 在每一轮 tool-call 后打印 (round, name, arguments, result_brief)，
     方便直观看到 LLM 调用了哪个算法、传了什么参数、得到什么数值

运行:
    python scripts/demo_algo_tools.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm_advisor.config import build_llm_client
from llm_advisor.diagnoser import diagnose
from llm_advisor.proposer import propose
from llm_advisor.schemas import ConfigDelta, StopSignal
from tuner.memory import ExperienceMemory, TrialRecord
from tuner.metrics_parser import TrialMetrics
from tuner.tools import ToolRegistry
from tuner.optimizer import B_TOOL_SPECS


def make_metrics(**kw) -> TrialMetrics:
    base = dict(
        throughput_req_per_s=1.0, throughput_tok_per_s=1000.0,
        ttft_p95_ms=200.0, tpot_p95_ms=20.0, latency_p95_ms=2000.0,
        preemptions_total=0, preemption_rate_per_min=0.0,
        kv_cache_usage_p95_pct=0.5, queue_time_p95_ms=10.0,
        success=True, early_killed=False, wall_time_s=60.0,
    )
    base.update(kw)
    return TrialMetrics(**base)


def _rec(tid: str, cfg: dict, *, source="agent", **mkw) -> TrialRecord:
    return TrialRecord(trial_id=tid, config=cfg,
                       metrics=make_metrics(**mkw), source=source)


def banner(t: str) -> None:
    print()
    print("=" * 78); print(t); print("=" * 78)


# ---- 预填一个有 6 个 trial 的搜索历史，吞吐数据呈"中等 batch 最优"的非线性形态 ----
HISTORY = [
    # (trial_id, max_num_seqs, max_num_batched_tokens, gpu_mem_util, throughput, ttft, tpot, lat, kv)
    ("baseline_0",  64, 4096, 0.85, 1500.0, 180.0, 18.0, 1500.0, 0.55),
    ("trial_1",    32, 2048, 0.85, 1200.0, 150.0, 16.0, 1300.0, 0.40),
    ("trial_2",    96, 4096, 0.85, 1700.0, 200.0, 22.0, 1700.0, 0.72),
    ("trial_3",   128, 4096, 0.85, 1650.0, 240.0, 28.0, 1900.0, 0.85),
    ("trial_4",    96, 8192, 0.90, 1850.0, 210.0, 24.0, 1750.0, 0.78),
    ("trial_5",    64, 4096, 0.92, 1600.0, 175.0, 18.5, 1550.0, 0.62),
]


def main() -> int:
    mem = ExperienceMemory()
    for tid, ns, mbt, gmu, tput, ttft, tpot, lat, kv in HISTORY:
        cfg = {"max_num_seqs": ns, "max_num_batched_tokens": mbt,
               "gpu_memory_utilization": gmu, "block_size": 16,
               "enable_prefix_caching": False, "enable_chunked_prefill": True}
        src = "baseline" if tid == "baseline_0" else "agent"
        mem.add(_rec(tid, cfg, source=src,
                     throughput_tok_per_s=tput, ttft_p95_ms=ttft,
                     tpot_p95_ms=tpot, latency_p95_ms=lat,
                     kv_cache_usage_p95_pct=kv))

    # 当前 trial：吞吐 1850 算最好，但 KV 已到 0.78，目标是再榨一点吞吐
    current = mem.all()[-1]   # trial_4

    banner("演示场景：已有 6 条 trial 历史，让 P-LLM 用算法工具寻找下一组参数")
    print(f"  历史 trial 数: {len(mem.all())}（baseline + 5 次实验）")
    print(f"  current best : {current.trial_id}  cfg={current.config}")
    print(f"                  tput={current.metrics.throughput_tok_per_s} "
          f"ttft={current.metrics.ttft_p95_ms} kv={current.metrics.kv_cache_usage_p95_pct}")

    client = build_llm_client()
    print(f"\n[setup] LLM={client.cfg.base_url}  model={client.cfg.model}")

    # ---------- A-LLM 诊断 ----------
    banner("Step 1 — A-LLM 诊断")
    diag = diagnose(client, mem, current)
    print(f"  bottleneck={diag.bottleneck}  conf={diag.confidence:.2f}")
    print(f"  evidence  ={diag.evidence}")
    print(f"  source    ={diag.source}")

    # ---------- 构建 ToolRegistry：A 类（6 个）+ B 类（5 个算法工具）----------
    tools = ToolRegistry(memory=mem, extra_tools=B_TOOL_SPECS)
    print(f"\n[tools] A 类（6）+ B 类（5）共 {len(tools.openai_tools_schema())} 个工具:")
    for spec in tools.openai_tools_schema():
        print(f"   - {spec['function']['name']:24s}  {spec['function']['description'][:60]}...")

    # ---------- P-LLM 提议 ----------
    banner("Step 2 — P-LLM 提议（多轮 function-calling，可调用算法工具）")
    proposal = propose(
        client, tools, mem, diag,
        current_config=current.config,
        max_rounds=6,
    )

    if isinstance(proposal, StopSignal):
        print(f"  StopSignal: {proposal.reason}")
        return 0

    # tool_trace 记录在 proposal.tools_used 里只有名字；要看完整调用细节，
    # 直接复刻一遍 chat_with_tools 的中间结果会再花一次额外的 token；
    # 这里改用从 ToolRegistry 同步抽取最后一次实际调用的工具历史（来自 propose 内部）。
    print(f"  param          : {proposal.param}")
    print(f"  old -> new     : {proposal.old_value} -> {proposal.new_value}")
    print(f"  expected_effect: {proposal.expected_effect}")
    print(f"  tools_used     : {proposal.tools_used}")
    print(f"  source         : {proposal.source}")

    # ---------- 直接展示算法工具的本地输出（与 LLM 实际看到的一致）----------
    banner("Step 3 — 直接演示这些算法工具的输出（不经 LLM）")
    print("\n[bo_suggest] Optuna TPE 在 6 个观测点上拟合，提议下一组参数：")
    out = tools.dispatch("bo_suggest", {
        "target": "throughput_tok_per_s",
        "direction": "maximize",
        "params": ["max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization"],
    })
    print("  " + json.dumps(out, ensure_ascii=False))

    print("\n[param_sensitivity] 用 Spearman 等级相关 + 极差，给出参数对吞吐的影响排序：")
    out = tools.dispatch("param_sensitivity", {
        "target": "throughput_tok_per_s",
        "params": ["max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization"],
    })
    print("  " + json.dumps(out, ensure_ascii=False))

    print("\n[pareto_front] (吞吐 max, 延迟 min) 二维 Pareto 前沿：")
    out = tools.dispatch("pareto_front", {
        "obj_max": "throughput_tok_per_s",
        "obj_min": "latency_p95_ms",
    })
    print("  " + json.dumps(out, ensure_ascii=False))

    print("\n[local_grid] 在当前 best 附近 ±1 邻域生成局部候选格：")
    around = {"max_num_seqs": current.config["max_num_seqs"],
              "max_num_batched_tokens": current.config["max_num_batched_tokens"]}
    out = tools.dispatch("local_grid", {"around": around, "radius": 1})
    print("  " + json.dumps(out, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
