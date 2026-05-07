#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/demo_fake_metrics.py — 假设 vLLM 已存在，给 agent 喂捏造 metrics，
观察其完整的诊断 / 工具调用 / 调参链路。

运行:
    python scripts/demo_fake_metrics.py
    python scripts/demo_fake_metrics.py --scenario kv_pressure
    python scripts/demo_fake_metrics.py --scenario decode_bound
    python scripts/demo_fake_metrics.py --scenario underutilized

打印内容:
    1. 输入：baseline 与 current trial 的 config + metrics
    2. A-LLM 诊断：bottleneck / confidence / evidence / source(llm or fallback)
    3. P-LLM 提议：调用了哪些工具（多轮 function-calling 全 trace）+ 最终 ConfigDelta
    4. （可选）模拟 vLLM 跑出新 metrics → R-LLM 反思 verdict + new_note
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm_advisor.config import build_llm_client
from llm_advisor.diagnoser import diagnose
from llm_advisor.proposer import propose
from llm_advisor.reflector import reflect
from llm_advisor.schemas import ConfigDelta, StopSignal
from tuner.memory import ExperienceMemory, TrialRecord
from tuner.metrics_parser import TrialMetrics
from tuner.tools import ToolRegistry


# ---------------- helpers ----------------
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


# ---------------- 三个捏造场景 ----------------
SCENARIOS = {
    # KV 缓存压力：kv_p95 极高、有少量抢占、TPOT 略升
    "kv_pressure": {
        "desc": "KV 缓存压力（kv_p95=0.97，有 12 次抢占，TPOT 比 baseline 上升）",
        "baseline_cfg": {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.85,
            "block_size": 16,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
        },
        "baseline_metrics": dict(
            throughput_tok_per_s=1800.0, ttft_p95_ms=180.0, tpot_p95_ms=18.0,
            latency_p95_ms=1500.0, kv_cache_usage_p95_pct=0.55,
            preemptions_total=0, preemption_rate_per_min=0.0,
            queue_time_p95_ms=8.0,
        ),
        "current_metrics": dict(
            throughput_tok_per_s=1700.0, ttft_p95_ms=210.0, tpot_p95_ms=24.0,
            latency_p95_ms=1900.0, kv_cache_usage_p95_pct=0.97,
            preemptions_total=12, preemption_rate_per_min=2.4,
            queue_time_p95_ms=18.0,
        ),
    },
    # decode-bound：TPOT 高且 TTFT 正常
    "decode_bound": {
        "desc": "Decode-bound（TPOT_p95=42ms 远超 baseline，TTFT 与 baseline 相当）",
        "baseline_cfg": {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.90,
            "block_size": 16,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
        "baseline_metrics": dict(
            throughput_tok_per_s=2400.0, ttft_p95_ms=160.0, tpot_p95_ms=20.0,
            latency_p95_ms=1800.0, kv_cache_usage_p95_pct=0.62,
            queue_time_p95_ms=10.0,
        ),
        "current_metrics": dict(
            throughput_tok_per_s=1900.0, ttft_p95_ms=170.0, tpot_p95_ms=42.0,
            latency_p95_ms=3200.0, kv_cache_usage_p95_pct=0.65,
            queue_time_p95_ms=12.0,
        ),
    },
    # 资源未充分利用：吞吐低但延迟正常
    "underutilized": {
        "desc": "资源未充分利用（KV 仅 0.30、TTFT/TPOT 都低，吞吐远低于 baseline）",
        "baseline_cfg": {
            "max_num_seqs": 32,
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.75,
            "block_size": 16,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
        },
        "baseline_metrics": dict(
            throughput_tok_per_s=1200.0, ttft_p95_ms=120.0, tpot_p95_ms=15.0,
            latency_p95_ms=1200.0, kv_cache_usage_p95_pct=0.40,
            queue_time_p95_ms=5.0,
        ),
        "current_metrics": dict(
            throughput_tok_per_s=900.0, ttft_p95_ms=110.0, tpot_p95_ms=14.0,
            latency_p95_ms=1100.0, kv_cache_usage_p95_pct=0.30,
            queue_time_p95_ms=4.0,
        ),
    },
}


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def dump_metrics(label: str, m: TrialMetrics) -> None:
    print(f"  {label}: tput={m.throughput_tok_per_s:.0f} tok/s  "
          f"ttft_p95={m.ttft_p95_ms:.0f}ms  tpot_p95={m.tpot_p95_ms:.1f}ms  "
          f"lat_p95={m.latency_p95_ms:.0f}ms  kv_p95={m.kv_cache_usage_p95_pct:.2f}  "
          f"preempt={m.preemptions_total}  qt_p95={m.queue_time_p95_ms:.0f}ms")


def run(scenario: str, do_reflect: bool) -> int:
    sc = SCENARIOS[scenario]
    banner(f"Scenario: {scenario}  —  {sc['desc']}")

    # 构建 baseline + current trial
    baseline = TrialRecord(
        trial_id="baseline_0",
        config=dict(sc["baseline_cfg"]),
        metrics=make_metrics(**sc["baseline_metrics"]),
        source="baseline",
    )
    current = TrialRecord(
        trial_id="trial_current",
        config=dict(sc["baseline_cfg"]),       # 还没改参数，跟 baseline 同 config
        metrics=make_metrics(**sc["current_metrics"]),
        source="agent",
    )
    mem = ExperienceMemory()
    mem.add(baseline)
    mem.add(current)

    print("\n[输入] baseline config:")
    print("  " + json.dumps(sc["baseline_cfg"], ensure_ascii=False))
    print("[输入] metrics:")
    dump_metrics("baseline", baseline.metrics)
    dump_metrics("current ", current.metrics)

    # 构建 LLM 客户端
    client = build_llm_client()
    print(f"\n[setup] LLM={client.cfg.base_url} model={client.cfg.model}")

    # ---------- A-LLM 诊断 ----------
    banner("Step 1 — A-LLM 诊断")
    diag = diagnose(client, mem, current)
    print(f"  source       : {diag.source}")
    print(f"  bottleneck   : {diag.bottleneck}")
    print(f"  confidence   : {diag.confidence:.2f}")
    print(f"  evidence     : {diag.evidence}")
    print(f"  should_stop  : {diag.should_stop}")
    print(f"  hypothesis   : {diag.hypothesis}")
    print(f"  slo_pressure : {diag.slo_pressure}")

    # ---------- P-LLM 提议 ----------
    banner("Step 2 — P-LLM 提议（多轮 function-calling）")
    tools = ToolRegistry(memory=mem)
    print(f"  registered tools: {[t['function']['name'] for t in tools.openai_tools_schema()]}")

    proposal = propose(
        client, tools, mem, diag,
        current_config=current.config,
        max_rounds=4,
    )

    if isinstance(proposal, StopSignal):
        print(f"  -> StopSignal: reason={proposal.reason} source={proposal.source}")
        print("\n=== 演示结束（agent 主动停止）===")
        return 0

    print(f"  source         : {proposal.source}")
    print(f"  param          : {proposal.param}")
    print(f"  old_value      : {proposal.old_value}")
    print(f"  new_value      : {proposal.new_value}")
    print(f"  hypothesis     : {proposal.hypothesis_ref}")
    print(f"  expected_effect: {proposal.expected_effect}")
    print(f"  rollback_if    : {proposal.rollback_if}")
    print(f"  tools_used     : {proposal.tools_used}")

    # ---------- 可选：模拟新 trial → R-LLM 反思 ----------
    if not do_reflect:
        return 0

    banner("Step 3 — 模拟 vLLM 应用新参数后的 trial，然后 R-LLM 反思")
    new_cfg = dict(current.config)
    new_cfg[proposal.param] = proposal.new_value

    # 简单启发式：根据 bottleneck 给一个"看起来合理"的新 metrics
    bn = diag.bottleneck
    new_m = sc["current_metrics"].copy()
    if bn == "kv_cache_pressure":
        new_m["kv_cache_usage_p95_pct"] = 0.78
        new_m["preemptions_total"] = 1
        new_m["preemption_rate_per_min"] = 0.2
        new_m["throughput_tok_per_s"] *= 1.08
    elif bn == "decode_bound":
        new_m["tpot_p95_ms"] *= 0.7
        new_m["latency_p95_ms"] *= 0.78
        new_m["throughput_tok_per_s"] *= 1.05
    elif bn == "underutilized":
        new_m["throughput_tok_per_s"] *= 1.20
        new_m["kv_cache_usage_p95_pct"] = 0.55
    else:
        new_m["throughput_tok_per_s"] *= 1.03

    new_trial = TrialRecord(
        trial_id="trial_after_change",
        config=new_cfg,
        metrics=make_metrics(**new_m),
        source="agent",
    )
    mem.add(new_trial)
    print(f"  applied: {proposal.param} {proposal.old_value} -> {proposal.new_value}")
    dump_metrics("before ", current.metrics)
    dump_metrics("after  ", new_trial.metrics)

    cc = {"pass_": True, "violations": [], "reason": ""}
    refl = reflect(client, proposal, current, new_trial, cc, mem.notes)
    print()
    print(f"  R-LLM source   : {refl.source}")
    print(f"  verdict        : {refl.verdict}")
    print(f"  next_move_hint : {refl.next_move_hint}")
    print(f"  new_note       : {refl.new_note}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fake-metrics agent demo")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        default="kv_pressure")
    parser.add_argument("--no-reflect", action="store_true",
                        help="跳过 R-LLM 反思（只做 diagnose+propose）")
    args = parser.parse_args()
    return run(args.scenario, do_reflect=not args.no_reflect)


if __name__ == "__main__":
    raise SystemExit(main())
