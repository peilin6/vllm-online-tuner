#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/smoke_llm_api.py — 真实 DeepSeek API 端到端烟囱测试。

测试维度:
  T1. /chat/completions 基本连通性 + JSON 输出
  T2. A-LLM diagnose() 真实调用，验证 source==llm 且 bottleneck 命中规则
  T3. P-LLM propose() 真实调用 + tool-call 多轮，验证输出符合 playbook
  T4. R-LLM reflect() 真实调用，验证 verdict 合法
  T5. VtaAgent 端到端跑 3 步（runner_fn 用 mock 指标）

运行:
    python scripts/smoke_llm_api.py            # 全部
    python scripts/smoke_llm_api.py --only T2  # 只跑某项

每项失败不会阻塞后续项。最终给出汇总。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# 让脚本能直接 python scripts/smoke_llm_api.py 运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm_advisor.config import build_llm_client
from llm_advisor.diagnoser import diagnose
from llm_advisor.proposer import propose
from llm_advisor.reflector import reflect
from llm_advisor.schemas import ConfigDelta, StopSignal
from tuner.judge import Judge
from tuner.memory import ExperienceMemory, TrialRecord
from tuner.metrics_parser import TrialMetrics
from tuner.tools import ToolRegistry


# ---------- helpers ----------
def _metrics(**kw) -> TrialMetrics:
    base = dict(
        throughput_req_per_s=1.0, throughput_tok_per_s=1000.0,
        ttft_p95_ms=200.0, tpot_p95_ms=20.0, latency_p95_ms=2000.0,
        preemptions_total=0, preemption_rate_per_min=0.0,
        kv_cache_usage_p95_pct=0.5, queue_time_p95_ms=10.0,
        success=True, early_killed=False, wall_time_s=60.0,
    )
    base.update(kw)
    return TrialMetrics(**base)


def _record(trial_id: str, source: str = "agent", **mkw) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        config={"max_num_seqs": 64, "max_num_batched_tokens": 4096,
                "gpu_memory_utilization": 0.85},
        metrics=_metrics(**mkw), source=source,
    )


# ---------- tests ----------
def test_T1_connectivity(client) -> str:
    """最小连通性：让模型返回一个 JSON 对象。"""
    resp = client.chat(
        [
            {"role": "system", "content": "你是 JSON 输出助手。仅输出严格 JSON。"},
            {"role": "user", "content": '回复 {"ok": true, "model": "deepseek"} 这个 JSON。'},
        ],
        response_format={"type": "json_object"},
        use_cache=False,
        temperature=0.0,
    )
    msg = resp["choices"][0]["message"]["content"]
    obj = json.loads(msg)
    assert obj.get("ok") is True, f"unexpected payload: {obj}"
    return f"model returned: {obj}"


def test_T2_diagnose_kv_pressure(client) -> str:
    mem = ExperienceMemory()
    mem.add(_record("baseline_0", source="baseline"))
    # 制造 kv_cache_usage_p95 > 0.95（应该命中 R2 -> kv_cache_pressure）
    rec = _record("trial_kv", kv_cache_usage_p95_pct=0.97,
                  ttft_p95_ms=240.0, tpot_p95_ms=22.0)
    mem.add(rec)
    res = diagnose(client, mem, rec)
    assert res.source == "llm", f"diagnose 未走 LLM 路径，source={res.source}"
    assert res.bottleneck == "kv_cache_pressure", \
        f"bottleneck 期望 kv_cache_pressure，实际 {res.bottleneck}; evidence={res.evidence}"
    return f"bottleneck={res.bottleneck}, conf={res.confidence:.2f}, evidence={res.evidence!r}"


def test_T3_propose_kv_pressure(client) -> str:
    mem = ExperienceMemory()
    mem.add(_record("baseline_0", source="baseline"))
    rec = _record("trial_kv", kv_cache_usage_p95_pct=0.97)
    mem.add(rec)
    diag = diagnose(client, mem, rec)
    tools = ToolRegistry(memory=mem)
    proposal = propose(
        client, tools, mem, diag,
        current_config=rec.config,
        max_rounds=3,
    )
    assert isinstance(proposal, ConfigDelta), f"propose 返回 {type(proposal).__name__}"
    assert proposal.source == "llm", f"propose 未走 LLM 路径，source={proposal.source}"
    # playbook[kv_cache_pressure] 白名单
    legal = {"gpu_memory_utilization", "max_num_seqs",
             "max_num_batched_tokens", "block_size"}
    assert proposal.param in legal, f"param={proposal.param} 不在 playbook 白名单"
    return (f"param={proposal.param} {proposal.old_value}->{proposal.new_value}, "
            f"tools={proposal.tools_used}")


def test_T4_reflect(client) -> str:
    mem = ExperienceMemory()
    base = _record("baseline_0", source="baseline")
    mem.add(base)
    new = _record("trial_after",
                  throughput_tok_per_s=1100.0,   # +10%
                  kv_cache_usage_p95_pct=0.7)
    mem.add(new)
    proposal = ConfigDelta(
        param="gpu_memory_utilization", old_value=0.85, new_value=0.92,
        reason="release more KV", expected_effect={"throughput": "+5~10%"},
    )
    cc = {"pass_": True, "violations": [], "reason": ""}
    res = reflect(client, proposal, base, new, cc, mem.notes)
    assert res.verdict in ("accept", "partial", "reject"), \
        f"verdict 非法: {res.verdict}"
    assert res.next_move_hint in ("double_down", "explore_other", "rollback", "stop"), \
        f"hint 非法: {res.next_move_hint}"
    return (f"verdict={res.verdict} hint={res.next_move_hint} "
            f"new_note={res.new_note!r} src={res.source}")


def test_T5_agent_loop(client) -> str:
    """跑 3 步 VtaAgent，runner_fn 用确定性 mock 指标。"""
    from tuner.agent import VtaAgent

    mem = ExperienceMemory()
    judge = Judge(memory=mem, max_steps=10)
    tools = ToolRegistry(memory=mem)

    # mock runner: 每改一次都把 throughput +5%，但 kv 略升
    state = {"step": 0, "tput": 1000.0, "kv": 0.5}

    def fake_runner(cfg, *, baseline_throughput_tok_per_s):
        state["step"] += 1
        state["tput"] *= 1.05
        state["kv"] = min(0.99, state["kv"] + 0.05)
        return _metrics(throughput_tok_per_s=state["tput"],
                        kv_cache_usage_p95_pct=state["kv"])

    baseline_metrics = _metrics()
    baseline_cfg = {"max_num_seqs": 64, "max_num_batched_tokens": 4096,
                    "gpu_memory_utilization": 0.85}

    agent = VtaAgent(
        memory=mem, tools=tools, judge=judge,
        runner_fn=fake_runner, client=client,
        run_id="smoke_t5", use_llm=True,
    )
    report = agent.run(baseline_metrics, baseline_cfg, max_steps=3)
    return (f"steps={report.n_steps}, llm_calls={report.llm_call_counts}, "
            f"stop={report.stop_reason}, wall={report.total_wall_time_s}s")


TESTS = [
    ("T1", "connectivity",         test_T1_connectivity),
    ("T2", "A-LLM diagnose",       test_T2_diagnose_kv_pressure),
    ("T3", "P-LLM propose",        test_T3_propose_kv_pressure),
    ("T4", "R-LLM reflect",        test_T4_reflect),
    ("T5", "VtaAgent end-to-end",  test_T5_agent_loop),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="只跑指定 ID，如 --only T1 T2")
    args = parser.parse_args()

    client = build_llm_client()
    print(f"[setup] base_url={client.cfg.base_url}  model={client.cfg.model}")
    print(f"[setup] api_key=***{client.cfg.api_key[-6:]}\n")

    results = []
    for tid, name, fn in TESTS:
        if args.only and tid not in args.only:
            continue
        t0 = time.perf_counter()
        try:
            detail = fn(client)
            dt = time.perf_counter() - t0
            print(f"[PASS] {tid} {name} ({dt:.1f}s)")
            print(f"        {detail}")
            results.append((tid, True, detail))
        except Exception as e:    # noqa: BLE001
            dt = time.perf_counter() - t0
            print(f"[FAIL] {tid} {name} ({dt:.1f}s): {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            results.append((tid, False, str(e)))
        print()

    n_ok = sum(1 for _, ok, _ in results if ok)
    n = len(results)
    print(f"=== 汇总: {n_ok}/{n} pass ===")
    sys.exit(0 if n_ok == n else 1)


if __name__ == "__main__":
    main()
