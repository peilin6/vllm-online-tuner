#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""proposer.py — P-LLM Proposer（带工具调用）

Task 7.2。多轮 function-calling，LLM 通过 ToolRegistry 查 memory / 参数文档 /
触发 BO，然后输出 ConfigDelta。LLM 失败时降级到规则版（按优先级轮询未试过的 candidate）。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from tuner import param_registry
from tuner.memory import ExperienceMemory

from .prompts import P_LLM_SYSTEM, P_LLM_USER_TMPL
from .schemas import ConfigDelta, ParseError, StopSignal, parse_proposal

logger = logging.getLogger(__name__)


# 规则版 fallback 的参数优先级（与 prompt 文档一致）
FALLBACK_PRIORITY = [
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "block_size",
]


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _final_content(final_msg: dict | None) -> str:
    if not final_msg:
        return ""
    return final_msg.get("content") or ""


def _tried_values(memory: ExperienceMemory, param: str) -> set:
    return {r.config[param] for r in memory.all()
            if isinstance(r.config, dict) and param in r.config}


def _fallback_propose(
    memory: ExperienceMemory, current_config: dict,
) -> ConfigDelta | StopSignal:
    """轮询优先级表中第一个还有未试过 candidate 的参数，取候选集中位（偏未试过的方向）。"""
    for name in FALLBACK_PRIORITY:
        try:
            spec = param_registry.get_spec(name)
        except KeyError:
            continue
        cands = list(spec.candidates)
        if not cands:
            continue
        tried = _tried_values(memory, name)
        untried = [c for c in cands if c not in tried]
        if not untried:
            continue
        # 取 untried 中接近候选集中位的那个
        untried.sort(key=lambda v: abs(cands.index(v) - len(cands) // 2))
        new_value = untried[0]
        old_value = current_config.get(name, spec.default)
        if new_value == old_value:
            continue
        return ConfigDelta(
            param=name,
            old_value=old_value,
            new_value=new_value,
            hypothesis_ref="fallback: priority round-robin",
            tools_used=[],
            reason=f"fallback：未试过 {name}={new_value}（candidates={cands}）",
            expected_effect={"throughput_tok_per_s": "?", "risk": "未知"},
            rollback_if="throughput 不升或 SLO 违反",
            source="fallback",
        )
    return StopSignal(reason="fallback: 所有优先参数候选均已试过", source="fallback")


def _build_user_prompt(
    diagnosis_dict: dict, current_config: dict, memory: ExperienceMemory,
) -> str:
    return P_LLM_USER_TMPL.format(
        diagnosis_json=json.dumps(diagnosis_dict, ensure_ascii=False),
        current_config_json=json.dumps(current_config, ensure_ascii=False),
        memory_summary=json.dumps(memory.summarize(top_k=3, recent_n=5),
                                  ensure_ascii=False),
        notes_list=json.dumps(memory.notes[-10:], ensure_ascii=False),
    )


def propose(
    client: Any,                          # LlmClient
    tools: Any,                           # ToolRegistry
    memory: ExperienceMemory,
    diagnosis,                            # DiagnosisResult
    current_config: dict,
    *,
    max_rounds: int = 3,
    use_llm: bool = True,
) -> ConfigDelta | StopSignal:
    """主入口。LLM 失败 / 解析失败时自动降级。"""
    if getattr(diagnosis, "should_stop", False):
        return StopSignal(reason=f"diagnosis.should_stop=True: {diagnosis.bottleneck}",
                          source=getattr(diagnosis, "source", "llm"))

    if client is None or tools is None or not use_llm:
        return _fallback_propose(memory, current_config)

    user = _build_user_prompt(diagnosis.to_dict(), current_config, memory)
    messages = [
        {"role": "system", "content": P_LLM_SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        out = client.chat_with_tools(messages, tools, max_rounds=max_rounds)
    except RuntimeError as e:
        logger.warning("P-LLM 调用失败，fallback: %s", e)
        return _fallback_propose(memory, current_config)

    content = _strip_fence(_final_content(out.get("final")))
    if not content:
        logger.warning("P-LLM 返回空 content，fallback")
        return _fallback_propose(memory, current_config)
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("P-LLM JSON 解析失败 (%s)，fallback", e)
        return _fallback_propose(memory, current_config)
    # 把 LLM 实际调用过的工具名注入 tools_used（若 LLM 没填或填了空）
    if not raw.get("tools_used"):
        raw["tools_used"] = [step["name"] for step in out.get("tool_trace", [])]
    try:
        return parse_proposal(raw)
    except ParseError as e:
        logger.warning("P-LLM Proposal 解析失败 (%s)，fallback", e)
        return _fallback_propose(memory, current_config)
