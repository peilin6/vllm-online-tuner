#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""proposer.py — P-LLM Proposer（带工具调用 + Playbook 规则约束）

Task 7.2 + Playbook 强化版。多轮 function-calling，LLM 必须按 bottleneck-playbook
给出的 allowed_params 与 direction 来调参；越权或方向不符会被外层校验拒绝并降级到
规则版（按 playbook.allowed_params 优先级 + direction 选未试过的 candidate）。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from tuner import param_registry
from tuner.memory import ExperienceMemory

from .playbook import PlaybookEntry, get_entry, render_for_prompt
from .prompts import P_LLM_SYSTEM, P_LLM_USER_TMPL
from .schemas import ConfigDelta, ParseError, StopSignal, parse_proposal

logger = logging.getLogger(__name__)


# 全局兜底优先级（仅当 diagnosis 缺失或 playbook 白名单为空时使用）
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


def _recent_rejected_pairs(memory: ExperienceMemory, n: int = 3) -> list[tuple]:
    """最近 n 条 rejected_proposals 的 (param, new_value) 对。

    memory.rejected_proposals 存的是 (config_dict, reason) 元组，
    其中 config_dict 通常形如 {param: new_value}（VtaAgent 拒绝时这么记），
    也可能是完整 next_cfg；这里取所有 (k, v) 全部当作禁忌对。
    """
    out: list[tuple] = []
    for rec in list(memory.rejected_proposals)[-n:]:
        if isinstance(rec, tuple) and len(rec) >= 1 and isinstance(rec[0], dict):
            for k, v in rec[0].items():
                out.append((k, v))
        elif isinstance(rec, dict):    # 兼容旧/外部写入格式
            p = rec.get("param")
            v = rec.get("new_value")
            if p is not None:
                out.append((p, v))
    return out


def _direction_ok(old: Any, new: Any, direction: str) -> bool:
    """检查 (old, new) 是否符合 direction 要求。"""
    if direction == "toggle":
        return new != old
    if isinstance(old, (int, float)) and isinstance(new, (int, float)) \
            and not isinstance(old, bool) and not isinstance(new, bool):
        if direction == "up":
            return new > old
        if direction == "down":
            return new < old
    # 非数值（bool/str）一律按 toggle
    return new != old


def _validate_against_playbook(
    delta: ConfigDelta, entry: PlaybookEntry, current_config: dict,
    memory: ExperienceMemory,
) -> str | None:
    """返回 None 表示通过；否则返回拒绝原因（字符串）。"""
    if not entry.allowed_params:
        return f"playbook 不允许调参（bottleneck={entry.bottleneck}）"
    if delta.param not in entry.allowed_params:
        return (f"参数 {delta.param} 不在 allowed_params={list(entry.allowed_params)} 内")
    # 候选合法性
    try:
        spec = param_registry.get_spec(delta.param)
    except KeyError:
        return f"未登记参数: {delta.param}"
    if not spec.in_range(delta.new_value):
        return f"new_value={delta.new_value} 不在 candidates/range 内"
    # 方向校验
    direction = entry.direction.get(delta.param)
    old = current_config.get(delta.param, spec.default)
    if direction and not _direction_ok(old, delta.new_value, direction):
        return f"方向不符 playbook.direction[{delta.param}]={direction} (old={old} -> new={delta.new_value})"
    # 不得复用最近 rejected
    if (delta.param, delta.new_value) in _recent_rejected_pairs(memory, n=3):
        return f"({delta.param}={delta.new_value}) 命中 recent_rejected"
    return None


# -------------------- Fallback：bottleneck 感知 --------------------
def _pick_fallback_value(
    name: str, current_config: dict, memory: ExperienceMemory,
    direction: str | None,
) -> Any | None:
    """从 candidates 中按 direction 选一个未试过、与当前不同的值。"""
    try:
        spec = param_registry.get_spec(name)
    except KeyError:
        return None
    cands = list(spec.candidates)
    if not cands:
        return None
    tried = _tried_values(memory, name)
    old = current_config.get(name, spec.default)
    rejected = {v for (p, v) in _recent_rejected_pairs(memory, n=3) if p == name}

    def candidate_pool() -> list:
        if direction == "up" and isinstance(old, (int, float)) and not isinstance(old, bool):
            pool = [c for c in cands if isinstance(c, (int, float)) and c > old]
        elif direction == "down" and isinstance(old, (int, float)) and not isinstance(old, bool):
            pool = [c for c in cands if isinstance(c, (int, float)) and c < old]
        else:                       # toggle / 非数值 / 无 direction
            pool = [c for c in cands if c != old]
        return pool

    pool = candidate_pool()
    untried = [c for c in pool if c not in tried and c not in rejected]
    if untried:
        # 选离 old 最近的一档（小步走）
        if isinstance(old, (int, float)) and not isinstance(old, bool):
            untried.sort(key=lambda v: abs(cands.index(v) - cands.index(old))
                         if v in cands else 999)
        return untried[0]
    return None


def _fallback_propose(
    memory: ExperienceMemory, current_config: dict,
    diagnosis=None,
) -> ConfigDelta | StopSignal:
    """规则版 fallback：优先按 diagnosis 的 playbook 选参数与方向。

    - 有 diagnosis -> 从 entry.allowed_params 按出现顺序找第一个能产出新值的参数。
    - 无 diagnosis 或 playbook 为空 -> 退回到全局 FALLBACK_PRIORITY。
    """
    entry: PlaybookEntry | None = None
    if diagnosis is not None:
        entry = get_entry(getattr(diagnosis, "bottleneck", "underutilized"))
        if not entry.allowed_params:
            return StopSignal(
                reason=f"playbook[{entry.bottleneck}] 不允许再调参",
                source="fallback",
            )

    candidate_params = (list(entry.allowed_params) if entry else FALLBACK_PRIORITY)

    for name in candidate_params:
        direction = entry.direction.get(name) if entry else None
        new_value = _pick_fallback_value(name, current_config, memory, direction)
        if new_value is None:
            continue
        try:
            spec = param_registry.get_spec(name)
        except KeyError:
            continue
        old_value = current_config.get(name, spec.default)
        if new_value == old_value:
            continue
        bn = entry.bottleneck if entry else "n/a"
        return ConfigDelta(
            param=name,
            old_value=old_value,
            new_value=new_value,
            hypothesis_ref=f"fallback: playbook[{bn}].{name}",
            tools_used=[],
            reason=(f"fallback：bottleneck={bn}, direction={direction or 'any'}, "
                    f"未试过 {name}={new_value}"),
            expected_effect={"throughput_tok_per_s": "?", "risk": "未知"},
            rollback_if="throughput 不升或 SLO 违反",
            source="fallback",
        )

    return StopSignal(
        reason=f"fallback: 当前 bottleneck={entry.bottleneck if entry else 'n/a'} "
               f"的允许参数候选均已试过",
        source="fallback",
    )


def _build_user_prompt(
    diagnosis_dict: dict, current_config: dict, memory: ExperienceMemory,
) -> str:
    bottleneck = diagnosis_dict.get("bottleneck", "underutilized")
    return P_LLM_USER_TMPL.format(
        diagnosis_json=json.dumps(diagnosis_dict, ensure_ascii=False),
        playbook_block=render_for_prompt(bottleneck),
        current_config_json=json.dumps(current_config, ensure_ascii=False),
        memory_summary=json.dumps(memory.summarize(top_k=3, recent_n=5),
                                  ensure_ascii=False),
        rejected_recent=json.dumps(list(memory.rejected_proposals)[-3:],
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
    """主入口。LLM 失败 / 解析失败 / playbook 校验失败时自动降级。"""
    if getattr(diagnosis, "should_stop", False):
        return StopSignal(reason=f"diagnosis.should_stop=True: {diagnosis.bottleneck}",
                          source=getattr(diagnosis, "source", "llm"))

    entry = get_entry(getattr(diagnosis, "bottleneck", "underutilized"))
    if not entry.allowed_params:
        return StopSignal(
            reason=f"playbook[{entry.bottleneck}] 不允许再调参（已收敛）",
            source=getattr(diagnosis, "source", "llm"),
        )

    if client is None or tools is None or not use_llm:
        return _fallback_propose(memory, current_config, diagnosis=diagnosis)

    user = _build_user_prompt(diagnosis.to_dict(), current_config, memory)
    messages = [
        {"role": "system", "content": P_LLM_SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        out = client.chat_with_tools(messages, tools, max_rounds=max_rounds)
    except RuntimeError as e:
        logger.warning("P-LLM 调用失败，fallback: %s", e)
        return _fallback_propose(memory, current_config, diagnosis=diagnosis)

    content = _strip_fence(_final_content(out.get("final")))
    raw: dict | None = None
    if content:
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning("P-LLM 首轮 JSON 解析失败 (%s)，尝试 nudge 补轮", e)
            raw = None
    if raw is None:
        # nudge: 让模型基于已有 tool 历史，输出最终 JSON（不带 tools，用 json_object 收口）
        nudge_msgs = list(out.get("messages") or messages)
        nudge_msgs.append({
            "role": "user",
            "content": "请基于以上工具调用结果，仅输出最终 proposal JSON 对象（严格匹配 schema，"
                       "不要 markdown、不要解释）。",
        })
        try:
            resp2 = client.chat(
                nudge_msgs,
                response_format={"type": "json_object"},
                use_cache=False,
            )
            ch = (resp2.get("choices") or [{}])[0].get("message") or {}
            content2 = _strip_fence(ch.get("content") or "")
            raw = json.loads(content2) if content2 else None
        except (RuntimeError, json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.warning("P-LLM nudge 也失败 (%s)，fallback", e)
            raw = None
    if not isinstance(raw, dict):
        return _fallback_propose(memory, current_config, diagnosis=diagnosis)
    if not raw.get("tools_used"):
        raw["tools_used"] = [step["name"] for step in out.get("tool_trace", [])]
    try:
        proposal = parse_proposal(raw)
    except ParseError as e:
        logger.warning("P-LLM Proposal 解析失败 (%s)，fallback", e)
        return _fallback_propose(memory, current_config, diagnosis=diagnosis)

    if isinstance(proposal, StopSignal):
        return proposal

    # Playbook 后置校验
    err = _validate_against_playbook(proposal, entry, current_config, memory)
    if err is not None:
        logger.warning("P-LLM 输出违反 playbook (%s)，fallback", err)
        memory.record_rejected(
            {proposal.param: proposal.new_value},
            f"playbook_violation: {err}",
        )
        return _fallback_propose(memory, current_config, diagnosis=diagnosis)

    return proposal
