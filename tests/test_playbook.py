#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_playbook.py — 瓶颈→参数白名单+方向 规则表测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_advisor.playbook import (
    BOTTLENECK_PLAYBOOK, get_entry, render_for_prompt,
)
from llm_advisor.schemas import BOTTLENECK_ENUM
from tuner import param_registry


def test_playbook_covers_all_bottlenecks():
    assert set(BOTTLENECK_PLAYBOOK.keys()) == BOTTLENECK_ENUM


def test_all_allowed_params_are_registered():
    for entry in BOTTLENECK_PLAYBOOK.values():
        for p in entry.allowed_params:
            param_registry.get_spec(p)              # 不抛 KeyError 即通过


def test_all_directions_keys_subset_of_allowed():
    for entry in BOTTLENECK_PLAYBOOK.values():
        assert set(entry.direction.keys()).issubset(set(entry.allowed_params))


def test_directions_use_legal_tokens():
    legal = {"up", "down", "toggle"}
    for entry in BOTTLENECK_PLAYBOOK.values():
        for d in entry.direction.values():
            assert d in legal


@pytest.mark.parametrize(
    "bottleneck, must_have_param, must_have_direction",
    [
        # 与 rule.txt 表 3.x 对齐的硬性方向断言
        ("kv_cache_pressure", "gpu_memory_utilization", "up"),
        ("kv_cache_pressure", "max_num_seqs", "down"),
        ("preempt_storm", "gpu_memory_utilization", "up"),
        ("preempt_storm", "max_num_seqs", "down"),
        ("prefill_bound", "max_num_batched_tokens", "up"),
        ("decode_bound", "max_num_batched_tokens", "down"),
        ("queue_backlog", "max_num_seqs", "up"),
        ("underutilized", "max_num_batched_tokens", "up"),
        ("slo_margin_low", "max_num_batched_tokens", "down"),
    ],
)
def test_canonical_directions_match_vllm_doc(bottleneck, must_have_param, must_have_direction):
    e = get_entry(bottleneck)
    assert must_have_param in e.allowed_params
    assert e.direction[must_have_param] == must_have_direction


def test_converged_disallows_all_changes():
    e = get_entry("converged")
    assert e.allowed_params == ()
    block = render_for_prompt("converged")
    assert "stop" in block.lower() or "停止" in block or "stop" in block


def test_render_for_prompt_includes_whitelist_and_directions():
    block = render_for_prompt("kv_cache_pressure")
    assert "gpu_memory_utilization" in block
    assert "↑" in block or "增大" in block


def test_unknown_bottleneck_falls_back_to_underutilized():
    e = get_entry("not_a_real_one")
    assert e.bottleneck == "underutilized"
