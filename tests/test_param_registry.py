#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_param_registry.py — Task 6.5 单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.param_registry import (
    ParamSpec, all_specs, get_spec, names, validate_overrides,
)


def test_nine_restart_params_registered():
    assert len(all_specs()) == 9
    expected = {
        "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
        "block_size", "enable_chunked_prefill", "enable_prefix_caching",
        "swap_space", "cuda_graph_sizes", "tensor_parallel_size",
    }
    assert set(names()) == expected


def test_each_spec_has_required_fields():
    for s in all_specs():
        assert s.dotted_path.startswith("server.")
        assert s.type in {"int", "float", "bool", "str"}
        assert s.requires_restart is True
        assert s.notes  # 非空


def test_get_spec_unknown_raises():
    with pytest.raises(KeyError):
        get_spec("nonexistent_param")


# ---------- in_range ----------

def test_in_range_int_candidates():
    s = get_spec("max_num_seqs")
    assert s.in_range(64) is True
    assert s.in_range(33) is True   # 在 range 内即使不在 candidates
    assert s.in_range(99999) is False


def test_in_range_bool():
    s = get_spec("enable_chunked_prefill")
    assert s.in_range(True) is True
    assert s.in_range(False) is True


def test_in_range_float_within_bounds():
    s = get_spec("gpu_memory_utilization")
    assert s.in_range(0.85) is True
    assert s.in_range(0.99) is False
    assert s.in_range(0.40) is False


def test_in_range_str_candidates():
    s = get_spec("cuda_graph_sizes")
    assert s.in_range("default") is True
    assert s.in_range("nonsense") is False  # 离散候选不命中


# ---------- clamp ----------

def test_clamp_to_nearest_candidate():
    s = get_spec("max_num_seqs")
    assert s.clamp(70) == 64       # 64 与 96 中更近
    assert s.clamp(150) == 128     # 128 与 192 中更近


def test_clamp_within_range_stays():
    s = get_spec("gpu_memory_utilization")
    # 0.91 不在 candidates 但在 range → 走候选最近邻 = 0.92 (距 0.01) vs 0.90 (0.01) 取一
    out = s.clamp(0.91)
    assert out in {0.90, 0.92}


def test_clamp_out_of_range_int_clamps():
    # 用一个有 range 但无 candidates 的 spec 测 numeric clamp 行为
    spec = ParamSpec(name="x", dotted_path="server.x", type="int",
                     default=10, range=(0, 100))
    assert spec.clamp(150) == 100
    assert spec.clamp(-5) == 0
    assert spec.clamp(50) == 50


def test_clamp_out_of_range_float():
    spec = ParamSpec(name="y", dotted_path="server.y", type="float",
                     default=0.5, range=(0.0, 1.0))
    out = spec.clamp(1.5)
    assert out == pytest.approx(1.0)
    assert isinstance(out, float)


# ---------- validate_overrides ----------

def test_validate_overrides_all_valid():
    ok, errs = validate_overrides({
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.90,
        "enable_chunked_prefill": True,
    })
    assert ok is True
    assert errs == []


def test_validate_overrides_unknown_key():
    ok, errs = validate_overrides({"foo_bar": 1})
    assert ok is False
    assert any("foo_bar" in e for e in errs)


def test_validate_overrides_out_of_range():
    ok, errs = validate_overrides({"gpu_memory_utilization": 1.5})
    assert ok is False
    assert any("gpu_memory_utilization" in e for e in errs)


def test_validate_overrides_mixed():
    ok, errs = validate_overrides({
        "max_num_seqs": 64,                      # ok
        "gpu_memory_utilization": 0.99,          # 超界
        "garbage": 1,                            # 未登记
    })
    assert ok is False
    assert len(errs) == 2
