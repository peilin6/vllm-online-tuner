#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_generator.py — 把 baseline 配置 + overrides 渲染为临时实验配置

Task 6.1 配套模块。Agent 每提出一次 ConfigDelta，就调一次 render + write_temp_config，
把结果写到 /tmp/vllm_trial_<uuid>.json，再交给 VllmLauncher.restart()。
"""
import copy
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any


def _set_by_dotted_path(d: dict, dotted: str, value: Any) -> None:
    """按点路径写入，例如 'server.max_num_seqs' -> d['server']['max_num_seqs']=value"""
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def render_experiment_config(base: dict, overrides: dict) -> dict:
    """深拷贝 base，按 overrides 的 key（点路径或顶层 vLLM engine arg）覆写。

    overrides 支持两种 key:
    - 点路径: "server.max_num_seqs" -> base["server"]["max_num_seqs"]
    - 扁平 vLLM engine arg（如 'max_num_seqs', 'enable_prefix_caching'）-> 自动写到 base["server"][...]
      （server 段是 launcher 真正读取的字段集）
    """
    if not isinstance(base, dict):
        raise TypeError("base config must be dict")
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError("overrides must be dict or None")

    result = copy.deepcopy(base)
    if not overrides:
        return result

    for key, val in overrides.items():
        if "." in key:
            _set_by_dotted_path(result, key, val)
        else:
            # 扁平 key 自动归到 server 段（vLLM engine args 都属于 server）
            result.setdefault("server", {})
            result["server"][key] = val
    return result


def write_temp_config(config: dict, *, tmp_dir: str | None = None) -> Path:
    """落盘到临时文件，返回路径。tmp_dir 为 None 时使用系统 tempdir。"""
    tmp = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / f"vllm_trial_{uuid.uuid4().hex[:12]}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return path
