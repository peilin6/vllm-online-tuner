#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizer.py — B 类算法工具（纯函数）

Task 6.7。所有工具签名都是 (memory, **kwargs) -> dict，便于由 ToolRegistry 统一
暴露给 LLM。不依赖外部 IO，只读 memory 即可计算结果。

工具：
1. bo_suggest         基于已观测 trial 用 Optuna 提议下一组 RESTART 参数
2. param_sensitivity  逐参数 Spearman / 极差 灵敏度估计
3. pareto_front       (throughput, latency) 二维 Pareto 前沿
4. local_grid         在指定参数附近生成局部网格候选
5. cluster_workload_phases  对每条 trial 的 (rate, in_len, out_len) 做简单 KMeans 分群
"""
from __future__ import annotations

import math
from typing import Any

from .memory import ExperienceMemory, TrialRecord
from . import param_registry
from .tools import ToolSpec


# =====================================================================
# 内部工具函数
# =====================================================================

def _successful(records: list[TrialRecord]) -> list[TrialRecord]:
    return [r for r in records if r.metrics.success and not r.metrics.early_killed]


def _spearman(xs: list[float], ys: list[float]) -> float:
    """无依赖 Spearman 等级相关系数。"""
    n = len(xs)
    if n < 2:
        return 0.0

    def _rank(arr: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: arr[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based 平均秩
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = math.sqrt(sum((v - mx) ** 2 for v in rx))
    den_y = math.sqrt(sum((v - my) ** 2 for v in ry))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


# =====================================================================
# B 类工具实现
# =====================================================================

def _tool_bo_suggest(
    memory: ExperienceMemory, *,
    target: str = "throughput_tok_per_s",
    direction: str = "maximize",
    params: list[str] | None = None,
    n_warmup: int = 3,
) -> dict:
    """用 Optuna TPE 在已观测 trial 上拟合代理模型并提议下一组参数。

    若已成功 trial 数 < n_warmup，则降级为返回各参数的中位候选作为冷启动。
    """
    succ = _successful(memory.all())
    if not params:
        params = ["max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization"]
    specs = [param_registry.get_spec(p) for p in params]

    # 冷启动：返回每个参数候选集中位
    if len(succ) < n_warmup:
        suggestion = {}
        for s in specs:
            if s.candidates:
                suggestion[s.name] = s.candidates[len(s.candidates) // 2]
            elif s.range is not None:
                lo, hi = s.range
                v = (lo + hi) / 2.0
                suggestion[s.name] = int(v) if s.type == "int" else v
            else:
                suggestion[s.name] = s.default
        return {"ok": True, "suggestion": suggestion, "mode": "cold_start",
                "n_observed": len(succ)}

    try:
        import optuna  # type: ignore
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return {"ok": False, "error": "optuna 未安装；运行 pip install optuna"}

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction=direction, sampler=sampler)

    # 把已观测点用 add_trial 注入
    for rec in succ:
        ps = {}
        skip = False
        for s in specs:
            v = rec.config.get(s.name)
            if v is None:
                skip = True
                break
            ps[s.name] = v
        if skip:
            continue
        try:
            distros = {}
            for s in specs:
                if s.candidates:
                    distros[s.name] = optuna.distributions.CategoricalDistribution(
                        list(s.candidates))
                elif s.range is not None:
                    lo, hi = s.range
                    if s.type == "int":
                        distros[s.name] = optuna.distributions.IntDistribution(int(lo), int(hi))
                    else:
                        distros[s.name] = optuna.distributions.FloatDistribution(float(lo), float(hi))
                else:
                    continue
            value = float(getattr(rec.metrics, target, 0.0))
            study.add_trial(optuna.trial.create_trial(
                params=ps, distributions=distros, value=value,
            ))
        except (ValueError, TypeError):
            continue

    if len(study.trials) == 0:
        return {"ok": False, "error": "无有效观测点可注入"}

    trial = study.ask()
    suggestion = {}
    for s in specs:
        if s.candidates:
            suggestion[s.name] = trial.suggest_categorical(s.name, list(s.candidates))
        elif s.range is not None:
            lo, hi = s.range
            if s.type == "int":
                suggestion[s.name] = trial.suggest_int(s.name, int(lo), int(hi))
            else:
                suggestion[s.name] = trial.suggest_float(s.name, float(lo), float(hi))
    return {"ok": True, "suggestion": suggestion, "mode": "tpe",
            "n_observed": len(succ)}


def _tool_param_sensitivity(
    memory: ExperienceMemory, *,
    target: str = "throughput_tok_per_s",
    params: list[str] | None = None,
) -> dict:
    succ = _successful(memory.all())
    if len(succ) < 3:
        return {"ok": False, "error": f"成功 trial 数不足 (n={len(succ)})，至少 3 条"}
    if not params:
        params = list({k for r in succ for k in r.config.keys()})
    sens: dict[str, dict] = {}
    for p in params:
        xs, ys = [], []
        for r in succ:
            v = r.config.get(p)
            y = getattr(r.metrics, target, None)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and y is not None:
                xs.append(float(v))
                ys.append(float(y))
        if len(xs) < 3:
            sens[p] = {"n": len(xs), "spearman": None, "y_range": None}
            continue
        sens[p] = {
            "n": len(xs),
            "spearman": round(_spearman(xs, ys), 4),
            "y_range": round(max(ys) - min(ys), 3),
        }
    ranked = sorted(
        [(k, v) for k, v in sens.items() if v.get("spearman") is not None],
        key=lambda kv: abs(kv[1]["spearman"]), reverse=True,
    )
    return {"ok": True, "target": target,
            "sensitivity": sens,
            "ranking": [k for k, _ in ranked]}


def _tool_pareto_front(
    memory: ExperienceMemory, *,
    obj_max: str = "throughput_tok_per_s",
    obj_min: str = "latency_p95_ms",
) -> dict:
    succ = _successful(memory.all())
    pts = []
    for r in succ:
        x = getattr(r.metrics, obj_max, None)
        y = getattr(r.metrics, obj_min, None)
        if x is None or y is None or x < 0 or y < 0:
            continue
        pts.append((r.trial_id, float(x), float(y)))
    front = []
    for tid, x, y in pts:
        dominated = False
        for tid2, x2, y2 in pts:
            if tid == tid2:
                continue
            # tid2 严格支配 tid: x2>=x, y2<=y, 且至少一个严格
            if x2 >= x and y2 <= y and (x2 > x or y2 < y):
                dominated = True
                break
        if not dominated:
            front.append({"trial_id": tid, obj_max: x, obj_min: y})
    front.sort(key=lambda d: d[obj_max], reverse=True)
    return {"ok": True, "obj_max": obj_max, "obj_min": obj_min,
            "front": front, "n_total": len(pts)}


def _tool_local_grid(
    memory: ExperienceMemory, *,
    around: dict,
    radius: int = 1,
    params: list[str] | None = None,
) -> dict:
    """在 around 指定的中心点附近，对每个参数取 candidates 中相邻 ±radius 个候选，
    生成完整笛卡尔积候选集（去掉中心点）。"""
    if not params:
        params = list(around.keys())
    options: dict[str, list] = {}
    for p in params:
        spec = param_registry.get_spec(p)
        cands = list(spec.candidates)
        if not cands or around.get(p) not in cands:
            options[p] = [around.get(p, spec.default)]
            continue
        idx = cands.index(around[p])
        lo = max(0, idx - radius)
        hi = min(len(cands), idx + radius + 1)
        options[p] = cands[lo:hi]
    # 笛卡尔积
    keys = list(options.keys())
    grid: list[dict] = [{}]
    for k in keys:
        grid = [{**g, k: v} for g in grid for v in options[k]]
    grid = [g for g in grid if g != around]
    return {"ok": True, "around": around, "radius": radius,
            "n": len(grid), "grid": grid}


def _tool_cluster_workload_phases(
    memory: ExperienceMemory, *,
    k: int = 3,
    feature_keys: list[str] | None = None,
) -> dict:
    """对历史 trial 的 workload 特征做简单 1-D 投影聚类（不依赖 sklearn）。

    feature_keys 默认从 record.config 的 'workload.*' 字段提取；若没有则回退到
    每条 trial 的 (throughput_tok_per_s, ttft_p95_ms) 特征做演示性聚类。
    """
    recs = _successful(memory.all())
    if len(recs) < k:
        return {"ok": False, "error": f"trial 数 ({len(recs)}) 少于 k={k}"}

    feats = []
    for r in recs:
        # 优先使用 config 里 workload.* 字段
        if any(key.startswith("workload.") for key in r.config):
            row = [float(r.config.get(key, 0.0)) for key in (feature_keys or [])]
            if not row:
                row = [float(v) for v in r.config.values()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)]
        else:
            row = [r.metrics.throughput_tok_per_s, r.metrics.ttft_p95_ms]
        feats.append(row)

    # 朴素 K-Means（欧式 + 随机初始化用前 k 个点）
    centroids = [list(feats[i]) for i in range(k)]
    labels = [0] * len(feats)
    for _ in range(20):
        # assign
        for i, x in enumerate(feats):
            best, best_d = 0, float("inf")
            for c, ctr in enumerate(centroids):
                d = sum((a - b) ** 2 for a, b in zip(x, ctr))
                if d < best_d:
                    best, best_d = c, d
            labels[i] = best
        # update
        new_centroids = []
        for c in range(k):
            members = [feats[i] for i in range(len(feats)) if labels[i] == c]
            if not members:
                new_centroids.append(centroids[c])
                continue
            dim = len(members[0])
            new_centroids.append([
                sum(m[d] for m in members) / len(members) for d in range(dim)
            ])
        if new_centroids == centroids:
            break
        centroids = new_centroids
    clusters: dict[int, list[str]] = {c: [] for c in range(k)}
    for i, lab in enumerate(labels):
        clusters[lab].append(recs[i].trial_id)
    return {
        "ok": True, "k": k,
        "centroids": [[round(v, 3) for v in ctr] for ctr in centroids],
        "clusters": {str(c): tids for c, tids in clusters.items()},
    }


# =====================================================================
# 导出 ToolSpec 列表（供 ToolRegistry(extra_tools=...) 注入）
# =====================================================================
B_TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="bo_suggest",
        description="基于已观测 trial 用 Optuna TPE 提议下一组 RESTART 参数；冷启动时返回中位候选。",
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "default": "throughput_tok_per_s"},
                "direction": {"type": "string", "enum": ["maximize", "minimize"],
                              "default": "maximize"},
                "params": {"type": "array", "items": {"type": "string"}},
                "n_warmup": {"type": "integer", "minimum": 1, "default": 3},
            },
        },
        handler=_tool_bo_suggest,
    ),
    ToolSpec(
        name="param_sensitivity",
        description="逐参数计算 Spearman 相关与极差，给出对 target 影响的排序。",
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "default": "throughput_tok_per_s"},
                "params": {"type": "array", "items": {"type": "string"}},
            },
        },
        handler=_tool_param_sensitivity,
    ),
    ToolSpec(
        name="pareto_front",
        description="基于已观测 trial 计算 (吞吐 max, 延迟 min) 的 Pareto 前沿。",
        parameters={
            "type": "object",
            "properties": {
                "obj_max": {"type": "string", "default": "throughput_tok_per_s"},
                "obj_min": {"type": "string", "default": "latency_p95_ms"},
            },
        },
        handler=_tool_pareto_front,
    ),
    ToolSpec(
        name="local_grid",
        description="围绕指定中心点在 candidates 中取 ±radius 邻域，生成局部网格候选。",
        parameters={
            "type": "object",
            "properties": {
                "around": {"type": "object",
                           "description": "中心点 {param: value} 配置"},
                "radius": {"type": "integer", "minimum": 1, "default": 1},
                "params": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["around"],
        },
        handler=_tool_local_grid,
    ),
    ToolSpec(
        name="cluster_workload_phases",
        description="对 memory 中 trial 的 workload 特征做 K-Means 聚类。",
        parameters={
            "type": "object",
            "properties": {
                "k": {"type": "integer", "minimum": 2, "default": 3},
                "feature_keys": {"type": "array", "items": {"type": "string"}},
            },
        },
        handler=_tool_cluster_workload_phases,
    ),
]
