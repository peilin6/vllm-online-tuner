#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory.py — ExperienceMemory：记录每次 trial 的 config + metrics + 备注

Task 6.4。提供给 A/P/R-LLM 紧凑历史视图（top_k by score + recent_n），并支持
JSONL 持久化（results/<run_id>/memory.jsonl）。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .metrics_parser import TrialMetrics


@dataclass
class TrialRecord:
    """一次 trial 的完整存档：配置 + 指标 + 时间戳 + 备注 + 来源工具。"""
    trial_id: str
    config: dict
    metrics: TrialMetrics
    timestamp: float = field(default_factory=lambda: time.time())
    source: str = "agent"          # agent / baseline / random / bo / grid ...
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "metrics": self.metrics.to_dict() if isinstance(self.metrics, TrialMetrics)
                       else dict(self.metrics),
            "timestamp": self.timestamp,
            "source": self.source,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrialRecord":
        m = d.get("metrics", {}) or {}
        # 仅取 TrialMetrics 已声明的字段，忽略未来扩展导致的多余键
        valid = {k: m[k] for k in TrialMetrics.__dataclass_fields__ if k in m}
        return cls(
            trial_id=d["trial_id"],
            config=dict(d.get("config") or {}),
            metrics=TrialMetrics(**valid),
            timestamp=float(d.get("timestamp", 0.0)),
            source=str(d.get("source", "agent")),
            notes=list(d.get("notes") or []),
        )


def _score(rec: TrialRecord) -> float:
    """打分：成功 trial 用 token 吞吐；失败/早停打负分用于排序兜底。"""
    m = rec.metrics
    if not m.success or m.early_killed:
        return -1.0
    if m.throughput_tok_per_s < 0:
        return -1.0
    return float(m.throughput_tok_per_s)


class ExperienceMemory:
    """轻量内存 + JSONL 落盘。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._records: list[TrialRecord] = []
        # Week 7 扩展：R-LLM 长期笔记 + Judge 拒绝过的 (config, reason)
        self.notes: list[str] = []
        self.rejected_proposals: list[tuple[dict, str]] = []
        if self.path and self.path.exists():
            self.load()

    # ---------- mutation ----------
    def add(self, record: TrialRecord) -> None:
        self._records.append(record)
        if self.path is not None:
            self._append_jsonl(record)

    # Week 7 别名（VtaAgent 主循环用 append 语义更自然）
    append = add

    def add_note(self, note: str) -> None:
        if note and note.strip():
            self.notes.append(note.strip())

    def record_rejected(self, config: dict, reason: str) -> None:
        self.rejected_proposals.append((dict(config), str(reason)))

    def recent(self, n: int = 1) -> list[TrialRecord]:
        """recent_n 的别名，与 prompt 文档措辞一致。"""
        return self.recent_n(n)

    def __len__(self) -> int:
        return len(self._records)

    def all(self) -> list[TrialRecord]:
        return list(self._records)

    # ---------- queries ----------
    def top_k(self, k: int = 3) -> list[TrialRecord]:
        """按 token 吞吐降序取前 k；失败/早停的 trial 自然排在最后。"""
        return sorted(self._records, key=_score, reverse=True)[: max(0, int(k))]

    def recent_n(self, n: int = 3) -> list[TrialRecord]:
        return list(self._records[-max(0, int(n)):])

    def best(self) -> TrialRecord | None:
        if not self._records:
            return None
        cand = max(self._records, key=_score)
        return cand if _score(cand) > 0 else None

    def summarize(self, top_k: int = 3, recent_n: int = 3) -> dict:
        """生成给 LLM 的紧凑视图：只保留参数 + 关键指标 + 备注。"""
        def _view(rec: TrialRecord) -> dict:
            m = rec.metrics
            return {
                "trial_id": rec.trial_id,
                "source": rec.source,
                "config": rec.config,
                "throughput_tok_per_s": round(m.throughput_tok_per_s, 2),
                "ttft_p95_ms": round(m.ttft_p95_ms, 1),
                "tpot_p95_ms": round(m.tpot_p95_ms, 1),
                "preemptions_total": int(m.preemptions_total),
                "kv_p95_pct": round(m.kv_cache_usage_p95_pct, 3),
                "success": bool(m.success),
                "early_killed": bool(m.early_killed),
                "notes": rec.notes[-2:],
            }

        best = self.best()
        return {
            "n_trials": len(self._records),
            "best": _view(best) if best is not None else None,
            "top_k": [_view(r) for r in self.top_k(top_k)],
            "recent": [_view(r) for r in self.recent_n(recent_n)],
            "notes": list(self.notes[-5:]),
            "rejected_recent": [
                {"config": cfg, "reason": rsn}
                for cfg, rsn in self.rejected_proposals[-3:]
            ],
        }

    def dump_compact(self) -> dict:
        """Reporter 用的全量紧凑视图：所有 trial + 所有笔记。"""
        return {
            "n_trials": len(self._records),
            "trials": [
                {
                    "trial_id": r.trial_id,
                    "source": r.source,
                    "config": r.config,
                    "metrics": {
                        "throughput_tok_per_s": round(r.metrics.throughput_tok_per_s, 2),
                        "ttft_p95_ms": round(r.metrics.ttft_p95_ms, 1),
                        "tpot_p95_ms": round(r.metrics.tpot_p95_ms, 1),
                        "latency_p95_ms": round(r.metrics.latency_p95_ms, 1),
                        "preemptions_total": int(r.metrics.preemptions_total),
                        "kv_p95_pct": round(r.metrics.kv_cache_usage_p95_pct, 3),
                        "success": bool(r.metrics.success),
                        "early_killed": bool(r.metrics.early_killed),
                    },
                }
                for r in self._records
            ],
            "notes": list(self.notes),
        }

    # ---------- persistence ----------
    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("save() 需要 path 参数或在构造时传入 path")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        return target

    def load(self, path: str | Path | None = None) -> int:
        target = Path(path) if path else self.path
        if target is None or not target.exists():
            return 0
        self._records = []
        with target.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._records.append(TrialRecord.from_dict(json.loads(line)))
        return len(self._records)

    def _append_jsonl(self, rec: TrialRecord) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
