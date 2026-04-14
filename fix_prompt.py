#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix corrupted sections in week3-8_execution.prompt.md"""

import os

filepath = r"d:\vlllm\prompts\week3-8_execution.prompt.md"

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines are 0-indexed in the list, 1-indexed in the file
# Clean content: lines 0-2121 (file lines 1-2122, up to end of Task 7.3 + "---")
# Corrupted: lines 2122-2629 (file lines 2123-2630, Tasks 7.4-8.5) 
# Clean tail: lines 2630-end (file lines 2631+, §11-§13) but §11 header is garbled

# Keep lines before corruption (up to and including the "---" after Task 7.3)
head = lines[:2122]  # lines 1-2122

# Keep lines after corruption (§11 onwards)
tail = lines[2630:]  # lines 2631+

# Fix the §11 header in tail (first line should be "## 十一、最终交付清单（约 50 个）")
# Current: "## 十一、最终交付50个）"
for i, line in enumerate(tail):
    if '## 十一、最终交付50个）' in line:
        tail[i] = '## 十一、最终交付清单（约 50 个）\n'
        break

# Also fix duplicate lines in tail for scripts section
# Find and fix "├── run_comparison.sh\n├── run_ablation.sh\n└── test_analyzer_integration.sh"
# that appears duplicated

# Clean replacement for Tasks 7.4 through 8.5
clean_section = r"""### Task 7.4: BaseController + FixedController

**目标**：定义 Controller 抽象接口和最简单的对照实现。

**操作**：
1. 新建 `controllers/` 目录
2. 新建 `controllers/__init__.py`
3. 新建 `controllers/base.py`：

```python
"""
BaseController — Controller 抽象基类

所有 controller 必须实现 decide() 方法，返回统一格式的决策。
"""
from abc import ABC, abstractmethod


class BaseController(ABC):
    @abstractmethod
    async def decide(self, analyzer_state: dict) -> dict:
        """
        输入: Analyzer 输出的完整状态
        输出:
        {
            "controller_type": str,
            "policy_mode": "conservative" | "balanced" | "aggressive" | "fixed",
            "action_type": "fast_only" | "slow_only" | "both" | "none",
            "fast_action": {"batching_window_ms": int, "max_concurrency": int, "admission_threshold": float} | null,
            "slow_action": {"target_profile": str} | null,
            "reasoning": str,
            "confidence": float,
            "llm_source": str | null,
            "decision_timestamp": float
        }
        """
        pass
```

4. 新建 `controllers/fixed_controller.py`：

```python
class FixedController(BaseController):
    """固定配置 controller，作为实验对照基线"""

    def __init__(self, config: dict):
        """config 包含固定的 fast_action 参数"""

    async def decide(self, analyzer_state: dict) -> dict:
        """始终返回初始配置，不变"""
```

5. 定义 `controller_decisions.jsonl` 日志格式：

```json
{
  "timestamp": 1234567890.123,
  "decision_cycle": 1,
  "controller_type": "llm_strategy",
  "analyzer_state_summary": {"load_level": "high", "bottleneck_type": "decode", "risk_level": "warning"},
  "decision": {"policy_mode": "conservative", "action_type": "fast_only", "fast_action": {...}},
  "execution_result": {"success": true, "path": "fast", "duration_ms": 15},
  "llm_stats": {"latency_ms": 1200, "source": "deepseek"}
}
```

6. 新建 `configs/controllers/controller_fixed.json`:
```json
{
  "controller": {
    "type": "fixed",
    "params": {
      "batching_window_ms": 0,
      "max_concurrency": 32,
      "admission_threshold": 1.0
    }
  }
}
```

**产出物**：4 个新文件 + 1 个配置文件

---

### Task 7.5: Safety Guard (Layer 0)

**目标**：实现硬编码规则安全层，优先级最高，不经 LLM。

**操作**：
1. 新建 `controllers/safety_guard.py`：

```python
"""
SafetyGuard — Layer 0 安全规则检查器

最高优先级，所有 controller 的决策都必须经过 Safety Guard 校验。
使用硬编码规则，不调用 LLM，确保确定性和低延迟。
"""

class SafetyGuard:
    def __init__(self, config: dict = None):
        """
        config 可覆盖默认阈值:
        {
            "slo_ttft_p95_ms": 200,
            "slo_latency_p95_ms": 5000,
            "critical_reject_rate": 0.5,
            "critical_kv_cache_pct": 95,
            "rollback_after_consecutive_violations": 2
        }
        """

    def check_emergency(self, analyzer_state: dict) -> dict | None:
        """
        检查是否需要紧急干预。
        如果不需要返回 None；
        如果需要返回紧急动作:
        {
            "action_type": "fast_only" | "both",
            "fast_action": {"max_concurrency": 4, "admission_threshold": 0.7, ...},
            "slow_action": {"target_profile": "L"} | null,
            "reasoning": "reject_rate 超过 50%，紧急降低并发",
            "is_emergency": True
        }
        """
        # 规则 1: reject_rate > 50% → 立即降并发到最小，收紧准入
        # 规则 2: TTFT_p95 > SLO × 2.5 (500ms) → 紧急降并发
        # 规则 3: KV cache > 95% → 切换到 Profile L（最小 context）
        # 规则 4: 连续 2 个周期 SLO 违规 → 触发 rollback

    def validate_decision(self, decision: dict, analyzer_state: dict) -> dict:
        """
        校验 controller 决策是否安全。
        修正不安全的决策（如 risk=critical 时禁止 aggressive）。
        返回修正后的 decision。
        """
```

**产出物**：`controllers/safety_guard.py`

---

### Task 7.6: LLM Strategy Controller (Layer 1)

**目标**：实现以 LLM 为核心的策略决策器。

**操作**：
1. 新建 `controllers/llm_strategy_controller.py`：

```python
"""
LlmStrategyController — Layer 1 LLM 策略控制器

核心调用: F-LLM-1（策略选择 + 动作决策）
每个决策周期 (30s) 调用一次 LLM。
"""

class LlmStrategyController(BaseController):
    def __init__(self, llm_advisor: 'LlmAdvisor',
                 action_space: dict,
                 config: dict = None):
        """
        action_space: 可用的快慢动作空间
        config: {
            "decision_interval_s": 30,
            "history_window": 5,
            "current_fast_config": {...},
            "current_profile": "B"
        }
        """

    async def decide(self, analyzer_state: dict) -> dict:
        """
        1. 将 analyzer_state + current_config + history 组装为 F-LLM-1 输入
        2. 调用 llm_advisor.select_strategy()
        3. 解析 LLM 输出
        4. 记录到决策历史
        5. 返回统一格式的 decision
        """

    def update_config(self, new_config: dict):
        """更新当前配置（Executor 执行后回调）"""

    def record_outcome(self, decision: dict, execution_result: dict, post_metrics: dict):
        """记录决策结果用于历史参考"""

    def get_decision_history(self, n: int = 5) -> list[dict]:
        """返回最近 n 次决策及结果"""
```

2. 新建 `configs/controllers/controller_llm.json`:
```json
{
  "controller": {
    "type": "llm_strategy",
    "decision_interval_s": 30,
    "history_window": 5,
    "action_space": {
      "fast": {
        "batching_window_ms": [0, 10, 20, 50],
        "max_concurrency": [4, 8, 16, 32],
        "admission_threshold": [0.7, 0.8, 0.9, 1.0]
      },
      "slow": {
        "backend_profile": ["L", "B", "T"]
      }
    }
  }
}
```

**产出物**：`controllers/llm_strategy_controller.py` + `configs/controllers/controller_llm.json`

---

### Task 7.7: 闭环集成 + 稳定性测试

**目标**：把所有模块串成完整闭环并验证稳定性。

**操作**：
1. 修改 `proxy/proxy_server.py`，新增 `--controller` 参数：
   - 加载 controller 配置
   - 后台启动决策循环 (`asyncio.create_task`)：
     ```
     每 decision_interval_s 秒:
       1. 从 monitor 获取最新指标
       2. 注入 analyzer
       3. 调用 analyzer.get_full_state(llm_advisor) → state
       4. safety_guard.check_emergency(state) → emergency?
       5. 如果 emergency: 直接执行 emergency action
       6. 否则: controller.decide(state) → decision
       7. safety_guard.validate_decision(decision, state) → validated
       8. executor.check_cooldown / check_dwell_time
       9. executor.execute(validated)
       10. controller.record_outcome(...)
       11. 写入 controller_decisions.jsonl
     ```
   - 决策循环必须 catch all exceptions，确保单次失败不崩溃

2. 新建 `scripts/experiment/run_closed_loop.sh`：
```bash
#!/bin/bash
# 用法: bash scripts/experiment/run_closed_loop.sh <controller_config> <workload_config>
# 1. 启动 vLLM（默认 Profile B）
# 2. 启动 proxy（带 --controller 参数）
# 3. 启动 WorkloadGenerator 压测（指向 proxy port 9000）
# 4. 收集所有结果
# 5. 停止 proxy
```

3. 在 `results/<experiment_id>/` 中额外输出：
   - `controller_decisions.jsonl`: 每次决策的完整记录

4. **稳定性测试**：运行以下 4 个场景，每个 5 分钟：
   - `workload_baseline.json` (burst 混合)
   - `workload_rate4.json` (constant rate 4 req/s)
   - `workload_long_only.json` (全长请求)
   - `workload_phase_switch.json` (中途切换负载)
5. 验证标准：
   - 无崩溃
   - `controller_decisions.jsonl` 有 ≥5 条非 `none` 决策
   - SafetyGuard 在 inject 高压时正确触发（如果出现高压场景）
   - LLM 调用成功率 > 80%

**产出物**：`proxy/proxy_server.py` 更新、`scripts/experiment/run_closed_loop.sh`

**验证**：4 个场景各运行 5 分钟，产出完整数据，无崩溃

---

## 十、Week 8 — Optuna Controller + 对比实验

### Task 8.1: 添加 Optuna 依赖 + 离散搜索空间

**目标**：在 requirements.txt 中加入 Optuna 并定义离散搜索空间。

**操作**：
1. 在 `requirements.txt` 中添加 `optuna>=3.6.0` 和 `openai>=1.0.0`（如尚未添加）
2. 在 WSL2 中安装：`pip install optuna openai`
3. 定义搜索空间配置 `configs/controllers/controller_optuna.json`：

```json
{
  "controller": {
    "type": "optuna",
    "study_name": "vllm_proxy_optimization",
    "sampler": "tpe",
    "trial_duration_s": 60,
    "max_trials": 50,
    "llm_pruning_enabled": true,
    "action_space": {
      "batching_window_ms": [0, 10, 20, 50],
      "max_concurrency": [4, 8, 16, 32],
      "admission_threshold": [0.7, 0.8, 0.9, 1.0],
      "backend_profile": ["L", "B", "T"]
    },
    "reward": {
      "alpha": 0.4,
      "beta": 0.4,
      "gamma": 0.2,
      "slo_ttft_p95_ms": 200,
      "slo_latency_p95_ms": 5000
    }
  }
}
```

**产出物**：`requirements.txt` 更新 + `configs/controllers/controller_optuna.json`

**验证**：`python3 -c "import optuna; print(optuna.__version__)"` 无报错

---

### Task 8.2: LLM 搜索空间裁剪 (F-LLM-2)

**目标**：在每个 Optuna trial 前用 LLM 缩小候选范围。

**操作**：
1. 确保 `llm_advisor/llm_advisor.py` 的 `prune_search_space()` 方法已实现（Task 6.1 中定义）
2. 实现裁剪逻辑：
   - 调用 F-LLM-2 prompt 模板
   - LLM 返回 `constrained_space`（每个参数的受限候选集）和 `fixed_params`（直接固定的参数）
   - 裁剪后的空间传给 Optuna study 的 suggest 方法
3. 降级：LLM 不可用时使用完整搜索空间

**产出物**：`llm_advisor/llm_advisor.py` 的 `prune_search_space()` 完善

**验证**：
```bash
python3 -c "
import asyncio, json, os
from llm_advisor.llm_advisor import LlmAdvisor

config = json.load(open('configs/llm_advisor/llm_advisor_config.json'))['llm_advisor']
config['api_key'] = os.environ.get('DEEPSEEK_API_KEY', '')
advisor = LlmAdvisor(config)

state = {'bottleneck_type': 'decode', 'dominant_pattern': 'long_decode', 'risk_level': 'warning'}
space = {'batching_window_ms': [0,10,20,50], 'max_concurrency': [4,8,16,32], 'admission_threshold': [0.7,0.8,0.9,1.0], 'backend_profile': ['L','B','T']}
result = asyncio.run(advisor.prune_search_space(state, space, []))
print('Constrained space:', result.get('constrained_space'))
print('Fixed params:', result.get('fixed_params'))
"
```

---

### Task 8.3: OptunaController

**目标**：实现 Optuna ask-and-tell 控制器。

**操作**：
1. 新建 `controllers/optuna_controller.py`：

```python
"""
OptunaController — Layer 2 贝叶斯优化控制器

使用 Optuna ask-and-tell 接口进行在线配置搜索。
每个 trial = 一个控制窗口（60s），运行该配置后观测 reward。
LLM (F-LLM-2) 在每个 trial 前裁剪搜索空间。
"""
import optuna


class OptunaController(BaseController):
    def __init__(self, llm_advisor: 'LlmAdvisor',
                 safety_guard: 'SafetyGuard',
                 config: dict):
        """
        创建 Optuna study (TPESampler)
        """

    async def decide(self, analyzer_state: dict) -> dict:
        """
        1. 如果当前 trial 还在运行中（未到 trial_duration_s）→ 返回 none
        2. 如果当前 trial 已结束:
           a. 计算 reward
           b. tell(trial, reward)
           c. 记录 trial 结果
        3. 开始新 trial:
           a. 调用 LLM 裁剪搜索空间 (F-LLM-2)
           b. study.ask() 在裁剪后的空间中采样
           c. 转换为 fast_action + slow_action
           d. 返回 decision
        """

    def _compute_reward(self, trial_metrics: dict) -> float:
        """
        R = alpha * throughput_norm - beta * slo_violation_rate - gamma * reject_rate

        throughput_norm: 归一化到 [0, 1]，以 baseline 0 的 75 tps 为基准
        slo_violation_rate: SLO 违规请求比例
        reject_rate: 被拒绝请求比例
        """

    def get_trial_history(self) -> list[dict]:
        """返回所有 trial 的参数和 reward"""

    def get_best_params(self) -> dict:
        """返回当前最优配置"""
```

**产出物**：`controllers/optuna_controller.py`

**验证**：
```bash
# 在闭环中运行 10 分钟（~10 个 trial）
bash scripts/experiment/run_closed_loop.sh configs/controllers/controller_optuna.json configs/workloads/workload_baseline.json
# 检查 trial 历史
python3 -c "
import json
with open('results/<最新实验ID>/controller_decisions.jsonl') as f:
    trials = [json.loads(l) for l in f if 'trial' in json.loads(l).get('decision',{}).get('reasoning','')]
print(f'{len(trials)} trials recorded')
"
```

---

### Task 8.4: 4 组对比实验

**目标**：在相同 workload 下对比 4 种 controller 的性能。

**操作**：
1. 新建 `scripts/experiment/run_comparison.sh`：

```bash
#!/bin/bash
# 4 组对比实验，同一 workload 下:
#
# Group A: 直连 vLLM（无 proxy）
#   python3 benchmarks/run_benchmark.py --port 8000 --workload configs/workloads/workload_baseline.json
#
# Group B: proxy + FixedController（对照基线）
#   bash scripts/experiment/run_closed_loop.sh configs/controllers/controller_fixed.json configs/workloads/workload_baseline.json
#
# Group C: proxy + LlmStrategyController（规则+LLM）
#   bash scripts/experiment/run_closed_loop.sh configs/controllers/controller_llm.json configs/workloads/workload_baseline.json
#
# Group D: proxy + OptunaController（LLM+BO）
#   bash scripts/experiment/run_closed_loop.sh configs/controllers/controller_optuna.json configs/workloads/workload_baseline.json
#
# 每组运行 10 分钟，固定 workload seed=42
# 输出到 results/comparison_<timestamp>/
```

2. 每组实验使用相同的 workload 配置和随机种子
3. 建议使用 `workload_phase_switch.json`（包含负载切换）以展示 controller 的适应能力
4. 汇总 4 组的 summary.json 到一个对比表

**产出物**：`scripts/experiment/run_comparison.sh`

**验证**：4 组实验均成功完成，`results/comparison_*/` 下各有完整数据

---

### Task 8.5: 消融实验 + 结果汇总

**目标**：通过消融实验验证各功能的贡献，生成论文所需数据。

**操作**：
1. **消融实验**（关闭单个功能观察退化）：
   - Ablation 1: 关闭 batching window（固定 0ms）
   - Ablation 2: 关闭 concurrency 控制（固定 32）
   - Ablation 3: 关闭 admission control（固定 1.0）
   - Ablation 4: 关闭 LLM（纯规则 controller）
   - Ablation 5: 关闭 Optuna（纯 LLM controller）

2. 新建 `scripts/experiment/run_ablation.sh`：
```bash
#!/bin/bash
# 对每个消融组运行 5 分钟，记录性能退化
```

3. **结果汇总**：
   - 新建 `results/comparison_summary.json`，包含所有组的对比数据
   - 格式：
   ```json
   {
     "experiment_date": "2026-05-XX",
     "workload": "workload_phase_switch.json",
     "groups": {
       "direct": {"throughput_tps": ..., "ttft_p95_ms": ..., "latency_p95_ms": ..., "reject_rate": ...},
       "fixed": {...},
       "llm_strategy": {...},
       "optuna": {...}
     },
     "ablation": {
       "no_batching": {...},
       "no_concurrency_control": {...},
       "no_admission": {...},
       "no_llm": {...},
       "no_optuna": {...}
     }
   }
   ```

4. 为论文准备的关键数据点：
   - 4 组 controller 的吞吐量/延迟/SLO 违规率对比
   - Optuna 收敛曲线（trial_id vs reward）
   - Controller 决策轨迹（时序图数据）
   - 消融实验退化程度

**产出物**：`scripts/experiment/run_ablation.sh` + `results/comparison_summary.json`

**验证**：所有组实验完整，对比表数据自洽

"""

# Now fix the tail section
# Fix duplicate lines in §11 deliverables scripts section
tail_text = ''.join(tail)

# Fix the scripts section that has duplicates
old_scripts = """├── experiment/
│   ├── run_experiment_suite.sh
│   ├── run_closed_loop.sh
│   ├── run_comparison.sh
│   └── run_ablation.sh
└── test/
    ├── run_comparison.sh
├── run_ablation.sh
└── test_analyzer_integration.sh"""

new_scripts = """├── experiment/
│   ├── run_experiment_suite.sh
│   ├── run_closed_loop.sh
│   ├── run_comparison.sh
│   └── run_ablation.sh
└── test/
    └── test_analyzer_integration.sh"""

tail_text = tail_text.replace(old_scripts, new_scripts)

# Fix "experiment/experiment/" doubled paths
tail_text = tail_text.replace('experiment/experiment/', 'experiment/')

# Fix "gt_params" typo in any remaining content
# (this was in the corrupted Task 8.3 but we replaced it fully above)

# Fix stray "s/controller" fragments that may exist in tail
tail_text = tail_text.replace('\ns/controller\n', '\n')

# Compose final file
result = ''.join(head) + '\n' + clean_section + '\n---\n\n' + tail_text

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(result)

print(f"Done. File rewritten. New line count: {result.count(chr(10)) + 1}")
