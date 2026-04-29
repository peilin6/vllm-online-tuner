# vLLM 第3-8周执行任务书：LLM 驱动的自主优化 Agent

> **用法**：将本文件完整交给 AI 编码助手，让它按 Task 顺序执行。每完成一个 Task 后，让 AI 汇报产出物并等你确认，再继续下一个 Task。
>
> **替代**：本文件替代旧版 `week3-5_execution.prompt.md`。

---

## 一、项目背景

你正在帮助一个毕业设计项目：**vLLM 在线推理服务性能评测与优化**。

论文定位：*an autonomous LLM-driven runtime adaptation agent for vLLM inference optimization*。

### 1.1 硬件与环境

**Week 1–4（已完成，历史平台，baseline_0 数据来源）**

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB VRAM |
| 驱动 | 566.26, CUDA 12.4 |
| OS | Windows + WSL2 (Ubuntu-22.04, kernel 6.6.87.2) |
| Python venv | `~/vllm-venv` |
| 模型 | Qwen/Qwen2.5-3B-Instruct-AWQ (~2.69 GB) |
| 模型路径 | `~/models/Qwen2.5-3B-Instruct-AWQ` |

**Week 5 起（目标平台，详见 §七 迁移计划）**

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA RTX A6000, 48 GB VRAM |
| 驱动 | ≥ 550, CUDA 12.4 |
| Python venv | `~/vllm-venv-a6000` |
| 模型 | Qwen/Qwen3-8B (bf16, ~16 GB) |
| 模型路径 | `~/models/Qwen3-8B` |

**软件栈（两套平台共用版本）**：vLLM 0.6.6.post1、PyTorch 2.5.1+cu124、transformers 4.49.0、Python 3.10。vLLM 服务端口固定 8000；**本方案不引入 proxy 层**，所有参数调优直接作用于 vLLM engine args（见 §二）。

### 1.2 关键约束（必须遵守）

1. **所有命令在 WSL2 Ubuntu-22.04 中执行**，4060 用 `~/vllm-venv`，A6000 用 `~/vllm-venv-a6000`
2. **代理问题**：任何使用 Python `requests` 或 `aiohttp` 的脚本都必须在运行前 `export no_proxy="*"`
3. **PowerShell→WSL 引号问题**：不要在 PowerShell 中内联执行含引号/括号的 Python 命令，务必写成 .sh 脚本再 `wsl -- bash xxx.sh`
4. **显存限制**：4060 平台只能跑 3B-AWQ；A6000 平台默认跑 Qwen3-8B bf16（不做量化，见 Task 5.2）；**不能同时运行两个 vLLM 实例**
5. **不修改 vLLM 源码**，只在外部做 workload 生成、agent 决策与指标采集
6. **所有新文件使用 UTF-8 编码**
7. **所有 Python 脚本**开头加 `#!/usr/bin/env python3` 和 `# -*- coding: utf-8 -*-`
8. **日志输出**使用中文，代码注释使用中文，变量名使用英文
9. **不要安装不必要的新依赖**，新增依赖仅限 `openai>=1.0.0`、`tiktoken`、`python-dotenv`、`optuna>=3.6.0`、`scipy`（BO / 敏感度分析工具用）

### 1.3 当前工程结构（第1-4周已完成，含重构）

```
d:/vlllm/
├── configs/
│   ├── experiments/
│   │   └── baseline_0.json          # 基线实验配置
│   ├── workloads/
│   │   ├── workload_schema.json     # workload JSON Schema
│   │   ├── workload_baseline.json   # 默认 workload 配置
│   │   ├── workload_burst.json
│   │   ├── workload_poisson.json
│   │   ├── workload_prefix.json
│   │   ├── workload_phase_switch.json
│   │   └── ...                      # 其他 workload 预设（共14个）
│   └── llm_prompts/
│       └── few_shot_examples.json   # LLM few-shot 示例数据
├── benchmarks/                      # 纯压测执行（Module A 的执行入口）
│   ├── run_benchmark.py             # 核心压测脚本（asyncio+aiohttp，SSE流式）
│   ├── visualize_speed.py           # 逐token速度可视化
│   └── prompts.json                 # 5条固定prompt样本（向后兼容）
├── workloads/                       # Module A: Workload 生成与数据
│   ├── __init__.py
│   ├── workload_generator.py        # WorkloadGenerator 类
│   ├── prompts_pool.json            # 30+ prompt 语料池
│   └── prefix_pool.json             # 5+ 共享前缀模板
├── monitors/                        # Module D: 基础设施监控
│   ├── __init__.py
│   ├── gpu_monitor.py               # GPU 实时采样器（pynvml）
│   └── vllm_metrics_collector.py    # vLLM /metrics 端点采集
├── scripts/
│   ├── server/                      # 服务启停
│   │   ├── launch_server.sh         # 前台启动vLLM
│   │   ├── start_server.sh          # 后台启动+等待就绪
│   │   └── stop_server.sh           # 停止服务（SIGTERM→SIGKILL）
│   ├── setup/                       # 环境安装
│   │   ├── install_all.sh
│   │   ├── setup_env.sh
│   │   └── download_model.sh
│   ├── experiment/                  # 实验运行
│   │   ├── run_baseline.sh          # 一键baseline流水线
│   │   ├── run_experiment_suite.sh
│   │   └── run_initial_experiments.sh
│   ├── verify/                      # 验证脚本
│   │   ├── verify_server.py         # 健康检查+模型验证+生成测试
│   │   ├── verify_week3.sh
│   │   └── check_and_run.sh
│   ├── test/                        # 测试运行
│   │   ├── run_tests.sh
│   │   └── run_all_tests.sh
│   └── tools/                       # 工具脚本
│       └── collect_sysinfo.py       # GPU/CPU/CUDA/包版本采集
├── tests/                           # 单元测试（76个，全部通过）
├── docs/                            # 研究说明、实验模板、环境文档
├── results/                         # 实验结果
├── logs/                            # vLLM服务日志
├── plan.md
├── README.md
└── requirements.txt
```

### 1.4 最新 baseline 0 实测数据

| 指标 | 值 |
|------|-----|
| 成功率 | 100.0% (50/50) |
| 吞吐量 | 0.37 req/s |
| Token 吞吐 | 75.37 tokens/s |
| TTFT Mean | 62.6 ms |
| TTFT P95 | 90.8 ms |
| Avg Latency | 2717.2 ms |
| P95 Latency | 3571.1 ms |
| P99 Latency | 3624.7 ms |
| 平均输出 Tokens/req | 204.8 |
| 总耗时 | ~136s（50请求串行 concurrency=1） |

### 1.5 Phase 0 已完成

- ✅ Task 0.1: 文档模型版本统一（6个文件修复 7B→3B-AWQ）
- ✅ Task 0.2: Token 计数精度修复（stream_options.include_usage + usage 解析）

---

## 二、系统架构总览

> **设计口径**：本项目**不做 proxy 在线控制**。自 Week 5 起采用 **VTA-Agent**（vLLM Tuning Agent）——一个 **LLM 驱动 + 算法增强** 的闭环调参 agent：
>
> - **输入**：vLLM 运行时 metrics（throughput / TTFT / TPOT / KV / preempt / queue）。
> - **输出**：修改后的 vLLM engine args（一次一改，重启生效）。
> - **驱动方**：LLM 担任 Diagnoser / Proposer / Reflector / Reporter，**决策每一步该做什么**。
> - **加速器**：把 **Bayesian Optimization (Optuna TPE)、参数敏感度分析、Pareto 过滤** 等算法封装成 LLM 可调用工具——LLM 不必亲自做数值优化，但可以在合适时机让算法给出建议、再用领域知识审阅/采纳/否决。
>
> 即：**LLM 是 orchestrator，算法是 inner-loop tool**。这样既保留 LLM 的语义推理与解释能力，又借力成熟优化算法在数值密集子问题上的效率。

### 2.1 VTA-Agent 闭环架构

```
                    ┌──────────────────────────────────────┐
                    │    workload.json（固定，可复现）      │
                    └──────────────┬───────────────────────┘
                                   │
    ┌──────────────────────────────┴──────────────────────────────┐
    │                     VTA-Agent Control Loop                   │
    │                                                              │
    │   ┌────────────────────────────────────────────────────┐    │
    │   │  1. Observe  (Tool: observe_trial)                 │    │
    │   │     Launcher 重启 vLLM with current config         │    │
    │   │     Runner + Monitor 跑一轮 bench (60–120s)        │    │
    │   │     → TrialMetrics (throughput/TTFT/TPOT/KV/...)   │    │
    │   └────────────────────┬───────────────────────────────┘    │
    │                        ▼                                     │
    │   ┌────────────────────────────────────────────────────┐    │
    │   │  2. Diagnose  (🤖 A-LLM)                           │    │
    │   │     input: 最近 N 个 trial 的 metrics + memory      │    │
    │   │     output: bottleneck + hypothesis + confidence   │    │
    │   └────────────────────┬───────────────────────────────┘    │
    │                        ▼                                     │
    │   ┌────────────────────────────────────────────────────┐    │
    │   │  3. Propose   (🤖 P-LLM + Tools)                   │    │
    │   │     tools: query_param_docs / list_tried_configs / │    │
    │   │            compare_trials                          │    │
    │   │     output: ConfigDelta {param: new_value, reason} │    │
    │   └────────────────────┬───────────────────────────────┘    │
    │                        ▼                                     │
    │   ┌────────────────────────────────────────────────────┐    │
    │   │  4. Safety Check  (硬编码 Judge，不经 LLM)          │    │
    │   │     · value 在 REGISTRY 候选范围内                  │    │
    │   │     · 不违反 SLO 守门（预估值）                      │    │
    │   │     · 不重复最近 3 步已否决的组合                    │    │
    │   │     unsafe → 反馈给 P-LLM 重试一次                  │    │
    │   └────────────────────┬───────────────────────────────┘    │
    │                        ▼                                     │
    │   ┌────────────────────────────────────────────────────┐    │
    │   │  5. Act       (Tool: apply_config)                 │    │
    │   │     写临时 config → Launcher.restart()             │    │
    │   │     失败（OOM / 不就绪）→ 自动 rollback            │    │
    │   │     成功 → 返回 observe(步骤 1)测新 trial           │    │
    │   └────────────────────┬───────────────────────────────┘    │
    │                        ▼                                     │
    │   ┌────────────────────────────────────────────────────┐    │
    │   │  6. Reflect   (🤖 R-LLM)                           │    │
    │   │     input: (prev_trial, new_trial, hypothesis)     │    │
    │   │     output: 验证是否与预期一致 + 新笔记 + next_move │    │
    │   │     笔记写入 ExperienceMemory                      │    │
    │   └────────────────────┬───────────────────────────────┘    │
    │                        ▼                                     │
    │   ┌────────────────────────────────────────────────────┐    │
    │   │  7. Terminate?                                     │    │
    │   │     · 连续 3 步改动 <2% → converged                │    │
    │   │     · step_budget 用尽                             │    │
    │   │     · R-LLM 主动 stop=true                         │    │
    │   │     否则回到步骤 2                                  │    │
    │   └────────────────────┬───────────────────────────────┘    │
    │                        ▼                                     │
    │              best_trial + full memory                        │
    └──────────────────────────┬──────────────────────────────────┘
                               ▼
        ┌──────────────────────────────────────────┐
        │  Reporter  (🤖 Summary LLM)              │
        │  memory → final_report.md                │
        └──────────────────────────────────────────┘
```

### 2.2 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| **VtaAgent** | `tuner/agent.py` | 主循环，编排 Observe/Diagnose/Propose/Act/Reflect |
| **ExperienceMemory** | `tuner/memory.py` | 存所有 trial + LLM 笔记；支持 `summarize(top_k, recent_n)` 压缩为 LLM 可读摘要 |
| **ToolRegistry** | `tuner/tools.py` | LLM 可调用的工具集合（见 §2.4） |
| **VllmLauncher** | `tuner/launcher.py` | 重启 + 就绪检查 + 计时 + 失败检测 |
| **Runner** | `tuner/runner.py` | 单次 bench + metrics 聚合 → `TrialMetrics` |
| **ParamRegistry** | `tuner/param_registry.py` | 参数元数据：候选值、影响维度、先验说明（LLM 通过工具读取） |
| **Judge** | `tuner/judge.py` | 硬安全阈值：值域检查、SLO 守门、去重 |
| **LlmClient** | `llm_advisor/llm_client.py` | OpenAI/DeepSeek API 封装 + 限流 + 重试 + 缓存 |
| **Prompts** | `llm_advisor/prompts.py` | 生成 A/P/R/Reporter prompts |

### 2.3 LLM 参与边界

**原则**：**LLM 是 orchestrator，不是数值优化器**。每步该问什么、查谁、是否调用 BO 给建议、是否采纳建议、是否回滚——全部由 LLM 推理给出；BO / 敏感度 / Pareto 等算法只在 LLM 主动调用时执行，结果作为参考喂回 LLM。

**LLM 做的事**：

| 角色 | 模型调用 | 输入 | 输出 | 频次 |
|------|---------|------|------|------|
| **Diagnoser (A-LLM)** | 每步 1 次 | 最近 N 个 trial metrics + memory 摘要 | bottleneck + 假设 + 置信度 | 每步 |
| **Proposer (P-LLM)** | 每步 1–3 次（可用工具） | 诊断 + 历史 + 参数文档 | ConfigDelta | 每步 |
| **Reflector (R-LLM)** | 每步 1 次 | 前后 trial 对比 + 原假设 | 笔记 + 下一步建议 + stop 标志 | 每步 |
| **Reporter** | 结束时 1 次 | 完整 memory + best_trial | 最终报告 markdown 段 | 每 run 1 次 |

**LLM 不做的事**（硬编码）：

| 职责 | 实现 |
|------|------|
| 原始 metrics 采集 | `monitors/` (pynvml + `/metrics` + SSE 时间戳) |
| Metrics 聚合为 `TrialMetrics` | `tuner/metrics.py` |
| 值域校验 / SLO 守门 / 去重 | `tuner/judge.py` |
| 重启 / 就绪检测 / 回滚 | `tuner/launcher.py` |
| LLM API 调用的限流/重试/缓存 | `llm_advisor/llm_client.py` |
| 终止条件判断（收敛 / 预算用尽） | `tuner/agent.py` main loop |

### 2.4 Agent 工具集（LLM 可调用）

P-LLM 采用 OpenAI function-calling 风格，可在单次推理中多次调用工具再输出最终 ConfigDelta。工具分两类：**A. 只读查询**（轻量、永远可用）；**B. 优化算法**（重计算，由 LLM 按需触发，结果作为建议参考）。

#### A. 只读查询工具

| 工具名 | 参数 | 返回 | 用途 |
|--------|------|------|------|
| `list_params()` | — | `[{name, mutation_class, candidates, affects, notes}, ...]` | 枚举全部可调参数 |
| `query_param_docs(name)` | `name: str` | `ParamSpec` 详情 + 该参数在 memory 中被试过的所有值及结果 | 深挖某个参数 |
| `list_tried_configs(top_k=5)` | `top_k: int` | 历史 trial 按 score 排序的前 k 条 | 查"以前什么好" |
| `compare_trials(a_id, b_id)` | ids | 两 trial 的 metrics diff + params diff | 做 A/B 对比 |
| `get_baseline()` | — | `TrialMetrics` | 对照基线 |
| `get_memory_notes()` | — | R-LLM 累积的自然语言笔记 | 查"agent 自己总结过什么" |

#### B. 优化算法工具（LLM 驱动的算法）

这些工具把成熟的数值/优化算法封装成函数调用——LLM 决定**何时调用、传哪些参数、是否采纳结果**，算法负责数值密集的子任务。

| 工具名 | 参数 | 返回 | 何时该调用 |
|--------|------|------|-----------|
| `bo_suggest(scope, n=1, objective="score", min_history=4)` | `scope`: 待搜索参数子集；`objective`: 单目标或加权多目标 | `[{config, expected_score, acquisition, uncertainty}, ...]` | memory ≥ 4 条且 LLM **不确定**该往哪走时，让 TPE 给候选；LLM 再审阅是否符合直觉与笔记 |
| `param_sensitivity()` | — | `[{param, score_variance, spearman, monotonic, best_value}, ...]` 按敏感度降序 | 想知道"哪些参数最值得继续动" / 收敛时定位还能挖什么 |
| `pareto_front(metrics=["throughput","latency_p95"])` | 二/三目标 | 非支配 trial 列表 + 支配关系 | 当 SLO 紧、需要 trade-off 时帮 LLM 看清楚边界 |
| `local_grid(param, neighborhood=2)` | 参数名 + 邻域宽度 | 候选值列表（已剔除试过的） | 微调阶段，LLM 想"在 best 附近扫一圈" |
| `cluster_workload_phases(window_s=30)` | — | metrics 时序的 phase 切分 + 每 phase 指纹 | workload 有 phase-switch 时给 LLM 看清楚分段 |

**算法实现**（程序侧，文件 `tuner/optimizer.py`）：

- `bo_suggest` → `optuna.create_study(sampler=TPESampler(n_startup_trials=2))`，把 memory 中所有 trial enqueue 后 `study.ask()` 出 n 个候选；返回 `acquisition` 用 EI 近似 / `uncertainty` 用 TPE l(x)/g(x) 比值
- `param_sensitivity` → score 关于 param 的方差 + Spearman 相关 + 单调性检验（n≥4 才有效，否则返回 `insufficient_data=True`）
- `pareto_front` → 朴素 O(n²) 扫描非支配集
- `local_grid` → 从 `REGISTRY[param].candidates` 中按距离选邻域 ∖ 已试过
- `cluster_workload_phases` → metrics_timeseries 的 1D KMeans / change-point（基于 throughput 序列）

**关键设计**：算法工具的输出**永远是建议**，不是命令。P-LLM 必须在 ConfigDelta 的 `reason` 字段里给出"采纳/修改/拒绝该建议"的理由（例如："BO 推荐 max_num_seqs=128 但 memory 笔记显示>64 触发 preempt，采纳但下调到 96"）。

#### 禁止工具（防止 LLM 作弊 / 越权）

- LLM **不能**直接调用 `apply_config`——必须由 P-LLM 输出结构化 ConfigDelta，Judge 审核后由程序调用
- LLM **不能**调用 shell / 文件系统 / 网络
- 所有工具均为**纯函数 / 只读**，无副作用（除写一条调用日志到 `llm_calls.jsonl`）

### 2.5 Memory 结构

```python
@dataclass
class TrialRecord:
    trial_id: int
    config: dict              # vLLM engine args 快照
    delta_from_prev: dict     # 相对上一 trial 的改动
    hypothesis: str           # P-LLM 提出该配置时的假设
    metrics: TrialMetrics
    score: float
    constraint_violations: list[str]  # SLO 违反项
    reflection: str           # R-LLM 写的"是否符合预期 / 学到什么"
    timestamp: str

@dataclass
class ExperienceMemory:
    baseline: TrialMetrics
    trials: list[TrialRecord]
    notes: list[str]          # R-LLM 累积的跨 trial 笔记（如"max_num_seqs>64 会触发 preempt"）
    rejected_proposals: list[tuple[dict, str]]  # (config, reject_reason)

    def summarize(self, top_k=3, recent_n=5) -> str:
        """返回给 A/P/R-LLM 的压缩摘要：
        - baseline 关键数字
        - Top-k 最佳 trial 的 config + 关键 metrics
        - 最近 n 步的 delta + reflection
        - 累积 notes
        """
```

Memory 是 agent 的"学习成果"；LLM 的每次调用都看到它的摘要，而不是 raw trial 数组，以控 token。

### 2.6 参数分层与动作空间

**RESTART 类（绝大多数调优旋钮，每次改动需重启 vLLM）**

| 参数 | 候选 | 影响维度 | 先验（供 P-LLM 参考） |
|------|------|---------|----------------------|
| `max_num_batched_tokens` | 2048 / 4096 / 8192 / 16384 | throughput, TTFT, TPOT | 大→吞吐好 TTFT 好，小→TPOT 好 |
| `max_num_seqs` | 16 / 32 / 64 / 128 | throughput, KV 压力 | 大→吞吐上限高，但易 preempt |
| `gpu_memory_utilization` | 0.85 / 0.90 / 0.93 / 0.95 | KV 容量 | 高→KV 多，但 activation 余量小 |
| `max_model_len` | 2048 / 4096 / 8192 | KV block 分配 | 按实际 p95 剪裁，省 KV |
| `enable_prefix_caching` | False / True | prefill 节省 | prefix 共享高时开 |
| `enable_chunked_prefill` | False / True | TTFT/TPOT 平衡 | 长短混合时开 |
| `max_num_partial_prefills` | 1 / 2 / 4 | chunked prefill 并发 | 长短混合时调高 |
| `max_long_partial_prefills` | 1 / 2 | 长 prompt 并发上限 | 同上 |
| `long_prefill_token_threshold` | 1024 / 2048 / 4096 | 长短 prompt 分界 | 由 profiler 推出 |

**RUNTIME 类**：仅 `reset_prefix_cache`（cold/warm 对照用，不是调优旋钮）。

**PER_REQUEST 类**：sampling params、`priority`。本方案不作为调优目标。

**结论**：本 agent **每改一个参数就必须重启 vLLM**，没有"便宜动作"。因此 agent loop 每步 ~3 min（重启 60s + bench 90s + LLM 决策 5–10s），这个 **step 成本就是约束 LLM 必须高效推理**的客观理由。

### 2.7 优化目标与约束

- **SLO 硬约束（违反则 score = −∞，该 trial 仍被记入 memory 用于学习）**：
  - `TTFT_p95 ≤ baseline.TTFT_p95 × 1.2`
  - `latency_p95 ≤ baseline.latency_p95 × 1.2`
  - `preemption_rate_per_min ≤ 5`
- **主目标**：
  ```
  score = (throughput_tok_per_s − baseline.throughput_tok_per_s)
        /  baseline.throughput_tok_per_s
  ```
- **验收门槛**：至少 2/3 的 workload 类型达到 ≥ 10% 提升，3/3 不违反 SLO。

### 2.8 预算估算（Qwen3-8B on A6000）

**单步成本**（一次 Observe+Diagnose+Propose+Act+Reflect）

| 阶段 | 时间 |
|------|------|
| Launcher.restart (CUDA graph) | ~50 s |
| Warmup + Bench | ~90 s |
| A-LLM + P-LLM + R-LLM 三次调用 | ~5 s |
| 算法工具（BO / 敏感度 / Pareto，按需触发，纯本地计算） | <1 s |
| Judge + memory 更新 | <1 s |
| **合计** | **~145 s ≈ 2.5 min** |

**单 workload 预算**：最多 20 步 × 2.5 min ≈ **50 min**（远快于无约束网格搜索）

**减时间策略**：

1. **早期步骤用 `enforce_eager`**（步骤 1–5）：每步 −25 s，A-LLM 拿到的信号足够粗判；最优候选 ≥3 次出现在 Top-3 后自动切回 graph 模式。
2. **保留 OS 页缓存**（不 `drop_caches`）：第 2 次起 restart −15 s。
3. **在线早停**：bench 窗口内 `preempt_rate > 2/s` 或 `throughput < baseline × 0.5` → 20 s 内终止，返回"崩盘 trial"给 LLM 学习。
4. **LLM 短路**：A-LLM 判定"已收敛 + SLO 充裕"时直接输出 stop，跳过 P-LLM。

**3 种 workload 总预算**：~3 h（含消融 ~5 h）。

**LLM API 成本**：每步 ~2000 tokens，20 步 × 3 次 = ~120k tokens/run，DeepSeek 单价 ¥1-2/M → ~¥0.3/run。全实验 < ¥5。

---

## 三、LLM 调用详细设计

### 3.1 调用全景图

VTA-Agent 每一步调用 **3 个 LLM 角色**，加上结束时 1 次 Reporter。所有调用通过同一个 `LlmClient`，走 OpenAI 兼容接口。

```
每个 agent step (~2.5 min, 重启+bench+LLM):

  ┌────────────────────────────────────────────┐
  │  A-LLM (Diagnoser)   ~1 s                  │
  │  input : memory.summarize() + 最新 trial    │
  │  output: diagnosis JSON                    │
  └──────────────────┬─────────────────────────┘
                     ▼
  ┌────────────────────────────────────────────┐
  │  P-LLM (Proposer)    ~2-4 s                │
  │  input : diagnosis + memory                │
  │  tools : list_params / query_param_docs /  │
  │          list_tried_configs / compare_trials│
  │  output: ConfigDelta JSON                  │
  └──────────────────┬─────────────────────────┘
                     ▼
           Judge 审核（硬编码）
                     ▼
              apply_config → observe
                     ▼
  ┌────────────────────────────────────────────┐
  │  R-LLM (Reflector)   ~1-2 s                │
  │  input : (prev_trial, new_trial, hypothesis)│
  │  output: reflection JSON + note            │
  └──────────────────┬─────────────────────────┘
                     ▼
         (loop or terminate)

tuning run 结束后:

  ┌────────────────────────────────────────────┐
  │  Reporter LLM        ~3 s                  │
  │  input : 完整 memory + best_trial          │
  │  output: markdown 段落（写入 report.md）   │
  └────────────────────────────────────────────┘
```

**失败降级**：
- A/P/R 任何一次 LLM 失败 → 走 fallback 规则（硬编码最小决策：P-LLM fallback 用"轮询下一个未试过的参数中位值"）；该 step 的 trial 仍然执行并记入 memory，标记 `llm_source=fallback`。
- Reporter 失败 → report 只输出结构化数据表，不生成自然语言段。

### 3.2 A-LLM：Diagnoser（每步诊断瓶颈）

**System Prompt**:

```
你是 vLLM 推理服务的性能诊断专家。你在一个闭环调参 agent 内部工作，每一步都会收到最新 trial 的
运行时指标、历史最佳 trial 与 agent 自己积累的笔记。你的任务是定位当前的首要瓶颈并提出一个
明确的、可验证的假设（下一步如果朝某方向调参会得到什么样的指标变化）。

瓶颈枚举: prefill_bound | decode_bound | kv_cache_pressure | preempt_storm |
          queue_backlog | underutilized | slo_margin_low | converged

硬性规则:
- 基于实际数字推理，不要编造；如果数据不足以判断某瓶颈，confidence 给低分。
- 如果最近 3 步 score 改动 <2% 且 SLO 无余量，应输出 converged。
- 只输出 JSON，不要 markdown 围栏以外的任何解释。
```

**User Prompt 模板**:

```
## Baseline（对照组）
{baseline_metrics_json}

## Memory 摘要（Top-3 最佳 + 最近 5 步）
{memory_summary}

## 最新 Trial
- trial_id: {tid}
- config: {config_delta_from_baseline}
- metrics:
  throughput_tok_per_s={tput}  (vs baseline {base_tput}, {pct:+.1%})
  TTFT_p95_ms={ttft}, latency_p95_ms={lat}, TPOT_p95_ms={tpot}
  preempt_rate_per_min={preempt}, kv_cache_usage_p95={kv}
  queue_time_p95_ms={qt}, early_killed={ek}
- SLO 余量:
  TTFT headroom = {ttft_headroom:.1%}   # (limit - current) / limit
  latency headroom = {lat_headroom:.1%}

请输出 diagnosis JSON。
```

**输出 JSON Schema**:

```json
{
  "bottleneck": "decode_bound",
  "confidence": 0.78,
  "evidence": "TPOT_p95 从 baseline 45ms 升到 58ms；GPU util 95%；preempt_rate=0",
  "hypothesis": "max_num_seqs=32 时 decode 并发不足，尝试 64 预期 throughput +15%，TPOT 基本不变",
  "slo_pressure": "low | medium | high",
  "should_stop": false
}
```

### 3.3 P-LLM：Proposer（每步提出参数改动，可调用工具）

**System Prompt**:

```
你是 vLLM 推理服务的参数调优决策者。你接到一份诊断（bottleneck + hypothesis）后，
必须选择唯一的一次参数改动并输出结构化 ConfigDelta。你可以通过调用工具完成两类工作：
查询历史/参数文档；触发数值优化算法获得候选建议。最终必须输出一条 ConfigDelta。

工具列表（A. 只读）:
- list_params() -> [ParamSpec]
- query_param_docs(name) -> ParamSpec + 该参数在 memory 中被试过的值及结果
- list_tried_configs(top_k=5) -> Top-k 最佳历史 trial
- compare_trials(a_id, b_id) -> metrics diff + params diff
- get_memory_notes() -> R-LLM 累积笔记

工具列表（B. 优化算法，按需调用）:
- bo_suggest(scope, n=1, objective="score") -> Optuna TPE 在 memory 上拟合后的 n 个候选 config + 不确定度
- param_sensitivity() -> 各参数 score 方差 / Spearman 相关 / 单调性，按敏感度降序
- pareto_front(metrics=["throughput","latency_p95"]) -> 非支配 trial 列表
- local_grid(param, neighborhood=2) -> best 附近未试过的候选值
- cluster_workload_phases() -> metrics 时序的 phase 切分（仅 mixed workload 有意义）

调用建议（agent 工作流）:
- memory < 4 条 -> 依赖 diagnosis + query_param_docs 直接决策（BO 数据不足）
- memory ≥ 4 条且 diagnosis.confidence < 0.6 -> 调用 bo_suggest 获取候选，再用领域知识审阅
- 连续 2 步 score 改动 <3% -> 调用 param_sensitivity 找下一个值得动的参数
- SLO headroom < 10% -> 调用 pareto_front 看清楚 trade-off 边界
- 接近收敛 -> 调用 local_grid 在 best 邻域微调

硬性规则:
- 同一步只改 1 个参数（除非 diagnosis.should_stop=true）。
- 新值必须来自 ParamSpec.candidates（不接受自由数值）。
- 不能重复 memory.rejected_proposals 中最近 3 条。
- 每一个 ConfigDelta 必须附带 expected_effect。
- 若调用了优化算法工具，reason 字段必须明确写"采纳 / 修改 / 拒绝该建议"及理由。
- 若 diagnosis.should_stop=true，输出 {"action": "stop", "reason": "..."}。

只输出 JSON。
```

**User Prompt 模板**:

```
## Diagnosis（来自 A-LLM）
{diagnosis_json}

## 当前 config（完整）
{current_config_json}

## Memory 摘要
{memory_summary}

## 累积笔记（来自 R-LLM）
{notes_list}

请按需调用工具（至少 1 次 query_param_docs 验证候选合法性；视情况调用 bo_suggest /
param_sensitivity / pareto_front / local_grid），然后输出 ConfigDelta。
```

**输出 JSON Schema**:

```json
{
  "action": "change_param | stop",
  "param": "max_num_seqs",
  "old_value": 32,
  "new_value": 64,
  "hypothesis_ref": "验证 A-LLM 的 decode_bound 假设",
  "tools_used": ["query_param_docs", "bo_suggest"],
  "reason": "BO 推荐 max_num_seqs=128（acquisition=0.42），但 memory 笔记显示该 workload >64 触发 preempt；采纳 BO 方向但下调到 64 作为渐进步骤",
  "expected_effect": {
    "throughput_tok_per_s": "+10% ~ +20%",
    "TTFT_p95_ms": "不变或 +5%",
    "TPOT_p95_ms": "不变",
    "risk": "preempt_rate 可能升高"
  },
  "rollback_if": "throughput 提升 <3% 或 preempt_rate > 3"
}
```

### 3.4 R-LLM：Reflector（每步复盘 + 写长期笔记）

**System Prompt**:

```
你是 vLLM 调参 agent 的"学习主脑"。每一步 agent 执行完 ConfigDelta 并测到新 trial 后，
你会收到改动前后的 trial 对比、原假设、以及该 trial 是否违反 SLO。你的任务:
1. 判断实际结果是否符合 P-LLM 的 expected_effect（accept / partial / reject）。
2. 产出一条可复用的长期笔记（自然语言，一句话），写入 memory.notes。
3. 建议下一步方向（explore 其他参数 / double-down 当前方向 / rollback）。

硬性规则:
- 笔记必须是参数级的通用规律，不是具体数字（好:"max_num_seqs 超过 64 时此 workload 触发 preempt"；
  坏:"trial_7 的吞吐是 203 tok/s"）。
- 如果最近 3 步 accept 率 <30%，建议 explore 新参数；如果 >70% 且 SLO 余量低，建议 stop。
- 只输出 JSON。
```

**User Prompt 模板**:

```
## P-LLM 的原假设
{proposal_json}

## 改动前 trial
{prev_trial_json}

## 改动后 trial
{new_trial_json}

## 约束检查结果（硬编码 Judge 给出）
{constraint_check_json}

## 当前 memory notes
{notes_list}

请输出 reflection JSON。
```

**输出 JSON Schema**:

```json
{
  "verdict": "accept | partial | reject",
  "reason": "throughput +14% 符合 expected_effect，preempt_rate 仍为 0",
  "new_note": "max_num_seqs 32→64 在 decode_heavy workload 上带来 +14% 吞吐且无 preempt",
  "next_move_hint": "double_down | explore_other | rollback | stop",
  "hint_detail": "继续试 max_num_seqs=128 以摸 preempt 上限"
}
```

### 3.5 Reporter（结束时 1 次）

**System Prompt**:

```
你是 vLLM 调参 agent 的报告撰写者。你会收到完整的 memory（所有 trial + 所有笔记）和 best_trial，
需要写一段中文自然语言报告（3–5 段），直接嵌入 final_report.md。要求:
1. 总结最优配置相对 baseline 的改动和效果；
2. 讲搜索过程中的关键转折（哪些 trial 让 agent 改变了方向）；
3. 指出 agent 未能覆盖的部分（未试过的参数组合、可能的更优方向）。

基于数据推理，不要编造数字。纯文本段落，不要 JSON。
```

**User Prompt**：`{baseline}` + `{full_memory}` + `{best_trial}`。

**输出**：纯 markdown 段落，直接粘贴到 `docs/final_report.md`。

### 3.6 API 成本估算（DeepSeek-Chat）

| 指标 | 值 |
|------|-----|
| 单步 tokens（A+P+R，含工具往返） | ~2000 |
| 单 run step 数（含收敛提前停） | 15–20 |
| 单 run 总 tokens | ~40k |
| Reporter | ~3k |
| **单 run 成本** | **~¥0.1** |
| 3 种 workload + 5 种消融 | ~¥1 |

### 3.7 安全与作弊防护

- **P-LLM 不能直接写 vLLM config**：它的输出必须是结构化 ConfigDelta，Judge 校验值域后才由程序执行。
- **工具不可伪造**：`query_param_docs` 返回的是内存中的 `ParamSpec` 对象序列化，不是 LLM 自己生成。
- **去重**：Judge 检查 `(param, value)` 是否与最近 3 步 rejected_proposals 冲突，冲突则反馈给 P-LLM 要求重试（最多重试 1 次）。
- **无限循环防护**：agent 主循环最多 `max_steps=25` 硬上限，超出立即 stop。

---

## 四、执行总览

```
Week 3: Module A — Workload Generator（已完成 ✅）
  ├── Task 3.1: workload 配置 schema + 预设
  ├── Task 3.2: prompt 语料池 30+
  ├── Task 3.3: 共享前缀池 5+
  ├── Task 3.4: WorkloadGenerator 类（三种到达模式）
  ├── Task 3.5: phase-switch 能力
  └── Task 3.6: 集成 run_benchmark.py

Week 4: Module D — Monitor + Data Pipeline（已完成 ✅）
  ├── Task 4.1: TPOT 逐 token 时间戳
  ├── Task 4.2: GPU 实时采样器
  ├── Task 4.3: vLLM /metrics 采集器
  ├── Task 4.4: 数据落盘重构
  ├── Task 4.5: 实验套件配置集
  └── Task 4.6: 运行初始实验集

Week 5: 平台迁移 — A6000 + Qwen3-8B
  ├── Task 5.1: A6000 环境与依赖安装
  ├── Task 5.2: Qwen3-8B 模型获取与量化选型（默认 bf16）
  ├── Task 5.3: configs / profiles / start_server.sh 适配
  ├── Task 5.4: tests 与 monitor 适配（76 tests 继续全绿）
  └── Task 5.5: A6000 + Qwen3-8B 新 Baseline

Week 6: VTA-Agent 基础设施（agent 框架 + 外围工具）
  ├── Task 6.1: VllmLauncher（热重启 + ready 探测 + 页缓存保留 + eager 透传）
  ├── Task 6.2: 扩展 metrics 采集 + 产物字段补齐（queue_time + summary 聚合 vllm 指标）
  ├── Task 6.3: TrialMetrics + Runner（单 trial 闭环 + 在线早停）
  ├── Task 6.4: ExperienceMemory + TrialRecord + summarize()
  ├── Task 6.5: ParamRegistry（RESTART/RUNTIME 分层 + candidates）
  ├── Task 6.6: ToolRegistry — A 只读工具（6 个 read-only）
  ├── Task 6.7: Optimizer — B 算法工具（bo_suggest / param_sensitivity / pareto_front / local_grid / cluster_workload_phases）
  └── Task 6.8: LlmClient（OpenAI 兼容 + function-calling + 重试/缓存/限速）

Week 7: VTA-Agent 闭环决策主脑
  ├── Task 7.1: A-LLM Diagnoser（prompt + 解析 + 单测）
  ├── Task 7.2: P-LLM Proposer（prompt + tool-calling + ConfigDelta 解析）
  ├── Task 7.3: R-LLM Reflector（prompt + note append + next_move hint）
  ├── Task 7.4: Judge（值域 / SLO 门 / rejected 去重 / 循环防护）
  └── Task 7.5: VtaAgent 主循环（Observe→Diagnose→Propose→Act→Reflect + 终止条件 + 失败回退）

Week 8: 对比实验与最终报告
  ├── Task 8.1: Reporter LLM + final_report.md 生成器
  ├── Task 8.2: run_tuning.sh 端到端 + 3 种 workload（prefill / decode / mixed）
  ├── Task 8.3: 5 组消融（random-proposer / no-memory / no-reflect / fixed-config / no-early-stop）
  └── Task 8.4: 3 张核心图 + final_report.md 定稿
```

---

## 五、Week 3 — Module A: Workload Generator

### Task 3.1: 设计 workload 配置模型

**目标**：定义 workload 配置格式，作为后续所有实验的统一入口。

**操作**：
1. 新建 `configs/workloads/workload_schema.json`，定义 JSON Schema
2. 新建 `configs/workloads/workload_baseline.json`，作为默认 workload 配置

workload 配置必须包含以下字段：

```json
{
  "workload": {
    "name": "baseline_workload",
    "seed": 42,
    "num_requests": 50,
    "warmup_requests": 5,
    "cooldown_requests": 5,
    "arrival": {
      "pattern": "burst",
      "rate": null,
      "options": {}
    },
    "prompt": {
      "source": "pool",
      "pool_file": "workloads/prompts_pool.json",
      "length_distribution": {
        "short": {"range": [20, 100], "weight": 0.3},
        "medium": {"range": [100, 500], "weight": 0.5},
        "long": {"range": [500, 1500], "weight": 0.2}
      }
    },
    "output": {
      "max_tokens_distribution": {
        "short": {"value": 64, "weight": 0.2},
        "medium": {"value": 256, "weight": 0.6},
        "long": {"value": 512, "weight": 0.2}
      }
    },
    "shared_prefix": {
      "enabled": false,
      "ratio": 0.0,
      "prefix_pool_file": "workloads/prefix_pool.json"
    },
    "phase_switch": {
      "enabled": false,
      "phases": []
    }
  }
}
```

3. 新建 4 个预设 workload 配置文件：
   - `configs/workloads/workload_burst.json` — arrival.pattern = "burst"
   - `configs/workloads/workload_poisson.json` — arrival.pattern = "poisson", rate = 2.0
   - `configs/workloads/workload_prefix.json` — shared_prefix.enabled = true, ratio = 0.5
   - `configs/workloads/workload_phase_switch.json` — phase_switch.enabled = true，30s 后从 burst 切换到 poisson

**产出物**：`configs/workloads/` 下 5 个新 JSON 文件 + 1 个 schema 文件

**验证**：`python3 -c "import json; json.load(open('configs/workloads/workload_baseline.json'))"` 无报错

---

### Task 3.2: 扩展 prompt 语料池

**目标**：把 prompt 数据源从固定 5 条升级为可按分布采样的语料池。

**操作**：
1. **保留** `benchmarks/prompts.json` 不动（兼容旧接口）
2. 新建 `workloads/prompts_pool.json`，格式如下：

```json
{
  "version": "1.0",
  "metadata": {
    "description": "Workload generator prompt pool",
    "total_prompts": 30
  },
  "prompts": [
    {
      "id": "short_01",
      "category": "short",
      "estimated_tokens": 30,
      "messages": [
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "什么是Python？"}
      ]
    }
  ]
}
```

要求：
   - 至少 30 条 prompt，覆盖 short(10条, 20-100 tokens)、medium(12条, 100-500 tokens)、long(8条, 500-1500 tokens)
   - 内容多样性：包含中文问答、代码生成、文本摘要、翻译、数学推理等场景
   - 每条标注 `estimated_tokens`（可用 tokenizer 离线估算，或手动估计）
   - 不要包含有害、敏感内容

**产出物**：`workloads/prompts_pool.json`

**验证**：`python3 -c "import json; d=json.load(open('workloads/prompts_pool.json')); print(len(d['prompts']), 'prompts')"` 输出 ≥30

---

### Task 3.3: 创建共享前缀池

**目标**：为共享前缀场景提供模板，测试 prefix caching 效果。

**操作**：
1. 新建 `workloads/prefix_pool.json`：

```json
{
  "version": "1.0",
  "prefixes": [
    {
      "id": "code_assistant",
      "system_message": "你是一个专业的Python编程助手，擅长代码审查、性能优化和调试。请用简洁准确的语言回答。",
      "user_prefix": "请帮我分析以下代码：\n",
      "estimated_prefix_tokens": 60,
      "suffix_pool": [
        "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)",
        "for i in range(1000000): result = sum(range(i))",
        "data = open('file.txt').read().split('\\n')"
      ]
    }
  ]
}
```

要求：
   - 至少 5 组前缀模板
   - 每组包含固定的 system_message + user_prefix + 至少 3 个 suffix
   - 标注 `estimated_prefix_tokens`

**产出物**：`workloads/prefix_pool.json`

**验证**：`python3 -c "import json; d=json.load(open('workloads/prefix_pool.json')); print(len(d['prefixes']), 'prefixes')"` 输出 ≥5

---

### Task 3.4: 实现三种到达模式

**目标**：新建 workload generator 模块，支持 burst、constant-rate、Poisson 三种请求到达模式。

**操作**：
1. 新建 `workloads/workload_generator.py`
2. 实现 `WorkloadGenerator` 类：

```python
class WorkloadGenerator:
    """根据 workload 配置生成请求序列及其调度时间"""

    def __init__(self, config: dict, seed: int = 42):
        """
        config: workload 配置字典（从 JSON 加载后的 "workload" 部分）
        seed: 随机种子，保证可复现
        """

    def generate(self) -> list[dict]:
        """
        生成完整的请求序列，每个请求包含：
        - request_id: str
        - scheduled_time_s: float  (相对于实验开始的调度时间)
        - messages: list[dict]
        - max_tokens: int
        - prompt_length_bucket: str  ("short"/"medium"/"long")
        - actual_prompt_tokens: int  (估算的 prompt token 数)
        - target_max_tokens: int
        - shared_prefix_group: str | None
        - metadata: dict
        """

    def _sample_arrival_times(self, n: int) -> list[float]:
        """根据 arrival pattern 生成 n 个到达时间"""
        # burst: 全部为 0.0
        # constant_rate: i / rate
        # poisson: 累加 exponential(1/rate) 间隔

    def _sample_prompt(self) -> tuple[list[dict], str, int]:
        """按长度分布从 pool 采样一条 prompt，返回 (messages, bucket, est_tokens)"""

    def _sample_max_tokens(self) -> int:
        """按 max_tokens_distribution 加权采样"""

    def _apply_shared_prefix(self, messages: list[dict]) -> tuple[list[dict], str | None]:
        """按共享前缀比例决定是否替换为前缀模板+随机后缀"""
```

3. 要求：
   - 同一 seed 下 `generate()` 输出完全一致（可复现）
   - 所有随机操作使用传入的 seed 初始化的 `random.Random` 实例，**不要**用全局 random
   - `_sample_prompt()` 根据 `length_distribution` 中各 bucket 的 weight 加权随机采样
   - 如果某 bucket 的 pool 中 prompt 不足，输出警告并从最近的 bucket 借用

**产出物**：`workloads/workload_generator.py`

**验证**：
```bash
cd /mnt/d/vlllm
python3 -c "
from workloads.workload_generator import WorkloadGenerator
import json
cfg = json.load(open('configs/workloads/workload_baseline.json'))
wg = WorkloadGenerator(cfg['workload'])
reqs = wg.generate()
print(f'{len(reqs)} requests generated')
print(f'First: id={reqs[0][\"request_id\"]}, time={reqs[0][\"scheduled_time_s\"]:.3f}s')
wg2 = WorkloadGenerator(cfg['workload'])
reqs2 = wg2.generate()
assert reqs[0]['request_id'] == reqs2[0]['request_id'], 'NOT REPRODUCIBLE'
print('Reproducibility: OK')
"
```

---

### Task 3.5: 实现 phase-switch 能力

**目标**：支持在实验运行中途按时间表切换负载场景，用于测试 controller 的适应能力。

**操作**：
1. 在 `workloads/workload_generator.py` 中扩展 `WorkloadGenerator`：
   - 新增 `_apply_phase_switch()` 方法
   - 当 `phase_switch.enabled = true` 时，按 `phases` 列表中的 `start_time_s` 在不同阶段使用不同的 arrival/prompt/output 配置
2. phase_switch 配置格式：

```json
{
  "phase_switch": {
    "enabled": true,
    "phases": [
      {
        "name": "warm_up",
        "start_time_s": 0,
        "arrival": {"pattern": "constant_rate", "rate": 1.0},
        "output": {"max_tokens_distribution": {"medium": {"value": 256, "weight": 1.0}}}
      },
      {
        "name": "burst_phase",
        "start_time_s": 30,
        "arrival": {"pattern": "burst"},
        "output": {"max_tokens_distribution": {"long": {"value": 512, "weight": 1.0}}}
      },
      {
        "name": "cool_down",
        "start_time_s": 60,
        "arrival": {"pattern": "constant_rate", "rate": 0.5}
      }
    ]
  }
}
```

3. 每个 phase 可以覆盖 arrival、prompt、output、shared_prefix 的部分字段，未指定的继承全局默认值
4. 在生成的请求中标注 `phase_name` 字段

**产出物**：`workloads/workload_generator.py` 的更新 + `configs/workloads/workload_phase_switch.json`

**验证**：
```bash
python3 -c "
from workloads.workload_generator import WorkloadGenerator
import json
from collections import Counter
cfg = json.load(open('configs/workloads/workload_phase_switch.json'))
wg = WorkloadGenerator(cfg['workload'])
reqs = wg.generate()
phases = Counter(r.get('phase_name', 'default') for r in reqs)
print('Phase distribution:', dict(phases))
assert len(phases) >= 2, 'Phase switch not working'
print('Phase switch: OK')
"
```

---

### Task 3.6: 集成到 run_benchmark.py

**目标**：改造 `run_benchmark.py`，让它使用 `WorkloadGenerator` 替代硬编码的请求循环。

**操作**：
1. 在 `benchmarks/run_benchmark.py` 中新增 `--workload` 参数，接受 workload 配置文件路径
2. 当 `--workload` 指定时：
   - 用 `WorkloadGenerator` 生成请求序列
   - 按 `scheduled_time_s` 字段控制请求注入时间（使用 `asyncio.sleep` 等待到调度时间）
   - 每个请求使用自带的 `messages` 和 `target_max_tokens`
   - `warmup_requests` 和 `cooldown_requests` 生成但标记 `is_warmup`/`is_cooldown`，统计时排除
3. 当 `--workload` 未指定时：保持原有行为不变（向后兼容）
4. 在结果 JSON 中新增 `workload_config` 字段，保存完整的 workload 配置快照
5. 在每个请求的 raw_result 中保留 workload generator 生成的所有元数据

**产出物**：`benchmarks/run_benchmark.py` 的修改

**验证**：
```bash
# 旧模式仍然工作
python3 benchmarks/run_benchmark.py --config configs/experiments/baseline_0.json --host 127.0.0.1 --num-requests 3
# 新模式也能工作
python3 benchmarks/run_benchmark.py --config configs/experiments/baseline_0.json --workload configs/workloads/workload_baseline.json --host 127.0.0.1
```

---

## 六、Week 4 — Module D: Monitor + Data Pipeline

### Task 4.1: 实现 TPOT 采集

**目标**：在流式响应中记录逐 token 到达时间，计算 TPOT（Time Per Output Token）。

**操作**：
1. 修改 `benchmarks/run_benchmark.py` 中 `send_request()` 函数
2. 在 SSE 处理循环中，为每个有效 `delta.content` 的 chunk 记录 `time.perf_counter()` 时间戳
3. 在 `result` dict 中新增：
   - `token_timestamps_ms`: list[float] — 每个 token 相对于请求发出的时间（毫秒）
   - `tpot_ms`: float — 平均 TPOT（首 token 之后各 token 间隔的均值）
   - `tpot_p95_ms`: float — TPOT P95
4. 在 `compute_stats()` 中新增全局 TPOT 聚合：
   - `tpot_ms.mean` / `.median` / `.p95` / `.p99`
5. 在 `format_summary()` 中新增 TPOT 展示段
6. `token_timestamps_ms` 数组较大，提供 `--save-token-timestamps` 参数控制是否落盘（默认不保存原始数组，只保存统计值）

**产出物**：`benchmarks/run_benchmark.py` 的修改

**验证**：运行 3 个请求的小规模测试，summary 中出现：
```
  输出Token间隔 (TPOT):
    Mean   : XX.X ms
    Median : XX.X ms
    P95    : XX.X ms
```

---

### Task 4.2: 实现 GPU 实时采样器

**目标**：在压测运行期间，后台线程定时采样 GPU 指标。

**操作**：
1. 新建 `monitors/gpu_monitor.py`
2. 实现 `GpuMonitor` 类：

```python
class GpuMonitor:
    """后台 daemon 线程定时采样 GPU 指标"""

    def __init__(self, interval_ms: int = 500, gpu_index: int = 0):
        """interval_ms: 采样间隔（毫秒），建议 500ms"""

    def start(self):
        """启动后台采样线程"""

    def stop(self) -> list[dict]:
        """
        停止采样并返回时序数据。
        每条记录：
        {
            "timestamp_s": float,    # 相对于 start() 时的秒数
            "gpu_util_pct": float,   # GPU 利用率 %
            "mem_used_mib": float,   # 已用显存 MiB
            "mem_total_mib": float,  # 总显存 MiB
            "mem_util_pct": float,   # 显存利用率 %
            "temperature_c": int,    # GPU 温度 ℃
            "power_w": float         # 功耗 W
        }
        """
```

3. 使用 `pynvml`（已在 requirements.txt 中）
4. 采样线程必须是 daemon 线程，不阻塞主进程退出
5. 处理 pynvml 初始化失败的情况（优雅降级，只打印警告）

**产出物**：`monitors/gpu_monitor.py`

**验证**：
```bash
python3 -c "
from monitors.gpu_monitor import GpuMonitor
import time
m = GpuMonitor(interval_ms=500)
m.start()
time.sleep(3)
samples = m.stop()
print(f'{len(samples)} samples collected')
if samples:
    s = samples[0]
    print(f'GPU util: {s[\"gpu_util_pct\"]}%, Mem: {s[\"mem_used_mib\"]}/{s[\"mem_total_mib\"]} MiB')
"
```

---

### Task 4.3: 实现 vLLM metrics 端点采集

**目标**：采集 vLLM 内部指标（队列长度、KV cache 使用率等）。

**操作**：
1. 先检查 vLLM 0.6.6 是否暴露 `/metrics` 端点：
   ```bash
   curl -s http://127.0.0.1:8000/metrics | head -50
   ```
2. 新建 `monitors/vllm_metrics_collector.py`：

```python
class VllmMetricsCollector:
    """后台线程定时采集 vLLM Prometheus metrics"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", interval_ms: int = 1000):
        pass

    def start(self):
        """启动后台采集线程"""

    def stop(self) -> list[dict]:
        """
        返回时序数据，每条记录至少包含：
        {
            "timestamp_s": float,
            "num_requests_running": int,
            "num_requests_waiting": int,
            "gpu_cache_usage_pct": float,
            "cpu_cache_usage_pct": float
        }
        """
```

3. 需要解析的关键 Prometheus 指标名：
   - `vllm:num_requests_running`
   - `vllm:num_requests_waiting`
   - `vllm:gpu_cache_usage_perc`
   - `vllm:cpu_cache_usage_perc`
   - `vllm:num_preemptions_total`
4. 如果 `/metrics` 不可用，实现降级方案：标记 `"source": "estimated"` 并在输出中显式标注

**产出物**：`monitors/vllm_metrics_collector.py`

**验证**：启动 vLLM 服务后测试采集 3 秒

---

### Task 4.4: 重构数据落盘结构

**目标**：每轮实验输出四类标准化产物到独立子目录。

**操作**：
1. 修改 `benchmarks/run_benchmark.py` 的结果保存逻辑
2. 集成 `GpuMonitor` 和 `VllmMetricsCollector`：
   - 在压测开始前启动两个采样器
   - 压测结束后停止并收集数据
3. 每次实验创建子目录：`results/<experiment_id>/`
4. 输出四类文件：

```
results/<experiment_id>/
├── config_snapshot.json     # 完整实验配置快照
├── request_trace.jsonl      # 请求级 trace（JSON Lines，每行一个请求）
├── metrics_timeseries.jsonl # 时序指标（GPU + vLLM metrics）
└── summary.json             # 聚合统计结果
```

5. `request_trace.jsonl` 每行格式：
```json
{
  "request_id": "req_001",
  "scheduled_time_s": 0.0,
  "dispatch_time_s": 0.012,
  "first_token_time_s": 0.075,
  "complete_time_s": 2.850,
  "success": true,
  "ttft_ms": 62.5,
  "tpot_ms": 12.3,
  "latency_ms": 2838.0,
  "output_tokens": 210,
  "output_tokens_source": "usage",
  "prompt_length_bucket": "medium",
  "target_max_tokens": 256,
  "shared_prefix_group": null,
  "phase_name": "default",
  "is_warmup": false,
  "error": null
}
```

6. `metrics_timeseries.jsonl` 合并 GPU 和 vLLM metrics：
```json
{"timestamp_s": 0.5, "source": "gpu", "gpu_util_pct": 85.0, "mem_used_mib": 5200}
{"timestamp_s": 1.0, "source": "vllm", "num_requests_running": 1, "gpu_cache_usage_pct": 0.45}
```

7. 同时保留旧的 `results/benchmark_<timestamp>.txt` 人可读摘要（向后兼容）

**产出物**：`benchmarks/run_benchmark.py` 的修改

**验证**：运行一次小规模压测后，检查 `results/` 下生成了正确的目录结构和四类文件

---

### Task 4.5: 实验套件配置集

**目标**：建立三组单因子对照实验的配置集。

**操作**：
1. 新建以下 workload 配置文件：

**到达率轴**：
- `configs/workloads/workload_rate2.json` — constant_rate = 2 req/s
- `configs/workloads/workload_rate4.json` — constant_rate = 4 req/s
- `configs/workloads/workload_poisson2.json` — Poisson, lambda = 2

**长度分布轴**：
- `configs/workloads/workload_short_only.json` — 100% short prompts, max_tokens=64
- `configs/workloads/workload_long_only.json` — 100% long prompts, max_tokens=512
- `configs/workloads/workload_mixed.json` — 混合分布（baseline 的默认分布）

**共享前缀轴**：
- `configs/workloads/workload_prefix_0.json` — shared_prefix.ratio = 0.0
- `configs/workloads/workload_prefix_50.json` — shared_prefix.ratio = 0.5
- `configs/workloads/workload_prefix_90.json` — shared_prefix.ratio = 0.9

2. 所有配置固定 seed=42, num_requests=50
3. 新建 `scripts/experiment/run_experiment_suite.sh`，按顺序运行实验并等待完成：

```bash
#!/bin/bash
# 使用方法: bash scripts/experiment/run_experiment_suite.sh [axis]
# axis: arrival | length | prefix | all
```

**产出物**：9 个配置文件 + 1 个批量运行脚本

**验证**：`python3 -c "import json; json.load(open('configs/workloads/workload_rate2.json'))"` 无报错

---

### Task 4.6: 运行初始实验集

**目标**：获取 baseline 数据用于后续 LLM prompt 的 few-shot 示例。

**操作**：
1. 确保 vLLM 服务在运行
2. 运行以下 3 个具有代表性的实验：
   - `workload_baseline.json`（burst 模式，混合长度）
   - `workload_rate2.json`（constant rate 2 req/s）
   - `workload_long_only.json`（全长 prompt）
3. 每个实验至少 50 个请求
4. 收集产出物到 `results/` 下
5. 从三组实验中提取 3-5 个典型指标快照，作为后续 LLM prompt 的 few-shot 示例数据
6. 将提取的示例数据保存到 `configs/llm_prompts/few_shot_examples.json`

**产出物**：
- `results/` 下 3 组实验数据
- `configs/llm_prompts/few_shot_examples.json`

**验证**：三组实验均产出完整的 4 类文件；`few_shot_examples.json` 至少 3 个示例

---

## 七、Week 5 — 平台迁移：A6000 + Qwen3-8B

> **本周目标**：把现有项目从 RTX 4060 Laptop (8 GB) 迁到 A6000 (48 GB)，模型从 Qwen2.5-3B-Instruct-AWQ 换为 Qwen3-8B，重建 baseline 并验证所有既有模块（workloads / monitors / benchmarks / tests）在新平台上可用。
>
> **重要前提**：此前基于 "proxy 在线控制" 的设计（旧 Week 5–8）全部作废。新的 Week 6–8 采用 **VTA-Agent**（离线重启调参 agent）方案，见 §八–§十。

### Task 5.1: A6000 环境与依赖安装

**目标**：在 A6000 主机上搭好 WSL2/Linux + CUDA + vLLM 环境，版本与 4060 侧保持一致以减少差异。

**操作**：
1. 登录 A6000 主机，确认硬件：
   ```bash
   nvidia-smi                   # 确认 A6000 48 GB, 驱动 ≥ 550, CUDA 12.4
   free -h && lscpu             # 确认内存 ≥ 64 GB
   ```
2. 建立独立的 Python 3.10 venv：`python3.10 -m venv ~/vllm-venv-a6000 && source ~/vllm-venv-a6000/bin/activate`
3. 安装依赖，版本与 `requirements.txt` 对齐：
   - `pip install "vllm==0.6.6.post1" "torch==2.5.1" "transformers==4.49.0"`
   - `pip install -r requirements.txt`
4. 运行 `python3 scripts/tools/collect_sysinfo.py`，产出物保存为 `docs/env_a6000.md`（和原 4060 的 `env_4060.md` 并列保留）。
5. 在仓库根新建 `docs/migration_a6000.md` 记录：驱动版本、CUDA、Python、vLLM 及安装过程中的任何坑点。

**产出物**：`~/vllm-venv-a6000/`、`docs/env_a6000.md`、`docs/migration_a6000.md`

**验证**：
```bash
python3 -c "import torch, vllm; print(torch.cuda.get_device_name(0), vllm.__version__)"
# 期望输出: NVIDIA RTX A6000 0.6.6.post1
```

---

### Task 5.2: Qwen3-8B 模型获取与量化选型

**目标**：选定 Qwen3-8B 的发布版本与量化精度，下载模型到 `~/models/`。

**选型决策**（写入 `docs/model_choice_qwen3_8b.md`）：

| 候选 | 权重大小 | 预估 KV cache 余量（A6000 48 GB）| 适用场景 |
|------|----------|-----------------------------------|----------|
| Qwen3-8B bf16 | ~16 GB | ~28 GB | **本项目默认**（A6000 显存充足，精度无损，避免量化带来的性能噪声） |
| Qwen3-8B-AWQ (4-bit) | ~5 GB | ~39 GB | 想放大 `max_num_seqs` 看 KV 压力场景时备用 |
| Qwen3-8B-GPTQ-Int8 | ~9 GB | ~34 GB | 仅做对照备用 |

**决策**：**默认使用 Qwen3-8B bf16**。理由：
1. A6000 48 GB 显存足够，无需量化换空间；
2. 避免量化 kernel（AWQ/GPTQ）与 chunked prefill、partial prefills 的未知交互；
3. 作为调参实验的基准，权重精度对调参相对影响 **必须可复现**，bf16 是最干净的控制变量。

**操作**：
1. 下载模型到 `~/models/Qwen3-8B`（使用 `scripts/setup/download_model.sh`，修改默认模型 ID）：
   ```bash
   huggingface-cli download Qwen/Qwen3-8B --local-dir ~/models/Qwen3-8B
   ```
2. 校验：
   ```bash
   ls ~/models/Qwen3-8B/config.json
   python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('~/models/Qwen3-8B'); print(c.model_type, c.hidden_size, c.num_hidden_layers)"
   ```

**产出物**：`~/models/Qwen3-8B/`、`docs/model_choice_qwen3_8b.md`

---

### Task 5.3: 配置与 Profile 适配

**目标**：把所有 `configs/` 下的模型名、端口、上下文长度等参数从 3B-AWQ 替换为 Qwen3-8B，并重新定义 baseline 与 profile。

**操作**：
1. 新建 `configs/experiments/baseline_a6000_0.json`，关键字段：
   ```json
   {
     "experiment": {"name": "baseline_a6000_0", "description": "A6000 + Qwen3-8B 首个 baseline"},
     "model": {"name": "Qwen/Qwen3-8B", "dtype": "bfloat16", "quantization": null, "max_model_len": 4096, "trust_remote_code": true},
     "server": {"host": "0.0.0.0", "port": 8000, "tensor_parallel_size": 1, "gpu_memory_utilization": 0.90, "max_num_seqs": 64, "enforce_eager": false, "api_type": "openai"},
     "sampling": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 256, "stop": null},
     "benchmark": {"num_requests": 50, "concurrency": 1}
   }
   ```
2. 新建 `configs/profiles/profile_{L,B,T}.json`（替换旧的 3B 版 profile 概念，新 profile 针对 A6000 + 8B 重新标定）：
   - **L (低延迟)**：`max_num_seqs=32, max_model_len=2048, enable_prefix_caching=false, enable_chunked_prefill=true`
   - **B (平衡，默认)**：`max_num_seqs=64, max_model_len=4096, enable_prefix_caching=false, enable_chunked_prefill=true`
   - **T (高吞吐)**：`max_num_seqs=128, max_model_len=4096, enable_prefix_caching=true, enable_chunked_prefill=true`
3. 修改 `scripts/server/start_server.sh`：
   - 默认 config 指向 `configs/experiments/baseline_a6000_0.json`；
   - 模型路径解析时，`~/models/Qwen3-8B` 命中则走离线；
   - 增加 `--enforce-eager` 可选透传（`$2` 为 `eager` 时追加该 flag），供 Week 6 粗搜调用。
4. 更新 `configs/workloads/*.json` 中依赖上下文长度的字段：
   - prompt 长度上限从 1024 放宽到 2048（8B 能处理更长上下文）；
   - `max_tokens` 默认从 256 提到 512；
   - 保留原 14 份 workload 预设（到达/长度/前缀三轴），仅调整长度范围。

**产出物**：`configs/experiments/baseline_a6000_0.json`、`configs/profiles/profile_{L,B,T}.json`、改动后的 `scripts/server/start_server.sh`、14 份 workload 更新

**验证**：`bash scripts/server/start_server.sh configs/experiments/baseline_a6000_0.json`，观察 `/health` 返回 200，`/metrics` 含 `vllm:cache_config_info{model_name="/home/.../Qwen3-8B"}` 行。

---

### Task 5.4: 测试与监控适配

**目标**：确保 76 个既有单元测试在新平台 + 新模型上全部通过。

**操作**：
1. 全仓库搜索硬编码字符串 `Qwen2.5-3B-Instruct-AWQ`、`max_model_len.*2048`、`max_num_seqs.*32`，凡是作为测试 fixture 默认值的都改为 Qwen3-8B 的对应值；
2. `monitors/vllm_metrics_collector.py` 的指标名校验若有硬编码 `model_name` 匹配，改为通配或参数化；
3. `tests/` 下涉及到启动 vLLM 的集成测试若用到真实模型，统一通过 fixture `model_path` 注入，默认 `~/models/Qwen3-8B`；
4. 在 `tests/conftest.py`（若无则新建）中加入一个 `skip_if_no_gpu` marker，跳过没有 GPU 的 CI 环境。

**产出物**：更新后的 `tests/`、`tests/conftest.py`

**验证**：
```bash
bash scripts/test/run_all_tests.sh
# 期望: 76 passed（如有因 8B 模型加载时间超过 pytest timeout 而失败的，调大 timeout，不允许 skip）
```

---

### Task 5.5: A6000 + Qwen3-8B 新 Baseline

**目标**：在新平台上重新跑一次 50 请求、concurrency=1 的冷启动 baseline，作为后续调参的对照基准。

**操作**：
1. 启动服务：`bash scripts/server/start_server.sh configs/experiments/baseline_a6000_0.json`；
2. 运行压测：
   ```bash
   python3 benchmarks/run_benchmark.py \
     --config configs/experiments/baseline_a6000_0.json \
     --workload configs/workloads/workload_baseline.json \
     --host 127.0.0.1
   ```
3. 记录 `results/<experiment_id>/summary.json` 中的：
   - `throughput_req_per_s`, `throughput_tokens_per_s`
   - `ttft_ms` (mean/p95)
   - `tpot_ms` (mean/p95)
   - `latency_ms` (mean/p95/p99)
   - `preemption_total`（从 metrics_timeseries 推出，Week 6 会正式纳入）
4. 在 `_issue_body.md` 中追加 **"Baseline A6000-0"** 条目，格式与 Baseline 0 一致。

**产出物**：`results/baseline_a6000_0/`、`_issue_body.md` 更新

**验证**：新 baseline 吞吐 ≥ 3B-AWQ baseline 的 2× 以上（8B 模型在 A6000 上 token/s 预期 >150），P95 TTFT < 300 ms。

---

## 八、Week 6 — VTA-Agent 基础设施（框架 + 工具 + LLM 客户端）

> **本周目标**：把 agent 所需的**所有"可被 LLM 调用或被程序调用的"基础设施**全部打通，包括热重启、指标、内存、工具注册、LLM 客户端。本周 **不写 agent 主循环**（留到 Week 7），目标是一个 trial 能被 runner 闭环跑通，且 LlmClient 能完成一次带工具调用的对话。

### Task 6.1: VllmLauncher（热重启 + ready + eager 透传 + 页缓存保留）

**目标**：把一份参数字典 → 重启 vLLM + 就绪检查封装成原子调用，并尽量压低重启墙钟。

**操作**：
1. 新建 `tuner/launcher.py`：
   ```python
   class VllmLauncher:
       def __init__(self, keep_page_cache: bool = True, log_dir: Path = ...): ...
       def start(self, config_path: Path, enforce_eager: bool = False,
                 ready_timeout_s: int = 120) -> LaunchResult:
           """nohup 启动 + 轮询 /health；超时抛 TimeoutError。"""
       def stop(self, grace_s: int = 10):
           """SIGTERM→SIGKILL；等端口释放；**不** drop_caches（保页缓存）。"""
       def restart(self, config_path, enforce_eager=False) -> LaunchResult: ...
   ```
2. 新建 `tuner/config_generator.py`：`render_experiment_config(base, overrides)` 深拷贝 + 点路径覆写，`write_temp_config` 落到 `/tmp/vllm_trial_<uuid>.json`。
3. `LaunchResult` 记录 `restart_wall_time_s`、`model_load_time_s`、`cuda_graph_capture_time_s`（从 log 正则抓，缺失置 None）。
4. `scripts/server/start_server.sh` 新增 `ENFORCE_EAGER=1` 环境变量透传 `--enforce-eager`。

**验证**：eager 热启动 ≤ 30 s，graph 热启动 ≤ 60 s；两次 restart 间页缓存保留（观察 `free -m` 的 cached 列）。

---

### Task 6.2: 扩展 metrics 采集 + 产物字段补齐（Week 4 收尾）

**目标**：在已有 [monitors/vllm_metrics_collector.py](monitors/vllm_metrics_collector.py) 与 [benchmarks/run_benchmark.py](benchmarks/run_benchmark.py) 基础上**只补缺口**，不改职责边界——只让"队列等待时间"指标进得来，并把已采但未聚合的 vllm 指标写进 `summary.json`。**TrialMetrics dataclass 与从产物聚合的逻辑都不在本任务，留到 Task 6.3。**

**现状盘点**（无需重做）：
- `VllmMetricsCollector` 已采 `num_requests_running / num_requests_waiting / gpu_cache_usage_pct / cpu_cache_usage_pct / num_preemptions_total`。
- `run_benchmark.py` 已落盘 `summary.json` / `request_trace.jsonl` / `metrics_timeseries.jsonl` / `config_snapshot.json`，已聚合 ttft/tpot/latency/throughput/token_throughput。

**操作**：
1. **`monitors/vllm_metrics_collector.py`**：在 `_METRIC_PATTERNS` 中加一条 `queue_time_seconds_sum`，对应 `vllm:time_in_queue_requests_seconds_sum`（counter，单位秒）。其它 4 条已存在，**不要重复添加**。
2. **`benchmarks/run_benchmark.py compute_stats()`**：收尾处增加一段"vLLM 时序 → summary 聚合"，输入新增参数 `vllm_samples: list[dict] | None`，写入：
   - `preemptions_total = max(num_preemptions_total) - min(...)`（counter 差分）
   - `preemption_rate_per_min = preemptions_total / wall_time_s * 60`
   - `kv_cache_usage_p95_pct`：对 `gpu_cache_usage_pct` 取 P95
   - `queue_time_delta_s`：`max(queue_time_seconds_sum) - min(...)`（供 6.3 算 P95，需要每请求队列时间则在 6.3 阶段从 `request_trace` 推导）
   - `vllm_metrics_available: bool`：当所有样本 `source=="unavailable"` 时为 False
3. `format_summary()` 同步追加一段"vLLM 内部指标"展示。
4. 现有 `tests/test_vllm_metrics_collector.py` 加一个 case：mock 一份含 `vllm:time_in_queue_requests_seconds_sum 12.345` 的响应，验证字段被解析。

**验证**：
- 对一次新 baseline 跑 50 请求，`summary.json` 新增 5 个字段全部非空且数值合理（`preemption_rate_per_min ≥ 0`、`kv_cache_usage_p95_pct ∈ [0, 1]`）。
- `metrics_timeseries.jsonl` 中 `source=="vllm"` 的行包含 `queue_time_seconds_sum` 键。

---

### Task 6.3: TrialMetrics + Runner（单 trial 闭环 + 在线早停）

**目标**：把 6.1 的 `VllmLauncher` + 现成的 `run_benchmark.py` 串成"一个函数调一次 = 一个 trial"，并将落盘产物归一化成 `TrialMetrics` 供 agent 使用。

**操作**：
1. **新建 `tuner/metrics_parser.py`**：从 `results/<exp_id>/` 目录解析（不重新计算压测）：
   ```python
   @dataclass
   class TrialMetrics:
       throughput_req_per_s: float
       throughput_tok_per_s: float
       ttft_p95_ms: float
       tpot_p95_ms: float
       latency_p95_ms: float
       preemptions_total: int
       preemption_rate_per_min: float
       kv_cache_usage_p95_pct: float
       queue_time_p95_ms: float            # 由 metrics_parser 从 request_trace.jsonl 派生
       success: bool
       early_killed: bool
       wall_time_s: float

   def parse_trial(exp_dir: Path, *, early_killed: bool, wall_time_s: float) -> TrialMetrics:
       """读 summary.json + request_trace.jsonl，组装 TrialMetrics。
       summary 已有的字段直接搬运；queue_time_p95_ms 由请求级 (dispatch - scheduled) 取 P95。"""
   ```
   说明：把"按字段重命名 / 取 P95 / 兜底缺失值"的全部脏活集中在这里，run_benchmark 不再背负 trial-level 语义。
2. **新建 `tuner/runner.py`**：
   ```python
   def run_trial(config_overrides: dict, workload_path: Path,
                 enforce_eager: bool, bench_duration_s: int,
                 early_stop_cfg: dict | None,
                 launcher: VllmLauncher) -> TrialMetrics:
       """
       1. render_experiment_config + write_temp_config
       2. launcher.restart(config_path, enforce_eager)
       3. subprocess.run(['python','benchmarks/run_benchmark.py',
                          '--config', cfg, '--workload', wl, '--host', '127.0.0.1'])
          —— 复用 Week 4 已验证的压测路径，不重写
       4. 期间另起轻量轮询线程，每 5s 抓一次 /metrics，命中早停规则 → SIGTERM 子进程 + early_killed=True
       5. parse_trial(results/<exp_id>) → TrialMetrics
       6. launcher.stop()
       """
   ```
3. **早停规则**（默认）：启动 ≥ 20 s 后任一命中即终止——`preempt_rate > 2/s` / `throughput_tok_per_s < baseline*0.5` / `kv_cache_usage_pct > 0.98` 连续 3 次。

**验证**：
- baseline + `workload_long_only` 跑 30s：`run_trial` 返回 `early_killed=False`，全部字段非空，并能在 `results/baseline_a6000_0/` 上独立调 `parse_trial()` 复现同一 `TrialMetrics`（验证 parser 解耦）。
- 手工把 `max_num_seqs=256, max_num_batched_tokens=1024` → 应在 ≤ 60 s 内 `early_killed=True`。

---

### Task 6.4: ExperienceMemory + TrialRecord + summarize()

**目标**：让 agent 能在 20 步内不"失忆"——写入 / 查询 / 摘要三件事。

**操作**：新建 `tuner/memory.py`：
```python
@dataclass
class TrialRecord:
    trial_id: int
    config: dict                # 完整 config
    delta_from_prev: dict       # {param: (old, new)}
    hypothesis: str             # A-LLM 的 hypothesis + P-LLM 的 expected_effect
    metrics: TrialMetrics
    score: float
    constraint_violations: list[str]
    reflection: str             # R-LLM verdict + reason
    llm_source: str             # "llm" | "fallback"
    timestamp: float

class ExperienceMemory:
    def __init__(self, baseline: TrialMetrics): ...
    def append(self, record: TrialRecord) -> None: ...
    def best(self) -> TrialRecord | None: ...
    def recent(self, n: int) -> list[TrialRecord]: ...
    def rejected_proposals(self, n: int = 3) -> list[dict]:
        """最近 n 条 reject 的 (param, value)。"""
    def add_note(self, note: str) -> None: ...
    def summarize(self, top_k: int = 3, recent_n: int = 5) -> dict:
        """给 LLM 看的紧凑视图: {baseline, top_k_best, recent_n, notes,
                                  tried_values_per_param, rejected}。"""
    def dump(self, path: Path) -> None: ...   # JSONL 落盘
```

**验证**：构造 10 条伪 TrialRecord，`summarize(top_k=3, recent_n=5)` 输出 token 数 ≤ 1500（tiktoken 估算），`tried_values_per_param` 正确去重。

---

### Task 6.5: ParamRegistry（RESTART / RUNTIME 分层）

**目标**：代码化"哪些参数可调、候选、先验说明"——`query_param_docs` 工具的数据源。

**操作**：新建 `tuner/param_registry.py`：
```python
class MutationClass(Enum):
    RESTART = "restart"; RUNTIME = "runtime"

@dataclass
class ParamSpec:
    name: str
    mutation_class: MutationClass
    candidates: list
    default: object
    affects: list[str]        # ["throughput","ttft","tpot","kv","preempt"]
    notes: str                # 一句话先验
    depends_on: list[str] = field(default_factory=list)

REGISTRY: dict[str, ParamSpec] = {
    "max_num_batched_tokens": ParamSpec(..., RESTART, [2048,4096,8192,16384], 4096,
        ["throughput","ttft","tpot"], "大→prefill 吞吐好，小→decode TPOT 好"),
    "max_num_seqs":           ParamSpec(..., RESTART, [16,32,64,128], 64,
        ["throughput","preempt","kv"], "decode 并发上限；大→吞吐高但 preempt/KV 压力升"),
    "gpu_memory_utilization": ParamSpec(..., RESTART, [0.85,0.90,0.93,0.95], 0.90, ...),
    "max_model_len":          ParamSpec(..., RESTART, ["auto","p95","p99"], "auto", ...),
    "enable_prefix_caching":  ParamSpec(..., RESTART, [False,True], False, ...),
    "enable_chunked_prefill": ParamSpec(..., RESTART, [False,True], True, ...),
    "max_num_partial_prefills":    ParamSpec(..., RESTART, [1,2,4], 1, ...),
    "max_long_partial_prefills":   ParamSpec(..., RESTART, [1,2], 1, ...),
    "long_prefill_token_threshold":ParamSpec(..., RESTART, ["p75","p90"], "p90", ...),
    # RUNTIME: 仅 reset_prefix_cache，由 agent 在 accept 新 config 前触发
}
```

**验证**：`REGISTRY` 恰有 9 条 RESTART；`candidates` 不含 registry 默认值的非法值（如负数）。

---

### Task 6.6: ToolRegistry — A 类只读工具（6 个）

**目标**：给 P-LLM 定义一套**安全只读**的查询接口；同时充当所有工具（A 只读 + B 算法）的统一调度入口。

**操作**：新建 `tuner/tools.py`：
```python
class ToolRegistry:
    def __init__(self, registry: dict, memory: ExperienceMemory,
                 optimizer: "Optimizer | None" = None): ...

    # ===== A. 只读工具 =====
    def list_params(self) -> list[dict]: ...
    def query_param_docs(self, name: str) -> dict:
        """ParamSpec + 该参数在 memory 中被试过的 (value, score) 对。"""
    def list_tried_configs(self, top_k: int = 5) -> list[dict]: ...
    def compare_trials(self, a_id: int, b_id: int) -> dict:
        """{params_diff, metrics_diff, score_diff}。"""
    def get_baseline(self) -> dict: ...
    def get_memory_notes(self) -> list[str]: ...

    # ===== 给 LlmClient 用 =====
    def openai_tools_schema(self) -> list[dict]:
        """A + B 全部工具的 OpenAI function-calling JSON schema。"""
    def dispatch(self, name: str, args: dict) -> dict:
        """LLM 返回 tool_call 时查表调用；未知 name → UnknownToolError。
        每次调用都向 llm_calls.jsonl 追加一条 {tool, args, result_summary}。"""
```

**安全**：`apply_config / rollback / restart` **绝不**进入 ToolRegistry；LLM 无法通过任何工具直接执行副作用。

**验证**：`openai_tools_schema()` 能被 `openai` SDK 直接接受；`dispatch("query_param_docs", {"name": "max_num_seqs"})` 返回含 `candidates` 字段；`dispatch("apply_config", ...)` 抛 `UnknownToolError`。

---

### Task 6.7: Optimizer — B 类算法工具（LLM 驱动的优化算法）

**目标**：把 BO / 敏感度 / Pareto / 局部网格 / 时序聚类封装成 LLM 可调用工具。LLM 决定何时触发，算法负责数值密集子任务。

**操作**：新建 `tuner/optimizer.py`：
```python
class Optimizer:
    def __init__(self, memory: ExperienceMemory, registry: dict):
        self.memory = memory
        self.registry = registry

    def bo_suggest(self, scope: list[str], n: int = 1,
                   objective: str = "score",
                   min_history: int = 4) -> list[dict]:
        """Optuna TPE：把 memory 中所有 trial enqueue 后 ask n 个候选。
        返回 [{config, expected_score, acquisition, uncertainty}, ...]。
        memory < min_history 时返回 [] + insufficient_data=True。"""

    def param_sensitivity(self) -> list[dict]:
        """对每个 param 计算: score 方差 / Spearman ρ / 单调性。
        按敏感度降序返回。memory <4 返回 insufficient_data=True。"""

    def pareto_front(self, metrics: list[str] = ("throughput_tok_per_s", "latency_p95_ms")) -> list[dict]:
        """二/三目标朴素 O(n²) 非支配集；throughput 越大越好，latency 越小越好。"""

    def local_grid(self, param: str, neighborhood: int = 2) -> list:
        """以当前 best trial 的 param 值为中心，从 candidates 中
        取距离 ≤ neighborhood 的、未试过的候选值。"""

    def cluster_workload_phases(self, window_s: int = 30) -> dict:
        """读 monitors 落盘的 metrics_timeseries.jsonl，
        对 throughput 时序做 1D KMeans / change-point 检测，
        返回 {phases: [{start_s, end_s, fingerprint}], n_phases}。"""
```

ToolRegistry 在初始化时把 `Optimizer` 实例的 5 个方法也注册进 `dispatch`，并合并到 `openai_tools_schema()`。

**实现要点**：
- `bo_suggest` 仅在 `scope` 内的参数搜索；其他参数用当前 current_config 的值固定（用 `study.enqueue_trial(fixed)` 传给 TPE）。
- 所有工具均为**纯函数**：不写 vLLM 配置、不重启服务、不改动 memory（除日志）。
- `acquisition` 估计：拿 TPE 的 PDF 比 `l(x)/g(x)` 作为代理；`uncertainty` 取候选周围未观测体积的简易估计。

**验证**：
- 喂入 6 条合成 trial，`bo_suggest(scope=["max_num_seqs","max_num_batched_tokens"], n=2)` 返回 2 个 config，且都在候选集内、不重复历史。
- `param_sensitivity()` 对 6 条 trial 给出非空排序，最敏感参数与人工预期一致。
- `pareto_front(metrics=["throughput_tok_per_s","latency_p95_ms"])` 不返回任何被支配 trial。
- `local_grid("max_num_seqs", 2)` 在 best=64 时返回 {32, 128}（不含 64）。

---

### Task 6.8: LlmClient（OpenAI 兼容 + function-calling + 重试 + 缓存 + 限速）

**目标**：唯一的 LLM 通道，A/P/R/Reporter 都走它。

**操作**：新建 `llm_advisor/llm_client.py`：
```python
class LlmClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 rate_limit_qps: float = 1.0, cache_ttl_s: int = 60): ...

    def chat(self, system: str, user: str,
             response_format: str = "json",         # "json" 时 Prompt 末尾加 "仅输出 JSON" 保险
             tools: list[dict] | None = None,
             tool_handler: Callable[[str, dict], dict] | None = None,
             max_tool_rounds: int = 5,             # A+B 工具组合可能走多轮
             temperature: float = 0.2) -> LlmResponse:
        """
        完整执行 function-calling 多轮对话:
        - 若 tools 非空: 每收到 tool_calls 就用 tool_handler 解析并回填 role=tool
        - 最终 assistant content 解析为 JSON (若 response_format="json")
        - 3 次指数退避重试; 同 (system+user+tools hash) 60s TTL 缓存
        - 令牌 / 成本累计写入 LlmResponse.usage（含 tool_calls 计数）
        """
```

**降级**：任何一次 `chat` 失败最终抛 `LlmUnavailable`；调用方（agent）捕获后走 fallback 路径。

**配置**：`.env` 支持 `LLM_PROVIDER=deepseek|openai`、`LLM_MODEL`、`LLM_API_KEY`、`LLM_BASE_URL`。

**验证**：
- 对 `system="回复 OK"` 一次 chat 返回 "OK"；
- 注入一个 `echo(msg) -> msg` 工具，让 LLM 必须调用才能回答，验证 tool_calls 多轮对话能正确闭环；
- 同样请求第二次命中缓存（`LlmResponse.cache_hit=True`）。

---

## 九、Week 7 — VTA-Agent 闭环决策主脑

> **本周目标**：把 Week 6 的工具 + 内存 + LlmClient 串起来，实装 A-LLM / P-LLM / R-LLM 三个决策角色 + Judge 安全网 + VtaAgent 主循环。本周跑通一种 workload（`decode_heavy`）的 end-to-end。

### Task 7.1: A-LLM Diagnoser

**目标**：输入最新 trial + memory 摘要，输出 diagnosis JSON。

**操作**：新建 `llm_advisor/prompts.py` + `llm_advisor/diagnoser.py`：
```python
# prompts.py
A_LLM_SYSTEM = """..."""    # 见 §3.2
A_LLM_USER_TMPL = """..."""

# diagnoser.py
def diagnose(client: LlmClient, memory: ExperienceMemory,
             latest: TrialRecord) -> DiagnosisResult:
    """
    1. 渲染 user prompt = A_LLM_USER_TMPL.format(...)
    2. client.chat(A_LLM_SYSTEM, user, response_format="json") -> dict
    3. 校验字段 bottleneck/confidence/hypothesis 存在；非法值抛 ParseError
    4. 返回 DiagnosisResult
    """
```

`DiagnosisResult` 含 `bottleneck / confidence / evidence / hypothesis / slo_pressure / should_stop`。

**Fallback**：LLM 失败 → 规则版（preempt_rate>5 → `preempt_storm`，kv>95% → `kv_cache_pressure`，否则 `underutilized`）。

**验证**：构造 3 种合成 trial（preempt 爆/吞吐低/SLO 紧），diagnoser 输出的 `bottleneck` 与预期一致 ≥ 2/3（用 fallback 分支做 golden）。

---

### Task 7.2: P-LLM Proposer（带工具调用）

**目标**：输入 diagnosis + memory，输出 ConfigDelta。

**操作**：新建 `llm_advisor/proposer.py`：
```python
def propose(client: LlmClient, tools: ToolRegistry,
            memory: ExperienceMemory,
            diagnosis: DiagnosisResult,
            current_config: dict) -> ConfigDelta | StopSignal:
    resp = client.chat(
        system=P_LLM_SYSTEM,
        user=P_LLM_USER_TMPL.format(...),
        tools=tools.openai_tools_schema(),
        tool_handler=tools.dispatch,
        max_tool_rounds=3,
    )
    raw = json.loads(resp.content)
    if raw["action"] == "stop":
        return StopSignal(reason=raw["reason"])
    # 字段校验 + 值域校验 + rejected 去重预检
    return ConfigDelta(param=raw["param"], old_value=raw["old_value"],
                       new_value=raw["new_value"], hypothesis_ref=...,
                       expected_effect=raw["expected_effect"],
                       rollback_if=raw.get("rollback_if"))
```

**Fallback**：LLM 失败 → 规则 proposer：在未试过的参数中按优先级 `[max_num_seqs, max_num_batched_tokens, gpu_memory_utilization, ...]` 选第一个未试过的 candidate 中位数。

**验证**：P-LLM 至少调用一次 `query_param_docs`（观察 `LlmResponse.usage.tool_calls≥1`）；输出的 `new_value` 必在 `REGISTRY[param].candidates` 内。

---

### Task 7.3: R-LLM Reflector

**目标**：写复盘 + 产出可复用的长期笔记。

**操作**：新建 `llm_advisor/reflector.py`：
```python
def reflect(client: LlmClient,
            proposal: ConfigDelta,
            prev: TrialRecord, new: TrialRecord,
            constraint_check: dict,
            notes: list[str]) -> ReflectionResult:
    ...
```

`ReflectionResult` 含 `verdict / reason / new_note / next_move_hint / hint_detail`。执行后：
- `memory.add_note(result.new_note)`；
- `new.reflection = result.reason`；
- `next_move_hint` 作为下一步 A/P 的提示（写入 user prompt 的 "上一步复盘" 字段）。

**Fallback**：LLM 失败 → 规则：score 提升 > 3% 且无 SLO 违反 → `accept/double_down`；否则 `reject/explore_other`；note 用模板填空。

**验证**：给定已知的 "accept/improve 14%" 场景，R-LLM 输出 verdict=accept；"reject/worse" 场景输出 verdict=reject，且 `new_note` 含参数名。

---

### Task 7.4: Judge（值域 + SLO + 去重 + 循环防护）

**目标**：agent 唯一的"硬门"，所有 P-LLM 输出必经。

**操作**：新建 `tuner/judge.py`：
```python
class Judge:
    def __init__(self, registry, memory, max_steps=25,
                 slo_ttft_mult=1.2, slo_lat_mult=1.2, slo_preempt=5): ...

    def check_delta(self, delta: ConfigDelta) -> JudgeVerdict:
        """
        1. param 在 REGISTRY 且 mutation_class=RESTART
        2. new_value ∈ candidates
        3. (param, new_value) 不在最近 3 条 rejected_proposals
        4. 不重复当前最优配置
        -> {pass, reason, suggestion_for_retry}
        """
    def check_trial_constraints(self, m: TrialMetrics, baseline: TrialMetrics) -> dict:
        """返回 {pass, violations: [...]}。"""
    def should_terminate(self, memory: ExperienceMemory) -> tuple[bool, str]:
        """max_steps 硬上限；近 3 步 score 改动<2% 且 SLO 余量<5% → converged。"""
    def should_early_stop_trial(self, intermediate: dict) -> tuple[bool, str]: ...
```

**验证**：给出 `new_value=999`（超出 candidates）→ Judge 拒绝；max_steps=25 到达后 `should_terminate` 返回 True。

---

### Task 7.5: VtaAgent 主循环

**目标**：把上面全部拼成一个 `run()` 入口。

**操作**：新建 `tuner/agent.py`：
```python
class VtaAgent:
    def __init__(self, launcher, runner, memory, tools, judge, client, config): ...

    def run(self, workload_path: Path, baseline: TrialMetrics,
            max_steps: int = 20) -> AgentReport:
        current_cfg = DEFAULT_CONFIG.copy()
        # step 0: baseline 已在外部跑过，直接入 memory
        memory.append(baseline_record)

        for step in range(max_steps):
            # 1. Observe (上一次 trial 已入 memory)
            latest = memory.recent(1)[0]

            # 2. Diagnose
            diag = diagnose(client, memory, latest)      # fallback on LlmUnavailable
            if diag.should_stop or judge.should_terminate(memory)[0]:
                break

            # 3. Propose (+ tools)
            proposal = propose(client, tools, memory, diag, current_cfg)
            if isinstance(proposal, StopSignal): break

            # 4. Safety check
            verdict = judge.check_delta(proposal)
            if not verdict.pass_:
                memory.record_rejected(proposal, verdict.reason)
                continue                                  # 同 step 重试或直接下一步

            # 5. Act
            new_cfg = apply_delta(current_cfg, proposal)
            metrics = runner.run_trial(new_cfg, workload_path, ...)
            constraint = judge.check_trial_constraints(metrics, baseline)

            # 6. Record
            record = build_trial_record(step, new_cfg, proposal, metrics,
                                        constraint, diag.hypothesis)
            memory.append(record)

            # 7. Reflect
            refl = reflect(client, proposal, latest, record, constraint,
                           memory.notes)
            record.reflection = refl.reason
            memory.add_note(refl.new_note)

            # 8. Accept/rollback
            if refl.verdict == "reject" or not constraint["pass"]:
                # 不更新 current_cfg，下一步从 current_cfg 继续
                pass
            else:
                current_cfg = new_cfg

        return build_report(memory)
```

**验证**：对 `decode_heavy` 跑一次 end-to-end，step 数 ≤ 20，墙钟 ≤ 60 min，`AgentReport.best_trial.score > 0`，无崩溃，`results/tuning/<run_id>/memory.jsonl` 完整。

---

## 十、Week 8 — 对比实验与最终报告

> **本周目标**：3 种 workload 完整跑 + 5 组消融 + Reporter LLM + 三张图 + final_report。

### Task 8.1: Reporter LLM + final_report 生成器

**目标**：结束时一次 LLM 调用，产出可直接嵌入 final_report.md 的自然语言段落。

**操作**：新建 `llm_advisor/reporter.py`：
```python
def generate_report_section(client: LlmClient, memory: ExperienceMemory,
                            baseline: TrialMetrics) -> str:
    """
    client.chat(REPORTER_SYSTEM, REPORTER_USER_TMPL.format(
        baseline=..., full_memory=memory.dump_compact(), best=memory.best()),
        response_format="text")
    """
```

新建 `scripts/report/build_final_report.py`：汇总 3 份 `AgentReport` + 消融结果 → `docs/final_report.md`，章节包含:
- 背景与目标（Score / SLO）
- VTA-Agent 架构（ReAct 闭环 + 工具 + 内存）
- 实验：3 workload 对比 + 5 消融对比
- Reporter LLM 段（每个 workload 一段）
- 讨论：LLM 贡献 / 重启成本瓶颈 / 限制
- 附录：REGISTRY、所有 prompts、完整 memory dump

**验证**：Reporter 段落 ≥ 300 字、无数字幻觉（脚本 diff：Reporter 段内每个数字都应在 memory.dump 中出现）。

---

### Task 8.2: run_tuning.sh 端到端 + 3 种 workload

**目标**：一条命令跑完"baseline → agent → report"。

**操作**：
1. 新建 `scripts/experiment/run_tuning.sh`：
   ```bash
   # 用法: bash run_tuning.sh <workload_config> [--ablation=random|nomem|noreflect|fixed|noearly]
   # 1. 若 baseline 结果不存在 → 先跑 baseline
   # 2. python -m tuner.cli run --workload ... --ablation ...
   # 3. 输出 results/tuning/<run_id>/{memory.jsonl, report.json, report.md}
   ```
2. 新建 `tuner/cli.py`（argparse 入口），`AgentReport` 落 json，`report.md` 含 Reporter 段 + 关键表格。
3. 正式跑 3 种 workload：
   - `workload_prefix_50.json`（prefill-heavy）
   - `workload_long_only.json`（decode-heavy）
   - `workload_phase_switch.json`（mixed）
4. 汇总到 `results/tuning/summary_week8.md`（workload / baseline / best / 提升 / SLO 是否守住 / step 数 / wall time）。

**验证**：3 份 report 产出；≥ 2/3 达到 ≥ 10% 吞吐提升；3/3 无 SLO 违反；单 workload 墙钟 ≤ 60 min。

---

### Task 8.3: 5 组消融实验

**目标**：验证 agent 各组件的贡献。对 `decode_heavy` 跑：

| 代号 | 消融 | 实现方式 | 预期退化点 |
|------|------|---------|-----------|
| **A. random-proposer** | 替换 P-LLM 为随机挑 param+value | `--ablation=random` 注入 `RandomProposer` | 找到 best 所需 step↑，最终 score↓ |
| **B. no-memory** | `summarize()` 只返回 baseline + 最近 1 步 | `--ablation=nomem` | agent 反复试同一参数、收敛慢 |
| **C. no-reflect** | 跳过 R-LLM，不写 notes；accept 仅看 score 差 | `--ablation=noreflect` | 无法积累 workload-specific 规律 |
| **D. fixed-config** | 不跑 agent，直接用 `DEFAULT_CONFIG` | `--ablation=fixed` | 作为绝对下限对照（= baseline） |
| **E. no-early-stop** | Judge 关闭 trial 早停 + 循环终止条件 | `--ablation=noearly` | 墙钟大幅↑（期望 +40%） |

**产出物**：`results/tuning/ablation/{A..E}/report.json`、`ablation_summary.md`（柱状对比）。

**验证**：A/B/C 的 best_score < full 的 best_score；E 的 wall_time > full 的 1.3×。

---

### Task 8.4: 3 张核心图 + final_report.md 定稿

**目标**：毕设可用的终稿。

**操作**：生成 `results/tuning/figures/`：
- `throughput_improvement_bar.png`：3 workload × (baseline / agent best) 柱状图；
- `agent_trajectory.png`：step id vs rolling-best score（3 workload 一张图）；
- `ablation_bar.png`：full + 5 消融的 best_score / wall_time 对比。

`docs/final_report.md` 定稿 + `_issue_body.md` 勾选 Week 5–8 全部 Task。

**验证**：自读 report，每个数字都能从 `results/tuning/` 反查到；无孤立数据。

---

## 十一、最终交付物清单

```
configs/
├── experiments/
│   ├── baseline_0.json                  # Week 1-2 (4060+3B-AWQ) 历史基线
│   └── baseline_a6000_0.json            # Week 5 A6000+Qwen3-8B 新基线
├── workloads/                           # workload 配置（Week 3-4, Week 5 调整长度范围）
│   ├── workload_schema.json
│   ├── workload_baseline.json
│   ├── workload_burst.json
│   ├── workload_poisson.json
│   ├── workload_phase_switch.json
│   ├── workload_rate2.json
│   ├── workload_rate4.json
│   ├── workload_poisson2.json
│   ├── workload_short_only.json
│   ├── workload_long_only.json
│   ├── workload_mixed.json
│   ├── workload_prefix_0.json
│   ├── workload_prefix_50.json
│   └── workload_prefix_90.json
├── profiles/                            # vLLM 启动预设（Week 5 Task 5.3 重标定）
│   ├── profile_L.json                   # 低延迟
│   ├── profile_B.json                   # 平衡（默认）
│   └── profile_T.json                   # 高吞吐
├── llm_prompts/                         # LLM prompt 模板（Week 4 旧 + Week 7-8 新）
│   ├── few_shot_examples.json
│   ├── a_llm_diagnose.md                # A-LLM Diagnoser（Week 7）
│   ├── p_llm_propose.md                 # P-LLM Proposer（Week 7）
│   ├── r_llm_reflect.md                 # R-LLM Reflector（Week 7）
│   └── reporter.md                      # Reporter（Week 8）
└── llm_advisor/
    └── llm_advisor_config.json          # DeepSeek / OpenAI 切换 + 限流 + 缓存

benchmarks/                              # 压测执行（Week 2-4）
├── run_benchmark.py                     # 已被 tuner.runner 以模块方式调用
├── visualize_speed.py
└── prompts.json

workloads/                               # Module A: Workload 生成（Week 3）
├── __init__.py
├── workload_generator.py
├── prompts_pool.json
└── prefix_pool.json

monitors/                                # 基础设施监控（Week 4 + Week 6 Task 6.2 扩展）
├── __init__.py
├── gpu_monitor.py
└── vllm_metrics_collector.py

tuner/                                   # VTA-Agent 核心（Week 6-8）
├── __init__.py
├── config_generator.py                  # Task 6.1: 渲染 + 写临时 config
├── launcher.py                          # Task 6.1: VllmLauncher（eager / 页缓存 / 重启计时）
├── metrics_parser.py                    # Task 6.3: results/<exp_id>/ → TrialMetrics（产物解析）
├── runner.py                            # Task 6.3: run_trial + 在线早停
├── memory.py                            # Task 6.4: TrialRecord + ExperienceMemory + summarize
├── param_registry.py                    # Task 6.5: ParamSpec REGISTRY
├── tools.py                             # Task 6.6: ToolRegistry — A 只读 + dispatch B 算法
├── optimizer.py                         # Task 6.7: Optimizer — BO/敏感度/Pareto/局部网格/相位聚类
├── judge.py                             # Task 7.4: 值域 / SLO / 去重 / 循环防护
├── agent.py                             # Task 7.5: VtaAgent 闭环主循环
└── cli.py                               # Task 8.2: argparse 入口（支持 --ablation）

llm_advisor/                             # LLM 调用与角色（Week 6-8）
├── __init__.py
├── llm_client.py                        # Task 6.8: OpenAI 兼容 + function-calling + 重试/缓存/限速
├── prompts.py                           # Task 7.1-7.3 + 8.1: A/P/R-LLM + Reporter system/user 模板
├── diagnoser.py                         # Task 7.1: A-LLM + fallback
├── proposer.py                          # Task 7.2: P-LLM + tool-calling
├── reflector.py                         # Task 7.3: R-LLM + note append
└── reporter.py                          # Task 8.1: Reporter

scripts/
├── server/
│   ├── launch_server.sh
│   ├── start_server.sh                  # Week 5 改: 本地模型 + ENFORCE_EAGER 透传
│   └── stop_server.sh
├── setup/
│   ├── install_all.sh
│   ├── setup_env.sh
│   └── download_model.sh                # Week 5 改: 默认 Qwen3-8B
├── experiment/
│   ├── run_baseline.sh
│   ├── run_experiment_suite.sh
│   ├── run_initial_experiments.sh
│   └── run_tuning.sh                    # Week 8 新增: baseline + VtaAgent + report 一键入口
├── report/
│   └── build_final_report.py            # Week 8 Task 8.1: 汇总 3 workload + 消融 → final_report.md
├── verify/
├── test/
└── tools/

tests/                                   # Week 5 Task 5.4 适配后保持全绿
├── conftest.py                          # Week 5 新增: skip_if_no_gpu / model_path fixture
└── test_*.py

docs/
├── research_scope.md
├── troubleshooting.md
├── env_4060.md                          # 历史平台
├── env_a6000.md                         # Week 5 新增
├── migration_a6000.md                   # Week 5 新增
├── model_choice_qwen3_8b.md             # Week 5 Task 5.2
└── final_report.md                      # Week 8 Task 8.5

results/
├── baseline_0/                          # 历史
├── baseline_a6000_0/                    # Week 5
└── tuning/                              # Week 7-8
    ├── <run_id>/                        # 单次 agent run（workload × ablation）
    │   ├── memory.jsonl                 # 全部 TrialRecord
    │   ├── notes.jsonl                  # R-LLM 累积笔记
    │   ├── llm_calls.jsonl              # A/P/R/Reporter 每次调用的 input/output/usage
    │   ├── trials/trial_<n>/            # 每 trial 的 summary / timeseries / trace
    │   ├── report.json                  # AgentReport 结构化
    │   └── report.md                    # 人可读 + Reporter 段
    ├── summary_week8.md                 # 3 workload 汇总
    ├── ablation/{A..E}/                 # 5 组消融
    ├── ablation_summary.md
    └── figures/{throughput_improvement_bar,agent_trajectory,ablation_bar}.png
```

### 修改 / 新增关键文件一览

```
benchmarks/run_benchmark.py    # Week 3-4 已改，Week 6.2 追加 preempt/KV/queue summary 字段
monitors/vllm_metrics_collector.py  # Week 6.2: 增采 preempt/kv/queue 指标
scripts/server/start_server.sh # Week 5.3 改: 本地模型 + ENFORCE_EAGER 透传
requirements.txt               # +openai>=1.0.0, +tiktoken, +python-dotenv, +optuna>=3.6.0, +scipy
```

### 已废弃（不再产出）

- `proxy/` 整个包（proxy_server.py / batching / admission 等）
- `controllers/`（FixedController / LlmStrategyController / OptunaController / safety_guard）
- `executor/` + `profile_manager/` + `analyzer/`
- `workload_profiler/`（被 agent 的 Diagnoser 替代，不再做预运行分类）
- `tuner/coarse_search.py` / `tuner/fine_search.py` / `tuner/planner.py` / `tuner/runtime_ops.py`（搜索版产物，本方案不产出）
- 历史 prompt：`state_analysis.json` / `strategy_selection.json` / `search_pruning.json`（E-LLM-1/2, F-LLM-1/2）
- 历史 prompt：`flm_a_classify.md` / `flm_b_prune.md` / `flm_c_explain.md`（上一搜索版 F-LLM-A/B/C）

---

## 十二、执行注意事项

1. **严格按 Task 顺序执行**，后续 Task 依赖前序产出物
2. **每个 Task 完成后先验证**，再进入下一个
3. **不要过度工程化**：只做描述中要求的内容，不要提前加功能
4. **保持向后兼容**：旧的 `python3 benchmarks/run_benchmark.py --config configs/experiments/baseline_0.json` 必须继续能工作
5. **Agent 的 prompt 模板要严格要求 JSON-only 输出**：A/P/R-LLM 的 system prompt 末尾必有“仅输出 JSON”，parser 失败走 fallback
6. **LLM 调用必须可追溯**：每次 A/P/R/Reporter 调用的 input/output/tool_calls/usage 全部落盘到 `llm_calls.jsonl`，方便复现与 debug
7. **P-LLM 不得旁路**：LLM 的 ConfigDelta 必须经 Judge 校验；`apply_config / restart / rollback` 任何情况下都不出现在 ToolRegistry
8. **重启成本是硬成本**：维持页缓存、早期 step 用 eager、在线早停、`should_terminate` 收敛检测 — 四件缺一不可
9. **Memory 不要无节制填 LLM**：`summarize(top_k=3, recent_n=5)` 是默认，单次 prompt token 守在 ≤ 2000
10. **消融 D (fixed-config)** 即 baseline——用作绝对下限对照，不要尝试优化它

---

## 十三、每周验证检查单

### Week 3 验证
- [x] `WorkloadGenerator(seed=42).generate()` 两次结果完全一致
- [x] prompt_pool ≥ 30 条，prefix_pool ≥ 5 组
- [x] phase-switch 在 t=30s 时切换可观测
- [x] `--workload` 模式和旧模式都能正常运行

### Week 4 验证
- [x] 小规模压测 summary 中出现 TPOT 段（mean/median/p95）
- [x] `GpuMonitor` 3 秒内采集 ≥5 个样本
- [x] `VllmMetricsCollector` 采集到 num_requests_running 或优雅降级
- [x] `results/<id>/` 包含 config_snapshot + request_trace.jsonl + metrics_timeseries.jsonl + summary.json
- [x] `few_shot_examples.json` 至少 3 个示例

### Week 5 验证（平台迁移）
- [ ] `python3 -c "import torch, vllm; print(torch.cuda.get_device_name(0))"` → `NVIDIA RTX A6000`
- [ ] `~/models/Qwen3-8B/config.json` 存在且 `model_type == "qwen3"`（或相应实际名）
- [ ] `scripts/server/start_server.sh configs/experiments/baseline_a6000_0.json` 启动后 `/health` 返回 200
- [ ] `/metrics` 含 `vllm:cache_config_info` 中的 Qwen3-8B 模型路径
- [ ] `bash scripts/test/run_all_tests.sh` 全绿（原 76 tests + conftest fixture）
- [ ] `results/baseline_a6000_0/summary.json` 吞吐 ≥ 3B-AWQ baseline 的 2×，P95 TTFT < 300 ms

### Week 6 验证（VTA-Agent 基础设施）
- [ ] `VllmLauncher.restart(enforce_eager=True)` 热启动 ≤ 30 s；`enforce_eager=False` ≤ 60 s；两次 restart 间页缓存保留
- [ ] `VllmMetricsCollector` 对 mock 响应能解析 `vllm:time_in_queue_requests_seconds_sum`；`run_benchmark.py` summary 新增 5 个 vllm 聚合字段（preemptions_total / preemption_rate_per_min / kv_cache_usage_p95_pct / queue_time_delta_s / vllm_metrics_available）非空
- [ ] `metrics_parser.parse_trial(results/baseline_a6000_0/)` 返回的 `TrialMetrics` 全字段非空，且与 summary.json 中 ttft/tpot/throughput 完全一致
- [ ] `run_trial()` 对畸形 config（max_num_seqs=256, batched_tokens=1024）触发早停，返回 `early_killed=True`
- [ ] `ExperienceMemory.summarize(top_k=3, recent_n=5)` 对 10 条伪记录的 token 数 ≤ 1500，`tried_values_per_param` 去重正确
- [ ] `REGISTRY` 恰有 9 条 RESTART 参数；`ToolRegistry.dispatch("apply_config", ...)` 抛 `UnknownToolError`
- [ ] `Optimizer.bo_suggest(scope=["max_num_seqs","max_num_batched_tokens"], n=2)` 在 6 条伪 trial 上返回 2 个不重复候选；memory <4 时返回 `[]` 且 `insufficient_data=True`
- [ ] `Optimizer.param_sensitivity()` 输出按 score 方差降序，最敏感参数与人工预期一致
- [ ] `Optimizer.pareto_front(["throughput_tok_per_s","latency_p95_ms"])` 不返回任何被支配 trial；`local_grid("max_num_seqs", 2)` 在 best=64 时返回 {32, 128}
- [ ] `LlmClient.chat` 能完成一次 function-calling 多轮对话（注入 echo 工具验证）；同请求第二次命中缓存

### Week 7 验证（VTA-Agent 闭环决策）
- [ ] `diagnose()` 对三种合成 trial（preempt 爆 / 吞吐低 / SLO 紧）输出的 `bottleneck` 与预期一致 ≥ 2/3
- [ ] `propose()` 至少调用一次 `query_param_docs`（`LlmResponse.usage.tool_calls ≥ 1`），`new_value` 必在 `REGISTRY[param].candidates` 内
- [ ] 当 `memory ≥ 4` 且 `diagnosis.confidence < 0.6` 的合成场景下，P-LLM 实际触发 `bo_suggest`（在 `LlmResponse.usage.tool_calls` 中可观测，且 `ConfigDelta.tools_used` 含 `bo_suggest`）
- [ ] `reflect()` 在 accept/improve 场景输出 `verdict=accept`；reject/worse 场景输出 `verdict=reject` 且 `new_note` 含参数名
- [ ] `Judge.check_delta` 对超出 candidates 的值拒绝；`should_terminate` 在 max_steps=25 达到时返回 True
- [ ] `VtaAgent.run` 对 `decode_heavy` 端到端跑通：step 数 ≤ 20、墙钟 ≤ 60 min、`best_trial.score > 0`、`memory.jsonl` 完整，LLM 不可用时全链路 fallback 也能跑完
- [ ] 单元测试总数 ≥ 85

### Week 8 验证（对比实验 + 最终报告）
- [ ] Reporter 段落 ≥ 300 字，脚本 diff 确认毕 Reporter 结果中的数字都能在 `memory.jsonl` 找到
- [ ] `run_tuning.sh` 对 3 种 workload 各跑一次，每记墙钟 ≤ 60 min，共产出 3 份 report
- [ ] 3 workload 中 ≥ 2/3 达到 ≥ 10% 吞吐提升，3/3 不违反 SLO
- [ ] 5 组消融 A/B/C 的 best_score < full版；E 的 wall_time > full版 × 1.3
- [ ] 3 张核心图（throughput_improvement_bar / agent_trajectory / ablation_bar）生成
- [ ] `docs/final_report.md` 定稿；每个数字能从 `results/tuning/` 反查到
