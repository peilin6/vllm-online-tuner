# vLLM 第3-8周执行任务书：LLM 驱动的自主优化 Agent

> **用法**：将本文件完整交给 AI 编码助手，让它按 Task 顺序执行。每完成一个 Task 后，让 AI 汇报产出物并等你确认，再继续下一个 Task。
>
> **替代**：本文件替代旧版 `week3-5_execution.prompt.md`。

---

## 一、项目背景

你正在帮助一个毕业设计项目：**vLLM 在线推理服务性能评测与优化**。

论文定位：*an autonomous LLM-driven runtime adaptation agent for vLLM inference optimization*。

### 1.1 硬件与环境（已验证，不要更改）

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB VRAM |
| 驱动 | 566.26, CUDA 12.4 |
| OS | Windows + WSL2 (Ubuntu-22.04, kernel 6.6.87.2) |
| Python | 3.10.12 (WSL2 内, venv 在 ~/vllm-venv) |
| vLLM | 0.6.6.post1 |
| PyTorch | 2.5.1+cu124 |
| transformers | 4.49.0 |
| 模型 | Qwen/Qwen2.5-3B-Instruct-AWQ (4-bit AWQ, ~2.69GB) |
| 模型路径 | ~/models/Qwen2.5-3B-Instruct-AWQ |
| vLLM 服务端口 | 8000 |
| Proxy 端口 | 9000（Week 5 起） |

### 1.2 关键约束（必须遵守）

1. **所有命令在 WSL2 Ubuntu-22.04 中执行**，虚拟环境在 `~/vllm-venv`
2. **代理问题**：任何使用 Python `requests` 或 `aiohttp` 的脚本都必须在运行前 `export no_proxy="*"`
3. **PowerShell→WSL 引号问题**：不要在 PowerShell 中内联执行含引号/括号的 Python 命令，务必写成 .sh 脚本再 `wsl -- bash xxx.sh`
4. **显存限制 8GB**：不要使用 7B fp16 模型，当前只能跑 3B AWQ；**不能同时运行两个 vLLM 实例**
5. **不修改 vLLM 源码**，只在外部做 workload、proxy、controller
6. **所有新文件使用 UTF-8 编码**
7. **所有 Python 脚本**开头加 `#!/usr/bin/env python3` 和 `# -*- coding: utf-8 -*-`
8. **日志输出**使用中文，代码注释使用中文，变量名使用英文
9. **不要安装不必要的新依赖**，新增依赖仅限 `optuna>=3.6.0` 和 `openai>=1.0.0`

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

### 2.1 七模块架构图

```
┌─────────────────────────────────────────────────────────────┐
│                Module A: Workload Generator                  │
│     burst / constant-rate / poisson / phase-switch           │
│     工具: random.Random(seed) + asyncio + aiohttp            │
└────────────────────────────┬────────────────────────────────┘
                             │ POST /v1/chat/completions
                             ▼
┌─────────────────────────────────────────────────────────────┐
│             Module B: Proxy / Gateway (port 9000)            │
│  batching_window │ max_concurrency │ admission_threshold     │
│  /health │ /proxy/metrics │ /internal/config/proxy           │
│  工具: aiohttp.web + asyncio.Semaphore + asyncio.Queue       │
└────────────────────────────┬────────────────────────────────┘
                             │ 转发
                             ▼
┌─────────────────────────────────────────────────────────────┐
│           vLLM Backend 单实例 (port 8000)                    │
│    Profile L (低延迟) / B (平衡) / T (高吞吐)                │
│    通过重启加载不同配置，不并行运行多实例                      │
└────────────────────────────┬────────────────────────────────┘
              ┌──────────────┴──────────────────┐
              ▼                                 ▼
┌───────────────────────┐       ┌───────────────────────────┐
│  Module D: Monitor    │       │  Module C: Profile Manager │
│  pynvml + /metrics    │       │  3 profiles, restart-switch│
│  + proxy stats        │       │  stop→start→verify (~45s)  │
│  + request-level      │       └───────────────────────────┘
└──────────┬────────────┘                       ▲
           ▼                                    │
┌───────────────────────┐       ┌───────────────────────────┐
│  Module E: Analyzer   │──────▶│  Module F: Controller      │
│  ┌─────────────────┐  │       │  ┌───────────────────────┐ │
│  │ 规则层:         │  │       │  │Layer 0: Safety Guard  │ │
│  │ 滑动窗口聚合    │  │       │  │  硬编码规则，最高优先 │ │
│  │ 数值特征计算    │  │       │  │  不经 LLM             │ │
│  ├─────────────────┤  │       │  ├───────────────────────┤ │
│  │ 🤖 LLM 认知层: │  │       │  │Layer 1: LLM Strategy  │ │
│  │ E-LLM-1 状态    │  │       │  │  🤖 F-LLM-1 策略选择 │ │
│  │   语义化理解    │  │       │  │  conservative/balanced│ │
│  │ E-LLM-2 负载    │  │       │  │  /aggressive + 动作   │ │
│  │   模式识别      │  │       │  ├───────────────────────┤ │
│  └─────────────────┘  │       │  │Layer 2: Optuna BO     │ │
└───────────────────────┘       │  │  🤖 F-LLM-2 空间裁剪 │ │
                                │  │  + TPESampler 精搜    │ │
                                │  └───────────┬───────────┘ │
                                └──────────────┼─────────────┘
                                               ▼
                                ┌───────────────────────────┐
                                │  Module G: Executor        │
                                │  Fast: 更新proxy内存配置   │
                                │  Slow: 重启vLLM切profile   │
                                │  Rollback + cooldown       │
                                │  纯代码，不用 LLM          │
                                └───────────────────────────┘
```

### 2.2 LLM 参与边界（认知层 vs 控制层）

**核心原则**：LLM 负责"认知层"（理解、判断、规划），程序逻辑负责"控制层"（采集、计算、执行、安全）。

**LLM 只做三件事**：

| # | 职责 | 调用点 | 输入 | 输出 |
|---|------|--------|------|------|
| 1 | **状态语义化理解** | E-LLM-1/2 | 数值指标表 | 瓶颈类型、趋势、负载模式 |
| 2 | **高层策略规划** | F-LLM-1 | 语义状态 + 历史 + 动作空间 | conservative/balanced/aggressive + 具体参数 |
| 3 | **缩小搜索空间** | F-LLM-2 | 当前状态 + 完整搜索空间 | 受限候选集合 |

**LLM 不做四件事**：

| # | 职责 | 负责模块 | 实现方式 |
|---|------|---------|---------|
| 1 | 原始指标采集 | Module D (Monitor) | pynvml / /metrics / SSE 时间戳 |
| 2 | 数值特征计算 | Module E 规则层 | 滑动窗口 / statistics / numpy |
| 3 | 安全约束与回滚判断 | Module F Layer 0 + Module G | 硬编码 if-else 阈值 |
| 4 | 底层 API 执行 | Module G (Executor) | aiohttp POST / shell 脚本 |

### 2.3 LLM Advisor 技术栈

| 项目 | 选型 |
|------|------|
| API 提供商 | DeepSeek / OpenAI（可配置切换） |
| 客户端 SDK | `openai` Python 包（兼容 DeepSeek API） |
| 限流 | `asyncio.Semaphore(1)`，同一时刻最多 1 个 LLM 调用 |
| 缓存 | TTL dict，相同输入 30s 内复用 |
| 重试 | 3 次指数退避（1s→2s→4s） |
| 降级 | LLM 不可用时退化为纯规则引擎，日志标注 `"llm_source": "fallback"` |
| 输出格式 | JSON（系统 prompt 强制 + `json.loads` 校验） |

### 2.4 动作空间定义

**快动作（Proxy 层，毫秒级生效）**

| 参数 | 候选值 | 默认 |
|------|--------|------|
| `batching_window_ms` | {0, 10, 20, 50} | 0 |
| `max_concurrency` | {4, 8, 16, 32} | 32 |
| `admission_threshold` | {0.7, 0.8, 0.9, 1.0} | 1.0 |

**慢动作（vLLM 重启，~45s 生效）**

| Profile | max_num_seqs | max_model_len | enable_prefix_caching | 场景 |
|---------|-------------|--------------|----------------------|------|
| **L** (低延迟) | 16 | 1024 | false | 短请求快响应 |
| **B** (平衡) | 32 | 2048 | false | 通用场景（当前 baseline） |
| **T** (高吞吐) | 48 | 2048 | true | 大量请求+前缀共享 |

### 2.5 SLO 与 Reward

- **SLO**: TTFT_p95 < 200ms，latency_p95 < 5000ms
- **Reward**: R = 0.4 × throughput_norm − 0.4 × slo_violation_rate − 0.2 × reject_rate
- **Optuna**: TPESampler, ask-and-tell，每个 trial = 1 个控制窗口（60s）

---

## 三、LLM 调用详细设计

### 3.1 调用全景图

```
每个决策周期 (30s):

  ┌──────────────────────────────────────────┐
  │  API 调用 #1（Analyzer → LLM）           │
  │                                          │
  │  合并 E-LLM-1 + E-LLM-2                 │
  │  状态语义化 + 负载模式识别                │
  │  输入 ~600 tokens，输出 ~300 tokens       │
  │  延迟 ~1-2s                              │
  └──────────────────┬───────────────────────┘
                     │ AnalyzerState
                     ▼
  ┌──────────────────────────────────────────┐
  │  API 调用 #2（Controller → LLM）          │
  │                                          │
  │  F-LLM-1 策略选择 + 动作决策              │
  │  输入 ~800 tokens，输出 ~300 tokens       │
  │  延迟 ~1.5-2s                            │
  └──────────────────┬───────────────────────┘
                     │ ControllerDecision
                     ▼
            Safety Guard 校验 → Executor 执行

每个 Optuna trial (60s，仅 Layer 2 启用时):

  ┌──────────────────────────────────────────┐
  │  API 调用 #3（Controller → LLM）          │
  │                                          │
  │  F-LLM-2 搜索空间裁剪                    │
  │  输入 ~600 tokens，输出 ~200 tokens       │
  │  延迟 ~1s                                │
  └──────────────────────────────────────────┘
```

### 3.2 E-LLM-1+2：状态语义化 + 负载模式识别（合并为 1 次调用）

**System Prompt**:

```
你是 vLLM 推理服务的性能分析专家。你的任务是根据系统指标判断当前的运行状态。

你会收到一张包含三个时间窗口（10s/30s/60s）的指标表和实时快照。
你必须输出一个 JSON 对象，包含状态语义化描述和负载模式分类。

可用的瓶颈类型: prefill | decode | memory | queue | none
可用的负载模式: short_burst | long_prompt | long_decode | prefix_heavy | mixed
可用的趋势: degrading | stable | improving
可用的风险等级: safe | warning | critical
可用的到达特征: bursty | steady | ramping

你必须且只能以 JSON 格式回复，不要添加任何其他文字。
```

**User Prompt 模板**:

```
## 系统指标快照

### 实时值
- GPU 利用率: {gpu_util_pct}%
- 显存占用: {mem_used_mib}/{mem_total_mib} MiB ({mem_util_pct}%)
- KV Cache 使用率: {kv_cache_pct}%
- 队列深度: {queue_depth}
- 活跃请求数: {in_flight}
- 拒绝率（最近30s）: {reject_rate}%

### 滑动窗口聚合（10s / 30s / 60s）
| 指标 | 10s | 30s | 60s |
|------|-----|-----|-----|
| TTFT mean (ms) | {ttft_10s_mean} | {ttft_30s_mean} | {ttft_60s_mean} |
| TTFT P95 (ms) | {ttft_10s_p95} | {ttft_30s_p95} | {ttft_60s_p95} |
| TPOT mean (ms) | {tpot_10s_mean} | {tpot_30s_mean} | {tpot_60s_mean} |
| latency P95 (ms) | {lat_10s_p95} | {lat_30s_p95} | {lat_60s_p95} |
| throughput (req/s) | {tput_10s} | {tput_30s} | {tput_60s} |

### 请求统计（最近 30s）
- 平均 prompt 长度: {avg_prompt_len} tokens
- 平均 output 长度: {avg_output_len} tokens
- prefix group 占比: {prefix_ratio}%
- 到达率: {arrival_rate} req/s
- 到达 CV（变异系数）: {arrival_cv}

### 当前配置
- Proxy: batching_window={bw}ms, max_concurrency={mc}, admission_threshold={at}
- Backend Profile: {profile} (max_num_seqs={mns}, max_model_len={mml})

请分析上述指标并输出 JSON。
```

**输出 JSON Schema**:

```json
{
  "state_description": "string — 2-3句中文描述系统当前状态",
  "bottleneck_type": "prefill | decode | memory | queue | none",
  "bottleneck_confidence": 0.85,
  "trend": "degrading | stable | improving",
  "risk_level": "safe | warning | critical",
  "risk_signals": ["kv_cache_near_full", "tpot_rising"],
  "dominant_pattern": "short_burst | long_prompt | long_decode | prefix_heavy | mixed",
  "pattern_confidence": 0.78,
  "secondary_pattern": "string | null",
  "arrival_characteristic": "bursty | steady | ramping",
  "reasoning": "string — 分析推理过程"
}
```

**Few-shot 示例**（至少 3 个，应在 Week 4 用真实实验数据填充）:

```
示例 1: GPU 高利用率 + TPOT 上升 → bottleneck=decode, pattern=long_decode
示例 2: 队列深度快速增长 + TTFT 飙升 → bottleneck=queue, pattern=short_burst
示例 3: KV cache 87% + 显存接近满 → bottleneck=memory, risk=warning
```

### 3.3 F-LLM-1：策略选择 + 动作决策

**System Prompt**:

```
你是 vLLM 推理服务的自主优化控制器。你的目标是在满足 SLO 约束的前提下最大化吞吐量。

SLO 约束:
- TTFT P95 < 200ms
- 端到端延迟 P95 < 5000ms

你会收到系统的语义状态分析和可用动作空间。
你必须选择一个策略模式，并给出具体的参数调整动作。

策略模式:
- conservative: 优先保证 SLO，降低并发，收紧准入
- balanced: 在 SLO 安全范围内适度提升吞吐
- aggressive: SLO 余量充足时激进提升吞吐

动作类型:
- fast_only: 仅调整 Proxy 层快参数
- slow_only: 仅切换 Backend Profile（需要重启 vLLM，~45s 服务中断）
- both: 同时调整快参数和切换 Profile
- none: 维持当前配置不变

重要安全约束（你必须遵守）:
- 如果 risk_level=critical，只能选 conservative
- 如果最近一次 Profile 切换不到 60s，不能再次切换
- 如果最近一次快参数调整不到 20s，不能同方向再次调整
- 不要在 bottleneck=memory 时增大 max_concurrency

你必须且只能以 JSON 格式回复。
```

**User Prompt 模板**:

```
## 当前状态（来自 Analyzer）
{analyzer_state_json}

## 当前配置
- Proxy: batching_window={bw}ms, max_concurrency={mc}, admission_threshold={at}
- Backend Profile: {profile}

## 可用动作空间
### 快动作（Proxy 层）
- batching_window_ms: 可选 {0, 10, 20, 50}，当前 {bw}
- max_concurrency: 可选 {4, 8, 16, 32}，当前 {mc}
- admission_threshold: 可选 {0.7, 0.8, 0.9, 1.0}，当前 {at}

### 慢动作（Profile 切换，需 ~45s 重启）
- 可选 Profile: L(低延迟), B(平衡), T(高吞吐)，当前 {profile}

## 最近 5 次决策历史
{recent_decisions_json}

请选择策略模式和具体动作。
```

**输出 JSON Schema**:

```json
{
  "policy_mode": "conservative | balanced | aggressive",
  "action_type": "fast_only | slow_only | both | none",
  "fast_action": {
    "batching_window_ms": 20,
    "max_concurrency": 16,
    "admission_threshold": 0.9
  },
  "slow_action": {
    "target_profile": "T | B | L | null"
  },
  "reasoning": "string — 决策理由",
  "confidence": 0.82,
  "expected_effect": "string — 预期效果"
}
```

### 3.4 F-LLM-2：Optuna 搜索空间裁剪

**System Prompt**:

```
你是 vLLM 推理服务的参数优化顾问。你的任务是根据当前系统状态，
缩小 Optuna 贝叶斯优化器的搜索空间，去掉明显不合理的参数组合。

你会收到当前状态和完整搜索空间，以及最近几次 trial 的结果。
你需要输出一个受限的候选空间，以及可以直接固定的参数。

你必须且只能以 JSON 格式回复。
```

**User Prompt 模板**:

```
## 当前状态
{analyzer_state_json}

## 完整搜索空间
- batching_window_ms: {0, 10, 20, 50}
- max_concurrency: {4, 8, 16, 32}
- admission_threshold: {0.7, 0.8, 0.9, 1.0}
- backend_profile: {L, B, T}

## 最近 trial 历史
{recent_trials_json}

请给出受限候选空间和可固定参数。
```

**输出 JSON Schema**:

```json
{
  "constrained_space": {
    "batching_window_ms": [0, 10],
    "max_concurrency": [8, 16, 32],
    "admission_threshold": [0.9, 1.0],
    "backend_profile": ["B", "T"]
  },
  "fixed_params": {
    "admission_threshold": 0.9
  },
  "reasoning": "string — 裁剪理由"
}
```

### 3.5 API 成本估算（DeepSeek-Chat）

| 指标 | 值 |
|------|-----|
| 每决策周期 tokens | ~2000（两次调用合计） |
| 决策周期 | 30s |
| 10 分钟实验 | ~40,000 tokens |
| DeepSeek 单价 | ¥1/百万 input, ¥2/百万 output |
| **单次 10 分钟实验成本** | **~¥0.05** |
| 一天 20 次实验 | ~¥1 |

---

## 四、执行总览

```
Week 3: Module A — Workload Generator
  ├── Task 3.1: workload 配置 schema + 预设
  ├── Task 3.2: prompt 语料池 30+
  ├── Task 3.3: 共享前缀池 5+
  ├── Task 3.4: WorkloadGenerator 类（三种到达模式）
  ├── Task 3.5: phase-switch 能力
  └── Task 3.6: 集成 run_benchmark.py

Week 4: Module D — Monitor + Data Pipeline
  ├── Task 4.1: TPOT 逐 token 时间戳
  ├── Task 4.2: GPU 实时采样器
  ├── Task 4.3: vLLM /metrics 采集器
  ├── Task 4.4: 数据落盘重构
  ├── Task 4.5: 实验套件配置集
  └── Task 4.6: 运行初始实验集

Week 5: Module B — Proxy / Gateway
  ├── Task 5.1: 最小 proxy 骨架
  ├── Task 5.2: batching window
  ├── Task 5.3: max concurrency
  ├── Task 5.4: admission threshold + 429
  ├── Task 5.5: /proxy/metrics 端点
  └── Task 5.6: /internal/config/proxy 热更新

Week 6: LLM Advisor + Module E + Module C
  ├── Task 6.1: LlmAdvisor 基础设施
  ├── Task 6.2: 4 套 Prompt 模板
  ├── Task 6.3: Analyzer 规则核心
  ├── Task 6.4: Analyzer LLM 集成
  ├── Task 6.5: 3 个 backend profile 配置
  ├── Task 6.6: ProfileManager 类
  └── Task 6.7: Analyzer 集成测试

Week 7: Module G + Module F — Executor + Controller + 闭环
  ├── Task 7.1: Executor 快路径
  ├── Task 7.2: Executor 慢路径
  ├── Task 7.3: Rollback + 安全机制
  ├── Task 7.4: BaseController + FixedController
  ├── Task 7.5: Safety Guard (Layer 0)
  ├── Task 7.6: LLM Strategy Controller (Layer 1)
  └── Task 7.7: 闭环集成 + 稳定性测试

Week 8: Optuna + 对比实验
  ├── Task 8.1: Optuna 依赖 + 离散空间
  ├── Task 8.2: LLM 搜索空间裁剪 (F-LLM-2)
  ├── Task 8.3: OptunaController
  ├── Task 8.4: 4 组对比实验
  └── Task 8.5: 消融实验 + 结果汇总
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

## 七、Week 5 — Module B: Proxy / Gateway

### Task 5.1: 实现最小 proxy 骨架

**目标**：新建轻量级反向代理，接收 OpenAI-compatible 请求并转发给 vLLM。

**操作**：
1. 新建 `proxy/` 目录
2. 新建 `proxy/__init__.py`（空文件）
3. 新建 `proxy/proxy_server.py`，基于 `aiohttp.web` 实现：

```python
"""
最小 vLLM Proxy 服务器

架构:
  Client (run_benchmark.py)
    → Proxy (port 9000)
      → vLLM Backend (port 8000)

功能:
  - 接收 /v1/chat/completions 请求
  - SSE 流式逐 chunk 转发（不缓冲）
  - 排队、控制并发、按 window 分组
  - 记录自身指标
  - /proxy/metrics 暴露指标 JSON
  - /internal/config/proxy 热更新参数
"""

class ProxyServer:
    def __init__(self, config: dict):
        """
        config 包含:
        - backend_url: str (如 "http://127.0.0.1:8000")
        - host: str (如 "0.0.0.0")
        - port: int (如 9000)
        - batching_window_ms: int (初始 0)
        - max_concurrency: int (初始 32)
        - admission_threshold: float (初始 1.0)
        - queue_size_limit: int (如 100)
        - request_timeout_s: int (如 120)
        """

    async def handle_chat_completions(self, request):
        """处理 /v1/chat/completions 请求"""
        # 1. admission check
        # 2. 入队
        # 3. 等待 batching window 或队列满
        # 4. 获取 semaphore
        # 5. 转发给后端
        # 6. 流式回传响应给客户端

    async def handle_health(self, request):
        """代理层健康检查，同时检查后端"""

    async def handle_proxy_metrics(self, request):
        """GET /proxy/metrics — 返回 proxy 自身指标 JSON"""

    async def handle_config_update(self, request):
        """POST /internal/config/proxy — 热更新 proxy 参数"""

    async def handle_passthrough(self, request):
        """非 /v1/chat/completions 的请求直接透传给后端"""

    def get_metrics_snapshot(self) -> dict:
        """
        返回当前指标快照:
        {
            "queue_depth": int,
            "in_flight": int,
            "total_requests": int,
            "total_forwarded": int,
            "total_rejected": int,
            "total_batches": int,
            "avg_batch_size": float,
            "avg_queue_latency_ms": float,
            "rejection_rate": float,
            "current_config": {
                "batching_window_ms": int,
                "max_concurrency": int,
                "admission_threshold": float
            }
        }
        """
```

4. 新建 `configs/proxy/proxy_config.json`：

```json
{
  "proxy": {
    "host": "0.0.0.0",
    "port": 9000,
    "backend_url": "http://127.0.0.1:8000",
    "batching_window_ms": 0,
    "max_concurrency": 32,
    "admission_threshold": 1.0,
    "queue_size_limit": 100,
    "request_timeout_s": 120,
    "log_level": "INFO"
  }
}
```

5. 新建 `scripts/server/launch_proxy.sh`：

```bash
#!/bin/bash
source ~/vllm-venv/bin/activate
export no_proxy="*"
cd /mnt/d/vlllm
python3 proxy/proxy_server.py --config configs/proxy/proxy_config.json
```

**关键要求**：
- 第一版 proxy 的 `batching_window_ms=0` 和 `admission_threshold=1.0`（即不做任何控制），验证纯转发不引入额外延迟
- SSE 流式响应必须逐 chunk 转发，不能缓冲后一次性返回
- 对非 `/v1/chat/completions` 的请求直接透传给后端

**产出物**：`proxy/proxy_server.py`、`proxy/__init__.py`、`configs/proxy/proxy_config.json`、`scripts/server/launch_proxy.sh`

**验证**：
1. 启动 vLLM 后端 (port 8000)
2. 启动 proxy (port 9000)
3. `curl http://127.0.0.1:9000/health` 返回 200
4. `python3 scripts/verify/verify_server.py --port 9000` 三项检查全部通过
5. `python3 benchmarks/run_benchmark.py --host 127.0.0.1 --port 9000 --num-requests 3` 成功
6. 对比直连和经 proxy 的延迟差异小于 10%

---

### Task 5.2: 实现 batching window

**目标**：proxy 支持在 window 时间内累积请求后批量触发转发。

**操作**：
1. 在 `proxy/proxy_server.py` 中实现 batching window 机制：
   - 收到请求后放入 pending 队列
   - 启动 window 计时器（`batching_window_ms`）
   - window 到期时触发 batch 转发
   - 如果 pending 队列在 window 未到期时已达到 max_concurrency，提前触发
2. 每个 batch 记录 `batch_id`、`batch_size`、`formation_time_ms`
3. 在 proxy metrics 中新增 `total_batches`、`avg_batch_size`

**产出物**：`proxy/proxy_server.py` 的更新

**验证**：
```bash
# 修改 configs/proxy/proxy_config.json: "batching_window_ms": 200
# 运行并发测试
python3 benchmarks/run_benchmark.py --host 127.0.0.1 --port 9000 --num-requests 10 --concurrency 4
# 检查 curl http://127.0.0.1:9000/proxy/metrics 中 avg_batch_size > 1
```

---

### Task 5.3: 实现 max concurrency 控制

**目标**：proxy 限制同时转发给后端的活跃请求数。

**操作**：
1. 用 `asyncio.Semaphore(max_concurrency)` 控制并发
2. 超过并发限制的请求在 proxy 内排队（不是拒绝）
3. 在 proxy metrics 中实时更新 `in_flight`、`queue_depth`、`avg_queue_latency_ms`
4. 日志记录并发饱和事件

**产出物**：`proxy/proxy_server.py` 的更新

**验证**：
```bash
# 设置 max_concurrency=2, 发送 10 个请求 concurrency=8
# curl http://127.0.0.1:9000/proxy/metrics 应显示 in_flight <= 2
```

---

### Task 5.4: 实现 admission threshold

**目标**：当 proxy 队列压力超过阈值时拒绝新请求。

**操作**：
1. 计算 `load_ratio = queue_depth / queue_size_limit`
2. 当 `load_ratio >= admission_threshold` 时，返回 HTTP 429 + JSON 错误体：
   ```json
   {"error": {"message": "Server overloaded, please retry later", "type": "overloaded", "queue_depth": 50, "threshold": 0.9}}
   ```
3. 在 proxy metrics 中新增 `total_rejected`、`rejection_rate`
4. 在 `benchmarks/run_benchmark.py` 中处理 HTTP 429：记录 `error: "rejected_429"`，统计中单独显示

**产出物**：`proxy/proxy_server.py` 和 `benchmarks/run_benchmark.py` 的更新

**验证**：
```bash
# 设置 queue_size_limit=5, admission_threshold=0.8, max_concurrency=1
# 发送 20 个并发请求
python3 benchmarks/run_benchmark.py --host 127.0.0.1 --port 9000 --num-requests 20 --concurrency 20
# summary 中应出现部分 rejected_429 错误
```

---

### Task 5.5: /proxy/metrics 端点

**目标**：暴露 proxy 运行时指标。

**操作**：
1. 实现 `GET /proxy/metrics`，返回 JSON 格式的 `get_metrics_snapshot()` 结果
2. 包含字段（Task 5.1-5.4 中已定义）：
   - `queue_depth`, `in_flight`, `total_requests`, `total_forwarded`, `total_rejected`
   - `total_batches`, `avg_batch_size`, `avg_queue_latency_ms`, `rejection_rate`
   - `current_config`（当前 proxy 参数快照）
   - `uptime_s`

**产出物**：如已在前面 Task 中实现则确认完整性即可

**验证**：`curl -s http://127.0.0.1:9000/proxy/metrics | python3 -m json.tool` 输出格式正确

---

### Task 5.6: /internal/config/proxy 热更新 API

**目标**：支持外部控制器动态更新 proxy 参数。

**操作**：
1. 实现 `POST /internal/config/proxy`：
   ```json
   {
     "batching_window_ms": 20,
     "max_concurrency": 16,
     "admission_threshold": 0.9
   }
   ```
2. 更新立即生效（修改公共配置对象，新请求使用新参数）
3. 返回更新后的完整配置：
   ```json
   {"status": "ok", "previous": {...}, "current": {...}}
   ```
4. 校验参数合法性：
   - `batching_window_ms` ∈ [0, 500]
   - `max_concurrency` ∈ [1, 64]
   - `admission_threshold` ∈ [0.5, 1.0]
5. 非法参数返回 400

**产出物**：`proxy/proxy_server.py` 的更新

**验证**：
```bash
# 更新参数
curl -X POST http://127.0.0.1:9000/internal/config/proxy \
  -H "Content-Type: application/json" \
  -d '{"max_concurrency": 8}'
# 查看生效
curl http://127.0.0.1:9000/proxy/metrics | python3 -c "import json,sys; print(json.load(sys.stdin)['current_config'])"
```

---

## 八、Week 6 — LLM Advisor + Module E (Analyzer) + Module C (Profile Manager)

### Task 6.1: LlmAdvisor 基础设施

**目标**：建立统一的 LLM 调用封装层，供 Analyzer 和 Controller 共享。

**操作**：
1. 新建 `llm_advisor/` 目录
2. 新建 `llm_advisor/__init__.py`
3. 新建 `llm_advisor/llm_advisor.py`：

```python
"""
LLM Advisor — 统一 LLM 调用封装

支持 DeepSeek / OpenAI API，提供:
- 异步调用 (asyncio)
- 限流: 同一时刻最多 1 个调用
- 缓存: 相同输入 30s TTL 复用
- 重试: 3 次指数退避 (1s→2s→4s)
- 降级: API 不可用时返回 fallback 标记
- 监控: 调用延迟 / 成功率 / token 计数
"""
import asyncio
import hashlib
import json
import time
from openai import AsyncOpenAI


class LlmAdvisor:
    def __init__(self, config: dict):
        """
        config:
        {
            "provider": "deepseek" | "openai",
            "api_key": "sk-...",
            "base_url": "https://api.deepseek.com" | null,
            "model": "deepseek-chat" | "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 500,
            "timeout_s": 30,
            "cache_ttl_s": 30,
            "max_retries": 3
        }
        """

    async def analyze_state(self, metrics_snapshot: dict) -> dict:
        """
        E-LLM-1+2: 状态语义化 + 负载模式识别
        输入: 聚合指标快照
        输出: AnalyzerState dict（含 bottleneck_type, dominant_pattern 等）
        降级: 返回 {"llm_source": "fallback", ...} + 规则引擎基础分类
        """

    async def select_strategy(self, analyzer_state: dict,
                               current_config: dict,
                               action_space: dict,
                               decision_history: list[dict]) -> dict:
        """
        F-LLM-1: 策略选择 + 动作决策
        输入: 语义状态 + 当前配置 + 可用动作 + 历史
        输出: ControllerDecision dict
        降级: 返回 {"action_type": "none", "llm_source": "fallback"}
        """

    async def prune_search_space(self, analyzer_state: dict,
                                  full_space: dict,
                                  trial_history: list[dict]) -> dict:
        """
        F-LLM-2: Optuna 搜索空间裁剪
        输入: 状态 + 完整空间 + trial 历史
        输出: {"constrained_space": {...}, "fixed_params": {...}}
        降级: 返回完整空间不裁剪
        """

    async def _call_llm(self, system_prompt: str,
                         user_prompt: str,
                         output_schema: dict | None = None) -> dict:
        """
        底层 LLM 调用（含缓存/限流/重试/解析）
        """

    def get_stats(self) -> dict:
        """
        返回 LLM 调用统计:
        {
            "total_calls": int,
            "successful_calls": int,
            "failed_calls": int,
            "fallback_calls": int,
            "cache_hits": int,
            "avg_latency_ms": float,
            "total_input_tokens": int,
            "total_output_tokens": int
        }
        """
```

2. 新建 `configs/llm_advisor/llm_advisor_config.json`：

```json
{
  "llm_advisor": {
    "provider": "deepseek",
    "api_key": "${DEEPSEEK_API_KEY}",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "temperature": 0.3,
    "max_tokens": 500,
    "timeout_s": 30,
    "cache_ttl_s": 30,
    "max_retries": 3
  }
}
```

**注意**：`api_key` 从环境变量 `DEEPSEEK_API_KEY`（或 `OPENAI_API_KEY`）读取，配置文件中的 `"${...}"` 只是占位符。

**产出物**：`llm_advisor/llm_advisor.py`、`llm_advisor/__init__.py`、`configs/llm_advisor/llm_advisor_config.json`

**验证**：
```bash
export DEEPSEEK_API_KEY="your-key-here"
python3 -c "
import asyncio
from llm_advisor.llm_advisor import LlmAdvisor
import json

config = json.load(open('configs/llm_advisor/llm_advisor_config.json'))['llm_advisor']
config['api_key'] = '$DEEPSEEK_API_KEY'  # 从环境变量

advisor = LlmAdvisor(config)

# 测试基础连通性
result = asyncio.run(advisor._call_llm(
    system_prompt='你是一个测试助手，请只返回JSON',
    user_prompt='请返回 {\"status\": \"ok\"}'
))
print('LLM 连通:', result)
print('Stats:', advisor.get_stats())
"
```

---

### Task 6.2: 4 套 Prompt 模板

**目标**：为每个 LLM 调用点创建可维护的 prompt 模板。

**操作**：
1. 新建 `configs/llm_prompts/` 目录
2. 新建 4 个模板文件（JSON 格式，每个包含 system_prompt + user_prompt_template + output_schema + few_shot_examples）：
   - `configs/llm_prompts/state_analysis.json` — E-LLM-1+2 合并模板
   - `configs/llm_prompts/strategy_selection.json` — F-LLM-1 模板
   - `configs/llm_prompts/search_pruning.json` — F-LLM-2 模板
   - `configs/llm_prompts/few_shot_examples.json` — 共享的 few-shot 数据（Task 4.6 产出）

3. 模板格式：

```json
{
  "template_id": "state_analysis",
  "version": "1.0",
  "system_prompt": "你是 vLLM 推理服务的性能分析专家...",
  "user_prompt_template": "## 系统指标快照\n\n### 实时值\n- GPU 利用率: {gpu_util_pct}%\n...",
  "output_schema": {
    "type": "object",
    "required": ["bottleneck_type", "dominant_pattern", "risk_level"],
    "properties": {}
  },
  "few_shot_examples": [
    {
      "scenario": "长输出解码瓶颈",
      "input_summary": "GPU 92%, TPOT P95 18ms 上升, KV cache 87%",
      "expected_output": {"bottleneck_type": "decode", "dominant_pattern": "long_decode", "risk_level": "warning"}
    }
  ]
}
```

4. system_prompt 和 user_prompt_template 的内容来自本文件第三章（3.2/3.3/3.4 节）
5. few-shot 示例应使用 Task 4.6 的真实实验数据。如果 Task 4.6 尚未运行，先用合理的占位数据，后续替换

**产出物**：`configs/llm_prompts/` 下 4 个 JSON 文件

**验证**：`python3 -c "import json; [json.load(open(f'configs/llm_prompts/{f}.json')) for f in ['state_analysis','strategy_selection','search_pruning']]"`

---

### Task 6.3: Analyzer 规则核心

**目标**：实现 Analyzer 的数值计算层（不含 LLM 部分）。

**操作**：
1. 新建 `analyzer/` 目录
2. 新建 `analyzer/__init__.py`
3. 新建 `analyzer/analyzer.py`：

```python
"""
Analyzer — 系统状态分析器

架构:
  规则层（本模块）: 滑动窗口聚合 + 数值特征计算 + 基础阈值分类
  认知层（LLM）: 状态语义化理解 + 负载模式识别（在 Task 6.4 集成）
"""
import collections
import statistics
import time


class MetricsWindow:
    """固定时间窗口的指标聚合器"""

    def __init__(self, window_seconds: int):
        """window_seconds: 窗口大小（秒）"""

    def add(self, timestamp: float, metrics: dict):
        """添加一个指标数据点"""

    def get_aggregated(self) -> dict:
        """返回窗口内的聚合统计（mean/median/p95/min/max/slope）"""


class Analyzer:
    """系统状态分析器（规则核心）"""

    def __init__(self, config: dict = None):
        """
        内部维护三个窗口: 10s, 30s, 60s
        """

    def ingest_request_metrics(self, request_result: dict):
        """注入单个请求的指标（TTFT/TPOT/latency/output_tokens）"""

    def ingest_gpu_sample(self, sample: dict):
        """注入一个 GPU 采样点"""

    def ingest_vllm_metrics(self, metrics: dict):
        """注入一次 vLLM /metrics 采集结果"""

    def ingest_proxy_metrics(self, metrics: dict):
        """注入 proxy /proxy/metrics 快照"""

    def get_rule_based_state(self) -> dict:
        """
        规则引擎初筛 — 仅用阈值和统计方法，不调用 LLM。
        返回:
        {
            "windows": {
                "10s": {"ttft_mean": ..., "ttft_p95": ..., "tpot_mean": ..., ...},
                "30s": {...},
                "60s": {...}
            },
            "realtime": {
                "gpu_util_pct": ...,
                "mem_util_pct": ...,
                "kv_cache_pct": ...,
                "queue_depth": ...,
                "in_flight": ...,
                "reject_rate_30s": ...
            },
            "request_stats_30s": {
                "avg_prompt_len": ...,
                "avg_output_len": ...,
                "prefix_ratio": ...,
                "arrival_rate": ...,
                "arrival_cv": ...
            },
            "rule_classification": {
                "load_level": "low" | "medium" | "high" | "burst",
                "risk_level": "safe" | "warning" | "critical",
                "preliminary_bottleneck": "prefill" | "decode" | "memory" | "queue" | "none"
            }
        }
        """

    def _classify_load_level(self, metrics: dict) -> str:
        """基于 queue_depth 阈值的负载分级"""
        # low: queue_depth <= 2
        # medium: 3-10
        # high: 11-30
        # burst: >30

    def _classify_risk_level(self, metrics: dict) -> str:
        """基于 SLO 阈值的风险分级"""
        # safe: TTFT_p95 < 150ms & latency_p95 < 3000ms
        # warning: TTFT_p95 < 200ms & latency_p95 < 5000ms (SLO 边界)
        # critical: 任一指标超 SLO

    def _classify_bottleneck(self, metrics: dict) -> str:
        """基于数值特征的初步瓶颈判定"""
        # decode: TPOT 上升 & GPU 利用率高
        # prefill: TTFT 上升 & 队列不长
        # memory: KV cache > 85%
        # queue: queue_depth 快速增长
        # none: 无明显瓶颈
```

**产出物**：`analyzer/analyzer.py`、`analyzer/__init__.py`

**验证**：
```bash
python3 -c "
from analyzer.analyzer import Analyzer
import time

a = Analyzer()
# 模拟注入数据
for i in range(20):
    a.ingest_request_metrics({
        'ttft_ms': 60 + i, 'tpot_ms': 12 + i*0.5,
        'latency_ms': 2700 + i*50, 'output_tokens': 200,
        'prompt_length_bucket': 'medium'
    })
    a.ingest_gpu_sample({'gpu_util_pct': 80+i, 'mem_util_pct': 70+i, 'timestamp_s': i*0.5})
    time.sleep(0.01)

state = a.get_rule_based_state()
print('Load:', state['rule_classification']['load_level'])
print('Risk:', state['rule_classification']['risk_level'])
print('Bottleneck:', state['rule_classification']['preliminary_bottleneck'])
print('Rules OK')
"
```

---

### Task 6.4: Analyzer LLM 集成

**目标**：在规则核心之上集成 LLM 认知层。

**操作**：
1. 在 `analyzer/analyzer.py` 的 `Analyzer` 类中新增方法：

```python
    async def get_full_state(self, llm_advisor: 'LlmAdvisor' = None) -> dict:
        """
        完整状态分析 = 规则初筛 + LLM 语义理解

        如果 llm_advisor 提供且可用:
          调用 E-LLM-1+2，返回完整语义状态
        否则:
          返回规则初筛结果 + llm_source="fallback"
        """

    def _prepare_llm_input(self, rule_state: dict) -> dict:
        """将规则引擎的输出格式化为 LLM prompt 需要的指标快照"""
```

2. 输出的 `AnalyzerState` 统一格式：

```python
{
    # 来自规则层
    "windows": {...},
    "realtime": {...},
    "request_stats_30s": {...},
    "rule_classification": {...},

    # 来自 LLM 认知层（或降级为规则映射）
    "state_description": "中文状态描述",
    "bottleneck_type": "decode",
    "bottleneck_confidence": 0.85,
    "trend": "degrading",
    "risk_level": "warning",      # LLM 可能修正规则层的判定
    "risk_signals": [...],
    "dominant_pattern": "long_decode",
    "pattern_confidence": 0.78,
    "arrival_characteristic": "steady",
    "reasoning": "...",

    # 元数据
    "llm_source": "deepseek" | "openai" | "fallback",
    "llm_latency_ms": 1234,
    "timestamp": 1234567890.123
}
```

3. LLM 认知analyzer/analyzer.py` 的更新

**验证**：
```bash
# 需要设置 DEEPSEEK_API_KEY 环境变量
export DEEPSEEK_API_KEY="your-key"
python3 -c "
import asyncio
from analyzer

config = json.load(open('configs/llm_advisor/llm_advisor_config.json'))['llm_advisor']
import os; config['api_key'] = os.environ['DEEPSEEK_API_KEY']
advisor = LlmAdvisor(config)
a = Analyzer()

# 注入模拟数据
for i in range(10):
    a.ingest_request_metrics({'ttft_ms': 60+i, 'tpot_ms': 15, 'latency_ms': 3000, 'output_tokens': 200, 'prompt_length_bucket': 'medium'})
    a.ingest_gpu_sample({'gpu_util_pct': 90, 'mem_util_pct': 80, 'timestamp_s': i})

state = asyncio.run(a.get_full_state(advisor))
print('LLM source:', state.get('llm_source'))
print('Bottleneck:', state.get('bottleneck_type'))
print('Pattern:', state.get('dominant_pattern'))
print('Description:', state.get('state_description', '')[:80])
"
```

---

### Task 6.5: 定义 3 个 backend profile 配置

**目标**：创建 L/B/T 三个 vLLM 启动配置文件。

**操作**：
1. 每个 profile 是一个完整的 vLLM 启动配置 JSON，格式兼容 `start_server.sh`：

`configs/profiles/profile_L.json`（低延迟）:
```json
{
  "profile": {
    "name": "L",
    "description": "低延迟安全型：小 seq 数，短 context，不开 prefix caching",
    "model": {
      "name": "Qwen/Qwen2.5-3B-Instruct-AWQ",
      "dtype": "auto",
      "quantization": "awq",
      "max_model_len": 1024,
      "trust_remote_code": true
    },
    "server": {
      "host": "0.0.0.0",
      "port": 8000,
      "tensor_parallel_size": 1,
      "gpu_memory_utilization": 0.90,
      "max_num_seqs": 16,
      "enforce_eager": false,
      "swap_space": 1
    }
  }
}
```

`configs/profiles/profile_B.json`（平衡，与当前 baseline 一致）:
```json
{
  "profile": {
    "name": "B",
    "description": "平衡型：中等 seq 数和 context，通用场景",
    "model": { "max_model_len": 2048 },
    "server": { "max_num_seqs": 32 }
  }
}
```

`configs/profiles/profile_T.json`（高吞吐）:
```json
{
  "profile": {
    "name": "T",
    "description": "高吞吐型：大 seq 数，开 prefix caching，适合大量请求",
    "model": { "max_model_len": 2048 },
    "server": {
      "max_num_seqs": 48,
      "enable_prefix_caching": true
    }
  }
}
```

2. 注意：Profile T 的 `max_num_seqs=48` 可能在高并发时接近显存上限。如果实测 OOM 则降到 40

**产出物**：3 个 profile 配置文件

**验证**：`python3 -c "import json; [json.load(open(f'configs/profiles/profile_{p}.json')) for p in 'LBT']"`

---

### Task 6.6: ProfileManager 类

**目标**：封装 vLLM 实例的 profile 切换逻辑。

**操作**：
1. 新建 `profile_manager/` 目录
2. 新建 `profile_manager/__init__.py`
3. 新建 `profile_manager/profile_manager.py`：

```python
"""
ProfileManager — vLLM Backend Profile 切换管理

通过 stop → start → verify 三步完成 profile 切换。
属于"慢动作"，每次切换约 30-60 秒。
"""
import asyncio
import json
import time

class ProfileManager:
    def __init__(self, profiles_dir: str = "configs/profiles",
                 scripts_dir: str = "scripts/server",
                 base_url: str = "http://127.0.0.1:8000"):
        """
        加载 profile_L.json, profile_B.json, profile_T.json
        """

    @property
    def current_profile(self) -> str:
        """当前活跃 profile 名称 (L/B/T)"""

    @property
    def available_profiles(self) -> list[str]:
        """可用 profile 列表"""

    async def switch(self, target_profile: str, timeout_s: int = 120) -> dict:
        """
        切换到目标 profile。
        步骤:
        1. 调用 stop_server.sh 停止当前实例
        2. 根据目标 profile 配置生成启动参数
        3. 调用 start_server.sh 启动新实例（传入 profile 配置）
        4. 轮询 /health 等待就绪
        5. 调用 verify_server.py 确认可用

        返回:
        {
            "success": True,
            "previous_profile": "B",
            "current_profile": "T",
            "switch_duration_s": 42.3,
            "error": None
        }
        """

    async def _stop_backend(self) -> bool:
        """停止当前 vLLM 实例"""

    async def _start_backend(self, profile_config: dict) -> bool:
        """用指定 profile 配置启动 vLLM"""

    async def _wait_healthy(self, timeout_s: int) -> bool:
        """轮询 /health 端点等待就绪"""

    def get_profile_config(self, profile_name: str) -> dict:
        """返回指定 profile 的完整配置"""
```

2. 要求：
   - 使用 `asyncio.create_subprocess_exec` 调用 shell 脚本
   - 切换过程中锁定，防止并发切换
   - 切换失败时尝试回退到之前的 profile
   - 切换耗时记录到日志

**产出物**：`profile_manager/profile_manager.py`、`profile_manager/__init__.py`

**验证**：
```bash
# 注意：此测试会重启 vLLM 服务，确保无正在进行的实验
python3 -c "
import asyncio
from profile_manager.profile_manager import ProfileManager
pm = ProfileManager()
print('Current profile:', pm.current_profile)
print('Available:', pm.available_profiles)
# 如果有时间可以测试切换:
# result = asyncio.run(pm.switch('L'))
# print('Switch result:', result)
"
```

---

### Task 6.7: Analyzer 集成测试

**目标**：端到端验证 Monitor → Analyzer → LLM 输出链路。
/test
**操作**：

**操作**：
1. 新建 `scripts/test
#!/bin/bash
# 1. 确认 vLLM 运行中
# 2. 运行 10 个请求的小规模压测，同时采集 GPU 和 vLLM 指标
# 3. 将采集到的数据注入 Analyzer
# 4. 调用 get_full_state() 获取 LLM 分析结果
# 5. 打印完整输出，验证格式和语义
```

2. 验证点：
   - 三个窗口聚合数值正确（不为空/NaN）
   - 规则引擎分类结果合理
   - LLM 返回 JSON 可解析
   - `bottleneck_type` 和 `dominant_pattern` 为合法枚举值
   - `state_description` 是有意义的中文描述
   - `llm_source` 为实际使用的 provider

**产出物**：`scripts/test/test_analyzer_integration.sh` + 测试输出确认

**验证**：运行脚本后无报错，输出包含完整的 `AnalyzerState` 结构

---

## 九、Week 7 — Module G (Executor) + Module F (Controller) + 闭环

### Task 7.1: Executor 快路径

**目标**：实现 Executor 对 proxy 快参数的热更新。

**操作**：
1. 新建 `executor/` 目录
2. 新建 `executor/__init__.py`
3. 新建 `executor/executor.py`：

```python
"""
Executor — 控制决策执行器

两条执行路径:
  快路径: HTTP POST 更新 proxy 参数（毫秒级）
  慢路径: 调用 ProfileManager 切换 backend（~45s）
"""
import aiohttp
import asyncio
import time


class Executor:
    def __init__(self, proxy_url: str = "http://127.0.0.1:9000",
                 profile_manager: 'ProfileManager' = None):
        pass

    async def execute_fast(self, fast_action: dict) -> dict:
        """
        快路径: POST /internal/config/proxy 更新 proxy 参数
        输入: {"batching_window_ms": 20, "max_concurrency": 16, "admission_threshold": 0.9}
        输出: {"success": True, "path": "fast", "duration_ms": 12, "previous": {...}, "current": {...}}
        """

    async def execute_slow(self, slow_action: dict) -> dict:
        """
        慢路径: 切换 vLLM backend profile
        输入: {"target_profile": "T"}
        输出: {"success": True, "path": "slow", "duration_s": 42.3, "previous_profile": "B", "current_profile": "T"}
        """

    async def execute(self, decision: dict) -> dict:
        """
        统一入口: 根据 action_type 选择路径
        - fast_only: 只执行 execute_fast
        - slow_only: 只执行 execute_slow
        - both: 先 fast 再 slow
        - none: 不执行
        """

    # 配置快照栈（用于 rollback）
    def push_config_snapshot(self, config: dict):
        """保存当前配置快照到栈（最多 3 层）"""

    def get_rollback_config(self) -> dict | None:
        """取出上一个安全配置"""
```

**产出物**：`executor/executor.py`、`executor/__init__.py`

**验证**：
```bash
python3 -c "
import asyncio
from executor.executor import Executor
e = Executor()
# 测试快路径（需要 proxy 在 9000 运行）
result = asyncio.run(e.execute_fast({'max_concurrency': 16}))
print('Fast execute:', result)
"
```

---

### Task 7.2: Executor 慢路径

**目标**：实现 Executor 对 vLLM backend profile 的切换。

**操作**：
1. 在 `executor/executor.py` 中实现 `execute_slow()` 方法：
   - 调用 `ProfileManager.switch(target_profile)`
   - 切换前保存当前完整配置快照（含 proxy + profile）
   - 切换后验证新 profile 正常运行
   - 切换过程中 proxy 应继续响应排队请求，返回 503（后端不可用）给新请求
2. 在 proxy 中新增对后端不可用状态的处理：
   - 当 Executor 通知 "switching"，proxy 对新请求返回 503 而非转发到死连接

**产出物**：`executor/executor.py` 的更新 + `proxy/proxy_server.py` 中新增 503 状态处理

**验证**：切换 Profile B → L → B 后服务正常（如果时间允许）

---

### Task 7.3: Rollback + 安全机制

**目标**：实现三大安全保护机制。

**操作**：
1. 在 `executor/executor.py` 中新增：

```python
    async def rollback(self, reason: str) -> dict:
        """
        回退到上一个安全配置。
        触发条件（由 Safety Guard 调用）:
        - 连续 2 个窗口 SLO 违规
        - reject_rate > 50%
        - profile 切换后性能明显下降
        """

    def check_cooldown(self, action_type: str) -> bool:
        """
        检查冷却期:
        - 快动作: 上次快动作后 20s 内不能同方向调整
        - 慢动作: 上次 profile 切换后 60s 内不能再次切换
        返回 True 表示可以执行，False 表示在冷却中
        """

    def check_dwell_time(self) -> bool:
        """
        profile 最小驻留时间: 当前 profile 至少驻留 60s 后才能切换
        """

    def apply_action_mask(self, decision: dict, risk_level: str) -> dict:
        """
        动作掩码:
        - risk=critical: 禁止 aggressive 策略，禁止增大 concurrency
        - risk=warning: 禁止 slow_action（不在危险时切换 profile）
        返回修正后的 decision
        """
```

2. 所有安全机制都是**代码逻辑**，不经过 LLM

**产出物**：`executor/executor.py` 的更新

**验证**：单元测试冷却期和 action mask 逻辑

---

### Task 7.4: BaseController + FixedController

**目标**：定义 Controller 抽象接口和最简单的对照实现。

**操作**：
1. 新建 `controllers/` 目录
2. 新建 `controllers/__init__.py`
3. 新建 `controllers/base.py`：

```python
"""
BaseController -- Controller 抽象基类

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
  "decision": {"policy_mode": "conservative", "action_type": "fast_only", "fast_action": {}},
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
SafetyGuard -- Layer 0 安全规则检查器

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
            "fast_action": {"max_concurrency": 4, "admission_threshold": 0.7},
            "slow_action": {"target_profile": "L"} | null,
            "reasoning": "reject_rate 超过 50%，紧急降低并发",
            "is_emergency": True
        }
        """
        # 规则 1: reject_rate > 50% -> 立即降并发到最小，收紧准入
        # 规则 2: TTFT_p95 > SLO x 2.5 (500ms) -> 紧急降并发
        # 规则 3: KV cache > 95% -> 切换到 Profile L（最小 context）
        # 规则 4: 连续 2 个周期 SLO 违规 -> 触发 rollback

    def validate_decision(self, decision: dict, analyzer_state: dict) -> dict:
        """
        校验 controller 决策是否安全。
        修正不安全的决策（如 risk=critical 时禁止 aggressive）。
        返回修正后的 decision。
        """
```

**产出物**：`controllers/safety_guard.py`

**验证**：单元测试各种紧急场景下的输出

---

### Task 7.6: LLM Strategy Controller (Layer 1)

**目标**：实现以 LLM 为核心的策略决策器。

**操作**：
1. 新建 `controllers/llm_strategy_controller.py`：

```python
"""
LlmStrategyController -- Layer 1 LLM 策略控制器

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
            "current_fast_config": {},
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
       3. 调用 analyzer.get_full_state(llm_advisor) -> state
       4. safety_guard.check_emergency(state) -> emergency?
       5. 如果 emergency: 直接执行 emergency action
       6. 否则: controller.decide(state) -> decision
       7. safety_guard.validate_decision(decision, state) -> validated
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
# 1. 确认 vLLM 后端运行中
# 2. 启动 proxy + controller
# 3. 运行 workload generator 压测（指向 proxy port 9000）
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
   - `controller_decisions.jsonl` 有 >= 5 条非 `none` 决策
   - SafetyGuard 在 inject 高压时正确触发（如果出现高压场景）
   - LLM 调用成功率 > 80%

**产出物**：`proxy/proxy_server.py` 更新、`scripts/experiment/run_closed_loop.sh`

**验证**：4 个场景各运行 5 分钟，产出完整数据，无崩溃

---

## 十、Week 8 -- Optuna Controller + 对比实验

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
OptunaController -- Layer 2 贝叶斯优化控制器

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
        1. 如果当前 trial 还在运行中（未到 trial_duration_s）-> 返回 none
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
# 在闭环中运行 10 分钟（约 10 个 trial）
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

**目标**：运行标准化对比实验，评估每种 controller 的效果。

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
       "direct": {"throughput_tps": 0, "ttft_p95_ms": 0, "latency_p95_ms": 0, "reject_rate": 0},
       "fixed": {},
       "llm_strategy": {},
       "optuna": {}
     },
     "ablation": {
       "no_batching": {},
       "no_concurrency_control": {},
       "no_admission": {},
       "no_llm": {},
       "no_optuna": {}
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


---

## 十一、最终交付物清单

```
configs/
├── workloads/                       # workload 配置（Week 3-4）
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
├── proxy/                           # proxy 配置（Week 5）
│   └── proxy_config.json
├── profiles/                        # backend profile 配置（Week 6）
│   ├── profile_L.json
│   ├── profile_B.json
│   └── profile_T.json
├── controllers/                     # controller 配置（Week 7-8）
│   ├── controller_fixed.json
│   ├── controller_llm.json
│   └── controller_optuna.json
├── llm_advisor/                     # LLM advisor 配置（Week 6）
│   └── llm_advisor_config.json
└── llm_prompts/                     # LLM prompt 模板（Week 6）
    ├── state_analysis.json
    ├── strategy_selection.json
    ├── search_pruning.json
    └── few_shot_examples.json

workloads/                           # Module A: Workload 生成（Week 3）
├── __init__.py
├── workload_generator.py
├── prompts_pool.json
└── prefix_pool.json

monitors/                            # Module D: 基础设施监控（Week 4）
├── __init__.py
├── gpu_monitor.py
└── vllm_metrics_collector.py

analyzer/                            # Module E: 系统状态分析（Week 6）
├── __init__.py
└── analyzer.py

llm_advisor/                         # 共享 LLM 调用封装（Week 6）
├── __init__.py
└── llm_advisor.py

profile_manager/                     # Module C: Profile 切换管理（Week 6）
├── __init__.py
└── profile_manager.py

proxy/                               # Module B: Proxy/Gateway（Week 5）
├── __init__.py
└── proxy_server.py

controllers/                         # Module F: 控制器（Week 7-8）
├── __init__.py
├── base.py
├── fixed_controller.py
├── safety_guard.py
├── llm_strategy_controller.py
└── optuna_controller.py

executor/                            # Module G: 执行器（Week 7）
├── __init__.py
└── executor.py

scripts/
├── server/
│   └── launch_proxy.sh
├── experiment/
│   ├── run_experiment_suite.sh
│   ├── run_closed_loop.sh
│   ├── run_comparison.sh
│   └── run_ablation.sh
└── test/
    ├── run_comparison.sh
├── run_ablation.sh
└── test_analyzer_integration.sh
```

### 修改文件（~2个）

```
benchmarks/run_benchmark.py    # workload集成、TPOT采集、数据落盘重构、429处理
requirements.txt               # +optuna, +openai
```

---

## 十二、执行注意事项

1. **严格按 Task 顺序执行**，后续 Task 依赖前序产出物
2. **每个 Task 完成后先验证**，再进入下一个
3. **不要过度工程化**：只做描述中要求的内容，不要提前加功能
4. **保持向后兼容**：旧的 `python3 benchmarks/run_benchmark.py --config configs/baseline_0.json` 必须继续能工作
5. **LLM prompt 模板的 few-shot 示例**应优先使用 Task 4.6 的真实实验数据，而非编造
6. **Profile T 的 max_num_seqs=48** 可能在高并发下接近 OOM 边界，实测时如 OOM 则降到 40
7. **Optuna 探索效率**取决于 trial 数量，60s/trial 意味着 10 分钟 ≈ 10 个 trial，建议至少跑 20 分钟
8. **phase-switch 场景**是论文亮点（证明 controller 能适应动态负载变化），Week 7 稳定性测试重点验证

---

## 十三、每周验证检查单

### Week 3 验证
- [ ] `WorkloadGenerator(seed=42).generate()` 两次结果完全一致
- [ ] prompt_pool ≥ 30 条，prefix_pool ≥ 5 组
- [ ] phase-switch 在 t=30s 时切换可观测
- [ ] `--workload` 模式和旧模式都能正常运行

### Week 4 验证
- [ ] 小规模压测 summary 中出现 TPOT 段（mean/median/p95）
- [ ] `GpuMonitor` 3 秒内采集 ≥5 个样本
- [ ] `VllmMetricsCollector` 采集到 num_requests_running 或优雅降级
- [ ] `results/<id>/` 包含 config_snapshot + request_trace.jsonl + metrics_timeseries.jsonl + summary.json
- [ ] `few_shot_examples.json` 至少 3 个示例

### Week 5 验证
- [ ] `curl http://127.0.0.1:9000/health` 返回 200
- [ ] 经 proxy 压测与直连延迟差异 <10%
- [ ] batching_window=200ms 时 avg_batch_size > 1
- [ ] max_concurrency=2 时 in_flight ≤ 2
- [ ] admission_threshold=0.8 + 高并发 → 出现 429 拒绝
- [ ] `POST /internal/config/proxy` 更新后 `/proxy/metrics` 反映新值

### Week 6 验证
- [ ] `LlmAdvisor` 成功调用 DeepSeek/OpenAI API 并返回合法 JSON
- [ ] Analyzer 规则层输出三级分类（load_level/risk_level/preliminary_bottleneck）
- [ ] Analyzer LLM 层返回 bottleneck_type + dominant_pattern + state_description
- [ ] `llm_source` 正确标注（deepseek/openai/fallback）
- [ ] ProfileManager 有 3 个 profile 可列出
- [ ] 集成测试 `test_analyzer_integration.sh` 无报错

### Week 7 验证
- [ ] Executor 快路径更新 proxy 参数成功（<100ms）
- [ ] Executor 慢路径切换 profile 成功（如测试）
- [ ] cooldown 机制阻止了过于频繁的调整
- [ ] Safety Guard 在 inject 高指标时正确触发紧急动作
- [ ] 闭环运行 5 分钟无崩溃
- [ ] `controller_decisions.jsonl` 有 ≥5 条非 none 决策
- [ ] LLM 调用成功率 > 80%

### Week 8 验证
- [ ] `import optuna` 无报错
- [ ] F-LLM-2 裁剪后搜索空间变小
- [ ] OptunaController 有 ≥10 条 trial 记录
- [ ] 4 组对比实验均成功完成，各有完整数据
- [ ] 消融实验展示了各功能的性能贡献
- [ ] `comparison_summary.json` 数据自洽且可用于论文
