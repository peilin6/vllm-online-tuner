<div align="center">

# ⚡ vLLM Online Tuner — VTA-Agent

**面向 vLLM 推理服务的 LLM 驱动闭环调参 Agent 与性能评测平台**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![vLLM 0.6.x](https://img.shields.io/badge/vLLM-0.6.x-green?logo=v&logoColor=white)](https://github.com/vllm-project/vllm)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Tests 256 passed](https://img.shields.io/badge/Tests-256%20passed-success)](./tests)
[![License: Academic](https://img.shields.io/badge/License-Academic-yellow)](./README.md)

</div>

---

## 📖 项目简介

本项目是一个面向 **vLLM 推理框架** 的全栈性能评测 + 自动调参平台。项目从"手工压测 + 手动选参"起步，逐步演进到"**LLM 驱动的闭环 Agent 调参 (VTA-Agent)**"：由 LLM 扮演 Diagnoser / Proposer / Reflector 三重角色，配合哨兵式 Judge 与 BO/敏感度分析等纯算法工具，自动探索 9 项 RESTART 参数的最佳组合。

### 🎯 研究目标

- 在固定硬件与模型条件下，建立 **可复现的性能基准线**（Baseline）
- 以 **WorkloadGenerator** 生成 prefill / decode / mixed / phase-switch 多种负载场景
- 构建 **VTA-Agent**：LLM orchestrator + 算法 inner-loop tools，针对不同 workload 自动选择最优 vLLM 运行参数
- 验证在 prefill_heavy / decode_heavy / mixed 三种负载上、达到 ≥ 10% 吞吐提升

### 📊 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 推理框架 | [vLLM 0.6.x](https://github.com/vllm-project/vllm) | PagedAttention + Continuous Batching |
| 模型 | Qwen3-8B (bf16) / Qwen2.5-3B-Instruct-AWQ | A6000 / 4060 双平台适配 |
| 压测引擎 | aiohttp + asyncio | 异步并发，支持流式 SSE |
| 监控 | pynvml + Prometheus | GPU 指标 + vLLM 内部指标 |
| 负载生成 | 自研 WorkloadGenerator | 可编程、可复现的请求序列 |
| Agent 主脑 | OpenAI 兼容 LLM (GPT-4 / DeepSeek / …) | function-calling 多轮工具调用 |
| 算法工具 | Optuna TPE / 纯函数分析 | BO / Pareto / 敏感度 / 局部网格 |

---

## ✨ 核心特性

<table>
<tr>
<td width="50%">

### 🚀 异步并发压测引擎
- 支持 **Burst / Constant-rate / Poisson** 三种请求到达模式
- 流式 SSE 解析 + `stream_options.include_usage` 精确 token 计数
- 可配置并发数、请求速率、超时策略

</td>
<td width="50%">

### 📈 全链路指标采集
- **TTFT** — 首 Token 时延
- **TPOT** — Token 间平均间隔
- **E2E Latency** — 端到端时延 (Mean / P95 / P99)
- **Throughput** — 请求吞吐 & Token 吞吐

</td>
</tr>
<tr>
<td>

### 🔧 可编程负载生成
- Prompt 长度分布采样（短/中/长加权）
- 共享前缀注入（Prefix Caching 实验）
- 多阶段 Phase-switch 负载切换
- 基于 seed 的可复现请求序列

</td>
<td>

### 📡 实时基础设施监控
- GPU 利用率 / 显存 / 温度 / 功耗（500ms 采样）
- vLLM Prometheus `/metrics` 端点采集
- 运行中/等待请求数、KV Cache 占用率、抢占次数

</td>
</tr>
</table>

### 🏆 Baseline 0 实测结果

<table>
<tr>
<th>指标</th><th>值</th><th>指标</th><th>值</th>
</tr>
<tr>
<td>🔄 吞吐量</td><td><b>0.37 req/s</b></td>
<td>⚡ Token 吞吐</td><td><b>75.37 tokens/s</b></td>
</tr>
<tr>
<td>🎯 TTFT 均值</td><td>62.6 ms</td>
<td>📊 TTFT P95</td><td>90.8 ms</td>
</tr>
<tr>
<td>⏱️ 端到端延迟均值</td><td>2717.2 ms</td>
<td>📈 端到端延迟 P95</td><td>3571.1 ms</td>
</tr>
<tr>
<td>✅ 成功率</td><td colspan="3"><b>100%</b></td>
</tr>
</table>

---

## 🏗️ 系统架构

```
                            ┌────────────────────────────────┐
                            │    VtaAgent (run loop)         │
                            │  Observe → Diagnose → Propose  │
                            │  → Safety → Act → Reflect      │
                            └──────┬──────┬──────────────────┘
      ┌──────────────────────────┬┴───┐  │
      ▼                          ▼    ▼  ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    ┌────────────────┐
  │ A-LLM   │ │ P-LLM   │ │ R-LLM   │ │ Judge   │    │ ToolRegistry   │
  │Diagnose │ │Propose  │ │Reflect  │ │(rules)  │    │ A: read-only 6 │
  │ (JSON)  │ │+ tools  │ │+ notes  │ └─────────┘    │ B: BO/grid 5   │
  └─────┬───┘ └────┬────┘ └────┬────┘                └───────┬────────┘
        └─────────┴────┬───────┴──── LlmClient (OpenAI compat) ────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
   │ Memory  │   │ Param   │   │ Runner   │   │ Launcher │
   │+notes   │   │Registry │   │+early    │   │  vLLM    │
   │+rejected│   │ 9 specs │   │ stop     │   │  ctrl    │
   └────┬────┘   └─────────┘   └────┬─────┘   └────┬─────┘
        │                            │              │
        ▼                       ┌────┴───────────────┴─────┐
   memory.jsonl                  │ Benchmark Engine          │
                                 │ + GPU/vLLM Monitors       │ ◄── vLLM Server
                                 └────────────┬──────────────┘     (OpenAI API)
                                              ▼
                                  results/<exp_id>/summary.json
```

### 🔄 主循环一轮运作

1. **Observe**：Memory 返回最近 trial + top-k
2. **Diagnose**：A-LLM 给出 `bottleneck` + `hypothesis` + `should_stop`
3. **Propose**：P-LLM 多轮 function-calling，调用 BO/sensitivity 等工具 → `ConfigDelta`
4. **Safety**：Judge 检查值域 / 是否最近被拒 / 是否重复 → 不通过则记一笔 `rejected` 后重取
5. **Act**：Runner 重启 vLLM + 跑 benchmark + 轮询早停 → `TrialMetrics`
6. **Reflect**：R-LLM 给出 `verdict` + `next_move_hint`，写一条 `note` 进 Memory
7. **Loop / Stop**：Judge.should_terminate 检查 max_steps / 收敛；R-LLM 主动建议 stop

### 🧭 VTA-Agent 优化 vLLM 参数的基本规则、限制与逻辑

当前实现把 `rule.txt` 中的 vLLM 调优经验收敛为三层硬约束：**A-LLM 只负责按指标诊断瓶颈**，**P-LLM 只能按 Playbook 白名单提议 1 个 RESTART 参数改动**，**Judge 在执行前后做安全闸与 SLO 校验**。因此 agent 的实际调参单位不是“自由改配置”，而是“观测指标 → 命中规则 → 从对应参数白名单中按方向选一个候选值 → 重启 vLLM 压测 → 接受或回滚”。

#### 1. 闭环与安全边界

| 环节 | 输入/依据 | 硬性限制 | 输出/动作 |
|:-----|:---------|:---------|:---------|
| Observe | 最新 trial、baseline、Memory top-k、最近 trial、rejected 记录 | 指标缺失值为 `-1` 时不得用作诊断依据 | 构造 A-LLM 诊断上下文 |
| Diagnose | `preempt_rate_per_min`、`kv_cache_usage_p95`、`queue_time_p95`、TTFT/TPOT/latency 与 baseline/SLO 对比 | 必须按 R1→R8 顺序命中；瓶颈只能从固定枚举中选择 | `bottleneck`、`confidence`、单参数方向性 `hypothesis` |
| Propose | Diagnosis + Playbook + ParamRegistry + Memory | 每轮只改 **1 个** 参数；必须在 Playbook `allowed_params` 内；方向必须匹配 `up/down/toggle`；新值必须来自登记候选或合法范围；不得复用最近 3 条 rejected 的 `(param, value)` | `ConfigDelta(param, old_value, new_value, reason, rollback_if)` 或 stop |
| Safety | Judge + ParamRegistry + Memory | 参数必须已登记；值必须在 `candidates/range` 内；不能与当前 best/current 完全重复；不能命中最近 rejected | 通过则执行 trial；拒绝则写入 `rejected_proposals` |
| Act | Runner + Launcher + benchmark | 当前登记参数均为 RESTART 类参数，生效需要重启 vLLM 后重新压测 | 生成新 trial metrics |
| Constraint | 新 trial metrics + baseline | trial 失败/早停拒绝；TTFT P95 或 latency P95 超过 baseline×1.2 拒绝；preemption > 5/min 拒绝 | `pass/violations` |
| Reflect | proposal、改动前后 trial、约束检查、历史 notes | 只有 `accept` 且约束通过才更新 current config；否则回滚到上一个 accepted config | 写入经验 note，给出下一步 `explore/double_down/rollback/stop` |
| Stop | Judge / Diagnoser / Reflector | 达到 `max_steps`；最近窗口吞吐变化 <2%；诊断为 `converged`；R-LLM 建议 stop | 输出最终报告与 best trial |

#### 2. 诊断规则到调参动作映射

| 优先级 | 命中信号 | 诊断瓶颈 | P-LLM 允许调整的参数与方向 | 主要逻辑 |
|:------|:---------|:---------|:--------------------------|:---------|
| R1 | `preempt_rate_per_min > 5` | `preempt_storm` | ↑ `gpu_memory_utilization`；↓ `max_num_seqs`；↓ `max_num_batched_tokens`；↑ `swap_space`；toggle `enable_chunked_prefill` | 抢占说明 KV/批调度压力过高，优先给 KV 腾显存，其次降并发和 token budget；`swap_space` 只作为重算成本高时的补救 |
| R2 | `kv_cache_usage_p95 > 0.95` | `kv_cache_pressure` | ↑ `gpu_memory_utilization`；↓ `max_num_seqs`；↓ `max_num_batched_tokens`；toggle `block_size` | KV cache 接近上限，按 `rule.txt` 优先增加可用显存比例，必要时降低同轮序列数或 token 数 |
| R3 | `queue_time_p95_ms > latency_p95_ms × 0.3` | `queue_backlog` | ↑ `max_num_seqs`；↑ `max_num_batched_tokens`；toggle `enable_chunked_prefill` | 排队占比高说明调度容量不足，增大 batch/并发以提升吞吐，但仍受 SLO 与 KV 安全闸约束 |
| R4 | `TTFT` 或 `latency` 的 SLO headroom <10% | `slo_margin_low` | ↓ `max_num_batched_tokens`；↓ `max_num_seqs`；toggle `enable_chunked_prefill` | 延迟余量不足时不再激进扩吞吐，优先缩小 batch/token budget 保护 P95/P99 |
| R5 | TTFT 比 baseline 高 ≥20%，且 TPOT 增幅 <10% | `prefill_bound` | ↑ `max_num_batched_tokens`；toggle `enable_chunked_prefill`；toggle `enable_prefix_caching`；↓ `max_num_seqs` | 长 prompt/prefill 慢时增大 prefill token 处理能力；共享前缀场景启用 prefix caching；KV 紧张才降并发 |
| R6 | TPOT 比 baseline 高 ≥20%，且 TTFT 增幅 <10% | `decode_bound` | ↓ `max_num_batched_tokens`；↓ `max_num_seqs`；toggle `enable_prefix_caching` | decode 慢通常受 memory bandwidth 与 prefill 干扰影响，降低 token budget 和并发以改善 ITL/TPOT |
| R7 | 最近 3 步吞吐极差 <2%，且 SLO 余量充足 | `converged` | 禁止继续调参 | 认为搜索收益已经很小，Proposer 输出 stop |
| R8 | 以上均不满足 | `underutilized` | ↑ `max_num_batched_tokens`；↑ `max_num_seqs`；↑ `gpu_memory_utilization` | 无明显 KV/preempt/queue/SLO 风险时按吞吐优先策略扩大批处理能力 |

#### 3. vLLM 可观测 Metrics 完整指标表

Agent / Judge / Playbook 决策所依赖的全部 trial-level 标量指标，由 [`tuner/metrics_parser.py`](tuner/metrics_parser.py) 的 `TrialMetrics` 数据类承载，从 `results/<exp_id>/summary.json` 解析得出。

**A. 吞吐与延迟（核心 SLO 指标）**

| 字段 | 单位 | 含义 | 主要用途 |
|:-----|:----|:----|:--------|
| `throughput_req_per_s` | req/s | 请求级吞吐 | 评分主指标之一；R7 收敛检测 |
| `throughput_tok_per_s` | tok/s | Token 级吞吐 | `_score()` 默认排序键；BO/敏感度的 target |
| `ttft_p95_ms` | ms | 首 token 时间 P95 | R5 prefill_bound 判定；TTFT SLO 余量计算 |
| `tpot_p95_ms` | ms | 单 token 间隔 P95 | R6 decode_bound 判定 |
| `latency_p95_ms` | ms | 端到端延迟 P95 | SLO 余量；Pareto 前沿目标 |

**B. KV / 调度健康度（瓶颈侧标）**

| 字段 | 单位 | 含义 | 主要用途 |
|:-----|:----|:----|:--------|
| `preemptions_total` | 次 | trial 内累计 preempt 次数 | R1 preempt_storm 触发条件之一 |
| `preemption_rate_per_min` | 次/分钟 | 单位时间 preempt 速率 | R1 主信号（≥ 阈值 → preempt_storm） |
| `kv_cache_usage_p95_pct` | 0–1 | KV cache 占用率 P95 | R2 kv_cache_pressure (>0.95)；R8 underutilized (<阈值) |
| `queue_time_p95_ms` | ms | 请求排队时间 P95 | R3 queue_bound 判定（queue/latency 比例） |

**C. 任务可信度与状态**

| 字段 | 单位 | 含义 | 主要用途 |
|:-----|:----|:----|:--------|
| `success` | bool | trial 是否成功（success_rate ≥ 0.95 且未早停） | `_score` 失败置 −1；过滤 BO 训练点 |
| `early_killed` | bool | 是否被 Judge 提前终止 | 区分主动失败 vs 自然失败 |
| `wall_time_s` | s | trial 实际墙钟时间 | 预算控制 |
| `success_rate` | 0–1 | 成功请求占比 | 进入 success 判定 |
| `total_requests` | 个 | 该 trial 发出的请求总数 | 数据可信度参考 |
| `notes` | list[str] | 解析阶段附加说明 | 调试用 |

**D. 上游原始指标来源（采集层 → 解析层映射）**

以上字段由 `summary.json` 中以下原始字段汇总而来（[`metrics_parser.py`](tuner/metrics_parser.py) 中的映射）：

| TrialMetrics 字段 | summary.json 源路径 |
|:------------------|:-------------------|
| `throughput_req_per_s` | `throughput_rps` |
| `throughput_tok_per_s` | `token_throughput_tps` |
| `ttft_p95_ms` | `ttft_ms.p95` |
| `tpot_p95_ms` | `tpot_ms.p95` |
| `latency_p95_ms` | `latency_ms.p95` |
| `preemptions_total` | `vllm_aggregates.preemptions_total` |
| `preemption_rate_per_min` | `vllm_aggregates.preemption_rate_per_min` |
| `kv_cache_usage_p95_pct` | `vllm_aggregates.kv_cache_usage_p95_pct` |
| `queue_time_p95_ms` | 由 `vllm_aggregates.queue_time_delta_s` 推导（histogram 仅含 `_sum`） |
| `success_rate` | `success_rate` |
| `wall_time_s` | `wall_time_s`（可由 Runner 覆盖） |

> 采集源：vLLM `/metrics` Prometheus 端点（KV/preempt/queue）+ aiohttp 压测客户端（throughput/TTFT/TPOT/latency）+ pynvml（GPU 利用率，用于诊断但不存入 TrialMetrics）。

#### 4. 当前可调 RESTART 参数事实表

| 参数 | 候选/范围 | 默认值 | 主要影响 | Agent 使用限制与调参含义 |
|:-----|:----------|:------|:---------|:--------------------------|
| `max_num_seqs` | 候选 `[8,16,32,64,96,128,192,256]`；范围 `1–512` | `32` | 吞吐、KV cache、TTFT | ↑ 提升并发和吞吐但增加 KV 压力；↓ 用于 preempt、KV 压力、decode 慢或 SLO 余量低 |
| `max_num_batched_tokens` | 候选 `[1024,2048,4096,8192,16384]`；范围 `256–32768` | `2048` | 吞吐、TTFT、TPOT | ↑ 增强 prefill/吞吐；↓ 降低 decode 干扰、P99 和 KV 压力；是吞吐/延迟权衡的核心旋钮 |
| `gpu_memory_utilization` | 候选 `[0.80,0.85,0.88,0.90,0.92,0.95]`；范围 `0.50–0.97` | `0.90` | KV cache、吞吐、稳定性 | ↑ 给 KV cache 更多空间；超过 `0.95` 附近 OOM/CUDA graph 风险升高，必须由 Judge 和 trial 验证 |
| `block_size` | 候选 `[8,16,32]`；范围 `8–32` | `16` | KV cache、吞吐 | 只在 KV 压力场景 toggle；小 block 提高内存利用率但元数据更多，大 block 相反 |
| `enable_chunked_prefill` | 候选 `[True, False]` | `True` | TTFT、TPOT、公平性 | prefill、queue、preempt、SLO 场景可切换；通常建议开启以让 prefill/decode 混合调度并降低 decode 抖动 |
| `enable_prefix_caching` | 候选 `[True, False]` | `False` | TTFT、KV cache | prefill 或 decode 场景可切换；大量重复系统 prompt/RAG 模板时优先开启，但需观察 KV 占用 |
| `swap_space` | 候选 `[0,2,4,8,16]`；范围 `0–64` GB | `4.0` | preempt recovery、吞吐 | 仅 preempt storm 场景允许 ↑；可能减少重算但 swap-in 慢，不能替代降低 KV 压力 |
| `cuda_graph_sizes` | 候选 `default/small/wide/off` | `default` | 启动时间、TPOT | 当前 Playbook 未主动使用；保留给 CUDA Graph 命中率/启动时间权衡实验 |
| `tensor_parallel_size` | 候选 `[1,2,4,8]`；范围 `1–8` | `1` | 吞吐、显存、延迟 | 当前 Playbook 未主动使用；多卡时可减轻单卡显存压力，但通信开销可能抵消收益 |

> 说明：`rule.txt` 还列出了 `max_model_len`、`kv_cache_dtype`、`api_server_count`、`renderer_num_workers`、`async_scheduling`、`pipeline_parallel_size`、`data_parallel_size` 等 vLLM 重要参数；当前 Agent 为了保证可复现实验与安全搜索，只把上表 9 个 RESTART 参数纳入 `ParamRegistry` 和 Judge 白名单。新增参数前必须先补充候选值、范围、Playbook 方向、测试与回滚策略。

---

## 📁 目录结构

<details>
<summary><b>点击展开完整目录</b></summary>

```
vlllm/
├── tuner/                          # ⏳ VTA-Agent 核心 (Week 6 + 7)
│   ├── agent.py                    #   VtaAgent 主循环
│   ├── launcher.py                 #   VllmLauncher—启停/重启 vLLM
│   ├── runner.py                   #   单 trial 闭环执行器 + 轮询早停
│   ├── config_generator.py         #   实验配置渲染 + 临时文件写入
│   ├── metrics_parser.py           #   results/<exp_id>/ → TrialMetrics
│   ├── memory.py                   #   ExperienceMemory + JSONL 落盘
│   ├── param_registry.py           #   9 项 RESTART 参数事实表
│   ├── tools.py                    #   ToolRegistry (A 类只读 6 个)
│   ├── optimizer.py                #   B 类算法工具 (5 个)
│   └── judge.py                    #   安全闸 / SLO / 收敛检测
├── llm_advisor/                    # 🧠 LLM 客户端 + 三重角色
│   ├── llm_client.py               #   OpenAI 兼容 + tool-calling 多轮
│   ├── prompts.py                  #   A/P/R-LLM + Reporter system/user 模板
│   ├── schemas.py                  #   DiagnosisResult / ConfigDelta / Reflection
│   ├── diagnoser.py                #   A-LLM—诊断瓶颈
│   ├── proposer.py                 #   P-LLM—提议参数改动与 BO
│   └── reflector.py                #   R-LLM—反思 + 长期笔记
├── benchmarks/                     # 📊 压测引擎
│   ├── run_benchmark.py            #   异步并发 + summary.json 聚合
│   └── prompts.json
├── monitors/                       # 📈 指标采集
│   ├── gpu_monitor.py              #   pynvml 周期采集
│   └── vllm_metrics_collector.py   #   /metrics Prometheus 解析
├── workloads/                      # 🎬 负载生成
│   ├── workload_generator.py
│   ├── prompts_pool.json           #   30+ 条分类 prompt 语料池
│   └── prefix_pool.json            #   5 组共享前缀模板
├── configs/                        # ⚙️ 配置
│   ├── experiments/                #   baseline_0 / baseline_a6000_0
│   ├── profiles/                   #   profile_{L,B,T}.json (Week 5)
│   ├── workloads/                  #   15 个 workload 预设
│   └── llm_prompts/few_shot_examples.json
├── scripts/                        # 🔨 自动化脚本
│   ├── setup/  server/  experiment/  verify/  test/  tools/
├── tests/                          # ✅ 219 passed / 2 skipped
│   ├── test_run_benchmark.py / test_workload_generator.py / test_configs.py
│   ├── test_gpu_monitor.py / test_vllm_metrics_collector.py / test_metrics_aggregation.py
│   ├── test_tuner_launcher.py / test_metrics_parser_and_runner.py
│   ├── test_memory.py / test_param_registry.py / test_tools.py / test_optimizer.py
│   ├── test_llm_client.py
│   ├── test_diagnoser.py / test_proposer.py / test_reflector.py
│   ├── test_judge.py / test_agent.py
│   └── test_integration_week6_7.py # 🔗 Week 6+7 联合集成测试
├── docs/                           # 📖 研究范围 / 环境 / 迁移 / 故障排查
├── results/                        # 💾 实验结果
├── logs/                           #     运行日志
└── requirements.txt                #     Python 依赖
```

</details>

---

## 🔨 环境要求

| 条件 | 要求 | 检查命令 |
|:-----|:-----|:---------|
| 🖥️ 操作系统 | Linux (Ubuntu 20.04/22.04 推荐) | `lsb_release -a` |
| 🎮 GPU | NVIDIA GPU, ≥8GB VRAM | `nvidia-smi` |
| ⚙️ CUDA Driver | ≥525.60 (CUDA 12.x) | `nvidia-smi` |
| 🐍 Python | 3.10.x | `python3 --version` |
| 💾 磁盘空间 | ≥10GB | `df -h` |

> 💡 **Windows 用户**：请通过 WSL2 (Ubuntu 22.04) 运行。安装方式：`wsl --install -d Ubuntu-22.04`

---

## 🚀 快速开始

### 1️⃣ 安装环境

```bash
# 克隆项目
git clone https://github.com/peilin6/vllm-online-tuner.git
cd vllm-online-tuner

# 一键安装（自动创建虚拟环境 + 安装依赖）
bash scripts/setup/install_all.sh
```

<details>
<summary>📋 手动安装</summary>

```bash
python3 -m venv ~/vllm-venv
source ~/vllm-venv/bin/activate
pip install --upgrade pip
pip install 'vllm>=0.6.0,<0.7.0' aiohttp requests numpy pandas pynvml tqdm
pip install 'transformers>=4.40.0,<4.50.0'
```

</details>

### 2️⃣ 下载模型

```bash
source ~/vllm-venv/bin/activate
bash scripts/setup/download_model.sh
```

> 模型下载到 `~/models/Qwen2.5-3B-Instruct-AWQ`（~2.69GB，hf-mirror.com 镜像，支持断点续传）

### 3️⃣ 启动服务

```bash
source ~/vllm-venv/bin/activate
bash scripts/server/launch_server.sh
```

看到 `Uvicorn running on http://0.0.0.0:8000` 表示启动成功。

### 4️⃣ 验证 & 压测

在**另一个终端**中执行：

```bash
source ~/vllm-venv/bin/activate

# 验证服务
python3 scripts/verify/verify_server.py

# 运行基准压测
python3 benchmarks/run_benchmark.py \
    --config configs/experiments/baseline_0.json \
    --host localhost --port 8000
```

### 🔁 一键全流程（可选）

```bash
bash scripts/experiment/run_baseline.sh
```

> 自动完成：环境信息采集 → 启动服务 → 验证 → 压测 → 结果汇总

### 5️⃣ 运行 VTA-Agent 自动调参

```python
# Python 脚本示例（详见 tuner/agent.py）
from llm_advisor.llm_client import LlmClient, LlmClientConfig
from tuner.agent import VtaAgent
from tuner.judge import Judge
from tuner.memory import ExperienceMemory
from tuner.optimizer import B_TOOL_SPECS
from tuner.runner import run_trial
from tuner.tools import ToolRegistry

memory = ExperienceMemory(path="results/tuning/memory.jsonl")
tools  = ToolRegistry(memory, extra_tools=B_TOOL_SPECS)
judge  = Judge(memory, max_steps=20)

# client=None 走 fallback；填入 API key 启用 LLM 决策
client = LlmClient(LlmClientConfig(api_key="sk-...", base_url="https://api.openai.com/v1"))

def runner_fn(cfg, *, baseline_throughput_tok_per_s):
    return run_trial(cfg,
                     base_config_path="configs/experiments/baseline_a6000_0.json",
                     workload_path="configs/workloads/workload_decode_heavy.json",
                     baseline_throughput_tok_per_s=baseline_throughput_tok_per_s)

agent = VtaAgent(memory, tools, judge, runner_fn, client=client, run_id="demo")
report = agent.run(baseline_metrics, baseline_config={"max_num_seqs": 32}, max_steps=20)
print(report.to_dict())
```

> Week 8 将提供 `scripts/experiment/run_tuning.sh` 一键 CLI 包装。

---

## ⚙️ 实验配置

### Baseline 0 参数

| 参数 | 值 | 说明 |
|:-----|:----|:-----|
| 请求数 | 50 | 总请求数量 |
| 并发数 | 1 | 串行发送 |
| max_tokens | 256 | 单请求最大生成 token |
| temperature | 0.7 | 采样温度 |
| 请求模式 | burst | 不限速发送 |

### 自定义压测

```bash
# 高并发
python3 benchmarks/run_benchmark.py --concurrency 4 --num-requests 100

# 限速 (2 req/s)
python3 benchmarks/run_benchmark.py --request-rate 2.0
```

### 多维度实验套件

```bash
bash scripts/experiment/run_experiment_suite.sh
```

自动扫描 **到达模式** × **prompt 长度** × **前缀比例** 三个实验轴。

### 📋 预定义 Workload 配置

| 配置文件 | 到达模式 | 特点 |
|:---------|:---------|:-----|
| `workload_baseline.json` | burst | 混合长度，默认基准 |
| `workload_rate2.json` | constant 2 req/s | 恒定速率 |
| `workload_rate4.json` | constant 4 req/s | 高速率 |
| `workload_poisson.json` | poisson λ=2 | 真实流量模拟 |
| `workload_short_only.json` | burst | 短 prompt 专项 |
| `workload_long_only.json` | burst | 长 prompt 专项 |
| `workload_prefix_50.json` | burst | 50% 共享前缀 |
| `workload_prefix_90.json` | burst | 90% 共享前缀 |
| `workload_phase_switch.json` | 多阶段 | 负载动态切换 |

---

## 📏 核心指标

| 指标 | 英文名 | 单位 | 说明 |
|:-----|:-------|:-----|:-----|
| 🔄 吞吐量 | Throughput | req/s | 每秒完成的请求数 |
| ⚡ Token 吞吐 | Token Throughput | tokens/s | 每秒生成的 token 数 |
| 🎯 首 Token 时延 | TTFT | ms | 发出请求到收到第一个 token |
| ⏱️ Token 间隔 | TPOT | ms | 输出 token 间的平均间隔 |
| 📊 端到端时延 | E2E Latency | ms | 请求到完整回复的时间 |
| 📈 尾部延迟 | P95 / P99 | ms | 尾部延迟分位数 |
| ✅ 成功率 | Success Rate | % | 成功请求占比 |

---

## 🧪 测试

```bash
source ~/vllm-venv/bin/activate
python3 -m pytest tests/ -v
```

当前状态：**219 passed / 2 skipped**（skip 项为需要真 vLLM 进程的 launcher 集成测）。

| 层级 | 测试模块 | 覆盖 |
|:-----|:---------|:-----|
| Week 1–4 | `test_run_benchmark.py` / `test_workload_generator.py` / `test_configs.py` / `test_gpu_monitor.py` / `test_vllm_metrics_collector.py` | 压测、负载生成、监控、指标聚合 |
| Week 5–6 | `test_tuner_launcher.py` / `test_metrics_parser_and_runner.py` / `test_metrics_aggregation.py` | Launcher / Runner / TrialMetrics |
| Week 6 | `test_memory.py` / `test_param_registry.py` / `test_tools.py` / `test_optimizer.py` / `test_llm_client.py` | Memory / 9 参数 / 工具调度 / BO / OpenAI 客户端 |
| Week 7 | `test_diagnoser.py` / `test_proposer.py` / `test_reflector.py` / `test_judge.py` / `test_agent.py` | A/P/R-LLM 三重角色 + Judge + 主循环 |
| 🔗 联合 | `test_integration_week6_7.py` | Mock LLM + Mock Runner 跑完整 VtaAgent.run 路径 |

> 所有 LLM 调用点（A/P/R-LLM）都有 **fallback 规则版**：client=None 也能完整跑完，所以 CI 不依赖任何 LLM API。

---

## 📈 项目路线图

| Week | 主题 | 状态 |
|:-----|:-----|:----:|
| 1 | 研究定义 + 工程骨架 | ✅ |
| 2 | 压测链路 + Baseline 0 | ✅ |
| 3 | Workload Generator | ✅ |
| 4 | Monitor + Data Pipeline | ✅ |
| 5 | A6000 + Qwen3-8B 迁移 | ✅ |
| 6 | VTA-Agent 基础设施（Runner / Memory / Tools / LlmClient） | ✅ |
| 7 | VTA-Agent 闭环决策主脑（A/P/R-LLM + Judge） | ✅ |
| 8 | Reporter + 5 组消融 + 最终报告 | ⏳ |

进度看板：[Issue #1 毕设任务看板](https://github.com/peilin6/vllm-online-tuner/issues/1)

---

## 📦 依赖

| 包名 | 版本 | 用途 |
|:-----|:-----|:-----|
| `vllm` | ≥0.6.0, <0.7.0 | LLM 推理框架 |
| `torch` | 随 vllm 安装 | 深度学习框架 |
| `transformers` | ≥4.40.0, <4.50.0 | 模型加载 / Tokenizer |
| `aiohttp` | ≥3.9.0 | 异步 HTTP 客户端 |
| `optuna` | ≥4.0.0 | TPE 贝叶斯优化（`bo_suggest` 工具）|
| `pynvml` | ≥11.5.0 | GPU 监控 |
| `numpy` / `pandas` | latest | 指标聚合与分析 |
| `pytest` + `pytest-timeout` | ≥8.0 / ≥2.4 | 测试框架 |
| `requests` / `tqdm` | latest | HTTP / 进度条 |

> LLM API 客户端内置于 [`llm_advisor/llm_client.py`](llm_advisor/llm_client.py)，只依赖标准库 `urllib`，不需额外 SDK；OpenAI 兼容接口均可接入（OpenAI / DeepSeek / 智谱 / 本地 vLLM 等）。

---

## ❓ FAQ

<details>
<summary><b>python3 -m venv 报错</b></summary>

```bash
sudo apt install python3.10-venv -y
```
</details>

<details>
<summary><b>transformers 版本不兼容 (all_special_tokens_extended)</b></summary>

```bash
pip install 'transformers>=4.40.0,<4.50.0'
```
</details>

<details>
<summary><b>OOM — GPU 显存不足</b></summary>

本项目使用 AWQ 4-bit 量化模型（~2.69GB）。如仍 OOM，降低 `gpu_memory_utilization` 或 `max_model_len`。
</details>

<details>
<summary><b>模型下载失败 / DNS 超时</b></summary>

`download_model.sh` 默认使用 hf-mirror.com 镜像，重新运行即可断点续传。
</details>

<details>
<summary><b>WSL2 代理警告（仅 Windows 用户）</b></summary>

不影响使用。如网络不通：`unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`
</details>

---

## 📄 License

本项目为毕业设计学术研究用途。
