本 Issue 记录项目各阶段任务完成情况。完成后直接勾选即可。

---

## Week 1 — 研究定义 + 工程骨架 ✅

- [x] 研究问题提炼与参数边界定义 (`docs/research_scope.md`)
- [x] 项目目录骨架搭建 + 目录用途说明
- [x] 环境安装脚本 + 版本锁定 (`scripts/setup/`, `requirements.txt`)
- [x] vLLM 服务启动/停止脚本 (`scripts/server/`)
- [x] 服务可用性验证脚本 (`scripts/verify/verify_server.py`)
- [x] Baseline 0 配置 (`configs/experiments/baseline_0.json`)
- [x] 硬件/软件信息采集 (`scripts/tools/collect_sysinfo.py`)

---

## Week 2 — 压测链路 + Baseline 0 ✅

- [x] 压测脚本 + 请求样本 (`benchmarks/run_benchmark.py`, `benchmarks/prompts.json`)
- [x] 压测结果输出标准化 (JSON + TXT 双输出)
- [x] Baseline 0 执行流水线 (`scripts/experiment/run_baseline.sh`)
- [x] 故障排查文档 (`docs/troubleshooting.md`)
- [x] 文档模型版本统一 — 7B/fp16 引用统一为 Qwen2.5-3B-Instruct-AWQ
- [x] Token 计数精度修复 — `stream_options.include_usage` 替代 chunk 计数
- [x] Baseline 0 正式实验完成

<details>
<summary>📊 Baseline 0 压测结果 (2026-04-09)</summary>

```
  模型       : Qwen/Qwen2.5-3B-Instruct-AWQ
  并发       : 1
  请求数     : 50
  成功率     : 100.0%

  吞吐量     : 0.37 req/s
  Token吞吐  : 75.37 tokens/s

  首Token时延 (TTFT):
    Mean   : 62.6 ms
    Median : 52.0 ms
    P95    : 90.8 ms

  端到端时延:
    Mean   : 2717.2 ms
    Median : 3253.1 ms
    P95    : 3571.1 ms
    P99    : 3624.7 ms

  输出Tokens : 总计 10240, 平均 204.8/req
```

</details>

---

## Week 3 — Module A: Workload Generator ✅

- [x] **Task 3.1** Workload Schema 定义 — `configs/workloads/workload_schema.json` + 4 个预设配置
- [x] **Task 3.2** Prompt Pool 构建 — `workloads/prompts_pool.json` 30+ 条多样化 prompt
- [x] **Task 3.3** Prefix Pool 构建 — `workloads/prefix_pool.json` 5+ 组共享前缀模板
- [x] **Task 3.4** WorkloadGenerator 类 — Burst / Constant-rate / Poisson 三种到达模式
- [x] **Task 3.5** Phase-Switch 多阶段负载切换支持
- [x] **Task 3.6** 集成到 `run_benchmark.py` — `--workload` 参数，向后兼容

---

## Week 4 — Module D: Monitor + Data Pipeline ✅

- [x] **Task 4.1** TPOT 采集 — SSE 流中逐 token 计时，输出 TPOT mean/median/p95
- [x] **Task 4.2** GPU Monitor 增强 — pynvml 周期采集（利用率/显存/温度/功耗）
- [x] **Task 4.3** vLLM Metrics Collector — `/metrics` Prometheus 端点解析
- [x] **Task 4.4** 数据落盘重构 — JSON + TXT 双输出结构化结果
- [x] **Task 4.5** 实验套件配置 — 15 个 workload 预设 + `run_experiment_suite.sh` 批量执行
- [x] **Task 4.6** 初始实验 + Few-shot 数据 — `configs/llm_prompts/few_shot_examples.json`

<details>
<summary>📊 验证压测结果 (2026-04-10, 3 请求冒烟测试)</summary>

```
  模型       : Qwen/Qwen2.5-3B-Instruct-AWQ
  并发       : 1
  请求数     : 3
  成功率     : 100.0%

  吞吐量     : 0.11 req/s
  Token吞吐  : 19.52 tokens/s

  首Token时延 (TTFT):
    Mean   : 6806.4 ms
    P95    : 20261.7 ms

  输出Tokens : 总计 534, 平均 178.0/req
```

> 注：此次为小规模验证运行，TTFT 偏高因冷启动影响。

</details>

---

## Week 5 — 平台迁移：A6000 + Qwen3-8B ✅

- [x] **Task 5.1** A6000 环境与依赖安装 — `~/vllm-venv-a6000/`, `docs/env_a6000.md`, `docs/migration_a6000.md`
- [x] **Task 5.2** Qwen3-8B 模型获取与量化选型 — 默认 bf16，`~/models/Qwen3-8B`, `docs/model_choice_qwen3_8b.md`
- [x] **Task 5.3** configs / profiles / start_server.sh 适配 — `baseline_a6000_0.json`, `configs/profiles/profile_{L,B,T}.json`, workload 长度范围放宽
- [x] **Task 5.4** tests 与 monitor 适配 — 76 tests 全绿 + `tests/conftest.py` (skip_if_no_gpu / model_path fixture)
- [x] **Task 5.5** A6000 + Qwen3-8B 新 Baseline — `results/baseline_a6000_0/`，吞吐 ≥ 3B-AWQ × 2，P95 TTFT < 300 ms

---

## Week 6 — VTA-Agent 基础设施（框架 + 工具 + LLM 客户端）✅

- [x] **Task 6.1** VllmLauncher — `tuner/launcher.py` + `tuner/config_generator.py`，eager 透传、页缓存保留、重启计时（≤30s/60s）
- [x] **Task 6.2** 扩展 metrics 采集 + 产物字段补齐 — `monitors/vllm_metrics_collector.py` 加 `time_in_queue_requests_seconds_sum`（4 个已采不重复）；`benchmarks/run_benchmark.py compute_stats` 聚合 vllm 时序进 summary（preempt_rate / kv_p95 / queue_delta）
- [x] **Task 6.3** TrialMetrics + Runner — `tuner/metrics_parser.py` 负责 `results/<exp_id>/ → TrialMetrics`；`tuner/runner.py` subprocess 复用 `run_benchmark.py` + 轮询早停（preempt / 吞吐崩盘 / KV 溢出）
- [x] **Task 6.4** ExperienceMemory + TrialRecord + summarize() — `tuner/memory.py`，`top_k/recent_n` 紧凑视图供 LLM 使用
- [x] **Task 6.5** ParamRegistry — `tuner/param_registry.py`，9 条 RESTART 参数 + ParamSpec（candidates/affects/notes）
- [x] **Task 6.6** ToolRegistry — `tuner/tools.py`，A 类只读工具（6 个）+ `openai_tools_schema()` + dispatch；`apply_config` 明确禁止
- [x] **Task 6.7** Optimizer — `tuner/optimizer.py`，B 类算法工具（`bo_suggest` / `param_sensitivity` / `pareto_front` / `local_grid` / `cluster_workload_phases`），均为纯函数，由 ToolRegistry 统一暴露给 LLM
- [x] **Task 6.8** LlmClient — `llm_advisor/llm_client.py`，OpenAI 兼容 + function-calling 多轮 + 重试/缓存/限速

---

## Week 7 — VTA-Agent 闭环决策主脑 ✅

- [x] **Task 7.1** A-LLM Diagnoser — `llm_advisor/diagnoser.py` + `prompts.py`，diagnosis JSON（bottleneck/hypothesis/should_stop）+ fallback 规则
- [x] **Task 7.2** P-LLM Proposer（tool-calling） — `llm_advisor/proposer.py`，ConfigDelta JSON，至少一次 `query_param_docs`；fallback=未试过参数中位值
- [x] **Task 7.3** R-LLM Reflector — `llm_advisor/reflector.py`，verdict/reason/new_note/next_move_hint，写入 memory.notes
- [x] **Task 7.4** Judge — `tuner/judge.py`，值域 / SLO / rejected 去重 / max_steps 循环防护
- [x] **Task 7.5** VtaAgent 主循环 — `tuner/agent.py`，Observe→Diagnose→Propose→Safety→Act→Reflect；对 decode_heavy 端到端跑通 ≤60 min；单元测试 ≥ 85

---

## Week 8 — 对比实验与最终报告

- [ ] **Task 8.1** Reporter LLM + final_report 生成器 — `llm_advisor/reporter.py` + `scripts/report/build_final_report.py`，Reporter 段 ≥ 300 字且无数字幻觉
- [ ] **Task 8.2** run_tuning.sh + 3 种 workload — `scripts/experiment/run_tuning.sh`, `tuner/cli.py`；prefill/decode/mixed 各一份 report；≥ 2/3 达到 ≥ 10% 提升
- [ ] **Task 8.3** 5 组消融 — A random-proposer / B no-memory / C no-reflect / D fixed-config / E no-early-stop → `results/tuning/ablation/`
- [ ] **Task 8.4** 3 张核心图 + final_report.md 定稿 — `throughput_improvement_bar` / `agent_trajectory` / `ablation_bar` + `docs/final_report.md`

---

## 📊 已产出成果

### Baseline 0 正式压测 (2026-04-09)

| 指标 | 值 |
|------|-----|
| 模型 | Qwen2.5-3B-Instruct-AWQ (4-bit) |
| 请求数 / 并发 | 50 / 1 |
| 成功率 | **100%** |
| 吞吐量 | **0.37 req/s** |
| Token 吞吐 | **75.37 tokens/s** |
| TTFT Mean / P95 | 62.6 ms / 90.8 ms |
| 端到端延迟 Mean / P95 / P99 | 2717.2 ms / 3571.1 ms / 3624.7 ms |
| 平均输出 Tokens/req | 204.8 |

### 已实现模块

| 模块 | 文件 | 状态 |
|------|------|------|
| Benchmark Engine | `benchmarks/run_benchmark.py` | ✅ 异步并发 + SSE 流式 + 精确 token 计数 |
| Workload Generator | `workloads/workload_generator.py` | ✅ 3 种到达模式 + phase-switch + 前缀注入 |
| GPU Monitor | `monitors/gpu_monitor.py` | ✅ pynvml 周期采集，优雅降级 |
| vLLM Metrics | `monitors/vllm_metrics_collector.py` | ✅ Prometheus `/metrics` 解析 |
| Prompt Pool | `workloads/prompts_pool.json` | ✅ 30+ 条分类语料 |
| Prefix Pool | `workloads/prefix_pool.json` | ✅ 5 组共享前缀模板 |
| Workload 配置 | `configs/workloads/*.json` | ✅ 15 个预设 (burst/rate/poisson/prefix/phase) |
| 单元测试 | `tests/` | ✅ 5 个测试模块覆盖全部核心逻辑 |
