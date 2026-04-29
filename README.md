<div align="center">

# ⚡ vLLM Online Tuner — VTA-Agent

**面向 vLLM 推理服务的 LLM 驱动闭环调参 Agent 与性能评测平台**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![vLLM 0.6.x](https://img.shields.io/badge/vLLM-0.6.x-green?logo=v&logoColor=white)](https://github.com/vllm-project/vllm)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Tests 219 passed](https://img.shields.io/badge/Tests-219%20passed-success)](./tests)
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
