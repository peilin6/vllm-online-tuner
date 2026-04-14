<div align="center">

# ⚡ vLLM Online Inference Benchmark & Optimization

**基于 vLLM 的大语言模型在线推理服务性能评测与优化系统**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![vLLM 0.6.x](https://img.shields.io/badge/vLLM-0.6.x-green?logo=v&logoColor=white)](https://github.com/vllm-project/vllm)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: Academic](https://img.shields.io/badge/License-Academic-yellow)](./README.md)

</div>

---

## 📖 项目简介

本项目是一个面向 **vLLM 推理框架**的端到端性能评测与优化平台。通过系统化的基准测试，量化分析不同负载模式、请求速率、输入长度和前缀复用策略对在线推理服务的吞吐量与延迟的影响，为 LLM 推理服务的部署优化提供数据支撑和决策依据。

### 🎯 研究目标

- 在固定硬件与模型条件下，建立 **可复现的性能基准线**（Baseline）
- 通过多维度参数扫描，分析 **请求到达模式**、**输入/输出长度分布**、**共享前缀比例** 对推理性能的影响
- 探索基于实时监控指标的 **自适应参数调优策略**

### 📊 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 推理框架 | [vLLM](https://github.com/vllm-project/vllm) | PagedAttention + Continuous Batching |
| 模型 | Qwen2.5-3B-Instruct-AWQ | AWQ 4-bit 量化, ~2.69GB |
| 压测引擎 | aiohttp + asyncio | 异步并发，支持流式 SSE |
| 监控 | pynvml + Prometheus | GPU 指标 + vLLM 内部指标 |
| 负载生成 | 自研 WorkloadGenerator | 可编程、可复现的请求序列 |

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
┌─────────────────────────────────────────────────────────────────┐
│                        Experiment Config                        │
│              (baseline_0.json + workload_*.json)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   WorkloadGenerator     │
              │  (burst/rate/poisson)   │
              └────────────┬────────────┘
                           │ 请求序列
              ┌────────────▼────────────┐
              │   Benchmark Engine      │
              │  (aiohttp + asyncio)    │
              │  ┌──────────────────┐   │        ┌──────────────────┐
              │  │ Streaming SSE    │───┼───────► │  vLLM Server     │
              │  │ Token Counter    │   │         │  (OpenAI API)    │
              │  └──────────────────┘   │         └──────────────────┘
              └────────────┬────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
   ┌────────▼───────┐ ┌───▼──────┐ ┌─────▼─────────┐
   │  GPU Monitor   │ │  vLLM    │ │   Results      │
   │  (pynvml)      │ │ Metrics  │ │  (.json/.txt)  │
   │  500ms sample  │ │ 1s pull  │ │                │
   └────────────────┘ └──────────┘ └────────────────┘
```

---

## 📁 目录结构

<details>
<summary><b>点击展开完整目录</b></summary>

```
vlllm/
├── benchmarks/                     # 压测执行
│   ├── run_benchmark.py            #   核心压测脚本（异步并发 + 多模式请求）
│   └── prompts.json                #   5 条测试 prompt 样本（短/中/长）
├── monitors/                       # 基础设施监控
│   ├── gpu_monitor.py              #   GPU 利用率/显存/温度/功耗采集
│   └── vllm_metrics_collector.py   #   vLLM Prometheus metrics 采集
├── workloads/                      # 负载生成与语料管理
│   ├── workload_generator.py       #   可编程负载生成器
│   ├── prompts_pool.json           #   30+ 条分类 prompt 语料池
│   └── prefix_pool.json            #   5 组共享前缀模板
├── configs/                        # 实验配置
│   ├── experiments/
│   │   └── baseline_0.json         #   Baseline 0 完整配置
│   ├── workloads/                  #   15 个 workload 配置
│   └── llm_prompts/
│       └── few_shot_examples.json  #   LLM Advisor few-shot 示例
├── scripts/                        # 自动化脚本
│   ├── setup/                      #   环境安装与初始化
│   ├── server/                     #   服务启停与健康检查
│   ├── experiment/                 #   实验执行
│   ├── verify/                     #   验证与检查
│   ├── test/                       #   测试脚本
│   └── tools/                      #   工具脚本
├── tests/                          # 单元测试（5 个模块）
├── docs/                           # 文档
├── results/                        # 实验结果
├── logs/                           # 运行日志
└── requirements.txt                # Python 依赖
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

| 测试模块 | 覆盖范围 |
|:---------|:---------|
| `test_run_benchmark.py` | 统计计算、格式化输出、trace 构建 |
| `test_workload_generator.py` | 到达模式、prompt 采样、可复现性 |
| `test_configs.py` | 配置完整性、prompt 池、prefix 池格式 |
| `test_gpu_monitor.py` | 生命周期、daemon 线程、采样、优雅降级 |
| `test_vllm_metrics_collector.py` | Prometheus 解析、端点不可用降级 |

---

## 📦 依赖

| 包名 | 版本 | 用途 |
|:-----|:-----|:-----|
| `vllm` | ≥0.6.0, <0.7.0 | LLM 推理框架 |
| `torch` | 随 vllm 安装 | 深度学习框架 |
| `transformers` | ≥4.40.0, <4.50.0 | 模型加载 / Tokenizer |
| `aiohttp` | ≥3.9.0 | 异步 HTTP 客户端 |
| `requests` | ≥2.31.0 | HTTP 客户端 |
| `numpy` | ≥1.26.0 | 数据处理 |
| `pandas` | ≥2.1.0 | 数据分析 |
| `pynvml` | ≥11.5.0 | GPU 监控 |
| `tqdm` | ≥4.66.0 | 进度条 |

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
