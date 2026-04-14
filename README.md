# vLLM 在线推理服务性能评测与优化

基于 vLLM 的大语言模型在线推理性能评测工程（毕业设计项目）。在 RTX 4060 Laptop GPU (8GB) + WSL2 环境下，对 Qwen2.5-3B-Instruct-AWQ 模型进行系统化的推理性能基准测试与优化研究。

> **硬件**: RTX 4060 Laptop GPU (8GB VRAM) + CUDA 12.4  
> **运行环境**: WSL2 (Ubuntu 22.04) — 所有脚本均在 WSL2 内执行  
> **模型**: Qwen/Qwen2.5-3B-Instruct-AWQ (AWQ 4-bit 量化, ~2.69GB)  
> **框架**: vLLM 0.6.6 + PyTorch 2.5.1

---

## 项目特性

- **异步并发压测引擎** — 基于 aiohttp，支持 Burst / Constant-rate / Poisson 三种请求到达模式
- **精确指标采集** — TTFT、TPOT、端到端延迟、吞吐量，流式 SSE + `stream_options.include_usage` 精确 token 计数
- **可编程负载生成** — 支持 prompt 长度分布采样、共享前缀注入（Prefix Caching）、多阶段 Phase-switch 负载切换
- **实时基础设施监控** — 后台 daemon 线程采集 GPU 利用率/显存/温度/功耗 + vLLM Prometheus `/metrics` 指标
- **完善的实验配置体系** — 15 个预定义 workload 配置覆盖速率/长度/前缀/突发等多维实验轴
- **全流程自动化** — 一键环境安装 → 模型下载 → 服务启动 → 验证 → 压测 → 结果汇总
- **完整单元测试** — 5 个测试模块覆盖压测统计、负载生成、配置校验、监控采集

### Baseline 0 实测结果

| 指标 | 值 |
|------|-----|
| 吞吐量 | 0.37 req/s |
| Token 吞吐 | 75.37 tokens/s |
| TTFT 均值 / P95 | 62.6 ms / 90.8 ms |
| 端到端延迟均值 / P95 | 2717.2 ms / 3571.1 ms |
| 成功率 | 100% |

---

## 目录结构

```
vlllm/
├── benchmarks/                     # 压测执行
│   ├── run_benchmark.py            #   核心压测脚本（异步并发 + 多模式请求）
│   └── prompts.json                #   5 条测试 prompt 样本（短/中/长）
├── monitors/                       # 基础设施监控模块
│   ├── gpu_monitor.py              #   GPU 利用率/显存/温度/功耗采集（pynvml）
│   └── vllm_metrics_collector.py   #   vLLM Prometheus metrics 采集
├── workloads/                      # 负载生成与语料管理
│   ├── workload_generator.py       #   可编程负载生成器（burst/rate/poisson + 前缀注入）
│   ├── prompts_pool.json           #   30+ 条分类 prompt 语料池
│   └── prefix_pool.json            #   5 组共享前缀模板（代码/SQL/API/数学/DevOps）
├── configs/                        # 实验配置
│   ├── experiments/
│   │   └── baseline_0.json         #     Baseline 0 完整配置
│   ├── workloads/                  #   15 个 workload 配置文件
│   │   ├── workload_baseline.json  #     默认 burst 混合长度
│   │   ├── workload_rate2/4.json   #     恒定速率 2/4 req/s
│   │   ├── workload_poisson*.json  #     泊松到达模式
│   │   ├── workload_*_only.json    #     短/长 prompt 专项
│   │   ├── workload_prefix_*.json  #     共享前缀比例 0%/50%/90%
│   │   ├── workload_phase_switch.json  # 多阶段负载切换
│   │   └── workload_schema.json    #     JSON Schema 定义
│   └── llm_prompts/
│       └── few_shot_examples.json  #     LLM Advisor few-shot 示例
├── scripts/                        # 工具脚本集合
│   ├── setup/                      #   环境安装与初始化
│   │   ├── install_all.sh          #     一键安装 vLLM 及所有依赖
│   │   ├── setup_env.sh            #     环境初始化（含验证）
│   │   └── download_model.sh       #     从 hf-mirror 下载模型
│   ├── server/                     #   服务启停与健康检查
│   │   ├── launch_server.sh        #     启动 vLLM 服务（推荐，WSL2 适配）
│   │   ├── start_server.sh         #     启动 vLLM 服务（从 JSON 读配置 + 后台）
│   │   ├── stop_server.sh          #     停止 vLLM 服务
│   │   └── _check_health.sh        #     快速健康检查
│   ├── experiment/                 #   实验执行
│   │   ├── run_baseline.sh         #     Baseline 一键流水线
│   │   ├── run_experiment_suite.sh #     批量实验套件（到达/长度/前缀三轴扫描）
│   │   └── run_initial_experiments.sh
│   ├── verify/                     #   验证与检查
│   │   ├── verify_server.py        #     服务健康+模型可用+生成能力验证
│   │   ├── verify_week3.sh         #     Week 3 Task 验证
│   │   └── check_and_run.sh        #     检查服务+启动+运行全套测试
│   ├── test/                       #   测试
│   │   ├── run_tests.sh            #     运行单元测试
│   │   ├── run_all_tests.sh        #     运行全套测试
│   │   └── test_stream.py          #     快速 streaming 请求测试
│   └── tools/
│       └── collect_sysinfo.py      #     系统信息采集（GPU/CPU/CUDA/版本）
├── tests/                          # 单元测试
│   ├── test_run_benchmark.py       #   压测统计/格式化/trace 构建
│   ├── test_workload_generator.py  #   负载生成/到达模式/可复现性
│   ├── test_configs.py             #   配置完整性/prompt池/prefix池格式
│   ├── test_gpu_monitor.py         #   GPU 监控生命周期/采样/降级
│   └── test_vllm_metrics_collector.py  # Prometheus 解析/端点降级
├── docs/                           # 文档
│   ├── research_scope.md           #   研究范围与参数边界定义
│   ├── environment_guide.md        #   环境搭建指南
│   ├── experiment_template.md      #   实验记录模板
│   ├── phase0_changelog.md         #   Phase 0 修改记录
│   └── troubleshooting.md          #   常见问题排查
├── results/                        # 实验结果
├── logs/                           # 运行日志
├── requirements.txt                # Python 依赖列表
└── README.md                       # 本文件
```

---

## 前置条件

| 条件 | 要求 | 检查命令 |
|------|------|----------|
| 操作系统 | Windows 10/11 + WSL2 (Ubuntu 22.04) | `wsl --list --verbose` |
| GPU | NVIDIA GPU, ≥8GB VRAM | `nvidia-smi` |
| CUDA Driver | ≥525.60 (支持 CUDA 12.x) | `nvidia-smi` 右上角显示 |
| Python | 3.10.x (WSL2 内) | `python3 --version` |
| python3-venv | 已安装 | `sudo apt install python3.10-venv` |
| 磁盘空间 | ≥10GB（模型 2.69GB + 依赖 ~5GB） | `df -h` |

### 安装 WSL2（如未安装）

```powershell
# 在 Windows PowerShell (管理员) 中运行
wsl --install -d Ubuntu-22.04
```

### 确认 GPU 在 WSL2 中可用

```bash
# 在 WSL2 Ubuntu 终端中
nvidia-smi
```

如果报错，需要安装 [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)。

---

## 快速开始

> 以下所有命令均在 **WSL2 Ubuntu 终端**中执行。

### 1. 安装环境

```bash
cd /mnt/d/vlllm
bash scripts/setup/install_all.sh
```

脚本会自动在 `~/vllm-venv` 创建 Python 虚拟环境并安装全部依赖。

如需手动安装：

```bash
python3 -m venv ~/vllm-venv
source ~/vllm-venv/bin/activate
pip install --upgrade pip
pip install 'vllm>=0.6.0,<0.7.0' aiohttp requests numpy pandas pynvml tqdm
pip install 'transformers>=4.40.0,<4.50.0'
```

> **重要**: 虚拟环境必须创建在 WSL2 本地路径（如 `~/vllm-venv`），不要放在 `/mnt/d/` 上。

### 2. 下载模型

```bash
source ~/vllm-venv/bin/activate
cd /mnt/d/vlllm
bash scripts/setup/download_model.sh
```

模型下载到 `~/models/Qwen2.5-3B-Instruct-AWQ`（~2.69GB，使用 hf-mirror.com 镜像，支持断点续传）。

### 3. 启动 vLLM 服务

```bash
source ~/vllm-venv/bin/activate
bash /mnt/d/vlllm/scripts/server/launch_server.sh
```

启动约 30-60 秒，看到 `Uvicorn running on http://0.0.0.0:8000` 表示成功。服务会占据当前终端，后续操作需**另开一个 WSL2 终端**。

### 4. 验证服务

```bash
source ~/vllm-venv/bin/activate
cd /mnt/d/vlllm
python3 scripts/verify/verify_server.py
```

验证 `/health` 可达、模型已加载、`/v1/chat/completions` 可正常生成。

### 5. 运行基准压测

```bash
python3 benchmarks/run_benchmark.py \
    --config configs/experiments/baseline_0.json \
    --host localhost --port 8000
```

结果自动保存到 `results/`：
- `benchmark_<时间戳>.json` — 完整原始数据
- `benchmark_<时间戳>.txt` — 人可读摘要

### 6. 一键完整流水线（可选）

```bash
bash scripts/experiment/run_baseline.sh
```

自动执行：采集环境信息 → 启动服务 → 验证 → 压测 → 汇总结果。

---

## 压测参数说明

### Baseline 0 配置 (`configs/experiments/baseline_0.json`)

| 参数 | 值 | 说明 |
|------|-----|------|
| 请求数 | 50 | 总请求数量 |
| 并发数 | 1 | 串行发送 |
| max_tokens | 256 | 单请求最大生成 token 数 |
| temperature | 0.7 | 采样温度 |
| 请求模式 | burst | 不限速，尽快发送 |

### 自定义参数

```bash
# 调整并发数和请求数
python3 benchmarks/run_benchmark.py --concurrency 4 --num-requests 100

# 限速发送 (2 req/s)
python3 benchmarks/run_benchmark.py --request-rate 2.0
```

### 批量实验套件

```bash
bash scripts/experiment/run_experiment_suite.sh
```

自动扫描三个实验轴：到达模式 (burst / 2 req/s / 4 req/s / poisson)、prompt 长度分布、共享前缀比例。

---

## 核心指标

| 指标 | 英文名 | 单位 | 说明 |
|------|--------|------|------|
| 吞吐量 | Throughput | req/s | 每秒完成的请求数 |
| Token 吞吐 | Token Throughput | tokens/s | 每秒生成的 token 数 |
| 首 Token 时延 | TTFT | ms | 发出请求到收到第一个 token 的时间 |
| Token 间隔 | TPOT | ms | 输出 token 间的平均间隔 |
| 端到端时延 | E2E Latency | ms | 发出请求到收到完整回复的时间 |
| P95 / P99 时延 | P95 / P99 Latency | ms | 尾部延迟 |
| 成功率 | Success Rate | % | 成功完成的请求占比 |

---

## 运行测试

```bash
cd /mnt/d/vlllm
source ~/vllm-venv/bin/activate
python3 -m pytest tests/ -v
```

---

## 依赖

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| vllm | ≥0.6.0, <0.7.0 | LLM 推理框架 |
| torch | 随 vllm 安装 | 深度学习框架 |
| transformers | ≥4.40.0, <4.50.0 | 模型加载 / Tokenizer |
| aiohttp | ≥3.9.0 | 异步 HTTP 客户端（压测） |
| requests | ≥2.31.0 | HTTP 客户端（验证） |
| numpy | ≥1.26.0 | 数据处理 |
| pandas | ≥2.1.0 | 数据分析 |
| pynvml | ≥11.5.0 | GPU 监控 |
| tqdm | ≥4.66.0 | 进度条 |

---

## 常见问题

<details>
<summary><b>Q: python3 -m venv 报错</b></summary>

```bash
sudo apt install python3.10-venv -y
```
</details>

<details>
<summary><b>Q: transformers 版本不兼容 (all_special_tokens_extended 错误)</b></summary>

```bash
pip install 'transformers>=4.40.0,<4.50.0'
```
</details>

<details>
<summary><b>Q: OOM — GPU 显存不足</b></summary>

8GB VRAM 无法运行 fp16 模型。本项目使用 `Qwen2.5-3B-Instruct-AWQ` (4-bit, ~2.69GB 权重)。如仍 OOM，降低 `gpu_memory_utilization` 或 `max_model_len`。
</details>

<details>
<summary><b>Q: WSL2 代理警告 "检测到 localhost 代理配置"</b></summary>

不影响使用。如网络不通：
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```
</details>

<details>
<summary><b>Q: 模型下载失败 / DNS 超时</b></summary>

使用 hf-mirror.com 镜像（`download_model.sh` 默认配置）。重新运行脚本会自动断点续传。
</details>

---

## License

本项目为毕业设计学术用途。
