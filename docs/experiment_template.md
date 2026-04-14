# 实验记录模板

每次实验完成后复制本模板填写，保存到 `results/` 目录。

---

## 实验信息

| 项目 | 值 |
|------|-----|
| 实验名称 | baseline_0 |
| 日期 | YYYY-MM-DD |
| 执行者 | |
| 配置文件 | configs/baseline_0.json |

## 环境信息

| 项目 | 值 |
|------|-----|
| GPU | |
| 显存 | |
| CPU | |
| 内存 | |
| OS | |
| CUDA | |
| Python | |
| PyTorch | |
| vLLM | |

## 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen/Qwen2.5-3B-Instruct-AWQ |
| dtype | auto |
| quantization | awq |
| max_model_len | 2048 |
| gpu_memory_utilization | 0.90 |
| 并发数 | 1 |
| 请求数 | 50 |
| max_tokens | 256 |
| temperature | 0.7 |

## 结果摘要

| 指标 | 值 |
|------|-----|
| 吞吐量 (req/s) | |
| Token 吞吐 (tokens/s) | |
| TTFT Mean (ms) | |
| TTFT P95 (ms) | |
| Avg Latency (ms) | |
| P95 Latency (ms) | |
| P99 Latency (ms) | |
| 成功率 | |

## 原始数据路径

- JSON: `results/benchmark_XXXXXXXX_XXXXXX.json`
- 摘要: `results/benchmark_XXXXXXXX_XXXXXX.txt`
- 系统信息: `results/sysinfo_XXXXXXXX_XXXXXX.json`
- 服务日志: `logs/vllm_server_XXXXXXXX_XXXXXX.log`

## 观察与结论

<!-- 记录本次实验的关键观察、异常现象和初步结论 -->

