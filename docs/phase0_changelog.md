# Phase 0 修改记录

> 完成时间：2026-04-10

## 概述

Phase 0 共完成两项任务，目的是在进入 Week 3-5 功能开发前，修复历史遗留的文档不一致和数据精度问题。

---

## Task 0.1：文档模型版本统一

### 背景

项目初期计划使用 Qwen2.5-7B-Instruct (fp16)，但因 RTX 4060 Laptop 8GB 显存不足，实际切换为 **Qwen2.5-3B-Instruct-AWQ**（AWQ 4-bit 量化，约 2.69GB 显存）。切换后多份文档仍残留 7B / fp16 / max_model_len 4096 等旧信息。

### 修改文件及内容

| 文件 | 修改内容 |
|------|----------|
| `docs/research_scope.md` | 模型名 7B → 3B-AWQ；添加量化方式 AWQ 4-bit；补充模型选型理由（8GB 显存约束） |
| `docs/experiment_template.md` | 模型默认值改为 `Qwen2.5-3B-Instruct-AWQ`；新增 `quantization: awq` 字段；`max_model_len` 4096 → 2048 |
| `docs/environment_guide.md` | 模型下载命令从 `Qwen2.5-7B-Instruct` 改为 `Qwen2.5-3B-Instruct-AWQ` |
| `docs/troubleshooting.md` | 模型路径和 curl 示例中的模型名统一改为 3B-AWQ |
| `plan.md` | 顶部添加状态横幅，说明实际使用 3B-AWQ 及原因 |

### 未修改

- `README.md` 中 FAQ 提到 7B fp16 无法在 8GB 显存运行——这是正确的说明，无需修改。
- `configs/baseline_0.json` 已是正确的 3B-AWQ 配置，无需修改。

### 验证

```bash
grep -rn "7B" docs/ configs/ benchmarks/ --include="*.md" --include="*.json" --include="*.py"
```

确认所有残留 7B 引用均为合理上下文（如 README FAQ）。

---

## Task 0.2：Token 计数精度修复

### 背景

原 `benchmarks/run_benchmark.py` 通过 SSE 流中 `data:` chunk 的数量来估算输出 token 数。实测发现 chunk 数比实际 token 数高约 34%（SSE 分块不等于 token 粒度），导致吞吐量虚高。

### 修改文件

**`benchmarks/run_benchmark.py`**

1. **请求 payload 新增 `stream_options`**

   ```python
   "stream_options": {"include_usage": True}
   ```

   启用后 vLLM 会在 SSE 流最后一个 chunk 的 `usage` 字段返回精确的 `completion_tokens` 数。

2. **SSE 解析逻辑新增 usage 提取**

   在 `send_request()` 的 SSE 循环中，检测 `chunk["usage"]["completion_tokens"]`，若存在则用该值覆盖 chunk 计数。

3. **结果新增 `output_tokens_source` 字段**

   每条请求记录标注 token 来源：
   - `"usage"` — 来自 vLLM 返回的精确统计（优先）
   - `"chunk_count"` — 降级为 SSE chunk 计数（兼容旧版本）

### 验证

运行 3 条请求的小规模测试：

```
req 0: output_tokens=22,  source=usage
req 1: output_tokens=256, source=usage
req 2: output_tokens=256, source=usage
```

3/3 请求均使用 `usage` 来源，确认 vLLM 0.6.6 支持 `stream_options.include_usage`。

### 此前已修复（关联问题）

`compute_stats()` 的吞吐量计算此前已从 `max(latencies)` 改为传入 `wall_time_s`（实际墙钟时间），修正了约 33 倍的吞吐量虚高（2488 → 75 tps）。

---

## 当前基准数据参考

| 指标 | 值 |
|------|----|
| 模型 | Qwen2.5-3B-Instruct-AWQ |
| 请求数 | 50 |
| 并发 | 1 |
| 成功率 | 100% |
| 吞吐量 (req/s) | 0.37 |
| 吞吐量 (tokens/s) | 75.37 |
| 平均 TTFT | 62.6 ms |

数据来源：`results/benchmark_20260409_223743.json`
