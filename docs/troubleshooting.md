# 故障排查指南

## 排查流程总览

```
环境安装失败 → 检查 Step 1
服务启动失败 → 检查 Step 2
API 验证失败 → 检查 Step 3
压测异常     → 检查 Step 4
结果异常     → 检查 Step 5
```

---

## Step 1: 环境安装失败

### 症状
- `pip install` 报错
- `torch.cuda.is_available()` 返回 `False`
- `import vllm` 失败

### 排查步骤

1. **检查 Python 版本**
   ```bash
   python3 --version  # 应为 3.10.x
   ```

2. **检查 CUDA 是否可用**
   ```bash
   nvidia-smi                    # 应显示 GPU 信息
   nvcc --version                # 应为 12.1+
   ```

3. **检查 PyTorch CUDA 编译版本**
   ```python
   import torch
   print(torch.__version__)        # 如 2.4.0+cu121
   print(torch.version.cuda)       # 应与系统 CUDA 匹配
   print(torch.cuda.is_available())
   ```

4. **重装 PyTorch (指定 CUDA 版本)**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

5. **vLLM 安装问题**
   ```bash
   pip install vllm>=0.6.0 --no-cache-dir
   ```

---

## Step 2: 服务启动失败

### 症状
- 启动脚本超时 (>300s)
- 进程立即退出
- 显存不足 (OOM)

### 排查步骤

1. **查看服务日志**
   ```bash
   cat logs/vllm_server_*.log | tail -50
   ```

2. **检查显存占用**
   ```bash
   nvidia-smi  # 确认没有其他进程占用显存
   ```

3. **显存不足时的调整方案**
   编辑 `configs/baseline_0.json`:
   ```json
   {
     "server": {
       "gpu_memory_utilization": 0.85  // 从 0.90 降低
     },
     "model": {
       "max_model_len": 2048           // 从 4096 降低
     }
   }
   ```

4. **检查端口占用**
   ```bash
   lsof -i :8000   # 确认端口未被占用
   # 或换一个端口
   ```

5. **检查模型文件完整性**
   ```bash
   ls -la models/Qwen2.5-3B-Instruct-AWQ/  # 确认文件存在且大小正常
   ```

---

## Step 3: API 验证失败

### 症状
- `verify_server.py` 报连接拒绝
- 健康检查通过但 Chat 请求失败
- 返回空内容

### 排查步骤

1. **确认服务进程存在**
   ```bash
   cat logs/server.pid
   ps aux | grep vllm
   ```

2. **手动测试健康端点**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/v1/models
   ```

3. **手动发送 Chat 请求**
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Qwen/Qwen2.5-3B-Instruct-AWQ","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
   ```

4. **检查模型是否加载完成**
   - 在服务日志中搜索 "ready" 或 "started" 关键字
   - 模型加载需要 1-5 分钟，不要过早发送请求

---

## Step 4: 压测异常

### 症状
- 大量请求超时
- 成功率低
- 吞吐量异常低

### 排查步骤

1. **确认服务仍在运行**
   ```bash
   curl http://localhost:8000/health
   ```

2. **检查并发是否合理**
   - 首次测试建议 `concurrency=1`
   - 并发过高可能导致显存不足或排队超时

3. **放宽超时时间**
   编辑 `configs/baseline_0.json`:
   ```json
   {
     "benchmark": {
       "timeout_per_request": 300  // 从 120 增加到 300
     }
   }
   ```

4. **减少请求数先验证**
   ```bash
   python benchmarks/run_benchmark.py --num-requests 5 --concurrency 1
   ```

5. **检查 GPU 利用率**
   ```bash
   watch -n 1 nvidia-smi  # 观察 GPU 利用率和显存
   ```

---

## Step 5: 结果异常

### 症状
- TTFT 过高 (>10s)
- 吞吐量过低 (<0.1 req/s)
- 输出 token 数为 0

### 排查

1. **首次请求通常较慢** — CUDA kernel 编译和缓存预热，可先跑 warmup 再记录正式结果

2. **检查 `enforce_eager` 设置** — 设为 `true` 可跳过 CUDA graph 编译，避免首次延迟，但整体性能略降

3. **确认 prompt 格式** — 检查 `benchmarks/prompts.json` 中的 messages 格式是否符合模型要求

4. **排除机器负载干扰** — 确认没有其他 GPU 任务在运行
