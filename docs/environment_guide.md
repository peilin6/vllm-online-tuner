# 环境搭建指南

## 前置条件

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux (Ubuntu 20.04/22.04 推荐) |
| GPU | NVIDIA GPU，显存 ≥ 16GB（7B 模型 @ fp16） |
| CUDA | 12.1 及以上 |
| 驱动 | nvidia-smi 可用，驱动版本 ≥ 530 |
| Python | 3.10.x |
| 磁盘 | 模型权重约需 14GB，建议预留 30GB |

## 安装步骤

### 1. 环境初始化

```bash
bash scripts/setup/setup_env.sh
```

该脚本会：
- 创建 `.venv` 虚拟环境
- 安装 `requirements.txt` 中锁定的依赖
- 验证 PyTorch CUDA 可用性和 vLLM 版本

### 2. 模型下载

```bash
# 方式 A: huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-AWQ --local-dir ./models/Qwen2.5-3B-Instruct-AWQ

# 方式 B: modelscope（国内网络友好）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct-AWQ', cache_dir='./models')"
```

### 3. 服务启动

```bash
bash scripts/server/start_server.sh
```

### 4. 验证服务

```bash
python scripts/verify/verify_server.py
```

## 常见问题

| 问题 | 排查方向 |
|------|---------|
| `torch.cuda.is_available()` 返回 False | 检查 CUDA 版本与 PyTorch 编译版本是否匹配 |
| OOM (显存不够) | 降低 `gpu_memory_utilization` 或 `max_model_len` |
| 模型下载慢/失败 | 使用 modelscope 或配置 HF 镜像 |
| vLLM 版本不兼容 | 确认 PyTorch 版本，参考 vLLM release notes |
