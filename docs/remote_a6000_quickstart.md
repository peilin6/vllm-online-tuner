# A6000 服务器快速连接与实验流程

> 本文只记录连接别名和实验步骤，不保存密码。密码只在 SSH 交互提示时输入。

## 1. 本机 SSH 别名

已在 Windows 用户 SSH 配置中新增：

- `nus3090`：3090 跳板机
- `xtraa6k04`：A6000 实验机，经 `nus3090` 跳转

若连接失败，优先检查 `cloudflared`、NUS 跳板机密码、A6000 账号密码，以及目标主机是否允许从跳板机解析 `xtraa6k04`。

## 2. 登录后准备代码

首次登录到 `xtraa6k04` 后，建议在服务器上使用 GitHub 作为代码备份源：

1. 克隆或更新仓库到 `~/vlllm`
2. 进入仓库根目录
3. 创建并激活 `~/vllm-venv-a6000`
4. 安装 `requirements.txt` 与 vLLM 0.6.6.post1 / PyTorch 2.5.1
5. 下载模型到 `~/models/Qwen3-8B`

## 3. 运行 A6000 baseline

进入服务器上的仓库根目录后执行：

```bash
export no_proxy="*"
source ~/vllm-venv-a6000/bin/activate
bash scripts/experiment/run_a6000_baseline.sh
```

完成后重点查看：

- `results/baseline_a6000_0/summary.txt`
- `results/baseline_a6000_0/summary.json`
- `logs/vllm_server_*.log`

## 4. 注意事项

- 做完实验或暂时不用时，执行 `bash scripts/server/stop_server.sh` 关闭 vLLM。
- NUS 机器上的代码和结果需要及时 `git push` 或下载备份。
- 同一张卡不要同时运行两个 vLLM 实例。