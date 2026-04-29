#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launcher.py — VllmLauncher：vLLM 子进程的启动 / 停止 / 重启 + 就绪探测 + 计时

Task 6.1 核心模块。每次 trial 由 agent 调用 launcher.restart(config_path) 完成
"换配置 + 拉起 vLLM + 等 /health 200"，并把模型加载、CUDA graph 捕获、
端到端重启墙钟落到 LaunchResult。

设计要点:
- 使用 scripts/server/start_server.sh 拉起服务（保持与 Week 1-4 行为一致）；通过环境变量
  `ENFORCE_EAGER=1` 透传 --enforce-eager。
- stop() 只发 SIGTERM/SIGKILL，**不** drop_caches，让 OS 页缓存留给下次重启的模型权重读取。
- 子进程通过 nohup + setsid 后台运行（脚本里已经 disown），launcher 端只负责拿 PID + 轮询健康。
"""
from __future__ import annotations

import logging
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LaunchResult:
    """一次启动 / 重启的耗时与状态汇报。"""
    success: bool
    pid: int | None
    restart_wall_time_s: float
    model_load_time_s: float | None = None        # 从日志解析；解析不到为 None
    cuda_graph_capture_time_s: float | None = None
    error: str | None = None
    log_file: Path | None = None


# 用于从 vLLM 日志里抓"模型加载耗时" / "CUDA graph 捕获耗时"
# 例: "Loading model weights took 8.45 seconds" / "Capturing CUDA graph shapes: 100% ... 18.2s"
_MODEL_LOAD_RE = re.compile(
    r"(?:loading model weights took|model weights loaded in)\s+([\d.]+)\s*(?:s|seconds)",
    re.IGNORECASE,
)
_CUDA_GRAPH_RE = re.compile(
    r"(?:capturing cuda graph[^\n]*?|cuda graphs captured in)\s+([\d.]+)\s*(?:s|seconds)",
    re.IGNORECASE,
)


class VllmLauncher:
    """vLLM 服务子进程管理器。"""

    def __init__(
        self,
        project_dir: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        keep_page_cache: bool = True,
        log_dir: str | Path | None = None,
        start_script: str | Path | None = None,
    ):
        """
        Args:
            project_dir: 项目根目录（默认推断为本文件所在仓库根）。
            host/port: vLLM 服务地址。
            keep_page_cache: True 时 stop() 不调用 drop_caches；保留 OS 页缓存加速下次启动。
            log_dir: 日志目录（默认 <project_dir>/logs）。
            start_script: scripts/server/start_server.sh 路径（默认推断）。
        """
        if project_dir is None:
            project_dir = Path(__file__).resolve().parent.parent
        self.project_dir = Path(project_dir)
        self.host = host
        self.port = port
        self.keep_page_cache = keep_page_cache
        self.log_dir = Path(log_dir) if log_dir else self.project_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_script = (
            Path(start_script)
            if start_script
            else self.project_dir / "scripts" / "server" / "start_server.sh"
        )
        self._pid: int | None = None
        self._current_log_file: Path | None = None

    # ---------------- 公开 API ---------------- #
    def start(
        self,
        config_path: str | Path,
        enforce_eager: bool = False,
        ready_timeout_s: int = 300,
    ) -> LaunchResult:
        """启动一次 vLLM；返回 LaunchResult。已有进程在跑则先 stop。"""
        if self.is_alive():
            logger.info("检测到 vLLM 服务已在运行，先 stop 再 start")
            self.stop()

        config_path = Path(config_path)
        if not config_path.exists():
            return LaunchResult(False, None, 0.0, error=f"config 不存在: {config_path}")

        t0 = time.perf_counter()
        try:
            log_file = self._spawn_subprocess(config_path, enforce_eager)
        except Exception as e:
            return LaunchResult(False, None, time.perf_counter() - t0, error=f"spawn 失败: {e}")

        ready = self._wait_ready(ready_timeout_s)
        wall = time.perf_counter() - t0

        if not ready:
            self._read_pid_file()  # 尝试拿到 pid，便于 stop 兜底
            self.stop()
            return LaunchResult(
                False,
                self._pid,
                wall,
                error=f"等待 /health 超时 ({ready_timeout_s}s)",
                log_file=log_file,
            )

        self._read_pid_file()
        model_load_s, cuda_graph_s = self._parse_log_timings(log_file)
        return LaunchResult(
            success=True,
            pid=self._pid,
            restart_wall_time_s=round(wall, 3),
            model_load_time_s=model_load_s,
            cuda_graph_capture_time_s=cuda_graph_s,
            log_file=log_file,
        )

    def stop(self, grace_s: int = 10) -> bool:
        """SIGTERM → SIGKILL → 等端口释放。返回是否成功停止。"""
        if self._pid is None:
            self._read_pid_file()
        if self._pid is None and not self._port_in_use():
            return True

        if self._pid is not None and self._pid_alive(self._pid):
            try:
                os.kill(self._pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            t0 = time.time()
            while time.time() - t0 < grace_s and self._pid_alive(self._pid):
                time.sleep(0.5)
            if self._pid_alive(self._pid):
                logger.warning("SIGTERM 未生效，发 SIGKILL")
                try:
                    os.kill(self._pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        # 等端口释放
        t0 = time.time()
        while time.time() - t0 < grace_s and self._port_in_use():
            time.sleep(0.3)

        # 注意: keep_page_cache=True 时不做 drop_caches；此处仅日志说明
        if not self.keep_page_cache:
            logger.info("keep_page_cache=False，但出于安全考虑 launcher 不主动 drop_caches")

        self._pid = None
        return not self._port_in_use()

    def restart(
        self,
        config_path: str | Path,
        enforce_eager: bool = False,
        ready_timeout_s: int = 300,
    ) -> LaunchResult:
        """stop + start 的原子调用。"""
        self.stop()
        return self.start(config_path, enforce_eager=enforce_eager, ready_timeout_s=ready_timeout_s)

    def is_alive(self) -> bool:
        """端口在用 + (可选) PID 存活就视为活着。"""
        return self._port_in_use()

    # ---------------- 内部辅助 ---------------- #
    def _spawn_subprocess(self, config_path: Path, enforce_eager: bool) -> Path:
        """实际拉起 start_server.sh；返回 log 文件路径（从 stdout 第一行抓取）。"""
        env = os.environ.copy()
        if enforce_eager:
            env["ENFORCE_EAGER"] = "1"

        ts = time.strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"vllm_server_{ts}.log"
        self._current_log_file = log_file

        # start_server.sh 自身已经 nohup + disown；这里同步等它打印"OK"或"ERROR"
        cmd = ["bash", str(self.start_script), str(config_path)]
        proc = subprocess.run(
            cmd,
            cwd=str(self.project_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"start_server.sh 失败 (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
            )
        # log 文件名由脚本内部决定，再从 server.pid 旁边的 logs/ 取最新一份
        latest = self._find_latest_log()
        if latest is not None:
            self._current_log_file = latest
        return self._current_log_file

    def _wait_ready(self, timeout_s: int) -> bool:
        url = f"http://{self.host}:{self.port}/health"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if 200 <= resp.status < 300:
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(1.0)
        return False

    def _port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            try:
                s.connect((self.host, self.port))
                return True
            except OSError:
                return False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except OSError:
            return False

    def _read_pid_file(self) -> None:
        f = self.log_dir / "server.pid"
        if f.exists():
            try:
                self._pid = int(f.read_text().strip())
            except ValueError:
                self._pid = None

    def _find_latest_log(self) -> Path | None:
        candidates = sorted(self.log_dir.glob("vllm_server_*.log"))
        return candidates[-1] if candidates else None

    @staticmethod
    def _parse_log_timings(log_file: Path | None) -> tuple[float | None, float | None]:
        """从 vLLM 日志里 best-effort 抓模型加载 / CUDA graph 捕获耗时。"""
        if log_file is None or not log_file.exists():
            return None, None
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None, None
        m1 = _MODEL_LOAD_RE.search(text)
        m2 = _CUDA_GRAPH_RE.search(text)
        return (
            float(m1.group(1)) if m1 else None,
            float(m2.group(1)) if m2 else None,
        )
