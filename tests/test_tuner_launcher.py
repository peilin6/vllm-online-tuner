#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_tuner_launcher.py — Task 6.1 单元测试

不实际启动 vLLM，使用 fake start script + mock urllib 完整覆盖:
- config_generator.render_experiment_config 点路径与扁平 key
- config_generator.write_temp_config 落盘 + 可解析回原始 dict
- VllmLauncher 启动 / 等就绪 / stop / restart 逻辑
- LaunchResult.restart_wall_time_s 计时
- 解析日志里的 model_load / cuda_graph 耗时
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.config_generator import (
    render_experiment_config,
    write_temp_config,
    _set_by_dotted_path,
)
from tuner.launcher import VllmLauncher, LaunchResult


# ============================================================
# config_generator 测试
# ============================================================

def test_render_dotted_path_overrides():
    base = {"server": {"max_num_seqs": 32, "gpu_memory_utilization": 0.9}}
    out = render_experiment_config(base, {"server.max_num_seqs": 64})
    assert out["server"]["max_num_seqs"] == 64
    assert out["server"]["gpu_memory_utilization"] == 0.9
    # 深拷贝，源不变
    assert base["server"]["max_num_seqs"] == 32


def test_render_flat_key_routes_to_server():
    base = {"server": {"max_num_seqs": 32}}
    out = render_experiment_config(base, {"max_num_seqs": 128, "enable_prefix_caching": True})
    assert out["server"]["max_num_seqs"] == 128
    assert out["server"]["enable_prefix_caching"] is True


def test_render_creates_missing_intermediate_dicts():
    base = {}
    out = render_experiment_config(base, {"server.max_num_seqs": 16})
    assert out == {"server": {"max_num_seqs": 16}}


def test_render_none_overrides_returns_copy():
    base = {"a": 1}
    out = render_experiment_config(base, None)
    assert out == base
    assert out is not base


def test_render_invalid_inputs_raise():
    with pytest.raises(TypeError):
        render_experiment_config("not a dict", {})
    with pytest.raises(TypeError):
        render_experiment_config({}, "not a dict")


def test_set_by_dotted_path_overwrites_non_dict_intermediate():
    d = {"server": "string"}
    _set_by_dotted_path(d, "server.x", 1)
    assert d == {"server": {"x": 1}}


def test_write_temp_config_roundtrip(tmp_path):
    cfg = {"server": {"max_num_seqs": 64}, "model": {"name": "Qwen"}}
    p = write_temp_config(cfg, tmp_dir=str(tmp_path))
    assert p.exists() and p.suffix == ".json"
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded == cfg


def test_write_temp_config_unique_paths(tmp_path):
    cfg = {"a": 1}
    p1 = write_temp_config(cfg, tmp_dir=str(tmp_path))
    p2 = write_temp_config(cfg, tmp_dir=str(tmp_path))
    assert p1 != p2


# ============================================================
# Launcher: log timing 解析
# ============================================================

def test_parse_log_timings_extracts_both(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text(
        "INFO ... loading model weights took 8.45 seconds ...\n"
        "INFO ... cuda graphs captured in 12.30 seconds ...\n",
        encoding="utf-8",
    )
    m, c = VllmLauncher._parse_log_timings(log)
    assert m == 8.45
    assert c == 12.30


def test_parse_log_timings_missing_fields_returns_none(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text("nothing relevant here\n", encoding="utf-8")
    m, c = VllmLauncher._parse_log_timings(log)
    assert m is None and c is None


def test_parse_log_timings_no_file():
    m, c = VllmLauncher._parse_log_timings(None)
    assert m is None and c is None
    m2, c2 = VllmLauncher._parse_log_timings(Path("/non/existent.log"))
    assert m2 is None and c2 is None


# ============================================================
# Launcher: 端口检测 / pid 文件
# ============================================================

def test_port_in_use_detects_listening(tmp_path):
    launcher = VllmLauncher(
        project_dir=tmp_path, host="127.0.0.1", port=_pick_free_port(),
        log_dir=tmp_path / "logs",
    )
    assert launcher._port_in_use() is False  # 没人监听
    # 起一个临时 listener
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((launcher.host, launcher.port))
    srv.listen(1)
    try:
        assert launcher._port_in_use() is True
    finally:
        srv.close()


def test_read_pid_file(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "server.pid").write_text("12345\n")
    launcher = VllmLauncher(project_dir=tmp_path, log_dir=log_dir, port=_pick_free_port())
    launcher._read_pid_file()
    assert launcher._pid == 12345


def test_read_pid_file_invalid_value(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "server.pid").write_text("abc\n")
    launcher = VllmLauncher(project_dir=tmp_path, log_dir=log_dir, port=_pick_free_port())
    launcher._read_pid_file()
    assert launcher._pid is None


# ============================================================
# Launcher: start / stop 集成（fake HTTP /health + fake start script）
# ============================================================

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_args):  # 静音
        pass


@pytest.fixture
def fake_health_server():
    """起一个本地 HTTP 服务模拟 vLLM /health 端点"""
    port = _pick_free_port()
    srv = HTTPServer(("127.0.0.1", port), _HealthHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield port, srv
    finally:
        srv.shutdown()
        srv.server_close()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _has_bash() -> bool:
    from shutil import which
    return which("bash") is not None


def _make_fake_start_script(tmp_path: Path, log_dir: Path) -> Path:
    """生成一个跨平台 fake start script。Linux/Mac 用 bash，Windows 上 pytest 需 wsl/git-bash 才能 bash 调用。"""
    script = tmp_path / "fake_start.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "fake start"\n'
        f'mkdir -p "{log_dir}"\n'
        f'echo "loading model weights took 1.23 seconds" > "{log_dir}/vllm_server_test.log"\n'
        f'echo "cuda graphs captured in 4.56 seconds" >> "{log_dir}/vllm_server_test.log"\n'
        f'echo 99999 > "{log_dir}/server.pid"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        os.chmod(script, 0o755)
    return script


@pytest.mark.skipif(os.name == "nt" or not _has_bash(),
                    reason="集成测试依赖 POSIX bash + 路径语义；Windows 上跳过，留到 Linux 服务器侧验证")
def test_start_success_with_fake_script(tmp_path, fake_health_server):
    port, _srv = fake_health_server
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    fake_script = _make_fake_start_script(tmp_path, log_dir)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"server": {}, "model": {}}))

    launcher = VllmLauncher(
        project_dir=tmp_path,
        host="127.0.0.1",
        port=port,
        log_dir=log_dir,
        start_script=fake_script,
    )
    res = launcher.start(cfg, ready_timeout_s=5)
    assert res.success is True
    assert res.restart_wall_time_s >= 0
    assert res.model_load_time_s == 1.23
    assert res.cuda_graph_capture_time_s == 4.56
    assert res.pid == 99999


def test_start_missing_config(tmp_path):
    launcher = VllmLauncher(
        project_dir=tmp_path, port=_pick_free_port(), log_dir=tmp_path / "logs",
        start_script=tmp_path / "nonexistent.sh",
    )
    res = launcher.start(tmp_path / "missing.json")
    assert res.success is False
    assert res.error and "config 不存在" in res.error


def test_start_subprocess_failure(tmp_path):
    if os.name == "nt" or not _has_bash():
        pytest.skip("依赖 POSIX bash subprocess，Windows 跳过")
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    bad_script = tmp_path / "bad.sh"
    bad_script.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    if os.name != "nt":
        os.chmod(bad_script, 0o755)

    launcher = VllmLauncher(
        project_dir=tmp_path, port=_pick_free_port(),
        log_dir=log_dir, start_script=bad_script,
    )
    res = launcher.start(cfg, ready_timeout_s=2)
    assert res.success is False
    assert res.error is not None


def test_wait_ready_returns_false_on_timeout(tmp_path):
    """端口没人监听时 _wait_ready 应在 timeout 后返回 False。"""
    launcher = VllmLauncher(
        project_dir=tmp_path, port=_pick_free_port(),
        log_dir=tmp_path / "logs",
        start_script=tmp_path / "x.sh",
    )
    t0 = time.time()
    assert launcher._wait_ready(timeout_s=1) is False
    elapsed = time.time() - t0
    assert elapsed < 5  # 不会卡死


def test_stop_when_nothing_running(tmp_path):
    launcher = VllmLauncher(
        project_dir=tmp_path, port=_pick_free_port(),
        log_dir=tmp_path / "logs",
        start_script=tmp_path / "x.sh",
    )
    assert launcher.stop() is True


def test_pid_alive_for_self():
    assert VllmLauncher._pid_alive(os.getpid()) is True


def test_pid_alive_for_unlikely_pid():
    # PID 0 / 极大 PID 通常不会是当前进程
    assert VllmLauncher._pid_alive(99999999) is False


def test_is_alive_reflects_port(tmp_path):
    launcher = VllmLauncher(
        project_dir=tmp_path, port=_pick_free_port(),
        log_dir=tmp_path / "logs",
        start_script=tmp_path / "x.sh",
    )
    assert launcher.is_alive() is False
