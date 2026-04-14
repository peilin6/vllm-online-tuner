#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_monitor.py — 后台线程定时采样 GPU 指标

使用 pynvml 采集 GPU 利用率、显存、温度、功耗等实时数据。
在压测期间以 daemon 线程方式后台运行，不阻塞主进程退出。
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)

# 延迟导入 pynvml，允许在无 GPU 环境优雅降级
_pynvml = None


def _ensure_pynvml():
    """尝试导入并初始化 pynvml"""
    global _pynvml
    if _pynvml is not None:
        return True
    try:
        import pynvml
        pynvml.nvmlInit()
        _pynvml = pynvml
        return True
    except Exception as e:
        logger.warning(f"pynvml 初始化失败，GPU 监控不可用: {e}")
        return False


class GpuMonitor:
    """后台 daemon 线程定时采样 GPU 指标"""

    def __init__(self, interval_ms: int = 500, gpu_index: int = 0):
        """
        interval_ms: 采样间隔（毫秒），建议 500ms
        gpu_index: GPU 设备索引
        """
        self._interval_s = interval_ms / 1000.0
        self._gpu_index = gpu_index
        self._samples: list[dict] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0
        self._available = False

    def start(self):
        """启动后台采样线程"""
        self._available = _ensure_pynvml()
        if not self._available:
            logger.warning("GPU 监控降级：pynvml 不可用，将不采集 GPU 数据")
            return

        self._samples = []
        self._stop_event.clear()
        self._start_time = time.perf_counter()

        self._thread = threading.Thread(
            target=self._sample_loop,
            name="gpu-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"GPU 监控已启动，采样间隔 {self._interval_s * 1000:.0f}ms")

    def stop(self) -> list[dict]:
        """
        停止采样并返回时序数据。
        每条记录：
        {
            "timestamp_s": float,    # 相对于 start() 时的秒数
            "gpu_util_pct": float,   # GPU 利用率 %
            "mem_used_mib": float,   # 已用显存 MiB
            "mem_total_mib": float,  # 总显存 MiB
            "mem_util_pct": float,   # 显存利用率 %
            "temperature_c": int,    # GPU 温度 ℃
            "power_w": float         # 功耗 W
        }
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        result = list(self._samples)
        logger.info(f"GPU 监控已停止，共采集 {len(result)} 个样本")
        return result

    def _sample_loop(self):
        """采样主循环"""
        try:
            handle = _pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        except Exception as e:
            logger.error(f"无法获取 GPU {self._gpu_index} 句柄: {e}")
            return

        while not self._stop_event.is_set():
            try:
                sample = self._read_sample(handle)
                self._samples.append(sample)
            except Exception as e:
                logger.warning(f"GPU 采样异常: {e}")

            self._stop_event.wait(self._interval_s)

    def _read_sample(self, handle) -> dict:
        """读取一次 GPU 指标"""
        timestamp_s = time.perf_counter() - self._start_time

        # GPU 利用率
        try:
            util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
        except Exception:
            gpu_util = -1.0

        # 显存信息
        try:
            mem_info = _pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mib = mem_info.used / (1024 * 1024)
            mem_total_mib = mem_info.total / (1024 * 1024)
            mem_util_pct = (mem_info.used / mem_info.total * 100) if mem_info.total > 0 else 0.0
        except Exception:
            mem_used_mib = -1.0
            mem_total_mib = -1.0
            mem_util_pct = -1.0

        # 温度
        try:
            temperature_c = _pynvml.nvmlDeviceGetTemperature(
                handle, _pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            temperature_c = -1

        # 功耗
        try:
            power_mw = _pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = power_mw / 1000.0
        except Exception:
            power_w = -1.0

        return {
            "timestamp_s": round(timestamp_s, 3),
            "gpu_util_pct": float(gpu_util),
            "mem_used_mib": round(mem_used_mib, 1),
            "mem_total_mib": round(mem_total_mib, 1),
            "mem_util_pct": round(mem_util_pct, 1),
            "temperature_c": temperature_c,
            "power_w": round(power_w, 1),
        }


if __name__ == "__main__":
    # 简单自测
    logging.basicConfig(level=logging.INFO)
    m = GpuMonitor(interval_ms=500)
    m.start()
    time.sleep(3)
    samples = m.stop()
    print(f"{len(samples)} 个样本已采集")
    if samples:
        s = samples[0]
        print(f"GPU 利用率: {s['gpu_util_pct']}%, "
              f"显存: {s['mem_used_mib']}/{s['mem_total_mib']} MiB, "
              f"温度: {s['temperature_c']}℃, "
              f"功耗: {s['power_w']}W")
    else:
        print("未采集到样本（pynvml 可能不可用）")
