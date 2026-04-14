#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vllm_metrics_collector.py — 后台线程定时采集 vLLM Prometheus metrics

采集 vLLM /metrics 端点暴露的关键指标：
- 运行中请求数 (num_requests_running)
- 等待中请求数 (num_requests_waiting)
- GPU KV cache 使用率 (gpu_cache_usage_pct)
- CPU KV cache 使用率 (cpu_cache_usage_pct)
- 抢占次数 (num_preemptions_total)

如果 /metrics 端点不可用，优雅降级并在输出中标注 source="unavailable"。
"""
import logging
import re
import threading
import time
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# vLLM 0.6.x Prometheus 指标名映射
_METRIC_PATTERNS = {
    "num_requests_running": re.compile(
        r'^vllm:num_requests_running\b.*?\s+([\d.]+)', re.MULTILINE
    ),
    "num_requests_waiting": re.compile(
        r'^vllm:num_requests_waiting\b.*?\s+([\d.]+)', re.MULTILINE
    ),
    "gpu_cache_usage_pct": re.compile(
        r'^vllm:gpu_cache_usage_perc\b.*?\s+([\d.]+)', re.MULTILINE
    ),
    "cpu_cache_usage_pct": re.compile(
        r'^vllm:cpu_cache_usage_perc\b.*?\s+([\d.]+)', re.MULTILINE
    ),
    "num_preemptions_total": re.compile(
        r'^vllm:num_preemptions_total\b.*?\s+([\d.]+)', re.MULTILINE
    ),
}


class VllmMetricsCollector:
    """后台线程定时采集 vLLM Prometheus metrics"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000",
                 interval_ms: int = 1000):
        """
        base_url: vLLM 服务地址
        interval_ms: 采集间隔（毫秒），建议 1000ms
        """
        self._metrics_url = f"{base_url}/metrics"
        self._interval_s = interval_ms / 1000.0
        self._samples: list[dict] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0
        self._available: bool | None = None  # None = 未检测

    def start(self):
        """启动后台采集线程"""
        self._samples = []
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self._available = None

        self._thread = threading.Thread(
            target=self._collect_loop,
            name="vllm-metrics-collector",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"vLLM metrics 采集器已启动，间隔 {self._interval_s * 1000:.0f}ms，"
                     f"端点: {self._metrics_url}")

    def stop(self) -> list[dict]:
        """
        停止采集并返回时序数据。
        每条记录至少包含：
        {
            "timestamp_s": float,
            "source": "vllm" | "unavailable",
            "num_requests_running": int,
            "num_requests_waiting": int,
            "gpu_cache_usage_pct": float,
            "cpu_cache_usage_pct": float,
            "num_preemptions_total": int
        }
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        result = list(self._samples)
        logger.info(f"vLLM metrics 采集器已停止，共采集 {len(result)} 个样本")
        return result

    def _collect_loop(self):
        """采集主循环"""
        while not self._stop_event.is_set():
            sample = self._fetch_metrics()
            if sample is not None:
                self._samples.append(sample)
            self._stop_event.wait(self._interval_s)

    def _fetch_metrics(self) -> dict | None:
        """拉取并解析一次 /metrics 端点"""
        timestamp_s = round(time.perf_counter() - self._start_time, 3)

        try:
            req = urllib.request.Request(self._metrics_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")

            if self._available is None:
                self._available = True
                logger.info("vLLM /metrics 端点可用")

            return self._parse_prometheus(body, timestamp_s)

        except urllib.error.URLError as e:
            if self._available is None:
                self._available = False
                logger.warning(f"vLLM /metrics 端点不可用: {e}")
            return {
                "timestamp_s": timestamp_s,
                "source": "unavailable",
                "num_requests_running": -1,
                "num_requests_waiting": -1,
                "gpu_cache_usage_pct": -1.0,
                "cpu_cache_usage_pct": -1.0,
                "num_preemptions_total": -1,
                "error": str(e),
            }
        except Exception as e:
            logger.warning(f"vLLM metrics 采集异常: {e}")
            return None

    @staticmethod
    def _parse_prometheus(text: str, timestamp_s: float) -> dict:
        """从 Prometheus 文本格式解析关键指标"""
        result = {
            "timestamp_s": timestamp_s,
            "source": "vllm",
        }

        for field, pattern in _METRIC_PATTERNS.items():
            match = pattern.search(text)
            if match:
                val = float(match.group(1))
                # 整数类型字段转 int
                if field in ("num_requests_running", "num_requests_waiting",
                             "num_preemptions_total"):
                    result[field] = int(val)
                else:
                    result[field] = round(val, 4)
            else:
                result[field] = -1 if "num_" in field else -1.0

        return result


if __name__ == "__main__":
    # 简单自测
    import sys
    logging.basicConfig(level=logging.INFO)

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    c = VllmMetricsCollector(base_url=base_url, interval_ms=1000)
    c.start()
    print("采集 3 秒...")
    time.sleep(3)
    samples = c.stop()
    print(f"{len(samples)} 个样本已采集")
    if samples:
        s = samples[0]
        print(f"来源: {s['source']}")
        if s["source"] == "vllm":
            print(f"  running: {s['num_requests_running']}, "
                  f"waiting: {s['num_requests_waiting']}, "
                  f"gpu_cache: {s['gpu_cache_usage_pct']}, "
                  f"cpu_cache: {s['cpu_cache_usage_pct']}")
        else:
            print(f"  /metrics 不可用: {s.get('error', 'N/A')}")
