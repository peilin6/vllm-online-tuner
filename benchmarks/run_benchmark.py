#!/usr/bin/env python3
"""
run_benchmark.py — vLLM 在线推理性能压测脚本

用法:
  python benchmarks/run_benchmark.py
  python benchmarks/run_benchmark.py --config configs/experiments/baseline_0.json
  python benchmarks/run_benchmark.py --concurrency 4 --num-requests 100

输出:
  - 终端实时进度与摘要
  - results/benchmark_<timestamp>.json  (机器可读完整结果)
  - results/benchmark_<timestamp>.txt   (人可读摘要)

核心指标: 吞吐量, TTFT, 平均时延, P95 时延, 成功率
"""
import argparse
import asyncio
import copy
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

from workloads.workload_generator import WorkloadGenerator
from monitors.gpu_monitor import GpuMonitor
from monitors.vllm_metrics_collector import VllmMetricsCollector


# ========== 配置 ==========

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts(prompts_path: str) -> list[dict]:
    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ========== 单次请求 ==========

async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict],
    sampling: dict,
    timeout: int,
) -> dict:
    """发送一次 chat/completions 请求并记录时延"""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": sampling.get("max_tokens", 256),
        "temperature": sampling.get("temperature", 0.7),
        "top_p": sampling.get("top_p", 0.9),
        "stream": True,  # 用 stream 来测量 TTFT
        "stream_options": {"include_usage": True},  # 让最后一个 chunk 返回精确 token 计数
    }

    result = {
        "success": False,
        "ttft_ms": None,
        "latency_ms": None,
        "output_tokens": 0,
        "output_tokens_source": "chunk_count",
        "tpot_ms": None,
        "tpot_p95_ms": None,
        "token_timestamps_ms": [],
        "error": None,
    }

    try:
        t_start = time.perf_counter()
        first_token_time = None
        token_count = 0
        usage_tokens = None
        token_timestamps = []  # 每个 token 的绝对时间戳

        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                result["error"] = f"HTTP {resp.status}"
                return result

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    # 尝试从 usage 字段获取精确 token 数
                    if chunk.get("usage"):
                        ct = chunk["usage"].get("completion_tokens")
                        if ct is not None:
                            usage_tokens = ct
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content"):
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        token_timestamps.append(now)
                        token_count += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        t_end = time.perf_counter()
        result["success"] = True
        result["latency_ms"] = (t_end - t_start) * 1000
        # 优先使用 usage 精确值，fallback 到 chunk 计数
        if usage_tokens is not None:
            result["output_tokens"] = usage_tokens
            result["output_tokens_source"] = "usage"
        else:
            result["output_tokens"] = token_count
            result["output_tokens_source"] = "chunk_count"
        if first_token_time is not None:
            result["ttft_ms"] = (first_token_time - t_start) * 1000
        # TPOT 计算：逐 token 间隔的统计值
        result["token_timestamps_ms"] = [
            (ts - t_start) * 1000 for ts in token_timestamps
        ]
        if len(token_timestamps) >= 2:
            intervals = [
                (token_timestamps[i] - token_timestamps[i - 1]) * 1000
                for i in range(1, len(token_timestamps))
            ]
            result["tpot_ms"] = statistics.mean(intervals)
            sorted_intervals = sorted(intervals)
            result["tpot_p95_ms"] = sorted_intervals[
                int(len(sorted_intervals) * 0.95)
            ]

    except asyncio.TimeoutError:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)

    return result


# ========== 并发压测驱动 ==========

async def run_benchmark(
    base_url: str,
    model: str,
    prompts: list[dict],
    sampling: dict,
    num_requests: int,
    concurrency: int,
    timeout: int,
    request_rate: float | None = None,
) -> list[dict]:
    """按指定并发数发送 num_requests 个请求"""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    # 循环复用 prompts
    tasks = []

    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            messages = prompt["messages"]

            async def _task(msgs=messages, idx=i):
                if request_rate is not None and idx > 0:
                    await asyncio.sleep(idx / request_rate)
                async with semaphore:
                    r = await send_request(
                        session, base_url, model, msgs, sampling, timeout,
                    )
                    r["request_id"] = idx
                    return r

            tasks.append(_task())

        # 进度显示
        completed = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            completed += 1
            if completed % max(1, total // 10) == 0 or completed == total:
                print(f"  进度: {completed}/{total} "
                      f"({completed * 100 // total}%)")

    return results


# ========== Workload 模式压测驱动 ==========

async def run_benchmark_workload(
    base_url: str,
    model: str,
    workload_requests: list[dict],
    sampling: dict,
    concurrency: int,
    timeout: int,
) -> list[dict]:
    """使用 WorkloadGenerator 生成的请求序列进行压测，按 scheduled_time_s 控制注入时间"""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        t_start = time.perf_counter()

        for wreq in workload_requests:
            sched_time = wreq["scheduled_time_s"]
            messages = wreq["messages"]
            max_tokens = wreq.get("max_tokens", wreq.get("target_max_tokens", 256))

            # 为每个请求覆盖 max_tokens
            req_sampling = copy.copy(sampling)
            req_sampling["max_tokens"] = max_tokens

            async def _task(msgs=messages, stime=sched_time, s=req_sampling,
                            meta=wreq):
                # 等待到调度时间
                elapsed = time.perf_counter() - t_start
                wait = stime - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)

                async with semaphore:
                    r = await send_request(
                        session, base_url, model, msgs, s, timeout,
                    )
                    # 附加 workload 元数据
                    r["request_id"] = meta["request_id"]
                    r["prompt_length_bucket"] = meta.get("prompt_length_bucket")
                    r["target_max_tokens"] = meta.get("target_max_tokens")
                    r["shared_prefix_group"] = meta.get("shared_prefix_group")
                    r["phase_name"] = meta.get("phase_name", "default")
                    r["is_warmup"] = meta.get("is_warmup", False)
                    r["is_cooldown"] = meta.get("is_cooldown", False)
                    r["scheduled_time_s"] = stime
                    return r

            tasks.append(_task())

        # 进度显示
        completed = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            completed += 1
            if completed % max(1, total // 10) == 0 or completed == total:
                print(f"  进度: {completed}/{total} "
                      f"({completed * 100 // total}%)")

    return results


# ========== 统计汇总 ==========

def compute_stats(results: list[dict], wall_time_s: float = 0) -> dict:
    """计算汇总统计指标"""
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    latencies = [r["latency_ms"] for r in successes if r["latency_ms"] is not None]
    ttfts = [r["ttft_ms"] for r in successes if r["ttft_ms"] is not None]
    output_tokens = [r["output_tokens"] for r in successes]

    # 使用实际墙钟时间计算吞吐量，而非 max(latencies)
    total_time_s = wall_time_s if wall_time_s > 0 else ((max(latencies) / 1000) if latencies else 0)

    stats = {
        "total_requests": len(results),
        "successful": len(successes),
        "failed": len(failures),
        "success_rate": len(successes) / len(results) if results else 0,
    }

    if latencies:
        sorted_lat = sorted(latencies)
        stats["latency_ms"] = {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": sorted_lat[int(len(sorted_lat) * 0.95)],
            "p99": sorted_lat[int(len(sorted_lat) * 0.99)],
        }
        # 吞吐量: 成功请求数 / 总耗时
        if total_time_s > 0:
            stats["throughput_rps"] = len(successes) / total_time_s

    if ttfts:
        sorted_ttft = sorted(ttfts)
        stats["ttft_ms"] = {
            "min": min(ttfts),
            "max": max(ttfts),
            "mean": statistics.mean(ttfts),
            "median": statistics.median(ttfts),
            "p95": sorted_ttft[int(len(sorted_ttft) * 0.95)],
        }

    if output_tokens:
        stats["output_tokens"] = {
            "total": sum(output_tokens),
            "mean": statistics.mean(output_tokens),
        }
        if total_time_s > 0:
            stats["token_throughput_tps"] = sum(output_tokens) / total_time_s

    # TPOT 聚合
    tpots = [r["tpot_ms"] for r in successes if r.get("tpot_ms") is not None]
    if tpots:
        sorted_tpots = sorted(tpots)
        stats["tpot_ms"] = {
            "min": min(tpots),
            "max": max(tpots),
            "mean": statistics.mean(tpots),
            "median": statistics.median(tpots),
            "p95": sorted_tpots[int(len(sorted_tpots) * 0.95)],
            "p99": sorted_tpots[int(len(sorted_tpots) * 0.99)],
        }

    if failures:
        error_counts: dict[str, int] = {}
        for r in failures:
            err = r.get("error", "unknown")
            error_counts[err] = error_counts.get(err, 0) + 1
        stats["errors"] = error_counts

    return stats


def format_summary(stats: dict, config_info: dict) -> str:
    """生成人可读摘要"""
    lines = [
        "=" * 60,
        " vLLM Baseline 压测结果摘要",
        "=" * 60,
        f"  时间       : {config_info.get('timestamp', 'N/A')}",
        f"  模型       : {config_info.get('model', 'N/A')}",
        f"  并发       : {config_info.get('concurrency', 'N/A')}",
        f"  请求数     : {stats['total_requests']}",
        f"  成功/失败  : {stats['successful']}/{stats['failed']}",
        f"  成功率     : {stats['success_rate']:.1%}",
        "",
    ]

    if "throughput_rps" in stats:
        lines.append(f"  吞吐量     : {stats['throughput_rps']:.2f} req/s")
    if "token_throughput_tps" in stats:
        lines.append(f"  Token吞吐  : {stats['token_throughput_tps']:.2f} tokens/s")

    if "ttft_ms" in stats:
        t = stats["ttft_ms"]
        lines += [
            "",
            "  首Token时延 (TTFT):",
            f"    Mean   : {t['mean']:.1f} ms",
            f"    Median : {t['median']:.1f} ms",
            f"    P95    : {t['p95']:.1f} ms",
        ]

    if "tpot_ms" in stats:
        tp = stats["tpot_ms"]
        lines += [
            "",
            "  输出Token间隔 (TPOT):",
            f"    Mean   : {tp['mean']:.1f} ms",
            f"    Median : {tp['median']:.1f} ms",
            f"    P95    : {tp['p95']:.1f} ms",
        ]

    if "latency_ms" in stats:
        l = stats["latency_ms"]
        lines += [
            "",
            "  端到端时延:",
            f"    Mean   : {l['mean']:.1f} ms",
            f"    Median : {l['median']:.1f} ms",
            f"    P95    : {l['p95']:.1f} ms",
            f"    P99    : {l['p99']:.1f} ms",
            f"    Min    : {l['min']:.1f} ms",
            f"    Max    : {l['max']:.1f} ms",
        ]

    if "output_tokens" in stats:
        lines += [
            "",
            f"  输出Tokens : 总计 {stats['output_tokens']['total']}, "
            f"平均 {stats['output_tokens']['mean']:.1f}/req",
        ]

    if "errors" in stats:
        lines += ["", "  错误分布:"]
        for err, cnt in stats["errors"].items():
            lines.append(f"    {err}: {cnt}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ========== 主流程 ==========

def _build_request_trace(results: list[dict], save_timestamps: bool) -> list[dict]:
    """构建 request_trace.jsonl 的记录列表"""
    traces = []
    for r in results:
        trace = {
            "request_id": r.get("request_id"),
            "scheduled_time_s": r.get("scheduled_time_s"),
            "success": r["success"],
            "ttft_ms": r.get("ttft_ms"),
            "tpot_ms": r.get("tpot_ms"),
            "tpot_p95_ms": r.get("tpot_p95_ms"),
            "latency_ms": r.get("latency_ms"),
            "output_tokens": r.get("output_tokens", 0),
            "output_tokens_source": r.get("output_tokens_source"),
            "prompt_length_bucket": r.get("prompt_length_bucket"),
            "target_max_tokens": r.get("target_max_tokens"),
            "shared_prefix_group": r.get("shared_prefix_group"),
            "phase_name": r.get("phase_name", "default"),
            "is_warmup": r.get("is_warmup", False),
            "is_cooldown": r.get("is_cooldown", False),
            "error": r.get("error"),
        }
        if save_timestamps:
            trace["token_timestamps_ms"] = r.get("token_timestamps_ms", [])
        traces.append(trace)
    return traces


def _build_metrics_timeseries(gpu_samples: list[dict],
                               vllm_samples: list[dict]) -> list[dict]:
    """合并 GPU 和 vLLM metrics 为统一时序序列"""
    timeseries = []
    for s in gpu_samples:
        entry = {"source": "gpu"}
        entry.update(s)
        timeseries.append(entry)
    for s in vllm_samples:
        if "source" not in s:
            s["source"] = "vllm"
        timeseries.append(s)
    # 按时间排序
    timeseries.sort(key=lambda x: x.get("timestamp_s", 0))
    return timeseries


def main():
    parser = argparse.ArgumentParser(description="vLLM 在线推理压测")
    parser.add_argument("--config", default="configs/experiments/baseline_0.json",
                        help="实验配置文件")
    parser.add_argument("--prompts", default="benchmarks/prompts.json",
                        help="请求样本文件")
    parser.add_argument("--workload", default=None,
                        help="workload 配置文件路径（启用 WorkloadGenerator 模式）")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--request-rate", type=float, default=None,
                        help="请求速率 (req/s)，None=burst")
    parser.add_argument("--save-token-timestamps", action="store_true",
                        default=False,
                        help="是否在结果中保存逐 token 时间戳数组")
    parser.add_argument("--experiment-id", default=None,
                        help="实验 ID（用于子目录命名），默认自动生成")
    parser.add_argument("--no-gpu-monitor", action="store_true", default=False,
                        help="禁用 GPU 监控")
    parser.add_argument("--no-vllm-metrics", action="store_true", default=False,
                        help="禁用 vLLM metrics 采集")
    parser.add_argument("--output-dir", default="results",
                        help="结果输出目录")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    host = args.host or config["server"]["host"].replace("0.0.0.0", "localhost")
    port = args.port or config["server"]["port"]
    model = config["model"]["name"]
    sampling = config["sampling"]
    concurrency = args.concurrency or config["benchmark"]["concurrency"]
    timeout = config["benchmark"]["timeout_per_request"]
    base_url = f"http://{host}:{port}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 实验 ID
    experiment_id = args.experiment_id or f"exp_{timestamp}"

    # ========== 启动监控器 ==========
    gpu_monitor = None
    vllm_collector = None

    if not args.no_gpu_monitor:
        gpu_monitor = GpuMonitor(interval_ms=500)
    if not args.no_vllm_metrics:
        vllm_collector = VllmMetricsCollector(base_url=base_url, interval_ms=1000)

    # ========== Workload 模式 ==========
    if args.workload:
        workload_config = load_config(args.workload)
        wl = workload_config["workload"]
        num_requests = wl["num_requests"]

        wg = WorkloadGenerator(wl)
        workload_requests = wg.generate()

        print("=" * 60)
        print(" vLLM 推理性能压测 (Workload 模式)")
        print("=" * 60)
        print(f"  实验 ID   : {experiment_id}")
        print(f"  服务地址  : {base_url}")
        print(f"  模型      : {model}")
        print(f"  并发      : {concurrency}")
        print(f"  请求数    : {num_requests}")
        print(f"  Workload  : {wl['name']}")
        print(f"  到达模式  : {wl['arrival']['pattern']}")
        print(f"  预热/冷却 : {wl.get('warmup_requests', 0)}/{wl.get('cooldown_requests', 0)}")
        print(f"  GPU监控   : {'启用' if gpu_monitor else '禁用'}")
        print(f"  vLLM采集  : {'启用' if vllm_collector else '禁用'}")
        print("=" * 60)

        # 启动监控器
        if gpu_monitor:
            gpu_monitor.start()
        if vllm_collector:
            vllm_collector.start()

        print("\n开始压测...\n")
        t0 = time.time()
        results = asyncio.run(run_benchmark_workload(
            base_url=base_url,
            model=model,
            workload_requests=workload_requests,
            sampling=sampling,
            concurrency=concurrency,
            timeout=timeout,
        ))
        wall_time = time.time() - t0

        # 停止监控器
        gpu_samples = gpu_monitor.stop() if gpu_monitor else []
        vllm_samples = vllm_collector.stop() if vllm_collector else []

        # 过滤掉 warmup 和 cooldown 请求，用于统计
        stats_results = [r for r in results
                         if not r.get("is_warmup") and not r.get("is_cooldown")]
        stats = compute_stats(stats_results, wall_time_s=wall_time)
        stats["wall_time_s"] = wall_time
        stats["warmup_excluded"] = sum(1 for r in results if r.get("is_warmup"))
        stats["cooldown_excluded"] = sum(1 for r in results if r.get("is_cooldown"))
        stats["gpu_samples_count"] = len(gpu_samples)
        stats["vllm_metrics_samples_count"] = len(vllm_samples)

        config_info = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "model": model,
            "concurrency": concurrency,
            "num_requests": num_requests,
            "config_file": args.config,
            "workload_file": args.workload,
            "workload_config": wl,
        }

    # ========== 经典模式（向后兼容） ==========
    else:
        prompts = load_prompts(args.prompts)
        num_requests = args.num_requests or config["benchmark"]["num_requests"]
        request_rate = args.request_rate or config["benchmark"].get("request_rate")

        print("=" * 60)
        print(" vLLM 推理性能压测")
        print("=" * 60)
        print(f"  实验 ID  : {experiment_id}")
        print(f"  服务地址 : {base_url}")
        print(f"  模型     : {model}")
        print(f"  并发     : {concurrency}")
        print(f"  请求数   : {num_requests}")
        print(f"  请求速率 : {request_rate or 'burst (不限速)'}")
        print(f"  样本数   : {len(prompts)} 条 prompt")
        print(f"  GPU监控  : {'启用' if gpu_monitor else '禁用'}")
        print(f"  vLLM采集 : {'启用' if vllm_collector else '禁用'}")
        print("=" * 60)

        # 启动监控器
        if gpu_monitor:
            gpu_monitor.start()
        if vllm_collector:
            vllm_collector.start()

        print("\n开始压测...\n")
        t0 = time.time()
        results = asyncio.run(run_benchmark(
            base_url=base_url,
            model=model,
            prompts=prompts,
            sampling=sampling,
            num_requests=num_requests,
            concurrency=concurrency,
            timeout=timeout,
            request_rate=request_rate,
        ))
        wall_time = time.time() - t0

        # 停止监控器
        gpu_samples = gpu_monitor.stop() if gpu_monitor else []
        vllm_samples = vllm_collector.stop() if vllm_collector else []

        stats = compute_stats(results, wall_time_s=wall_time)
        stats["wall_time_s"] = wall_time
        stats["gpu_samples_count"] = len(gpu_samples)
        stats["vllm_metrics_samples_count"] = len(vllm_samples)

        config_info = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "model": model,
            "concurrency": concurrency,
            "num_requests": num_requests,
            "request_rate": request_rate,
            "config_file": args.config,
            "prompts_file": args.prompts,
        }

    # ========== 输出: 新结构化格式 ==========
    output_base = Path(args.output_dir)
    exp_dir = output_base / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. config_snapshot.json — 完整实验配置快照
    config_snapshot = {
        "experiment": config_info,
        "server_config": config,
        "sampling": sampling,
    }
    config_snap_path = exp_dir / "config_snapshot.json"
    with open(config_snap_path, "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    # 2. request_trace.jsonl — 请求级 trace
    traces = _build_request_trace(results, args.save_token_timestamps)
    trace_path = exp_dir / "request_trace.jsonl"
    with open(trace_path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # 3. metrics_timeseries.jsonl — 时序指标
    timeseries = _build_metrics_timeseries(gpu_samples, vllm_samples)
    metrics_path = exp_dir / "metrics_timeseries.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for m in timeseries:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # 4. summary.json — 聚合统计结果
    summary_data = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "config": config_info,
        "stats": stats,
    }
    summary_path = exp_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # 5. 人可读摘要（向后兼容，同时写到实验目录和 results 根目录）
    summary_text = format_summary(stats, config_info)
    txt_path = exp_dir / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # 向后兼容: results/benchmark_<timestamp>.txt
    compat_txt_path = output_base / f"benchmark_{timestamp}.txt"
    with open(compat_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\n结果已保存到: {exp_dir}/")
    print(f"  配置快照      : {config_snap_path.name}")
    print(f"  请求 Trace    : {trace_path.name} ({len(traces)} 条)")
    print(f"  时序指标      : {metrics_path.name} ({len(timeseries)} 条)")
    print(f"  统计摘要      : {summary_path.name}")
    print(f"  可读摘要      : {txt_path.name}")
    if gpu_samples:
        print(f"  GPU 采样      : {len(gpu_samples)} 个样本")
    if vllm_samples:
        print(f"  vLLM metrics  : {len(vllm_samples)} 个样本")


if __name__ == "__main__":
    main()
