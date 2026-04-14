#!/usr/bin/env python3
"""
collect_sysinfo.py — 采集硬件与软件环境信息

用法: python scripts/tools/collect_sysinfo.py [--output results/sysinfo.json]

采集项: GPU、CPU、内存、操作系统、CUDA、Python、PyTorch、vLLM 版本。
输出: JSON 文件 + 终端摘要。
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_gpu_info() -> list[dict]:
    """通过 nvidia-smi 采集 GPU 信息"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_mib": int(parts[2]),
                "driver_version": parts[3],
            })
        return gpus
    except Exception as e:
        return [{"error": str(e)}]


def get_cuda_version() -> str:
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.split("\n"):
            if "release" in line.lower():
                return line.strip().split("release")[-1].strip().rstrip(",").strip()
        return "unknown"
    except Exception:
        # fallback: 从 torch 获取
        try:
            import torch
            return torch.version.cuda or "unknown"
        except Exception:
            return "unknown"


def get_python_packages() -> dict:
    """获取关键 Python 包版本"""
    pkgs = {}
    for name in ["torch", "vllm", "numpy", "pandas", "aiohttp", "pynvml"]:
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            pkgs[name] = "not installed"
    # torch CUDA 信息
    try:
        import torch
        pkgs["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            pkgs["torch_cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return pkgs


def collect() -> dict:
    import psutil  # 如果没装则 fallback
    mem_total = "unknown"
    cpu_count = os.cpu_count()
    try:
        import psutil
        mem_total = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
    except ImportError:
        try:
            # Linux fallback
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        mem_total = f"{kb / (1024**2):.1f} GB"
                        break
        except Exception:
            pass

    return {
        "timestamp": datetime.now().isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "model": platform.processor() or "unknown",
            "count": cpu_count,
        },
        "memory_total": mem_total,
        "gpu": get_gpu_info(),
        "cuda_version": get_cuda_version(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": get_python_packages(),
    }


def main():
    parser = argparse.ArgumentParser(description="采集系统环境信息")
    parser.add_argument("--output", default="results/sysinfo.json",
                        help="输出文件路径 (默认: results/sysinfo.json)")
    args = parser.parse_args()

    info = collect()

    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # 终端摘要
    print("=" * 50)
    print(" 系统环境信息")
    print("=" * 50)
    print(f"  时间     : {info['timestamp']}")
    print(f"  系统     : {info['os']['system']} {info['os']['release']}")
    print(f"  CPU      : {info['cpu']['model']} ({info['cpu']['count']} cores)")
    print(f"  内存     : {info['memory_total']}")
    for g in info["gpu"]:
        if "error" in g:
            print(f"  GPU      : [采集失败] {g['error']}")
        else:
            print(f"  GPU[{g['index']}]   : {g['name']} ({g['memory_mib']} MiB), "
                  f"Driver {g['driver_version']}")
    print(f"  CUDA     : {info['cuda_version']}")
    print(f"  Python   : {info['python']['version']}")
    for pkg, ver in info["packages"].items():
        print(f"  {pkg:20s}: {ver}")
    print("=" * 50)
    print(f"  已保存到: {output_path}")


if __name__ == "__main__":
    main()
