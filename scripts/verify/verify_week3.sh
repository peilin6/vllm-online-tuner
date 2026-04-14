#!/bin/bash
# Week 3 各 Task 验证脚本
source /home/lpl/vllm-venv/bin/activate
export no_proxy="*"
cd /mnt/d/vlllm

echo "=== Task 3.1: 验证 workload 配置文件 ==="
python3 << 'PYEOF'
import json, glob

files = sorted(glob.glob("configs/workload_*.json")) + ["configs/workload_schema.json"]
for f in files:
    d = json.load(open(f))
    if "workload" in d:
        print(f"  OK: {f} (name={d['workload']['name']})")
    else:
        print(f"  OK: {f} (schema)")
print(f"  共 {len(files)} 个文件全部验证通过")
PYEOF

echo ""
echo "=== Task 3.2: 验证 prompt 语料池 ==="
python3 << 'PYEOF'
import json
from collections import Counter
d = json.load(open("benchmarks/prompts_pool.json"))
prompts = d["prompts"]
cats = Counter(p["category"] for p in prompts)
print(f"  总计 {len(prompts)} 条 prompt")
for cat, cnt in sorted(cats.items()):
    print(f"    {cat}: {cnt} 条")
assert len(prompts) >= 30, f"不足 30 条: {len(prompts)}"
print("  验证通过: ≥30 条")
PYEOF

echo ""
echo "=== Task 3.3: 验证共享前缀池 ==="
python3 << 'PYEOF'
import json
d = json.load(open("benchmarks/prefix_pool.json"))
prefixes = d["prefixes"]
print(f"  总计 {len(prefixes)} 组前缀")
for p in prefixes:
    print(f"    {p['id']}: {len(p['suffix_pool'])} 个后缀, ~{p['estimated_prefix_tokens']} tokens")
assert len(prefixes) >= 5, f"不足 5 组: {len(prefixes)}"
print("  验证通过: ≥5 组")
PYEOF

echo ""
echo "=== Task 3.4: 验证 WorkloadGenerator 三种到达模式 ==="
python3 << 'PYEOF'
from workloads.workload_generator import WorkloadGenerator
import json

# burst 模式
cfg = json.load(open("configs/workloads/workload_burst.json"))
wg = WorkloadGenerator(cfg["workload"])
reqs = wg.generate()
assert all(r["scheduled_time_s"] == 0.0 for r in reqs), "burst 模式到达时间应全为 0"
print(f"  burst: {len(reqs)} 请求, 全部 t=0.0s ✓")

# constant_rate 模式
cfg2 = json.load(open("configs/workloads/workload_poisson.json"))
cfg2["workload"]["arrival"] = {"pattern": "constant_rate", "rate": 2.0}
wg2 = WorkloadGenerator(cfg2["workload"])
reqs2 = wg2.generate()
expected_gap = 0.5
actual_gap = reqs2[1]["scheduled_time_s"] - reqs2[0]["scheduled_time_s"]
assert abs(actual_gap - expected_gap) < 0.01, f"constant_rate 间隔异常: {actual_gap}"
print(f"  constant_rate: {len(reqs2)} 请求, 间隔={actual_gap:.3f}s ✓")

# poisson 模式
cfg3 = json.load(open("configs/workloads/workload_poisson.json"))
wg3 = WorkloadGenerator(cfg3["workload"])
reqs3 = wg3.generate()
times = [r["scheduled_time_s"] for r in reqs3]
assert times == sorted(times), "poisson 到达时间应单调递增"
assert times[0] > 0, "poisson 首个到达时间应 > 0"
print(f"  poisson: {len(reqs3)} 请求, 时间范围 [{times[0]:.3f}s, {times[-1]:.3f}s] ✓")

# 可复现性
wg3b = WorkloadGenerator(cfg3["workload"])
reqs3b = wg3b.generate()
assert reqs3[0]["request_id"] == reqs3b[0]["request_id"]
assert reqs3[0]["scheduled_time_s"] == reqs3b[0]["scheduled_time_s"]
print("  可复现性: 通过 ✓")
PYEOF

echo ""
echo "=== Task 3.5: 验证 phase-switch ==="
python3 << 'PYEOF'
from workloads.workload_generator import WorkloadGenerator
from collections import Counter
import json

cfg = json.load(open("configs/workloads/workload_phase_switch.json"))
wg = WorkloadGenerator(cfg["workload"])
reqs = wg.generate()
phases = Counter(r["phase_name"] for r in reqs)
print(f"  总计 {len(reqs)} 请求")
print(f"  Phase 分布: {dict(phases)}")
assert len(phases) >= 2, f"Phase switch 未生效: 只有 {len(phases)} 个 phase"
print("  验证通过: ≥2 个 phase ✓")
PYEOF

echo ""
echo "=== Task 3.6: 验证 run_benchmark.py 集成 ==="
python3 << 'PYEOF'
# 仅验证 import 和参数解析，不需要 vLLM 服务
import sys
sys.argv = ["run_benchmark.py", "--help"]
try:
    from benchmarks.run_benchmark import main
    # argparse --help 会抛 SystemExit(0)
    main()
except SystemExit as e:
    pass

# 验证 workload 模式代码路径可加载
from benchmarks.run_benchmark import run_benchmark_workload
print("  run_benchmark_workload 函数可导入 ✓")

# 验证 WorkloadGenerator 集成
from workloads.workload_generator import WorkloadGenerator
import json
cfg = json.load(open("configs/workloads/workload_baseline.json"))
wg = WorkloadGenerator(cfg["workload"])
reqs = wg.generate()
assert len(reqs) == 50
assert "messages" in reqs[0]
assert "scheduled_time_s" in reqs[0]
assert "is_warmup" in reqs[0]
print(f"  WorkloadGenerator 生成 {len(reqs)} 请求, 结构完整 ✓")
print("  集成验证通过 ✓")
PYEOF

echo ""
echo "========================================="
echo "  Week 3 全部 Task 验证完成！"
echo "========================================="
