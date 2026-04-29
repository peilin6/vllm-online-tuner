现在的问题已经从“在线控制”变成了“固定数据集、允许重启、追求性能提升”。在这个设定下，最合适的不是复杂 RL，而是一个可重启的自动调参 agent：它负责生成候选配置、启动 vLLM、跑统一基准、读取指标、更新搜索策略，最后输出最优参数组合。vLLM 官方也明确说明，engine arguments 是 LLM(...) 或 vllm serve 的配置参数；同时官方提供了 vllm bench latency / serve / throughput 这套基准工具，正好适合这种“重启—测试—比较”的调参闭环。

我会建议你把整个系统定义成：

VTA-Agent（vLLM Tuning Agent）
一个面向固定 workload 的、可解释的、分阶段搜索的 vLLM 自动调参 agent。

一、先定一个正确的目标

你这个项目里，agent 不应该直接优化“某一个单独指标”，而应该优化一个约束下的综合目标。

更稳的定义是：

主目标：吞吐提升至少 10%
约束：P95 latency、TTFT、TPOT/ITL 不能明显恶化，且不能频繁 preemption

这是因为 vLLM 的几个核心参数天然存在权衡：
max_num_batched_tokens 较小的时候，ITL 通常更好；较大时，TTFT 和吞吐往往更好，官方还特别建议在较小模型和大 GPU 上，为了吞吐可把它设到 8192 以上。另一方面，KV cache 空间不够时会发生 preemption，而 preemption/recomputation 会伤害端到端延迟；官方建议这时提高 gpu_memory_utilization，或者减小 max_num_seqs / max_num_batched_tokens。

所以你的 agent 最终优化的可以是：

Score=Throughput−λ
1
	​

⋅P95−λ
2
	​

⋅TTFT−λ
3
	​

⋅PreemptionPenalty

或者更简单一点，先做“约束过滤 + 主目标排序”：

先淘汰不满足 P95 / TTFT / preemption 约束的配置
再从剩下配置里选 throughput 最高的

这个做法最适合本科毕设，因为简单、稳、好解释。

二、这个 agent 的最佳形态

你现在最适合的不是“在线 agent”，而是离线工作流 agent。

组件设计

1. Workload Profiler
先扫描固定数据集，提取 workload 指纹：

prompt 长度分布
目标输出长度分布
是否大量共享前缀
请求到达模式，固定速率还是 burst
模型大小、GPU 数量、显存大小

2. Search Planner
根据 workload 类型，自动决定先搜哪些参数，后搜哪些参数。

3. Config Generator
生成候选 vLLM 配置。

4. Executor / Restarter
按配置启动 vllm serve 或 LLM(...)。

5. Benchmark Runner
统一调用 vllm bench serve 或 vllm bench throughput 测试。官方这两个工具就是分别做在线 serving throughput 和离线 throughput 基准的。

6. Metrics Parser
读取吞吐、TTFT、TPOT/ITL、P95/P99、preemption、KV cache 相关指标。vLLM 的 metrics 文档明确说明会记录每轮新生成 token、完成 prefill 的 prompt token、queue interval、TTFT、TPOT 等统计，还支持 KV cache residency metrics。

7. Optimizer
先粗搜，再精调。
不建议一开始就上强化学习，最合适的是：

第一阶段：规则筛选 + 粗网格搜索
第二阶段：Bayesian Optimization 或树模型代理搜索
第三阶段：局部精调

8. Report Generator
输出最优配置、对 baseline 的提升比例、以及“为什么它赢”。

三、最重要的参数分组

你不要一上来搜十几个参数。最稳的是分成三层。

第一层：必须优先调的 5 个核心参数

1. max_num_batched_tokens
每个 iteration 最多处理多少 token。官方说明它是最关键的吞吐/延迟旋钮之一：小值更利于 ITL，大值更利于 TTFT 和吞吐，且吞吐场景下常建议大于 8192。

2. max_num_seqs
每个 iteration 最多处理多少条 sequence。这个值过大容易加剧 KV cache 压力；官方建议 preemption 多时可以减小它。

3. gpu_memory_utilization
vLLM 为执行器/KV cache 预留的显存比例，默认 0.9；提高它能给 KV cache 更多空间。

4. max_model_len
prompt + output 的上下文长度上限；可以设为 auto，也可以按你的 workload 裁剪。对固定数据集来说，这个值如果设得过大，会白白吃掉可用于 KV cache 的空间。官方说明它支持自动选择能装进 GPU 的最大长度。

5. enable_prefix_caching
如果数据集里共享前缀很多，它很值得开；但它只减少 prefill 时间，不减少 decode 时间，因此对长输出、低共享前缀场景收益不大。

第二层：和长 prompt/混合负载强相关的参数

6. enable_chunked_prefill
在 V1 中，chunked prefill 在可能时默认启用；它会把大 prefill 切块，并和 decode 一起调度，有助于平衡 throughput 和 latency。

7. max_num_partial_prefills
chunked prefill 时，最多允许多少条序列被部分 prefill。

8. max_long_partial_prefills
控制长 prompt 并发部分 prefill 的数量；官方明确说把它设得低于 max_num_partial_prefills，可能让短 prompt 在某些情况下插队，从而改善 latency。

9. long_prefill_token_threshold
超过这个长度的 prompt 被视为 long prompt。

这三个参数对“短请求 + 长请求混合”的数据集尤其重要。

第三层：多 GPU 或更高阶的参数

如果你有多 GPU，再考虑：

10. tensor_parallel_size
增大它可以让每张 GPU 为 KV cache 留出更多空间，但会引入同步开销。

11. pipeline_parallel_size
也能减轻单卡权重压力、间接腾出 KV cache 空间，但可能增加 latency。

12. data_parallel_size
官方说明它适合“扩 throughput”而不是“扩单模型尺寸”，如果你 GPU 足够多、又是在线 serving 测试，可以把它列为扩展项。

四、最合理的 agent 工作流

我建议你做成 3 阶段搜索。

阶段 A：Workload 诊断

先别调参，先做 workload fingerprint。

agent 先跑一次 baseline，采集：

平均输入长度、P90 输入长度
平均输出长度、P90 输出长度
TTFT、TPOT、P95/P99
preemption 次数
prefix 重复率

然后把 workload 分成三类：

A 类：prefill-heavy
输入长、输出短、共享前缀高
→ 优先考虑 enable_prefix_caching、max_num_batched_tokens、chunked prefill 相关参数

B 类：decode-heavy
输入短、输出长
→ 少依赖 prefix caching，更关注 max_num_seqs、max_num_batched_tokens 和 preemption

C 类：mixed
长短 prompt 混合
→ 重点调 max_num_partial_prefills、max_long_partial_prefills、long_prefill_token_threshold

阶段 B：粗搜索

先搜小空间，快速淘汰烂配置。

推荐你先用下面这个搜索空间：

max_num_batched_tokens: {2048, 4096, 8192, 16384}
max_num_seqs: {16, 32, 64, 128}
gpu_memory_utilization: {0.85, 0.90, 0.93, 0.95}
enable_prefix_caching: {on, off}
max_model_len: {auto, p99(prompt+output), p95(prompt+output)}

如果是 mixed workload，再加：

max_num_partial_prefills: {1, 2, 4}
max_long_partial_prefills: {1, 2}
long_prefill_token_threshold: {p75_prompt, p90_prompt}

这里不要全排列暴力搜，不然组合太大。
你可以让 agent 用一个简单的分层规则先缩小空间：

如果 prefix 重复率低，直接把 enable_prefix_caching=off 的权重提高
如果 preemption 多，优先减小 max_num_seqs 或 max_num_batched_tokens，或者提高 gpu_memory_utilization，因为官方就这么建议
如果 TTFT 差但 ITL 还行，优先提高 max_num_batched_tokens
如果 ITL 差但 TTFT 还行，优先降低 max_num_batched_tokens
阶段 C：精搜索

从粗搜索里取 Top-5 配置，围绕它们再细调。

这时候再上：

Bayesian Optimization
TPE
或简单的局部邻域搜索

例如当前最好配置是：

max_num_batched_tokens=8192
max_num_seqs=64
gpu_memory_utilization=0.93

那精调阶段就只在邻域里试：

max_num_batched_tokens: 6144 / 8192 / 12288
max_num_seqs: 48 / 64 / 80
gpu_memory_utilization: 0.91 / 0.93 / 0.95

这样效率很高，也很符合“agent 在学习参数结构”的叙事。

五、这个 agent 要怎么“利用好参数特性”

你不能把所有参数当黑盒。要让 agent 带一点参数先验知识。

规则 1

如果 preemption 高频，优先：

提高 gpu_memory_utilization
降低 max_num_seqs
降低 max_num_batched_tokens
这是官方明确建议的方向。
规则 2

如果 TTFT 偏高，优先：

增大 max_num_batched_tokens
打开 prefix caching（前提是共享前缀高）
对 mixed workload，适当提高 partial prefill 并发
因为大 token budget 更利于 prefill 处理，prefix caching 也只主要帮 prefill。
规则 3

如果 ITL/TPOT 偏高，优先：

降低 max_num_batched_tokens
降低过多的长 prefill 并发
控制 long prompt 对 decode 的干扰
因为较小的 max_num_batched_tokens 会减少拖慢 decode 的 prefills。
规则 4

如果 吞吐不高但系统很稳，优先：

适度提高 max_num_seqs
适度提高 max_num_batched_tokens
多 GPU 时考虑 TP/DP
因为这通常说明你还没把机器吃满。
规则 5

如果 数据集共享前缀高，优先试 enable_prefix_caching=on；
如果是长输出、低共享前缀，就别太指望它。

六、实验上最容易踩坑的地方
1. 不要把“热缓存收益”误当成“参数收益”

因为你是同一个数据集反复跑，prefix cache 很容易让后面的实验天然变快。
vLLM 内部就提供了 reset prefix cache 的能力，文档明确提到它可用于 benchmarking 时重置 prefix caching 状态。

所以你要规定两种评测：

cold-run：每次重启或清 cache 后测
warm-run：缓存稳定后测

最后至少报告：

冷启动最优配置
热缓存最优配置
2. 固定请求顺序和 request-rate

因为 vllm bench serve 支持设置 dataset、request-rate、num-prompts；固定它们，实验才公平。

3. 先 warmup，再正式计时

否则第一次图编译、内核 warmup 会污染结果。

4. 不要一次调太多参数

第一版把搜索空间压在 5～8 个核心参数以内，否则很难收敛。

七、我建议你的最终 agent 结构

你可以直接按这个做：

模块 1：Profiler

输入数据集，输出 workload fingerprint。

模块 2：Heuristic Planner

根据 fingerprint 决定：

搜哪些参数
哪些参数固定
每个参数的初始候选范围
模块 3：Config Proposer

生成下一组参数。
先规则生成，再 BO 精调。

模块 4：Runner

执行：

写入 vLLM 启动参数
重启服务
warmup
跑 vllm bench serve / throughput
拉取 metrics
存档结果
模块 5：Judge

判断：

是否满足约束
是否 early stop
是否进入局部精调
模块 6：Explainer

最后生成：

最优配置
相比 baseline 提升多少
为什么提升
不同参数对不同指标的影响
八、最适合你的第一版可实现方案

如果你想要一个真能做出来的版本，我建议先只调这 6 个：

max_num_batched_tokens
max_num_seqs
gpu_memory_utilization
max_model_len
enable_prefix_caching
max_num_partial_prefills（仅 mixed workload 时启用）

然后工作流就设成：

baseline 跑一遍
粗搜索 20～30 组
选 Top-5
邻域精调 10～15 组
输出最优配置
报告 cold/warm 两套结果

这已经足够写成一个完整本科毕设，而且10% 提升是有机会做到的。我不能保证任何模型和任何机器都一定有 10%+，但在“固定 workload、baseline 不是手工最优”的前提下，这个目标是现实的。