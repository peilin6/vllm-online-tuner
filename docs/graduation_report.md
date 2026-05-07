**vLLM 在线推理服务性能评测与优化系统的设计与实现**

院系：计算机科学与技术  
专业班级：CS2207  
姓名：刘沛林  
学号：待补充  
指导教师：待补充  
日期：2026年5月

## 摘 要

大语言模型在线推理服务正在从离线批处理场景走向交互式应用场景。与传统 Web 服务相比，LLM 推理请求具有输入输出长度差异大、首 token 延迟敏感、显存 KV cache 压力随并发动态变化等特点。vLLM 通过 PagedAttention、continuous batching、prefix caching、chunked prefill 等机制显著提高了推理吞吐，但实际部署中仍需要根据硬件、模型和业务负载反复调整 engine 参数。手工调参依赖经验，难以在多种 workload 下稳定复现，也难以及时解释吞吐、TTFT、TPOT 和尾延迟之间的权衡。

围绕“vLLM 在线推理服务性能评测与优化”这一目标，设计并实现了一个面向 vLLM 的性能评测与闭环调参系统 VTA-Agent。系统首先构建了异步压测、可复现 workload 生成、GPU 与 vLLM 指标采集、实验结果落盘等基础设施；随后实现了由 LLM 驱动的 Diagnoser、Proposer、Reflector 三类角色，并将参数注册表、硬编码安全 Judge、经验记忆、贝叶斯优化、参数敏感度分析、Pareto 过滤、局部网格搜索等工具封装为闭环调参能力。系统不修改 vLLM 源码，不引入代理层，调参动作集中在 vLLM 启动和 engine 参数上，通过“观测、诊断、提议、校验、执行、反思、终止”的循环完成自动化探索。

实验以 Qwen2.5-3B-Instruct-AWQ 模型和 RTX 4060 Laptop GPU 为初始平台，建立了 baseline 0。实测结果显示，在 50 个串行请求场景下，服务成功率为 100.0%，请求吞吐为 0.37 req/s，token 吞吐为 75.37 tokens/s，TTFT 平均值为 62.6 ms，TTFT P95 为 90.8 ms，平均端到端延迟为 2717.2 ms，P95 端到端延迟为 3571.1 ms。基于该基线，系统进一步支持 prefix-heavy、decode-heavy、mixed、phase-switch 等 workload 预设，并通过单元测试和集成测试验证了压测、指标解析、工具分发、LLM fallback 和 agent 主循环的正确性。结果表明，该系统能够为 vLLM 调参实验提供可复现、可解释、可扩展的工程支撑。

**关键词**：vLLM；在线推理；性能评测；自动调参；大语言模型 Agent；KV cache

## Abstract

Online serving of large language models is becoming a core infrastructure for interactive applications. Compared with conventional web services, LLM inference workloads are highly dynamic: prompt length and generation length vary significantly, time to first token is user-visible, and KV cache pressure changes with concurrency and context length. vLLM improves serving efficiency through PagedAttention, continuous batching, prefix caching, and chunked prefill. However, practical deployments still require careful tuning of engine arguments under different models, hardware platforms, and workload patterns. Manual tuning is experience-driven, difficult to reproduce, and often insufficient to explain the trade-off among throughput, TTFT, TPOT, and tail latency.

This project designs and implements VTA-Agent, a performance evaluation and closed-loop tuning system for vLLM online inference services. The system first builds an engineering foundation that includes asynchronous benchmarking, reproducible workload generation, GPU and vLLM metric collection, and structured experiment persistence. On top of this foundation, the system introduces LLM-based Diagnoser, Proposer, and Reflector roles, together with a parameter registry, a rule-based safety Judge, experience memory, Bayesian optimization, parameter sensitivity analysis, Pareto filtering, and local grid search. The system does not modify vLLM source code and does not introduce an online proxy layer. Instead, all tuning actions operate on vLLM engine arguments and take effect through controlled restart. The control loop follows the process of observe, diagnose, propose, validate, act, reflect, and terminate.

The initial baseline is evaluated on Qwen2.5-3B-Instruct-AWQ with an RTX 4060 Laptop GPU. Under a 50-request serial workload, the service reaches a 100.0% success rate, 0.37 req/s request throughput, 75.37 tokens/s token throughput, 62.6 ms mean TTFT, 90.8 ms P95 TTFT, 2717.2 ms average end-to-end latency, and 3571.1 ms P95 end-to-end latency. Based on this baseline, the system supports prefix-heavy, decode-heavy, mixed, and phase-switch workloads. Unit tests and integration tests further validate the benchmark pipeline, metrics parser, tool registry, fallback LLM behavior, and the VTA-Agent control loop. The results show that the proposed system provides a reproducible, explainable, and extensible engineering framework for vLLM tuning experiments.

**Keywords:** vLLM; online inference; performance evaluation; automatic tuning; LLM agent; KV cache

# 绪论

本章介绍课题的研究背景、目的和意义，分析 vLLM 在线推理服务调参与评测的国内外研究现状，并说明毕业设计的主要内容和论文结构。随着大语言模型应用从演示系统走向生产服务，推理系统的吞吐、延迟、显存利用率和稳定性已经成为影响用户体验和部署成本的重要因素。本课题聚焦单机 GPU 环境下 vLLM 在线推理服务的可复现评测与自动调参，目标是将经验驱动的手工调参过程沉淀为一套可执行、可追踪、可解释的闭环系统。

## 课题背景、目的与意义

大语言模型推理服务的性能瓶颈具有明显的阶段性。Prefill 阶段负责处理输入 token，长 prompt、RAG 文档拼接、多轮对话历史会显著影响首 token 延迟；decode 阶段逐 token 自回归生成，更容易受到显存带宽、batch 大小和调度策略影响；并发请求和长上下文还会持续消耗 KV cache，在 cache 逼近上限时可能触发 preemption 或 recompute，从而拉高尾延迟。vLLM 通过 PagedAttention 改善 KV cache 管理，并通过 continuous batching 提高吞吐，但在真实服务中，`max_num_batched_tokens`、`max_num_seqs`、`gpu_memory_utilization`、`enable_prefix_caching`、`enable_chunked_prefill` 等参数仍然需要结合负载特征调整。

传统调参方式通常由开发者观察日志和监控曲线后手动修改参数，再重启服务对比结果。这种方法存在三个问题：第一，实验输入和环境信息容易漂移，导致结果不可复现；第二，调参动作缺少统一安全约束，容易产生 OOM、超时或尾延迟劣化；第三，实验经验没有被结构化保存，后续 workload 仍需重复试错。针对上述问题，本设计将性能评测、指标采集、规则诊断、LLM 推理和算法优化整合为闭环 agent，使系统能够基于历史 trial 自动提出下一步参数修改，并用硬编码 Judge 保证参数合法性和 SLO 约束。

本课题的意义体现在两个方面。工程上，系统为 vLLM 服务提供了从 workload 生成、压测执行、指标采集到报告生成的完整工具链，可以减少人工调参成本。研究上，系统探索了“LLM 作为调参编排器、传统优化算法作为内层工具”的方法，使 LLM 负责语义诊断、假设生成和结果解释，而将数值密集型搜索交给贝叶斯优化、敏感度分析和 Pareto 过滤完成。

## 国内外研究现状

大模型推理系统研究主要围绕显存管理、批处理调度、分布式并行和服务化部署展开。vLLM 提出的 PagedAttention 将操作系统虚拟内存分页思想引入 KV cache 管理，降低了长上下文场景中的显存碎片浪费，并通过 continuous batching 提高请求混合调度效率。HuggingFace Text Generation Inference、TensorRT-LLM、DeepSpeed-Inference 等系统也从不同角度优化推理服务性能，包括 kernel 融合、张量并行、流水线并行、量化和 speculative decoding 等。

在自动调参方面，传统方法包括网格搜索、随机搜索、贝叶斯优化和启发式规则。网格搜索实现简单但成本高；随机搜索更适合高维空间的粗粒度探索；贝叶斯优化可以在有限 trial 数下利用历史结果给出候选；启发式规则则适合具有明确工程经验的场景。近年来，LLM agent 被用于代码生成、运维诊断、数据库调参和配置推荐等任务，其优势在于能够读取自然语言规则、解释指标变化并生成结构化决策。但 LLM 直接进行数值优化容易出现数字幻觉和不稳定输出，因此本设计将 LLM 限定为 orchestrator，不让其绕过参数注册表、Judge 和算法工具。

结合本课题的 `rule.txt` 规则，vLLM 调参可抽象为若干瓶颈类型：KV cache 显存不足或 preemption、长 prompt 导致 TTFT 高、decode 阶段 ITL 高、batch 或 scheduler 吞吐与延迟权衡失衡、重复 prefix 计算浪费、CPU/tokenizer 前端瓶颈、多 GPU 通信瓶颈以及冷启动瓶颈。不同瓶颈对应不同参数方向，例如 preemption 增多时优先增大 `gpu_memory_utilization` 或降低 `max_num_seqs` 和 `max_num_batched_tokens`；TTFT 高时优先增大 `max_num_batched_tokens`、开启 prefix caching 或优化 chunked prefill；ITL 高时则倾向于降低 `max_num_batched_tokens`，减少 prefill 对 decode 的干扰。

## 论文的主要内容与结构

围绕 vLLM 在线推理性能评测与优化，完成了以下工作：

1. 设计并实现异步压测模块，支持 OpenAI 兼容接口、SSE 流式响应解析、TTFT、TPOT、端到端延迟和 token 吞吐统计。
2. 设计并实现 workload 生成模块，支持 burst、constant-rate、poisson、prefix-heavy、long-only、phase-switch 等可复现场景。
3. 设计并实现 GPU 与 vLLM 指标采集模块，记录 GPU 利用率、显存使用、vLLM `/metrics` 中的运行请求数、等待请求数、KV cache 和 preemption 信息。
4. 构建 VTA-Agent 闭环调参框架，实现 Diagnoser、Proposer、Reflector、ExperienceMemory、ToolRegistry、Optimizer 和 Judge 等核心模块。
5. 根据 `rule.txt` 中的 vLLM 调参规则建立瓶颈识别和参数调整策略，将调参动作限定在安全、可解释、可回滚的范围内。
6. 通过 baseline 实验、单元测试和集成测试验证系统的可运行性和可复现性。

论文结构如下：第一章介绍课题背景、研究现状和主要工作；第二章进行需求分析、可行性分析和关键技术论证；第三章说明系统总体设计和模块设计；第四章描述核心模块实现；第五章给出测试环境、功能测试和性能结果分析；第六章总结工作并展望后续方向。

# 方案论证

本章从需求、可行性、开发工具、关键技术和总体方案几个方面论证系统设计。由于 vLLM 调参涉及服务重启、显存安全、性能指标波动和 LLM 输出不确定性，系统方案需要同时满足工程可运行性和实验可复现性。

## 系统需求分析

系统需要满足四类需求。第一是性能评测需求：能够对 vLLM OpenAI 兼容接口发起固定规模或固定速率请求，记录请求成功率、吞吐、TTFT、TPOT、端到端延迟和输出 token 数。第二是 workload 可复现需求：同一 seed、同一 workload 配置下生成相同请求序列，避免 prompt 和到达过程变化干扰参数对比。第三是调参决策需求：根据历史 trial 和最新指标识别瓶颈，提出一次只修改少量参数的配置变更。第四是安全与可追溯需求：所有 LLM 输出必须经过结构化解析和 Judge 校验，所有 trial、拒绝记录、反思笔记和实验结果必须落盘。

系统的核心非功能需求包括可维护性、可扩展性和低侵入性。可维护性要求模块边界清晰，压测、监控、调参、LLM 调用和报告生成分别实现；可扩展性要求后续能够增加新的 workload、参数、工具和 LLM provider；低侵入性要求不修改 vLLM 源码，不接管请求转发路径，而是在外部完成评测和 engine 参数调优。

从用户视角看，系统需要提供三种使用方式。第一种是基础压测方式，用户只需启动 vLLM 服务并执行 benchmark 脚本，即可得到一组可复现的基准数据。第二种是实验套件方式，用户选择 workload 配置后，系统按统一输出格式生成实验目录，便于横向比较。第三种是自动调参方式，用户指定 baseline、workload 和最大步数，VTA-Agent 自动完成多轮 trial 并输出最佳配置与调参轨迹。三种方式分别对应“能测”“能比”“能优化”的递进目标。

从数据视角看，系统需要保存两类数据。一类是原始实验数据，包括请求 trace、监控 timeseries、vLLM metrics 快照和服务日志；另一类是决策数据，包括 LLM prompt、LLM response、tool calls、proposal、Judge verdict、reflection note 和 rejected proposal。前者保证性能结论可反查，后者保证 agent 决策可解释。若只保存最终指标，无法说明某个参数为什么被选择，也无法复盘失败 trial 的原因。因此系统将实验结果和决策过程都作为一等数据对象保存。

## 系统可行性分析

从技术可行性看，vLLM 已提供 OpenAI 兼容 API 和 Prometheus `/metrics` 端点，便于外部压测与指标采集；Python 的 `asyncio` 和 `aiohttp` 能够实现异步并发请求；`pynvml` 可以采集 GPU 利用率、显存和温度；LLM API 可通过 OpenAI 兼容接口封装；贝叶斯优化和敏感度分析可通过 Optuna 或纯函数实现。因此系统无需修改 vLLM 内部调度逻辑即可完成闭环实验。

从工程可行性看，项目已经形成明确目录结构：`benchmarks/` 负责压测，`workloads/` 负责 workload 生成，`monitors/` 负责基础设施监控，`tuner/` 负责 agent 主循环和工具，`llm_advisor/` 负责 LLM 角色和 prompt，`configs/` 负责实验配置，`results/` 负责结果保存，`tests/` 负责自动化测试。模块划分满足毕业设计实现和后续扩展需求。

从风险控制看，LLM 不直接执行重启、写配置和回滚动作，而只输出 `ConfigDelta`。Judge 对参数名、候选值、SLO、重复提议和终止条件进行硬校验；runner 对异常 trial 进行早停；launcher 负责服务重启和失败回滚。该设计降低了 LLM 幻觉或越权输出导致系统不可用的风险。

## 开发工具分析及选择

系统主要使用 Python 3.10 开发。压测模块采用 `asyncio` 和 `aiohttp` 实现异步 HTTP 请求，便于统计流式响应中的首 token 时间和 token 间隔。配置文件采用 JSON，便于版本管理和程序解析。测试框架采用 `pytest`，覆盖配置、压测、workload 生成、监控、参数注册、优化器、LLM client、diagnoser、proposer、reflector、judge 和 agent 主循环。

推理服务选用 vLLM 0.6.x。vLLM 支持 OpenAI 兼容接口、PagedAttention、continuous batching、prefix caching、chunked prefill 和 Prometheus 指标，适合作为在线推理服务调参对象。模型初始阶段选用 Qwen2.5-3B-Instruct-AWQ，原因是 AWQ 量化后模型体积较小，能够在 8GB 显存的 RTX 4060 Laptop GPU 上稳定加载；后续迁移方案支持 A6000 与 Qwen3-8B。

LLM 侧采用 OpenAI 兼容 API 封装，支持 OpenAI、DeepSeek 或本地兼容服务切换。调参算法工具包含 Optuna TPE、参数敏感度分析、Pareto 前沿、局部网格和 workload phase 聚类。系统保留 fallback 规则路径，在没有 LLM API 的情况下仍能跑通端到端闭环。

## 关键技术分析

vLLM 调参的关键在于正确区分瓶颈类型。根据 `rule.txt` 中总结的规则，常见瓶颈包括：

1. KV cache 显存不足或 preemption。典型信号是 preemption count 上升、KV cache 使用率接近上限、P99 延迟恶化。优先动作是增大 `gpu_memory_utilization`，降低 `max_num_seqs` 或 `max_num_batched_tokens`。
2. Prefill 瓶颈。典型信号是 TTFT 高、prompt tokens/s 低、长 prompt 拖慢短请求。优先动作是开启 chunked prefill、增大 `max_num_batched_tokens`、启用 prefix caching。
3. Decode 瓶颈。典型信号是 TTFT 可接受但 ITL/TPOT 高，流式输出卡顿。优先动作是降低 `max_num_batched_tokens` 和 `max_num_seqs`，减少 decode 阶段的 memory bandwidth 竞争。
4. Batch 或 scheduler 瓶颈。典型信号是 GPU 利用率低但队列高。优先动作是增大 batch 参数、开启 async scheduling 或优化前端处理。
5. Prefix 重复计算浪费。典型信号是大量请求共享系统 prompt 但 TTFT 仍高。优先动作是开启 `enable_prefix_caching`，并规范化上游 prompt 模板。

系统将这些规则写入 Diagnoser prompt、Playbook 和 fallback 逻辑中。LLM 不需要记忆所有 vLLM 参数细节，而是通过 `query_param_docs` 等工具读取参数注册表，通过 `bo_suggest`、`param_sensitivity` 等工具获得算法建议，再输出受约束的 `ConfigDelta`。

除瓶颈识别外，另一个关键技术是指标归一化。不同 trial 的请求吞吐、token 吞吐、TTFT、TPOT 和延迟单位不同，且优化方向并不一致：吞吐越高越好，而延迟越低越好；KV cache 使用率过低可能说明资源没有充分利用，过高又会带来 preemption 风险。因此系统没有将所有指标简单线性加权为唯一分数，而是把 token 吞吐作为主要排序指标，同时用 TTFT、TPOT、latency P95、KV cache 和 preemption 作为约束信号。这样既能保留优化方向，又能避免 agent 为了吞吐牺牲交互式服务体验。

第三个关键技术是 LLM 输出约束。自然语言 LLM 具有较强解释能力，但在工程控制路径中必须避免不确定输出。系统从 prompt、schema、parser、fallback 和 Judge 五个层面限制 LLM：prompt 要求 JSON-only；schema 限定字段和枚举值；parser 对非法 JSON 触发 fallback；fallback 规则保证无 LLM 也可运行；Judge 最终检查参数合法性和 SLO。多层约束使 LLM 只参与“建议”，不直接获得执行权限。

## 基本方案制定

最终方案采用“外部闭环调参 agent”架构，不做在线 proxy，不修改 vLLM 源码。系统以固定 workload 为输入，以 vLLM engine 参数为输出，主循环包括以下步骤：

1. Observe：重启 vLLM 并运行一轮 benchmark，采集 `TrialMetrics`。
2. Diagnose：A-LLM 或 fallback 根据最新 trial、baseline 和 memory 判断瓶颈。
3. Propose：P-LLM 读取参数文档和历史记录，必要时调用算法工具，输出一次参数改动。
4. Safety Check：Judge 检查参数合法性、候选值、SLO 和重复提议。
5. Act：执行新配置，运行 trial，失败时记录并回滚。
6. Reflect：R-LLM 判断本次修改是否符合预期，并写入长期经验笔记。
7. Terminate：若收敛、步数用尽或 LLM 建议停止，则输出最佳 trial 和报告。

该方案的特点是职责边界清晰：LLM 负责诊断、提议和解释；算法工具负责数值搜索；Judge 负责安全边界；runner 和 launcher 负责可执行性；memory 负责实验经验沉淀。

## 本章小结

本章分析了 vLLM 在线推理性能评测和自动调参的需求，论证了基于外部闭环 agent 的可行性。方案选择保留 vLLM 原生服务路径，将调参行为限制在 engine 参数层面，并通过 Judge、参数注册表和实验落盘机制保证安全性与可复现性。该设计为后续系统设计和实现奠定了基础。

# 系统设计

本章介绍 VTA-Agent 系统的总体架构、功能模块、数据结构和调参规则设计。系统以实验可复现为中心，将 benchmark、workload、monitor、memory、LLM advisor、optimizer 和 judge 组合为一条闭环流水线。

## 功能需求

系统功能需求可分为五个部分。

第一，基准实验管理。系统需要读取 `configs/experiments/baseline_0.json`，启动固定模型与参数的 vLLM 服务，运行固定请求数、固定采样参数和固定并发的 benchmark，并输出结构化 summary。

第二，workload 生成。系统需要支持短 prompt、长 prompt、共享 prefix、混合长度、poisson 到达、constant-rate 到达、phase-switch 等场景，并保证 seed 可复现。

第三，指标采集与解析。系统需要采集请求级 trace、summary 指标、GPU 指标和 vLLM metrics，将它们聚合为 `TrialMetrics`，供 agent 决策使用。

第四，闭环调参。系统需要存储历史 trial，调用 Diagnoser 识别瓶颈，调用 Proposer 生成参数修改，调用 Judge 校验，调用 runner 执行 trial，再调用 Reflector 写入经验。

第五，报告生成。系统需要汇总 baseline、best trial、调参轨迹、消融实验和 LLM reporter 段落，生成可嵌入毕业设计报告的结果说明。

## 系统总体设计

系统采用分层设计。最底层是 vLLM 服务和 GPU 硬件；其上是压测和监控层，包括 `benchmarks/run_benchmark.py`、`monitors/gpu_monitor.py` 和 `monitors/vllm_metrics_collector.py`；再上层是 trial 执行和数据管理层，包括 `tuner/runner.py`、`tuner/launcher.py`、`tuner/metrics_parser.py` 和 `tuner/memory.py`；最高层是 agent 决策层，包括 `llm_advisor/diagnoser.py`、`llm_advisor/proposer.py`、`llm_advisor/reflector.py`、`tuner/tools.py`、`tuner/optimizer.py`、`tuner/judge.py` 和 `tuner/agent.py`。

系统数据流如下：

1. `WorkloadGenerator` 生成 workload JSON。
2. `Runner` 根据当前配置启动 vLLM 并运行 benchmark。
3. benchmark 输出 `summary.json`、`request_trace.jsonl` 和 metrics timeseries。
4. `metrics_parser` 解析结果为 `TrialMetrics`。
5. `ExperienceMemory` 存储 trial、notes 和 rejected proposals。
6. `Diagnoser` 根据 baseline、latest trial 和 memory 判断瓶颈。
7. `Proposer` 通过 `ToolRegistry` 查询参数文档或调用优化算法，生成 `ConfigDelta`。
8. `Judge` 判断提议是否可执行。
9. `VtaAgent` 应用配置并进入下一轮。

系统总体结构可以抽象为“数据平面”和“控制平面”。数据平面负责实际运行请求、收集延迟和吞吐数据，不改变 vLLM 的请求处理路径。控制平面负责读取数据、做出决策、生成新配置并触发下一轮实验。二者之间通过文件系统中的结构化结果目录和 `TrialMetrics` 数据类交互。该设计减少了模块耦合：benchmark 不需要知道 LLM 决策细节，LLM advisor 也不直接接触 HTTP 请求和 GPU 采样。

与在线 proxy 方案相比，本设计牺牲了毫秒级实时调参能力，但获得了更强的可复现性和安全性。毕业设计场景更关注“如何评测、如何解释、如何形成可靠调参策略”，而不是在生产流量中实时拦截请求。因此系统选择“离线或准在线 trial 闭环”的方式，每一轮参数变更都通过完整 benchmark 评估，适合做对比实验、消融实验和论文结果分析。

## 功能模块设计

### 压测模块设计

压测模块负责向 vLLM OpenAI 兼容接口发送请求。请求支持流式 SSE 模式，记录请求发出时间、首 token 到达时间、每个 token 到达时间和响应完成时间。核心指标包括请求吞吐、token 吞吐、TTFT、TPOT、端到端延迟、成功率和输出 token 数。为了保证 token 计数准确性，压测请求启用 `stream_options.include_usage`，优先从 usage 字段读取 token 数。

### Workload 模块设计

Workload 模块通过配置描述请求数量、到达模式、prompt 长度分布、输出长度、prefix 共享比例和 phase 切换规则。系统提供多种预设：`workload_baseline.json`、`workload_burst.json`、`workload_poisson.json`、`workload_prefix_50.json`、`workload_long_only.json`、`workload_phase_switch.json` 等。固定 seed 使得同一 workload 可以在不同参数配置下复用，从而保证 trial 之间可比较。

Workload 模块的设计重点不是追求语料语义复杂度，而是覆盖推理系统的典型压力形态。短 prompt 场景更容易暴露 decode 和 batch 调度问题；长 prompt 场景更容易暴露 prefill 计算和 TTFT 问题；共享 prefix 场景用于验证 prefix caching 的收益；phase-switch 场景模拟业务流量在一段时间内从短请求切换到长请求或从低速率切换到突发速率的情况。通过这些场景，系统可以观察同一参数在不同负载下的收益差异，避免只针对单一 benchmark 过拟合。

### 指标采集模块设计

GPU 监控模块周期性采集 GPU 利用率、显存占用、温度和功耗。vLLM metrics 采集模块访问 `/metrics` 端点，解析运行请求数、等待请求数、KV cache 使用率、preemption 次数和队列时间等指标。当真实 vLLM 指标缺失时，系统以 `-1` 或默认值表示缺失，Diagnoser 规则明确禁止使用缺失指标作为命中依据。

### Memory 模块设计

`ExperienceMemory` 保存所有 trial 的配置、指标、来源和备注。它支持 `top_k`、`recent_n`、`best`、`summarize` 和 `dump_compact` 等查询方式。`summarize(top_k=3, recent_n=5)` 为 LLM 提供紧凑上下文，避免 prompt 无限制增长；`rejected_proposals` 保存最近被拒绝的参数和值，防止 agent 重复尝试已知失败组合。

### LLM 角色设计

系统将 LLM 拆分为三个角色。Diagnoser 输入 baseline、memory 和 latest trial，输出瓶颈类型、证据、假设和置信度。Proposer 输入诊断结果、Playbook、当前配置、memory 和 notes，输出单次 `ConfigDelta` 或 stop 信号。Reflector 输入修改前后 trial、proposal 和约束检查结果，输出 accept、partial 或 reject 结论，并写入长期经验笔记。

### 参数注册与工具设计

`ParamRegistry` 记录 9 类需要重启生效的参数，包括 `max_num_seqs`、`max_num_batched_tokens`、`gpu_memory_utilization`、`block_size`、`enable_chunked_prefill`、`enable_prefix_caching`、`swap_space`、`cuda_graph_sizes` 和 `tensor_parallel_size`。每个参数包含候选值、范围、影响维度和调参说明。`ToolRegistry` 向 LLM 暴露只读工具和算法工具，但不暴露 `apply_config` 或 restart 能力。

表3-2 给出本系统当前纳入调参闭环的主要参数。

表3-2 VTA-Agent 参数注册表

| 参数 | 类型 | 候选值或范围 | 主要影响 |
| --- | --- | --- | --- |
| `max_num_seqs` | int | 8, 16, 32, 64, 96, 128, 192, 256 | 并发、吞吐、KV cache 压力 |
| `max_num_batched_tokens` | int | 1024, 2048, 4096, 8192, 16384 | TTFT、TPOT、吞吐 |
| `gpu_memory_utilization` | float | 0.80, 0.85, 0.88, 0.90, 0.92, 0.95 | KV cache 容量、OOM 风险 |
| `block_size` | int | 8, 16, 32 | KV 分块粒度、显存利用率 |
| `enable_chunked_prefill` | bool | True, False | prefill/decode 混合调度、公平性 |
| `enable_prefix_caching` | bool | True, False | 共享 prefix 场景 TTFT |
| `swap_space` | float | 0, 2, 4, 8, 16 | preemption 恢复、CPU 交换开销 |
| `cuda_graph_sizes` | str | default, small, wide, off | 启动时间、decode 稳态性能 |
| `tensor_parallel_size` | int | 1, 2, 4, 8 | 多 GPU 显存与通信权衡 |

在当前单卡实验中，`tensor_parallel_size` 固定为 1，但仍保留在注册表中，目的是为后续 A6000 或多 GPU 平台迁移做准备。单卡主要关注 `max_num_seqs`、`max_num_batched_tokens` 和 `gpu_memory_utilization` 三个主旋钮；prefix workload 再关注 `enable_prefix_caching`；长 prompt workload 再关注 `enable_chunked_prefill`。

### Judge 模块设计

Judge 是系统安全边界。它检查参数是否存在于注册表、值是否在候选范围或合法区间、是否重复最近被拒绝组合、trial 是否违反 SLO，以及调参过程是否已经收敛或达到最大步数。该模块不依赖 LLM，避免 LLM 输出不稳定影响系统安全。

## 调参规则设计

根据 `rule.txt`，系统将瓶颈到动作的映射固化为 Playbook。表3-1 给出主要规则。

表3-1 瓶颈类型与首选调参动作

| 瓶颈类型 | 典型信号 | 首选动作 |
| --- | --- | --- |
| KV cache pressure | KV cache 高、preemption 增多、P99 升高 | 增大 `gpu_memory_utilization`，降低 `max_num_seqs` 或 `max_num_batched_tokens` |
| prefill bound | TTFT 高、TPOT 正常 | 增大 `max_num_batched_tokens`，开启 chunked prefill 或 prefix caching |
| decode bound | TPOT/ITL 高、TTFT 正常 | 降低 `max_num_batched_tokens`，降低 `max_num_seqs` |
| queue backlog | 队列时间占比较高 | 增大 batch 参数或启用更合适的调度策略 |
| underutilized | GPU/KV 不紧张但吞吐低 | 增大 `max_num_batched_tokens`、`max_num_seqs` |
| slo margin low | TTFT 或 latency 接近 SLO 上限 | 降低 batch 参数，使用 Pareto 前沿权衡吞吐和延迟 |

每轮调参原则是一次只改变一个主参数，减少因多参数同时变化导致的归因困难。LLM 可以调用算法工具给出候选，但最终参数仍必须来自注册表候选值，并由 Judge 校验。

为了进一步限制搜索空间，系统将每类瓶颈绑定到白名单参数。例如 `decode_bound` 只允许调整 `max_num_batched_tokens`、`max_num_seqs` 和 `enable_prefix_caching`；`kv_cache_pressure` 只允许调整 `gpu_memory_utilization`、`max_num_seqs`、`max_num_batched_tokens` 和 `block_size`；`converged` 不允许继续提出配置修改，只能输出停止信号。白名单机制可以避免 LLM 在不相关瓶颈下调整无关参数，也能提高实验解释性。

系统还区分探索阶段和微调阶段。早期 memory 较少时，LLM 主要依据 Playbook 和参数文档做启发式探索；当成功 trial 数达到一定规模后，可以调用 `bo_suggest` 获取 TPE 候选；当连续几步提升有限时，可以调用 `param_sensitivity` 判断最敏感参数；当 SLO 余量较低时，可以调用 `pareto_front` 分析吞吐和延迟边界；接近收敛时，通过 `local_grid` 在 best 附近做小范围搜索。这样既避免过早依赖不充分历史数据，也避免后期继续盲目探索。

## 本章小结

本章给出了系统的总体架构和功能模块设计。VTA-Agent 通过模块化方式整合 workload、benchmark、monitor、memory、LLM advisor、optimizer 和 judge，形成可复现的闭环调参流程。调参规则来自 vLLM 性能瓶颈分析，并被转化为 Playbook 和参数注册表，使 LLM 决策具有明确边界。

# 系统实现

本章介绍系统核心模块的实现方法，包括压测、workload、指标采集、LLM 调用、调参工具、Judge 和 VTA-Agent 主循环。实现过程强调可测试性和可追踪性，所有关键模块均提供独立单元测试。

## 压测与结果聚合实现

`benchmarks/run_benchmark.py` 是压测执行入口。脚本读取实验配置和 workload 配置，构造 OpenAI chat completions 请求，并使用异步协程并发发送。对于流式响应，脚本在收到第一个非空 token 时记录 TTFT，在每个 token 到达时记录时间戳，最终计算 TPOT 和端到端延迟。压测完成后，脚本输出人类可读摘要和机器可读 JSON。

结果聚合字段包括 `throughput_rps`、`token_throughput_tps`、`ttft_ms`、`tpot_ms`、`latency_ms`、`success_rate`、`total_requests`、`successful`、`failed` 和 `vllm_aggregates`。其中 `vllm_aggregates` 保存 preemption、KV cache 和 queue time 等指标，为 Diagnoser 判断瓶颈提供依据。

在实现中，TTFT 的计算方式为“请求发送完成后到首次接收有效 token 的时间差”，端到端延迟为“请求发送完成后到响应结束的时间差”。TPOT 用输出 token 时间戳序列计算，反映流式输出阶段 token 间隔。对于失败请求，系统记录异常类型和超时信息，但不将其纳入成功请求延迟分位数计算。这样既能避免失败请求污染延迟统计，又能通过成功率和错误数反映服务稳定性。

为减少实验过程中的环境干扰，所有使用 Python HTTP 客户端的脚本在 WSL2 中运行时需要设置 `no_proxy="*"`。这是因为代理环境变量可能导致 localhost 请求绕行或连接失败。该约束写入执行 prompt 和脚本说明中，保证后续重复实验时不会因网络代理导致不可复现问题。

## Workload 生成实现

`workloads/workload_generator.py` 根据 JSON 配置生成请求序列。请求内容从 `prompts_pool.json` 和 `prefix_pool.json` 中采样，支持共享 prefix 注入。到达过程支持 burst、constant-rate 和 poisson；phase-switch workload 通过时间窗口切换请求长度和速率分布。生成结果包含 prompt、max_tokens、temperature、到达时间和请求元数据。

为保证可复现性，生成器使用显式 seed。测试用例验证了 `WorkloadGenerator(seed=42).generate()` 多次执行结果一致，并验证 prompt pool 和 prefix pool 数量满足实验要求。

## 指标采集与解析实现

`monitors/gpu_monitor.py` 使用 NVML 周期采集 GPU 指标，记录采样时间、GPU 利用率、显存使用、温度和功耗。`monitors/vllm_metrics_collector.py` 访问 vLLM `/metrics` 端点并解析 Prometheus 文本格式，提取运行请求数、等待请求数、KV cache 使用和 preemption 指标。

`tuner/metrics_parser.py` 将 `summary.json` 转换为 `TrialMetrics` 数据类。`TrialMetrics` 包含请求吞吐、token 吞吐、TTFT P95、TPOT P95、latency P95、preemption rate、KV cache P95、queue time P95、success、early_killed、wall_time、success_rate 和 total_requests 等字段。该结构是 agent 决策的统一输入。

由于 vLLM `/metrics` 不同版本暴露的指标名称存在差异，collector 采用“优先解析已知字段，缺失则优雅降级”的策略。例如 queue time 在当前实现中通过 `queue_time_delta_s` 和成功请求数估算 P95；当未来接入 histogram bucket 后，可以替换为真实分位数计算。此处采用保守设计：缺失字段用 `-1` 表示，Diagnoser prompt 明确规定缺失值不能作为命中规则的依据。

## 参数注册表实现

`tuner/param_registry.py` 定义 `ParamSpec` 数据类和 9 个 RESTART 参数。每个参数包含名称、配置路径、类型、默认值、候选值、范围、影响维度和说明。`ParamSpec.in_range()` 用于判断参数值是否合法，`clamp()` 用于将越界数值裁剪到合法范围。该模块是 Judge、ToolRegistry、Optimizer 和 LLM prompt 的单一事实来源。

参数注册表的设计避免了 LLM 任意生成不存在的 vLLM 参数，也避免了字符串形式的配置散落在多个模块中。新增参数时只需在注册表中补充候选值和影响说明，再在 Playbook 中声明对应瓶颈即可。

## LLM Advisor 实现

`llm_advisor/prompts.py` 保存 A-LLM、P-LLM、R-LLM 和 Reporter 的 system/user 模板。Prompt 明确要求 JSON-only 输出，并要求 Diagnoser 按规则顺序判断瓶颈。`diagnoser.py` 负责解析诊断结果；当 LLM 不可用或输出非法时，使用 fallback 规则生成诊断。`proposer.py` 负责 function-calling 风格的工具调用和 `ConfigDelta` 解析；`reflector.py` 负责根据 trial 前后对比生成经验笔记。

LLM client 封装在 `llm_advisor/llm_client.py` 中，支持 OpenAI 兼容接口、重试、缓存和限流。系统在测试中使用 mock client 验证 tool-calling 流程，避免 CI 依赖真实 API。

Diagnoser 的 fallback 逻辑与 prompt 中的规则保持一致：优先判断 preemption、KV cache、queue backlog 和 SLO 余量，再判断 prefill/decode 瓶颈，最后回退到 underutilized 或 converged。Proposer 的 fallback 逻辑根据 Playbook 选择参数方向，并尽量避开最近被拒绝的提议。Reflector 的 fallback 逻辑根据新 trial 相比旧 trial 的吞吐和约束变化判断 accept 或 reject。通过这种方式，LLM 只是增强系统效果的可选能力，而不是系统正确性的唯一来源。

Reporter 角色用于实验结束后生成自然语言分析段落。它读取 baseline、best trial 和完整 memory，概括最佳配置、关键转折和未覆盖的搜索空间。为了避免报告中的数字幻觉，设计中要求 Reporter 段落中的数字必须能在 memory dump 中反查；后续可通过脚本扫描 Reporter 输出中的数字并与 memory 比对。

## 优化工具实现

`tuner/optimizer.py` 提供五类算法工具：

1. `bo_suggest`：基于历史成功 trial 调用 Optuna TPE 生成候选配置；数据不足时返回冷启动候选。
2. `param_sensitivity`：计算参数与目标指标之间的 Spearman 相关和指标极差，帮助判断最值得继续探索的参数。
3. `pareto_front`：在吞吐最大化和延迟最小化之间计算非支配 trial。
4. `local_grid`：围绕当前 best 配置在候选值邻域生成局部网格。
5. `cluster_workload_phases`：对 workload 特征或 trial 指标进行简单聚类，辅助 phase-switch 分析。

这些工具只读取 memory，不直接修改配置，不执行重启，输出结果仅作为 LLM 决策参考。

## VTA-Agent 主循环实现

`tuner/agent.py` 实现 `VtaAgent`。其 `run()` 方法接收 baseline metrics、baseline config 和最大步数。首次运行时，agent 将 baseline 写入 memory；随后进入循环：判断终止条件，调用 `diagnose()`，调用 `propose()`，用 Judge 检查 `ConfigDelta`，执行 runner，写入新 trial，检查 SLO，调用 `reflect()`，根据反思结果 accept 或 rollback。

主循环具有两种运行模式。LLM 模式下，Diagnoser、Proposer 和 Reflector 调用真实或 mock LLM；fallback 模式下，`client=None` 且 `use_llm=False`，系统仍通过规则跑完整流程。该设计既方便离线测试，也保证 API 不可用时系统不会完全失效。

`VtaAgent` 在 accept/rollback 上采用保守策略：只有当 Reflector 给出 `accept` 且 Judge 的 trial constraint 通过时，当前配置才更新为新配置；否则记录 rejected proposal，下一步仍从上一个 accepted 配置继续探索。这样可以避免一次失败 trial 将系统带入更差配置区域。对于 runner 异常，例如 vLLM 启动失败、健康检查超时或 OOM，agent 不会中断整个 run，而是将该配置写入拒绝记录并继续下一步。

主循环的终止条件包括最大步数、连续多步提升小于阈值、LLM 主动 stop 和 Judge 判断收敛。毕业设计当前默认最大步数为 20，主要考虑单次 trial 需要重启 vLLM 并运行 benchmark，过多步数会导致总实验时长过长。通过 early stop 和 convergence check，系统可以在明显无收益时提前结束，节省 GPU 时间。

## 异常处理与可追溯实现

vLLM 调参实验中常见异常包括模型加载失败、端口占用、显存不足、请求超时、metrics 端点不可访问和 LLM API 调用失败。系统分别在 launcher、runner、collector 和 LLM client 中处理这些异常。模型加载失败或端口不可用时，runner 返回失败 trial 或抛出异常，由 agent 记录 rejected proposal；metrics 缺失时，TrialMetrics 中相应字段设为缺失值；LLM API 失败时，Diagnoser、Proposer 和 Reflector 进入 fallback。

可追溯性通过结果目录实现。每次实验应在 `results/<exp_id>/` 下保存配置快照、请求 trace、summary、metrics timeseries 和日志。调参 run 则在 `results/tuning/<run_id>/` 下保存 `memory.jsonl`、`notes.jsonl`、`llm_calls.jsonl`、每个 trial 的子目录、`report.json` 和 `report.md`。这种目录结构使论文中的每个实验数字都可以追溯到原始结果，避免只保留截图或口头结论。

## 本章小结

本章说明了系统关键模块的实现。系统通过统一数据结构串联压测结果、指标解析、LLM 推理和调参执行；通过参数注册表、工具注册表和 Judge 限制 LLM 行为；通过 fallback 和自动化测试保证系统在无外部 API 时仍可验证核心链路。

# 性能测试与分析

本章介绍测试环境、功能测试、baseline 性能测试和 agent 相关验证。由于部分 A6000 迁移实验和 Week 8 完整消融仍依赖目标平台，本章以当前已落盘结果和自动化测试为主，同时说明后续实验的扩展方式。

## 测试环境

初始 baseline 平台如下。

表5-1 初始测试环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB VRAM |
| 驱动与 CUDA | Driver 566.26，CUDA 12.4 |
| 操作系统 | Windows + WSL2 Ubuntu-22.04 |
| Python | 3.10 |
| vLLM | 0.6.6.post1 |
| PyTorch | 2.5.1+cu124 |
| Transformers | 4.49.0 |
| 模型 | Qwen/Qwen2.5-3B-Instruct-AWQ |
| 服务端口 | 8000 |

baseline 配置固定模型、量化方式、最大上下文长度、采样参数、并发数和请求数。主要参数如下。

表5-2 baseline 0 配置

| 参数 | 值 |
| --- | --- |
| model | Qwen/Qwen2.5-3B-Instruct-AWQ |
| dtype | auto |
| quantization | awq |
| max_model_len | 2048 |
| tensor_parallel_size | 1 |
| gpu_memory_utilization | 0.90 |
| max_num_seqs | 32 |
| temperature | 0.7 |
| top_p | 0.9 |
| max_tokens | 256 |
| num_requests | 50 |
| concurrency | 1 |

## 功能测试

系统功能测试覆盖以下模块：

1. WorkloadGenerator：验证同一 seed 生成结果一致，验证 prefix 和 phase-switch 配置生效。
2. Benchmark：验证旧模式和 workload 模式均能运行，验证 summary 中包含 TTFT、TPOT、吞吐和延迟统计。
3. Monitor：验证 GPU monitor 能采样，vLLM metrics collector 能解析关键 Prometheus 指标或优雅降级。
4. Metrics Parser 与 Runner：验证 `summary.json` 能被解析为 `TrialMetrics`，异常配置可触发 early kill。
5. Memory：验证 JSONL 持久化、top-k、recent、best、summary 和 rejected proposals。
6. ParamRegistry 与 Judge：验证参数数量、候选值合法性、SLO 守门、重复拒绝和终止条件。
7. Optimizer：验证 BO、敏感度分析、Pareto 前沿和局部网格工具。
8. LLM Advisor：验证 Diagnoser、Proposer、Reflector 的 JSON 解析和 fallback 行为。
9. Agent：验证 mock LLM 和 fallback 两种模式下的完整闭环。

集成测试 `tests/test_integration_week6_7.py` 使用 mock LLM 和 mock runner 验证 VTA-Agent 主循环，确保 memory 中包含 baseline 和 agent trial，LLM tool-calling 被调用，best trial 能按吞吐选出。

表5-3 给出主要测试模块及其验证目标。

表5-3 测试模块与验证目标

| 测试文件 | 验证目标 |
| --- | --- |
| `test_workload_generator.py` | workload 可复现、prefix 与 phase-switch 生效 |
| `test_run_benchmark.py` | 压测参数解析、summary 输出和旧模式兼容 |
| `test_gpu_monitor.py` | GPU 采样器启动、采样和停止 |
| `test_vllm_metrics_collector.py` | Prometheus 文本解析和缺失指标降级 |
| `test_metrics_parser_and_runner.py` | summary 到 TrialMetrics 的解析，runner 早停 |
| `test_memory.py` | memory JSONL 持久化、best/top/recent 查询 |
| `test_param_registry.py` | 9 个 RESTART 参数元数据和值域校验 |
| `test_optimizer.py` | BO、敏感度、Pareto、local grid 工具 |
| `test_diagnoser.py` | 瓶颈规则和 fallback 诊断 |
| `test_proposer.py` | 参数提议、tool-calling 和 JSON 解析 |
| `test_reflector.py` | accept/reject 反思和 note 生成 |
| `test_judge.py` | 参数合法性、SLO、重复提议和终止条件 |
| `test_agent.py` | VTA-Agent fallback 主循环 |
| `test_integration_week6_7.py` | mock LLM 下完整闭环 |

## baseline 性能测试

根据 `prompts/week3-8_execution.prompt.md` 中记录的 baseline 0 实测数据，50 个串行请求的结果如下。

表5-4 baseline 0 性能结果

| 指标 | 值 |
| --- | --- |
| 成功率 | 100.0% (50/50) |
| 请求吞吐 | 0.37 req/s |
| Token 吞吐 | 75.37 tokens/s |
| TTFT Mean | 62.6 ms |
| TTFT P95 | 90.8 ms |
| Avg Latency | 2717.2 ms |
| P95 Latency | 3571.1 ms |
| P99 Latency | 3624.7 ms |
| 平均输出 Tokens/req | 204.8 |
| 总耗时 | 约 136 s |

从结果看，当前 baseline 在串行请求下成功率稳定，TTFT 较低，说明单请求 prefill 阶段没有明显排队压力；平均端到端延迟约 2.7 s，主要由 204.8 个平均输出 token 的 decode 时间决定。由于并发数为 1，该结果更适合作为服务正确性和基础性能基线，而不是吞吐上限。后续调参实验需要使用更高并发、prefix-heavy、long-only 和 phase-switch workload 才能充分暴露 batch、KV cache 和调度瓶颈。

从吞吐角度看，0.37 req/s 与 75.37 tokens/s 的组合说明在串行请求下请求吞吐受到单请求生成长度限制，而 token 吞吐更能反映 decode 阶段的实际产出能力。若后续提高并发，理论上 continuous batching 会提高 token 吞吐，但同时可能增加 TTFT、TPOT 或 KV cache 压力。因此 baseline 0 的作用不是证明配置最优，而是提供一个“稳定、可重复、可比较”的起点。

从延迟角度看，TTFT P95 仅 90.8 ms，远低于端到端延迟 P95 3571.1 ms，说明当前 workload 的用户等待主要发生在输出生成阶段，而不是首 token 返回阶段。这与 `max_tokens=256` 且平均输出 204.8 tokens/request 的设置一致。若后续 workload 改为长 prompt 或 RAG 场景，TTFT 将成为更关键指标；若 workload 改为短输入长输出，TPOT 和 token 吞吐将成为更关键指标。

## 调参系统测试分析

在 mock runner 场景中，agent 通过逐步增加 `max_num_seqs` 探索吞吐提升。当 `max_num_seqs=32` 时 baseline token 吞吐为 1000 tokens/s；mock profile 中 `max_num_seqs=64` 对应 1100 tokens/s，`max_num_seqs=96` 对应 1180 tokens/s，`max_num_seqs=128` 对应 1150 tokens/s。集成测试验证 best trial 能选择吞吐最高的 96 或至少优于 baseline 的 64，说明 memory 排序、proposal 执行和 best trial 选择逻辑有效。

真实调参时，系统不会简单追求吞吐最大化，而是将 TTFT、TPOT、latency P95、preemption 和 KV cache 作为约束。若吞吐提高但 latency P95 超过 SLO，Judge 会将 trial 标记为不通过，Reflector 也可能拒绝该方向。该机制使 agent 能够处理“吞吐提升但交互体验恶化”的常见冲突。

根据 `rule.txt`，不同 workload 的预期调参方向也不同。对于 prefix-heavy workload，`enable_prefix_caching=True` 应优先被尝试，因为共享系统 prompt 或模板能够复用 KV cache blocks，从而降低重复 prefill 成本。对于 decode-heavy workload，若 TPOT 升高，应尝试降低 `max_num_batched_tokens`，减少 prefill 对 decode 的干扰。对于 mixed workload，agent 需要在吞吐和尾延迟之间折中，可能需要 Pareto 前沿辅助判断。对于 phase-switch workload，单一 best 配置未必适合所有阶段，因此后续可以扩展 phase-aware 配置选择。

系统中算法工具的作用不是替代 LLM，而是减少 LLM 在数值搜索上的负担。例如在历史 trial 少于 4 条时，BO 建议的统计意义有限，Proposer 更适合依据 Playbook 做启发式探索；当 memory 积累到一定数量后，BO 可以给出下一组候选，LLM 再结合参数说明、SLO 余量和经验笔记判断是否采纳。该设计符合“LLM 是 orchestrator，算法是 inner-loop tool”的定位。

## 消融与后续测试设计

系统预留了 Week 8 消融实验接口，计划对 decode-heavy workload 运行五组对比：

1. random-proposer：将 P-LLM 替换为随机参数选择，验证 LLM proposer 的价值。
2. no-memory：只保留 baseline 和最近一步，验证长期经验记忆的作用。
3. no-reflect：跳过 R-LLM 反思，验证 notes 对后续决策的影响。
4. fixed-config：不运行 agent，作为 baseline 下限。
5. no-early-stop：关闭早停和收敛检测，验证安全机制对 wall time 的影响。

预期完整 agent 相比 A/B/C 消融具有更高 best score，no-early-stop 的 wall time 明显增加。消融结果将进一步支撑 LLM 编排、经验记忆和安全终止机制的有效性。

后续正式实验计划包含三类 workload。第一类是 `workload_prefix_50.json`，用于验证 prefix caching 和 prefill 优化；第二类是 `workload_long_only.json`，用于验证长输入或长输出下的 TTFT/TPOT 权衡；第三类是 `workload_phase_switch.json`，用于验证动态负载切换下 memory 和 reflection 的作用。每类 workload 都应报告 baseline、agent best、提升比例、SLO 是否违反、step 数和 wall time。图表方面，计划生成吞吐提升柱状图、agent trajectory 曲线和消融对比柱状图。

为了保证论文结论可靠，后续所有实验数字都应从 `results/tuning/` 中反查。具体做法是：每个 run 生成 `report.json`，汇总脚本读取 `report.json` 和 `memory.jsonl` 生成 `summary_week8.md`；论文只引用 summary 中的数字；Reporter 段落中的数字再通过脚本与 memory 做比对。这样可以降低手工复制表格时产生错误的概率。

## 本章小结

本章给出了测试环境、功能测试和 baseline 性能结果。实验表明，当前系统能够稳定完成 vLLM 基线压测，并已通过自动化测试验证闭环调参核心路径。baseline 结果为后续多 workload 调参和 A6000 迁移实验提供了对照基准。

# 总结与展望

围绕 vLLM 在线推理服务性能评测与优化，完成了以下工作：

1. 构建了可复现的 vLLM 性能评测工程，包含实验配置、服务启动、健康检查、异步压测、请求 trace、summary 输出和结果目录管理。
2. 实现了 workload 生成模块，支持 burst、constant-rate、poisson、prefix-heavy、long-only、mixed 和 phase-switch 等负载场景。
3. 实现了 GPU 与 vLLM 指标采集模块，将基础设施指标和服务内部指标聚合为 agent 可读的 `TrialMetrics`。
4. 设计并实现了 VTA-Agent 闭环调参框架，将 LLM Diagnoser、Proposer、Reflector 与参数注册表、工具注册表、经验记忆和安全 Judge 结合起来。
5. 根据 `rule.txt` 中总结的 vLLM 调参规则，建立了瓶颈识别与参数调整 Playbook，使调参方向能够解释为 TTFT、TPOT、KV cache、preemption、queue 和 GPU 利用率等指标变化。
6. 完成 baseline 0 实验，获得 100.0% 成功率、0.37 req/s 请求吞吐和 75.37 tokens/s token 吞吐等基线指标。
7. 通过单元测试和集成测试验证了压测、配置、workload、monitor、memory、optimizer、LLM advisor、Judge 和 VTA-Agent 主循环。

后续工作可以从以下方向展开：

1. 在 RTX A6000 与 Qwen3-8B 平台上运行完整三类 workload，对比 prefix-heavy、decode-heavy 和 mixed 场景下 agent 的收益。
2. 补全五组消融实验，量化 LLM proposer、memory、reflector 和 early-stop 对 best score 与 wall time 的影响。
3. 将真实 vLLM histogram 指标接入 queue time P95 和 KV cache 使用率计算，减少当前粗估带来的误差。
4. 扩展参数注册表，加入 async scheduling、renderer workers、api server count、prefix partial prefill 等前端和调度参数。
5. 增加图表生成和 Reporter 校验，使最终报告中的每个数字都能从 `results/tuning/` 反查。
6. 在多 GPU 或 MoE 模型场景中扩展 TP、PP、DP 和 EP 调参策略，研究通信瓶颈下的 agent 决策。

总体来看，VTA-Agent 将 vLLM 调参从一次性人工试验转化为可记录、可复现、可解释的闭环过程。该系统既可以作为毕业设计的工程实现，也可以作为后续大模型推理服务自动优化研究的基础平台。

# 参考文献

[1] Kwon W, Li Z, Zhuang S, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention[C]. Proceedings of the ACM SIGOPS Symposium on Operating Systems Principles, 2023.

[2] vLLM Project. Optimization and Tuning - vLLM[EB/OL]. https://docs.vllm.ai/en/stable/configuration/optimization/

[3] vLLM Project. vLLM serve - vLLM[EB/OL]. https://docs.vllm.ai/en/latest/cli/serve/

[4] vLLM Project. Automatic Prefix Caching - vLLM[EB/OL]. https://docs.vllm.ai/en/stable/design/prefix_caching/

[5] vLLM Project. Engine Arguments - vLLM[EB/OL]. https://docs.vllm.ai/en/stable/configuration/engine_args/

[6] OpenAI. OpenAI API Reference[EB/OL]. https://platform.openai.com/docs/api-reference

[7] Akiba T, Sano S, Yanase T, et al. Optuna: A Next-generation Hyperparameter Optimization Framework[C]. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019.

[8] Wolf T, Debut L, Sanh V, et al. Transformers: State-of-the-Art Natural Language Processing[C]. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2020.

[9] Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library[C]. Advances in Neural Information Processing Systems, 2019.

[10] NVIDIA. NVIDIA Management Library NVML Documentation[EB/OL]. https://developer.nvidia.com/nvidia-management-library-nvml

# 致谢

毕业设计完成过程中，指导教师在选题方向、系统架构、实验设计和论文撰写方面给予了耐心指导，使课题能够从最初的 vLLM 基线压测逐步扩展为完整的闭环调参系统。感谢计算机学院各位老师在专业课程和工程实践训练中的帮助，为本设计涉及的操作系统、并发编程、机器学习系统和软件工程能力打下了基础。

感谢同学和朋友在环境搭建、实验复现、问题排查和论文修改过程中提供的建议。vLLM 服务部署、GPU 显存约束、WSL 环境和 LLM API 调用都曾带来不少工程细节问题，这些讨论帮助系统逐步稳定下来。

最后，感谢家人在学习和毕业设计期间给予的支持与鼓励。
