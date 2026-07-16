# Qwen3.5/3.6 LoRA 并行与性能审计

本文记录当前 native Qwen3.5/3.6 LoRA 后端与 Megatron-LM 级训练栈的边界。结论按实际代码和 smoke/integration 结果整理，不把配置字段或通用拓扑类型当作已经实现的 kernel。

## 结论

- 模型语义：Qwen3.5 dense、Qwen3.6 dense/MoE 的 native forward/backward 路径已经覆盖 hybrid full attention、GDN、MoE、MTP 和 LoRA 目标模块；已有配置解析、集成测试及 H20 native smoke 证据。
- 已实现并可验证的分布式子集：MoE expert parallel，以及 replicated LoRA 的 data parallel；梯度累积和 dynamic multi-LoRA 已有 logical-step 边界。
- 性能：MoE grouped dispatch 相对逐 expert matmul 的已有 microbenchmark 为约 3.70x（E=32, N=4096, H=2048, I=768，结果误差为 0）；这不是端到端训练吞吐或 Megatron 对比。
- 尚未实现：Qwen native 路径的 tensor parallel、pipeline parallel、context parallel，以及 TP/PP/CP 与 EP/DP 的组合。当前训练上下文仍由单个进程持有完整 dense 权重和完整层栈。
- 因此当前实现不能宣称“Megatron-LM 级别”。它是一个计算集中在 C++ 的 LoRA/EP/DP 子集，离 Megatron 的完整并行和通信重叠仍有实质差距。

## 当前能力矩阵

| 能力 | 当前状态 | 证据/限制 |
| --- | --- | --- |
| Qwen3.5 full attention | 已实现 | native smoke、Qwen3.5 配置集成测试 |
| Qwen3.5 GDN/linear attention | 已实现 | CUDA delta-rule forward/backward 与 native smoke |
| Qwen3.6 MoE | 已实现 | grouped dispatch、EP smoke；完整模型仍需目标 GPU/权重运行 |
| MTP | 已实现 | C++ hidden gradient 检查和集成测试；可通过环境变量关闭 |
| fixed LoRA | 已实现 | attention/GDN/MLP/shared/routed expert 目标模块 |
| dynamic multi-LoRA | 已实现子集 | 请求按 adapter 分组，单个 logical step 统一 backward/Adam；adapter 仍共享 context optimizer step |
| microbatch accumulation | 已实现子集 | non-final microbatch 只 backward，final microbatch 才 optimizer；梯度仍累加在 BF16 leaf 上 |
| replicated data parallel | 已实现 | logical-step 边界同步 replicated LoRA；EP expert 参数不走该 reduction |
| expert parallel | 已实现子集 | 路由输出 all-reduce 和本地 expert 权重；没有 DeepEP 式 fused A2A/dispatch overlap |
| tensor parallel | 未实现于 Qwen native | 不切分 attention/MLP/LM-head 权重，也没有 Qwen TP communicator |
| pipeline parallel | 未实现于 Qwen native | 没有 stage 切分、microbatch scheduler 或 activation send/recv |
| context parallel | 未实现于 Qwen native | 没有 ring attention、跨 rank KV/索引合并 |
| TP/PP/CP 组合 checkpoint | 未实现 | 当前 checkpoint 不是 Megatron rank-sharded topology |

## 与 Megatron-LM 的关键差距

### 并行语义

Megatron 的 TP 会按 head、hidden/intermediate 和 vocab 维度切分权重，并在线性层边界执行必要的 reduce-scatter/all-reduce；PP 会把层分到不同 stage 并使用 1F1B 等调度；CP 会在序列和 attention state 上做跨 rank 通信。当前 Qwen native `TrainingContext` 仍加载完整模型并在一个 C++ forward 中执行全部层，因此仅增加 `tensor_model_parallel_size` 等配置不能得到正确的 TP/PP/CP。

当前 DP/EP 也不是完整 Megatron 语义：DP 只同步 replicated LoRA 梯度，expert 参数留在 EP rank；EP 使用已有 all-reduce，但没有 fused token dispatch/combine、异步 A2A 和通信计算重叠。

### 优化器与恢复

固定 LoRA 的 Adam 状态可导出/导入，且 native context 的 logical step 需要与 checkpoint step 对齐。dynamic adapter 的请求频率不同，但目前仍共享 context-level step；尚无每租户独立 optimizer step、FP32 gradient accumulator 或 accumulation window abort/zero API。这些差距会影响长时间多租户训练的数值一致性和故障恢复。

### 性能工程

当前粗粒度 C++ FFI、grouped MoE 和 activation checkpoint/offload 是有效优化，但尚无 Megatron/Transformer Engine 级别的端到端数据：没有完整模型在同一 GPU、序列长度、microbatch、精度和通信配置下的 tokens/s、显存、扩展效率对照，也没有 FP8/FP4 参数与 fused attention/DeepEP 的 Qwen 路径。

## 验证边界

已运行的验证包括 Rust 编译检查、core 单测、Qwen3.6 配置/集成测试、H20 ABI0 native smoke，以及已有 ABI1 环境中的单卡、EP 和 DP smoke。没有完成 Qwen3.5/3.6 完整大模型的长时间训练、跨节点通信、TP/PP/CP smoke 或与 Megatron-LM 的同条件 benchmark。因此“正确”应理解为已覆盖的模型/LoRA/EP/DP 子集，而不是所有并行配置。

## 继续达到 Megatron 级别所需的最小工作包

1. 建立 5D TP/PP/DP/EP/CP topology，并让 launcher、NCCL process groups 和 checkpoint 使用同一 rank 映射。
2. 为 Qwen full/GDN attention、dense MLP、MoE、LM-head/CE 实现 TP shard 和对应 collective；为 PP 实现 stage forward/backward 与 1F1B scheduler；为 CP 实现 ring attention/state exchange。
3. 将 EP dispatch/combine 替换为 fused/异步路径，并测量通信与计算重叠。
4. 为 LoRA 增加 FP32 accumulation、每 adapter optimizer step、可恢复的 accumulation 状态和 rank-sharded checkpoint。
5. 在固定硬件和 workload 上，与 Megatron-LM 记录 tokens/s、step time、峰值显存、通信占比和 loss 曲线。
