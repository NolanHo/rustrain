# Qwen3.5/3.6 LoRA 并行与性能审计

本文记录当前 native Qwen3.5/3.6 LoRA 后端与 Megatron-LM 级训练栈的边界。结论按实际代码和 smoke/integration 结果整理，不把配置字段或通用拓扑类型当作已经实现的 kernel。

## 结论

- 模型语义：Qwen3.5 dense、Qwen3.6 dense/MoE 的 native forward/backward 路径已经覆盖 hybrid full attention、GDN、MoE、MTP 和 LoRA 目标模块；已有配置解析、合成 oracle、集成测试及 H20 native smoke 证据，但尚未完成真实 35B/3.6 权重的长时间训练验证。
- 已实现并可验证的分布式子集：MoE expert parallel，以及 replicated LoRA 的 data parallel；梯度累积和 dynamic multi-LoRA 已有 logical-step 边界。DP 动态租户按 adapter token count 加权，sharded A2A native 路径会保留 source flattened row 来恢复租户，并按全局租户 token count 归一化。
- 性能：MoE grouped dispatch 相对逐 expert matmul 的已有 microbenchmark 为约 3.70x（E=32, N=4096, H=2048, I=768，结果误差为 0）；这不是端到端训练吞吐或 Megatron 对比。
- 已实现 LoRA latent-rank TP-only、frozen full-attention TP（Q/K/V ColumnParallel、O RowParallel），以及 frozen dense SwiGLU MLP 的 gate/up row shard、down column shard 和输出 all-reduce。Q/K/V/O fixed 与 selected dynamic LoRA 使用 projection-aware shard 和梯度 reduction；GDN/MoE/embedding/LM-head 仍复制，MLP LoRA 在 base TP 下暂拒绝；PP/CP 和 TP 与 EP/DP 的组合仍未实现。
- 因此当前实现不能宣称“Megatron-LM 级别”。它是一个计算集中在 C++ 的 LoRA/EP/DP 子集，离 Megatron 的完整并行和通信重叠仍有实质差距。

## 当前能力矩阵

| 能力 | 当前状态 | 证据/限制 |
| --- | --- | --- |
| Qwen3.5 full attention | 已实现 | native smoke、Qwen3.5 配置集成测试 |
| Qwen3.5 GDN/linear attention | 已实现 | CUDA delta-rule forward/backward 与 native smoke |
| Qwen3.6 MoE | 已实现 | grouped dispatch、EP smoke；完整模型仍需目标 GPU/权重运行 |
| MTP | 已实现 | C++ hidden gradient 检查和集成测试；可通过环境变量关闭 |
| fixed LoRA | 已实现 | attention/GDN/MLP/shared/routed expert 目标模块 |
| dynamic multi-LoRA | 已实现子集 | 请求按 adapter 分组，单个 logical step 统一 backward/Adam；每个 adapter 独立 optimizer clock 和 m/v，DP 与 native sharded A2A 按全局 token count 加权；零全局 token 租户跳过更新；dynamic+MTP 暂拒绝 |
| microbatch accumulation | 已实现子集 | non-final microbatch 只 backward，final microbatch 才 optimizer；FP32 accumulator 存储/聚合，autograd leaf backward 仍为 BF16 |
| replicated data parallel | 已实现 | logical-step 边界同步 replicated LoRA；EP expert 参数不走该 reduction |
| expert parallel | 已实现子集 | 默认 routed-output all-reduce；gated variable-split A2A 已验证 fixed-LoRA 和 native dynamic-LoRA data sharding；GPU-only split planning、异步 overlap 和 DeepEP backend 未实现 |
| tensor parallel | full attention + dense MLP 子集 | full attention 的 Q/K/V 输出头分片、O 输入列分片和 projection-aware fixed/dynamic LoRA 已通过 TP2 full-reference smoke；dense gate/up/down base 权重按 intermediate 维切分；GDN/MoE/embedding/LM-head 仍不切分，MLP LoRA/MTP 暂拒绝 |
| pipeline parallel | 未实现于 Qwen native | 没有 stage 切分、microbatch scheduler 或 activation send/recv |
| context parallel | 未实现于 Qwen native | 没有 ring attention、跨 rank KV/索引合并 |
| distributed checkpoint | 已实现子集 | v4 记录 projection layout、replicated tensor geometry 和 fixed slot identity，并验证 same-topology rank shard；v3 仅兼容 latent-rank checkpoint，旧 attention LoRA 因形状不可迁移而明确拒绝；跨 topology reshard 和 PP/CP 未实现 |

## 与 Megatron-LM 的关键差距

### 并行语义

Megatron 的通用 MLP 使用 ColumnParallel fc1、local gated activation 和 RowParallel fc2，并进一步提供 fused fc1/activation、sequence parallel、通信重叠和 sharded-state 支持。当前 Qwen native 已补上算法等价的 separate gate/up row shard 与 down column shard，但仍是两次独立 GEMM。

本地 Megatron-LM 的 `experimental/lite/megatron/lite/model/qwen3_5` 已包含直接的 Qwen3.5 实现和 LoRA adapter，而不是只能依赖外部 bridge：`primitive/modules/gqa.py` 使用 fused QKV ColumnParallel 和 O RowParallel；`gated_delta_net.py` 对输入/输出投影和 local heads 做 TP，并接入 sequence parallel、context parallel 与 FLA 高性能路径；`lora.py` 的 `LinearLoRA` 还能拆分 latent rank/output 并配套 gather、reduce-scatter 或 input-gradient all-reduce。没有发现显式的 Qwen3.6 model registration。当前 Qwen native 的 full-attention TP 数学布局已经对齐，但 Q/K/V 仍是三次独立 GEMM，LoRA 的 replicated 一侧也比 Megatron Lite 更保守；GDN、MoE、embedding/LM-head 和多轴并行仍是主要差距。

PP 会把层分到不同 stage 并使用 1F1B 等调度；CP 会在序列和 attention state 上做跨 rank 通信。当前 Qwen native `TrainingContext` 仍在每个进程执行完整层栈，因此仅增加 PP/CP 配置不能得到正确语义。

当前 DP/EP 也不是完整 Megatron 语义：DP 同步 replicated LoRA 梯度并按租户 token count 归一化，expert 参数留在 EP rank；EP 默认使用 routed-output all-reduce，实验 gate 已加入 variable-split dispatch/inverse combine 和 fixed/dynamic LoRA data sharding。dynamic native path 的 source row metadata 已随 token index 传输，但仍逐 top-k 执行 host-visible count sync，没有 fused permutation、GPU-only split planning、异步 overlap 或 DeepEP backend。server 的 `TrainMultiLora` 目前向各 worker 广播相同 batch，因此不能把它当作 source-sharded 服务吞吐。

### 优化器与恢复

固定 LoRA 的 Adam 状态可导出/导入，且 native context 的 logical step 需要与 checkpoint step 对齐。dynamic adapter 的请求频率不同，每租户拥有独立 optimizer step、m/v 与 FP32 accumulator；仍缺少跨 optimizer group 的事务性回滚。

### 性能工程

当前粗粒度 C++ FFI、grouped MoE 和 activation checkpoint/offload 是有效优化，但 full attention 仍通过 ATen linear/SDPA，QKV 未融合，TP collective 同步执行，GDN 和 vocab 路径仍复制。尚无 Megatron/Transformer Engine 级别的端到端数据：没有完整模型在同一 GPU、序列长度、microbatch、精度和通信配置下的 tokens/s、显存、扩展效率对照，也没有 FP8/FP4 参数与 fused attention/DeepEP 的 Qwen 路径。

本次 native benchmark 没有证明 gated A2A 的端到端 step-time 优势：在该小型 workload 上 sharded A2A 的中位 step 反而比 legacy 高约 `23%`。它没有实现 DeepEP 的 fused permutation、GPU-only split planning 或通信计算 overlap，也没有覆盖 H=`2048`/E=`256` 的完整 Qwen3.6 workload。legacy 模式复制输入 batch，因此必须同时报告唯一样本吞吐，不能只看所有 rank 的 processed tokens/s。

目标 H20 的 ABI1 环境有 PyTorch 2.12.1、Triton 和 NumPy，但没有 Megatron、Transformer Engine、FLA、DeepEP、flash-attn、Apex 或缓存的兼容 prebuilt wheel。虽然本地 Megatron Lite 源码包含 Qwen3.5 LoRA adapter，目标机仍不具备其高性能依赖。因此当前不能诚实地产出 matched Megatron-LoRA benchmark，且本工作没有通过 JIT 或自构建依赖绕过该限制。

## 验证边界

已运行的验证包括 Rust 编译检查、core 单测、Qwen3.6 配置/集成测试，以及 H20 ABI11 的单卡、TP2、EP2 和 DP2 native smoke。DP2 的 weighted m/grouped/v/Adam BF16 delta oracle 分别达到 `2.43e-8`、`2.27e-8`、`7.33e-8` 和 `0`；legacy EP2 与 full-expert reference 的 loss、LoRA、Adam state 和标准 Adam 首步 oracle 在两 rank 均为零差异。Replicated A2A 与 fixed-LoRA sharded A2A 也均通过两 rank full-expert reference；sharded token counts `[1,3]` 的加权 loss 与 global reference 相差约 `9.6e-7`，m/v 最大差 `1.22e-5` / `3.92e-9`，Adam oracle 差为 `0`。H20 ABI1 dynamic sharded full-reference smoke 在两 rank 返回 `0`：dynamic grouped-expert 参数最大差 `1.53e-5`，m/v 最大差 `4.88e-5` / `5.75e-8`。ABI13 dense base-MLP TP2 smoke 对 gate/up/down 使用半尺寸本地权重，eval/train loss 与完整权重参考相差 `1.84e-5`。

ABI14 的 4Q/2KV-head GQA full-attention TP2 smoke 覆盖 Q/K/V/O fixed 和 selected dynamic LoRA。fixed eval/loss 与完整参考最大差 `4.40e-4`；Q/K/V-B 与 O-A 梯度最大差分别为 `1.83e-4`、`1.14e-5`、`3.05e-4` 和 `1.83e-4`，FP32 Adam m/v 最大差 `1.53e-5` / `6.63e-9`，本地标准 Adam 公式误差小于 `3.73e-9`。selected dynamic loss 最大差 `6.82e-5`，Q/K/V/O 参数最大差 `3.05e-5`。另一个两层 GDN smoke 直接验证 latent-rank TP 的 input-dgrad backward all-reduce：A/B 梯度最大差 `9.54e-7` / `4.77e-7`，m/v 最大差 `4.77e-8` / `3.71e-14`，本地 Adam 误差为 `0`。BF16 参数直接对照最多跨约两个量化 bin，因此验收以 FP32 梯度、m/v 和本地 Adam 公式为主。没有完成完整大模型长时间训练、跨节点通信、GDN/MoE/vocab base TP、PP/CP smoke 或与 Megatron-LM 的同条件 benchmark。因此“正确”应理解为已覆盖且有直接 oracle 的子集，而不是所有并行配置。

## 继续达到 Megatron 级别所需的最小工作包

1. 把现有 5D TP/PP/DP/EP/CP topology contract 落成可组合 runtime process groups、launcher 和 checkpoint rank mapping；当前 native 仍拒绝多轴组合。
2. 补齐 Qwen GDN attention、MoE、embedding/LM-head/CE 的 TP shard，并为 dense MLP 增加 projection-aware LoRA、fused gate/up FC1 和 MTP 支持；为 full attention 融合 QKV/SDPA 并增加 sequence parallel；为 PP 实现 stage forward/backward 与 1F1B scheduler；为 CP 实现 ring attention/state exchange。
3. 将 EP dispatch/combine 替换为 fused/异步路径，并测量通信与计算重叠。
4. 为 checkpoint 增加跨 topology reshard、可恢复的 pending accumulation state，并为旧 v3 attention checkpoint 提供离线迁移工具。
5. 在固定硬件和 workload 上，与 Megatron-LM 记录 tokens/s、step time、峰值显存、通信占比和 loss 曲线。

## Native EP Benchmark Artifact

`crates/rustrain-qwen3-6/tests/native_ep_bench.cpp` is a dependency-free
synthetic baseline for the native Qwen C ABI. It times complete
`qwen36_train_step` calls (forward, backward, and Adam) with CUDA events and
accepts `BENCH_SEQ`, `BENCH_HIDDEN`, `BENCH_EXPERTS`, `BENCH_INTERMEDIATE`,
`BENCH_WARMUP`, and `BENCH_ITERS`. It reports processed and unique tokens/s,
per-rank step statistics, and free memory. This is not a Megatron-LM
comparison and does not claim DeepEP or Transformer Engine parity.

Fresh H20 ABI0 run (`seq=128, hidden=256, experts=8, intermediate=256,
warmup=2, iters=10`, two ranks):

| Mode | Rank | Median step | Mean step | Processed tokens/s | Unique tokens/s | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy EP (`QWEN36_EP_A2A=0`) | 0 | 5.4752 ms | 5.4902 ms | 46,391 | 23,196 | 0 |
| Legacy EP (`QWEN36_EP_A2A=0`) | 1 | 5.4672 ms | 5.4591 ms | 46,459 | 23,230 | 0 |
| Sharded A2A (`QWEN36_EP_A2A=1`, `QWEN36_EP_A2A_SHARDED=1`) | 0 | 6.7192 ms | 6.7459 ms | 37,802 | 37,802 | 0 |
| Sharded A2A (`QWEN36_EP_A2A=1`, `QWEN36_EP_A2A_SHARDED=1`) | 1 | 6.7261 ms | 6.7458 ms | 37,763 | 37,763 | 0 |

Each rank processed 127 local tokens and 254 processed tokens per step. In
replicated legacy EP, unique tokens/s is 127 tokens divided by step time;
sharded A2A has 254 unique global tokens per step. These are synthetic native
measurements only; workload, precision, packing, and communication semantics
are not matched to Megatron-LM.
