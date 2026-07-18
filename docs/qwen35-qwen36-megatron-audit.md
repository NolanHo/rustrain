# Qwen3.5/3.6 LoRA 并行与性能审计

本文记录当前 native Qwen3.5/3.6 LoRA 后端与 Megatron-LM 级训练栈的边界。结论按实际代码和 smoke/integration 结果整理，不把配置字段或通用拓扑类型当作已经实现的 kernel。

## 结论

- 模型语义：Qwen3.5 dense、Qwen3.6 dense/MoE 的 native forward/backward 路径已经覆盖 hybrid full attention、GDN、MoE、MTP 和 LoRA 目标模块；已有配置解析、合成 oracle、集成测试及 H20 native smoke 证据，但尚未完成真实 35B/3.6 权重的长时间训练验证。
- 已实现并可验证的分布式子集：ABI26 可按 `TP-CP-EP-DP-PP` 五维拓扑建立、缓存和 attach 正交 NCCL process groups；LoRA 模型执行已覆盖可组合的 TP、MoE EP 和 expert-DP，包含 TP2 x EP2、TP2 x DP2 和 TP2 x EP2 x expert-DP2 native oracle。CP2 x PP2 communicator smoke 已验证交叉分组和全网 max 传播，但 PP stage 与 CP sequence/attention 语义仍 fail-closed。DP 动态租户按 adapter token count 加权，sharded A2A 保留 source flattened row 来恢复租户，并按全局租户 token count 归一化。
- 性能：ABI19 将 top-k assignment 合并为单次 packed dispatch/combine，并对接收 token 做一次 expert sort 和 grouped GEMM。H20 TP2 x EP2 端到端 fixed-LoRA benchmark 按每步 TP x EP 全局最慢 rank 计时，p50 从 `9.007 ms` 降到 `6.499 ms`，unique-token throughput 从 `56.84k/s` 提升到 `78.78k/s`；这是 native legacy/packed 对照，不是 Megatron 对比。
- 已实现 LoRA latent-rank TP、frozen full-attention/GDN/dense SwiGLU MLP TP、routed/shared ETP，以及 embedding/LM-head/vocabulary TP。GDN 按 K/V head group 切分复合 QKV、depthwise conv、Z/A/B、A_log/dt_bias 和 out-proj input columns；fixed 与 selected dynamic LoRA 使用 projection-aware shard 和梯度 reduction。ABI20 对 routed fused gate/up 的 gate/up 两半分别切分后重排，并在 routing weight 前归约 local expert partial。TP 可与 EP 或 expert-DP 组合；新增 `QWEN36_SEQUENCE_PARALLEL=1` 的 guarded TP2 dense fixed-LoRA 子集，按 Megatron 语义执行 sequence scatter/all-gather/reduce-scatter/loss gather，但 CP/PP/EP/DP、MoE、MTP、dynamic LoRA 和 latent-rank target 仍 fail-closed。
- CLI 与 server 共享 checkpoint v5 实现。分布式 CLI 训练按 shared run ID 隔离 rank 日志，以唯一 save transaction generation 协调 rank shard，原子发布标准 PEFT 目录，并支持相同 topology 下恢复 fixed LoRA、FP32 Adam m/v 和独立 fixed optimizer step。checkpoint library 不再把 run-scoped attempt ID 隐式当作 save generation；缺失、空、非法或复用 generation 的分布式保存直接拒绝。新 writer 用内容 digest 绑定 manifest/tensor，并发布定长 compact rank receipt；每个进程预检 receipt set 后只解析、hash 和加载本地完整 manifest/tensor。distributed server 的 save/load coordinator 在同一个 dispatch lock 内执行两阶段事务：save 写入 exclusive sibling staging，要求所有 rank receipt 完整并用 shadow context 全量 hydrate 验证，再以 `RENAME_NOREPLACE` 发布 fresh destination；load 先在所有 rank 建立 no-sync shadow context，全部成功后才交换 live context，commit 不完整则 worker group fail-stop。当前不覆盖已有 destination，不提供 `LATEST` generation 指针或断电级目录 fsync 保证；跨 topology reshard 仍未实现。
- ABI22 server 训练数据面在 parent/worker IPC 间使用 64-byte aligned binary tensor slab：parent 只做一次 base64 decode/packing，worker 零拷贝借用 host `int64` span，按 topology 选取本地 rows，并通过一次粗粒度 C++ 调用完成 H2D 与完整 train/eval step。shared-memory layout、epoch、span 边界/重叠和 timeout poison 均有校验；默认 slab 为 32 MiB，可通过环境变量配置。tensor routes 在 body decode 前使用单许可 admission gate，与本来串行的 IPC dispatch 对齐并限制并发请求内存。HTTP 边界仍是 JSON/base64，host memory 也尚未 pin，因此这不是最终的高吞吐 ingress。
- 因此当前实现不能宣称“Megatron-LM 级别”。它是一个计算集中在 C++ 的 LoRA TP/EP/expert-DP 子集，离 Megatron 的完整并行和通信重叠仍有实质差距。

## 当前能力矩阵

| 能力 | 当前状态 | 证据/限制 |
| --- | --- | --- |
| Qwen3.5 full attention | 已实现 | native smoke、Qwen3.5 配置集成测试 |
| Qwen3.5 GDN/linear attention | 已实现 | CUDA delta-rule forward/backward 与 native smoke |
| Qwen3.6 MoE | 已实现 | grouped dispatch、EP smoke；完整模型仍需目标 GPU/权重运行 |
| MTP | 已实现 | C++ hidden gradient 检查和集成测试；可通过环境变量关闭 |
| fixed LoRA | 已实现 | attention/GDN/MLP/shared/routed expert 目标模块 |
| dynamic multi-LoRA | 已实现子集 | selected v2 默认把不同 rank/target 的租户按 projection-local 最大 rank 补零，保留各租户 `alpha / logical_rank`，在一个 activation batch 内完成一次 forward/backward，再以一次 FinalizeOnly 调用独立更新各租户参数、m/v 和 optimizer clock；未命中的 target 使用零张量且不更新。`QWEN36_HETERO_PADDED_BATCH=0` 保留按 signature 分组和跨组回滚。checkpoint 可独立恢复 heterogeneous rank/alpha/targets；HTTP 可在显式 `allow_aggregate_loss=true` 时有界合并兼容且 adapter ID 不相交的并发请求，响应保留 aggregate scalar、adapter loss 和真实 optimizer step；`expected_steps` 在原生 TP/EP/DP collective 前做全秩乐观并发校验，过期重试 fail closed；默认仍保持单请求 dispatch，dynamic+MTP 暂拒绝 |
| microbatch accumulation | 已实现子集 | non-final microbatch 只 backward，final microbatch 才 optimizer；FP32 accumulator 存储/聚合，autograd leaf backward 仍为 BF16 |
| replicated data parallel | 已实现 | logical-step 边界同步 replicated LoRA；EP expert 参数不走该 reduction |
| expert parallel | 已实现子集 | 默认 routed-output all-reduce；gated variable-split A2A 已验证 fixed-LoRA 和 native dynamic-LoRA data sharding；GPU-only split planning、异步 overlap 和 DeepEP backend 未实现 |
| tensor parallel | attention + GDN + dense/expert MLP + vocabulary 子集 | full attention 与 GDN 使用 head-aligned ColumnParallel input projection 和 RowParallel output projection；GDN 的 flat QKV/conv 按 `[Q_local|K_local|V_local]` 重排。embedding/LM-head/CE 使用 vocabulary shard；routed/shared expert 使用 gate/up output shard、down input shard，并已在 TP2 x EP2 验证。CLI/server 均可通过共享 v5 rank shard 合并标准 PEFT |
| sequence parallel | 已实现受限子集 | `QWEN36_SEQUENCE_PARALLEL=1` 仅允许 TP2、dense text-only、fixed projection-aware LoRA、CP/EP/DP/PP=1、MTP/aux loss=0；embedding scatter、column all-gather、row reduce-scatter、loss gather 在 H20 两进程 NCCL smoke 中通过，任意 latent-rank target fail-closed |
| pipeline parallel | communicator 已实现，模型语义未实现 | ABI26 建立/缓存 PP group；没有 stage 切分、microbatch scheduler 或 activation send/recv，非单例执行 fail-closed |
| context parallel | communicator 已实现，模型语义未实现 | ABI26 建立/缓存 CP group；没有 ring attention、跨 rank KV/索引合并，非单例执行 fail-closed |
| distributed checkpoint | 已实现子集 | v5 记录 rank order、完整五维坐标、TP/EP 多轴 placement、fused gate/up segments 及 fixed/dynamic slot identity；任意 multi-rank 拓扑使用 rank 目录。compact receipt 将每 rank preflight 限制为 `O(world_size)` 小 metadata + `O(local state)`；CLI/server 共享 same-topology restore 和标准 PEFT merge。server save/load 使用全 rank prepare/validate/commit-or-abort，load 通过 no-sync shadow context 保证失败不改 live state；fresh destination 以 `RENAME_NOREPLACE` 原子可见。v3/v4 及旧 digest-v5 保持只读兼容；可覆盖 generation/LATEST、断电级 durability、跨 topology reshard 和 PP/CP 未实现 |
| server tensor transport | 已实现 IPC 子集 | ABI22 parent/worker 共享 64-byte aligned binary slab 与 compact JSON descriptor；worker 不再反序列化三份 `Vec<i64>` 或在 Rust 热路径构造 `tch::Tensor`。普通 tensor HTTP routes 在 decode 前只允许一个 in-flight 请求，忙时快速返回 `429`；`/train_multi` 允许有界并发 admission，并仅对显式接受 aggregate loss 的兼容请求执行短窗口 GPU coalescing。普通控制面和 checkpoint transaction 使用 bounded FIFO 单消费者 scheduler（默认队列 32，可由 `RUSTRAIN_EP_DISPATCH_QUEUE_CAPACITY` 调整），每个 accepted job 在 `spawn_blocking` 上串行执行。HTTP ingress 仍为 JSON/base64，pinned staging、异步 H2D 和 raw binary endpoint 未实现 |

## 与 Megatron-LM 的关键差距

### 并行语义

Megatron 的通用 MLP 使用 ColumnParallel fc1、local gated activation 和 RowParallel fc2，并进一步提供 fused fc1/activation、sequence parallel、通信重叠和 sharded-state 支持。当前 Qwen native 已补上算法等价的 separate gate/up row shard 与 down column shard，但仍是两次独立 GEMM。

本地 Megatron-LM 的 `experimental/lite/megatron/lite/model/qwen3_5` 已包含直接的 Qwen3.5 实现和 LoRA adapter，而不是只能依赖外部 bridge：`primitive/modules/gqa.py` 使用 fused QKV ColumnParallel 和 O RowParallel；`gated_delta_net.py` 对输入/输出投影和 local heads 做 TP，并接入 sequence parallel、context parallel 与 FLA 高性能路径；`lora.py` 的 `LinearLoRA` 还能拆分 latent rank/output 并配套 gather、reduce-scatter 或 input-gradient all-reduce。没有发现显式的 Qwen3.6 model registration。当前 Qwen native 已补上一个真实但严格受限的 TP2 sequence-parallel dense 路径，数学布局与 Megatron 的 gather/reduce-scatter 方向一致；但 full attention 的 Q/K/V 仍是三次独立 GEMM，GDN 没有 CP/FLA chunk kernel，LoRA 的 replicated 一侧也比 Megatron Lite 更保守，MoE、PP/CP、多轴组合和通信重叠仍是主要差距。

PP 会把层分到不同 stage 并使用 1F1B 等调度；CP 会在序列和 attention state 上做跨 rank 通信。ABI26 已把 PP/CP communicator 纳入 `TrainingContext` 和进程级缓存，并通过五轴 size consensus 防止 rank 执行不同的 split 序列；4 卡 CP2 x PP2 smoke 的正交组全局 max 为 `3.0`。新增的 sequence-parallel 只覆盖 TP2 dense 单轴，不能替代 PP stage ownership、CP ring attention 或任意五维组合。NCCL 文件 rendezvous 按 generation 隔离：launcher 使用共享 `RUSTRAIN_LAUNCH_OUTPUT_DIR`，直接多节点任务必须提供唯一 `RUSTRAIN_NCCL_RUN_ID` 和共享 `RUSTRAIN_NCCL_SYNC_DIR`。当前所有非单例 PP/CP train/eval 入口仍在读取输入前拒绝执行。

当前 DP/EP 仍不是完整 Megatron 语义：DP 同步 replicated LoRA 梯度并按租户 token count 归一化，expert 参数留在 EP rank；sharded EP 使用 variable-split dispatch/inverse combine 和 fixed/dynamic LoRA data sharding。ABI19 已把所有 top-k assignment 合并为一次 dispatch，并把 local expert 计算合并为 grouped GEMM，但 count planning 仍可见于 host，且没有 fused permutation、异步 overlap 或 DeepEP backend。server 广播同一个 global batch descriptor，worker 按五维 topology 选择本 source rows：TP peers 保持相同，sharded EP 使用 `DP*EP` sources，replicated EP 只按 DP 分片。ABI22 将 parent/worker 数据从每 worker 全量 JSON vector 反序列化改为共享 binary slab；worker 直接借用 slab span 并调用 C++ host-i64 coarse ABI，Rust worker 热路径不再执行 `Tensor::from_slice`、`reshape` 或 `to_device`。普通控制面/检查点现在由 bounded FIFO 单消费者调度器承接，避免 Tokio handler 被阻塞式 IPC 占用；tensor 请求仍单 in-flight，避免多个 48 MiB body 同时驻留，因而它不是多租户 GPU batching。IPC timeout 会永久 poison channel，首次失败立即终止并回收 worker；pre-publish `InvalidInput` 不 poison channel。health 在 terminal failure 后返回 unavailable，partial launch 同样回收已启动 ranks。HTTP ingress 仍是 base64/JSON，且缺少 pinned host staging、异步 H2D 与请求/计算 overlap，因此仍不能视为最终的高吞吐服务传输。

### 优化器与恢复

固定 LoRA 的 Adam 状态可导出/导入，native context 的 logical step 与 checkpoint step 对齐；恢复时校验 canonical base-model path，CPU safetensors state 会复制到 shadow CUDA/FP32 allocation。dynamic adapter 的请求频率不同，每租户拥有独立 optimizer step、m/v 与 FP32 accumulator。server checkpoint load 只有在 fixed/dynamic LoRA、全部 m/v、各 optimizer clock 和 session metadata 都 hydrate 成功后才交换 context。heterogeneous selected v2 默认把不同 signature 的 adapter 安装到同一个 padded registry，一次 TrainOnly 后再执行一次 FinalizeOnly；低 rank 的真实 leaf 通过可微 `cat` 补零，inactive projection 使用同 geometry 的零张量，因此不会破坏 autograd 或错误更新 target hole。旧的 deterministic signature grouping 与 persistent shadow 跨组回滚仍可通过环境变量启用；同时仍缺少基于模型内容 fingerprint 的身份校验。

### 性能工程

ABI24 adds a packed LoRA gradient synchronization path: FP32 accumulators are
grouped by EP/DP/TP reduction mask and each populated bucket/axis uses one NCCL
all-reduce. `QWEN36_PACKED_LORA_SYNC=0` keeps the per-tensor grouped fallback.
This reduces collective launch count, but it still runs after full backward and
the token-count CPU fence; it is not Megatron-style backward bucket overlap or
reduce-scatter. H20 TP2 fixed-LoRA smoke and TP2 x DP2 oracle passed with both
settings; the synthetic TP2 benchmark was `77.970 ms` packed versus `78.024 ms`
fallback p50, within measurement noise.

The GDN backward path now has a separately validated CUDA optimization. The
reverse recurrence restores `R_t` while reading `S_t` for the direct output
gradient, so the old standalone state-undo sweep and three per-token barriers
are removed. Fusion is enabled by default and can be disabled with
`QWEN36_GDN_RECURRENT_FUSION=0`. On H20, the matched `B=2,S=512,H=2048,L=3`
benchmark improved single-rank p50 from `80.272` to `75.685 ms` and TP2 from
`75.908` to `70.885 ms`; TP2 at `B=8` improved from `157.986` to `148.216 ms`.
An independent ATen recurrence-backward TP2 smoke preserved fixed/dynamic loss,
FP32 m/v and parameter deltas (`adam_error=0` on both ranks). This is a local
GDN-kernel improvement, not evidence of FLA, sequence/context parallelism, or
matched Megatron end-to-end throughput.

Server control-plane and checkpoint commands now use a bounded FIFO
single-consumer dispatcher. Accepted jobs run one-at-a-time on
`spawn_blocking`, preserving IPC collective order while keeping Tokio handlers
responsive; queue pressure returns `429` and scheduler/worker failure returns
`503`. The default capacity is 32 (hard maximum 4096). Tensor train/eval routes
still admit only one 48 MiB body and reject while busy, so this is runtime
backpressure/fairness rather than cross-tenant GPU batching. The separate
`/train_multi` route now supports the opt-in bounded coalescer described below.

The EP `/train_multi` route now adds an opt-in cross-request coalescer. A client
must set `allow_aggregate_loss=true`; otherwise the legacy one-request dispatch
and request-local loss are preserved. Opt-in requests are grouped only when
session, sequence/source layout, LoRA rank, and adapter IDs are compatible and
disjoint. The coalescer rebuilds source-major rows into one native heterogeneous
batch, is bounded by request/adapter/payload limits, and seals its window before
ordinary tensor, registry, or checkpoint operations. Responses expose
`loss_scope=coalesced_batch` and the number of merged requests. ABI25 requires
the native report symbols and returns adapter-ordered losses: it sums token-loss
numerators and supervised-token counts only over DP and source-sharded EP,
without double-counting TP or replicated EP. The server slices that vector back
to each request while retaining the explicitly labelled aggregate scalar for
wire compatibility. Older native libraries are rejected at load time; within
ABI25, report/legacy call mode is negotiated collectively before report-only
reductions. IPC slab wire version 2 rejects old parent/worker layouts, and the
coordinator verifies rank-consistent report values. The coalescer is capped at
2048 adapters for the fixed 256 KiB result slot; an oversized serialized result
is replaced by a compact error and still signals the waiting parent. This is a
scheduling/transport optimization, not PP/CP or DeepEP communication overlap.

GDN fused backward 现在可通过
`QWEN36_GDN_STATE_CHECKPOINT_STRIDE` 保存固定 token 边界的 FP32 recurrent
state，并在每个 reverse chunk 开始时从精确 chunk-end state 重启。默认值
为 `0`，因为每层额外占用约
`BH * (ceil(S / stride) + 1) * DK * DV * 4` bytes。该路径限制跨 chunk 的
反向重建误差累积，但 chunk 内仍通过 clamp 后的 decay 反除，不能宣称已
解决单 token 极小 decay 的数值问题，也没有提供 FLA/sequence-parallel
chunk kernel 或 packed `cu_seqlens`。

在此 checkpoint 基础上，显式 `QWEN36_GDN_CHUNKWISE_BWD=1` 启用两阶段 AOT
backward：轻量串行 pass 生成精确 `dS` chunk 边界，随后以
`(batch-head, chunk)` CTA 并行 replay 完整参数梯度。H20 TP2
`B=2,S=512,H=2048,L=3,stride=32` 的重复 matched p50 为
`71.484 -> 53.458 ms`（`-25.2%`），TP2 stride-2 fixed/dynamic oracle 的
`adam_error=0`。该 flag 和 checkpoint stride 已进入 distributed topology
hash，非法 checkpoint 配置 fail-fast，checkpoint address 使用 `size_t`。
该路径不能默认开启：single-rank p50
`76.195 -> 81.843 ms`（回退 `7.4%`），TP2 `B=8` 仅改善约 `1.15%`，且
该 workload 的 state checkpoint 约增加 `100 MiB/GPU`。

当前粗粒度 C++ FFI、packed/grouped MoE、GDN head TP、vocabulary TP 和 activation checkpoint/offload 是有效优化，但 full attention 仍通过 ATen linear/SDPA，QKV 未融合，TP collective 同步执行。heterogeneous selected v2 的 padded 路径消除了按 signature 重复完整 trainer 和组间 GPU success fence；在 H20 TP2 x EP2 的四租户 synthetic workload 上，p50 从 grouped 的 `17.750 ms` 降到 `13.841 ms`，unique-token throughput 从 `57.69k/s` 提升到 `73.98k/s`，但 allocator peak 从约 `0.240 GiB` 增至 `0.354 GiB`，且 10 次样本的 p95 为 `20.152 ms`，未优于 grouped 的 `19.441 ms`。这证明 unified padded batch 对当前 native 路径有效；HTTP 仅在显式 `allow_aggregate_loss=true` 时合并兼容请求，且 aggregate loss 会被明确标记，不是 Megatron 对比。尚无 Megatron/Transformer Engine 级别的完整模型端到端数据：没有同一 GPU、序列长度、microbatch、精度和通信配置下的 tokens/s、显存、扩展效率对照，也没有 FP8/FP4 参数与 fused attention/DeepEP 的 Qwen 路径。

ABI15 synthetic GDN TP benchmark 使用 3 层、S=512、H=2048、16 K heads、32 V heads 和 LoRA rank 8。在 B=2 时 single/TP2 p50 为 `81.10/约 76.06 ms`，只有约 `1.07x`；在 B=8 时为 `303.25/约 158.04 ms`，达到约 `1.92x`，每卡 observed resident 从 `4.59 GiB` 降到 `3.54 GiB`。`nsys` 的 B=2 TP2 trace 中 fused delta-rule backward 占 GPU kernel time `80.3%`、forward 占 `7.4%`，NCCL all-reduce 合计低于 `0.4%`。小 batch 下 single 和 TP2 都只需相近的 persistent-block wave 数，因此不能期待 head TP 自动加速；dynamic multi-LoRA batching 提高 BH 后才接近线性 scaling。这是 synthetic native 结果，不是完整模型或 matched Megatron 对比。

本次 native benchmark 没有证明 gated A2A 的端到端 step-time 优势：在该小型 workload 上 sharded A2A 的中位 step 反而比 legacy 高约 `23%`。它没有实现 DeepEP 的 fused permutation、GPU-only split planning 或通信计算 overlap，也没有覆盖 H=`2048`/E=`256` 的完整 Qwen3.6 workload。legacy 模式复制输入 batch，因此必须同时报告唯一样本吞吐，不能只看所有 rank 的 processed tokens/s。

上述结果是 ABI0 的纯 EP2 小模型历史基线。ABI19 在 TP2 x EP2、相同 source-sharded 语义下做了 matched routing-slot/packed 对照，并在每步训练后沿 EP、expert-DP、TP 依次做 MAX，从而让 4 个 rank 使用相同的全局最慢 rank 样本：p50 `9.007 -> 6.499 ms`，unique-token throughput `56.84k -> 78.78k/s`。Nsight Systems process-tree trace 同时记录到 CUDA kernel launch `6812 -> 5168`、NCCL SendRecv `96 -> 48` 和 AllGather `24 -> 12`；该 trace 采集于 benchmark-only world-max metrics collective 加入之前，因此不包含这个计时区间外的聚合。它证明 packed dispatcher 对当前 native 路径有效，但仍不等于 DeepEP 或 Megatron 的重叠能力。新增 TP2 dense sequence-parallel H20 smoke 只验证 all-gather/reduce-scatter/loss-gather 与固定 LoRA 更新，不提供吞吐结论。

目标 H20 的 ABI1 环境有 PyTorch 2.12.1、Triton 和 NumPy，但没有 Megatron、Transformer Engine、FLA、DeepEP、flash-attn、Apex 或缓存的兼容 prebuilt wheel。虽然本地 Megatron Lite 源码包含 Qwen3.5 LoRA adapter，目标机仍不具备其高性能依赖。因此当前不能诚实地产出 matched Megatron-LoRA benchmark，且本工作没有通过 JIT 或自构建依赖绕过该限制。

## 验证边界

已运行的验证包括 Rust 编译检查、core 单测、Qwen3.6 配置/集成测试，以及 H20 ABI11 的单卡、TP2、EP2 和 DP2 native smoke。DP2 的 weighted m/grouped/v/Adam BF16 delta oracle 分别达到 `2.43e-8`、`2.27e-8`、`7.33e-8` 和 `0`；legacy EP2 与 full-expert reference 的 loss、LoRA、Adam state 和标准 Adam 首步 oracle 在两 rank 均为零差异。Replicated A2A 与 fixed-LoRA sharded A2A 也均通过两 rank full-expert reference；sharded token counts `[1,3]` 的加权 loss 与 global reference 相差约 `9.6e-7`，m/v 最大差 `1.22e-5` / `3.92e-9`，Adam oracle 差为 `0`。H20 ABI1 dynamic sharded full-reference smoke 在两 rank 返回 `0`：dynamic grouped-expert 参数最大差 `1.53e-5`，m/v 最大差 `4.88e-5` / `5.75e-8`。ABI13 dense base-MLP TP2 smoke 对 gate/up/down 使用半尺寸本地权重，eval/train loss 与完整权重参考相差 `1.84e-5`。

ABI14 的 4Q/2KV-head GQA full-attention TP2 smoke 覆盖 Q/K/V/O fixed 和 selected dynamic LoRA。fixed eval/loss 与完整参考最大差 `4.40e-4`；Q/K/V-B 与 O-A 梯度最大差分别为 `1.83e-4`、`1.14e-5`、`3.05e-4` 和 `1.83e-4`，FP32 Adam m/v 最大差 `1.53e-5` / `6.63e-9`，本地标准 Adam 公式误差小于 `3.73e-9`。selected dynamic loss 最大差 `6.82e-5`，Q/K/V/O 参数最大差 `3.05e-5`。

ABI15 的两层 GDN base-TP smoke 覆盖复合 QKV/conv head shard、Z/A/B/A_log/dt_bias、replicated norm、out-proj input columns，以及五种 GDN projection 的 fixed/selected dynamic LoRA。fixed loss 最大差 `1.30e-4`，rank-local factor 梯度最大差 `2.44e-4`，FP32 Adam m/v 最大差 `3.43e-6` / `2.49e-10`；dynamic loss 最大差 `1.16e-3`，m/v 最大差 `6.10e-6` / `1.22e-9`，标准 Adam 公式误差均为 `0`。latent-rank-only GDN TP2 回归也通过，loss 差 `5.46e-5`。BF16 参数直接对照最多跨约两个量化 bin，因此验收以 FP32 梯度、m/v 和本地 Adam 公式为主。ABI21 world8 TP2 x EP2 x expert-DP2 shadow resume 在 no-sync attach 前故意扰动 DP 非零 rank 的 fixed LoRA，attach 后保持 bitwise 不变；hydrate 后 fixed/dynamic LoRA、Adam m/v 和 tenant clock 精确一致，第二步的参数/m/v 与连续训练零差，step 为 `2`。同一 world8 fixture 已分别覆盖 source-sharded 与 replicated-source A2A；replicated-source 的 fixed 参数/m 最大差 `1.96e-3` / `2.58e-6`，dynamic 参数/m 最大差 `1.99e-3` / `4.16e-5`，两者 Adam 公式误差均为 `0`。ABI22 的单卡 native smoke 验证 borrowed host-i64 eval 与 tensor ABI loss 零差、fixed train Adam oracle 零差，以及 selected dynamic multi-LoRA 仅更新已选 adapter；同一 ABI22 world8 TP2 x EP2 x expert-DP2 `tri-smoke` 也通过 checkpoint resume、selected isolation 与 Adam oracle。heterogeneous restore smoke 在已有 rank-8 expert adapters 后以 no-sync hydration 注册 rank-9 `q_proj` adapter，验证 checkpoint registry 可保留独立 signature。ABI23 进一步覆盖 live heterogeneous registration、tensor/borrowed host-i64 selected v2 dispatch，以及第二组注入失败后参数、Adam m/v 和 tenant clock 的精确回滚。TP2 x EP2 x expert-DP2 world8 使用 rank-3 `q_proj` 与 rank-4 multi-module tenants 验证所有 rank 的成功更新和失败回滚，未出现 collective 顺序分叉。新的 padded 路径在单卡和 TP2 x EP2 smoke 中覆盖 rank-8/rank-9、dense/expert target holes、独立 clock、零 token tenant 与 rollback；显式 grouped fallback 的 TP2 x EP2 smoke 也通过。没有完成完整大模型长时间训练、真实跨节点通信、PP/CP smoke 或与 Megatron-LM 的同条件 benchmark。因此“正确”应理解为已覆盖且有直接 oracle 的子集，而不是所有并行配置。

## 继续达到 Megatron 级别所需的最小工作包

1. 为已实现的 opt-in server batch coalescing 增加 HTTP response capability/version negotiation，使支持 per-adapter loss 的客户端可取消显式 aggregate opt-in；补齐 padding rank inflation 容量模型和长尾 benchmark。
2. 实现 PP/CP runtime process groups、launcher、scheduler 和 checkpoint rank mapping；为 PP 实现 stage forward/backward 与 1F1B scheduler，为 CP 实现 ring attention/state exchange。
3. 将 EP dispatch/combine 替换为 fused/异步路径，并测量通信与计算重叠。
4. 为 dense/expert MLP 增加 fused gate/up FC1 和 MTP 支持；为 GDN 消除 chunk 内 decay 反除并增加 sequence/context parallel 与 packed `cu_seqlens`，为 full attention 融合 QKV/SDPA 并增加 sequence parallel。
5. 为 server 增加 raw binary HTTP ingress、pinned staging、异步 H2D/request overlap；为 checkpoint 增加 generation/`LATEST`、断电级 durability、跨 topology reshard、可恢复的 pending accumulation state 和旧 v3 attention checkpoint 离线迁移工具。
6. 在固定硬件和 workload 上，与 Megatron-LM 记录 tokens/s、step time、峰值显存、通信占比和 loss 曲线。

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
