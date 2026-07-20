# Qwen3.5/3.6 LoRA 并行与性能审计

本文记录当前 native Qwen3.5/3.6 LoRA 后端与 Megatron-LM 级训练栈的边界。结论按实际代码和 smoke/integration 结果整理，不把配置字段或通用拓扑类型当作已经实现的 kernel。

本轮补上了 native host-i64 入口的 opt-in `QWEN36_HOST_PINNED_STAGING=1`：context 缓存独立的 pinned input/target/attention staging，并以 non-blocking H2D 替代 pageable blocking copy。H20 TP2 x EP2 matched native benchmark（`B=2,S=64,H=1024,E=8,I=2048,L=1`，5 warmup/50 iterations）loss 保持 `5.411833`，pageable/pinned p50 为 `6.336/6.959 ms`，p95 为 `7.148/7.108 ms`；因此默认关闭，且不能把该路径等同于已经实现 request/compute overlap。

## 结论

- 模型语义：Qwen3.5 dense、Qwen3.6 dense/MoE 的 native forward/backward 路径已经覆盖 hybrid full attention、GDN、MoE、MTP 和 LoRA 目标模块；已有配置解析、合成 oracle、集成测试及 H20 native smoke 证据，但尚未完成真实 35B/3.6 权重的长时间训练验证。
- 已实现并可验证的分布式子集：ABI26 可按 `TP-CP-EP-DP-PP` 五维拓扑建立、缓存和 attach 正交 NCCL process groups；LoRA 模型执行已覆盖可组合的 TP、MoE EP 和 expert-DP，包含 TP2 x EP2、TP2 x DP2 和 TP2 x EP2 x expert-DP2 native oracle。固定 LoRA PP 已覆盖固定 shape、单 chunk 的 non-interleaved 1F1B，并在 H20 PP2/PP3 通过。严格受限的 dense GDN CP2 已实现 sequence-to-head/head-to-sequence exchange、fixed/dynamic LoRA、loss recovery 和 grouped gradient sync；full attention、ring CP、CP4 及 CP 与其他并行轴的模型执行仍 fail-closed。DP 动态租户按 adapter token count 加权，sharded A2A 保留 source flattened row 来恢复租户，并按全局租户 token count 归一化。
- 性能：ABI19 将 top-k assignment 合并为单次 packed dispatch/combine，并对接收 token 做一次 expert sort 和 grouped GEMM。H20 TP2 x EP2 端到端 fixed-LoRA benchmark 按每步 TP x EP 全局最慢 rank 计时，p50 从 `9.007 ms` 降到 `6.499 ms`，unique-token throughput 从 `56.84k/s` 提升到 `78.78k/s`；这是 native legacy/packed 对照，不是 Megatron 对比。
- 性能：ABI28 在 heterogeneous selected-v2 入口把 health、request、loss-report capability、report buffer 和 accumulation-clear 状态打包成一个 `int32[6]`，每个 EP/DP/TP 轴只做一次 MIN collective。H20 TP2 x EP2、B=8/S=128/8 active tenants 相对旧六次 collective 实现的三次交错 A/B，中位 p50 约从 `13.394 ms` 降到 `13.199 ms`，Nsight Systems 的 `u32` all-reduce 实例从 `1032` 降到 `752`；这是 native synthetic benchmark，不是 Megatron 对比。
- 性能：dynamic multi-LoRA 的 activation chunk 现在在 device 上累积 loss、hidden-gradient finite 和 MTP token-count validity，整个 logical step 末尾只保留一次 host read/collective consensus；selected loss report 的一次 D2H 也移动到 finalizer 前。这样把数值检查从 `O(chunks)` 个 host fence 降为 `O(1)`，同时保持 Adam/clock commit 前的 fail-closed 语义。H20 TP2 x EP2 uneven-source MTP、TP2 fused-MoE MTP、TP2 vocabulary-parallel MTP 和 EP dynamic regression 均通过；这属于 runtime synchronization 优化，不是 backward bucket overlap 或通信计算重叠。
- 已实现 LoRA latent-rank TP、frozen full-attention/GDN/dense SwiGLU MLP TP、routed/shared ETP，以及 embedding/LM-head/vocabulary TP。GDN 按 K/V head group 切分复合 QKV、depthwise conv、Z/A/B、A_log/dt_bias 和 out-proj input columns；fixed 与 selected dynamic LoRA 使用 projection-aware shard 和梯度 reduction。ABI20 对 routed fused gate/up 的 gate/up 两半分别切分后重排，并在 routing weight 前归约 local expert partial。TP 可与 EP 或 expert-DP 组合；新增 `QWEN36_SEQUENCE_PARALLEL=1` 的 guarded TP2 dense path，fixed 和 projection-aware dynamic LoRA 均可按 Megatron 语义执行 sequence scatter/all-gather/reduce-scatter/loss gather；CP/PP/EP/DP、MoE、MTP、aux loss，以及 latent-rank dynamic target 仍 fail-closed。
- CLI 与 server 共享 checkpoint v5 实现。分布式 CLI 训练按 shared run ID 隔离 rank 日志，以唯一 save transaction generation 协调 rank shard，原子发布标准 PEFT 目录，并支持相同 topology 下恢复 fixed LoRA、FP32 Adam m/v 和独立 fixed optimizer step。checkpoint library 不再把 run-scoped attempt ID 隐式当作 save generation；缺失、空、非法或复用 generation 的分布式保存直接拒绝。新 writer 用内容 digest 绑定 manifest/tensor，并发布定长 compact rank receipt；每个进程预检 receipt set 后只解析、hash 和加载本地完整 manifest/tensor。distributed server 的 save/load coordinator 在同一个 dispatch lock 内执行两阶段事务：save 写入 exclusive sibling staging，要求所有 rank receipt 完整并用 shadow context 全量 hydrate 验证，再以 `RENAME_NOREPLACE` 发布 fresh destination；load 先在所有 rank 建立 no-sync shadow context，全部成功后才交换 live context，commit 不完整则 worker group fail-stop。当前不覆盖已有 destination，不提供 `LATEST` generation 指针或断电级目录 fsync 保证；跨 topology reshard 仍未实现。
- ABI22 server 训练数据面在 parent/worker IPC 间使用 64-byte aligned binary tensor slab：parent 只做一次 base64 decode/packing，worker 零拷贝借用 host `int64` span，按 topology 选取本地 rows，并通过一次粗粒度 C++ 调用完成 H2D 与完整 train/eval step。shared-memory layout、epoch、span 边界/重叠和 timeout poison 均有校验；默认 slab 为 32 MiB，可通过环境变量配置。tensor routes 在 body decode 前使用单许可 admission gate，与本来串行的 IPC dispatch 对齐并限制并发请求内存。HTTP 边界仍是 JSON/base64，host memory 也尚未 pin，因此这不是最终的高吞吐 ingress。
- 因此当前实现不能宣称“Megatron-LM 级别”。它是一个计算集中在 C++ 的 LoRA TP/EP/expert-DP 子集，离 Megatron 的完整并行和通信重叠仍有实质差距。

## 当前能力矩阵

| 能力 | 当前状态 | 证据/限制 |
| --- | --- | --- |
| Qwen3.5 full attention | 已实现 | native smoke、Qwen3.5 配置集成测试 |
| Qwen3.5 GDN/linear attention | 已实现 | CUDA delta-rule forward/backward 与 native smoke |
| Qwen3.6 MoE | 已实现 | grouped dispatch、EP smoke；完整模型仍需目标 GPU/权重运行 |
| MTP | 已实现受限子集 | fixed LoRA 和 dynamic selected LoRA 的 C++ hidden gradient；单轴 DP、TP2 dense/真实 MoE prediction-layer shard、TP2xEP2 不均匀 source-token oracle 均有 H20 证据。TP2 MTP 现在支持 vocabulary-parallel embedding/LM-head/CE（loss diff `3.62e-3`），仍要求 CP/PP/sequence-parallel 关闭；PP/CP MTP 与完整模型长跑仍未验证；fixed-LoRA EP MTP 继续 fail-closed |
| fixed LoRA | 已实现 | attention/GDN/MLP/shared/routed expert 目标模块 |
| dynamic multi-LoRA | 已实现子集 | selected v2 默认把不同 rank/target 的租户按 projection-local 最大 rank 补零，保留各租户 `alpha / logical_rank`，在一个 activation batch 内完成一次 forward/backward，再以一次 FinalizeOnly 调用独立更新各租户参数、m/v 和 optimizer clock；入口 preflight 默认合并 6 个布尔状态 collective，未命中的 target 使用零张量且不更新。`QWEN36_HETERO_PADDED_BATCH=0` 保留按 signature 分组和跨组回滚。checkpoint 可独立恢复 heterogeneous rank/alpha/targets；HTTP 可在显式 `allow_aggregate_loss=true` 或 `X-Rustrain-Multi-LoRA-Capability: v1` capability 协商后有界合并兼容且 adapter ID 不相交的并发请求，响应返回 capability version、aggregate scalar、adapter loss 和真实 optimizer step；`expected_steps` 在原生 TP/EP/DP collective 前做全秩乐观并发校验，过期重试 fail closed；dynamic+MTP 已有单轴 DP、TP2 和 TP2xEP2 uneven-source oracle，默认仍保持单请求 dispatch |
| microbatch accumulation | 已实现子集 | non-final microbatch 只 backward，final microbatch 才 optimizer；FP32 accumulator 存储/聚合，autograd leaf backward 仍为 BF16 |
| replicated data parallel | 已实现 | logical-step 边界同步 replicated LoRA；EP expert 参数不走该 reduction |
| expert parallel | 已实现子集 | 默认 routed-output all-reduce；gated variable-split A2A 已验证 fixed-LoRA 和 native dynamic-LoRA data sharding；`QWEN36_EP_A2A_COMPACT_COUNTS=1` 在大 EP world 只将本 rank send row/receive column/validity flags 拷回 host，避免完整 `world×world` metadata D2H，TP2×EP2×DP2 world8 smoke 已通过；GPU-only variable-count dispatch、异步 overlap 和 DeepEP backend 未实现 |
| tensor parallel | attention + GDN + dense/expert MLP + vocabulary 子集 | full attention 与 GDN 使用 head-aligned ColumnParallel input projection 和 RowParallel output projection；GDN 的 flat QKV/conv 按 `[Q_local|K_local|V_local]` 重排。embedding/LM-head/CE 使用 vocabulary shard；routed/shared expert 使用 gate/up output shard、down input shard，并已在 TP2 x EP2 验证。CLI/server 均可通过共享 v5 rank shard 合并标准 PEFT |
| sequence parallel | 已实现受限子集 | `QWEN36_SEQUENCE_PARALLEL=1` 仅允许 TP2、dense text-only、projection-aware fixed/dynamic LoRA、CP/EP/DP/PP=1、MTP/aux loss=0；embedding scatter、column all-gather、row reduce-scatter、loss gather 及三租户 selected training 在 H20 两进程 NCCL smoke 中通过，latent-rank dynamic target fail-closed |
| pipeline parallel | 已实现受限 non-interleaved 子集 | ABI27 stage-local ownership 加上任意 `PP_SIZE>=2` 的固定 shape、单 chunk 1F1B 窗口。新增 opt-in dynamic-LoRA flag：共享 batch 行按注册 tenant 展开，反向恢复 per-tenant token numerator，并复用动态归一化/clip/事务 Adam finalizer；selected tenant 的 ordered request identity 现在在 TP/EP/DP/CP/PP 轴统一校验，并提供 per-tenant loss-report ABI；server 可通过 `QWEN36_PP_MICROBATCHES` 将 `[tenant, seq]` 行批次拆成真实多微批次 1F1B。H20 ABI1 PP2 fused-MLP dynamic smoke 已通过，包含 runtime mismatch、异构 rank 两 tenant/两 microbatch 和独立 loss/Adam 状态。dynamic window 在窗口期间固定 heterogeneous padding，后续本地 shape/mask/tick 错误用 selected-row contract-shape dummy 激活完成固定 P2P，并在 finish 做全 rank fail-closed consensus、禁止 optimizer commit。H20 PP2/PP3 `pp-train-smoke` 均通过固定 LoRA 的正常 loss/gradient/Adam parity、late shape+phase failure 和 divergent-target negative case。仍无 MTP/aux、CP、interleaved/chunked schedule 和 PP-aware reshard |
| context parallel | 已实现受限 GDN CP2 子集 | 仅 `CP_SIZE=2, TP=EP=DP=PP=1` 的 dense GDN；支持 fixed 和 grouped selected dynamic LoRA、序列/头置换、loss gather 与梯度 reduction，拒绝 MTP、aux loss、sequence chunk 和非 grouped dynamic sync。full attention、ring CP、CP4 及多轴组合仍 fail-closed |
| distributed checkpoint | 已实现子集 | v5 记录 rank order、完整五维坐标、TP/EP 多轴 placement、fused gate/up segments 及 fixed/dynamic slot identity；任意 multi-rank 拓扑使用 rank 目录。compact receipt 将每 rank preflight 限制为 `O(world_size)` 小 metadata + `O(local state)`；CLI/server 共享 same-topology restore 和标准 PEFT merge。server save/load 使用全 rank prepare/validate/commit-or-abort，load 通过 no-sync shadow context 保证失败不改 live state；fresh destination 以 `RENAME_NOREPLACE` 原子可见。v3/v4 及旧 digest-v5 保持只读兼容；可覆盖 generation/LATEST、断电级 durability、跨 topology reshard 和 PP/CP 未实现 |
| server tensor transport | 已实现 IPC 子集 | ABI22 parent/worker 共享 64-byte aligned binary slab 与 compact JSON descriptor；worker 不再反序列化三份 `Vec<i64>` 或在 Rust 热路径构造 `tch::Tensor`。普通 tensor HTTP routes 在 decode 前只允许一个 in-flight 请求，忙时快速返回 `429`；`/train_multi` 允许有界并发 admission，并仅对显式 `allow_aggregate_loss=true` 或 `X-Rustrain-Multi-LoRA-Capability: v1` 协商的兼容请求执行短窗口 GPU coalescing，响应声明 capability version 和 per-adapter result contract。带 `expected_steps` 的 optimistic 请求现在保持 request-local failure domain，不参与跨请求 coalescing；无版本 guard 的请求仍走 bounded batching。新增 `/train_multi_binary` 固定版本 little-endian wire，直接解析 int64 tensor sections 后写入 IPC slab，绕过 base64；普通控制面和 checkpoint transaction 使用 bounded FIFO 单消费者 scheduler（默认队列 32，可由 `RUSTRAIN_EP_DISPATCH_QUEUE_CAPACITY` 调整），每个 accepted job 在 `spawn_blocking` 上串行执行。native host-i64 已有 opt-in pinned staging，但仍缺稳定异步 H2D 和跨请求 GPU step overlap |

## 与 Megatron-LM 的关键差距

### 并行语义

Megatron 的通用 MLP 使用 ColumnParallel fc1、local gated activation 和 RowParallel fc2，并进一步提供 fused fc1/activation、sequence parallel、通信重叠和 sharded-state 支持。当前 Qwen native 已补上算法等价的 separate gate/up row shard 与 down column shard，但仍是两次独立 GEMM。

本地 Megatron-LM 的 `experimental/lite/megatron/lite/model/qwen3_5` 已包含直接的 Qwen3.5 实现和 LoRA adapter，而不是只能依赖外部 bridge：`primitive/modules/gqa.py` 使用 fused QKV ColumnParallel 和 O RowParallel；`gated_delta_net.py` 对输入/输出投影和 local heads 做 TP，并接入 sequence parallel、context parallel 与 FLA 高性能路径；`lora.py` 的 `LinearLoRA` 还能拆分 latent rank/output 并配套 gather、reduce-scatter 或 input-gradient all-reduce。没有发现显式的 Qwen3.6 model registration。当前 Qwen native 已补上真实但严格受限的 TP2 sequence-parallel dense 和 dense GDN CP2 路径，数学布局与 Megatron 的 gather/reduce-scatter 或 sequence/head exchange 方向一致；但 full attention 的 Q/K/V 仍是三次独立 GEMM，也没有通用 ring/full-attention CP 或 FLA chunk kernel。LoRA 的 replicated 一侧也比 Megatron Lite 更保守，MoE、PP/CP、多轴组合和通信重叠仍是主要差距。

PP 会把层分到不同 stage 并使用 1F1B 等调度；CP 会在序列和 attention state 上做跨 rank 通信。ABI26 已把 PP/CP communicator 纳入 `TrainingContext` 和进程级缓存，并通过五轴 size consensus 防止 rank 执行不同的 split 序列；ABI27 补上 stage ownership，当前实现进一步把固定 LoRA 窗口推广到任意 `PP_SIZE>=2` 的固定 shape、单 chunk non-interleaved 1F1B；随后新增了受限 dynamic-LoRA window，PP 版本仍明确拒绝 CP/MTP/aux/interleaving，GDN CP2 dynamic-LoRA 则走独立的 sequence gather/gradient all-reduce 路径。PP control collective 使用独立 communicator，并只在首个微批次建立全局 contract，避免把不同 stage 锁步而与 activation/gradient P2P 互锁；后续本地 shape/mask/tick 错误不直接 reset 单 rank，而是用固定 contract shape 的 dummy 激活完成同样的 P2P 计数，再在 finish 以 control all-reduce fail closed 并清理累积梯度。固定 LoRA 的 H20 PP2/PP3 `pp-train-smoke` 正常 loss、gradient、parameter 和 Adam parity，以及 late shape/phase failure 均通过；H20 ABI1 PP2 dynamic window 也已通过。CP2 x PP2 smoke 仍只验证正交 group 和全网 max 传播；新增的 sequence-parallel 只覆盖 TP2 dense 单轴，不能替代 CP ring attention 或任意五维组合。NCCL 文件 rendezvous 按 generation 隔离：launcher 使用共享 `RUSTRAIN_LAUNCH_OUTPUT_DIR`，直接多节点任务必须提供唯一 `RUSTRAIN_NCCL_RUN_ID` 和 `RUSTRAIN_NCCL_SYNC_DIR`。除严格受限的 dense GDN CP2 外，其余 CP model execution 仍拒绝；MTP/aux 和 interleaved/chunked PP 也仍在读取输入前 fail-closed。

当前 DP/EP 仍不是完整 Megatron 语义：DP 同步 replicated LoRA 梯度并按租户 token count 归一化，expert 参数留在 EP rank；sharded EP 使用 variable-split dispatch/inverse combine 和 fixed/dynamic LoRA data sharding。ABI19 已把所有 top-k assignment 合并为一次 dispatch，并把 local expert 计算合并为 grouped GEMM，但 count planning 仍可见于 host，且没有 fused permutation、异步 overlap 或 DeepEP backend。server 广播同一个 global batch descriptor，worker 按五维 topology 选择本 source rows：TP peers 保持相同，sharded EP 使用 `DP*EP` sources，replicated EP 只按 DP 分片。ABI22 将 parent/worker 数据从每 worker 全量 JSON vector 反序列化改为共享 binary slab；worker 直接借用 slab span 并调用 C++ host-i64 coarse ABI，Rust worker 热路径不再执行 `Tensor::from_slice`、`reshape` 或 `to_device`。普通控制面/检查点现在由 bounded FIFO 单消费者调度器承接，避免 Tokio handler 被阻塞式 IPC 占用；tensor 请求仍单 in-flight，避免多个 48 MiB body 同时驻留，因而它不是多租户 GPU batching。IPC timeout 会永久 poison channel，首次失败立即终止并回收 worker；pre-publish `InvalidInput` 不 poison channel。health 在 terminal failure 后返回 unavailable，partial launch 同样回收已启动 ranks。HTTP ingress 现在同时支持 binary slab wire 与 JSON fallback；native host-i64 有 opt-in pinned staging，coalescing deadline 也从 admission 起算以覆盖前一个 GPU step，但跨请求 H2D/compute overlap 仍未实现，因此仍不能视为最终的高吞吐服务传输。

H20 ABI1 `cp-attention-smoke` 进一步验证了受限 dense full-attention CP2 bridge：local Q 使用 CP rank 的 RoPE offset，normalized K/V 经可微 sequence all-gather 后以显式 global causal mask 做 SDPA，right-padding 跨 CP boundary 的 CP2 与 CP1 eval/loss 差为 `1.117e-3`，Adam `m/v` 差为 `5.595e-6/1.284e-9`；rank-local `QWEN36_CP_FULL_ATTENTION_KV_GATHER` mismatch 在 collective 前 fail-closed。Rust CLI/server 入口仅允许 CP2、TP=EP=DP=PP=1 且显式开启该 flag；这不是 ring attention，也未证明长序列吞吐。

### 优化器与恢复

固定 LoRA 的 Adam 状态可导出/导入，native context 的 logical step 与 checkpoint step 对齐；恢复时校验 canonical base-model path，CPU safetensors state 会复制到 shadow CUDA/FP32 allocation。dynamic adapter 的请求频率不同，每租户拥有独立 optimizer step、m/v 与 FP32 accumulator。server checkpoint load 只有在 fixed/dynamic LoRA、全部 m/v、各 optimizer clock 和 session metadata 都 hydrate 成功后才交换 context。heterogeneous selected v2 默认把不同 signature 的 adapter 安装到同一个 padded registry，一次 TrainOnly 后再执行一次 FinalizeOnly；低 rank 的真实 leaf 通过可微 `cat` 补零，inactive projection 使用同 geometry 的零张量，因此不会破坏 autograd 或错误更新 target hole。旧的 deterministic signature grouping 与 persistent shadow 跨组回滚仍可通过环境变量启用。若二次恢复（registry、梯度清理、rollback 或 restore）失败，native context 现在标记为 poisoned；后续动态请求先在 TP/EP/DP 做健康共识并把全 worker group 一致隔离，EP parent 收到 terminal result 后立即标记 unavailable 并回收全部 worker。H20 world4 `TP2 x EP2` 用 rank0-only 注入验证四 rank 均 fail-closed 且无 collective hang；正常 Adam 注入失败仍恢复为 healthy。同时仍缺少基于模型内容 fingerprint 的身份校验。

### 性能工程

ABI24 adds a packed LoRA gradient synchronization path: FP32 accumulators are
grouped by EP/DP/TP reduction mask and each populated bucket/axis uses one NCCL
all-reduce. `QWEN36_PACKED_LORA_SYNC=0` keeps the per-tensor grouped fallback.
This reduces collective launch count, but it still runs after full backward and
the token-count CPU fence; it is not Megatron-style backward bucket overlap or
reduce-scatter. H20 TP2 fixed-LoRA smoke and TP2 x DP2 oracle passed with both
settings; the synthetic TP2 benchmark was `77.970 ms` packed versus `78.024 ms`
fallback p50, within measurement noise.

ABI28 also packs the selected-v2 preflight booleans (health, request validity,
loss-report mode/capacity, and accumulation state) into one per-axis MIN
all-reduce. On H20 TP2 x EP2 with the synthetic
heterogeneous workload (`B=8`, `S=128`, eight active tenants), three interleaved
runs measured p50 old/packed pairs of `13.394/13.256`, `13.219/13.153`, and
`13.638/13.199 ms`; the median improved by about `1.5%`. An Nsight Systems
process-tree trace reduced `ncclDevKernel_AllReduce_Sum_u32_RING_LL` instances
from `1032` to `752`; u64 registry/hash collectives were unchanged. This is a
control-plane launch reduction, not communication/compute overlap or a matched
Megatron result.

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
may set `allow_aggregate_loss=true`, or advertise
`X-Rustrain-Multi-LoRA-Capability: v1`; the latter is a versioned opt-in for
per-adapter loss/step results and the explicit loss scope. Without either
signal, the legacy one-request dispatch and request-local loss are preserved.
Opt-in requests are grouped only when
session, sequence/source layout, and adapter IDs are compatible and disjoint.
The scheduler keeps up to `RUSTRAIN_MULTI_LORA_BATCH_MAX_OPEN_WINDOWS`
compatible layout buckets open (default 8, hard maximum 64), so interleaved
sequence/source layouts no longer seal otherwise mergeable windows early.
Admission also enforces `RUSTRAIN_MULTI_LORA_BATCH_MAX_RANK_WORK` over
`adapter_count * max_lora_rank` so heterogeneous rank padding cannot silently
turn a small request window into an oversized GEMM. The coalescer rebuilds
source-major rows into one native heterogeneous batch, is bounded by
request/adapter/rank-work/payload limits, and seals its window before
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
The bounded multi-bucket scheduler passed 53/53 server tests in both the local
ABI1 environment and on H20, including interleaved-layout retention, deadline,
capacity, binary ingress, and rank-work admission cases.

An opt-in asynchronous A2A metadata prototype was tested against the
compact-count path on H20. EP4 `H=4096, I=8192, S=128` regressed p50 from
`5.6632 ms` to `5.8951 ms`; EP8 repeated runs were effectively tied at p50
(`5.6856` vs `5.6524 ms`) while async p95 worsened to `7.1466` from `5.9853 ms`.
The uncommitted prototype was removed; `QWEN36_EP_A2A_ASYNC_METADATA` is not a
supported runtime flag.

The `/train_multi_binary` endpoint accepts version-1 `RLM1` requests with a
56-byte header, adapter IDs/optional expected steps, and three contiguous
little-endian int64 tensor sections. It validates exact `[batch, seq]` geometry,
alignment, counts, and trailing bytes before dispatching the same coalescer and
native ABI as JSON requests. This removes base64 expansion and JSON tensor
materialization from the ingress path; the IPC worker still receives the same
validated slab contract.

Variable-split A2A also has an opt-in compact count transfer. With
`QWEN36_EP_A2A_COMPACT_COUNTS=1` (enabled by default for communicator worlds
of at least eight), the host receives only `O(world)` counts instead of the
complete `world x world` matrix; the variable-count NCCL enqueue remains on the
host because the current prebuilt NCCL API still requires receiver counts before
`ncclRecv`. H20 TP2 x EP2 x DP2 world8 `tri-smoke` passed with the default path.
On the small TP2 x EP2 benchmark the compact path was not enabled by default;
forced compact A/B was `22.707 ms` p50 versus `22.385 ms` legacy, so EP2/EP4
keep the legacy path.

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

H20 ABI1 的 TP2 full-attention oracle 已在 `QWEN36_FUSED_QKV` 关闭和开启两种模式下通过：fixed eval/loss 差为 `4.40e-4`，selected dynamic loss 差为 `6.82e-5`，各 Q/K/V/O 参数差不超过 `3.05e-5`。这证明 fused 路径与未融合路径等价，不代表已完成完整模型性能胜出。

full-attention dynamic LoRA 另新增 `QWEN36_FUSED_LORA_QKV_A=1` opt-in：当 Q/K/V 的 adapter layout、rank、batch、input width、device 都兼容时，将三个 A stack 沿 rank 维拼接，只执行一次 `A@x` batched matmul，再切分 latent activation 并执行各自的 B matmul；异构请求自动回退旧路径。latent-rank TP 只复制一次输入，三个 output delta 仍分别执行既有 all-reduce；flag 纳入 distributed runtime hash，rank-local 不一致会 fail-closed。H20 local smoke 和 TP2 fixed/dynamic oracle 均通过。TP2 x EP2 synthetic A/B 在 `B=2,S=128,H=1024,rank=16,2 active tenants` 时 p50 为 legacy/fused `9.354/9.291 ms`（约 `0.7%`）；`B=8,S=512,H=2048,8 active tenants` 的重复结果在约 `20.2-21.4 ms` 间波动，未形成稳定收益。因此默认关闭，当前证据只支持正确、可回滚的实验路径，不支持“高性能已完成”的结论。

MoE shared expert 另新增 `QWEN36_MOE_SHARED_OVERLAP=1` opt-in：每个 `TrainingContext` 懒创建一个稳定的 nonblocking CUDA stream，producer/completion event 将 shared dense expert 与 routed sort/A2A/grouped GEMM 排队到不同 stream，residual combine 前才等待；runtime hash 包含该 flag，context free 会销毁 stream，默认路径不改变。H20 TP2 x EP2 fixed/dynamic smoke 及 rollback/poison 负测通过。交错 H20 synthetic A/B 中，`B=2,S=128,H=1024` 没有稳定收益；`B=8,S=512,H=2048` p50 legacy/overlap 为 `53.195/53.576 ms`，allocator reserved 为 `10.004/12.182 GiB`。由于 overlap 未带来稳定 step-time 改善且增加 workspace/cache 压力，保持默认关闭；这不是 DeepEP/Megatron overlap 等价证明。

本轮新增 `QWEN36_GDN_FUSED_CP_EXCHANGE=1` opt-in：在 CP2 dense GDN 中把 peer-major `[Q|K|V|Z|A|B]` payload 的两次 sequence-to-head exchange 合为一次，autograd backward 也合为一次 inverse exchange；总 payload bytes 不变，配置进入 runtime hash，rank-local flag 不一致会在 P2P 前 fail-closed。H20 `B=2,S=512,H=2048,L=3,rank=8,warmup=5,iters=30` 的 rank0 p50/p90 为 legacy/fused `103.811/105.456 ms` 与 `104.011/107.565 ms`；`B=8` 为 `227.745/231.473 ms` 与 `227.952/236.727 ms`。B=2 fused allocator peak `0.904172/0.996094 GiB`（allocated/reserved），legacy `0.904271/0.990234 GiB`；B=8 fused `2.531722/2.777344 GiB`，legacy `2.531722/2.783203 GiB`。最小 Nsight trace（1 layer、warmup 1、iters 2）确认 `ncclDevKernel_SendRecv` 从 `36` 降至 `24`（每步每层 `6 -> 4`），但端到端 p50 没有稳定改善，故默认关闭；这是 native synthetic CP2 A/B，不是 Megatron/FLA 对照，也没有证明通信计算 overlap。

当前粗粒度 C++ FFI、packed/grouped MoE、GDN head TP、vocabulary TP 和 activation checkpoint/offload 是有效优化；full attention 的 activation-level LoRA 路径新增 `QWEN36_FUSED_QKV=1` opt-in，可将 frozen Q/K/V base projection 合成一次 GEMM，但由于会复制冻结权重，默认关闭，必须在目标 GPU 做显存/延迟 A/B。TP collective 仍同步执行。heterogeneous selected v2 的 padded 路径消除了按 signature 重复完整 trainer 和组间 GPU success fence；在 H20 TP2 x EP2 的四租户 synthetic workload 上，p50 从 grouped 的 `17.750 ms` 降到 `13.841 ms`，unique-token throughput 从 `57.69k/s` 提升到 `73.98k/s`，但 allocator peak 从约 `0.240 GiB` 增至 `0.354 GiB`，且 10 次样本的 p95 为 `20.152 ms`，未优于 grouped 的 `19.441 ms`。这证明 unified padded batch 对当前 native 路径有效；HTTP 仅在显式 `allow_aggregate_loss=true` 时合并兼容请求，且 aggregate loss 会被明确标记，不是 Megatron 对比。尚无 Megatron/Transformer Engine 级别的完整模型端到端数据：没有同一 GPU、序列长度、microbatch、精度和通信配置下的 tokens/s、显存、扩展效率对照，也没有 FP8/FP4 参数与 fused attention/DeepEP 的 Qwen 路径。

新增 `QWEN36_FUSED_MLP_FC1=1` opt-in，把冻结 dense/shared gate/up projection 合成一次 FC1 GEMM，再保留每个租户 LoRA delta 的独立累加；该配置也纳入 PP runtime identity。H20 TP2 x EP2 synthetic dynamic-MoE benchmark（B=2、8 tenants、2 active）中，p50 `11.133 -> 10.672 ms`（约 `4.1%`），但 p95 `12.231 -> 13.777 ms`，allocator peak 约增加 `3.6 MiB/GPU`；B=8、8 active tenants 中 p50 `14.034 -> 13.930 ms`（约 `0.7%`），p95 `15.488 -> 15.706 ms`，allocator peak 约增加 `2.4 MiB/GPU`。因此该路径默认关闭，只作为目标硬件上的可回滚 A/B 选项；native smoke 和 PP2 dynamic isolation 均已通过。

ABI15 synthetic GDN TP benchmark 使用 3 层、S=512、H=2048、16 K heads、32 V heads 和 LoRA rank 8。在 B=2 时 single/TP2 p50 为 `81.10/约 76.06 ms`，只有约 `1.07x`；在 B=8 时为 `303.25/约 158.04 ms`，达到约 `1.92x`，每卡 observed resident 从 `4.59 GiB` 降到 `3.54 GiB`。`nsys` 的 B=2 TP2 trace 中 fused delta-rule backward 占 GPU kernel time `80.3%`、forward 占 `7.4%`，NCCL all-reduce 合计低于 `0.4%`。小 batch 下 single 和 TP2 都只需相近的 persistent-block wave 数，因此不能期待 head TP 自动加速；dynamic multi-LoRA batching 提高 BH 后才接近线性 scaling。这是 synthetic native 结果，不是完整模型或 matched Megatron 对比。

本次 native benchmark 没有证明 gated A2A 的端到端 step-time 优势：在该小型 workload 上 sharded A2A 的中位 step 反而比 legacy 高约 `23%`。它没有实现 DeepEP 的 fused permutation、GPU-only split planning 或通信计算 overlap，也没有覆盖 H=`2048`/E=`256` 的完整 Qwen3.6 workload。legacy 模式复制输入 batch，因此必须同时报告唯一样本吞吐，不能只看所有 rank 的 processed tokens/s。

上述结果是 ABI0 的纯 EP2 小模型历史基线。ABI19 在 TP2 x EP2、相同 source-sharded 语义下做了 matched routing-slot/packed 对照，并在每步训练后沿 EP、expert-DP、TP 依次做 MAX，从而让 4 个 rank 使用相同的全局最慢 rank 样本：p50 `9.007 -> 6.499 ms`，unique-token throughput `56.84k -> 78.78k/s`。Nsight Systems process-tree trace 同时记录到 CUDA kernel launch `6812 -> 5168`、NCCL SendRecv `96 -> 48` 和 AllGather `24 -> 12`；该 trace 采集于 benchmark-only world-max metrics collective 加入之前，因此不包含这个计时区间外的聚合。它证明 packed dispatcher 对当前 native 路径有效，但仍不等于 DeepEP 或 Megatron 的重叠能力。新增 TP2 dense sequence-parallel H20 smoke 只验证 all-gather/reduce-scatter/loss-gather 与固定 LoRA 更新，不提供吞吐结论。

目标 H20 的 ABI1 环境是 CPython 3.12、PyTorch 2.12.1+cu130（ABI=1）、cuDNN 9.20、NCCL 2.29.7、Triton 3.7.1 和 NVSHMEM 3.4.5；机器为 8 张 H20 SM90。Megatron Core、Transformer Engine、FLA、DeepEP、flash-attn 和 Apex 均未安装。PyTorch 自带 SDPA flash backend、`_scaled_mm`、cuBLAS/cuDNN/NCCL，因此仍是当前直接满足约束的生产计算栈。

依赖审计的结论需要区分“预构建产物能运行”和“能无 Python 接入 rustrain”：PrimeIntellect `prime-rl v0.5.0` 提供了可运行的 DeepEP wheel `deep_ep-1.2.1+73b6ea4.cu13-cp312-cp312-linux_x86_64.whl`，SHA256 为 `80369bcbf664d8931950f529e71b549a0e0808c6953b5c8c3dccc91a75770f36`。在 H20 Torch2.12.1/NVSHMEM3.4.5 上，8-GPU `get_dispatch_layout -> dispatch -> combine` 误差为 `0.0`，SM90 检测为真；top1 `4096x7168` BF16 roundtrip p50 约 `0.779 ms`。但 wheel 只有 `PyInit_deep_ep_cpp` 动态业务入口，DeepEP 初始化/IPC 使用 hidden C++ symbols 和 `pybind11::bytearray`，没有稳定 C ABI 或头文件；直接集成需要嵌 CPython/pybind，或者供应方提供预构建 C-ABI shim。自己重编 DeepEP 或 shim 会违背当前“依赖必须预构建、禁止 Python JIT/自构建依赖”的约束，因此本轮只记录为候选验证通过，不接入生产路径。

FlashAttention 另有社区预构建 `flash_attn-2.8.3+cu130torch2.12-cp312` wheel，在当前 H20 可 import 并完成 BF16 GQA forward/backward，forward 约 `0.280 ms`，与 PyTorch SDPA 约 `0.286 ms` 基本相同；由于供应链和 pybind ABI 风险，不能证明值得替换当前 SDPA。TE 2.16.1 的 cp312/cu13/ABI1 wheel 针对 NVIDIA Torch 26.05 的另一 commit；FLA 的 GDN 使用 `triton.jit`/autotune；FA4 依赖 CuTe compile；这些仍不满足当前生产约束。Megatron Core、TE、FLA、DeepEP、flash-attn 和 Apex 仍未安装在目标 Pod。

本地 `/root/code/Megatron-LM` 为 0.19.0（commit `ec2aff43`）。其声明的 GDN/MoE 高性能路径依赖 TE、FLA、FlashAttention/DeepEP，其中锁定环境还包含 git/source build 或 Triton JIT，不能拿到当前 H20 环境直接做合规的 matched run。当前 H20 原语 microbenchmark 只能说明硬件基础可用：SDPA GQA `B=2,Hq=32,Hkv=8,S=1024,D=128` forward 约 `0.286 ms`、forward+backward 约 `1.099 ms`；BF16/FP8 GEMM 约 `132.7/270.6 TFLOP/s`；NCCL A2A 约 `206 GB/s/rank` median。它们不是完整模型 benchmark。因此当前不能诚实地产出 matched Megatron-LoRA tokens/s、显存和扩展效率对照，也没有通过 Python JIT 或自构建依赖绕过限制。

## 验证边界

已运行的验证包括 Rust 编译检查、core 单测、Qwen3.6 配置/集成测试，以及 H20 ABI11 的单卡、TP2、EP2 和 DP2 native smoke。DP2 的 weighted m/grouped/v/Adam BF16 delta oracle 分别达到 `2.43e-8`、`2.27e-8`、`7.33e-8` 和 `0`；legacy EP2 与 full-expert reference 的 loss、LoRA、Adam state 和标准 Adam 首步 oracle 在两 rank 均为零差异。Replicated A2A 与 fixed-LoRA sharded A2A 也均通过两 rank full-expert reference；sharded token counts `[1,3]` 的加权 loss 与 global reference 相差约 `9.6e-7`，m/v 最大差 `1.22e-5` / `3.92e-9`，Adam oracle 差为 `0`。H20 ABI1 dynamic sharded full-reference smoke 在两 rank 返回 `0`：dynamic grouped-expert 参数最大差 `1.53e-5`，m/v 最大差 `4.88e-5` / `5.75e-8`。ABI13 dense base-MLP TP2 smoke 对 gate/up/down 使用半尺寸本地权重，eval/train loss 与完整权重参考相差 `1.84e-5`。

ABI14 的 4Q/2KV-head GQA full-attention TP2 smoke 覆盖 Q/K/V/O fixed 和 selected dynamic LoRA。fixed eval/loss 与完整参考最大差 `4.40e-4`；Q/K/V-B 与 O-A 梯度最大差分别为 `1.83e-4`、`1.14e-5`、`3.05e-4` 和 `1.83e-4`，FP32 Adam m/v 最大差 `1.53e-5` / `6.63e-9`，本地标准 Adam 公式误差小于 `3.73e-9`。selected dynamic loss 最大差 `6.82e-5`，Q/K/V/O 参数最大差 `3.05e-5`。

H20 ABI1 PP2 native smoke 进一步覆盖两 tenant、两 microbatch、每 tenant 不同 token 权重的 dynamic 1F1B。两 rank 对乱序 selected-adapter 请求一致拒绝；正确请求的 report loss 在两 rank 一致，aggregate loss 与按 tenant token 数加权的 report 相差小于 `1e-5`。选中 tenant 的 q_proj-B 更新而未选 tenant 的参数、Adam m/v 与 step 保持逐位不变；PP2 fixed-LoRA parity 仍通过。追加的异构 rank (2/3) selected 请求通过 `qwen36_add_lora_v2` 注册并完成同一隔离检查。第二个 microbatch 注入 malformed target 后，PP 继续消费 contract-shaped dummy work，fallback batch 保持 selected tenant 数量而不是 registry 总数，最终两 rank 在 finish 一致失败且无 P2P count mismatch。该测试仍限于单 chunk、固定 shape 的 1F1B，不代表 interleaved PP 或 CP model execution。

ABI15 的两层 GDN base-TP smoke 覆盖复合 QKV/conv head shard、Z/A/B/A_log/dt_bias、replicated norm、out-proj input columns，以及五种 GDN projection 的 fixed/selected dynamic LoRA。fixed loss 最大差 `1.30e-4`，rank-local factor 梯度最大差 `2.44e-4`，FP32 Adam m/v 最大差 `3.43e-6` / `2.49e-10`；dynamic loss 最大差 `1.16e-3`，m/v 最大差 `6.10e-6` / `1.22e-9`，标准 Adam 公式误差均为 `0`。latent-rank-only GDN TP2 回归也通过，loss 差 `5.46e-5`。BF16 参数直接对照最多跨约两个量化 bin，因此验收以 FP32 梯度、m/v 和本地 Adam 公式为主。ABI21 world8 TP2 x EP2 x expert-DP2 shadow resume 在 no-sync attach 前故意扰动 DP 非零 rank 的 fixed LoRA，attach 后保持 bitwise 不变；hydrate 后 fixed/dynamic LoRA、Adam m/v 和 tenant clock 精确一致，第二步的参数/m/v 与连续训练零差，step 为 `2`。同一 world8 fixture 已分别覆盖 source-sharded 与 replicated-source A2A；replicated-source 的 fixed 参数/m 最大差 `1.96e-3` / `2.58e-6`，dynamic 参数/m 最大差 `1.99e-3` / `4.16e-5`，两者 Adam 公式误差均为 `0`。ABI22 的单卡 native smoke 验证 borrowed host-i64 eval 与 tensor ABI loss 零差、fixed train Adam oracle 零差，以及 selected dynamic multi-LoRA 仅更新已选 adapter；同一 ABI22 world8 TP2 x EP2 x expert-DP2 `tri-smoke` 也通过 checkpoint resume、selected isolation 与 Adam oracle。heterogeneous restore smoke 在已有 rank-8 expert adapters 后以 no-sync hydration 注册 rank-9 `q_proj` adapter，验证 checkpoint registry 可保留独立 signature。ABI23 进一步覆盖 live heterogeneous registration、tensor/borrowed host-i64 selected v2 dispatch，以及第二组注入失败后参数、Adam m/v 和 tenant clock 的精确回滚。TP2 x EP2 x expert-DP2 world8 使用 rank-3 `q_proj` 与 rank-4 multi-module tenants 验证所有 rank 的成功更新和失败回滚，未出现 collective 顺序分叉。新的 padded 路径在单卡和 TP2 x EP2 smoke 中覆盖 rank-8/rank-9、dense/expert target holes、独立 clock、零 token tenant 与 rollback；显式 grouped fallback 的 TP2 x EP2 smoke 也通过。没有完成完整大模型长时间训练、真实跨节点通信、CP model execution 或与 Megatron-LM 的同条件 benchmark；PP 目前只有固定 LoRA、固定 shape、单 chunk 的 PP2/PP3 smoke。因此“正确”应理解为已覆盖且有直接 oracle 的子集，而不是所有并行配置。

本轮新增 H20 ABI1 CP2 GDN full-reference smoke：`QWEN36_GDN_FUSED_CP_EXCHANGE=0/1` 两种模式均通过 fixed eval/train 与 dynamic selected training。fixed eval/loss 差为 `5.1117e-4`，五种 GDN LoRA projection 的 A/B 参数差为 `1.9531e-3/1.9989e-3`，Adam m/v 差为 `3.0518e-6/1.4198e-10`；dynamic aggregate/per-tenant loss 差均为 `0`，LoRA 参数差 `9.8801e-4`，Adam m/v 差 `1.5259e-6/1.5669e-10`。两个 selected tenant 正常更新，第三个未选 tenant 的参数、m/v 与 optimizer step 均逐位不变；rank-local fused flag mismatch 也在 P2P 前一致 fail-closed。实现将 CP local hidden 先 gather 后计算 CE，再由 autograd/手动 checkpoint backward 切回 local gradient，并在 dynamic grouped gradient sync 中加入 CP all-reduce；非 grouped dynamic CP 继续 fail-closed。

本轮新增 H20 ABI1 `mtp-dynamic-smoke` 和 `mtp-dp-smoke`：注册三个 dense LoRA tenant，选择两个不同主/MTP token-count 的租户，安装一层真实 frozen MTP prediction layer；单卡 dynamic per-tenant report 与单租户 oracle 的 loss 和参数差均为 `0`，DP2 complementary-source rows 对两个 singleton oracle 的 loss、参数、Adam m/v 差均为 `0`，第三个未选择租户参数/m/v/step 逐位不变。DP2 MTP objective 的 loss effect 为 `1.0821`、parameter effect 为 `2.0142e-3`，证明 MTP 分支实际参与训练。实现一次性 all-reduce `[main_counts, mtp_counts]`（DP>1 时）并将 MTP hidden gradient/report numerator 转到主 loss denominator；fixed micro-step 使用同一比例，seq<3 和 `TP/CP/EP/PP>1` 组合 fail-closed。

本轮再新增 H20 ABI1 `mtp-tp-smoke`：TP2 两 rank 对主层和 dense MTP prediction layer 同时使用 attention/MLP 列/行并行，embedding/LM-head 保持 replicated，两个 selected dynamic tenant 与一个未选 tenant 对照单卡 full-weight oracle。两 rank 的 `loss_diff` 最大 `3.53e-3`、参数差最大 `1.87e-9`、Adam m/v 差最大 `2.14e-5/3.23e-9`，未选状态逐位不变；开启 `QWEN36_FUSED_QKV=1` 与 `QWEN36_FUSED_MLP_FC1=1` 后仍通过。后续真实 MoE prediction-layer TP2 MTP oracle 的 `loss_diff=1.50e-3`、参数差 `3.05e-5`；vocabulary-parallel MTP 的 `loss_diff=3.62e-3`、参数差 `2.01e-3`，未选状态均逐位不变。MTP token counts 在 TP 只做 min/max 一致性检查，不做求和，避免重复放大 hidden gradient。TP2xEP2 uneven source-token-count MTP 也已通过（main/MTP tokens `7/5`，loss diff `4.38e-4`，参数差 `2.01e-3`）。这些仍是受限 oracle，不覆盖 PP/CP MTP 或完整模型长跑。

本轮对 heterogeneous selected-v2 的 registry 管理做了小范围优化：恢复阶段按保存的 canonical index 原位放回 selected adapter，并维护持久排序的 `adapter_id -> canonical slot` 索引，供 distributed validator、selected-v2、expected-step 和 selected-eval 使用；临时 selected/chunk registry detach 时显式使索引失效，canonical vector 恢复后再启用。H20 TP2 x EP2 synthetic dynamic-MoE（`B=8,S=128,H=1024,I=2048`、rank 16、q/expert targets、50 iterations）中，8 registered/8 active tenants 的 p50/p95 为 `13.859/17.561 ms`，1024 registered/8 active 为 `14.390/19.579 ms`，p50 增加约 `3.8%`；allocator reserved 从约 `1.04` 增至 `9.46 GiB/GPU`。local smoke 和 TP2 x EP2 distributed smoke 均通过。该结果只证明 registry scaling 的查找开销受控，不等于完整模型端到端收益；大注册表的 adapter 状态显存仍是容量约束。

这里“尚未完成 CP model execution”指任意五维组合、ring attention 和长序列 CP；当前新增的 full-attention CP2 仅是 opt-in KV all-gather correctness bridge，和 GDN CP2 一样不应外推为完整 Megatron CP。

## 继续达到 Megatron 级别所需的最小工作包

1. 为已实现的 opt-in server batch coalescing 增加 HTTP response capability/version negotiation，使支持 per-adapter loss 的客户端可取消显式 aggregate opt-in；补齐 padding rank inflation 容量模型和长尾 benchmark。
2. 为现有任意 PP size 的受限窗口增加 interleaved/chunked scheduler、dynamic LoRA、PP-aware checkpoint/loader，并为 CP 实现 ring attention/state exchange。
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
