---
type: ChangeSpec
title: kernel-first rustrain — operator ABI, declarative registry, plan-driven execution
description: Rebuild rustrain around orchestration: a plan IR with parallel layout and communication scheduling, an operator registry resolved from a recipe, and kernels as pluggable plugins.
tags: [rustrain, architecture, orchestration, kernel-first, spec]
timestamp: 2026-08-03T00:00:00Z
---

# kernel-first rustrain

> 状态：`ready-for-agent`
> 基线：`main` @ 历史压缩后的单一基线 commit（见上下文层）
> 本文件是唯一 checkpoint。执行者不需要对话历史，只读本文件即可接手。

---

## 1. 目标

### 定位

rustrain 是**训练框架**，不是 kernel 库。差异化价值在并行策略、通信调度、显存与精度编排——
不在手写 GEMM。计算交给 cuBLAS / CUTLASS / ATen / FlashAttention / Tilelang。

### 问题

重启前，"哪个算子实现会运行"由五处互不相通的机制决定（编译进的 crate、`dlopen` 符号探测、
`LD_LIBRARY_PATH`、进程环境变量、Rust 侧手写布尔），**每一处都以静默降级收场**。同时并行切分
被写死成训练循环里的手写通信调用，导致换并行策略要改代码。结果是：一次 run 结束后无法回答
"跑了哪个 kernel、什么精度、通信插在哪"，也无法在不改代码的前提下改变这两者。

### 目标结果

框架不认识任何模型，也不认识任何具体 kernel：

1. **解析** —— 把模型表达成算子图（plan），并行切分是图上的数据。
2. **解析实现** —— 按 recipe 为每个算子选出具体实现。
3. **驱动** —— 编译计划、插入并调度通信、执行。

模型是数据。算子是插件。精度与并行是配置。

### 一句话验收

改一个 TOML 字段就能切换某个算子的前向/反向实现或精度方案，改一处并行配置就能改变切分与
随之插入的通信，**都不重编译框架**；计划摘要与插件版本写进 run manifest，可复现。

---

## 2. 契约层（durable）

### 2.1 分层与依赖方向

```
rustrain-cli         train | ops | plan
rustrain-runtime     执行器 / 显存池 / 流与 collective 调度
rustrain-plan        Plan IR / 切分传播 / 校验 / 编译 / digest
rustrain-parallel    进程组 / rank 布局 / 切分规格 / 通信规划
rustrain-ops         算子描述 / 注册表 / requires / numerics / expansion / recipe
rustrain-abi         插件 ABI v1 / 装载
──────────────────────────────────────────────────────────────
plugins (.so)        aten | reference | 第三方高性能 kernel
```

**不变式 I-1：`rustrain-{abi,ops,parallel,plan,runtime}` 的依赖闭包中不得出现 `tch`、`libtorch`、`cuda`。**

理由：这是"换 kernel 不重编框架"的全部依据，也是核心能在无 GPU/无 torch 的机器上跑完整测试的前提。

**不变式 I-2：算子实现只能通过 ABI 注册，不能被框架静态引用。**
`rustrain-kernels` 是*一个插件*，不是框架的一部分。

**不变式 I-3：并行切分是 plan 里的数据，不是训练循环里的手写通信。**
`ParallelLayout` 承载切分规格；布局不匹配处的 `all_reduce` / `all_gather` / `reduce_scatter`
由传播 pass 插入。手写通信调用只允许出现在 `rustrain-parallel` 的通信规划内部。

### 2.2 ABI v1

C ABI 是唯一跨编译单元的契约。手写头文件 `crates/rustrain-abi/include/rustrain_op.h`，
Rust 侧 `#[repr(C)]` 镜像并做尺寸与偏移断言。

关键结构：`rs_tensor`（POD 张量描述，`reserved[]` 存后端私有指针）、`rs_op_id`、`rs_numerics`、
`rs_requires`、`rs_attrs`、`rs_mem_req`、`rs_expansion`、`rs_collective`、`rs_services`、`rs_ctx`、
`rs_op_desc`、`rs_plugin`。

**契约 C-1（单一入口）**：每个插件 `.so` 只导出 `rustrain_plugin_v1`。框架不按名字找函数，
只问插件"你提供什么"。

**契约 C-2（版本协商）**：`abi_version != RUSTRAIN_ABI_VERSION` 时拒绝装载并报错，不做兼容猜测。

**契约 C-3（服务注入）**：插件不自己 `malloc` 显存、不自己建 stream、不自己建 NCCL communicator，
全部通过 `rs_services` 向框架申请。这让框架能统一做显存池、流调度与 collective 排序。

**契约 C-4（POD 边界）**：跨 ABI 传递的都是 POD。`at::Tensor*` 只能藏在 `rs_tensor.reserved[]`。

### 2.3 算子注册表

注册表是全系统关于"什么算子可用"的**唯一真值来源**。

**契约 R-1（解析失败必须报错）**：plan 中出现无法解析的算子时构建失败，并列出该算子的全部候选
及每个候选被拒绝的具体原因（dtype 不符 / SM 不足 / world_size 不满足 / 未注册）。
**禁止任何静默 fallback。**

**契约 R-2（回退必须显式声明）**：只有写进 `fallback` 列表的候选才允许被降级选用，
且降级事实必须记入 plan digest 与日志。

**契约 R-3（精度是数据）**：量化格式、粒度、block 尺寸、scale dtype、scale 模式、amax 历史长度
全部由 `rs_numerics` + recipe 表达。任何 kernel 不得硬编码 block 尺寸或从张量形状反推量化方案。

**契约 R-4（融合必须声明展开）**：任何复合/融合算子必须提供 `expansion`。展开是三重用：
一致性校验的等价基准、fused 与 primitive 的 A/B 依据、plan 可读性。

### 2.4 原语集

固定词表。新增原语是一次框架演进（需 review），新增**实现**不需要。

| 类别 | 算子 |
|---|---|
| 元数据（零计算，planner 级） | `view` `reshape` `transpose` `narrow` `cat` `broadcast` |
| L0 计算 | `matmul` `linear` `bmm` `elementwise_unary` `elementwise_binary` `compare` `reduce` `softmax` `rmsnorm` `layernorm` `rope` |
| 量化 | `quantize` `dequantize` `amax_update` |
| 数据搬运 | `embedding` `gather` `scatter` |
| 通信 | `all_reduce` `all_gather` `reduce_scatter` `send_recv` |
| 复合（必须声明 expansion） | `sdpa` `flash_attn` `topk_router` `expert_dispatch` `expert_combine` `cross_entropy` `adamw` |
| 模型块（必须声明 expansion） | `mlp_swiglu` `moe_layer` `transformer_layer` `dsa_attention` `gated_delta_rule` |

**契约 P-1**：复合算子的 `expansion` 深度不得超过 2 层（块 → 原语）。

### 2.5 并行与切分

**进程组拓扑**：`world_size` + `ParallelConfig { tp, ep, cp, dp, pp }` 决定 rank 到各组的映射。
`rustrain-parallel` 提供 `RankLayout`：给定 rank 与拓扑，解析出该 rank 在各并行维上的坐标，
以及各组的成员集合。

**切分规格 `ParallelLayout`**：

| 变体 | 含义 |
|---|---|
| `Replicate` | 每个 rank 持有完整副本 |
| `Shard(dim)` | 沿 `dim` 均分 |
| `Partial(Sum)` | 每 rank 持有一个部分和，需要 all-reduce 才能得到完整值 |
| `ExpertShard` | 按专家维度切分（EP） |
| `SequenceShard` | 沿序列维切分（CP） |

**契约 S-1（切分传播）**：plan 编译期执行一次布局传播。规则由算子的 `shard_rule` 声明
（例如 `linear` 的权重按输出维切分时输出为 `Partial(Sum)`，需 all-reduce）。传播后仍存在
布局冲突的边，编译器必须插入显式转换算子；无法插入时报错。

**契约 S-2（通信是显式的）**：传播插入的通信在 plan 中是可见节点（`all_reduce` 等），
不在 kernel 内部隐式发生。算子若自带通信，必须通过 `rs_collective` 声明，供调度器排序。

**契约 S-3（overlap 可声明）**：通信节点可标记 `side_stream`，调度器负责生成
"计算-通信 overlap" 的执行顺序，并在需要的点插入同步。默认不 overlap，需显式开启。

### 2.6 Plan IR

```rust
pub struct Plan {
    pub meta:  PlanMeta,          // model id, recipe digest, parallel config, seed, phase
    pub slots: Vec<Slot>,         // 命名张量槽：shape/dtype/layout/lifetime
    pub nodes: Vec<PlanNode>,
}

pub struct PlanNode {
    pub op:        OpRef,
    pub inputs:    Vec<SlotId>,
    pub outputs:   Vec<SlotId>,
    pub attrs:     Attrs,
    pub phase:     Phase,               // Forward | Backward | Update
    pub precision: Option<PrecisionOverride>,
    pub checkpoint: CheckpointPolicy,   // None | Recompute | Offload
    pub stream:    StreamPolicy,        // Default | Side
    pub source:    Trace,               // "layers.17.mlp"
}
```

**契约 PL-1（先校验后运行）**：`compile` 之前必须通过五道校验，任一失败即拒绝：
1. **shape/dtype 推导** —— 逐节点 `infer`，类型与形状必须收敛。
2. **能力匹配** —— 选中 variant 的 `requires` 必须被当前环境满足。
3. **精度相容** —— 相邻节点 numerics 必须相容；不相容需显式插入 `quantize`/`dequantize`。
4. **切分相容** —— 每个节点的输入布局必须满足其 `shard_rule` 的前提。
5. **确定性** —— 声明为非确定性的算子若出现在开启确定性开关的 plan 中，拒绝。

**契约 PL-2（compile 一次，扁平执行）**：`compile` 产出扁平调用序列 + 已解析函数指针 +
预分配 slot 表 + 通信调度表。step 热路径上不做字符串查找、不做 `HashMap` 查询。

**契约 PL-3（digest 覆盖全部决策）**：digest = 拓扑 + 每节点 (op, variant, numerics, layout) +
插件名/版本 + 并行配置。**recipe 通过它产生的决策进入 digest，而不是以原文进入**——
两条解析出相同算子/变体/精度的 recipe 必须得到相同 digest，否则一次无关紧要的格式调整就会让两次
完全相同的 run 看起来不同。相同 digest 必然产生相同执行序列。

### 2.7 Recipe（配置即控制面）

```toml
[kernel]
default = "aten"
strict  = true

[kernel.ops.rmsnorm]
forward  = "cuda.fused_bf16"
backward = "cuda.fused_bf16"

[kernel.ops.mlp_swiglu]
forward         = "cuda.fp8_block128"
backward        = "autodiff"
check_expansion = true

[kernel.precision]
compute        = "bf16"
accumulate     = "fp32"
master_weights = "fp32"
grad           = "bf16"
weights        = "fp8_e4m3"
quant_scheme   = "per_block"
block          = [128, 128]
scale_mode     = "delayed"
amax_history   = 8

[kernel.parallel]
tensor   = 8
expert   = 1
context  = 1
data     = 1
overlap_collectives = true
```

**解析优先级**：节点级覆盖 → 算子级 recipe → 相位级 recipe → `[kernel].default` →
显式 `fallback` → **报错**。

**契约 CF-1**：前向与反向精度独立可选。
**契约 CF-2**：算子路径上不得读取进程环境变量。配置来自文件。

### 2.8 一致性门禁

`rustrain ops check` 是让"配置化换 kernel"安全成立的唯一机制。对每个
`(op, variant, shape, dtype, precision)` 组合执行四项检查：

| 检查 | 内容 | 判据 |
|---|---|---|
| numeric | variant vs reference | 相对误差 ≤ 算子声明容差 |
| expansion | fused variant vs 其 `expansion` 组合 | 同上 |
| gradient | 解析梯度 vs 有限差分/自动微分参考 | 相对误差 ≤ 梯度容差 |
| determinism | 同输入两次 | 逐位相同（除非显式声明非确定性并给出理由） |

**契约 K-1**：任何 variant 未经 `ops check` 通过，不得被 plan 选中。

### 2.9 run manifest 与可复现性

每次 run 在输出目录写入 `manifest.json`：plan digest、recipe 原文、每个选中 variant 的
`(plugin_name, plugin_version, op_id)`、并行配置、随机种子。

**契约 M-1**：给定同一个 manifest，`rustrain plan digest --manifest <path>` 必须复现相同 digest。


### 2.10 显存策略

训练时的绑定约束是**激活寿命**，不是 KV cache。KV cache 只在三处重要：rollout/生成（推理缓存，
需分页与淘汰）、有状态注意力（GLM-5 IndexShare / Qwen3.6 gated delta rule —— 序列内持久且带梯度）、
长序列分块或 ring attention（当前 chunk 的 K/V 必须实体化）。两者共用分配器，策略完全不同。

要管的是五类：权重与主权重、梯度、优化器状态、**激活**、workspace 与通信缓冲。

**契约 MEM-1（策略是声明的）**：每个节点的激活策略来自 recipe，优先级与算子选择一致
（node 级 → 算子级 → 默认）。任何 kernel 不得自行决定是否 offload / recompute。

**契约 MEM-2（峰值在编译期计算）**：slot 的寿命由图上"生产者 → 最后消费者"静态确定
（forward 产出的激活要到 backward 才死）。编译器据此投影整个 step 的峰值显存：
存活槽位 + workspace + 通信缓冲。**不得依赖运行时 OOM 才发现放不下。**

**契约 MEM-3（预算门禁）**：峰值超过 `budget_bytes` 时编译失败，错误必须指出峰值出现在哪一步、
以及哪些策略能把它压下来。`auto` 模式按 keep → offload → recompute 的确定顺序降级，
并把每一次降级决策记入 plan digest 与日志（与 R-2 同一原则：降级必须显式且可追溯）。

**契约 MEM-4（寿命复用）**：寿命区间不相交的槽位必须复用同一块缓冲，产出 `MemoryPlan`。
复用决策是 digest 的一部分（它决定执行期的地址，而地址影响 CUDA graph 捕获）。

**契约 MEM-5（state 是独立一类）**：`SlotKind` 区分 `State { kind: Recurrent | Kv { capacity,
paging, block } }` 与激活。state 有容量维度与淘汰/分页策略，激活没有；把两者混为一类会让
KV 的分页策略污染激活的寿命复用。

**配置面**：

```toml
[kernel.memory]
budget_bytes      = 120_000_000_000
activation_policy = "auto"        # auto | keep | recompute | offload
auto_target       = 0.85
recompute_groups  = 4
prefetch_depth    = 2
optimizer_state   = "shard"       # replicate | shard
pool              = "slab"

[kernel.memory.ops.mlp_swiglu]
activation_policy = "offload"

[kernel.memory.state]
kv_capacity = 131072
kv_paging   = "block"
kv_block    = 64
```


### 2.11 反向推导

**三种反向语义，一套机制**：

| `rs_backward_kind` | 谁提供 | 用途 |
|---|---|---|
| `EXPLICIT` | 注册的反向实现（通常是融合 kernel） | 性能路径 |
| `AUTODIFF` | 由其声明的 `expansion` 推导 | 组合路径与验收基准 |
| `NONDIFF` | 无 | 梯度不流过 |

妙处在于 `AUTODIFF` 推导出的反向**正好是 `EXPLICIT` 反向的验收基准** —— §2.8 的"展开等价"检查即是此用。
三者不是三套机制，而是一套机制加一道数值门禁。

**契约 B-1（推导机制）**：反向拓扑遍历 forward 图。每个节点按其 `backward` 种类展开：
`EXPLICIT` 发一个调用已注册反向实现的节点；`AUTODIFF` 对其 `expansion` 递归发 VJP 片段；
`NONDIFF` 的输出若存在非空伴随即为错误。一个 slot 有多个消费者时，在汇合处插入累积节点。

**契约 B-2（VJP 表在框架侧，融合的 VJP 来自 expansion）**：原语词表由框架拥有（与 `shard_rule`
同一理由），所以原语的 VJP 规则也在框架侧。复合/融合算子不单独声明 VJP —— 它的 VJP 由自己声明的
`expansion` 推导（契约 R-4）。因此 R-4 不是"可选的文档"，而是反向可用性的前提。

**契约 B-3（save-for-backward 不是独立机制）**：ABI 的 `RsMemReq.save_for_backward_bytes` 不需要
单独实现。"为反向保存激活"就是**该 slot 的最后一个消费者落在 backward 阶段**，而 `MemoryPlan`
已经在算寿命。保存、释放、重算的显存账自动正确，不需要一套与 plan 并行的 checkpointing 子系统。
（旧代码正是在这里长出了 `QWEN36_SUBCKPT` / `megakernel` / `OFFLOAD_ACTIVATIONS` 三条手写路径。）

**契约 B-4（反向必须重跑切分传播）**：通信算子的伴随关系是
`all_reduce` 自伴随、`all_gather ↔ reduce_scatter`。反向图的结构与前向不同，切分需求也不同，
因此 **`shard::propagate` 必须在反向图生成之后对整张图再跑一次**。只对前向传播、反向复用其结论，
在并行训练下必然错。

**契约 B-5（量化在反向是策略，不是导数）**：FP8 训练惯例是直通估计（STE）—— 量化器的"导数"为 1，
梯度走高精度。因此 `quantize` / `dequantize` 的 VJP 由一个有名字的策略决定
（`straight_through` | `zero` | `custom`），而不是数学推导出来的。这是"配置控制精度与量化"在反向侧的落点。

**契约 B-6（adjoint 存储模型 —— 镜像 + 寿命驱动释放）**：每个需要梯度的 forward slot 配一个伴随 slot，
反向图是前向图的镜像。伴随 slot 的寿命由 `MemoryPlan` 自然给出（其最后消费者即最后一次累积），
不需要引用计数。选它是因为零新增机制，且与显存规划共用同一本账。

**契约 B-7（确定性）**：反向图是 forward 图与 VJP 表的纯函数，因此同一 forward 图必得同一反向图，
并可进入 digest。

**词表补全（原为 P3 的前置条件，现已闭合）。** 照原 §2.4 词表无法写出下列 VJP，缺的五样已按下表补齐：

| 缺什么 | 谁需要 | 补法 |
|---|---|---|
| `reduce` 的 `keepdim` | softmax / layernorm 的 VJP | `reduce` 增加 `keepdim` 属性（bool，默认 false） |
| `rsqrt` / `pow` | rmsnorm 的 VJP | `elementwise_unary` 增 `rsqrt`；`elementwise_binary` 增 `pow` |
| 比较与 mask | max-reduce 的 VJP、带 mask 的 loss | 新原语 `compare`：两输入同形状 → f32 输出恰好为 1.0/0.0，属性 `kind ∈ {eq,ne,lt,le,gt,ge}`，NaN 一律为 0.0（IEEE） |
| 激活函数的导数 | silu′ / gelu′ | `elementwise_unary` 增 `silu_grad` / `gelu_grad` / `sigmoid_grad` / `tanh_grad` / `relu_grad`，取值为 f′(x) |
| 部分累积 | `scatter_add` | `scatter` 增加 `reduce ∈ {assign, add}` 属性；`add` 按**升序源索引**累积以保证逐位可复现 |

**设计取舍**：一律用"新增 kind/属性"而非"新增原语"来补，只有比较必须成为新原语（它产生 mask，语义上不是逐元素算术）。
这样 VJP 片段可以用 `elementwise_binary(mul, elementwise_unary(silu_grad, x), dy)` 这种组合表达，词表增长最小。

**注意（词表与切分规则必须同步）**：新增原语若忘记加进 `shard::rule_for`，不会报错 —— 它会静默落到 `Declared`，
即"分布取自 plan 而非推导"，于是它周围永远不会有集合通信被插入。安全但静默，所以两者必须一起改。

**已知缺口（P3 的前置条件之二）：反向输出的接线约定。**

分两种情况，只有一种需要动 ABI：

- **`AUTODIFF` 不需要**。推导走的是 `expansion` 里的原语，原语的 VJP 表在框架侧（契约 B-2），
  框架本来就知道 `embedding` 的 indices、`gather`/`scatter` 的索引、`cross_entropy` 的 target
  不接伴随。所以逐输入可导性**不必进 ABI**。
- **`EXPLICIT` 需要**。框架要按名调用注册的反向实现，但 `rs_op_desc` 没有说明
  **反向实现返回的几个输出，分别对应前向的哪几个输入**。`linear_backward` 返回 `(dX, dW)`，
  而前向是 `(x, w, b)`；bias 存在时是三个，不存在时是两个。缺这条约定，接线只能靠猜，
  而猜错的表现是**静默错误的梯度** —— 比报错糟得多。

修法：给 `rs_op_desc` 加一个小的映射声明（反向输出的索引 → 前向输入的索引），装载时校验其自洽。
这是 ABI v1 → v2 的一处变更，应在**第一个 `EXPLICIT` 反向算子接入之前**完成，否则会先长出一套
隐式约定。

---

## 3. 交付物

> 状态标记：`- [ ]` 未开始 / `[-]` 进行中 / `- [x]` 完成。完成必须附验证证据。
> 排序即优先级：先把编排链路打通，计算层只接 ATen 一家即可跑通。

### P0 — 脚手架（本机可完整验证，不依赖 GPU/torch）

**D1 · ABI 与插件装载**
`rustrain-abi`：C 头 + Rust 镜像（尺寸/偏移断言）+ `dlopen` 装载 + 描述符校验。
验收：`cargo test -p rustrain-abi`；测试用 **C 编译**一个最小插件 `.so`，框架装载、枚举、调用成功；
版本不匹配被拒绝；缺 execute 的描述符被拒绝。
状态：`- [x]` — `cargo test -p rustrain-abi` 12 unit + 11 integration 通过；C 编译的插件 `add@c` 端到端装载/枚举/调用成功；C 侧 14 个 `_Static_assert` 与 Rust 侧尺寸/偏移断言一致（并修正了一处错误 pin：`sizeof(rs_plugin)` 是 48，不是 56）；版本不匹配、缺入口符号、缺 execute、空描述符表、init 失败均被拒绝；clippy `-D warnings` 干净。

**D2 · 算子注册表与 recipe**
`rustrain-ops`：算子描述、注册、按能力筛选、recipe 解析、拒绝原因聚合。
验收：`cargo test -p rustrain-ops`；解析失败时错误列出每个候选及拒绝原因。
状态：`- [x]` — `cargo test -p rustrain-ops` 55 通过（47 unit + 7 integration + 1 doctest）；`prefer` 绝对不回落，`fallback` 逐项记录被跳过的原因，`deny_unknown_fields` 让拼写错误成为硬错误，`backward = "autodiff"` 作为策略拼写而非变体名，`Registry::backward_of` 追踪声明的反向算子。

**D3 · 并行拓扑与切分规格**
`rustrain-parallel`：`RankLayout`、进程组解析、`ParallelLayout`、通信转换推导。
验收：`cargo test -p rustrain-parallel`；给定 tp=2,cp=2,ep=2,world=8 能解析出正确组与坐标；
由布局差异推导出正确的通信算子（Replicate↔Partial(Sum) 得 all_reduce，Shard(dim) 展开得 all_gather 等）。
状态：`- [x]` — `cargo test -p rustrain-parallel` 48 通过；rank 公式由三张手算表钉死（TP 最快 → CP → EP → DP → PP 最慢）；13 条转换规则逐条测试，含 `Replicate → Shard` 为本地操作（不产生集合通信）这类易错规则；变异测试证明 EP↔DP 换序会被检出。

**D4 · Plan IR 与编译器**
`rustrain-plan`：builder、切分传播、五道校验、编译、digest。
验收：`cargo test -p rustrain-plan`；同一 plan 两次 digest 相同；人为破坏 shape/精度/能力/切分
各触发对应校验失败且错误可读；切分传播能自动插入 all_reduce。
状态：`[-]` — IR、切分传播、校验、编译、digest 已实现，`cargo test -p rustrain-plan` 13 通过；`row_parallel_linear_inserts_all_reduce` 证明 row-parallel linear 会**自动**插入 all_reduce 并把消费者重连到转换后的槽位。未完成：AUTODIFF 的展开求导（当前显式报错，不静默跑错 kernel）、与 runtime 的端到端联调。

**D5 · 执行器**
`rustrain-runtime`：slot 显存池、扁平调用序列驱动、phase 排序、通信调度与 overlap。
验收：`cargo test -p rustrain-runtime`；含通信节点的 plan 在本地 mock 通信后端上执行，
结果与逐节点手写调用一致；`side_stream` 通信按声明的顺序出现。
状态：`- [x]` — 执行器、扁平调用序列驱动、collective 后端抽象（`Allocator` / `CollectiveBackend`）已实现，并且**执行器真正使用 `MemoryPlan` 的偏移**：只分配两块区域（常驻 + 激活池），槽位按规划器给的 offset 落位 —— 规划器判定寿命不相交的两个激活就是同一块字节。`cargo test -p rustrain-runtime` 15 通过，其中：
  - `row_parallel_weight_inserts_a_collective_the_runtime_drives`：Rust 插件经 ABI 注册 → 按 recipe 解析 → 编译期自动插入 all_reduce → 执行器真实驱动它；
  - `non_overlapping_activations_share_one_buffer`：5 个同尺寸激活串成链，复用 ≥3 次，池远小于 5×单槽，且规划器标记为共享的两个槽位解析到**同一个地址**；
  - `conformance_gate.rs` 6 个测试证明门禁会失败（见 D7）。
`SingleRank` 在 world_size>1 时**拒绝执行**而不是假装完成；执行器拒绝任何依赖运行时未实现显存策略的 plan。未完成：`rs_services` 的 alloc/stream 回调尚未接线（provider 目前在自己运行时内部分配）。

**D6 · reference provider（纯 Rust）**
`rustrain-kernels` 的 `reference` 实现：覆盖 §2.4 全部原语（CPU、纯 Rust、无 torch）。
定位是**数值基准与本地可测的执行后端**，不是性能路径。
验收：`cargo test -p rustrain-kernels`。
状态：`- [x]` — 26 个算子，变体统一为 `reference.f32`，每个都有 `infer`（纯计算、不分配）、`memory`、`execute`、`last_error`。`cargo test -p rustrain-kernels` 51 通过；clippy 零 warning；`cargo build -p rustrain-kernels` 产出 `librustrain_kernels.so`，并有测试经 `rustrain-abi` 的**真实 dlopen 装载器**驱动它。测试含：手算期望值、matmul 对比朴素三重循环（逐位）、数值稳定性（softmax 大值、CE 极端 logit、rmsnorm 近零）、量化往返在半 ULP 内、全部 26 个算子两次运行逐位一致、以及 sdpa/cross_entropy/adamw 的 fused≈expansion 逐步等价。

**D7 · CLI：ops check / plan explain**
`rustrain ops check`（四查门禁，机器可读报告，失败非零退出）；
`rustrain plan explain`（打印解析后的完整计划：每节点 op/variant/numerics/layout/通信）。
验收：`cargo run -p rustrain-cli -- ops check --json`；故意注入错误 kernel 被检出；
改 recipe 一个字段后 `plan explain` 输出随之改变，**期间不重编译**；
`rg -n 'getenv|env::var' crates/` 在算子路径上零命中。
状态：`- [x]` — `ops list` / `plan explain` / **`ops check`** 均已可用（文本 + `--json`）。
`rustrain ops check` 对 17 个算子跑数值（对照 `reference.f32`）、展开等价（重放声明的 `expansion`）、
确定性与梯度（**跳过并写明理由**：反向推导未实现）四项，通过退出码 0、失败非零。
**门禁已被证明会失败**：`conformance_gate.rs` 注入一个"把 add 算成减"的实现 → 数值检查抓到；
注入一个每次调用多加一个常数的实现 → 确定性检查抓到；参考实现自身被如实标为 `skip` 而非 `pass`（
"和自己是同一次执行"不是证据）；未注册的变体报错而不回落。无 case 的 8 个算子由 CLI 明确列出并给出理由。
`rustrain ops check` 对 17 个算子跑数值（对照 `reference.f32`）、展开等价（重放声明的 `expansion`）、
确定性与梯度（**跳过并写明理由**：反向推导未实现）四项，通过退出码 0、失败非零。
**门禁已被证明会失败**：`conformance_gate.rs` 注入一个"把 add 算成减"的实现 → 数值检查抓到；
注入一个每次调用多加一个常数的实现 → 确定性检查抓到；参考实现自身被如实标为 `skip` 而非 `pass`（
"和自己是同一次执行"不是证据）；未注册的变体报错而不回落。无 case 的 8 个算子由 CLI 明确列出并给出理由。

### P1 — 上 GPU

**D8 · aten provider**
`rustrain-kernels` 的 `aten` 实现：原语走 ATen（底层 cuBLAS/CUTLASS），`reserved[0]` 携带 `at::Tensor*`。
这是**唯一**需要 libtorch 的部分，以独立插件 crate 形式存在。
验收：验证宿主上 `cargo test -p rustrain-kernels --features aten`；`ops check` 中 aten 与 reference 在容差内一致。
状态：`- [ ]`

**D9 · 端到端训练**
一个真实形状的模型表达为 plan，在 GPU 上用 recipe 驱动训练，loss 下降；通信由切分传播插入。
验收：训练命令 + loss 曲线 + `manifest.json`；同一 manifest 重跑得到相同 digest；
改 recipe 中 forward 实现后 digest 改变而命令不变。
状态：`- [ ]`

**D10 · 多卡验证**
TP（或 CP/EP）≥2 的配置跑通，通信由传播插入而非手写。
验收：2 卡训练 loss 与单卡在容差内一致；plan 中可见自动插入的 `all_reduce` 节点。
状态：`- [ ]`

### P2 — 收敛

**D11 · 一个真实模型架构移植**
一个真实模型族完整表达为 plan fragments，加载真实 checkpoint 训练。
验收：与 archive 分支实现在相同输入上的 loss 对齐，误差在声明容差内。
状态：`- [ ]`

**D12 · 打通外部高性能 kernel**
至少一个第三方高性能 kernel（Tilelang / CUTLASS / FlashAttention 之一）以插件形式接入 ABI，
声明 expansion 并通过等价检查。
验收：`ops check --op <该算子>` 通过；fused 与 primitive 展开结果在容差内一致。
状态：`- [ ]`

### P3 — 显存策略

**D13 · 寿命分析与峰值投影**
`rustrain-plan` 增加 memory pass：slot 寿命区间、按策略投影峰值、不相交寿命的槽位复用，
产出 `MemoryPlan`。编译期调用每个算子的 `memory()` 取 workspace 与 save-for-backward。
验收：`cargo test -p rustrain-plan`；手工可算的小图峰值与预期一致；寿命不相交的两槽位被分到同一偏移；
`rustrain plan explain` 打印峰值与复用表。
状态：`- [x]` — 寿命分析、峰值投影、不相交寿命的槽位复用（首次适配）已实现，编译期调用每个算子的 `memory()` 取 workspace。`pool` 默认为 `slab`（复用正是寿命分析存在的理由，默认关掉等于白白浪费已证明可省的内存）。`cargo test -p rustrain-plan` 22 通过，含 8 个显存测试。**执行器已按这些偏移分配**（见 D5），所以 `MemoryPlan` 是执行事实而非建议。

**D14 · 预算门禁与策略降级**
超预算时编译失败并指出峰值步骤与可用策略；`activation_policy = "auto"` 按确定顺序降级并记录决策。
验收：`cargo test -p rustrain-plan`；把 `budget_bytes` 调到峰值以下必失败且错误可读；
`auto` 在两次运行中做出**相同**的降级序列（确定性）。
状态：`- [x]` — 超预算编译失败，错误指出峰值节点、该处活跃字节数、以及可用的缓解手段；`auto` 按 keep → offload → recompute 的确定顺序降级并逐条记录理由。`cargo test -p rustrain-plan` 覆盖：budget=1 必失败且错误可读、运行时无 offload/recompute 能力时**不谎报**（决策为空、门禁照常拒绝）、有支持时两次运行的降级序列**逐条相同**、运行时不支持却被 recipe 要求的策略进入 `unsupported` 并在 `explain` 中显式列出。

**D15 · state 槽位与 KV 策略**
`SlotKind` 区分 `Recurrent` 与 `Kv { capacity, paging, block }`；state 槽位按容量预分配，
KV 按 block 分页。为 rollout / 有状态注意力 / ring attention 提供统一的 state 分配。
验收：`cargo test -p rustrain-plan`；prefill 与 decode 两种访问模式下 KV 分配与回收次数可预测；
容量超限是显式错误而非静默重分配。
状态：`- [ ]`


**D16 · 词表补全与 VJP 表**
补上 §2.11 列出的词汇缺口（`reduce.keepdim`、`rsqrt`/`pow`、比较与 mask、激活导数 kind、`scatter_add`），
并在 `rustrain-plan` 建立原语 VJP 表。`rustrain-kernels` 的 reference provider 为每个新原语提供实现。
验收：`cargo test -p rustrain-kernels`；每个新原语的 VJP 与有限差分一致（容差内）。
状态：`- [ ]`

**D17 · `derive_backward` pass**
反向拓扑遍历 + 三种 `backward` 种类展开 + 汇合处累积 + 对整图重跑切分传播 + 进入 digest。
验收：`cargo test -p rustrain-plan`；小 MLP 的推导反向与手写反向逐位一致（同一次序）；
前向的一对 column/row-parallel linear 在反向图上得到**位置正确**的通信；同一 forward 图两次推导 digest 相同。
状态：`- [ ]`

**D18 · explicit 反向 ≈ 推导反向**
把 `ops check` 的等价检查从"融合 ≈ 展开"扩展到"`EXPLICIT` 反向 ≈ 推导反向"。
验收：`rustrain ops check` 对至少一个声明了显式反向的算子执行该检查并通过；故意注入不一致会被检出。
状态：`- [ ]`


---

## 4. 依赖与门

- **门 G1**：D1–D7 全部完成且有可跑验收证据，才能进入 P1。
- **门 G2**：D4 的切分传播通过，D5/D9 才有意义；否则通信仍是手写。
- **依赖**：D2←D1；D4←D2,D3；D5←D4；D7←D1..D6；D13←D4；D14←D13；D15←D13。
- **门 G3**：D13/D14 在 D9（端到端训练）**之前**完成。否则第一次真实训练就会用手改代码绕开它，
  而绕过一次之后就不会再回来。
- **门 G4**：D16 是 D17 的前置 —— 词表缺一个原语，对应算子的 VJP 就写不出来。
- **门 G5**：D17 在 D9 之前。没有推导反向，"训练"无从谈起。
- **环境事实**：本机有 `cargo 1.96`，**无 torch、无 CUDA、无 GPU**。P0 必须能在本机完整验证。
- **验证宿主**：`root@47.94.214.197:26002`（8× NVIDIA L20X 143GB，sm_89，CUDA 13.0 + nvcc，
  torch 2.11.0+cu130，1600GB RAM，Rust 1.98.1 已装）。

---

## 5. 已确认约束

| 约束 | 来源 |
|---|---|
| 核心 crate 不得依赖 tch/libtorch/cuda | 用户要求"不重编译即换算子"；本机无 torch |
| 算子路径禁止读环境变量 | 用户要求"通过配置控制算子"；项目 dev-sop 可复现性宪法 |
| 禁止静默 fallback | 现状诊断：五处静默降级是混乱根因 |
| 量化方案必须是数据 | 现状诊断：block 128 硬编码，方案从形状反推 |
| 融合算子必须声明展开 | 等价性验收唯一依据 |
| **不自研 kernel，计算交给 ATen/CUTLASS/第三方** | 用户明确：框架重点在通信与编排，kernel 次要 |
| **并行切分进 plan，通信由传播插入** | 用户确认"编排第一公民"；避免 Megatron 式手写通信 |
| **显存策略进 plan，峰值编译期门禁** | 用户要求"显存管理机制要可控"；旧代码这一维有 6 个 env var |
| **state（含 KV）是独立槽位类别** | 用户点名 KV cache；其容量/分页语义与激活不同 |
| **反向由 plan 推导（非新增层、非全手工）** | 用户 P3 决策；`expansion` 因此真正承重 |
| **VJP 表在框架侧，融合的 VJP 来自 expansion** | 与原语的切分规则同一理由：词表归框架所有 |
| **反向必须重跑切分传播** | `all_gather ↔ reduce_scatter` 伴随关系使反向的切分需求与前向不同 |
| 验证必须在 `47.94.214.197:26002` | 用户明确指定 |
| 旧 rustrain 副本与 legacy kernel 源码已删除 | 用户明确要求（归档中保全） |

---

## 6. 待解决

*（执行前必须为空。当前为空。）*

---

## 7. 上下文层（volatile — 快照于 2026-08-03；会过时，执行者须重新核对）*

### 7.1 探索证据

诊断细节见 `_internal_docs/kernel-first/exploration.md`（私有）。要点：

- 4 个 `.so`、67 个导出算子入口，粒度从"整个训练步"到"单个 rmsnorm"到"显存分配器旋钮"混在同一 `extern "C"` 块。
- `qwen3_6_kernels.cpp` 内 38 处 `getenv`，在 4 条并行 forward 实现间切换。
- `session_ep.rs` 手写布尔把 C++ attention/loss/optimizer 绑在同一开关上，且要求 `world_size == 1`
  → 真实 EP run 跑的是 Rust 路径（AGENTS.md 宣称的"一层一次 FFI"在生产配置上是死代码）。
- `rustrain-qwen` 与 `rustrain-qwen3` 95.1% 逐行相同（20,645/21,708 行）。
- FP8 block 128 硬编码，量化方案从 scale 张量形状反推；`scale_fmt`/`weight_block_size`/`expert_dtype`
  解析后从未被读取。
- 当次 checkout 中 `libv4_flash_kernels.so` 与 `libqwen36_kernels.so` 均编译失败，但只输出
  `cargo:warning`，构建仍然成功。
- 直通式 all-reduce 的 detach 技巧 `x + (allreduce(x) - x).detach()` 被复制粘贴在 3 处
  —— 分布式 autograd 语义被手工内联，这正是 I-3 要消除的。

### 7.2 被放弃的方案

- **渐进式（保留现状 + 加一层注册表）**：用户否决——"不要在屎山上雕花"。
- **自研 kernel 路线**：用户否决——框架重点在通信与编排，计算交给成熟实现。
- **通用 IR 编译器（mini-XLA）**：否决。原语固定、展开限 2 层、切分传播只支持声明的 `shard_rule`，
  不做通用图优化。

### 7.3 环境与资产

- 本机 = 编辑/编译盒：`cargo 1.96`（`/root/.cargo/bin`），无 torch/CUDA/GPU，仓库在 gpfs。
- 验证宿主 NAS 上的旧 rustrain 副本（6.0G）**已删除**；新代码需同步过去。
- 旧实现完整保全：tag `archive/pre-rewrite-20260803`、分支 `archive/legacy-743commits`、
  bundle `/data/user/nolanho/backups/rustrain-full-history-20260803.bundle`（含 11 个 kernel 源文件）。
- 本机 `~/.ssh/config` 增加了 `rustrain-verify` 条目（回收：`~/.ssh/config.bak-dsh-*`）。

### 7.4 交付物履行状态

见 §3 各条目的状态标记。
