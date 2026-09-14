# EP dispatch/combine 怎么落地 —— 待裁定的取舍说明

**状态：待用户裁定。** 这份文档只回答一个问题：`moe_layer` 声明的两个 `ALL_TO_ALL`
（expert dispatch / combine）要由谁、以什么形式实现。裁定前的默认行为已经是对的——
**拒绝运行**，不是静默少通信：

```
check --ep 4 --dtype f32  →
fail l1.layout_propagation: node NodeId(32) (moe_layer) declares communication the planner
cannot express: the operator declares expert-parallel routing (all_to_all dispatch/combine) over 4
on tensor 0, and the planner has no expression for it yet; the plan would be missing
communication the operator's math owes, so it is refused rather than run
```

---

## 1. 现状（全部实测/源码位置，不含推测）

| 事实 | 证据 |
|---|---|
| `moe_layer` 是**一个**显式算子，声明 3 条集合通信：`ALL_TO_ALL{TP\|EP}` on tensor 0（dispatch）、`ALL_TO_ALL{TP\|EP}` on tensor 10（combine）、`ALL_REDUCE{TP}` on tensor 10 | `crates/rustrain-kernels/src/lib.rs:480-522`；ATen 插件同样三条 `plugins/aten/src/ops_moe.cpp:204-206` |
| 描述里 router（`topk_router`）在**上游**，`moe_layer` 的输入 1/2 就是它的输出（routing weights / indices） | `crates/rustrain-kernels/src/lib.rs:177`（`MOE_LAYER_DOC`） |
| 编译器对 `ALL_REDUCE` 有表达（转成 `partial(sum, tp)`），对 `ALL_TO_ALL` **没有**：ep>1 → `PlanError::DeclaredCollective` 硬失败；ep=1 → 记一条 note 跳过 | `crates/rustrain-plan/src/compile.rs:309-445`（`apply_declared_collectives`），错误类型 `crates/rustrain-plan/src/lib.rs:140-144` |
| 运行期**已经能执行** `all_to_all`：`intrinsic.all_to_all` 是词汇表里的内在算子，编译期校验 `group`/`dim`/`split`，执行走 NCCL `ncclSend`/`ncclRecv` 对（`ncclGroupStart/End`），支持不均匀 split，单进程 world>1 时硬失败 | `crates/rustrain-plan/src/ir.rs:484`、`crates/rustrain-plan/src/compile.rs:779-930`、`crates/rustrain-runtime/src/lib.rs:687-701`、`crates/rustrain-runtime/src/nccl.rs:785-882` |
| 两个 provider 的 `moe_layer` **body 都是纯本地**的：dropless、不做任何交换（"per token EVERY selected expert runs — dropless, no capacity truncation"） | `crates/rustrain-kernels/src/op/moe.rs:276`；`plugins/aten/src/ops_moe.cpp:1-15`（`moe_execute` :128） |
| 插件的 service table **不提供**集合通信，且这是写下来的设计理由（kernel 与拓扑无关，I-7） | `crates/rustrain-runtime/src/lib.rs:953-963`（`RsServices.collective` 为 `None`）；理由在 `crates/rustrain-plan/src/ir.rs:478-484` |
| 唯一需要翻转的测试：`the_five_axis_mesh_is_refused_naming_the_unwired_expert_routing`（钉住 exit 1 与原因） | `crates/rustrain-cli/tests/check_report_contract.rs:904` |

**所以缺口只有一处**：编译器没有把"声明的 ALL_TO_ALL"变成 plan 里的步骤。传输、校验、
CPU 模拟、NCCL 路径全都已经在仓库里并有测试。

### 1.1 为什么这不是"加个 split 就行"

`all_to_all` 的 split = **每个 rank 发给每个 rank 多少行**，它取决于 router 的 top-k 结果：

- 稀疏 dropless 调度下，rank 发给 rank *j* 的行数 = 本 rank 上被路由到 *j* 的专家集合的行数，
  **运行期才知道**（`MOE_LAYER_DOC` 原话："whose per-rank token counts are only known at run time"）。
- 而 `intrinsic.all_to_all` 的 split 今天是**编译期常量**（`ATTR_SPLIT`，`ir.rs:509-513`；
  `CollectiveRequest.split: Option<Vec<i64>>`，`collective.rs:100-102`）。
- NCCL 的 `ncclSend/ncclRecv` 本来就要求调用时给出 host 侧 count —— 所以运行期 split
  必然意味着"每次交换前把 ≤ ep 个整数从设备读回 host"，这个成本的实测值见 §3。

### 1.2 本模型的 EP 几何（实测自 fixture）

`E=256`、`topk=8`、`H=2048`、`I=moe_inter=512`、40 层 + 1 层 MTP。
`ep=4` → 每 rank 64 个专家；`--tokens` 的 8 个 token 是现在的 debug 尺寸，`params.seq=512`，
真实训练序列会是 2K–8K。所有性能数字按 S=8192、ep=4、每 rank 一层给出（宿主实测，单卡 L20X，
bf16，`torch.matmul` 组合 gate/up/down 三次 GEMM）。

---

## 2. 数学上的三条路

记 `S` = token 数、`K=topk`、`E` = 专家数、`ep` = 专家并行度。dropless 语义 =
"每 token 的每个被选中专家都算，按专家下标升序加权求和"。

| 路线 | dispatch 怎么切 | 本地算什么 | combine | 数值 |
|---|---|---|---|---|
| **A-dense**（静态、稠密） | token 轴均分：每 rank 把 S/ep 行发给每个目的 rank | 本地 E/ep 个专家 × **全部 S 个 token** | 反向 a2a + 本地加权求和 | 与 HF/dropless **完全一致**（未选中的权重为 0） |
| **A-cap**（静态、容量） | 容量常量 split：每 rank 每目的最多 `capacity` 行，超出丢弃 | 只算收到的行 × 本地专家 | 反向 a2a + 加权求和 | **与 HF 不一致**（丢 token，且 drop 位置依赖运行期数据） |
| **B**（运行期 split、稀疏） | 按 router 的真实计数发（每目的行数 = 被路由到该 rank 的行数） | 只算收到的行 × 本地专家 | 反向 a2a（同一份计数矩阵的转置）+ 加权求和 | 与 HF **完全一致**，dropless、无容量丢弃 |

通信量（S=8192、ep=4、bf16、每行 H=2048 即 4 KiB）：
- **B / A-cap**：每 rank 每方向 `S·K/ep = 16384` 行 = **64 MiB**（集群 256 MiB/方向/层）；
- **A-dense**：每 rank 每方向 `ep × S/ep = S = 8192` 行 = **32 MiB**（集群 128 MiB）——
  dense 把网络流量**减半**，代价是算力 ×32.6（见 §3）。A-cap 的通信量与 B 相同，但会丢。

---

## 3. 后果（宿主实测数字 + 契约代价）

### 性能

| | 每 rank 每层的专家 FLOPs（S=8192, ep=4） | 每次交换的流量（单向） | 实测耗时 | 41 层合计 | 相对 |
|---|---|---|---|---|---|
| **B（稀疏）** | 103 GFLOP（16384 行 × 3 GEMM） | 64 MiB | **0.14 ms** | 5.7 ms/rank | 1.00× |
| **A-cap（容量）** | 与 B 同量级（丢 token 后更少） | 64 MiB | ≈0.14 ms | ≈5.7 ms/rank | ≈1.00× |
| **A-dense** | 3299 GFLOP（524288 行） | 32 MiB | **4.69 ms** | 192 ms/rank | **32.6×** |

运行期 split 的唯一额外成本 = 每次交换前一次 `D2H(ep 个 int64) + sync`：**实测 14.7 µs**，
两次交换 × 41 层 = **1.2 ms/前向**（对照：一次 tp=4 前向的墙钟是 9.6 s，占比 0.01%）。
dispatch 的载荷（每 rank 每层 64 MiB）本地拷贝实测 0.03 ms（3864 GB/s），跨卡只受互连带宽限制。

**结论：在"性能 > 灵活性 > 正确性"的排序下，A-cap 相对 B 的性能优势是 1.2 ms/前向（噪声级），
而 A-dense 的代价是 32.6× 专家 FLOPs。B 在性能上不输、在灵活性上唯一通用、在正确性上最好。**

### 灵活性

- **B** 让"数据依赖的集合通信"成为框架的一等表达：任何运行期才知道 split 的交换
  （MoE routing、序列 packing、变长 attention 的 all_to_all 都是这一类）都能落进同一套
  plan/校验/遥测。这是本框架"通信调度"差异化价值能覆盖的最后一大类通信。
- **A** 只能表达静态 split：以后每个数据依赖的通信都要重复一次同样的取舍讨论。
- 代价面（B）：多一个"元数据输入槽"概念（plan 里记 slot id，形状静态 `[ep]` i64，编译期仍可校验），
  split 的**和**只能在运行期校验（`nccl.rs:788-815` 已经有这条硬失败），编译期校验降级为
  "形状/组/长度对得上"。

### 正确性

- **A-cap** 会静默改变数值：丢 token = 训练梯度与 HF/dropless 不一致，而且丢在哪里依赖
  routing 数据（同一个 batch 重放才可复现）。用户把正确性排在最后，但在训练框架里这一类
  错误是"梯度悄悄错"，与"测试红"不是一个量级，所以我把它单独列出来，不当作 0 成本。
- **B / A-dense** 都是精确的；B 还是**逐位可复现**的（固定升序累加，同现有 `MOE_LAYER_DOC`）。

### 契约面（每个路线要动什么）

| | 编译期 | 运行期 | 描述/算子词汇 | 需要用户裁定 |
|---|---|---|---|---|
| **B** | `apply_declared_collectives` 为 ALL_TO_ALL 插 `intrinsic.all_to_all` 两步；新增"split 来自元数据槽"的 attr；校验元数据槽形状/组 | 执行前读元数据槽（D2H，14.7 µs），按计数发/收；和不为 extent → 硬失败 | 需要一个产生计数矩阵的元数据槽（router 侧：`topk_router` 多一个输出，或一个小的 local 算子） | **是**：契约新增"数据依赖集合通信"与"元数据输入槽" |
| **A-dense** | 同上，但 split 是常量（token 轴均分），无新 attr | 无需 D2H | 不动 | **是**：接受 32.6× 专家 FLOPs |
| **A-cap** | 同 A-dense + 容量参数（描述侧） | 无需 D2H | 描述里要声明 capacity | **是**：接受丢 token 与数值不一致 |
| **C**（算子自持通信） | 不动 | 接上 `RsServices.collective`，插件自己 dispatch/combine | 不动 | **是**：与 I-7"kernel 与拓扑无关"冲突，框架失去调度/遥测可见性 |

### 三条路线共有的一个绕不开的问题：交换之后的"反置换 + 加权求和"住哪

无论 A 还是 B，被交换的都是"部分贡献"，最终 `out[token] = Σ_k w_k · expert_k(x_token)`
需要**在交换之后**对"回到本 rank 的那些行"做反置换 + 加权求和（外加 shared expert 项）。
今天这个求和住在 `moe_layer` 的 body 里（单进程 world=1 时它是对的），而交换插在 body 之前/之后，
所以必须选一个：

1. **描述里多一个本地小算子**（`moe_combine`：反置换 + 加权求和 + shared expert，无集合通信）
   ——`moe_layer` 缩小成"本地专家对已分发行做三次 GEMM"。改动：描述 + 算子词汇（T2，丢一个 `.so`），
   框架不动 body 语义。
2. **ABI 允许一个节点分两段执行**（框架在段间插交换）—— 框架改动最大，描述不动。
3. **算子自己做交换**（= 路线 C）—— 不新增算子，但违反 I-7。

我的建议：**B + 方案 1**。理由：它是唯一同时满足"性能不输、通用、数值精确"的组合；
方案 1 的改动面是描述与算子词汇（T2，可插拔，不动框架不变式），而方案 2 会把"节点内分段"
变成框架不变式，方案 3 会删掉一条已经写进代码的架构理由。

---

## 4. 如果裁定 B：落地形状（供 review，不是承诺）

1. **描述**：`topk_router` 增加一个输出（或新增一个纯本地算子）产出
   `send_counts[ep]`（本 rank 发往每个 rank 的行数）与 `send_offsets[ep]`；combine 侧用
   `recv_counts = send_counts` 的转置（由一次 `all_to_all` 交换 ep 个整数得到，
   或由同一算子计算后交换）。
2. **plan**：`apply_declared_collectives` 为声明的 `ALL_TO_ALL` 插入两个
   `intrinsic.all_to_all` 步骤（dispatch 在 body 之前、combine 在之后），各带
   `group=TP|EP`、`dim=token 轴`、`split_slot=<元数据槽>`（新 attr，命名 split 的来源）；
   `ALL_REDUCE{TP}` 维持现状（转 `partial(sum, tp)`）。
3. **编译期校验**：元数据槽存在、dtype i64/i32、长度 = 组度数；`dim` 合法；组可用。
   split 的**和**与逐 rank 上界只能在运行期校验（已有硬失败）。
4. **运行期**：执行器在 a2a 步骤前读元数据槽到 host（≤ ep 个整数），构造 send/recv 计数矩阵，
   走现有 `NcclBackend::AllToAll`；计数不合法 → 硬失败并点名步骤。
5. **可行性验证（三层，全部要先做再做数值）**：
   - CPU：`ThreadBackend` 的 a2a 已有测试（不均匀 split / 中间维 / 相等 split），补一条
     "split 来自元数据槽"的用例；
   - plan：`check --ep 4 --dtype f32` 从 exit 1 变 exit 0，插入了 2 条 a2a（每条 moe 层），
     `collectives_by_kind` 里能看到 `all_to_all`；
   - 宿主：`launch --sweep "ep=2"` / `"ep=4"`（world 2/4）与 world=1 基线比 logits；
     以及 **ep=ep 的等价性**（`--ep 2` 与 `--ep 4` 结果应在 bf16 容差内一致，因为它只是把
     同一批专家切成更多份）。
6. **必须翻转的测试**：`check_report_contract.rs:904`（从"钉住拒绝"改成"钉住插入 2 条 a2a"）；
   新增 dispatch/combine 的数值测试（CPU 参考实现 + 宿主）。

---

## 5. 需要你回答的一句话

**选 B（运行期 split + 描述里多一个本地 `moe_combine` 算子，性能最优且数值精确）、
A-dense（不动契约但专家 FLOPs ×32.6）、A-cap（最快一点但丢 token）、还是 C（插件自己做交换，
与 I-7 冲突）？**

在裁定之前，我不会改契约面；`check --ep >1` 继续以现在这条明确的原因拒绝。
