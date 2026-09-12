# Plan IR 与编译器基线（现状）

> 只描述**今天**代码里的事实：`Plan` 的字段、`Compiler::compile` 的 pass 顺序、`CompiledPlan` 的产物、intrinsic 词汇表、`ParallelLayout` 的表达能力、运行期的消费方式，以及模型描述层必须满足的接口。每条断言带 `file:line`；不存在的东西写 "not found"。不含建议与设计意见。
>
> 标注约定：**死钩子** = 全仓无读者，或只有构造者 / 测试在写的字段、方法或变体。

## 1. 类型逐字段清点

`Plan`（`ir.rs:217-222`）= `meta: PlanMeta` + `slots: Vec<Slot>` + `nodes: Vec<PlanNode>`；访问器
`slot`/`slot_mut`/`node`/`slot_id`/`producers` 在 `ir.rs:225-250`。

| `PlanMeta` 字段 | 类型 | 含义 | 谁读它 |
|---|---|---|---|
| `name` | `String` | plan 名 | `compile.rs:122`（Debug）、`compile.rs:138`（explain）、`cli/main.rs:348` |
| `phase` | `Phase` | 整份 plan 的默认 phase | 只被 `PlanBuilder::node` 当默认值（`ir.rs:400`）；编译器读的是 `PlanNode.phase`（`compile.rs:347,384,507`） |
| `parallel` | `ParallelConfig` | 发射时的拓扑度数 | **死钩子**（`rustrain-plan` 内无读者）：只有 `cli/main.rs:340` 读出来再手喂 `Compiler::new`；`Compiler` 用自己那份（`compile.rs:212,314`） |
| `seed` | `u64` | 随机种子 | **死钩子**：唯一写入者 `PlanBuilder::seed`（`ir.rs:353-356`），无读者；但作为 `Plan` 的一部分进 digest（`compile.rs:696-714`） |

`SlotId(pub usize)`（`ir.rs:19-20`）与 `NodeId(pub usize)`（`ir.rs:23-24`）分别是 `slots`/`nodes` 的下标，
派生 `Ord/Hash/Serialize`。读者：`Plan::slot/node`（`ir.rs:225-235`）、`producers`（`ir.rs:242-250`）、
memory（`memory.rs:563-591`）、运行期 buffer 表（`runtime/lib.rs:335-407`）。`NodeId` 另存于
`CompiledStep`（`compile.rs:32,47`）；读它的 `CompiledStep::node()`（`compile.rs:60-64`）**无调用者 → 死钩子**。

`SlotKind`（`ir.rs:28-44`）七变体，读者只有三处：`memory.rs:548-553`（`Weight|Gradient|State` → 常驻区）、
`shard.rs:547-549`（`!State` → `is_distributable`，**该方法无调用者**）、`runtime/lib.rs:796-801`
（`required_inputs`，唯一调用者是测试 `runtime/tests/end_to_end.rs:229`）。

| `SlotKind` 变体 | 构造点 | 备注 |
|---|---|---|
| `Weight` | `cli/main.rs:257,307`、测试 | `memory.rs:551` 读 |
| `Activation` | `cli/main.rs:270,288`、`conformance.rs`、测试 | — |
| `Input` / `Output` | `cli/main.rs:250,317`、`conformance.rs:692,697`、测试 | — |
| `Temp` | 仅 `conformance.rs:606` | 无生产性构造者 |
| `Gradient` | not found | **死变体**：无任何构造点（`memory.rs:551` 会读，但永远读不到） |
| `State` | not found | **死钩子**：无构造点；唯一读它的 `shard.rs:548` 自身无调用者 |

| `Slot` 字段（`ir.rs:47-55`） | 类型 | 含义 | 谁读它 |
|---|---|---|---|
| `name` | `String` | 张量名 | `compile.rs:169-173`（explain）、`compile.rs:418`（转换后改名 `{name}__{op}`）、运行期报错文案（`runtime/lib.rs:34,38,46,63`）。**无唯一性校验**：`Plan::slot_id` 返回第一个匹配（`ir.rs:237-239`） |
| `dtype` | `RsDtype` | 元素类型 | `compile.rs:340-344`（选实现）、`compile.rs:600-606`、`runtime/lib.rs:455,550`；memory 经 `element_bytes()` → `byte_width()`（`ir.rs:64-67`） |
| `shape` | `Vec<i64>` | **具体**形状，不容符号维 | `ir.rs:6-8`；读者 `compile.rs:490`（与 infer 比对）、`shard.rs:366`（定 `DimNormalizer` 的 rank）、`memory.rs:556`、`runtime/lib.rs:392-398,466-468` |
| `layout` | `ParallelLayout` | 该张量的分布 | `shard.rs:244-245`（`effective` 起点）、`shard.rs:280`（声明的输出布局）、`shard.rs:417-419`（转换后写回）。**运行期完全不读**；`compile.rs:719` 的 `slot_layout()` 无调用者（**死钩子**） |
| `kind` | `SlotKind` | 用途 | 见上表 |

`Slot::numel()`（`ir.rs:58-60`）被 `element_bytes()` 消费；`Slot::dim()`（`ir.rs:70-78`）**只在自身测试被读**
（`ir.rs:581-584`）→ **死钩子**。

| `PlanNode` 字段（`ir.rs:183-197`） | 类型 | 含义 | 谁读它 |
|---|---|---|---|
| `op` | `OpRef` | 算子引用 | `op.name`：`shard.rs:270`（`rule_for`）、`compile.rs:257,280`、`memory.rs:230`；`op.variant`：`compile.rs:353-363`（显式 variant 绝对优先，fallback 为空）。`OpRef::display()`（`ir.rs:156-161`）经 `display_op`（`ir.rs:200-202`）**无调用者 → 死钩子** |
| `inputs` / `outputs` | `Vec<SlotId>` | 接线 | `compile.rs:450-459`（infer）、`shard.rs:275-358`（布局推导）、`memory.rs:563-591`（生命周期）、`runtime/lib.rs:582-591` |
| `attrs` | `Attrs` | 算子属性 | `compile.rs:289`（→ ABI）、`compile.rs:532-582`（intrinsic 属性）、`memory.rs:631`（workspace）、`runtime/lib.rs:631`；另经 `Plan` 整体进 digest |
| `phase` | `Phase` | forward/backward/update | `compile.rs:347,384`（backward 节点按 forward 选再走 `backward_of`）、`compile.rs:507`（取 recipe numerics）。**运行期不读** |
| `precision` | `PrecisionOverride` | 逐节点精度覆盖 | 读者 `compile.rs:508-519`；写入者只有 `Default`（`ir.rs:424`、`shard.rs:455`），唯一可变入口 `PlanBuilder::current_node`（`ir.rs:432-434`）**无调用者** → **死钩子（无写入者）** |
| `checkpoint` | `CheckpointPolicy` | 保留 / 重算 / 卸载 | **死钩子**：写入者 `ir.rs:425`、`shard.rs:456`；无读者；不进 `CompiledStep` |
| `stream` | `StreamPolicy` | CUDA stream | 读者 `compile.rs:308,585` → `stream_of`（`compile.rs:591-596`）。`StreamPolicy::Side`（`ir.rs:130`）**无生产者**，故恒为 `MAIN_STREAM`；`CompiledStep::stream()`（`compile.rs:72-76`）**无调用者 → 死钩子** |
| `source` | `Trace` | 诊断来源 | `compile.rs:150`（`is_inserted()` → explain 的 `*` 标记）、`compile.rs:157,163`（打印 path） |

`Trace`（`ir.rs:83-88`）：`path: String` 由 `PlanBuilder::node_in_phase` 拼 `trace_prefix + source`
（`ir.rs:412-416`），前缀来自 `scope()`（`ir.rs:358-361`），调用方直接传 `"mlp.up"`（`cli/main.rs:281`）、
`"layer0.linear"`（`shard.rs:655`）这类自由字符串；插入节点复制生产者的 path（`shard.rs:442,458`）。
**仓库内不存在模块树对象**，path 的格式无任何校验。`inserted_by: Option<String>` 写入者 `ir.rs:101` 与
`shard.rs:458`（`"shard-propagation"`），读者只有 `is_inserted()`（`ir.rs:105-107`）→ `compile.rs:150`。
`Phase` 定义在 `rustrain-ops/src/capability.rs:30-35`（`Forward`/`Backward`/`Update`），经 `ir.rs:16` 重导出；
`name()`（`capability.rs:40-47`）自称 "Used in plan digests"。`CheckpointPolicy`（`ir.rs:112-120`）与
`StreamPolicy`（`ir.rs:123-131`）见上表：前者无读者，后者的 `Side` 分支无生产者（`SIDE_STREAM` 常量
`compile.rs:27` 因此不可达）。

`Attrs` = `BTreeMap<String, AttrValue>`（`attrs.rs:79-80`，有序以保证 digest 稳定，`attrs.rs:76-78`），
`AttrValue` 五类 `I64`/`F64`/`Bool`/`Str`/`I64s`（`attrs.rs:18-25`）。**框架侧只读两种**：`Str`（`compile.rs:534`
的 `group`、`compile.rs:546` 的 `reduce`）与 `I64`（`compile.rs:582` 的 `dim`）；`F64`/`Bool`/`I64s` 无框架侧
读者，访问器 `f64()`/`bool()`/`i64s()`（`attrs.rs:104-118`）**死钩子**，但 `to_abi()` 会完整传递五类
（`attrs.rs:267-283`），实际消费者是插件（`runtime/lib.rs:631`）。`AbiAttrs`（`attrs.rs:212-218`）是第二套
表示，`from_attrs`（`attrs.rs:245-293`）把 key/str/slice 各自 `CString`/`Box` 化以保证移动安全。

## 2. 构造面

`PlanBuilder`（`ir.rs:331-336`）持有 `meta`/`slots`/`nodes`/`trace_prefix`：

- `new(name, phase, parallel)`（`ir.rs:339-351`）：`seed = 0`。
- `seed(u64)`（`ir.rs:353-356`）：只写 `meta.seed`；`scope(prefix)`（`ir.rs:358-361`）：只影响 `Trace.path`。
- `slot(name, dtype, shape, kind)`（`ir.rs:363-371`）→ `slot_with_layout(..., ParallelLayout::Replicate)`
  （`ir.rs:373-390`）：**不填 layout 就是 `Replicate`（`ir.rs:370`）**，要分布必须调 `slot_with_layout`。
- `node(...)`（`ir.rs:392-401`）：phase 取 `meta.phase`；`node_in_phase`（`ir.rs:403-430`）才逐节点指定。
  两者都把 `precision`/`checkpoint`/`stream` 固定为 `Default`。
- `current_node(id) -> &mut PlanNode`（`ir.rs:432-434`）与 `compare_nodes()`（`ir.rs:436-438`）**无调用者**。
- `build()`（`ir.rs:440-448`）：装配 `Plan` 后只调用 `check_structure()`。

`check_structure`（`ir.rs:257-300`）强制四件事：每个 `SlotId` 在范围内（`UnknownSlot`，`ir.rs:263,280`）；
input 的生产者必须是**更早**的节点（`NotTopological`，`ir.rs:271`）；一个 slot 不能被两个节点写
（`SlotWrittenTwice`，`ir.rs:288`）；每个节点至少一个 output（`NodeWithoutOutput`，`ir.rs:296`）。
它**不**检查：slot 名唯一、`layout` 里的 group 在拓扑下是否存在、`meta.parallel` 与 `Compiler` 的 `parallel`
是否一致、节点 phase 与 `meta.phase` 是否一致、dtype 与算子声明是否匹配（后者由 §3 pass 2 的 infer 兜底）。

今天 plan 的构造者只有三处：`cli/main.rs:238-329`（`demo_plan`，两节点 MLP）、`conformance.rs:595`
（算子 expansion）、`conformance.rs:689`（单算子 case），其余全是测试。**plan 的加载 / 保存入口：not found**
—— `Plan` 虽派生 `Serialize/Deserialize`（`ir.rs:217`），但全仓没有读写 plan 文件的路径（recipe 有
`Recipe::from_toml`，plan 没有对应物）。

## 3. 编译器 pass 逐个

`Compiler::compile`（`compile.rs:242-325`）的实际顺序。代码里的 pass 列表原文（`compile.rs:3-7`）：

```text
//! The order matters. Sharding propagation runs first because it *changes the
//! graph* (it splices in collectives); validating before that would validate a
//! graph that is not the one that runs. Resolution runs next so that every later
//! check has an implementation to ask. Shape inference runs last, because it is
//! the only check that needs the resolved descriptors.
```

`compile.rs:252-254` 与 `compile.rs:274`：

> `// Pass 1: resolve everything first. The memory pass has to ask each`
> `// implementation for its workspace before it can project a peak, and it`
> `// must do that before any step is emitted.`
> `// Pass 2: validate against the implementations and flatten.`

**前置门**（`compile.rs:243-246`）：`nodes.is_empty()` → `EmptyPlan`；然后 `plan.check_structure()`。

**Pass 0 `shard::propagate`**（`compile.rs:249` → `shard.rs:235-481`）。输入 `&Plan` + `&ProcessGroups`
（后者在 `shard.rs:236` 被 `let _ = groups;` 丢弃）；输出 `ShardPropagation { plan, inserted }`
（`shard.rs:210-214`）。动作：`rule_for`（`shard.rs:71-81`）按算子名分类（`linear` → `Linear`，
`matmul|bmm` → `MatMul`，20 个点名算子 → `Elementwise`，**其余一律 `Declared`**）；`derive`
（`shard.rs:94-193`）算 required/produced layout；不一致时 splice 一个 intrinsic 节点、追加一个复制自原
slot 的新 slot（`shard.rs:414-419`，改名 `{name}__{op}`）、重指消费者（`shard.rs:466-475`）、Kahn 重排
（`shard.rs:488-538`）。错误：`ShardDerivation`（`shard.rs:284`）、`LayoutConflict`
（`shard.rs:297,304,338,386,400`）、`Shard`（`shard.rs:368,372`）、`Digest`（`shard.rs:529`，环路）。
**故意不做**：同一 slot 不做第二次 fan-out 转换（`claimed` 一票否决，`shard.rs:295-303,337-344`）；
不为 local view 型转换插节点（`shard.rs:376-398` 报错）；不多步转换（`shard.rs:399-411` 报错）；
不校验 group 在拓扑里可用（`groups` 被丢弃，`ShardError::GroupMismatch` 因此永不触发）。

**Pass 1 resolve**（`compile.rs:252-266` → `compile.rs:334-411`）。输入 `&Plan` 与节点，输出
`(RegisteredOp, 被拒候选及原因)`。dtype 取输入 slot（`compile.rs:340-344`）；`op.variant` 存在时走
`registry.resolve` 且 `fallback: Vec::new()`（`compile.rs:356-363`），否则走 `recipe.resolve`
（`compile.rs:364-370`）；`Phase::Backward` 先按 `Forward` 选（`compile.rs:347-351`），再
`registry.backward_of`（`compile.rs:402-409`）。错误：`Resolve`（`compile.rs:372,405`）、
`NotValidatable`（`compile.rs:392`）。**故意不做**：不生成反向图 —— `backward = "autodiff"` 直接拒绝，
原话在 `compile.rs:395-397`（"this compiler does not generate yet"）。

**Pass 1b memory**（`compile.rs:271-272` → `memory.rs:207-310,316-376`）。输入 `&Plan`、
`Vec<Option<RegisteredOp>>`、`&recipe.memory`、`RuntimeCapabilities`；输出 lifetime（`memory.rs:563-591`）、
别名（`memory.rs:595-608`，intrinsic 输出复用输入存储）、workspace（`memory.rs:616-664`，问实现要
`RsMemReq`）、常驻区 + slab 池 + 峰值。必要时按 `keep → offload → recompute` 阶梯放宽
（`memory.rs:262-302`，只在 `recipe.target_bytes()` 为 `Some` 且 `caps` 支持时）。错误：`Digest`
（`memory.rs:214`，resolved ops 数与节点数不符，属内部一致性检查）、`MemoryBudgetExceeded`
（`memory.rs:368`）。**故意不做**：不实现 offload / recompute，只规划并拒绝（`memory.rs:9-12`）；
`RuntimeCapabilities` 两开关默认 `false`（`memory.rs:26-32`），`Compiler::new` 用 default（`compile.rs:221`），
全仓唯一调 `.capabilities()` 的 `conformance.rs:738` 传的也是 default。

**Pass 2 validate + flatten**（`compile.rs:274-312`）。intrinsic 节点走 `compile_intrinsic`
（`compile.rs:523-588`），错误 `UnknownIntrinsic`（`:530`）、`IntrinsicMissingAttr`（`:535`）、
`IntrinsicBadAttr`（`:540,551`）、`NotValidatable`（`:563,571`）；其余节点依次 `check_arity`
（`compile.rs:413-429`，**只判"输入输出不同时为空"**，`ArityMismatch` 变体从未被构造）、
`node.attrs.to_abi()`（`:289`）、`numerics_for`（`compile.rs:506-521`，recipe 的 phase numerics 叠加节点
覆盖）、`validate_shapes`（`compile.rs:434-503`：调实现自己的 `infer`；错误 `InferMissing` `:444`、
`InferFailed` `:480`、`InferredShapeMismatch` `:492`；注意 `compile.rs:491` 只在 `!inferred.is_empty()`
时比对，返回空 shape 的实现可绕过）、最后 push `CompiledStep::Op`（`compile.rs:301-311`）。
**故意不做**：不检查 arity（ABI 里没有，`compile.rs:419-420`）、不做 dtype 掩码校验（交给实现 infer）。

**Pass 3 digest**（`compile.rs:314` → `compile.rs:612-716`），见 §4.4。`Compiler` 另有两个只写不读的字段：`deterministic`（`compile.rs:202`，setter `:233-236`，无读者，故 `PlanError::Nondeterministic`（`lib.rs:138`）从未被构造）与 `caps`；`CompiledPlan::inputs()`（`compile.rs:177-179`）无调用者。

## 4. 编译产物

**`CompiledPlan`**（`compile.rs:102-114`）：`plan: Plan`（运行期全部读写，`runtime/lib.rs:337,439,467,472`）、
`steps: Vec<CompiledStep>`（`runtime/lib.rs:580-708`）、`digest: String`（`compile.rs:123,139`、
`cli/main.rs:349`、测试 `end_to_end.rs:377-389`）、`parallel: ParallelConfig`（`compile.rs:140` 与
`cli/main.rs:350`；**运行期不读**，`SingleRank` 自带 world_size）、`resolved: Vec<ResolvedNode>`
（`cli/main.rs:353` 与测试；`ResolvedNode.rejected`（`compile.rs:99`）除 Debug 外无读者）、
`inserted: Vec<InsertedCollective>`（`compile.rs:126,146,160-165`、`cli/main.rs:358`、测试；**运行期不读**）、
`memory: MemoryPlan`（`runtime/lib.rs:284-290,322-323,354-388`、`cli/main.rs:364-370`）。`Debug` 是手写的
（`compile.rs:116-130`）；`explain()`（`compile.rs:134-167`）打印 plan/digest/world、每个 step 的
label ← inputs → outputs @`Trace.path`，再拼 `memory.explain()` 与 inserted 通信表。

**`CompiledStep`**（`compile.rs:30-57`）两个 variant：

- `Op { node: NodeId, op: RegisteredOp, numerics: RsNumerics, attrs: AbiAttrs, inputs: Vec<SlotId>,
  outputs: Vec<SlotId>, stream: StreamId, phase: Phase, source: Trace }` —— 运行期只用 `op`/`attrs`/
  `inputs`/`outputs`（`runtime/lib.rs:626-696`）；`numerics`/`stream`/`phase`/`source` 运行期不读，
  `stream` 只进 digest（`compile.rs:668`）。
- `Intrinsic { node, op: String, group: GroupKind, reduce: Option<ReduceOp>, dim: Option<i64>,
  input: SlotId, output: SlotId, stream, source }` —— 运行期用 `op`/`group`/`reduce`/`dim`/`input`/`output`
  （`runtime/lib.rs:599-624`）。强制单输入单输出（`compile.rs:560-575`）。
- 辅助：`node()`（`:60`）、`source()`（`:66`，explain 用）、`stream()`（`:72`，**无调用者**）、`label()`
  （`:79`，explain 与运行期报错用）。`StreamId = u32`（`compile.rs:22`），取值只有 `MAIN_STREAM = 0` /
  `SIDE_STREAM = 1`（`compile.rs:25-27`）。

**`MemoryPlan` 及子类型**（`memory.rs:54-134`）：`Lifetime { slot, born, dies }`（`:54-59`）——`live_at` 被
`enforce_budget` 用（`memory.rs:338`），`overlaps`（`:66-68`）只有测试读。`Placement`（`:73-84`）四变体
`Persistent{offset}`/`Pool{offset}`/`Aliased(SlotId)`/`NonResident`，全部由 `runtime/lib.rs:362-388` 消费。
`SlotAllocation { slot, bytes, counted_bytes, placement, policy }`（`:87-99`）——`bytes`→`runtime/lib.rs:405`，
`placement`→`:362`，`policy`→`:385`（NonResident 报错路径），`counted_bytes` 只在 `enforce_budget`
（`memory.rs:340`）与 `reuse` 构造（`:519`）里读。`PolicyDecision`（`:105-112`）只被 `memory.rs:162-170`
（explain）读。`MemoryPlan`（`:115-134`）：`persistent_bytes`/`transient_pool_bytes` 是两个 region 的大小
（`runtime/lib.rs:322-323`），`peak_bytes` 被运行期记成 `resident_bytes`（`runtime/lib.rs:411`），
`allocation()` 被运行期查表（`:356`）；`decisions`/`unsupported`/`reuse`/`budget_bytes` 只被 `explain()`、
`enforce_budget()` 与测试读；`lifetimes` 在 `rustrain-plan` 之外**无读者**。

**digest 的具体内容**（`compile.rs:612-716`）。进入的（`DigestInput { plan, decisions, parallel }`，
`compile.rs:696-714`）：(1) **post-propagation 的整份 `Plan`**（`compile.rs:250` 起 `plan` 已被 propagation
结果覆盖）——含 `meta`（name/phase/parallel/**seed**）、每个 `Slot`（name/dtype/shape/layout/kind）、每个
`PlanNode` 的 `op.name`/`op.variant`/`inputs`/`outputs`/**`attrs`**/`phase`/**`precision`**/**`checkpoint`**/
`stream`/**`source`（`Trace.path` 与 `inserted_by`）**；(2) `decisions`（`compile.rs:631-694`）：step 序号、
`op.name()`、`"{plugin}:{spec_name}"`、9 个 numerics 字段、`stream`、输入输出的 usize，intrinsic 端
`implementation` 编成 `"intrinsic:{group}:{reduce:?}:{dim:?}"`（`compile.rs:684-687`）；(3) `Compiler` 的
`parallel`（`compile.rs:314,700`）。**被排除**：recipe 文本（`compile.rs:708` `let _ = recipe;`，注释
`:703-707` 说明 recipe 只通过 decision 间接进入）、**整个 `MemoryPlan`**（不在 `DigestInput` 里，
`compile.rs:696-701`）、`inserted` 的 `reason`/`source`/`produced_slot`/`consumed_slot`（op/group/reduce/dim
经 decision 间接进）、`ResolvedNode.rejected`、`CompiledStep` 的 ABI `attrs` 与 `phase`/`source`
（`Decision` 结构 `compile.rs:631-640` 里没有）。事实后果：改 `Trace.path`、`seed` 或 `checkpoint` 会改
digest 而不改执行；改内存策略不改 digest。

**`InsertedCollective`**（`shard.rs:197-207`）：`reason: String`、`op: &'static str`、`group: GroupKind`、
`reduce: Option<ReduceOp>`、`dim: Option<i64>`、`source: String`（复制生产者 path，`shard.rs:442`）、
`produced_slot`/`consumed_slot: SlotId`。读者：`compile.rs:322`、`compile.rs:160-165`、`cli/main.rs:358`、
测试（`shard.rs:667-686`、`end_to_end.rs:288-293`）。

## 5. intrinsic 词汇表

保留前缀 `intrinsic.`（`ir.rs:468`；命名空间理由见 `ir.rs:461-467`——`broadcast` 同时是原语）。
五个名字 `ALL_REDUCE`/`ALL_GATHER`/`REDUCE_SCATTER`/`BROADCAST`/`SYNC`（`ir.rs:470-474`）；`is_intrinsic`
是**白名单匹配**而非前缀判断（`ir.rs:476-482`）。属性键：`ATTR_GROUP = "group"`（`ir.rs:486`）、
`ATTR_REDUCE = "reduce"`（`:488`）、`ATTR_DIM = "dim"`（`:490`）；`group_name`/`parse_group`
（`ir.rs:492-514`）把 `GroupKind` 与 `"tp"/"cp"/"ep"/"dp"/"pp"/"global"` 双向映射，是**六值封闭映射**。

| 名字 | 编译期读取 | 运行期分派 |
|---|---|---|
| `intrinsic.all_reduce` | `group`（必填 `:534`）、`reduce`（`sum`/`max`/`min`，`:546-558`） | `runtime/lib.rs:601` → `ALL_REDUCE` |
| `intrinsic.all_gather` | `group`、`dim`（`i64`，`:582`） | `runtime/lib.rs:602` → `ALL_GATHER` |
| `intrinsic.reduce_scatter` | `group`、`dim` | `runtime/lib.rs:603-605` → `REDUCE_SCATTER` |
| `intrinsic.broadcast` | `group`（`dim`/`reduce` 允许但无意义） | **not found**：落入 `runtime/lib.rs:606-612` 的 `other` → `RuntimeError::Collective`（"no backend implements this intrinsic"） |
| `intrinsic.sync` | `group`（每个 intrinsic 都必填，`:532-538`） | **not found**：同上 `runtime/lib.rs:606-612` |

插入侧只产出四种：`shard.rs:217-226` 的 `intrinsic_for` 把 `Collective` 映射到 intrinsic；但 `transitions`
从不返回 `Collective::Broadcast`（`collective.rs:42-49` 自述 "the transition rules below never emit it"），
也从不返回 sync。即 `SYNC` 与 `BROADCAST` 只能由描述层手写，且 `BROADCAST` 的 `src_group_index`
（`collective.rs:46-49`）在 `shard.rs:224` 被丢弃 —— IR 里没有对应属性。

## 6. 并行 layout 类型

`ParallelLayout`（`layout.rs:68-91`，封闭枚举 5 变体）：`Replicate`、`Shard { dim: i64, group: GroupKind }`、
`Partial { op: ReduceOp, group: GroupKind }`、`ExpertShard { group }`、`SequenceShard { group }`。
`dim` 是**逻辑维**（可为负），由 `DimNormalizer` 在发转换时解析；`group()` 返回单值 `Option<GroupKind>`
（`layout.rs:101-109`）。`ReduceOp`（`layout.rs:29-38`）只有 `Sum`/`Max`/`Min`。
`GroupKind`（`group.rs:20-35`）封闭 6 值 `Tp`/`Cp`/`Ep`/`Dp`/`Pp`/`Global`，`ALL: [GroupKind; 6]`
（`group.rs:42-49`，注释 "Frozen"），`index()` 映射到固定 6 槽（`group.rs:87-96`）。`ProcessGroups`
（`group.rs:135-143`）是 `groups: [Vec<ProcessGroup>; 6]` 定长数组，由 `ParallelConfig`（五个 `usize`，
`config.rs:68-80`）经 `stride_extent`（`group.rs:287-301`，**五个轴的 stride 硬编码为运行乘积**）派生；
rank 顺序硬编码 `[tp, cp, ep, dp, pp]`、tp 最快（`rank.rs:9-31`）。`Collective`（`collective.rs:22-50`）
四变体：`AllReduce{group, op}`、`AllGather{group, dim}`、`ReduceScatter{group, dim}`、
`Broadcast{group, src_group_index}`。`DimNormalizer`（`layout.rs:137-141`）= `rank` + `sequence_dim`
（默认 `DEFAULT_SEQUENCE_DIM = 1`，`layout.rs:26`），`normalize`（`:189-198`）把负 dim 解析到 `0..rank`，
越界报 `ShardError::DimOutOfRange`；`with_sequence_dim`（`:163-166`）是覆盖序列轴假设的唯一入口；
`EXPERT_DIM = 0`（`layout.rs:15`）硬编码专家轴位置。`transitions(from, to, norm)`
（`collective.rs:168-209`）返回最小有序 `Vec<Collective>`，规则表原文在 `collective.rs:144-160`；
`two_sided`（`:274-325`）在两侧都非 `Replicate` 且 group 不同时直接 `GroupMismatch`（`:283-286`），
`Shard→Partial` 报 `ShardToPartial`（`:319-321`），`Max/Min` partial → shard 报
`ReduceScatterRequiresSum`（`:308-311`），其余未列举组合报 `UnsupportedTransition`（`:324`）。
`RankLayout`（`rank.rs:36-45`）**只被 `rustrain-parallel` 内部与测试使用**：plan 与 runtime 都不构造它。

能表达：单轴分布（沿任一维切一个 group、组内部分和、复制、专家轴切分、序列轴切分）与五类转换
（`Partial→Replicate` = all_reduce、`Shard→Replicate` = all_gather、`Partial(Sum)→Shard` = reduce_scatter、
`Shard(d1)→Shard(d2)` = all_gather + 本地切片、`Replicate→*Shard`/`Replicate→Expert|SequenceShard` 为空转换
即本地 narrow，`collective.rs:194-197`）。**今天不能表达：**

- **一个 slot 只能属于一个 group**：`ParallelLayout::group()` 返回单值（`layout.rs:101-109`），
  `Shard{dim, group}` 只带一个 `GroupKind` → "张量同时沿 tp 与 dp 切分"（tp×dp 组合）无表示。
- **hybrid mesh 无表示**：`GroupKind` 封闭 6 值、`GroupKind::ALL` 与 `ProcessGroups.groups` 都是定长 6
  （`group.rs:42-49,142`）、`stride_extent` 写死五轴 stride（`group.rs:287-301`）→ "tp 与 ep 的乘积轴"
  这类组合组没有名字，也进不了那 6 个槽位；`intrinsic.ATTR_GROUP` 是六值字符串映射
  （`ir.rs:492-514`），一个不透明 axis id 无处安放。
- **跨组重分布不表达**：`two_sided` 在 `g1 != g2` 时报 `GroupMismatch`（`collective.rs:283-286`），
  官方路线先绕 `Replicate`（注释 `collective.rs:133-138`）；多步转换也被拒（`shard.rs:399-411`）。
- **layout 不带轴名**：只有 `dim: i64` + `GroupKind`，`"shard(-1, tp)"` 这种 Display（`layout.rs:117-126`）
  里没有符号轴 / axis id；`DimNormalizer` 由 tensor rank 构造（`shard.rs:366`），序列轴位置靠常量与
  `with_sequence_dim`（`layout.rs:163`）。
- **没有 mesh 对象**：`ParallelConfig` 是五个 `usize` 的 `Copy` 结构（`config.rs:68-80`），不带轴名也不带
  rank 列表，组是派生的（`group.rs:165-200`）。
- **partial 语义受限**：`ReduceOp` 只三种（`layout.rs:29-38`），reduce_scatter 只接受 `Sum`
  （`collective.rs:301-311`）；`ExpertShard`/`SequenceShard` 与 `Shard` 即使 dim 相同也不等价
  （`collective.rs:157-160`），互转报 `UnsupportedTransition`。
- **拓扑可用性无人校验**：`shard::propagate` 收到 `groups` 后丢弃（`shard.rs:236`），`PlanError::GroupUnavailable`
  （`lib.rs:156`）从未被构造 —— plan 可以声明 `GroupKind::Ep` 而拓扑里 `expert = 1` 而不报错。

## 7. 运行期一侧

`Executor`（`runtime/lib.rs:254-268`）消费 `CompiledPlan` 的方式：

**按 `MemoryPlan` 分配**（`Executor::new`，`runtime/lib.rs:277-427`）：先拒绝 `plan.memory.unsupported` 非空
（`:284-290`）；从 steps 里的 `Intrinsic` 重建别名链（`:295-301`，与 `memory.rs:595-608` 的规则重复一次）；
**只开两个 region** —— `persistent_bytes` 与 `transient_pool_bytes`（`:322-323`），零字节 region 跳过
（`:310-312`）。每个 slot 按 `Placement` 取指针：`Persistent{offset}`/`Pool{offset}` 是 `base + offset`
（`:365-372`，planner 的 offset 就是真实地址复用）、`Aliased(root)` 抄 root 的 ptr（`:373-377`）、
`NonResident` 报 `UnsupportedMemoryPolicy`（`:381-387`）。初始 `shape`/`strides` 由 `slot.shape` 行优先算出
（`:390-398`），`elem_width` 取 dtype 宽度（`:404`），`bytes` 取 allocation 的 bytes（`:405`）。

**一步如何执行**（`run`，`runtime/lib.rs:574-712`）：严格按 `plan.steps` 顺序，无调度搜索（对应
`ir.rs:252-256` 的拓扑序约定）；每个 step 先取 in/out descriptor（`:595-596`）。intrinsic 分支
（`:599-624`）：按 `op` 字符串映射到 `RsCollectiveKind`，取 `out_tensors.remove(0)`，调
`CollectiveBackend::execute(kind, group, reduce, dim, &mut t)`。op 分支（`:626-696`）：`op.desc().execute`
（`:634-639`）、`RsCtx { user: null, svc }`（`:641-644`）、失败时用 `desc.last_error` 取消息（`:661-674`）。
`run` 不可续跑（`:569-573`）。

**在什么地方采纳算子返回的 `data`/`shape`/`stride`**（`runtime/lib.rs:688-707`）：op 执行后，对每个
`!t.data.is_null()` 的输出，把 `buf.ptr/shape/strides/rank` 覆盖为算子写回 descriptor 的值
（**不更新 `bytes`/`elem_width`/`dtype`**）。这是 view 算子（`transpose`/`narrow`/`reshape`/`broadcast`）
能工作的唯一原因；`SlotBuffer` 注释（`:220-241`）记录了不采纳时读到的是执行器自己未初始化的 buffer。
读回 `read_raw`（`:525-536`）用采纳后的 shape/strides 走 `materialise`（`:733-774`，能处理 stride=0）；
而 `write_f32`（`:476-492`）与 `slot_len`（`:466-468`）仍按 **plan 的 shape** 期望连续排布。

**collective 步骤怎么拿到 group**：只拿到 `GroupKind` 枚举值（`compile.rs:49`）加 `reduce`/`dim`。
`CollectiveBackend::execute` 的签名（`runtime/lib.rs:170-179`）里**没有 rank、没有 world size、没有组句柄、
没有 plan 引用**；`SingleRank` 只在 `self.world_size == 1` 时成功，否则报错（`:207-213`）。`RsServices` 的
`collective` 回调是 `None`（`:790`），`alloc`/`free`/`current_stream`/`log` 同样是 `None`（`:786-791`），
插件无法自行发集合通信或申请显存。执行器**不读** `Slot.layout`、`CompiledPlan.parallel`/`inserted`/`digest`/
`resolved`，也不读 `CompiledStep::Op.phase`。

## 8. 模型描述必须满足的接口

**必须填的字段**：

1. `PlanMeta { name, phase, parallel }`（`ir.rs:206-214`）—— `parallel` 必须与传给 `Compiler::new` 的
   `ParallelConfig` 是同一份度数；今天**没有校验**，CLI 靠手工传两次（`cli/main.rs:340-343`）。
2. `Slot { name, dtype, shape, layout, kind }`（`ir.rs:47-55`）—— 五个全必填，`shape` 必须是具体 `i64`
   （`ir.rs:6-8` 拒绝符号维度），不填 `layout` 就是 `Replicate`（`ir.rs:370`）。
3. `PlanNode { op, inputs, outputs, attrs, phase }`（`ir.rs:183-197`），节点必须**按拓扑序发射**
   （`ir.rs:252-256,271`），每节点至少一个 output（`ir.rs:295-297`）；intrinsic 节点还须自带 `group` 属性
   （`compile.rs:532-538`），`all_reduce` 带 `reduce`（`:546-558`），gather/scatter 带 `dim`（`:582`）。

**被推导、描述层不填的**：输出 layout（`shard.rs:94-193`）、插入的 collective 与转换后的新 slot
（`shard.rs:413-461`）、内存 lifetime/placement/pool/峰值（`memory.rs:207-310`）、实现 variant 与 numerics
（`compile.rs:334-411,506-521`）、digest（`compile.rs:612-716`）。

**schema 会迫使改动的地方（均为现状事实）**：

- **`GroupKind` 封闭** → hybrid mesh（tp×ep×dp 组合）无表示：六值枚举 + 定长数组 + 六值字符串映射
  （`group.rs:20-49,87-96,142`；`ir.rs:492-514`）。任何"轴由 mesh 决定"的 schema 都必须替换
  `Shard{group: GroupKind}`（`layout.rs:76`）与 `intrinsic.ATTR_GROUP` 的取值域。
- **`SlotKind::State` / `SlotKind::Gradient` 无构造点** → 常驻区（`memory.rs:548-553`）今天只可能出现
  `Weight`；依赖 `State` 的 `shard.rs:547-549` 本身无调用者。
- **`Trace.path` 来自一个不存在的模块树** → 由 `scope()` 前缀 + 调用方字符串拼成（`ir.rs:412-416`），
  插入节点复制生产者 path（`shard.rs:442,458`），且参与 digest（`compile.rs:696-714`）。
- **`PlanMeta.seed` / `PlanNode.checkpoint` 无读者但进 digest**（`ir.rs:213,193`；`compile.rs:696-714`）。
- **`StreamPolicy::Side` 无生产者**（`ir.rs:130`），`CompiledStep.stream` 只进 digest（`compile.rs:668,690`）。
- **`PrecisionOverride` 有读者、无写入者**（唯一入口 `ir.rs:432` 无调用者），逐节点精度今天只能靠 recipe。
- **`intrinsic.sync` / `intrinsic.broadcast` 在运行期没有分派**（`runtime/lib.rs:606-612`），且
  `Broadcast.src_group_index` 在 IR 里不存在（`shard.rs:224`）。
- **`Phase::Backward` 需要显式 backward 实现**：`backward = "autodiff"` 被 `NotValidatable` 拒绝
  （`compile.rs:388-400`）。
- **plan 的持久化入口：not found**，描述层今天唯一的产物通道是 `PlanBuilder` 的内存对象。
