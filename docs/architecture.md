# rustrain 架构

> **状态**：本次重构的架构定义。重构前的设计已作废，归档在 `_internal_docs/archive/pre-rewrite/`。
>
> 这份文件描述**架构是什么**。实现规则（边界契约的操作化、不变式、三类变更 checklist、禁止模式、
> 验证门禁）在 `skills/architecture/SKILL.md` —— **规则只有一份，这里不重复它。**

---

## 0. 边界契约（T1 / T2 / T3）

**T1 实现体自由，T2 声明契约自由，T3 数学形态重编框架。**

| 层 | 变什么 | 代价 |
|---|---|---|
| **T1 实现体** | 同一契约，不同代码：SIMD / CUDA / Tilelang / CUTLASS / fp8 变体 / 融合或拆开 | 换一个 `.so` + 改 recipe。**不重编框架。** |
| **T2 声明契约** | arity、形状规则、dtype 掩码、numerics 与量化方案、expansion、collectives、反向接线 | 换 `.so` + 换 plan。**不重编框架。** |
| **T3 数学形态** | 出现框架没见过的规则种类 | **重编框架，接受** —— 形状都变了。 |

**为什么这个自由值得**：Kernel 的正确性是 **Kernel 的责任，不是框架的责任**。框架只需要能
（a）拿到一个实现、(b) 知道它的契约、(c) 有办法验证它和别人算的是同一件事。在此之上，能**随时装卸
kernel 做对照**，对研究比编译期检查更有价值。

**因此明确放弃**：类型化 / 编译期的接口检查。插件是运行时对象，错的插件是运行时错误。
代价由一致性门禁承担 —— **门禁不是"顺手做的检查"，它是这条边界的承重结构**。

**推论**：

1. 声明错了编译器不会发现，只有门禁会发现 → 任何实现进入 plan 之前必须过门禁。
2. ABI 稳定性是纪律：`struct_size` 前向兼容、字段不得重排、C 枚举用 newtype 包住任意值。
3. digest 必须记插件身份（`plugin@version` + origin），否则同配置不可复现。
4. **规则不得按算子名查框架侧的表** —— 那会把 T2 泄漏成 T3。规则要作为**描述符里的声明**
   （`{kind, 参数}`）出现；新算子只要规则种类已存在就不重编框架。

---

## 1. 核心模型

### 1.1 三层产物：描述 → 编译 → plan

```
模型描述（数据，拓扑无关，可移植）
      │   × 拓扑（编译输入）
      ▼
   编译（纯 CPU，位于 launch 路径）
      ▼
   plan（具体：形状、layout、axis id、节点集合）   ← 每个 rank 一份
```

| 事实 | 归属 | 求值时机 |
|---|---|---|
| mesh：**有序**轴名 → degree | 编译输入 + 运行输入（`Mesh`；plan 只存 `MeshFingerprint`）。**不进 plan。** | 编译期 |
| 张量轴 → mesh 轴的**指派** | 模型描述（符号：**多个 `(dim, group)` 分片 + 至多一个 partial**） | 编译期求值 |
| 具体 layout（含度数）、axis id、形状、节点集合 | **plan 产物** | 编译后 |
| topology **指纹**（不是对象） | plan 的 meta / digest | 编译后 |

**Plan 里没有"topology"这个概念，只有它的结果。** 说不出"谁遍历它"的东西不进设计，拓扑对象没有消费者。
只有三个地方用到拓扑，各自的归属必须分清 —— 这是 topology 溜进 plan 的唯一通道：

- **collective 执行**：要组句柄 → plan 存 **axis id**，runtime 拿它在 mesh 里查句柄。
- **加载器**：要"我持有哪一片" → 由 (slot.layout, 组内 rank) 算出；mesh 是运行输入，
  所以需要拓扑的是 loader，不是 plan。
- **PP 实例化**：节点集合是拓扑的函数 → **编译期求值**，产物里节点集合已经具体。
  （**只有 PP**；EP 是 layout + 显式 routing，见 §1.4。）

**推论（重要）**：**描述是可移植产物，plan 是拓扑相关的产品。**
因此 checkpoint 映射挂在**描述**上（全局参数空间），不挂在 plan 上（局部）。
副产品：§4.4 的 L2 检查**完全不需要 topology**，只有 L1 需要。

**这枚钉子已经拆掉（D3，2026-09）**：`GroupKind` 那个封闭枚举（Tp/Dp/Pp/Ep）表达不了 hybrid mesh
（HSDP、tp×ep×dp 组合），现在换成了 **`GroupMask`（轴掩码，可表达 `tp|ep` 这类组合组）**。
掩码是结果、不是拓扑对象：名字与 degree 留在 `Mesh` 里，plan 只带 `MeshFingerprint`
（`docs/design/model-description.md` §1）。

### 1.2 模型是数据

- **结构**（哪些算子、什么顺序、怎么连）→ 描述文件里的**子图模板 + 重复**。
- **实现**（这个算子用哪个 kernel）→ recipe。
- **`expansion`（契约 R-4）** 只承载"融合实现 ↔ 原语分解"的**实现**语义，**不承载结构** ——
  否则"这条边是融合还是展开"会和"模型是什么"纠缠在一起。

框架**不提供任何模型模板**（框架不认识任何模型）：模板住在模型目录里，与 config / checkpoint 同处。
于是"支持一个模型" = 一份描述 +（必要时）原语 kernel；只用现有原语时**零 Rust**。

描述的四个部分：

| 部分 | 内容 |
|---|---|
| 参数 | 从 `config.json` 取；描述引用参数名，不重复数值 |
| 模板 | 子图（一层、一个 attention、一个 MLP），用参数名作形状 |
| 实例化 | **按下标取列表**；标量 → 结构的派生由生成器 materialize 成显式列表（§5.2） |
| 参数映射 | slot ↔ checkpoint 名字 + 变换 + **切分轴**（§1.4），直接喂 L2 |

**判据**：描述格式完成的标志是它能**把第一个验证样本（`Qwen/Qwen3.6-35B-A3B`）完整表达成数据**，
并且**不需要为别的模型改语言** —— 收窄的是验证范围，不是设计。
具体语法与展开语义见 `docs/design/model-description.md`（D8）。

### 1.3 切分不是一个算子

切分不产生运行时计算，因此**不进入计算图**。写成算子（`shard(x, dim, group) -> x_local`）有三个硬后果：
图变成"按 rank 切过的程序"（不再描述模型本身）；layout 从**声明**降级为算子的**输出**（传播的输入没了）；
权重切分表达不了（它发生在 step 0 之前，由加载器决定读哪一块）→ 于是出现两套切分机制。

正确形状：**切分是 slot 的属性，它在图里唯一的可见后果是一个集合通信节点。**

**三样不许混**：

| 概念 | 载体 | 是什么 |
|---|---|---|
| 度数（tp=8 / ep=2） | `Mesh`（由 `ParallelConfig` 建出） | 编译输入；进程生命周期常量 |
| 轴（这条边属于哪条 mesh 轴） | axis id + 节点属性 `ATTR_GROUP` | **节点属性**，不是 tensor operand |
| layout（哪个张量轴被切） | **多个 `(dim, group)` 分片 + 至多一个 partial**（`docs/design/model-description.md` §2.1） | **slot 声明** |

**group 不做 tensor operand 的三条理由**：

1. 它是常量 —— 做成 operand 等于把常量塞进数据流，并让图的**拓扑**变成数据依赖
   （"这条边要不要通信"运行时才知道），AOT 的扁平 step 列表不成立。
2. 它必须一致 —— tensor 值可以在 rank 间不一致；属性在 startup 校验一次。
3. 它不需要"被计算" —— 由 (mesh 拓扑, 轴) 唯一决定，是查表，不是 kernel。

### 1.4 五轴 = 一种 layout 机制 + 一种 PP 机制

| | 切什么 | 机制 | 与 Kernel 的关系 | 图中的体现 |
|---|---|---|---|---|
| **layout**（TP / CP / DP / **EP**） | 同一批节点的**张量轴** | 形状算术 + 通信插入 | **相关**：输入输出形状与是否通信变了 | 节点集合不变，多出通信节点 |
| **instantiation**（**只有 PP**） | **节点集合本身**（本 rank 有哪几层） | 按 (rank, 度数) 枚举模板实例 | **无关**：kernel 代码不会因 pp=4 而改变 | 节点集合随 rank 变 |

**EP 属于 layout，不属于 instantiation**（这一条推翻过早期分类，以 Qwen3.6 MoE 走通之后修正，
推导见 `docs/design/model-description.md` §6.1）：

- **专家权重就是 dim 0 的分片**：`experts.gate_up_proj [E, 2I, H]` 在 `ep=4` 下本地形状 `[E/4, 2I, H]`
  —— 纯形状算术，没有"哪些节点存在"的问题。
- **激活侧的 routing 是数据依赖的**（token 去哪个 rank 由 router 的 top-k 决定），因此它推不出来，
  必须是图中的显式算子（dispatch / combine），由 kernel 声明 `collectives: [all_to_all(ep)]`。
- 于是 **五轴里只有 PP 改变节点集合**，也因此只有 PP 需要微批与 send/recv 调度 —— 那是独立子系统。

对 PP 放弃"一个 plan 跑所有度数"：**同一份描述 + 不同切分参数 → 各实例化一个 plan，
且实例化可在无 GPU 机器上完成**（§4.4）。

**切分轴属于参数声明**：一个权重 slot 的 layout（**多个 `(dim, group)` 分片 + 至多一个 partial**，
`docs/design/model-description.md` §2.1）与"从 checkpoint 取哪一块"是**同一条事实**，
必须住在同一个地方（参数映射）—— 一份声明同时被加载器和形状算术读取。
按算子名或张量名查框架侧的表是禁止的（P6）。

**DP 的梯度归约组是布局的函数**：

```
grad_reduce_mask(param) = 全掩码 \ (该张量已切分的轴 ∪ {pp})
```

被切分的维度上每个 rank 持有的是**不同的参数**，跨它们求和是错的；只有复制出来的那份需要跨副本求和。
于是"哪些参数要 all-reduce、在哪个组上"**不需要任何额外标记位，也不需要训练循环手写** ——
它是 `axes` 声明的又一个后果。实测对照：Megatron 用逐参数的 `allreduce` 布尔标记表达同一件事
（`tensor_parallel/layers.py:928, 1278`），专家参数被分流到 expert-DP 组
（`distributed/distributed_data_parallel.py:219-223`）。DP 用 all-reduce 还是 reduce-scatter 取决于
优化器是否分片（`param_and_grad_buffer.py:761-783`），属于 D5。

### 1.5 推导的口径：兑现义务，不猜声明

- **可以推**：某 slot 声明沿 dim 0、组 `{tp}` 的分片 → 每个 rank 只有部分和 → 该处必须 all_reduce。
  这是**把声明的后果算出来**。
- **不可以推**：看到算子名叫 `linear` 就假定权重是 `[K,N]`，看到 `qkv` 就假定 column parallel。这是**猜**。

**推导的输入必须是描述符，不是名字。** 现有违规：`shard::rule_for(op: &str)`（P6）。

### 1.6 度数：编译输入，不是运行期参数

- **方案 A（符号 / 晚绑定）**：plan 里只有轴名，startup 时把度数代进去；一个产物跨度数复用。
- **方案 B（度数作为编译输入）—— 采用**：每次启动重新编译（编译是纯 CPU 的，本就在 launch 路径上）。
  编译器**内部**用符号算术（`local = global[dim] / N`），**产物里只有具体值**。

B 与"非解释"一致且机制更少：产物里没有"度"的痕迹，扁平循环，每步开销与 tp=1 相同；
`N ∤ shape[dim]` 在编译期就是错误，不是运行期 fallback。A 的唯一收益（一个产物跨度数）在训练里用不到，
而"同一份描述在不同度数下各产出一个 plan 再 diff"更便宜、也更好调试。

现状已经长成 B 的形状（`shard::propagate(plan, &groups)` 收度数），缺的是 `let _ = groups;`。

### 1.7 状态

训练里 **KV cache 不是必需的**（那是推理期的东西），§5.2 的实测也证实旧实现没有 KV cache。
训练中真正需要管理的状态是三类：**激活**（由 memory plan 管）、**optimizer state**（缺失）、
**循环 / 增量层的 state**（旧实现是 per-forward scratch）。`SlotKind::State` 与持久化策略见 §8 D10。

---

## 2. 算子与 Kernel 契约

### 2.1 "Kernel 与拓扑无关"的准确含义

kernel **不持有 mesh、不按度数分支**。它持有的只有三样：

1. **本地张量** —— 本地形状已经在张量里，所以多数 kernel 不需要任何拓扑信息
   （row/column parallel linear 都只做一个 local GEMM）。
2. **描述符里声明的轴 id** —— "我吸收 `axis='tp'` 上的 all_reduce"。
3. **执行期的组句柄** —— runtime 绑定，进程生命周期不变。

### 2.2 框架注入什么

| 注入物 | 时机 | 例子 |
|---|---|---|
| **静态常量** | 编译 / 实例化期，烤进 plan 或描述符 | 本地 expert 范围、vocab shard 偏移、stage 索引 |
| **组句柄**（opaque communicator） | 启动期绑定一次，进程内不变 | `axis("tp")` → 该轴的 communicator |
| ~~每步的 TP 参数~~ | ✗ | 需要它 = plan 没定下来 |

- **禁**：给 kernel 传独立的度数标量（`tp_size: usize`）。它是同一事实的第二个来源，一定会和句柄漂移。
  度数从 communicator 读。
- 看起来像拓扑、其实是数据的例子：**SP 的序列偏移** —— 它是输入数据（`position_ids` 按本地 chunk 生成），
  不是拓扑参数。这类东西最容易被误诊成"要注入的 TP 参数"。
- **白拿的对照能力**：`tp=1` 就是 size=1 的组，all_reduce 是恒等运算。**同一个 kernel 在 tp=1 直接能跑**，
  于是"装卸 TP 做对照"不需要两套代码 —— 这正是 §0 要的自由。

### 2.3 planner 只规划 expansion，不规划融合体

> **planner 永远规划 primitive expansion。融合是解析期的一次替换，只有当门禁证明"融合体 ≡ 它的 expansion"
> 时才合法。**

**替换点是"模板实例"**（精确化见 `docs/design/op-vocabulary.md` §8.1）：描述产生的是一张细粒度图，
融合就是把某个模板实例的整段子图**按名字**换成一个算子节点。不做结构模式匹配 —— 那要求框架"认识"某种
子图形状，是 T3 泄漏。于是粒度可以是每模板实例一个开关：原语 → 块 → 层 → （显式加 model 模板的）整模型，
**同一份 plan 换 recipe 就能做融合/展开的对照**。

由此，"这个 kernel 能不能被拆"有了确定答案 —— **不由框架猜，由声明决定**：

- **有 `expansion`** → 框架**总是**能规划它的原语分解。问题不是"能不能拆"，而是"拆了是否更慢"
  （recipe 的性能取舍，不是正确性问题）。
- **融合体声明的 `collectives` 必须与 planner 在它的 expansion 上决定插入的 collectives 集合相等**；
  不等 → **拒绝这次融合**，回落到分解形式（不是报错）。这是**机械校验**，不是猜。

所以"通信写在 kernel 里"不是问题；**问题是写在里面而没人知道**。

### 2.4 EXPLICIT 算子

做的事不在原语词表里时，必须声明：(a) 布局规则的**种类**，(b) 它内部吸收的 collectives。
规则种类已存在 → T2；需要新种类 → T3（应罕见，且应被注意到）。

### 2.5 设备纪律

**插件在 `init()` 之前不得碰设备。** 否则 `dlopen` 会把 CUDA 上下文拉起来，无 GPU check 就真的"执行"了。
`Plugin::load` 已经保证"校验通过之后才 `init`"（被拒的插件不执行任何代码）；这条纪律是它的另一半。

### 2.6 算子声明的显存：`workspace_bytes` 与 `save_for_backward_bytes`

`memory()` 回调填 `RsMemReq`（`ffi.rs:411-419`、`rustrain_op.h:139-144`）：

| 字段 | 语义 | 生存期 | 今天 |
|---|---|---|---|
| `workspace_bytes` | **这次调用期间**要的临时空间 | 调用结束即释放 | ✅ planner 计入峰值（`memory.rs:648`） |
| `save_for_backward_bytes` | **调用结束之后仍然持有**、直到反向才释放的字节 | 前向 → 反向，全程占着 | ❌ **死钩子**：ABI 里有，planner 从不累加 |
| `save_tensor_count` | 持有了几个张量（bookkeeping 与粒度） | 同上 | ❌ 同上 |

**为什么必须声明**：plan 的寿命分析**只看得到 slot**。普通算子的反向输入就是它的输入 slot，planner 看得见；
但**融合算子内部的中间量不是 slot** —— 它要留到反向，planner 一无所知。于是投影峰值偏低 ——
**告警本身就不可信**，而真实占用要到训练时才以 OOM 的形式暴露。

**为什么只能声明**：留多少是**实现选择**（T1）—— 参考实现可以什么都不留（反向重算），调优实现可能全留。
同一个算子的两个 variant 可以声明不同的值，所以它属于描述符，不能是框架假设。

**它是另一半的对手**：`PlanNode.checkpoint: CheckpointPolicy{None|Recompute|Offload}` 也是死钩子。
两个钩子其实是**同一个缺失子系统**的两半 —— 反向激活的显存管理：`save_for_backward_bytes` 是代价，
`CheckpointPolicy` 是杠杆。旧框架手写了这套东西（激活卸载到 CPU pinned、子层 checkpoint、手工 sequential
checkpoint），所以它确实需要，且不是新需求。

**落地时框架该做什么**：executor **预留**声明的字节；kernel 实际保存超过声明 → 在那个算子处分配失败，
而不是静默 OOM。声明错仍只能靠"跑起来"发现 —— **门禁证的是数值等价，不是内存**。

**但这一条现在不做**（§8 D12）：内存管理整个留空。**目标行为**是预算只警告、不拦编译；但**代码里
`enforce_budget` 目前仍是硬失败**（`memory.rs` 返回 `MemoryBudgetExceeded`），这一项由
`docs/design/qwen36-text/spec.md` 的 **D4** 交付。好消息是这两件事不冲突 —— 预算一旦不再拦编译，
**融合实验就不会被内存投影阻塞**，等做内存管理时再激活这两个钩子。

---

## 3. 计算路径

从"一份模型描述"到"一个 kernel 被执行"。

```
入口
 │
 ├─ 模型描述（数据）        ← 待建。参数 + 模板 + 实例化 + 参数映射
 │    展开 expand()          → 一份（拓扑无关的）结构化模型
 │    × 拓扑                 → PlanBuilder 调用序列
 │
 ├─ Plan            slots[] + nodes[]；每个 slot 带 dtype / shape / ParallelLayout
 │
 ├─ Compiler::compile
 │    1 check_structure      节点拓扑序、槽位唯一写者
 │    2 shard::propagate     按描述符规则校验/推导每个算子输入输出所需的布局，
 │     │                      在布局不匹配处**插入** intrinsic 集合通信节点
 │    3 resolve_node ×N      Registry + Recipe → 每个节点的具体实现
 │     │                      融合体在此替换其 expansion；替换前校验 collectives 集合相等（§2.3）
 │    4 memory::plan         寿命分析 → 偏移复用 → 峰值投影；调每个算子的 memory() 取 workspace
 │    5 enforce_budget       峰值**投影**超 budget_bytes → **目标**只 Warning；**今天仍硬失败**（§8 D12，D4 交付）
 │    6 validate_shapes      调插件的 infer()，与 plan 声明的形状比对
 │    7 compute_digest       把全部决策哈希（不含 recipe 原文，只含它产生的决策）
 │
 ├─ CompiledPlan    steps[]（扁平调用序列）+ digest + memory + inserted[]
 │
 ├─ Executor::new   按 MemoryPlan 分配**两块区域**（常驻 + 激活池），槽位按规划偏移落位
 │
 └─ Executor::run   按序走 steps
      ├─ Op       组 RsTensor 描述符 → desc.execute()；随后**采纳**算子返回的 data/shape/stride
      │             （view 算子会把 out.data 指回输入）
      └─ Intrinsic  CollectiveBackend::execute
```

**当前的真实边界**：这条路径**到 loss 为止**。反向图、优化器步、训练循环都不存在。
CLI 只走到 `Compiler::compile`（为了 `plan explain`），执行器只在测试与门禁里被驱动。

---

## 4. 加载路径

### 4.1 插件（kernel）加载 —— 已实现

```
插件 .so（只导出 rustrain_plugin_v1）
 │
 ├─ Plugin::load(path)          dlopen + dlsym
 │    1 版本协商                abi_version 必须相等；struct_size 不得小于本构建
 │    2 遍历并校验每个算子描述符  缺 execute / 空 expansion / EXPLICIT 但无反向接线 → 拒绝
 │    3 init(services)          **校验通过之后**才调用（被拒的插件不执行任何代码）
 │
 ├─ Registry::add_plugin        op@variant 全进程唯一；冲突报错并指出两个来源
 │
 ├─ Recipe::resolve             prefer 绝对（不存在/能力不满足 → 报错，不回落）
 │                              fallback 链（逐项记录被跳过的原因）
 │                              default provider（仅当两者都没给）
 │                              strict = true 时任何降级都拒绝
 │
 └─ digest                     记录 plugin@version 与 origin
```

**进程内插件**同一条路径：`Plugin::from_static` 跳过 dlopen，其余校验完全一致。

### 4.2 权重 / 状态加载 —— 缺失

没有权重加载、没有 checkpoint/resume、没有 `SlotKind::State` 的管理。
模型权重今天只是 plan 里的一个 slot，没有来源。
参数映射（slot ↔ checkpoint 名字 + 变换 + 切分轴）挂在**描述**上（§1.1）。

L2 的变换词表最小集（由真实 checkpoint 反推，不是想出来的）：`take(name)` / `slice(dim, range)` /
`transpose` / `split` / `concat(dim)`。Megatron 用到的全部机制都落在这几种里 —— 前缀重命名
（`sharded_state_dict_keys_map`）、按轴切片（`{"weight": 0}`）、MLA 的 `torch.cat([q, kv], dim=0)`。

### 4.3 插件发现 —— 缺失

今天必须显式 `--plugin <path>`。没有扫描目录、没有 manifests、没有版本约束求解。

### 4.4 无 GPU check 阶梯

**它不实际执行计算**，是一种**声明一致性检查**。这是"支持一个模型"的主循环。

| 级 | 需要什么 | 查什么 | 现状 |
|---|---|---|---|
| **L1 结构** | 无（零设备、零权重） | plan 可编译；每个节点的算子可解析到实现；每个算子的 `infer()` 与声明形状一致；layout 传播完成、每个 `Partial` 都被兑现、每个 collective 都绑了轴；每个 slot 都有分配、无别名冲突（**内存预算只报告，不拦**，D12） | 能力已具备（`validate_shapes`），**缺驱动**：今天跑的是 CLI 里手写的 demo plan，不是"指定 model 路径" |
| **L2 加载** | checkpoint 的 **metadata**，不读数据 | 每个 slot 都能从某个 checkpoint tensor 得到（名字 + 变换）；每个 checkpoint tensor 要么被消费、要么显式声明忽略；dtype / shape / 切片范围一致 | **缺失**。safetensors 头部即 JSON（名字 / dtype / shape / offset），读它不需要读权重。实测：4 GB / 1386 tensor 的 checkpoint，头部 198 KB |
| **L3 数值** | 设备（或 CPU 参考实现） | 同一算子的两个实现算同一件事 + 数值参考 | 已有（conformance gate） |

```
rustrain check --model <model-dir> [--checkpoint <dir>] [--tp N --cp N --ep N --dp N --pp N] [--json]
```

- **插件发现尚未实现**，当前靠 `--plugin <path>` 显式指定（§4.3）。
- 退出码非零即失败；每条 skip 必须写原因（沿用门禁纪律）。
- `--json` 的用途是**机器消费**：L2 的自然用户是从 HF 模型"拆碎"出描述 + 名字映射的生成器，
  `decompose → check → 修映射 → check` 全在笔记本 CPU 上跑。
- 顺带可查**算子覆盖**：模型需要的算子本机有没有实现、有没有目标精度的变体
  （`uncovered_operators()` 是种子）。

**它保证什么**（声明之间自洽）：每个节点都有实现（dtype / layout / target 满足）；每个边界的形状 /
strides / dtype 一致；每个 buffer 都被分配、无别名冲突；每个 collective 都有组
（"组在拓扑里存在"**自 D3 起是真的**：掩码要对着 plan 自己的 mesh 指纹校验，越界的组位报 `GroupUnavailable`）；
每个 slot 都有来源（L2）；每个算子有反向接线或可推导。

**它不保证什么**：**数值**（NaN / Inf / 精度 / 发散 —— 那是 kernel 的责任）；**模型是对的**
（形状全对而数学错，§5.2 有实物反例）；性能；确定性（那是门禁的另一条轴）。

**它不是类型论意义的类型检查**：声明是**不可信输入**（T2 自由的代价）。它证明的是"声明彼此自洽"，
不是"声明是真的" —— 后者只有 L3（两个实现 + 一个数值参考）能证。

---

## 5. 先例与证据

### 5.1 外部（实测，非转述）

**Megatron-LM**：层结构基本是**代码**（`ModuleSpec` 持有 Python 类对象，`spec_utils.py:29-31, 99-122`）。
数据只覆盖"哪一层用哪套参数"（`heterogeneous_config.py:157-179`）与"哪一层在第几个 stage"
（字符串 DSL `'Et*3|(tt|)*29,m|L'`，`transformer_config.py:108-128`）；**子模块顺序与残差接线永远是代码**
（`transformer_layer.py:362-460, 860-868`）。checkpoint → 参数是名字字符串 + 每个类自己的
`sharded_state_dict()` + 前缀重命名表（`gpt_layer_specs.py:484-487`）。
**切分轴是类里的字面 dict**：`ColumnParallelLinear` → `{"weight": 0, "bias": 0}`（`layers.py:1116-1126`）、
`RowParallelLinear` → `{"weight": 1}`（`:1379-1389`）—— 注意它在 `sharded_state_dict` 里，也就是**加载侧**。

**HuggingFace**：`layer_types` / `full_attention_interval` / `sliding_window_pattern` 是数据，
但 `layer_types` 是在 config 代码里**从一个标量生成**的；类别分派（`if block_type == "linear_attention"`）、
合法词表、子模块顺序、残差接线全是代码。GLM-4.6 两个字段都没有。

**vLLM / SGLang / JAX MaxText / torchtitan：没有一个把图本身表达成数据。** 最接近的四例都停在图的门口：

- vLLM 有 `splitting_ops: list[str]`，但那是"在哪些算子上切"；pass 流水线是代码，最后生成 Python 源码字符串 `exec` 掉。
- SGLang 手里是运行时的 `torch.cuda.CUDAGraph` 对象。
- MaxText 是 `layer_map = {DecoderBlockType.DEFAULT: [NNXDecoderLayer], ...}` —— 配置枚举选 Python 类。
- torchtitan 的 `Config(layers=[...])` 只描述**构造**；算子顺序与连接写死在 `forward()` 里。
  （`experiments/graph_trainer` 序列化 FX trace，但那是**派生的缓存**，不是被编写的源头。）

**根因**：它们的图就是 `forward()` —— 图存在，但不是一等产物，是 Python 控制流，没有东西可以序列化。
**它们的 `forward()` 就是我们的 `Plan{slots,nodes}`。**

**风险（要正视）**：据我们所知，这是第一个把图当作**编写对象**的框架。写错的图**编译不会报错**，
而且没有现成工具能替我们验证它。因此：

> **§4.4 的 check 阶梯不是便利功能，它是"模型即数据"的前置条件。**

### 5.2 我们自己的 legacy（`archive/pre-rewrite-20260803` 实测）

- **层图有两个来源**：同一层的算子顺序与残差布局在 Rust（`rustrain-qwen3-6/src/model.rs:664-692`）
  与 C++（`kernels/qwen3_6_kernels.cpp:777-853`）**各写一遍**。这不是"代码没整理好"，
  而是**结构没有数据来源**的必然后果。
- **切分轴按张量名硬编码**：`rustrain-glm5/src/tp_cp.rs:240, 264, 495-515, 650-666` ——
  column-parallel 取 `shape[0]`、row-parallel 取 `shape[1]`、`q_b_proj` 取 dim0 的
  `head_start*(qk_nope+qk_rope)`、`o_proj` 取 dim1。这正是 P6 要消灭的形态，
  也是 §1.4"切分轴属于参数声明"的实物依据。
- **配置解析了但从不读**：`full_attention_interval`、`mrope_*`、`attn_output_gate`、
  `shared_expert_intermediate_size`、`hidden_act`、`attention_bias` …（`Q/config.rs` 有定义，
  模型代码零引用）。**死钩子的实物**：声明了却没人读，等于没有声明，而且比没有更糟 ——
  它看起来像支持。
- **两种 layer-kind 表达形式并存**：Qwen3.6 是**纯列表**（`layer_types`，代码里没有取模路径，
  `full_attention_interval` 解析后不使用）；GLM5 的 indexer kind 是**列表 + 取模回退**
  （`glm5/src/model.rs:342-343, 361-363`）。**验证阶段只看前者**，但语言不需要为后者改：
  派生一律由生成器 materialize 成显式列表（§1.2 的"四部分"里 params 只做取值与算术表达式）。
- **状态**：Qwen3.6 **没有 KV cache**；delta-rule 的 state 是 per-forward scratch
  （`Q/model.rs:368`）；唯一的跨层状态是 GLM5 的 `IndexShareState`（`glm5/src/model.rs:810-819`）。
- **并行实现极不对称**：Qwen3.6 只有 EP（把 `WORLD_SIZE` 当 EP 用）；GLM5 有 TP + EP + CP；
  **两者都没有 PP**。

---

## 6. crate 职责与依赖

### 6.1 实际依赖图（`cargo tree` 实测）

```
rustrain-parallel     （无内部依赖）
rustrain-abi          （无内部依赖）
     ▲
     ├── rustrain-ops            → abi
     │        ▲
     │        ├── rustrain-plan  → abi, ops, parallel
     │        │        ▲
     │        │        └── rustrain-runtime → abi, ops, parallel, plan
     │        │                     ▲
     │        └── rustrain-kernels → abi          （**插件，不是框架的一部分**）
     │                             ▲
     └── rustrain-cli  → abi, ops, parallel, plan, runtime, kernels   （组合根）
```

### 6.2 职责

| crate | 职责 | 明确不负责 |
|---|---|---|
| `rustrain-abi` | ABI v1：POD 描述符、插件入口、装载、插件作者辅助 | 不依赖 tch/CUDA；不认识具体算子 |
| `rustrain-ops` | 算子字典 + recipe 解析 | 不懂图、不懂调度 |
| `rustrain-parallel` | 进程组、rank 布局、切分规格与转换规则 | 不懂算子、不懂 plan |
| `rustrain-plan` | Plan IR、切分校验与通信插入、显存规划、校验、编译、digest | 不执行 |
| `rustrain-runtime` | 执行器、显存池、collective 后端、一致性门禁 | 不决定什么该跑 |
| `rustrain-kernels` | reference provider（语义真值） | **不是框架**，可整体替换 |
| `rustrain-cli` | 组合根 | 不含业务逻辑 |

**不变式 I-1**：`abi / ops / parallel / plan / runtime` 的依赖闭包中不得出现 tch、libtorch、cuda。
实测为 0。这是"核心能在无 GPU 机器上跑完整测试"的依据，也是 §4.4 check 能在无设备机器上跑的前提。

### 6.3 已知的分层偏差

| # | 问题 | 修法 |
|---|---|---|
| P1 | `Recipe` 是跨层策略（算子/精度/并行/显存）却住在 `rustrain-ops`（算子字典） | 独立 `rustrain-recipe`，或提到组合根 |
| P2 | `rustrain-abi` 把宿主侧与插件侧捆在一起 —— 插件因此依赖它永不需要的 `libloading` | 拆出 `rustrain-loader` |
| P4 | 没有训练层：无 train / data / checkpoint / manifest | 新增 `rustrain-train`、`rustrain-data` |
| P5 | `rustrain-kernels` 同时是"语义真值（必须纯）"和 aten provider 的预定住址（必须链 libtorch） | 拆成 `-reference` 与 `-aten` |
| P6 | `shard::rule_for(op: &str)` 按算子名查表，把 T2 泄漏成 T3 | 规则进描述符，作为 `{kind, 参数}` 声明 |
| P7 | 模型表达层不存在 —— 78 层模型按今天的写法要手工标注每张量的布局 | 模型描述 + 结构化模板（§1.2） |

P2/P6 直接服务 §0 的边界契约，优先级高于 P1/P4/P5。

---

## 7. 缺口（重构输入）

**与 §1 / §2 定义的差距**（逐条实测见 `docs/design/plan-ir-baseline.md`）

**死钩子的完整清单见 `docs/design/plan-ir-baseline.md`** —— 本节只列与 §1 / §2 差距直接相关的那些，不复制全表。

- ~~`shard::propagate` 收到 `ProcessGroups` 后丢弃 → `GroupUnavailable` 从未被构造~~ **D3 已闭合**：
  `propagate` 现在拿 plan 的 mesh 指纹校验每个 layout 的掩码与维度，越界即报错并点名节点 / 算子 / 掩码。
- `CollectiveBackend::execute` 的签名里没有 rank / world size / 组句柄（`runtime/lib.rs:170-179`）→
  **§2.2 说的"执行期由 runtime 绑定句柄"今天没有通道**；`intrinsic.sync` 与 `intrinsic.broadcast`
  编译期被接受、运行期落空。
- digest 把 `seed` / `checkpoint` / `Trace.path`（后者只是诊断字符串）纳入，却把**整个 `MemoryPlan`** 排除
  （`compile.rs:696-716`）→ 改诊断路径会改 digest，改内存策略不会。与 §0 推论 3 的意图相反。
- ~~`GroupKind` 是封闭六值枚举 + `[_; 6]` 定长数组~~ **D3 已完成**：`Mesh` + `GroupMask` + 多分片
  `ParallelLayout`，旧算术有逐 rank 对拍测试。剩下的是 D4 的 `instantiate`。
- **`SlotKind::State` / `Gradient` 无任何构造点（`memory.rs:551` 会读）→ §1.7 的状态管理今天没有承载。**
- **`RsMemReq.save_for_backward_bytes` 是死钩子**（ABI 里有，`ffi.rs:418`；只有测试读，
  `memory.rs:644` 构造后从不累加）→ 融合 kernel 自己保存的激活不计入预算，**激活峰值被低估**。
  这是"支持任意粒度融合（含 Megakernel）"的前置条件，见 `docs/design/op-vocabulary.md` §8.3。
- `PlanMeta.parallel` 在 plan crate 内无读者，CLI 靠手工传两次（`cli/main.rs:340-343`）→
  §1.1 的"描述 × 拓扑"没有单一入口。
- 没有 plan 的持久化入口（`Plan` 派生 `Serialize` 但全仓无读写路径）→ 描述层的产物今天只能走内存对象。

**计算路径**
- 模型描述与展开（§1.2）—— 设计已完成（`docs/design/model-description.md`），待实现；执行入口 `docs/design/qwen36-text/spec.md`
- 反向图（`derive_backward`）—— 设计已定（spec §2.11），未实现
- 优化器步、梯度累积、微批调度
- collective 的流分配与 overlap 调度（`StreamPolicy::Side` 已存在但无消费者）
- 训练循环本身
- 融合替换的合法性检查（§2.3）—— 需要 ABI 的 `collectives` 语义明确

**加载路径**
- 权重加载（safetensors → slot）
- L2 加载检查（§4.4）
- checkpoint / resume，以及 `SlotKind::State` 的持久化（§1.7）
- 插件的发现与版本约束（现在靠手写 `--plugin`）
- 一个正式的插件 SDK：现在"新增一个实现"没有模板，只有散在测试里的样例

**模型面**
- 模型描述格式：结构由数据承载（§1.2 / D8）
- ~~`GroupKind` → `GroupMask`（轴掩码，§1.1 / D9）~~ **已完成（D3）**
- 模块树是否必要：今天它只贡献 `Trace.path` 这一个可读字符串，没有任何东西**遍历**它。
  只有当某个消费者必须走树时（checkpoint 映射、显式 optimizer 分区、state 管理）才值得引入

---

## 8. 待定

| # | 决定 | 状态 |
|---|---|---|
| D2 | 模型描述形态 + 切分谁决定 | **已定**：结构是数据（子图模板 + 重复），插件只提供原语，`expansion` 只承载实现语义（§1.2） |
| D4 | VJP 规则表归属：框架侧 / 描述符 / 两者 | 未定；倾向描述符（与 P6 同一理由） |
| D3 | recipe 作用域：算子名 vs 描述里的结构路径 | 依赖 D8 |
| D5 | 训练循环归属、微批与梯度累积对 plan 的影响 | 未定 |
| D6 | P1/P2/P4/P5/P6/P7 的落地顺序 | 待 D8 定 |
| D8 | **描述文件的具体语法与展开语义**（重复、逐层覆盖、按名接线；必须容纳 §5.2 的两种形式） | **已设计完成**：`docs/design/model-description.md` §3–§4；执行入口是 `docs/design/qwen36-text/spec.md` |
| D9 | `GroupKind` → `GroupMask`（轴掩码，可表达 `tp\|ep` 这类组合组）的迁移 | **已落地（D3）**：`Mesh` / `GroupMask` / 多分片 `ParallelLayout` / 形状算术；plan 带 `MeshFingerprint`、`ATTR_GROUP` 变成掩码整数、`GroupUnavailable` 成为真实错误。设计见 `docs/design/model-description.md` §1–§2，证据见 `docs/design/qwen36-text/spec.md` D3。**ABI 面未动**：`RsGroupKind` 是插件的**能力声明**（"我要在哪些轴上做集合通信"），与 plan 的 layout 掩码是两个概念；若将来要统一，那是 ABI v2 |
| D10 | `SlotKind::State` 与持久化：激活 / optimizer state / 循环层 state | 未定 |
| D11 | 融合替换的合法性检查落地（collectives 集合相等） | 未定 |
| D7 | `rustrain check` 的层级划分与 L1/L2 落地顺序（§4.4） | 依赖 D8 |
| D12 | **内存管理模型** | **留空（用户决定）**：峰值只报 Warning，不拦编译；未来按 vLLM 的方式做 —— 声明式的 `gpu_memory_utilization` 预算 + 框架自管的分配器 + **测量而非纯计算**。`save_for_backward_bytes` / `CheckpointPolicy` 两个钩子随之延后激活（§2.6） |

**D12 的理由与未来方向**

峰值是**估**出来的，而估错的代价是**单向**的：投影偏低（融合体保存量未声明、分配器碎片、重算/卸载策略尚未实现）
会在训练时 OOM；投影偏高会**错杀**一个本来跑得动的配置。**让不准的东西去否决准的东西，是划不来的。**
所以现在：`memory::plan` 照旧算（它仍是 executor 分配"常驻区 + 激活池"两块的依据）；**目标**是
`enforce_budget` 只 Warning，但代码里它**今天仍是硬失败**（`MemoryBudgetExceeded`，由
`docs/design/qwen36-text/spec.md` 的 D4 交付）。
**未来**告警要准确，前提是该字段被声明；现在字段与预算一起留空（D12）。

**vLLM 的先例支持这个方向**：它不解析式地算激活峰值，而是跑一次 `profile_run` **测量**，
再用 `总显存 × gpu_memory_utilization − 权重 − 非 torch 内存 − 激活峰值 = KV cache 预算`
（[PR #12126 的真实日志](https://github.com/vllm-project/vllm/pull/12126)：79.22GiB × 0.90 = 71.29GiB，权重 19.86 +
非 torch 0.16 + 激活峰值 36.63，余 14.65GiB 给 KV）。对我们训练框架的对应物是：
**预算由声明给出，分配由框架自管的池子做，峰值靠测量而不是靠计算**；届时 plan 里的偏移从"分配依据"退化为"复用提示"。
