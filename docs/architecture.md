# rustrain 架构定义

> 状态：**讨论中**。第 0 章（边界契约）已定；第 1–3 章是当前系统的事实描述，等待核对；
> 第 4 章列出两条路径上的缺口，是本次重构的输入。
>
> 这份文档描述**架构是什么**。实现规则（constraints、checklist、禁止模式）另有一份 skill，归属待定。

---

## 0. 边界契约（已定）

**T1 / T2 自由，T3 重编框架。**

| 层 | 变什么 | 代价 |
|---|---|---|
| **T1 实现体** | 同一契约，不同代码：SIMD / CUDA / Tilelang / CUTLASS / fp8 变体 / 融合或拆开 | 换一个 `.so` + 改 recipe。**不重编框架。** |
| **T2 声明契约** | arity、形状规则、dtype 掩码、numerics 与量化方案、expansion、collectives、反向接线 | 换 `.so` + 换 plan。**不重编框架。** |
| **T3 数学形态** | 出现框架没见过的规则种类 | **重编框架，接受** —— 形状都变了。 |

**为什么这个自由值得**：Kernel 的正确性是 **Kernel 的责任，不是框架的责任**。框架只需要能
（a）拿到一个实现、(b) 知道它的契约、(c) 有办法验证它和别人算的是同一件事。
在此之上，能**随时装卸 kernel 做对照**，对研究比编译期检查更有价值。

**因此明确放弃**：类型化/编译期的接口检查。插件是运行时对象，错的插件是运行时错误。
代价由一致性门禁承担 —— **门禁不是"顺手做的检查"，它是这条边界的承重结构**。

**推论（必须写进实现规则）**：

1. 声明错了编译器不会发现，只有门禁会发现 → 任何实现进入 plan 之前必须过门禁。
2. ABI 稳定性是纪律：`struct_size` 前向兼容、字段不得重排、C 枚举用 newtype 包住任意值。
3. digest 必须记插件身份（`plugin@version` + origin），否则同配置不可复现。
4. 切分规则与 VJP 规则**不得按算子名查表** —— 那会把 T2 泄漏成 T3。规则要作为**描述符里的声明**
   （`{kind, 参数}`），新算子只要规则种类已存在就不重编框架。
   现状违反这一条：`shard::rule_for(op: &str)` 是按名字 match 的。

---

## 1. 计算路径

从"用户下达一个训练任务"到"一个 kernel 被执行"。

```
入口
 │
 ├─ 模型构造        ← 缺失。今天只有 CLI 里手写的 demo_plan
 │    模块树 → PlanBuilder 调用序列，产出 Plan
 │
 ├─ Plan            slots[] + nodes[]；每个 slot 带 dtype / shape / ParallelLayout
 │
 ├─ Compiler::compile
 │    1 check_structure      节点拓扑序、槽位唯一写者
 │    2 shard::propagate     按规则校验/推导每个算子输入输出所需的布局，
 │     │                      在布局不匹配处**插入** intrinsic 集合通信节点
 │    3 resolve_node ×N      Registry + Recipe → 每个节点的具体实现
 │     │                      反向节点在此追踪其 backward_op
 │    4 memory::plan         寿命分析 → 偏移复用 → 峰值投影；调每个算子的 memory() 取 workspace
 │    5 enforce_budget       峰值超 budget_bytes 则**编译失败**
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

## 2. 加载路径

### 2.1 插件（kernel）加载 —— 已实现

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

### 2.2 权重 / 状态加载 —— 缺失

没有权重加载、没有 checkpoint/resume、没有 `SlotKind::State` 的管理。
模型权重今天只是 plan 里的一个 slot，没有来源。

### 2.3 插件发现 —— 缺失

今天必须显式 `--plugin <path>`。没有扫描目录、没有 manifests、没有版本约束求解。

---

## 3. crate 职责与依赖

### 3.1 实际依赖图（`cargo tree` 实测）

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

### 3.2 职责

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
实测为 0。这是"核心能在无 GPU 机器上跑完整测试"的依据（本机 215 个测试证明了这一点）。

### 3.3 已知的分层偏差（待这次重构处理）

| # | 问题 | 修法 |
|---|---|---|
| P1 | `Recipe` 是跨层策略（算子/精度/并行/显存）却住在 `rustrain-ops`（算子字典） | 独立 `rustrain-recipe`，或提到组合根 |
| P2 | `rustrain-abi` 把宿主侧与插件侧捆在一起 —— 插件因此依赖它永不需要的 `libloading` | 拆出 `rustrain-loader` |
| P4 | 没有训练层：无 train / data / checkpoint / manifest | 新增 `rustrain-train`、`rustrain-data` |
| P5 | `rustrain-kernels` 同时是"语义真值（必须纯）"和 aten provider 的预定住址（必须链 libtorch） | 拆成 `-reference` 与 `-aten` |
| P6 | `shard::rule_for(op: &str)` 按算子名查表，把 T2 泄漏成 T3 | 规则进描述符，作为 `{kind, 参数}` 声明 |
| P7 | 模型表达层不存在 —— 78 层模型按今天的写法要手工标注每张量的布局 | 模块树 + 类型级切分声明（见 D2，未定） |

P2/P6 直接服务第 0 章的边界契约，优先级高于 P1/P4/P5。

---

## 4. 两条路径上的缺口（重构输入）

**计算路径**
- 反向图（`derive_backward`）—— 设计已定（spec §2.11），未实现
- 优化器步、梯度累积、微批调度
- collective 的流分配与 overlap 调度（`StreamPolicy::Side` 已存在但无消费者）
- 训练循环本身

**加载路径**
- 权重加载（safetensors → slot）
- checkpoint / resume，以及 `SlotKind::State` 的持久化
- 插件的发现与版本约束（现在靠手写 `--plugin`）
- 一个正式的插件 SDK：现在"新增一个实现"没有模板，只有散在测试里的样例

**模型面**
- 模块树、模块类型、以及"策略挂在模块路径上"的寻址方案（D2/D3，未定）

---

## 5. 待定

| # | 决定 | 状态 |
|---|---|---|
| D2 | 模型如何表达 + 切分谁决定 | 未定 —— 用户要求**减少自动推导**，倾向"模块类型显式声明切分" |
| D3 | recipe 作用域：算子名 vs 模块路径 | 依赖 D2 |
| D4 | VJP 规则表归属：框架侧 / 描述符 / 两者 | 未定；倾向描述符（与 P6 同一理由） |
| D5 | 训练循环归属、微批与梯度累积对 plan 的影响 | 未定 |
| D6 | 上述 P1/P2/P4/P5/P6/P7 的落地顺序 | 待 D2 定 |
