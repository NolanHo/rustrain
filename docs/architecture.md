# rustrain 架构定义

> 状态：**讨论中**。第 0 章（边界契约）已定；第 1–3 章是当前系统的事实描述，等待核对；
> 第 4、5 章是讨论中形成的提案（4 切分与模型表达、5 无 GPU check 阶梯），第 6 章列出两条路径上的缺口。
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
| P7 | 模型表达层不存在 —— 78 层模型按今天的写法要手工标注每张量的布局 | 模型描述 + 结构化模板（形状见 §4，形态见 D2） |

P2/P6 直接服务第 0 章的边界契约，优先级高于 P1/P4/P5。

---

## 4. 切分与模型表达（提案）

> 4.4 是用户已同意的口径；4.1–4.3、4.5 是本次讨论中新提出的形状，**尚未定案**。

### 4.1 切分不是一个算子

切分不产生运行时计算，因此它**不进入计算图**。写成算子（`shard(x, dim, group) -> x_local`）有三个硬后果：

1. 图变成"按 rank 切过的程序"，不再描述模型本身 —— `plan explain` 在两个 rank 上输出不同，我们丢掉最便宜的调试工具（diff 两个 plan）。
2. layout 从**声明**降级成算子的**输出**：传播的输入没了，"这个 tensor 应该是什么布局"要顺着图跑一遍才知道。
3. 权重切分表达不了 —— 它发生在 step 0 之前（加载器决定从 checkpoint 读哪一块），不是图里的运算。于是出现两套切分机制：算子的给激活，加载器的给权重。

正确的形状：**切分是 slot 的属性，它在图里唯一的可见后果是一个集合通信节点。**

### 4.2 三样不许混的东西

| 概念 | 载体 | 是什么 |
|---|---|---|
| 度数（tp=8 / ep=2） | `ProcessGroups`（`ParallelConfig`） | 编译输入；进程生命周期常量 |
| 轴（这条边属于哪条 mesh 轴） | `GroupKind` + 节点属性 `ATTR_GROUP` | **节点属性**，不是 tensor operand |
| layout（哪个张量轴被切） | `ParallelLayout::{Shard,Partial}` | **slot 声明** |

`intrinsic_for()` 把 group 写成节点属性（`attrs.set(ATTR_GROUP, group_name(group))`）已经是正确形状，保持。
**group 不做 side input 的三条理由**：

1. 它是常量 —— 做成 operand 等于把常量塞进数据流，并且让图的**拓扑**变成数据依赖（"这条边要不要通信"运行时才知道），AOT 的扁平 step 列表不成立。
2. 它必须一致 —— tensor 值可以在 rank 间不一致；属性在 startup 校验一次。不一致的 group 是最不该在 step 5000 才发现的东西。
3. 它不需要"被计算" —— group 由 (mesh 拓扑, 轴) 唯一决定，是查表，不是 kernel。

### 4.3 度数：编译输入，不是运行期参数（方案 B）

- **方案 A（符号/晚绑定）**：plan 里只有轴名，startup 时把度数代进去；一个产物跨度数复用。
- **方案 B（度数作为编译输入）**：每次启动重新编译（编译是纯 CPU 的，且在 launch 路径上）；编译器**内部**用符号算术（`local = global[dim] / N`），**产物里只有具体值**。

**推荐 B**，与"非解释"一致且机制更少：产物里没有"度"的痕迹，扁平循环，每步开销与 tp=1 相同；
`N ∤ shape[dim]` 在编译期就是错误，不是运行期 fallback。A 的唯一收益（一个产物跨度数）在训练里用不到，
而"同一份描述在不同度数下各产出一个 plan 再 diff"更便宜、也更好调试。

现状已经长成 B 的形状（`shard::propagate(plan, &groups)` 收度数），缺的是 `let _ = groups;` ——
度数还没被用来算本地形状。

### 4.4 推导的口径：兑现义务，不猜声明（已同意）

- **可以推**：某 slot 声明 `Shard{dim:0, group:Tp}` → 每个 rank 只有部分和 → 该处必须 all_reduce。这是把声明的后果算出来。
- **不可以推**：看到算子名叫 `linear` 就假定权重是 `[K,N]`，看到 `qkv` 就假定 column parallel。这是猜。

今天两者混在 `shard.rs`：`propagate()` 是前者，`rule_for()` 是后者（P6 / I-5）。
修法：**推导的输入必须是描述符，不是名字。**

### 4.5 TP 和 EP 不是一类机制

| | 切什么 | 机制 | 图中的体现 |
|---|---|---|---|
| **layout**（TP / SP / DP） | 同一批节点的张量轴 | 形状算术 + 通信插入 | 节点集合不变，多出 intrinsic |
| **instantiation**（EP / PP） | **节点集合本身**（本地有哪几个 expert / 哪几层） | 按 (rank, 度数) 枚举节点 | 节点集合随 rank 变 |

EP 用形状算术表达不了：rank 0 有 expert 0–3、rank 1 有 4–7，"哪些节点存在"变了。PP 同理。

对 EP/PP 放弃"一个 plan 跑所有度数"，改成更弱但够用的性质：**同一份模型描述 + 不同切分参数 →
各实例化一个 plan，且实例化可在无 GPU 机器上完成**（接 §5）。

### 4.6 外部先例，以及它修正的一处（实测）

读到代码的事实（证据为 file:line，取自 `/data/user/nolanho/code/Megatron-LM`）：

- **层结构基本是代码。** 槽位名是类的字段；`ModuleSpec` 持有的是 Python **类对象**，`build_module` 直接实例化它
  （`spec_utils.py:13-41, 99-122`）。数据只覆盖两件事："哪一层用哪套参数"
  （`heterogeneous_config.py:157-179` 的 `block_configs` + `heterogeneous_layer_specs.py:196-215`）与
  "哪一层在第几个 stage"（`pipeline_model_parallel_layout` 字符串 DSL `'Et*3|(tt|)*29,m|L'`，
  `transformer_config.py:108-128`，解析在 `pipeline_parallel_layer_layout.py:283-321`）。
  **子模块顺序与残差接线永远是代码**（`transformer_layer.py:362-460`、`860-868`）—— 两种数据路径都表达不了
  "换一种子模块顺序"或"换一条连接"。
- **checkpoint → 参数**：名字字符串 + 每个类自己的 `sharded_state_dict()` + 前缀重命名表
  （`gpt_layer_specs.py:484-487` 的 `sharded_state_dict_keys_map`，应用在 `transformer_layer.py:1168-1173`）。
  没有全局表。
- **切分轴是类里的字面 dict**：`ColumnParallelLinear` → `{"weight": 0, "bias": 0}`
  （`tensor_parallel/layers.py:1116-1126`），`RowParallelLinear` → `{"weight": 1}`（`:1379-1389`）。

**它修正的一处**：Megatron 的切分轴出现在 `sharded_state_dict()` 里，也就是**加载侧**。
这说明**"切分轴"和"从 checkpoint 取哪一块"是同一条事实**——Megatron 把它存在类里（代码），
我们错在把它存在 `rule_for(op: &str)` 里（也是代码，而且是按名字猜）。正确位置是**参数/slot 的描述符**：
一份声明同时被 L2 的名字映射和形状算术读取。这比"把 `rule_for` 改成描述符"更准确。

**Q5 核实完毕：没有先例。** vLLM / SGLang / JAX MaxText / torchtitan 都不把图本身（哪些算子、什么顺序、怎么连）表达成数据。
四个最接近的形态，都停在图的门口：

- vLLM 有 `splitting_ops: list[str]` 这类数据，但那是"在哪些算子上切"；pass 流水线是代码（`if self.pass_config.enable_sp: ...`），
  最后还 `generate_execution_code()` 生成 Python 源码字符串再 `exec`。
- SGLang 手里是运行时的 `torch.cuda.CUDAGraph` 对象，`--cuda-graph-config` 只描述 batch 尺寸与 backend。
- MaxText 是 `layer_map = {DecoderBlockType.DEFAULT: [NNXDecoderLayer], ...}` —— 配置枚举选 Python 类。
- torchtitan 最接近：`Llama3Model.Config(layers=[...])` 是数据的层列表，但只描述**构造**；算子顺序与连接写死在
  `decoder.py` 的 `forward()` 里，`ModelSpec` 没有算子/边字段。（它的 `experiments/graph_trainer` 确实序列化了 FX trace，
  但那是**派生的缓存**，不是被编写的源头。）

**根因**：它们的图就是 `forward()` —— 图存在，但不是一等产物，是 Python 控制流，没有东西可以序列化。所以只能把"选择"
做成数据、把"连接"留给代码。**它们的 `forward()` 就是我们的 `Plan{slots,nodes}`。**

**风险（要正视）**：据我们所知，这是一个把图当作**编写对象**的框架。生态里唯一被序列化的图（vLLM codegen、
torchtitan graph_trainer）都是派生产物，只服务于性能，从不当真值来源。后果是：写错的图**编译不会报错**，
而且没有任何现成工具能替我们验证它。因此——

> **§5 的 check 阶梯不是便利功能，它是"模型即数据"的前置条件。**
> 没有 L1/L2/L3 加参考实现对账，"图即数据"就是莽撞的。

而这件事之所以可做，恰恰因为 rustrain 早就把图做成了一等产物（`Plan` 早于模型层存在），
所以"模型即数据"不是新机制，只是**生成这个 Plan 的输入**：参数 + 带重复的子图模板 + checkpoint 名字映射。

---

## 5. 无 GPU check 阶梯（提案）

用户要求：**指定 kernel + model 路径就能检查形状，不需要 GPU**。这是"支持一个模型"的主循环。

| 级 | 需要什么 | 查什么 | 现状 |
|---|---|---|---|
| **L1 结构** | 无（零设备、零权重） | plan 可编译；每个节点的算子可解析到实现；每个算子的 `infer()` 与声明形状一致；layout 传播完成、每个 `Partial` 都被兑现、每个 collective 都绑了轴；内存规划无重叠且不超预算 | 能力已具备（`validate_shapes`），**缺驱动**：今天跑的是 CLI 里手写的 demo plan（`plan explain --tp N`），不是"指定 model 路径" |
| **L2 加载** | checkpoint 的 **metadata**，不读数据 | 每个 slot 都能从某个 checkpoint tensor 得到（名字 + 变换：slice / transpose / qkv split）；每个 checkpoint tensor 要么被消费、要么显式声明忽略；dtype / shape / 切片范围一致 | **缺失**。safetensors 头部即 JSON（名字 / dtype / shape / offset），读它不需要读权重数据。实测：4 GB / 1386 tensor 的 checkpoint，头部 198 KB —— 开销可忽略。真实 checkpoint 命名很脏（`base_model.model.model.layers.0.mlp.down_proj.lora_A.weight`），所以映射必须是 pattern/前缀表，且 L2 要能报告"没被消费的 tensor" |
| **L3 数值** | 设备（或 CPU 参考实现） | 同一算子的两个实现算同一件事 + 数值参考 | 已有（conformance gate） |

L2 的变换词表最小集（由真实 checkpoint 反推，不是想出来的）：`take(name)` / `slice(dim, range)` / `transpose` /
`split` / `concat(dim)`。Megatron 用到的全部机制都落在这几种里 —— 前缀重命名（`sharded_state_dict_keys_map`，
`gpt_layer_specs.py:484-487`）、按轴切片（`{"weight": 0}`，`layers.py:1116-1126`）、
以及 MLA 的 `torch.cat([q_weight, kv_weight], dim=0)`（`multi_latent_attention.py:1479-1497`）。

CLI 形状：

```
rustrain check --model <model-dir> --plugin <p.so> [--tp N --pp N --ep N] [--json]
```

- 退出码非零即失败；每条 skip 必须写原因（沿用门禁纪律）。
- `--json` 的用途是**机器消费**：L2 的自然用户是从 HF 模型"拆碎"出 plan 描述 + 名字映射的生成器，
  `decompose → check → 修映射 → check` 全在笔记本 CPU 上跑。
- 顺带可查**算子覆盖**：模型需要的算子本机有没有实现、有没有目标精度的变体（`uncovered_operators()` 是种子）。

**边界（必须写死，否则这个 check 会骗人）**：L1+L2 只抓**接线错误**，抓不了"这个分解算的是别的东西"。
仓库内有现成反例：切分规则的权重约定反了，而**全部测试通过**。所以 check 是开发内环，不是正确性证明；
最后一道仍是 L3，以及与参考实现（HF logits/loss）对齐。

**成本分层**（"支持一个模型"的三种价位）：

1. 只用现有原语 → 纯数据：结构 + 名字映射，由 L1/L2 在 CPU 上验证。**零 Rust。**
2. 需要新原语 → + 一个 kernel（T2，丢一个 `.so`）。
3. 框架没见过的数学形态 → T3，重编框架。应罕见，且每次都应能说清"为什么它是 T3"。

目标：绝大多数模型支持落在第 1 档。

---

## 6. 两条路径上的缺口（重构输入）

**计算路径**
- 反向图（`derive_backward`）—— 设计已定（spec §2.11），未实现
- 优化器步、梯度累积、微批调度
- collective 的流分配与 overlap 调度（`StreamPolicy::Side` 已存在但无消费者）
- 训练循环本身

**加载路径**
- 权重加载（safetensors → slot）
- **L2 加载检查**：名字映射（checkpoint tensor → slot）的机械验证，见 §5
- checkpoint / resume，以及 `SlotKind::State` 的持久化
- 插件的发现与版本约束（现在靠手写 `--plugin`）
- 一个正式的插件 SDK：现在"新增一个实现"没有模板，只有散在测试里的样例

**模型面**
- 模型描述格式：结构（哪些算子、怎么连）是数据还是插件，见 D2 / §4
- 模块树是否必要：今天它只贡献 `Trace.path` 这一个可读字符串，没有任何东西**遍历**它。
  只有当某个消费者必须走树时（checkpoint 映射、显式 optimizer 分区、state/KV 管理）才值得引入

---

## 7. 待定

| # | 决定 | 状态 |
|---|---|---|
| D2 | 模型描述的具体形态 + 切分谁决定 | **已收窄**：切分 = slot 声明 + 通信插入（§4.1）、度数 = 编译输入（§4.3）、推导只兑现声明（§4.4）已定。**未定**：结构本身是数据（模板/子图）还是由插件提供的复合算子；用户倾向后者（b），待综合性能与可用性定案 |
| D3 | recipe 作用域：算子名 vs 结构路径 | 依赖 D2 |
| D4 | VJP 规则表归属：框架侧 / 描述符 / 两者 | 未定；倾向描述符（与 P6 同一理由） |
| D5 | 训练循环归属、微批与梯度累积对 plan 的影响 | 未定 |
| D7 | `rustrain check` 的层级划分与 L1/L2 落地顺序（§5） | 依赖 D2 |
| D6 | 上述 P1/P2/P4/P5/P6/P7 的落地顺序 | 待 D2 定 |
