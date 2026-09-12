# 模型描述（D8 / D9 设计）

> **状态：设计，待评审。** D2 已定"结构是数据"（`docs/architecture.md` §1.2）；本文定 **D8**（描述的语法与
> 展开语义）与 **D9**（轴与 mesh 的表示）。现状基线见 `docs/design/plan-ir-baseline.md`。
>
> 本文不含实现计划；落地顺序仍是 §8 的 D6。

---

## 0. 三个产物、两个阶段

```
描述（JSON，拓扑无关）
   │  expand
   ▼
全局模型（= 一个 Plan：layout 全 Replicate，形状全量，slot 带 symbolic binding）
   │  instantiate(mesh, rank)
   ▼
Plan（具体：本地形状、组掩码、本 rank 的节点集合）
   │  compile（现有七步）
   ▼
CompiledPlan
```

**只有一个表示。** 全局模型就是一个 `Plan` —— 不需要第二套 IR。这样做的三个后果：

1. **L2 在全局模型上做**（checkpoint ↔ slot 的对齐），因此**完全不需要拓扑** —— 这正是
   `architecture.md` §1.1 那条推论的落地。
2. **L1 在 instantiate 之后做**，因为它需要度数。
3. PP 的节点集合裁剪、其余四轴的形状除法，都发生在 instantiate 里；`compile` 的七个 pass 不变。

---

## 1. 轴与 mesh（D9）

### 1.1 mesh 是**有序**的轴列表

```rust
Mesh { axes: Vec<(name: String, degree: usize)> }   // 编译输入 + 运行输入，不进 plan
```

rank 分解约定：`axes[0]` 变化最快，`stride_i = Π_{j<i} degree_j`，即 `rank = Σ coord_i * stride_i`。
**声明轴的顺序就是声明 rank 布局**，不再硬编码 `[tp, cp, ep, dp, pp]`（现状见 `group.rs:287-301`、
`rank.rs:9-31`）。

### 1.2 组 = **轴掩码**

```rust
struct GroupMask(u32);   // bit i = mesh 的第 i 条轴参与这个组
```

- `Global` = 全掩码；"tp 组" = 单个 bit。
- **为什么不是枚举**：`GroupKind` 是封闭六值（`group.rs:20-49`）+ `ProcessGroups.groups: [_; 6]` 定长
  数组（`group.rs:142`）+ 五轴 stride 硬编码，因此 **tp×dp、tp×ep 这类组合组没有名字也没有槽位**。
  掩码天然支持任意组合。
- **掩码是结果，不是拓扑对象**（I-6）：度数已经在 instantiate 时烘进形状，plan 只存掩码；
  名字与 degree 留在 mesh 里。拿到 mesh 时掩码可打印成人话（`tp|ep`）。
- 32 轴上限，超出**报错**，不静默截断。

### 1.3 进 plan 的是什么

`PlanMeta.mesh: MeshFingerprint` —— 有序的 `[(name, degree)]`，供 digest 与 `plan explain`。
**不是**可遍历的 mesh 对象，不含 rank 列表。

---

## 2. Layout 的推广

### 2.1 一个 slot 的 layout = 多个 (dim, group) 分片 + 至多一个 partial

```rust
struct ParallelLayout {
    dims: Vec<ShardSpec>,           // ShardSpec { dim: i64, group: GroupMask }
    partial: Option<PartialSpec>,   // PartialSpec { op: ReduceOp, group: GroupMask }
}
```

**为什么必须是多个**：Qwen3.6 MoE 在 `tp=2, ep=4` 下，`mlp.experts.gate_up_proj [E, 2I, H]` 的本地形状是
`[E/4, 2I/2, H]` —— dim 0 沿 ep、dim 1 沿 tp，**同一个张量上两条互不相干的切分**。
今天的 `Shard { dim, group }` 是单值（`layout.rs:76`、`layout.rs:101-109`），表达不了。

`partial` 保持"至多一个"：今天没有出现需要两个部分和的真实情形，且转移规则表要能穷举才是安全的。
真要两个时是加法式扩展。

### 2.2 本地形状算术：唯一的推导，纯后果

```
local[d] = global[d] / Π { degree(轴) : 轴 ∈ spec.group, spec ∈ dims, normalize(spec.dim) == d }
```

- 不整除 → **编译期硬错误**（方案 B：不是运行期 fallback）。
- 这就是 `architecture.md` §1.5 说的"兑现义务"：描述声明了切分，框架算出本地形状。

### 2.3 转换保持**单步**

- partial 的兑现：`all_reduce` over `partial.group`；若目标在该 dim 上有同组分片，则 `reduce_scatter`。
- 同 dim 换组、或 shard ↔ replicate：单轴单步。
- 需要多轴或多步才能完成的转换 → **拒绝**（沿用 `shard.rs:399-411` 的纪律），要求描述显式写出中间 layout。
- 跨组不再一律 `GroupMismatch`（`collective.rs:283-286`）：先绕 `Replicate`（官方路线），或单步同轴。

### 2.4 与今天的关系

| 今天 | 目标 |
|---|---|
| `GroupKind` 封闭六值 + `[_; 6]` 数组 | `GroupMask(u32)` + mesh |
| `ParallelLayout` 五个封闭变体，单 group | `{ dims: Vec<ShardSpec>, partial }` |
| `intrinsic.ATTR_GROUP` 六值字符串（`ir.rs:492-514`） | 组掩码的稳定编号 |
| `let _ = groups;`（`shard.rs:236`），拓扑没人校验 | instantiate 必读 mesh；`GroupUnavailable` 成为真实错误 |

---

## 3. 描述文件（D8）

**格式：JSON。** 理由：主要由生成器产出（手工编辑是次要场景）、结构嵌套深、与 `config.json` 同族。
四个顶层键：`params` / `templates` / `stack` / `binding`。

```jsonc
{
  "format": "rustrain.model.v1",
  "name": "qwen3.6-35b-a3b",
  "dtype": "bf16",
  "params": { ... },       // 3.1
  "templates": { ... },    // 3.2
  "stack": [ ... ],        // 3.3
  "binding": [ ... ]       // 3.4
}
```

### 3.1 `params` —— 值的唯一来源

| 形式 | 含义 |
|---|---|
| `{"from": "text_config.hidden_size"}` | 从模型目录的 `config.json` 取 |
| `{"from": "...", "default": 4}` | 缺省 |
| `{"expr": "2 * heads * head_dim"}` | 参数表达式：整数、参数名、`+ - * / ( )` |
| `["full_attention", "linear_attention", ...]` | 列表值（逐层类型） |

**不做"标量 → 结构"的隐藏推导。** 例如 Qwen 的 `full_attention_interval: 4`（旧代码解析了却不读，
`config.rs:145-146`）与 GLM5 的 indexer 取模回退（`glm5/model.rs:342-343`）：这类派生由**生成器**算成
显式列表写进描述，描述语言本身只需要**按下标取列表**。理由：少自动推导；列表可见、可审、可 diff。

### 3.2 `templates` —— 具名子图，只管数学与连接

```jsonc
"templates": {
  "mlp_swiglu": {
    "inputs":  { "x": {"shape": ["seq", "hidden"], "kind": "activation"} },
    "outputs": { "y": {"shape": ["seq", "hidden"], "kind": "activation"} },
    "slots": [
      { "name": "wg", "kind": "weight", "shape": ["inter", "hidden"] },
      { "name": "wu", "kind": "weight", "shape": ["inter", "hidden"] },
      { "name": "wd", "kind": "weight", "shape": ["hidden", "inter"] }
    ],
    "nodes": [
      { "op": "linear", "in": ["x", "wg"], "out": ["g"] },
      { "op": "elementwise_unary", "in": ["g"], "out": ["gs"], "attrs": { "kind": "silu" } },
      { "op": "linear", "in": ["x", "wu"], "out": ["u"] },
      { "op": "elementwise_binary", "in": ["gs", "u"], "out": ["gu"], "attrs": { "kind": "mul" } },
      { "op": "linear", "in": ["gu", "wd"], "out": ["y"] }
    ]
  }
}
```

- slot 名是**模板内局部名**；形状是 `params` 表达式。
- 模板**不含任何切分信息** —— 切分住在 `binding`（§3.4），因为那是"参数从哪来"的同一个事实（§1.4）。

### 3.3 `stack` —— 实例化，按序展开

```jsonc
"stack": [
  { "template": "embed",   "prefix": "embed" },
  { "template": "decoder", "prefix": "layers.{l}",
    "repeat": { "count": "layers", "index": "l" },
    "select": { "by": "layer_types[l]",
                "cases": { "full_attention": "decoder_full",
                           "linear_attention": "decoder_linear" },
                "default": "decoder_full" } },
  { "template": "final_norm", "prefix": "norm" },
  { "template": "lm_head",    "prefix": "lm_head" },
  { "template": "mtp",        "prefix": "mtp",
    "until": "mtp_layers",          // 0 = 不展开
    "inputs": { "h": "layers.{last}.y", "ids": "input_ids" } }
]
```

- **接线显式**：每项的 `inputs` 是 `{局部名: 全局名}`；缺省是链式（上一项的 `outputs`）。
  首项的 inputs 来自描述的 `inputs` 段（`input_ids` 等）。模板是函数，stack 是带实参的调用。
- **选择只按列表下标**（§3.1），不做算术。
- `prefix` 里的 `{l}` / `{last}` 由实例化器替换；**名字就是 slot 的标识**，唯一的寻址方式是名字模式匹配
  （这也是"不需要模块树"的依据：只有名字约定 + 模式匹配，没有东西需要遍历树）。

### 3.4 `binding` —— 参数映射，也就是 L2 的定义

```jsonc
"binding": [
  { "slot":   "layers.*.self_attn.q_proj.weight",
    "source": "model.layers.{*}.self_attn.q_proj.weight",
    "transform": [],
    "axes": { "0": ["tp"] } },

  { "slot":   "layers.*.mlp.experts.gate_up_proj",
    "source": "model.layers.{*}.mlp.experts.gate_up_proj",
    "transform": [],
    "axes": { "0": ["ep"], "1": ["tp"] } },

  { "slot":   "layers.*.mlp.experts.down_proj",
    "source": "model.layers.{*}.mlp.experts.down_proj",
    "transform": [],
    "axes": { "0": ["ep"], "2": ["tp"] } }
]
```

- `slot` / `source` 的 `*` 是**共享捕获**：两边同一个 `{*}` 指同一个下标段。
- **`axes` 指的是 slot 的维度**（kernel 看到的那个张量），不是 checkpoint 的维度；
  `transform` 负责 checkpoint → slot 的映射。这样"切哪一维"无歧义。
- `transform` 词表：`take(name)` / `slice(dim, range)` / `transpose(i, j)` / `split(dims, sizes)` / `concat(dim)`。
  这五种覆盖了 Megatron 用到的全部机制（前缀重命名、按轴切片、MLA 的 `cat([q, kv])`）。
- `axes` 里是**符号轴名**；instantiate 时解析成 `GroupMask` 并执行 §2.2 的除法。
- **切分轴住在这里**：不是模板，也不是框架侧规则表（I-5 / P6）。它与"从 checkpoint 取哪一块"
  是同一条事实，所以必须同一处声明、同一处被加载器与形状算术读取。

### 3.5 强制性（L2 的判据）

- 全局模型里**每个 weight slot 必须被恰好一条 binding 命中**；未命中 → 报错。
- 每个 checkpoint tensor 要么被消费，要么在描述里显式 `"ignore": [...]`；静默丢弃 → 报错。
- `transform` + `axes` 推出的本地形状与 slot 形状必须一致 → 否则报错。
- `slice`/`split` 的区间必须在范围内。

---

## 4. 展开与实例化语义

### 4.1 `expand`：描述 → 全局 Plan（确定性）

1. 解析 `params`（表达式按依赖序求值，环 → 报错）。
2. 按 `stack` 顺序展开模板：分配 slot（前缀 + 局部名）、发射节点（模板内顺序即发射顺序）。
3. 形状求值（全量形状，`layout` 全 `Replicate`）。
4. `check_structure()`（拓扑序、唯一写者、非空输出）。
5. 挂上 `binding` 的符号声明（每个 weight slot 一条）。

确定性要求：同名冲突报错；顺序即定义序，不做隐式排序；同一份描述 + 同一份 config 必然得到同一个 Plan。

### 4.2 `instantiate`：全局 Plan × (mesh, rank) → 具体 Plan

| 步骤 | 做什么 |
|---|---|
| 1. **PP** | 只保留本 stage 的模板实例（由描述里的 `split_layers` 规则或每项的 `stage` 给出）；跨界 slot 变成 plan 的 input/output |
| 2. **形状** | 对每个 slot：把 `binding.axes` 与从激活传播来的 layout 解析成 `GroupMask`，执行 §2.2 除法；不整除 → 报错 |
| 3. **组可用性** | 每个用到的 `GroupMask` 必须在 mesh 里有定义（度数 > 1）；否则 `GroupUnavailable`（今天从不触发） |
| 4. **重写** | 重写 slot 名称（加 rank 无关的稳定后缀即可，形状已本地化），节点集合即为产物的节点集合 |

**PP 是唯一的"节点集合随 rank 变"的机制**（见 §6.1）。其余四轴都只改形状与通信。

---

## 5. 与 check 阶梯的对应

| 级 | 在哪一步做 | 输入 |
|---|---|---|
| **L2 加载检查** | `expand` 之后，`instantiate` 之前 | 全局 Plan + checkpoint metadata（safetensors 头部）+ `binding` |
| **L1 结构检查** | `instantiate` 之后 | 具体 Plan + mesh（形状、layout、每个 collective 的组、内存预算） |
| **L3 数值** | 运行期 | 两个实现 + 一个数值参考 |

L2 不需要 topology，L1 需要 —— 与 `architecture.md` §4.4 一致。CLI 统一入口：

```
rustrain check --model <model-dir|desc.json> [--tp N --cp N --ep N --dp N --pp N] [--json]
```

`--model` 只给描述文件时，L2 需要额外的 `--checkpoint <dir>`（或从描述指向模型目录）。

---

## 6. 这次定型修正的两处

### 6.1 EP **不是** instantiation，而是 layout + 显式 routing

`architecture.md` §1.4 原先把 EP 与 PP 归为同一类（"切节点集合本身"）。以 Qwen3.6 MoE 走一遍之后
这个归类是错的，修正为：

- **专家权重就是 dim 0 的分片**：`experts.gate_up_proj [E, 2I, H]` → `axes {0: [ep]}`，
  本地形状 `[E/4, ...]` 由 §2.2 的除法给出。**没有"节点集合变化"这件事。**
- **激活侧的 routing 是数据依赖的**（token 去哪个 rank 由 router 的 top-k 决定），
  所以它**不能**从 layout 推出来，必须是图中的显式算子（dispatch / combine），由 kernel 声明
  `collectives: [all_to_all(ep)]`。
- 推论：**五轴里只有 PP 改变节点集合**，其余四轴（TP/CP/DP/EP）都是 layout。
  PP 也因此是唯一需要**调度器**（微批、send/recv）的轴 —— 那是一个独立的子系统。

这条修正直接简化了设计：五种并行 = 一种 layout 机制 + 一种 PP 机制，而不是两类各两种。

### 6.2 MoE 的 dispatch 输出形状是数据依赖的，所以 MoE 是 EXPLICIT 算子

`Plan` 要求所有形状具体（不许符号维，`ir.rs:6-8`），而 dispatch 之后每个 rank 拿到的 token 数是
运行期才知道的。两个选项，选第二个：

- ~~把 `dispatch`/`expert_ffn`/`combine` 作为可规划的原语~~ —— 中间激活的形状无法声明；
- **把整块 MoE FFN 做成一个 `EXPLICIT` 算子**：输入输出形状都是静态的 `[seq, hidden]`，
  内部完成 routing + dispatch + 分组 GEMM + combine；专家权重是**独立的 weight slot**，
  由 `binding` 声明 `axes`（§3.4）；`collectives: [all_to_all(ep)]`。

这与 §0 的契约一致：**框架推不出来的东西就声明，声明由门禁兜底**，不假装推导。

---

## 7. 明确不做 / 待定

- **不做符号维**：所有形状在 instantiate 后必须具体（沿用 `ir.rs:6-8`）。
- **不做多轴/多步转换**（§2.3）；需要时要求显式写出中间 layout。
- **不做"标量 → 结构"的推导**（§3.1）；派生交给生成器。
- **描述语言暂不含条件/循环**，只有"按列表下标选择"与"计数重复"。若真实模型需要更多，先加用例再改语言。
- **待定**：`split_layers`（PP 的层分配规则）的具体形式；CP 对线性注意力层是否支持（取决于 §8 的证据）；
  `binding` 的 `ignore` 列表写法；描述文件的版本迁移策略。
