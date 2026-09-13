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

**不做"标量 → 结构"的隐藏推导。** 目标模型的层类型在 checkpoint 里就是**显式列表**
（`layer_types`，40 项），描述按 `list[i]` 取值即可。框架不提供"间隔/取模"这类派生语法 ——
一旦需要，由**生成器**算成显式列表写进描述。理由：少自动推导；列表可见、可审、可 diff。
（旧代码里 `full_attention_interval` 解析了却不读，`config.rs:145-146`，也是"标量派生不可信"的实物。）

### 3.2 `templates` —— 具名子图，只管数学与连接

```jsonc
"templates": {
  "mlp_swiglu": {
    "inputs":  { "x": {"shape": ["seq", "hidden"], "kind": "activation"} },
    "outputs": { "y": {"shape": ["seq", "hidden"], "kind": "activation"} },
    "slots": [
      { "name": "wg",  "kind": "weight",     "shape": ["inter", "hidden"] },
      { "name": "wu",  "kind": "weight",     "shape": ["inter", "hidden"] },
      { "name": "wd",  "kind": "weight",     "shape": ["hidden", "inter"] },

      // 中间激活也必须在这里声明（§3.7 #1）—— 不能让编译器 infer 回填
      { "name": "g",   "kind": "activation", "dtype": "bf16", "shape": ["seq", "inter"] },
      { "name": "gs",  "kind": "activation", "dtype": "bf16", "shape": ["seq", "inter"] },
      { "name": "u",   "kind": "activation", "dtype": "bf16", "shape": ["seq", "inter"] },
      { "name": "gu",  "kind": "activation", "dtype": "bf16", "shape": ["seq", "inter"] }
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

- slot 名是**模板内局部名**；形状是 `params` 表达式；`dtype` 可省（继承顶层，§3.6 #6）。
- **每个被节点写入的名字都必须在这里声明**（或属于模板的 `outputs`），否则报错 —— 见 §3.7 #1。
- 模板**不含任何切分信息** —— 切分住在 `binding`（§3.4），因为那是"参数从哪来"的同一个事实（§1.4）。

### 3.3 `stack` —— 实例化，按序展开

```jsonc
"stack": [
  { "template": "embed",   "prefix": "embed" },
  // `select` 与 `template` 互斥（§3.7 #13）：有 select 时兜底只有 default
  { "prefix": "layers.{l}",
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
  首项的 inputs 来自描述的 `inputs` 段（`input_ids` 等，语法见 §3.6 #2）。模板是函数，stack 是带实参的调用。
- **选择只按列表下标**（§3.1），不做算术。
- `prefix` 里的 `{l}` / `{last}` 由实例化器替换；**名字就是 slot 的标识**，唯一的寻址方式是名字模式匹配
  （这也是"不需要模块树"的依据：只有名字约定 + 模式匹配，没有东西需要遍历树）。

### 3.4 `binding` —— 参数映射，也就是 L2 的定义

**方向约定（必须写死）**：`axes` 永远指 **slot 的维度**，即 kernel 看到的那个张量。
`linear` 的权重在 slot 方向是 `[K, N]`（收缩维在前，由 reference provider 固定）；
HF / legacy 的 checkpoint 是 `[out, in]`，所以 binding 用 `transpose` 归一化。

```jsonc
"binding": [
  // HF [out, in] -> slot [in, out] = [K, N]
  { "slot":   "layers.*.self_attn.q_proj.weight",
    "source": "model.layers.{*}.self_attn.q_proj.weight",
    "transform": ["transpose(0,1)"],
    "axes": { "1": ["tp"] } },                       // column：切 N

  { "slot":   "layers.*.self_attn.o_proj.weight",
    "source": "model.layers.{*}.self_attn.o_proj.weight",
    "transform": ["transpose(0,1)"],
    "axes": { "0": ["tp"] } },                       // row：切 K -> partial

  // 三通道专家权重：HF [E, out, in] -> slot [E, in, out]
  { "slot":   "layers.*.mlp.experts.gate_up_proj",
    "source": "model.layers.{*}.mlp.experts.gate_up_proj",
    "transform": ["transpose(1,2)"],
    "axes": { "0": ["ep"], "2": ["tp"] } },

  // 一个 source -> 多个 slot：fused [gate|up] 在 [gate|up] 布局下按 tp 连续切是错的
  // （rank 0 会拿到全部 gate 行、rank 1 拿到全部 up 行，本地算不出来）。
  // 拆成两个 slot 各自声明分片；"融合存储"与"融合计算"是两件事，kernel 仍可同时读两个指针做融合。
  { "source": "model.layers.{*}.mlp.experts.gate_up_proj",
    "transform": ["transpose(1,2)"],
    "split": { "dim": 2, "sizes": ["moe_i", "moe_i"] },
    "targets": [
      { "slot": "layers.*.mlp.experts.gate_proj", "axes": { "0": ["ep"], "2": ["tp"] } },
      { "slot": "layers.*.mlp.experts.up_proj",   "axes": { "0": ["ep"], "2": ["tp"] } }
    ] }
]
```

- `slot` / `source` 的 `*` 是**共享捕获**：两边同一个 `{*}` 指同一个下标段。
  `source` 里**不得出现 `**`**（多段通配只属于 `ignore`）：一条 binding 的职责是"一个具体 checkpoint 张量 ↔ 一个具体 slot"。
- `transform` 词表**只有两个动词**：`transpose(i, j)` 与 `slice(dim, start, len)`。
  - `slice` 把该轴长度换成 `len`；要求 `start >= 0`、`len > 0`、`start + len <= size`（`checked_add`）；
    负 `dim` 从末尾数（与 `transpose` 一致）。
  - **多段拆分由 binding 的 `split` 字段表达**，不在 `transform` 里 —— 早期列的
    `take` / `concat` / `split(dim,sizes)` **已移除**（没有消费者也没有定义 = 死钩子）。
  - **未知动词在 `expand` 期报错**；`slice` 的越界是 `shape_mismatch`（fail）。
    **不存在"动词没实现所以 skip"的分支** —— skip 不得用来掩盖契约未实现。
- `axes` 里是**符号轴名**；instantiate 时解析成 `GroupMask` 并执行 §2.2 的除法。
- **切分轴住在这里**：不是模板，也不是框架侧规则表（I-5 / P6）。它与"从 checkpoint 取哪一块"
  是同一条事实，所以必须同一处声明、同一处被加载器与形状算术读取。

### 3.4.1 EXPLICIT 算子的 `collectives` 声明

有些通信**推不出来**，只能声明，因为它是数据依赖的或跨 rank 的：

| 场景 | 通信 | 为什么推不出来 |
|---|---|---|
| MoE dispatch / combine | `all_to_all({tp, ep})` | 目标 rank 由 router 的运行期 top-k 决定 |
| GDN 的 chunkwise CP | `all_gather({cp})` + fp32 链式 merge | 跨 chunk 的仿射映射必须在 rank 间传播 |
| 全注意力的 CP | `all_gather({cp})` / ring | 每个 rank 需要完整 KV |

形式：算子在描述符里声明 `collectives: [{ kind: "all_to_all", group: "tp|ep" }]`。

**校验（能查的都要查，查不到的要诚实）**：
- ✅ 声明的组的轴，必须是该张量**实际被切分的轴**之一 —— 在一个没有切分的轴上做 all_to_all 是错的。
- ✅ 组的轴必须在 mesh 里存在（度数 > 1）。
- ✅ 通信两侧的 layout 与本地形状必须自洽。
- ❌ **不能**验证它的正确性（数据依赖的 routing 对不对、仿射 merge 对不对）—— 那由 L3 门禁承担。
  这正是 §0 的边界契约：**推不出来的就声明，声明由门禁兜底，不假装推导。**

### 3.5 强制性（L2 的判据）

- 全局模型里**每个 weight slot 必须被恰好一条 binding 命中**；未命中 → 报错。
- 每个 checkpoint tensor 要么被消费，要么在描述里显式 `"ignore": [...]`；静默丢弃 → 报错。
- `transform` + `axes` 推出的本地形状与 slot 形状必须一致 → 否则报错。
- `slice`/`split` 的区间必须在范围内。

### 3.6 已裁定的细节（原先是空白；实现者不得自行发明）

D1 的验收测试暴露了十处未定义。以下裁定**是契约的一部分**。

| # | 问题 | 裁定 |
|---|---|---|
| 1 | 描述文件的位置与名字 | **模型目录下的 `model.json`**（与 `config.json` 同级） |
| 2 | 顶层键 | `format`(必填) / `name`(必填) / `dtype`(可选，默认值) / `inputs`(可选) / `params` / `templates` / `stack` / `binding`。`inputs` 与 `templates.*.inputs` 同构：`{shape, kind, dtype?}` |
| 3 | `plan explain --json` 的形状 | 顶层 `slots` 与 `nodes` 是**数组**，计数移进 `counts`：<br>`{"name","digest","world_size","mesh":{"axes":[["tp",2],["cp",1],…]},"counts":{"slots","nodes","steps"},"slots":[…] ,"nodes":[…] ,"implementations":[…],"collectives":[…],"memory":{…}}`。<br>这是对现有 CLI 的**破坏性修改**（今天 `slots`/`steps` 是计数、没有 `nodes`），由 D1 承担。<br>**D3 追加 `mesh`**：组掩码是 bit 位置，没有轴表就翻不回名字（§1.2 / §1.3）—— 所以轴表必须出现在人读的那份 JSON 里，而不是只存在于 digest 里 |
| 4 | `binding.axes` / `transform` 是否必填 | **都可省**。省 `axes` = 不切分（全局 Plan 全 `Replicate`）；省 `transform` = 恒等 |
| 5 | dtype 词表 | 小写字符串，与 `RsDtype::name()` 一致：`f32 f16 bf16 f8e4m3 f8e5m2 fp4e2m1 i32 i64 u8` |
| 6 | 模板 slot 是否要 dtype | **要**。`{name, kind, dtype, shape}`，缺省继承顶层 `dtype`。没有它表达不了索引输入（`embedding` 的 `i64`）——**这是本表里唯一修语言的一条** |
| 7 | 同名冲突由谁保证 | `expand` 保证 **slot 名全 plan 唯一**（冲突时指出重复的名字与两个来源）与 **stack 实例前缀唯一** |
| 8 | 报错是否允许 panic | **不允许**。六条错误路径都必须"非 0 退出 + stderr 说明名字 / 模式 / 路径"，不得出现 `panicked at` |
| 9 | 二进制名 | **`rustrain`**（`crates/rustrain-cli/Cargo.toml` 加 `[[bin]] name = "rustrain"`）。文档里所有 `rustrain …` 命令以此为准 |
| 10 | `dtype` 没有可用实现时 `explain` 的行为 | `plan explain` **不因缺实现而失败**：它在 `implementations` 里报告未解析的算子与原因，退出码仍为 0。**`check` 才要求解析成功**（那是 L1 的一部分）。这样"用真实 bf16 描述看结构"与"用 f32 跑门禁"两件事都能做 |

### 3.7 第二轮裁定（D1 实现反馈）

| # | 问题 | 裁定 |
|---|---|---|
| 1 | **中间激活的形状没有声明位置** | **模板必须声明所有被节点写入的 slot**（含中间激活）；节点的 `out` 只能引用已声明的 slot 或该模板的 `outputs`，否则**报错**（信息含实例前缀、节点号、算子、模板名、未声明的名字）。<br>`shape` **必填**；`dtype` 可省并继承顶层（§3.6 #6）—— 真实 fixture 写全只是风格，不是额外要求。<br>**为什么不能"由编译器 infer 回填"**：那会让无 GPU 的 L1 形状检查依赖"存在可解析的实现"，而真实描述是 bf16、reference provider 只有 f32 —— **整条 L1 就废了**。声明齐全后 L1 才能做"声明 vs infer"的全量比对。这条**修语言** |
| 2 | weight slot 数不是 > 900 | **实测 884**（= 712 + 2×30 `in_proj_qkv` + 2×30 `conv1d` + 11 `q_proj` + 41 `gate_up`）。**不去凑这个数字**：权威验收是 `nodes > 900` 与总 `slots > 900`（实测 1047 / 1943）。为了凑数把被拆掉的融合张量本身也留成 slot 是错的 —— 那会造出**无人读取的 slot**，正是本项目的头号禁忌 |
| 3 | `digest` / `steps` / `memory` 在未编译的模型路径上 | 照 `--model` 的语义定义：`counts.steps = 0`；`digest` = 全局 Plan JSON 的 blake3；`memory` 全 0；`collectives` 为空。它们是**编译产物的占位**，不是谎报 —— `plan explain --model` 的语义是"展开 + 报告"，不是"编译" |
| 4 | §3.5 的三条强制性在 expand 阶段做不了 | 划清边界：§3.5 的三条（本地形状一致、split 区间、张量消费）**属于 L2 / D2**；expand 阶段只做**模式层**校验（每个 weight slot 被恰好一条 binding 命中、source 不重复、两侧 `{*}` 数相等） |
| 5 | `attrs` 只能是字面量 | **接受**：本迭代不支持参数化属性（`rotary_dim`、`theta` 等写成字面量，与 config 重复）。这是**已知的重复**，等有需要再加表达式 |
| 6 | `params.*.from` 也要能取字符串列表 | **接受**：否则 40 项 `layer_types` 要抄进描述，成为第二个事实来源 |
| 7 | `--model` 与 `--tp` 同用 | `--model` 路径**忽略** `--tp`（全局 Plan 无 mesh），stderr 一行提示；`--tp` 属于 `check`（D3/D4） |
| 8 | `until` / `{last}` 的语义 | 按实现采纳并固化：`until: "<param>"` = 按该参数计数展开，索引变量 `l`，`0` = 不展开；`{last}` = 最近一次重复展开的最后一个下标 |
| 9 | dtype 在 plan JSON 里的拼写 | 呈现层用 `RsDtype::name()` 的小写拼写；IR 自身的 serde 形态不在本 spec 范围 |
| 10 | fixture 的 `seq` 取 `max_position_embeddings`（262144） | **改为显式小值（512）**：序列长度是**运行期**选择，不是模型常量；262144 会污染形状表并误导后续的本地形状推导 |
| 11 | 声明了却没人写的 slot | **应报错**（`expand` 期）：一个既不被任何节点读、也不被任何节点写的模板 slot 是**死钩子** —— 模板声明的 slot 必须至少是一个节点的输入或输出。**下发到下一单元实现**（本次未做），并补测试 |
| 12 | 新增的 `out` 校验在 CLI 侧没有端到端用例 | 冻结的 `model_description.rs` 覆盖不到它（四个 fixture 补齐后都不触发）。**允许新增一个 fixture + 新测试文件**（不得改冻结的那个），与 #11 一起做 |
| 13 | `select` 与 `template` 同时出现 | **互斥**：有 `select` 时 `template` 必须缺省，否则**报错** —— 两个兜底来源就是"同一个事实两个来源"；`select` 的兜底只有 `default`。§3.3 的示例据此修正 |
| 14 | #13 的校验放在哪一层 | **entry 级最前**（早于 `count <= 0` 早退）：静态错误不该因为别的错误先报而被漏掉；`(None, None)` 同样报错 |
| 15 | #11 的死钩子检查是模板级还是实例级 | **模板级**：只要是 `desc.templates` 里的模板就查（含从未被实例化的）。死数据不论用不用都该报出来 |
| 16 | **JSON `null` 在 `from` 路径上** | **`null` 视为缺失**（HF config 里 `"rope_scaling": null` 就是"没配"的惯用写法，不是类型错误），`default` 生效；只有**非 null 的错误类型**（数字/字符串/数组出现在期望对象的位置）才报错 |

### 3.8 第三轮裁定（D2 实现反馈）

| # | 问题 | 裁定 |
|---|---|---|
| 1 | `ignore` 的**锚定** | 固化：`ignore` 条目必须**以具体名字段开头**；`**`、`*`、`*.visual.**` 一律报错（C5 的"显式声明"：首段是通配符的条目什么也没声明，却长得像一条声明）。落地在 `expand.rs` 的 `check_ignore_patterns`，理由写在同一处注释。<br>**锚定只管第一段**：`model.**` 合法 —— 它声明的是"整个 `model` 子树"，可读、可审 |
| 2 | 模式里的**前后空白** | 视为**字面段**，**不 trim**（张量名匹配是字面的，不做隐式清洗）。后果有界：命中的张量数变 0 → `l2.ignore_coverage` 是 **warning**（C5：同一份描述可能对着别的 checkpoint 检查，0 命中不等于描述错了）；但它本该丢掉的张量若真实存在，就会变成**未消费** → `l2.tensor_consumption` **Fail**。<br>即：空白拼写错误永远不会静默放过一次真实的漏丢 —— 兜底的是"张量消费"那一条，不是模式本身 |

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
| 4. **位置常量** | 把与 rank 有关的**编译期常量**烘进节点属性：flat QKV 的通道偏移、CP 的序列偏移、本地专家范围（`rank * local`）。它们在进程生命周期内不变，是常量不是运行期参数（`architecture.md` §2.2） |
| 5. **重写** | 重写 slot 名称（加 rank 无关的稳定后缀即可，形状已本地化），节点集合即为产物的节点集合 |

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
rustrain check --model <model-dir|desc.json> [--checkpoint <dir>] [--tp N --cp N --ep N --dp N --pp N] [--json]
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

**退路**：若某个实现需要 dispatch 与专家计算分开（例如想单独用共享的 grouped-GEMM kernel），
那就在描述里声明一个**容量上界**（`capacity`），把它当作 dispatch 输出槽的静态形状 ——
代价是描述里多了一个与具体 kernel 相关的常量，收益是这一段回到可规划的原语。
两条路都合法，按 kernel 的形态选。

---

## 7. 明确不做 / 待定

- **不做符号维**：所有形状在 instantiate 后必须具体（沿用 `ir.rs:6-8`）。
- **不做多轴/多步转换**（§2.3）；需要时要求显式写出中间 layout。
- **不做"标量 → 结构"的推导**（§3.1）；派生交给生成器。
- **描述语言暂不含条件/循环**，只有"按列表下标选择"与"计数重复"。若真实模型需要更多，先加用例再改语言。
- **待定**：`split_layers`（PP 的层分配规则）的具体形式；CP 对线性注意力层是否支持（取决于 §8 的证据）；
  `binding` 的 `ignore` 列表写法；描述文件的版本迁移策略。
