# Qwen3.6-35B-A3B 算子词表与分解图

> **状态：设计。** 粒度已定（用户确认）：**可替换块为节点，声明 `expansion` 到原语**；
> 融合存储**拆成多个 slot**（不引入结构化分片）。
>
> 这份文件是**词表的唯一权威**（`docs/design/kernel-first/spec.md` §2.4 已改为指向本文）。
> 模型事实与真实形状见 `docs/design/qwen36-5d-example.md`；描述语言见 `docs/design/model-description.md`。

---

## 0. 粒度规则

| | 定义 | 要求 |
|---|---|---|
| **原语** | 数学上不可再分、且值得独立换实现的东西 | 无 `expansion`；有 `infer`/`memory`/`shard_rule`/`numerics` |
| **块** | 可整体替换的实现单元 | **必须声明 `expansion`** 到原语；P-1 的"深度 ≤ 2"只约束**描述符里自带展开声明**的复合算子（粗描述场景）—— 细描述路径下融合是"模板实例替换"，不产生深层展开（§8.1） |

**两个方向读同一份声明**：

- **描述里的模板**（`docs/design/model-description.md` §3.2）决定"模型结构" —— 迭代 1 直接按**展开形态**写；
- **描述符里的 `expansion`** 声明"这个融合实现等价于哪个子图" —— 解析期用它把子图替换成一个节点。

所以**迭代 1 不需要任何融合算子**：只要原语 + 描述模板。融合是后续的性能工作，
判据是门禁证明"融合体 ≡ 它的 expansion"（`architecture.md` §2.3）。

---

## 1. 现有原语对账（reference provider 已有 27 个）

| 类别 | 已有 | Qwen3.6 用得上 |
|---|---|---|
| 元数据 | `view` `reshape` `transpose` `narrow` `cat` `broadcast` | ✅ 全用（de-fuse、reshape 到 head、MTP 的 cat） |
| L0 计算 | `matmul` `linear` `bmm` `elementwise_unary` `elementwise_binary` `compare` `reduce` `softmax` `rmsnorm` `layernorm` `rope` | ✅ `softmax`/`layernorm` 暂不用；`rope` 需要 partial + 位置输入 |
| 量化 | `quantize` `dequantize` `amax_update` | ❌ 本迭代不用（BF16） |
| 数据搬运 | `embedding` `gather` `scatter` | ✅ `embedding`；`gather`/`scatter` 视 MoE 形态 |
| 复合 | `sdpa` `topk_router` `cross_entropy` `adamw` | ✅ 全用（`sdpa` 要支持 GQA 与 padding mask） |
| 通信 | `all_reduce` `all_gather` `reduce_scatter` | ✅ 全用 |

**结论：27 个里只有 5 个缺口** —— 词表设计经受住了真实模型的检验。

## 2. 需要新增的 5 个原语

| 原语 | 为什么是原语（不是块） | 关键属性 | 用在 |
|---|---|---|---|
| `l2norm` | 无可学参数、按 head dim 归一；没有值得单独调度的内部结构 | `dim`、`eps` | GDN 的 q/k 归一化 |
| `rmsnorm_gated` | 归一化与门控在同一个 reduce 里做才不浪费一趟显存 | `eps`、`gate_act="silu"` | GDN 的 `linear_attn.norm`（`[128]` = value_head_dim） |
| `causal_conv1d` | depthwise 因果卷积（kernel=4），是序列上的邻域操作，不是矩阵乘 | `kernel`、`groups=channels`、`fuse_act="silu"`、`pad` | GDN 的 `conv1d.weight [8192,1,4]` |
| `gated_delta_rule` | 递推本体；CP 的跨 rank 依赖在它内部（见 §4） | 状态精度（`mamba_ssm_dtype=float32`）、`chunk_size` | 30 个 linear 层 |
| `all_to_all` | 通信原语，现有通信集缺它 | 组掩码、split/ 等分 | MoE dispatch/combine、GDN headwise CP |

**`repeat_interleave` 不新增**：GQA 的 KV 头扩展由 `sdpa` 的契约承担（`num_kv_heads < num_heads`）。

**`topk_router` 已够用**：`softmax + top-k + norm_topk_prob` 已在它的 expansion 里；本模型 `topk=8`、`norm_topk_prob` 默认 true。

### 2.1 三个不需要新增、但 T2 要补声明的地方

| 位置 | 现状 | 要补 |
|---|---|---|
| `rmsnorm` | 约定未知 | **两种权重约定**：主干用 `1.0 + weight`，GDN 的门控归一化用原始 `weight`（legacy 实测）。作为属性声明，不是框架假设 |
| `rope` | 位置/布局约定未定（门禁里已记为 skip） | partial rotary（`rotary_dim = head_dim * 0.25 = 64`）、`theta = 1e7`、half-split 旋转、位置张量输入 |
| `sdpa` | 已有 | GQA（16:2）、causal + padding mask、`scale = 1/sqrt(head_dim)` |

---

## 3. 一层的分解图（真实形状）

### 3.1 `decoder_full`（10 层：3, 7, 11, … 39）

```
input_layernorm rmsnorm(1+w)                                    [2048]
  ↓
q_proj linear → [8192]   ⇒ de-fuse: narrow(-1, 0, 4096)=q, narrow(-1, 4096, 4096)=gate
  reshape(q, [b,s,16,256]) → rmsnorm(q_norm[256]) → qn
  k_proj linear → [512]  ⇒ reshape [b,s,2,256] → rmsnorm(k_norm[256]) → kn
  v_proj linear → [512]  ⇒ reshape [b,s,2,256] → v
  rope(qn, kn, rotary_dim=64, theta=1e7) → qr, kr          # 本迭代无 MRoPE
  sdpa(qr, kr, v, causal, gqa=16:2, padding_mask) → o       [b,s,16,256]
  sigmoid(gate) → g ; o * g → og                            # attn_output_gate 恒开
  reshape(og, [b,s,4096]) → linear(o_proj) → y              # row parallel ⇒ Partial
  x + y → x1
  ↓
post_attention_layernorm rmsnorm(1+w)
  ↓
moe_layer（§3.3）
  ↓
x1 + moe_out → x2
```

### 3.2 `decoder_linear`（30 层，GDN）

```
input_layernorm rmsnorm(1+w)
  ↓
in_proj_qkv linear → [8192]  ⇒ de-fuse: Q[0:2048] K[2048:4096] V[4096:8192]
causal_conv1d(Q|K|V, conv1d[8192,1,4], silu) → qkv_c
  in_proj_a → a[32] ; in_proj_b → b[32] ; in_proj_z → z[4096]
  g    = -exp(A_log) * softplus(a + dt_bias)                # elementwise
  beta = sigmoid(b)                                          # elementwise
  reshape q,k → [b,s,16,128] ; l2norm(q,k, dim=-1) ; reshape 回 [b,s,2048]   # 逐 128 维 head 归一（HF: l2norm(...,dim=-1)）
  # scale = 1/sqrt(128) 只作用于 q，且由 gated_delta_rule 内部施加（HF line 279）—— 不占图节点
  gated_delta_rule(qn, kn, v, g, beta) → o                  # 状态 float32
  rmsnorm_gated(o, norm[128], z) → og                        # 权重约定：原始 weight
  reshape(og, [b,s,4096]) → linear(out_proj) → y             # ⇒ Partial
  x + y → x1
  ↓
post_attention_layernorm → moe_layer → x1 + moe_out → x2
```

**`A_log` / `dt_bias` / `in_proj_a` / `in_proj_b` 的形状都是 `[32]`/`[32,2048]`** ——
它们按 **value head** 定义，必须沿该轴切分（`axes {0: [tp]}`），既不能复制也不能整段切。

### 3.3 `moe_layer`（40 层全有；1 个 EXPLICIT 算子）

```
topk_router(h, gate[256,2048], top_k=8, norm_topk_prob=true) → routing_weights, routing_indices
  # 两个输出 = 算子自己的 out0/out1 顺序（weights 先、indices 后）；indices 是 i32 索引，不是精度
  # HF 的 top-k 权重重归一化在这个 transformers 版本里是无条件的（line 773），所以该属性不是摆设
moe_layer(h, routing_weights, routing_indices,                   # EXPLICIT，10 个输入
          experts.gate_proj [256,2048,512],   # 拆开：融合 [E,2I,H] 的 2I 轴正是 TP 切过
          experts.up_proj   [256,2048,512],   # gate/up 语义边界的那条轴（HF 用时才 chunk(2,-1)）
          experts.down_proj [256,2048,512],
          shared_expert.{gate,up,down}_proj [512,2048]/[2048,512],
          shared_expert_gate [1,2048])
  collectives: [ all_to_all({tp, ep}) ] ×2（dispatch + combine 各一次）
  in/out shape: [b, s, 2048]（静态）；无属性 —— top_k / norm_topk_prob 属于 router 节点
```

**为什么是 EXPLICIT**：dispatch 之后每 rank 的 token 数是**运行期**才知道的，而 `Plan` 要求具体形状
（`ir.rs:6-8`）。§5 给三条退路。

### 3.4 `mtp_layer` / `embed` / `final_norm` / `lm_head`

真实张量：`mtp.fc.weight [2048,4096]`、`mtp.pre_fc_norm_{embedding,hidden}.weight [2048]`、
`mtp.norm.weight [2048]`，以及 `mtp.layers.0.*`（与 decoder 层**同构**：full attention + MoE）。

```
norm(embed(ids[t+1])) 与 norm(hidden[t]) ⇒ cat → [4096]
  ↓ mtp.fc linear → [2048]
  ↓ decoder_full 同构的一层（q/k/v/o + mlp）
  ↓ mtp.norm rmsnorm
  ↓ lm_head linear（与主 head 共享或独立，本模型 `tie_word_embeddings=false`）
```

`embed`：`embedding` 原语（词表 `[248320, 2048]`，按 tp 切词表）。
`lm_head`：`linear`（同上，列并行，`parallel_output` 不 gather）。
`final_norm`：`rmsnorm(1+w)`。

---

## 4. 每个位置的通信（按轴掩码）

| 位置 | 通信 | 掩码 | 来源 |
|---|---|---|---|
| `sdpa` 之前（CP>1） | `all_gather` KV（或 ring） | `{cp}` | 数据流推导 |
| `gated_delta_rule` 内部（CP>1） | `all_gather` 仿射映射 + **fp32** 链式 merge | `{cp}` | **算子声明** |
| `o_proj` / `out_proj` / expert `down_proj` 之后 | `all_reduce` | `{tp}` | Partial 兑现 |
| MoE dispatch / combine | `all_to_all` ×2 | `{tp, ep}` | **算子声明**（数据依赖） |
| embedding 之后 | `all_reduce`（或 SP 时 `reduce_scatter`） | `{tp}` | Partial 兑现 |
| PP 边界 | `send_recv` | `{pp}` | 调度器驱动 |
| 梯度 | 补集掩码 `全掩码 \ (切分轴 ∪ {pp})` | — | 布局后果（架构 §1.4） |

**只有两处是"声明"而非"推导"**：GDN 的 CP 仿射合并、MoE 的 dispatch/combine。
其余全部由布局算术推出 —— 这正是"推导只兑现声明"那条口径的实操检验。

---

## 5. MoE 形状的三条路

| 路 | 形状 | 语义 | 代价 |
|---|---|---|---|
| **A `moe_layer` 一体**（推荐，本迭代） | 进出静态 `[b,s,2048]` | 精确（dropless） | 参考实现是一个较大的 kernel；plan 里只有 1 个节点 |
| B capacity 展开 | 声明 `capacity` 作为 dispatch 输出形状 | capacity 足够大时精确；否则丢 token | 可组合（linear+silu），但要么浪费要么改变语义 |
| C 动态形状 | — | — | **不可行**：`Plan` 拒绝符号维 |

迭代 1 选 **A**：语义最干净，且参考实现只要对，不需要快。B 作为后续"想复用 grouped-GEMM kernel"时的退路，
届时要显式声明 capacity 并接受它的语义。

---

## 6. plan 规模估算

| 项 | 数量 |
|---|---|
| 每 linear 层节点 | 约 20（conv 1 + linear 5 + elementwise 6 + norm 3 + 递推 1 + 元数据 4） |
| 每 full 层节点 | 约 18 |
| MoE 每层 | 1（路 A） |
| 总节点 | 30×20 + 10×18 + 40×1 + 头尾 ≈ **1000 节点** |
| 权重 slot | 文本 693（根 3 + 层内合计 80+270+60+280）+ MTP 19 = **712**；其余 333 是视觉塔 |

plan 到千节点量级是正常的，**但这也说明"人读 plan"要靠 `plan explain` 的分层展示**（按模板前缀聚合），
而不是倒出 1000 行。

---

## 7. 迭代 1 的最小交付（reference provider 侧）

1. **新增 5 个原语**（§2）的 reference 实现 + `infer` + `memory` + `shard_rule` 声明。
2. **补齐 3 处 T2 声明**（§2.1）：`rmsnorm` 的两种权重约定、`rope` 的 partial/位置、`sdpa` 的 GQA/mask。
3. **描述文件**：`qwen3.6-35b-a3b.json`（params/templates/stack/binding），按展开形态写。
4. **一致性门禁 case**：每个新原语一条；`sdpa`/`topk_router` 已有。
5. **不做**：任何融合算子（`attn_full`/`gdn_layer`/`moe_layer` 的融合版本）、MRoPE、视觉塔、量化路径。

**验收**：`rustrain check --model <model-dir> [--checkpoint <dir>] [--tp N --cp N --ep N --dp N --pp N]` 在无 GPU 机器上过 L1 + L2；
L3 在验证宿主上跑通（数值对齐 HF）。

---

## 8. 融合的粒度：从原语到整模型（Megakernel 可行性）

"Megakernel" 底下混着三件不同的事，必须分开回答：

| | 是什么 | 支持情况 |
|---|---|---|
| **A 算子级融合** | 把 N 个原语合成一个 kernel（一层 = 1 个 kernel） | **支持**，但要把 `expansion` 的含义收紧（§8.1）+ 一个前置条件（§8.3） |
| **B 执行器级融合** | 一次 launch 跑完整个 step（CUDA graph / persistent kernel） | **天然支持**：`CompiledPlan` 本来就是"扁平 step 列表 + 预分配缓冲"，正是可 capture 的形状。graph-replay 是**执行器变体**，不是 plan 变更 |
| **C 整模型一个算子** | 一个 kernel 里跑完 40 层 | 可以表达（`EXPLICIT`，输入 ids 输出 logits），**但与 PP/EP 冲突**（§8.4） |

### 8.1 收紧 `expansion`：替换点是**模板实例**，不是深度 ≤ 2 的树

spec P-1（`expansion` 深度 ≤ 2）是为"粗描述"场景写的。我们选的是**细描述**：描述本身已经把一层
展开成子图（§3）。于是融合的定义变成：

> **融合 = 一次图重写：把"某个模板实例"的整段子图替换成一个算子节点。**

- 替换点由**名字**定位（`layers.17` 是 `decoder_full` 的一个实例），**不做结构模式匹配** ——
  否则框架就要"认识"某种子图形状，那是 T3 泄漏。
- **不需要在描述符里重复声明深层 expansion**：被替换的子图已经在 plan 里，
  "等价"由门禁数值证明（融合体 vs 展开体，即"两个实现 + 一个数值基准"）。
- 描述符只需要声明"我能替换模板 X 的展开形态"，外加它自己的 `infer`/`numerics`/`collectives`/`memory`。

于是粒度是**每模板实例一个开关**：不融合 → 展开图；融合 `attn_full`/`gdn_layer`/`moe_layer` → 块级；
融合 `decoder_full`/`decoder_linear` → 层级；融合整个 stack → 就要显式加一个"model 模板"（即 C）。

**同一份 plan + 两个 recipe = 融合与展开的对照实验**，门禁负责证等价 ——
这正是"随时装卸 kernel 做对照"的落地形态。

### 8.2 代价必须说清：每次融合都在拿"可检查性"换速度

| 粒度 | plan 节点数（本模型） | L1 能查什么 | 正确性由谁承担 |
|---|---|---|---|
| 原语 | ~1000 | 全查：形状、布局、每个 collective、内存 | 框架 + 门禁 |
| 块（attn / gdn / moe） | ~400 | 块边界 + 块内推导出的布局 | 块内部 = kernel + 门禁 |
| 层 | ~120 | 层边界 + PP/EP 位置 | kernel + 门禁 |
| 整模型 | ~1 | 只有 I/O | 全在 kernel |

**关键**：即使融合，**planner 仍然规划展开形态**，所以 L1 的检查照样全跑，只是 resolve/执行用融合体。
**可检查性不因融合而消失**；消失的是"框架能替你优化内部"的能力 —— 那是自愿放弃的，不是丢失的。

### 8.3 前置条件：让 `save_for_backward_bytes` 活过来

**语义与落地见 `docs/architecture.md` §2.6**（该字段今天在 ABI 里但从没被 planner 累加）。
一句话：融合 kernel 自己保存的中间激活不是 plan 的 slot，planner 看不见 → 投影峰值偏低、**告警不可信**。

**但它不阻塞融合实验**：内存管理整个留空（`architecture.md` §8 **D12**），预算只 Warning 不拦编译。
所以现在可以放心写融合 kernel，等做内存管理时再激活这个钩子。

（第二条已有：融合体声明的 `collectives` 必须等于它替换掉的子图里 planner 会插入的集合，
见 `architecture.md` §2.3。层融合会把 `all_reduce{tp}`、KV `all_gather{cp}`、`all_to_all{tp,ep}`
全吸进一个 kernel，必须声明，否则调度器不知道有通信、无法 overlap、也无法检查。）

### 8.4 边界（诚实说）

- **层融合一旦吸入 CP 的 KV gather**，它就得自己管跨 rank 的 KV —— 那是 attention kernel 的复杂度
  （FlashAttention + ring），"编排"帮不上。
- **C（整模型）与五轴并行正交甚至冲突**：PP 的微批调度与 EP 的专家放置都依赖 plan 里节点可见。
  一个整模型算子没有"哪些层在本 rank"这个概念。所以 C 是"放弃编排换极限性能"的显式选择，不是免费选项。
- 融合体**不能**跨 PP 边界（那是进程间通信），也**不能**跨 `EXPLICIT` 算子的数据依赖边界（MoE 的 routing）。

### 8.5 绕过 plan 是合法的（用户明确）

一个足够强的 kernel 开发者可以**彻底绕过 plan**：把整个模型（或一大块）做成一个算子，
内部自己调度、自己管通信、自己管显存。**这完全 OK**，只是"我们一般不是这么写的" ——
它是 C 层，是自愿放弃编排换极限性能。

框架在这种情况下要求的不是内部结构，而是**契约**：

| 声明 | 不声明的后果 |
|---|---|
| 它是**什么**（id、arity、dtype、numerics） | 描述无法引用它 |
| 它**吸收的通信**（`collectives`） | 调度器不知道有通信，无法 overlap，也无法检查 |
| 它**占的显存**（`workspace_bytes` / `save_for_backward_bytes`，§2.6） | 预算低估，训练时 OOM |
| 它**的等价物**（`expansion`，若有） | 门禁没有基准可比，一致性无从谈起 |

**绕过的是内部结构，不是契约。** 这正是 §0 的边界契约在"粒度"维度上的直接推论：
框架的职责只有"拿到实现、知道契约、验证它算的是同一件事"，**从不包括理解它的内部**。
所以 C 不需要任何新的架构机制 —— 它已经在这条边界之内。
