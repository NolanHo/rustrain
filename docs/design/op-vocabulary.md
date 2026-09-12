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
| **块** | 可整体替换的实现单元 | **必须声明 `expansion`** 到原语，深度 ≤ 2（spec 契约 P-1） |

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
  l2norm(q)·scale , l2norm(k)·scale                          # scale = 1/sqrt(128)
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
topk_router(h, gate[256,2048], topk=8, norm_topk_prob) → routing
moe_layer(h, routing,                                            # EXPLICIT
          experts.gate_up_proj [256,1024,2048],
          experts.down_proj     [256,2048,512],
          shared_expert.{gate,up,down}_proj [512,2048]/[2048,512],
          shared_expert_gate [1,2048])
  collectives: [ all_to_all({tp, ep}) ] ×2（dispatch + combine 各一次）
  in/out shape: [b, s, 2048]（静态）
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
| 权重 slot | 690（文本）+ MTP 16 |

plan 到千节点量级是正常的，**但这也说明"人读 plan"要靠 `plan explain` 的分层展示**（按模板前缀聚合），
而不是倒出 1000 行。

---

## 7. 迭代 1 的最小交付（reference provider 侧）

1. **新增 5 个原语**（§2）的 reference 实现 + `infer` + `memory` + `shard_rule` 声明。
2. **补齐 3 处 T2 声明**（§2.1）：`rmsnorm` 的两种权重约定、`rope` 的 partial/位置、`sdpa` 的 GQA/mask。
3. **描述文件**：`qwen3.6-35b-a3b.json`（params/templates/stack/binding），按展开形态写。
4. **一致性门禁 case**：每个新原语一条；`sdpa`/`topk_router` 已有。
5. **不做**：任何融合算子（`attn_full`/`gdn_layer`/`moe_layer` 的融合版本）、MRoPE、视觉塔、量化路径。

**验收**：`rustrain check --model <qwen36 dir>` 在无 GPU 机器上过 L1 + L2；
L3 在验证宿主上跑通（数值对齐 HF）。
