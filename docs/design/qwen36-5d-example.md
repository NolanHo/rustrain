# Qwen3.6-35B-A3B × 5D 并行（实例走查）

> **目的**：把 `docs/design/model-description.md` 的 schema 在一个**真实模型 + 五轴全开**的配置上走一遍，
> 找出 schema 表达不了的地方。本文是走查记录，不是新设计；暴露的缺口集中在 §8。
>
> **约定的两个方向**（全文一致，后面会用到）：
> - **slot 方向**：kernel 看到的张量，`linear` 的权重是 `[K, N]`（收缩维在前），由 `rustrain-kernels`
>   的 reference provider 固定。
> - **checkpoint 方向**：HF/legacy 的权重是 `[out, in]`。所以 binding 里靠 `transpose` 归一化，
>   **`axes` 永远指 slot 的维度**。
>
> 数字来源：`hidden=2048 / layers=40 / heads=16 / experts=256 / topk=8` 来自
> `_internal_docs/archive/pre-rewrite/qwen3-verification.md`；`linear in_proj_qkv = [Q 2048 | K 2048 | V 4096]`
> 来自同目录 `agent/linear-attention.md`（head_dim=128 由 2048/16 与 4096/32 反推）。
> **其余常量（moe_intermediate、shared_intermediate 等）用符号**，不臆造数字。

---

## 1. 模型事实（来自 legacy 清点）

40 层，层类型来自 `layer_types`（**显式列表，代码里没有取模路径**）：3 层 linear + 1 层 full 交替。

| 层类型 | 子结构 |
|---|---|
| **full attention** | q_proj **带 output gate**（输出 `2*heads*head_dim`，chunk 成 q + gate）→ q_norm/k_norm（**per-head RMSNorm**）→ **partial RoPE** → GQA（`repeat_interleave` KV）→ causal+padding mask → `sigmoid(gate)` 乘回 → o_proj |
| **linear attention（GDN）** | in_proj_qkv（**flat `[Q\|K\|V]`，不是 per-head 交错**）→ causal depthwise conv1d + SiLU → split → in_proj_a/b/z → `g=-exp(A_log)*softplus(a+dt_bias)`、`beta=sigmoid(b)` → L2 normalize q/k → delta rule → gated RMSNorm(SiLU(z)) → out_proj |
| **MoE** | router（softmax + top-8 + `norm_topk_prob` 重归一）→ 本地专家 fused `gate_up_proj` → SiLU×up → down → shared expert + `sigmoid(shared_expert_gate)` |
| **MTP** | `mtp.layers.{j}` 一层 full attention + MoE，`mtp.` 前缀硬编码（忽略 `weight_prefix`） |

RMSNorm 约定：主干用 `1.0 + weight`，gated norm 用原始 weight。

---

## 2. mesh 与 rank 分解

```
axes = [tp:2, cp:2, ep:4, dp:2, pp:2]        world = 2*2*4*2*2 = 64
rank = tp_coord*1 + cp_coord*2 + ep_coord*4 + dp_coord*8 + pp_coord*16
```

**轴顺序即 rank 布局**（§1.1）。几个派生组，全部是掩码：

| 组 | 掩码 | 度数 | 用途 |
|---|---|---|---|
| tp | `{tp}` | 2 | 张量并行 |
| cp | `{cp}` | 2 | 序列并行 |
| ep | `{ep}` | 4 | 专家并行 |
| **tp\|ep** | `{tp, ep}` | 8 | **专家权重同时被两者切**（见 §4.2）；MoE dispatch 的目标组 |
| dp | `{dp}` | 2 | 数据并行（默认前反向副本） |
| **cp\|dp** | `{cp, dp}` | 4 | 激活的 DP 域（cp 是序列切分不产生梯度冲突） |
| 全掩码 | 5 轴 | 64 | global |

`world_size` 校验：`Π degree == world_size`，否则**报错**（今天这条路没人走 —— `shard.rs:236` 丢弃拓扑）。

PP 的层分配：40 层 / 2 stage = 每 stage 20 层；embedding 只在 stage 0、`norm`+`lm_head` 只在 stage 1。

---

## 3. 描述文件（关键片段，非全文）

```jsonc
"params": {
  "hidden":  { "from": "text_config.hidden_size" },          // 2048
  "layers":  { "from": "text_config.num_hidden_layers" },     // 40
  "heads":   { "from": "text_config.num_attention_heads" },   // 16
  "hdim":    { "from": "text_config.head_dim" },              // 128
  "kv_heads":{ "from": "text_config.num_key_value_heads" },
  "q_out":   { "expr": "2 * heads * hdim" },                  // 4096（含 output gate）
  "kh":      { "from": "text_config.linear_num_key_heads" },  // 16
  "kd":      { "from": "text_config.linear_key_head_dim" },   // 128
  "vh":      { "from": "text_config.linear_num_value_heads" },"// 32
  "vd":      { "from": "text_config.linear_value_head_dim" }, // 128
  "qkv_dim": { "expr": "2 * kh * kd + vh * vd" },             // 8192
  "v_dim":   { "expr": "vh * vd" },                           // 4096
  "qt_dim":  { "expr": "kh * kd + vh * vd" },                 // 6144（conv1d 的 channels）
  "experts": { "from": "text_config.num_experts" },           // 256
  "topk":    { "from": "text_config.num_experts_per_tok" },   // 8
  "moe_i":   { "from": "text_config.moe_intermediate_size" },
  "shared_i":{ "from": "text_config.shared_expert_intermediate_size" },
  "layer_types": { "from": "text_config.layer_types" }        // 40 项显式列表
}
```

```jsonc
"stack": [
  { "template": "embed", "prefix": "embed" },
  { "template": "decoder", "prefix": "layers.{l}",
    "repeat": { "count": "layers", "index": "l" },
    "select": { "by": "layer_types[l]",
                "cases": { "full_attention": "decoder_full",
                           "linear_attention": "decoder_linear" },
                "default": "decoder_full" },
    "stage": { "of": "pp", "split": "layers" } },
  { "template": "final_norm", "prefix": "norm" },
  { "template": "lm_head",    "prefix": "lm_head" },
  { "template": "mtp", "prefix": "mtp", "until": "mtp_layers",
    "inputs": { "h": "layers.{last}.y", "ids": "input_ids" } }
]
```

binding 的六类（`axes` 全是 slot 维度）：

```jsonc
[ // 1) 词表切分：HF [vocab, hidden] -> 已经是 [N?, K?]，这里按"行=词表"直接切 dim 0
  { "slot": "embed.weight",  "source": "model.embed_tokens.weight",
    "axes": { "0": ["tp"] }, "transform": [] },
  { "slot": "lm_head.weight", "source": "lm_head.weight",
    "axes": { "0": ["tp"] }, "transform": [], "tied_with": "embed.weight" },

  // 2) full attention：HF [out, in] -> transpose 成 [in, out] = [K, N]
  { "slot": "layers.*.self_attn.q_proj.weight",
    "source": "model.layers.{*}.self_attn.q_proj.weight",
    "transform": ["transpose(0,1)"], "axes": { "1": ["tp"] } },   // column
  { "slot": "layers.*.self_attn.o_proj.weight",
    "source": "model.layers.{*}.self_attn.o_proj.weight",
    "transform": ["transpose(0,1)"], "axes": { "0": ["tp"] } },   // row -> partial
  { "slot": "layers.*.self_attn.q_norm.weight", "source": "...", "axes": {}, "transform": [] },

  // 3) linear attention（GDN）
  { "slot": "layers.*.linear_attn.in_proj_qkv.weight",
    "source": "model.layers.{*}.linear_attn.in_proj_qkv.weight",
    "transform": ["transpose(0,1)"], "axes": { "1": ["tp"] } },   // column，flat [Q|K|V]
  { "slot": "layers.*.linear_attn.conv1d.weight",
    "source": "model.layers.{*}.linear_attn.conv1d.weight",
    "transform": [], "axes": { "0": ["tp"] } },                   // depthwise，跟着 qkv 走
  { "slot": "layers.*.linear_attn.out_proj.weight",
    "source": "model.layers.{*}.linear_attn.out_proj.weight",
    "transform": ["transpose(0,1)"], "axes": { "0": ["tp"] } },   // row -> partial

  // 4) MoE 专家：HF [E, out, in] -> transpose(1,2) 成 [E, in, out]
  { "slot": "layers.*.mlp.experts.gate_up_proj",
    "source": "model.layers.{*}.mlp.experts.gate_up_proj",
    "transform": ["transpose(1,2)"], "axes": { "0": ["ep"], "2": ["tp"] } },
  { "slot": "layers.*.mlp.experts.down_proj",
    "source": "model.layers.{*}.mlp.experts.down_proj",
    "transform": ["transpose(1,2)"], "axes": { "0": ["ep"], "1": ["tp"] } },

  // 5) 复制的参数
  { "slot": "layers.*.mlp.gate.weight",        "source": "...", "axes": {}, "transform": [] },
  { "slot": "layers.*.mlp.shared_expert_gate.weight", "source": "...", "axes": {}, "transform": [] },

  // 6) 共享专家：跟着常规 TP（不是专家 TP）
  { "slot": "layers.*.mlp.shared_expert.gate_proj.weight", "source": "...",
    "transform": ["transpose(0,1)"], "axes": { "1": ["tp"] } }
]
```

---

## 4. 五轴逐轴走查

### 4.1 TP（`{tp}`，度数 2）

| 张量（slot 方向） | 切哪一维 | 本地形状 | 通信 |
|---|---|---|---|
| `q_proj` `[hidden, q_out]` | dim 1 | `[2048, 2048]` | 无（列并行，每 rank 出完整的一段 head） |
| `o_proj` `[q_out, hidden]` | dim 0 | `[2048, 2048]` | **partial → all_reduce({tp})** |
| `k/v_proj` `[hidden, kv_heads*hdim]` | dim 1 | — | 无 |
| `q_norm`/`k_norm` `[hdim]` | 无（复制） | `[128]` | 梯度要 all_reduce({tp})（见 §4.4） |
| `in_proj_qkv` `[hidden, qkv_dim]` | dim 1 | `[2048, 4096]` | 无 |
| `conv1d` `[qt_dim, 1, k]` | dim 0 | `[3072, 1, 4]` | 无（depthwise，跟着 qkv 的分片走） |
| `linear_attn.out_proj` `[v_dim, hidden]` | dim 0 | `[2048, 2048]` | **partial → all_reduce({tp})** |
| `embed`/`lm_head` `[vocab, hidden]` | dim 0（词表） | `[vocab/2, 2048]` | embedding 前向 `all_reduce`(或 SP 时 `reduce_scatter`)；lm_head 前向不 gather（`parallel_output`） |
| shared expert | dim 1 / dim 0 | — | 同 dense MLP |

**Flat QKV 的坑**：`in_proj_qkv` 的输出是 `[Q 2048 | K 2048 | V 4096]`，在 dim 1 上按 tp 切一刀会
**切开 Q/K/V 的边界**（tp=2 时 rank 0 拿 Q 全部 + K 前半，rank 1 拿 K 后半 + V 全部）。
这是**合法的**（每个 rank 拿到连续一段输出通道，conv1d 与后续 split 都按本地段处理），
但**要求 `conv1d` 与 `split` 的边界跟着走**：`linear_attn` op 必须知道自己那一段从全局的哪个偏移开始。

→ 这一类"位置相关"的量是**编译期常量**（`offset = rank * local`），由 instantiate 烘进节点属性，
不是运行期参数（`architecture.md` §2.2）。

### 4.2 EP（`{ep}`，度数 4）—— 与 TP 叠在同一个张量上

`experts.gate_up_proj`：全局 `[E=256, K=hidden, N=2*moe_i]`，声明 `{0: ["ep"], 2: ["tp"]}`。

```
local shape = [256/4, hidden, 2*moe_i/2] = [64, 2048, moe_i]
```

**两条切分互不相干**：dim 0 沿 ep（各 rank 管不同的专家），dim 2 沿 tp（每个专家自己的输出特征被切）。
这正是 `ParallelLayout` 必须从单值 `{dim, group}` 推广成**多个分片**的原因 —— 单值表达不了。

**激活侧的 routing 推不出来**：token 去哪个 rank 取决于 router 的运行期 top-8，所以它是图中的显式算子：

```
router(replicated [256, 2048])  ──topk──▶  dispatch  ──all_to_all({ep})──▶  expert_ffn(local)  ──combine──▶  out
```

- `dispatch` / `combine` 是 `EXPLICIT` 算子，声明 `collectives: [all_to_all({tp, ep})]`
  （Megatron 的 dispatcher 正是在 `tp_ep` 组上做 all-to-all，`token_dispatcher.py:84-85`）。
- 它们的输入输出形状必须**静态声明**（容量上限），因为 `Plan` 不允许符号维（`ir.rs:6-8`）。
- router 与 `shared_expert_gate` **复制**（Megatron 同样：`router.py:77-78`、`shared_experts.py:134`），
  每个 rank 独立算出相同的 top-k。
- **专家权重不参与 EP 之外的梯度归约**（见 §4.4）。

### 4.3 CP（`{cp}`，度数 2）—— 两种层两种机制

全注意力的三种方案（Megatron `cp_comm_type`）：ring / all-gather KV / Ulysses a2a。
Qwen3.6 的序列分片是**连续块**（与 legacy GLM5 一致：`narrow(1, cp_rank*s_local, s_local)`）。

**full attention 层**：序列切 cp 块，attention 前需要完整 KV →
`all_gather({cp})`（或 ring）；反向 `reduce_scatter`。GQA 的 KV heads 在所有 cp rank 上都需要。
causal mask 用**全局位置**（本地 chunk 的 offset 是编译期常量）。

**linear attention 层（GDN）—— 这里是最不确定的一块，实测结论是"可以做，但有明确约束"**：

- 门控 delta 规则是**可结合的仿射递推**：`S_t = S_{t-1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ`，
  组合算子 `(M_i, X_i) ∘ (M_j, X_j) = (M_i M_j, M_j X_i + X_j)`，即矩阵乘 + 加
  （DeltaNet 作者博客、Gated DeltaNet 论文 Eq.10、Mamba-2 附录 B.3.2）。所以**并行扫描存在**。
- Megatron 默认方案是 `linear_cp_mode="chunkwise"`（FLA 后端）：每个 rank 算本地 chunk 的仿射映射
  `(M, S_ext)` → **一次 all-gather** → **fp32 链式 merge `h' = M @ h + he`**。
  另一模式 `headwise` 是 Ulysses 式 all-to-all：每个 rank 拿部分 head 跑**完整序列**，
  文档自述"correct but memory-heavy"。
- 硬约束：`total_tokens % cp == 0`；**M 链必须 fp32**（bf16 反复回写累加器会显著放大误差）；
  CP 下不支持 `initial_state` / `output_final_state`；`gdn_conv_pad_alignment` 与之不兼容。
- **服务器侧还是坑**：vLLM 在 hybrid 上直接 `assert pcp_world_size == 1`；SGLang 的 PR 记录
  开启 prefill CP 后"不报错，但循环状态被静默破坏"。→ 这正好是我们 §0 那条边界要防的东西：
  **静默错误比报错更贵**，所以它在我们的设计里必须是**声明 + 门禁**，而不是默认打开。

**对 schema 的后果**：GDN 层是 `EXPLICIT` 算子，声明 `collectives: [all_gather({cp})]`
（chunkwise）或 `[all_to_all({cp})]`（headwise）；序列在两端的 layout 都是 `Shard{dim: 1, {cp}}`，
中间那步跨 rank 的 merge 在算子内部，plan 看不见 —— 与 MoE 的 dispatch/combine 完全同构。

### 4.4 DP（`{dp}`，度数 2）—— 梯度归约组 = 掩码补集

Megatron 的做法是给每个参数打布尔标记（`allreduce`），专家参数走 expert-DP 组
（`layers.py:928/1278`、`distributed_data_parallel.py:219-223`）。**在我们这里它可以被推导出来**：

```
grad_reduce_mask(param) = 全掩码 \ (该张量已切分的轴 ∪ {pp})
```

| 张量 | 已切分轴 | 梯度归约组 | 掩码度数 |
|---|---|---|---|
| `q_proj`（tp 切） | tp | `{cp, ep, dp}` | 8 |
| `q_norm`（复制） | ∅ | `{tp, cp, ep, dp}` | 16 |
| 专家权重（tp+ep 切） | tp, ep | `{cp, dp}` | 4 |
| `embed`（tp 切词表） | tp | `{cp, ep, dp}` | 8 |

**为什么必须这样**：被切分的维度上每个 rank 持有的是**不同的参数**，跨它们 all-reduce 是错的；
只有复制出来的那份才需要跨副本求和。**这就是"`axes` 声明"的又一个后果** ——
不需要额外的标记位，也不需要在训练循环里手写。

两个 Megatron 实测的补充：
- **tie 的 embedding 跨 PP stage** 要一次 `all_reduce`（`language_model.py:294`），因为 stage 0 与
  stage 1 各有一份；同 stage 内则用 `zero_out_wgrad`。
- **DP 用 all-reduce 还是 reduce-scatter** 取决于优化器是否分片（`param_and_grad_buffer.py:761-783`）——
  属于 D5（训练循环/优化器），本文不展开。

### 4.5 PP（`{pp}`，度数 2）—— 唯一改变节点集合的轴

| 项 | 结果 |
|---|---|
| 节点集合 | stage 0：embed + layers 0–19；stage 1：layers 20–39 + norm + lm_head |
| 边界张量 | `[seq/cp (=seq/2), micro_batch, hidden]`；**词表维不跨界**（在边界前已归约到 hidden） |
| 跨界通信 | `isend`/`irecv`（Megatron 用 `batch_isend_irecv`，`p2p_communication.py:29-49`） |
| MTP | `logsits`：MTP 层属于最后一个 stage；`mtp.layers.{j}` 的编号接在 trunk 之后 |
| 调度 | 微批 + 1F1B —— **独立子系统**，plan 只声明边界 slot |

**这印证了 §1.4 的修正**：五轴里只有 PP 改节点集合，也只有 PP 需要调度器。

---

## 5. 本地形状表（`tp=2, cp=2, ep=4`，PP 与 DP 不改形状）

| slot（slot 方向） | 全局形状 | 分片声明 | 本地形状 |
|---|---|---|---|
| `embed.weight` | `[vocab, 2048]` | dim0:{tp} | `[vocab/2, 2048]` |
| `self_attn.q_proj.weight` | `[2048, 4096]` | dim1:{tp} | `[2048, 2048]` |
| `self_attn.o_proj.weight` | `[4096, 2048]` | dim0:{tp} | `[2048, 2048]` + partial |
| `self_attn.q_norm.weight` | `[128]` | — | `[128]`（复制） |
| `linear_attn.in_proj_qkv.weight` | `[2048, 8192]` | dim1:{tp} | `[2048, 4096]` |
| `linear_attn.conv1d.weight` | `[6144, 1, 4]` | dim0:{tp} | `[3072, 1, 4]` |
| `linear_attn.out_proj.weight` | `[4096, 2048]` | dim0:{tp} | `[2048, 2048]` + partial |
| `mlp.experts.gate_up_proj` | `[256, 2048, 2*moe_i]` | dim0:{ep}, dim2:{tp} | `[64, 2048, moe_i]` |
| `mlp.experts.down_proj` | `[256, moe_i, 2048]` | dim0:{ep}, dim1:{tp} | `[64, moe_i/2, 2048]` + partial |
| `mlp.gate.weight` | `[256, 2048]` | — | `[256, 2048]`（复制） |
| 激活 `x`（序列） | `[seq, 2048]` | dim0:{cp} | `[seq/2, 2048]` |

---

## 6. 一个 decoder 层（full attention + MoE）的通信清单

| 位置 | 通信 | 组掩码 | 来源 |
|---|---|---|---|
| q/k/v 之后 | 无（列并行） | — | 布局算术 |
| attention 前 | `all_gather` KV | `{cp}` | CP 的数据流需求 |
| o_proj 后 | `all_reduce` | `{tp}` | partial 兑现 |
| MoE dispatch/combine | `all_to_all` ×2 | `{tp, ep}` | 算子声明（数据依赖） |
| shared expert 之后 | `all_reduce` | `{tp}` | partial 兑现 |
| 层末（梯度） | 按 §4.4 的补集掩码 | — | 布局后果 |

linear attention 层把第二行的 `all_gather KV` 换成 **GDN 内部的一次 `all_gather`（仿射映射）**。

---

## 7. 反向与优化器（一句话，属 D4/D5）

`Phase::Backward` 目前要求每个算子有显式反向实现（`compile.rs:388-400` 拒绝 autodiff）。
5D 不改变这一点，但把梯度归约组变成了**布局的函数**（§4.4）—— 这是训练循环需要的唯一新信息。

---

## 8. 这次走查暴露的缺口（需回填 design doc）

1. **`axes` 的基准方向必须写死**：本文用"slot 方向 + `transpose` 归一化"。design doc §3.4 的例子写在
   原始 checkpoint 方向（`{0:[ep], 2:[tp]}` 对 down_proj），**与本文不一致，必须改**。
2. **binding 需要"一个 source → 多个 slot"**：fused `gate_up_proj` 在 `[gate|up]` 布局下按 tp 切
   **不是连续切片**（rank 0 拿到全部 gate 行、rank 1 拿到全部 up 行，本地无法计算）。
   解法：binding 允许 `split(dim, sizes)` 把一个 checkpoint 张量拆成多个 slot（gate / up），
   各自声明 `axes`。**"融合存储"与"融合计算"是两件事** —— 拆开存储不影响 kernel 同时读两个指针做融合。
   （Megatron 靠 stride=2 的交错存储绕过这件事，那是它自己的 checkpoint 格式，HF 不是。）
3. **"位置相关常量"需要一个正式位置**：flat QKV 的偏移、CP 的序列偏移、本地专家范围都要在
   instantiate 时烘进节点属性。design doc §4.2 只写了"重写 slot 名称"，要补这一条。
4. **EXPLICIT 算子的 `collectives` 需要语法**：MoE 的 all_to_all、GDN 的 all_gather 都是声明而非推导。
   design doc §2.3 只覆盖了"融合体 vs expansion"的校验，没写 EXPLICIT 的 `collectives` 怎么声明与校验。
5. **形状静态性对 MoE 的含义**：dispatch 的容量上界要么由描述声明，要么整块 MoE 作为一个算子
   （本文按后者）。design doc §6.2 已经提到，但没有给出"容量"这条退路。
6. **梯度归约组 = 掩码补集**这条是从 5D 走查里得到的**新结论**，应该进 `architecture.md`（DP 的定义）
   与 training 层的设计。
