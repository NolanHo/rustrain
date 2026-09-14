# Qwen3.6-35B-A3B × 5D 并行（真实数据走查）

> **目的**：把 `docs/design/model-description.md` 的 schema 在真实模型 + 五轴全开上走一遍。
> 本文的模型事实**全部来自真实产物**，不是推测：
> - `config.json`：<https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json>（2026-04-24，sha `995ad96e`）
> - 张量命名：`model.safetensors.index.json`（1045 个张量，71.9 GB）
> - 张量形状：对分片头部做 HTTP Range 读取（**不下载 72 GB**），读的是 safetensors 头部 JSON
> - 本地副本：验证宿主 `/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B`
>
> **方向约定**：`axes` 永远指 **slot 维度**（`linear` 权重是 `[K, N]`，收缩维在前）；
> HF checkpoint 是 `[out, in]`，由 `transform` 的 `transpose` 归一化。

---

## 1. 配置事实（config.json 原文）

| 键 | 值 | 用途 |
|---|---|---|
| `architectures` | `Qwen3_5MoeForConditionalGeneration` | **多模态**（含视觉塔） |
| `text_config.model_type` | `qwen3_5_moe_text` | |
| `hidden_size` | **2048** | |
| `num_hidden_layers` | **40** | |
| `layer_types` | **40 项显式列表**（3 linear + 1 full 循环） | 层类型是**数据**，`full_attention_interval: 4` 解析了但用不上 |
| `num_attention_heads` | 16 | |
| `num_key_value_heads` | **2** | GQA 8:1 |
| `head_dim` | **256** | **不是 hidden/heads（128）** —— 旧代码两处注释都记着这件事 |
| `attn_output_gate` | true | q_proj 输出 **2×** |
| `partial_rotary_factor` | 0.25 | rotary_dim = 64 |
| `rope_parameters` | `mrope_interleaved: true`、`mrope_section: [11,11,10]`、`rope_theta: 1e7` | |
| `linear_num_key_heads` / `linear_key_head_dim` | 16 / 128 | Q=K=2048 |
| `linear_num_value_heads` / `linear_value_head_dim` | 32 / 128 | V=4096 |
| `linear_conv_kernel_dim` | 4 | |
| `mamba_ssm_dtype` | `float32` | delta 规则的状态精度 |
| `num_experts` / `num_experts_per_tok` | 256 / 8 | |
| `moe_intermediate_size` / `shared_expert_intermediate_size` | **512 / 512** | |
| `mtp_num_hidden_layers` | 1 | |
| `vocab_size` | 248320 | |
| `tie_word_embeddings` | **false** | `lm_head.weight` 是独立张量（形状与 embedding 相同） |
| `rms_norm_eps` | 1e-6 | |
| `max_position_embeddings` | 262144 | |
| `vision_config` | depth 27、hidden 1152、patch 16、spatial_merge 2、out_hidden 2048 | **本次范围外**（§7） |

---

## 2. 张量清单（真实，1045 个）

命名前缀随 `has_vision` 变化：本 checkpoint 是 **`model.language_model.`**（不是 `model.`），
MTP 是独立的 **`mtp.`** 前缀。

| 组 | 张量 | 数量 |
|---|---|---|
| 根 | `model.language_model.{embed_tokens,norm}.weight`、`lm_head.weight` | 3 |
| 每层公共 | `layers.{l}.{input_layernorm,post_attention_layernorm}.weight` | 80 |
| **linear attention**（30 层） | `.linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,conv1d,out_proj}.weight` + `.{A_log,dt_bias,norm.weight}` | 270 |
| **full attention**（10 层） | `.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight` | 60 |
| **MoE**（40 层） | `.mlp.gate.weight`、`.mlp.experts.{gate_up_proj,down_proj}`、`.mlp.shared_expert.{gate,up,down}_proj.weight`、`.mlp.shared_expert_gate.weight` | 280 |
| MTP | `mtp.{fc,norm,pre_fc_norm_embedding,pre_fc_norm_hidden}.weight` + `mtp.layers.0.*`（与 decoder 层同构） | 19 |
| 视觉 | `model.visual.*`（blocks ×27、merger、patch_embed、pos_embed） | 333 |

**注意两个没有 `.weight` 后缀的参数**：`mlp.experts.gate_up_proj` 与 `mlp.experts.down_proj` ——
它们是把 256 个专家堆在一个张量里的**参数**，不是子模块权重。

### 真实形状

```
model.language_model.embed_tokens.weight            [248320, 2048]
model.language_model.norm.weight                    [2048]
lm_head.weight                                      [248320, 2048]
layers.*.linear_attn.in_proj_qkv.weight             [8192, 2048]     # Q 2048 | K 2048 | V 4096
layers.*.linear_attn.in_proj_z.weight               [4096, 2048]
layers.*.linear_attn.in_proj_a.weight               [32, 2048]
layers.*.linear_attn.in_proj_b.weight               [32, 2048]
layers.*.linear_attn.conv1d.weight                  [8192, 1, 4]     # depthwise over Q|K|V
layers.*.linear_attn.A_log                          [32]
layers.*.linear_attn.dt_bias                        [32]
layers.*.linear_attn.norm.weight                    [128]            # = value_head_dim，不是 4096
layers.*.linear_attn.out_proj.weight                [2048, 4096]
layers.*.self_attn.q_proj.weight                    [8192, 2048]     # 2 * 16 * 256（含 output gate）
layers.*.self_attn.k_proj.weight                    [512, 2048]      # 2 * 256
layers.*.self_attn.v_proj.weight                    [512, 2048]
layers.*.self_attn.o_proj.weight                    [2048, 4096]     # 16 * 256
layers.*.self_attn.q_norm.weight                    [256]            # head_dim
layers.*.self_attn.k_norm.weight                    [256]
layers.*.mlp.gate.weight                            [256, 2048]
layers.*.mlp.experts.gate_up_proj                   [256, 1024, 2048]  # [E, 2*512, H]
layers.*.mlp.experts.down_proj                      [256, 2048, 512]   # [E, H, 512]
layers.*.mlp.shared_expert.{gate,up}_proj.weight    [512, 2048]
layers.*.mlp.shared_expert.down_proj.weight         [2048, 512]
layers.*.mlp.shared_expert_gate.weight              [1, 2048]
mtp.fc.weight                                       [2048, 4096]     # cat(e,h) -> hidden
```

---

## 3. 融合存储与 TP 边界：**两处不对齐，一处反而对齐**

> **本节曾被 HF 源码推翻一次。** 早期版本断言三处融合都需要 de-fuse，**`q_proj` 那条是错的**。
> 形状完全看不出来（两种排法都是 `[8192] → [4096, 4096]`），必须读源码。以下是从
> `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` 逐行确认的结果。

源码里的三行（原文）：

```python
query_states, gate = torch.chunk(self.q_proj(x).view(b, s, -1, head_dim * 2), 2, dim=-1)  # :797
gate, up          = F.linear(x, self.gate_up_proj[e]).chunk(2, dim=-1)                      # :875
query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)           # :605
```

| 张量 | 真实段序 | TP 连续切一刀 | 结论 |
|---|---|---|---|
| `q_proj` `[8192]` | **per-head 交错**：`[q₀(256) \| gate₀(256) \| q₁ \| gate₁ \| …]`（先 view 成 `[16, 512]` 再 `chunk(2, -1)`） | 每 rank 拿到**完整的 head 对** | **不需要 de-fuse** —— 连续切反而正确 |
| `in_proj_qkv` `[8192]` | 连续 `[Q(2048) \| K(2048) \| V(4096)]` | rank0 = Q+K，rank1 = V | **必须 de-fuse**（否则 delta 规则缺输入） |
| `experts.gate_up_proj` `[256, 1024]` | 连续 `[gate(512) \| up(512)]` | rank0 = 只有 gate 行，rank1 = 只有 up 行 | **必须 de-fuse** |

**解法**：

- **`in_proj_qkv` / `gate_up_proj`**：binding 的 `split` 把融合张量拆成语义段，每段再声明自己的 `axes`。
- **`q_proj`**：**保持一个 slot**，按输出维连续切分（`axes {1: ["tp"]}`）；q 与 gate 的**分离在激活上做**，
  即模板里用 `reshape` + `narrow` 两个原语节点取出 `[16,256]` 的两半 —— 它们是 plannable 原语，
  于是"哪一半是 q"这件事**写在图里**，不是加载期的隐式约定。

```jsonc
// 需要拆的（两处）
{ "source": "model.language_model.layers.{*}.linear_attn.in_proj_qkv.weight",
  "transform": ["transpose(0,1)"],                                    // -> [2048, 8192] = [K, N]
  "split": { "dim": 1, "sizes": ["lin_qk", "lin_qk", "lin_v"] },      // Q | K | V
  "targets": [ { "slot": "…in_proj_qkv.q", "axes": { "1": ["tp"] } }, … ] }

// 不需要拆的（一处）
{ "slot": "layers.*.self_attn.qg", "source": "…self_attn.q_proj.weight",
  "transform": ["transpose(0,1)"], "axes": { "1": ["tp"] } }          // 一个 slot，交错布局，切分正确
```

**教训**：段序是**语义**，形状证明不了它。这类事实只有两个来源：读参考实现的源码，或数值探针。
D5 与 HF 对齐时会再验一次（那是唯一能证伪的机械手段）。

**另一个必须的结构化切分**：GDN 的 `A_log` / `dt_bias` 是**按 value head** 定义的（`[32]`），
而 `in_proj_a/b` 的输出也是 32。它们必须沿 value-head 轴切（Megatron 同样：`partition_dim=0`，
`gated_delta_net/common.py:272-278`），不能复制也不能整段切。

---

## 4. TP 的可整除性约束（L1 能机械检查）

| 约束 | 值 | 违反时 |
|---|---|---|
| `num_attention_heads % tp == 0` | 16 % tp | 报错 |
| `num_key_value_heads % tp == 0` **或** 声明式 KV 复制 | 2 % tp | tp≥4 时用声明式复制：`"mode": "replicate", "unit": "head_dim"`，每个 rank 拿它的 q 头需要的 kv 头（`spec.md` §D6.6）。Megatron 那条退化路径（`attention.py:352` 置 1 后 all-gather）在这里不需要 —— 复制是声明出来的，不是把度数降成 1 绕过去 |
| `linear_num_value_heads % tp == 0` | 32 % tp | 报错 |
| `linear_num_key_heads % tp == 0` | 16 % tp | 报错 |
| `moe_intermediate_size % tp == 0` | 512 % tp | 报错 |
| `num_experts % ep == 0` | 256 % ep | 报错 |
| `vocab_size % tp == 0`（可 padding） | 248320 % tp | 248320 = 2^7×5×388，tp≤8 可整除 |
| `seq % cp == 0` | | 报错 |
| `num_hidden_layers` 在 PP 上分配 | 40 % pp | 报错 |

**这些是描述里的参数与 mesh 的纯算术关系**，所以 L1 能在无 GPU 机器上全部查掉 —— 正是
`architecture.md` §4.4 说的"声明之间自洽"。

---

## 5. 五轴走查（`tp=2, cp=2, ep=4, dp=2, pp=2`，world=64）

```
axes = [tp:2, cp:2, ep:4, dp:2, pp:2]
rank = tp*1 + cp*2 + ep*4 + dp*8 + pp*16
```

### 5.1 TP（`{tp}`=2）

> **方向约定**：本表**统一用 slot 方向**（`linear` 的权重是 `[K, N]`，收缩维在前），
> 与 binding 的 `axes` 完全一致 —— **binding 是唯一的执行事实**，本表只用于人工对照。
> 早期版本的表混用了 checkpoint 方向（`experts.*`、`in_proj_a/b`、`embed/lm_head` 那几行），已修正。
> checkpoint 方向的真实形状见 §2。

| slot（slot 方向） | 全局 | 切哪维 | 本地 | 通信 |
|---|---|---|---|---|
| `qg`（`q_proj`，**交错布局**）`[2048, 8192]` | 8192 = 16 × 512 | dim1 | `[2048, 4096]` | 无 —— 交错布局使连续切分正好给出完整 head 对（§3）；q/gate 的分离在**激活**上用 `reshape`+`narrow` 做 |
| `k_proj` / `v_proj` `[2048, 512]` | 2 kv 头 × 256 | dim1 | `[2048, 256]` | 无（tp≤2 够分；**tp≥4 需复制 KV**，见 §4） |
| `o_proj` `[4096, 2048]` | | dim0（收缩） | `[2048, 2048]` | **partial → all_reduce{tp}** |
| `q_norm` / `k_norm` `[256]` | | — | `[256]` | 复制；梯度需 all_reduce{tp} |
| `in_proj_qkv` → Q、K `[2048, 2048]`；V `[2048, 4096]` | | dim1 各自 | Q/K `[2048, 1024]`、V `[2048, 2048]` | 无（连续 `[Q\|K\|V]`，必须 de-fuse，§3） |
| `conv1d` 同拆 3 段：`[2048,1,4]`×2、`[4096,1,4]` | | dim0 | 各减半 | 无（depthwise，按通道切是精确的） |
| `A_log` / `dt_bias` `[32]` | | dim0 | `[16]` | 无 |
| `in_proj_a` / `in_proj_b` `[2048, 32]` | | dim1 | `[2048, 16]` | 无 |
| `in_proj_z` `[2048, 4096]` | | dim1 | `[2048, 2048]` | 无 |
| `linear_attn.norm` `[128]` | | — | `[128]` | 复制（按 head_dim，跨 head 共享） |
| `out_proj` `[4096, 2048]` | | dim0（收缩） | `[2048, 2048]` | **partial → all_reduce{tp}** |
| `experts.gate_up_proj` → gate / up `[256, 2048, 512]` | | dim2（**再叠 ep 切 dim0**） | `[256, 2048, 256]` | 无（连续 `[gate\|up]`，必须 de-fuse，§3） |
| `experts.down_proj` `[256, 512, 2048]` | | dim1（收缩，**再叠 ep 切 dim0**） | `[256, 256, 2048]` | **partial → all_reduce{tp}** |
| `mlp.gate` `[256, 2048]`、`shared_expert_gate` `[1, 2048]` | | — | 复制 | 复制（每 rank 算相同的 top-k） |
| `shared_expert.gate_proj`/`up_proj` `[2048, 512]`、`down_proj` `[512, 2048]` | | dim1 / dim0 | | all_reduce{tp} |
| `embed.w` `[248320, 2048]`（embedding，**不是 `[K,N]`**） | | dim0（词表） | `[124160, 2048]` | 前向 all_reduce{tp}（SP 时 reduce_scatter） |
| `lm_head.w` `[2048, 248320]` | | dim1（词表） | `[2048, 124160]` | 不 gather（parallel output） |

### 5.2 EP（`{ep}`=4）

专家张量 dim0 按 `[E/4] = 64` 切。**激活侧 routing 推不出来** → 显式算子：

```
router(复制 [256,2048]) ──top8──▶ dispatch ──all_to_all({tp,ep})──▶ expert_ffn(64 个本地专家) ──combine──▶ out
```

`dispatch`/`combine` 是 `EXPLICIT` 算子，声明 `collectives: [all_to_all({tp,ep})]`，输入输出形状静态。
**专家权重在 dim0{ep} + dim1/dim2{tp} 上同时被切** —— 这是 layout 必须是"多个 (dim,group)"的实证。

### 5.3 CP（`{cp}`=2）

| 层类型 | 机制 | 声明 |
|---|---|---|
| full attention（10 层） | 序列切连续块，attention 前要完整 KV | `all_gather({cp})`（或 ring） |
| linear attention（30 层） | **chunkwise 仿射合并**：每 rank 算本地 `(M, S_ext)` → 一次 all-gather → **fp32** 链式 `h' = M@h + he` | `all_gather({cp})` |

门控 delta 规则**可结合**（`S_t = S_{t-1}M_t + X_t`，组合 = 矩阵乘 + 加），所以并行扫描存在；
Megatron 默认 `linear_cp_mode="chunkwise"`（FLA 后端）。硬约束：`total % cp == 0`、M 链 fp32、
CP 下无 `initial_state`。服务器侧仍会**静默出错**（vLLM 直接禁、SGLang 有 PR 记录状态被破坏）——
所以这里必须"声明 + 门禁"，不能默认打开。

### 5.4 DP（`{dp}`=2）

```
grad_reduce_mask(param) = 全掩码 \ (该张量已切分的轴 ∪ {pp})
```

| 张量 | 已切分轴 | 归约组 | 度数 |
|---|---|---|---|
| `k_proj`（tp 切） | tp | `{cp,ep,dp}` | 8 |
| `q_norm`（复制） | ∅ | `{tp,cp,ep,dp}` | 16 |
| 专家权重（tp+ep） | tp, ep | `{cp,dp}` | 4 |
| `embed`（tp 切词表） | tp | `{cp,ep,dp}` | 8 |

### 5.5 PP（`{pp}`=2）

stage 0：`embed` + layers 0–19；stage 1：layers 20–39 + `norm` + `lm_head` + `mtp`。
边界张量 `[seq/cp=seq/2, micro_batch, 2048]`；词表维不跨界。**唯一改节点集合的轴**，也是唯一需要
微批 + send/recv 调度的轴。

---

## 6. MoE 是 EXPLICIT 算子的实证

`dispatch` 之后每个 rank 拿到的 token 数**运行期才知道**，而 `Plan` 要求形状具体（`ir.rs:6-8`
拒绝符号维）。所以整块 MoE FFN 是一个算子：输入输出都是静态 `[seq, 2048]`，内部做 routing +
dispatch + 分组 GEMM + combine；专家权重是独立 slot，由 binding 声明 `axes`。

**退路**：若要把 dispatch 与专家计算分开（用共享的 grouped-GEMM kernel），就在描述里声明
`capacity` 作为 dispatch 输出槽的静态形状。两条路都合法，按 kernel 形态选。

---

## 7. 明确的范围

**第一个验证样本 = 这个模型的文本解码路径 + 五轴并行。** 通过了再加其他模型；下面这些不是永久排除，
只是这一轮的边界：

- **视觉塔**（333 个张量，27 层 ViT + merger + patch_embed）。它是独立子图，文本路径不依赖它，
  也不需要改 schema —— 真要做多模态时它就是一个独立模板。
- **MRoPE 的三段分解**（`mrope_section: [11,11,10]`）：`mrope_*` 字段在旧代码里解析了但没用。
  文本位置编码先用标准 RoPE + partial rotary；MRoPE 作为**一个 op 的形态**后补（T2）。
