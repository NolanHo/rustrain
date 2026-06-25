# Plan: DeepSeek-V3 Support

## Summary

新增 `rustrain-deepseek` crate，支持 DeepSeek-V3.1 的 forward + training。
DeepSeek-V3 有两个 Qwen3 没有的核心架构：MLA (Multi-head Latent Attention) 和
shared expert MoE。

## DeepSeek-V3 架构

### Config

```
model_type: deepseek_v3
hidden_size: 7168, num_hidden_layers: 61, num_attention_heads: 128
tie_word_embeddings: false (有独立 lm_head.weight)
vocab_size: 129280

MLA:
  kv_lora_rank: 512        — KV 压缩维度
  q_lora_rank: 1536        — Q 压缩维度
  qk_nope_head_dim: 128    — 非旋转部分 head dim
  qk_rope_head_dim: 64     — 旋转部分 head dim
  v_head_dim: 128          — value head dim
  rope_theta: 10000
  rope_scaling: yarn (factor=40)

MoE:
  n_routed_experts: 256, num_experts_per_tok: 8
  n_shared_experts: 1, moe_intermediate_size: 2048
  layer 0-2: dense MLP, layer 3-60: MoE (shared + routed experts)
```

### 每层权重名

**Attention (MLA) — 所有层相同**：
```
model.layers.N.self_attn.q_a_proj.weight          # [q_lora_rank, hidden]
model.layers.N.self_attn.q_a_layernorm.weight      # [q_lora_rank]
model.layers.N.self_attn.q_b_proj.weight           # [heads*(qk_nope+qk_rope), q_lora_rank]
model.layers.N.self_attn.kv_a_proj_with_mqa.weight # [kv_lora_rank+qk_rope, hidden]
model.layers.N.self_attn.kv_a_layernorm.weight      # [kv_lora_rank]
model.layers.N.self_attn.kv_b_proj.weight           # [heads*(qk_nope+v_head), kv_lora_rank]
model.layers.N.self_attn.o_proj.weight              # [hidden, heads*v_head]
```

**Dense MLP (layer 0-2)**：
```
model.layers.N.mlp.gate_proj.weight    # [intermediate, hidden]
model.layers.N.mlp.up_proj.weight      # [intermediate, hidden]
model.layers.N.mlp.down_proj.weight    # [hidden, intermediate]
```

**MoE (layer 3-60)**：
```
model.layers.N.mlp.gate.weight                    # router: [256, hidden]
model.layers.N.mlp.shared_experts.gate_proj.weight
model.layers.N.mlp.shared_experts.up_proj.weight
model.layers.N.mlp.shared_experts.down_proj.weight
model.layers.N.mlp.experts.M.gate_proj.weight     # M=0..255
model.layers.N.mlp.experts.M.up_proj.weight
model.layers.N.mlp.experts.M.down_proj.weight
```

注意：DeepSeek-V3 使用 FP8 量化 (`weight_scale_inv`)，训练时需要去量化。

### MLA Forward

```
1. q = q_a_proj(hidden) → q_a_layernorm → q_b_proj → reshape to [heads, qk_nope+qk_rope]
2. kv = kv_a_proj(hidden) → split: kv_lora (first kv_lora_rank) + k_rope (last qk_rope)
3. kv_lora → kv_a_layernorm → kv_b_proj → reshape to [heads, qk_nope+v_head]
4. Apply RoPE to q_rope and k_rope parts
5. Concatenate qk_nope and qk_rope for Q and K
6. attention(Q, K, V) → [heads, v_head]
7. o_proj(context)
```

## Changes

### 1. `crates/rustrain-deepseek/` — 新 crate

#### `model.rs`
- `DeepSeekConfig`: MLA + MoE + yarn rope config
- `DeepSeekLayerWeights`: dense layer (MLA + dense MLP)
- `DeepSeekMoeLayerWeights`: MoE layer (MLA + router + shared expert + 256 routed experts)
- `deepseek_mla_attention()`: MLA forward
- `deepseek_mlp()`: SwiGLU (gate/up/down)
- `deepseek_moe_mlp()`: router → top-k → shared + routed expert combine
- `deepseek_layer()` / `deepseek_moe_layer()`: layer forward
- `deepseek_forward_from_ids()`: full model forward
- `read_deepseek_config()`: parse config.json

#### `session.rs`
- trainable tensors (MLA + MoE)
- training entry functions

#### `lora.rs`
- LoRA on MLA attention projections

#### `generate.rs`
- greedy/sampling with MLA KV cache

### 2. `src/main.rs` — dispatch

```rust
if is_tch && arch == "deepseek_trainable_session" { ... }
if is_tch && arch == "deepseek_lora_sft" { ... }
```

### 3. `Cargo.toml` — 依赖

### 4. `configs/` — 配置文件

## Definition of Done

- [ ] `cargo check -p rustrain-deepseek` 编译通过
- [ ] `train --config configs/deepseek_v3_session_single.toml` 完成 layer 0 (dense) 训练
- [ ] `train --config configs/deepseek_v3_moe_session_single.toml` 完成 layer 3 (MoE) 训练
- [ ] loss 下降

## Open Questions

- FP8 量化权重：训练时是否直接用 fp32/bf16 去量化？还是保留 FP8？
  建议：加载时去量化为 bf16，训练用 bf16。
- YaRN RoPE：是否需要完整实现 YaRN？建议第一阶段用标准 RoPE，忽略 yarn scaling。
