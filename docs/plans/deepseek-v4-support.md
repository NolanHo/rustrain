# Plan: DeepSeek-V4 Support

## Summary

新增 `rustrain-deepseek-v4` crate，支持 DeepSeek-V4-Flash 的 forward + LoRA SFT。
V4 是全新架构，与 V3 共享 MLA 思想但结构差异很大。

## V4 vs V3 差异

### Attention (MLA 变体)
- V3: q_a_proj → q_b_proj, kv_a_proj → kv_b_proj, o_proj
- V4: wq_a → wq_b, **单 wkv**（无 kv_b）, **wo_a + wo_b** (LoRA-style 输出投影)
- V4 新增: attn_sink, kv_norm, q_norm
- V4 权重全部带 `.scale` (FP8 量化)

### MLP
- V3: `mlp.experts.N.{gate,up,down}_proj.weight`
- V4: `ffn.experts.N.{w1,w2,w3}.weight` + `.scale`
- V4 全部是 MoE 层（无 dense MLP），但每层都有 shared expert
- V4 有 `swiglu_limit=10.0` (SwiGLU 输出限幅)

### Routing
- V3: sigmoid + group-based (n_group, topk_group)
- V4: **sqrtsoftplus** + hash clustering (hc_sinkhorn_iters, hc_mult, hc_eps)
- V4: topk_method=noaux_tc, routed_scaling_factor=1.5

### 其他
- V4: sliding_window=128, compress_ratios (序列压缩)
- V4: o_groups=8, o_lora_rank=1024 (输出分组 LoRA)
- V4: index_head_dim=128, index_n_heads=64 (indexer)

## V4-Flash Config

```
model_type: deepseek_v4
hidden_size: 4096, num_hidden_layers: 43, num_attention_heads: 64
q_lora_rank: 1024, qk_rope_head_dim: 64
kv_lora_rank: (implicit from wkv weight shape)
n_routed_experts: 256, num_experts_per_tok: 6, n_shared_experts: 1
moe_intermediate_size: 2048
scoring_func: sqrtsoftplus, routed_scaling_factor: 1.5
expert_dtype: fp8, swiglu_limit: 10.0
sliding_window: 128, o_groups: 8, o_lora_rank: 1024
```

## V4 Layer 0 权重名

```
layers.0.attn.attn_sink
layers.0.attn.kv_norm.weight
layers.0.attn.q_norm.weight
layers.0.attn.wkv.scale / .weight
layers.0.attn.wo_a.scale / .weight
layers.0.attn.wo_b.scale / .weight
layers.0.attn.wq_a.scale / .weight
layers.0.attn.wq_b.scale / .weight
layers.0.attn_norm.weight
layers.0.ffn.experts.N.{w1,w2,w3}.scale / .weight
layers.0.ffn.gate.{scale,weight} (router)
layers.0.ffn.shared_experts.{w1,w2,w3}.scale / .weight
```

## Changes

### 1. `crates/rustrain-deepseek-v4/` — 新 crate

#### `model.rs`
- `V4RuntimeConfig`: V4 config fields
- `V4AttentionWeights`: wq_a, wq_b, wkv, wo_a, wo_b, q_norm, kv_norm, attn_sink
- `V4MoeLayerWeights`: attention + router gate + shared expert + 256 experts
- `v4_attention()`: wq_a→wq_b + wkv + wo_a→wo_b forward
- `v4_moe_mlp()`: sqrtsoftplus routing + SwiGLU with swiglu_limit
- `v4_forward()`: full model forward
- FP8→bf16 loading via Python (same pattern as V3)

#### `lora.rs`
- LoRA on wq_a, wq_b, wkv, wo_a, wo_b + expert weights

#### `sft.rs`
- JSONL loader + tokenizer (reuse from rustrain-data)

#### `session.rs`
- LoRA SFT training with AdamW + checkpoint

#### `generate.rs`
- Greedy + sampling

### 2. `src/main.rs` — dispatch
### 3. `configs/deepseek_v4_*.toml`

## Definition of Done

- [ ] `cargo check -p rustrain-deepseek-v4` 编译通过
- [ ] `train --config configs/deepseek_v4_session_single.toml` forward + loss
- [ ] `train --config configs/deepseek_v4_lora_sft.toml` LoRA SFT loss 下降
- [ ] bf16 训练
- [ ] Checkpoint save/load + resume
