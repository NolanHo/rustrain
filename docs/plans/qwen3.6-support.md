# Plan: Qwen3.6-35B-A3B Support

## Summary

Add support for **Qwen3.6-35B-A3B** — a 35B-parameter MoE multimodal model with hybrid
linear/full attention, 256 experts (8 active), shared expert with gate, MTP, and a vision
encoder. New crate `rustrain-qwen3-6`, single-GPU forward + LoRA SFT first, EP=8 later.

---

## Architecture Analysis

### Config (from `config.json` text_config + vision_config)

| Component | Value |
|---|---|
| model_type | `qwen3_5_moe` |
| hidden_size | 2048 |
| num_hidden_layers | 40 |
| layer_types | 3× linear_attention + 1× full_attention (repeat 10×) |
| full_attention_interval | 4 |
| **Full attention** | |
| num_attention_heads | 16 |
| num_key_value_heads | 2 (GQA) |
| head_dim | 256 |
| partial_rotary_factor | 0.25 |
| rope_theta | 10,000,000 |
| mrope_interleaved | true |
| mrope_section | [11, 11, 10] |
| attn_output_gate | true |
| **Linear attention (Mamba2 SSM)** | |
| linear_num_key_heads | 16 |
| linear_key_head_dim | 128 |
| linear_num_value_heads | 32 |
| linear_value_head_dim | 128 |
| linear_conv_kernel_dim | 4 |
| mamba_ssm_dtype | float32 |
| **MoE** | |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 512 |
| shared_expert_intermediate_size | 512 |
| router_aux_loss_coef | 0.001 |
| **MTP** | |
| mtp_num_hidden_layers | 1 |
| mtp_use_dedicated_embeddings | false |
| **Other** | |
| vocab_size | 248,320 |
| rms_norm_eps | 1e-6 |
| hidden_act | silu |
| tie_word_embeddings | false |
| weight_prefix | `model.language_model.` (multimodal model) |

### Weight Map (text model)

```
model.language_model.embed_tokens.weight
model.language_model.layers.{N}.input_layernorm.weight
model.language_model.layers.{N}.post_attention_layernorm.weight

# Full attention layers (every 4th layer: 3,7,11,...,39)
model.language_model.layers.{N}.self_attn.q_proj.weight
model.language_model.layers.{N}.self_attn.q_norm.weight
model.language_model.layers.{N}.self_attn.k_proj.weight
model.language_model.layers.{N}.self_attn.k_norm.weight
model.language_model.layers.{N}.self_attn.v_proj.weight
model.language_model.layers.{N}.self_attn.o_proj.weight

# Linear attention layers (layers 0,1,2, 4,5,6, 8,9,10, ...)
model.language_model.layers.{N}.linear_attn.A_log
model.language_model.layers.{N}.linear_attn.conv1d.weight
model.language_model.layers.{N}.linear_attn.dt_bias
model.language_model.layers.{N}.linear_attn.in_proj_a.weight
model.language_model.layers.{N}.linear_attn.in_proj_b.weight
model.language_model.layers.{N}.linear_attn.in_proj_qkv.weight
model.language_model.layers.{N}.linear_attn.in_proj_z.weight
model.language_model.layers.{N}.linear_attn.norm.weight
model.language_model.layers.{N}.linear_attn.out_proj.weight

# MoE (all layers)
model.language_model.layers.{N}.mlp.gate.weight                    # router [256, 2048]
model.language_model.layers.{N}.mlp.shared_expert_gate.weight      # shared expert gate scalar/vector
model.language_model.layers.{N}.mlp.shared_expert.gate_proj.weight
model.language_model.layers.{N}.mlp.shared_expert.up_proj.weight
model.language_model.layers.{N}.mlp.shared_expert.down_proj.weight
model.language_model.layers.{N}.mlp.experts.{E}.gate_up_proj.weight # FUSED gate+up
model.language_model.layers.{N}.mlp.experts.{E}.down_proj.weight

# MTP
mtp.fc.weight
mtp.norm.weight
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
mtp.layers.{N}.input_layernorm.weight
mtp.layers.{N}.post_attention_layernorm.weight
mtp.layers.{N}.self_attn.q_proj.weight        # full attention only
mtp.layers.{N}.self_attn.q_norm.weight
mtp.layers.{N}.self_attn.k_proj.weight
mtp.layers.{N}.self_attn.k_norm.weight
mtp.layers.{N}.self_attn.v_proj.weight
mtp.layers.{N}.self_attn.o_proj.weight
mtp.layers.{N}.mlp.gate.weight
mtp.layers.{N}.mlp.shared_expert_gate.weight
mtp.layers.{N}.mlp.shared_expert.{gate,up,down}_proj.weight
mtp.layers.{N}.mlp.experts.{E}.gate_up_proj.weight
mtp.layers.{N}.mlp.experts.{E}.down_proj.weight

# Vision encoder
model.visual.patch_embed.proj.{weight,bias}
model.visual.pos_embed.weight
model.visual.blocks.{N}.norm1.{weight,bias}
model.visual.blocks.{N}.attn.qkv.{weight,bias}
model.visual.blocks.{N}.attn.proj.{weight,bias}
model.visual.blocks.{N}.norm2.{weight,bias}
model.visual.blocks.{N}.mlp.linear_fc1.{weight,bias}
model.visual.blocks.{N}.mlp.linear_fc2.{weight,bias}
model.visual.merger.norm.{weight,bias}
model.visual.merger.linear_fc1.{weight,bias}
model.visual.merger.linear_fc2.{weight,bias}

# Output
lm_head.weight
model.language_model.norm.weight
```

### Key Differences vs Existing Qwen3

| Feature | Qwen3 (existing) | Qwen3.6 (new) |
|---|---|---|
| Attention | Standard GQA | Hybrid: linear (Mamba2 SSM) + full attention |
| MoE experts | Separate gate_proj/up_proj | Fused gate_up_proj |
| Shared expert | None | shared_expert + shared_expert_gate |
| MTP | None | 1 MTP layer with MoE |
| RoPE | Standard | MRoPE (interleaved, partial_rotary=0.25) |
| attn output gate | No | Yes |
| Weight prefix | `model.` | `model.language_model.` (multimodal) |
| Vision | No | Yes (ViT + merger) |

---

## Implementation Plan

### Phase 1: Crate Setup + Config + Weight Loading

#### 1.1 Create `crates/rustrain-qwen3-6/`
- **What**: New crate scaffold: `Cargo.toml`, `src/lib.rs`
- **Why**: Independent model crate, matching project convention
- **Files**:
  - `crates/rustrain-qwen3-6/Cargo.toml` — deps: rustrain-core (tch), rustrain-data, rustrain-checkpoint (tch), rustrain-train, rustrain-parallel (nccl), rustrain-nccl, anyhow, serde, serde_json, tch, tokenizers, tracing
  - `crates/rustrain-qwen3-6/src/lib.rs` — module declarations
  - `crates/rustrain-qwen3/Cargo.toml` (root) — add `rustrain-qwen3-6` to workspace members (automatic via `crates/*` glob)

#### 1.2 Config parsing — `src/config.rs`
- **What**: Parse `config.json` → `Qwen36RuntimeConfig`
- **Key fields**: layer_types, linear attention dims, MoE config, MTP config, vision config, attn_output_gate
- **Weight prefix**: `model.language_model.` (not `model.`)
- **Why**: Need to handle hybrid layer types, linear attention config, and multimodal prefix

### Phase 2: Text Model Forward

#### 2.1 Full attention layer — `src/model.rs`
- **What**: GQA attention with MRoPE (interleaved, partial_rotary=0.25) + attn output gate
- **Components**:
  - `qwen36_rope()` — MRoPE with interleaved format, partial rotary (first 25% of head_dim gets RoPE)
  - `qwen36_attention()` — standard scaled dot-product attention with q_norm/k_norm
  - `qwen36_attn_output_gate()` — sigmoid gate on attention output (if `attn_output_gate=true`)
- **Reference**: `rustrain-qwen3/src/model.rs:qwen3_attention()` for base pattern
- **New**: MRoPE interleaved format, partial rotary, output gate

#### 2.2 Linear attention layer (Mamba2 SSM) — `src/model.rs`
- **What**: Selective state space model with conv1d, A_log, dt_bias
- **Components**:
  - `qwen36_linear_attn()` — selective scan forward pass
  - Weight loading: A_log, conv1d, dt_bias, in_proj_a/b/qkv/z, norm, out_proj
  - SSM core: discretize A/B (via dt), conv1d causal filtering, recurrent state update
  - `mamba_ssm_dtype`: float32 for SSM computation regardless of compute dtype
- **Why**: Completely new architecture component, no existing reference in codebase
- **Risk**: Selective scan in tch-rs may need manual loop or custom CUDA kernel for efficiency

#### 2.3 Hybrid layer dispatch — `src/model.rs`
- **What**: Forward pass dispatching between linear_attn and full_attn based on `layer_types`
- **Pattern**: Match `config.layer_types[layer_index]` → "linear_attention" or "full_attention"
- **Each layer**: input_layernorm → attention (linear or full) → residual → post_attention_layernorm → MoE → residual

#### 2.4 MoE with shared expert + gate — `src/model.rs`
- **What**: 256-expert MoE with fused gate_up_proj, shared expert, shared_expert_gate
- **Components**:
  - `qwen36_moe_mlp()` — router → top-8 → expert dispatch → shared expert → shared_expert_gate
  - Fused gate_up_proj: load single `gate_up_proj.weight`, split via `narrow(-1, 0, intermediate)` / `narrow(-1, intermediate, intermediate)` for SwiGLU
  - Shared expert: compute `shared_expert_mlp()` unconditionally
  - `shared_expert_gate`: scalar/vector gate applied to shared expert output before adding to routed sum
- **Reference**: `rustrain-qwen3/src/model.rs:qwen3_moe_mlp()` for routing pattern; `rustrain-glm5/src/model.rs:glm5_moe_mlp()` for shared expert pattern
- **New**: Fused gate_up_proj split, shared_expert_gate gating

#### 2.5 MTP forward + loss — `src/model.rs`
- **What**: 1 MTP layer with MoE, predict next token
- **Weights**: `mtp.fc.weight`, `mtp.norm.weight`, `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`, `mtp.layers.{N}.*`
- **Forward**: combine hidden + shifted embed → norm → attention (full) → MoE → norm → logits
- **Loss**: cross-entropy on shifted tokens, weight 0.5×
- **Reference**: `rustrain-glm5/src/model.rs:glm5_mtp_*` (closest naming convention)
- **New**: MTP layer uses full attention (not hybrid), MoE with shared expert + gate

#### 2.6 Full forward pass — `src/model.rs`
- **What**: `qwen36_forward_from_ids()` — embed → 40 hybrid layers → final norm → lm_head
- **Weight loading**: `read_safetensors_dir()` with `model.language_model.` prefix filtering (skip `model.visual.*`)
- **LoRA forward**: `qwen36_forward_from_ids_with_lora()` — apply LoRA adapters to target layers

### Phase 3: Vision Encoder Forward

#### 3.1 Vision encoder — `src/vision.rs`
- **What**: ViT forward (patch_embed → pos_embed → 27 blocks → merger)
- **Components**:
  - `Qwen36VisionConfig` — depth=27, hidden=1152, heads=16, patch_size=16, spatial_merge=2, temporal_patch=2
  - `Qwen36VisionWeights` — patch_embed, pos_embed, blocks (norm1, attn qkv+proj, norm2, mlp fc1+fc2), merger
  - `qwen36_vision_forward()` — process image → visual tokens
- **Why**: Multimodal support requires vision encoder to process image inputs
- **Note**: For text-only LoRA SFT, vision encoder uses frozen weights; LoRA targets text model only

#### 3.2 Multimodal integration — `src/model.rs`
- **What**: Combine vision tokens + text tokens for multimodal forward
- **Image token handling**: vision_start_token_id (248053), image_token_id (248056), vision_end_token_id (248054)
- **Token sequence**: text tokens + image tokens (replaced by visual embeddings)

### Phase 4: LoRA SFT (Single GPU)

#### 4.1 LoRA registry — `src/lora.rs`
- **What**: LoRA adapter registry for Qwen3.6
- **Target modules**: q_proj, k_proj, v_proj, o_proj (full attn), gate_up_proj, down_proj (MoE experts), shared_expert gate/up/down
- **New targets**: linear_attn.in_proj_qkv, in_proj_z, out_proj (linear attention projections)
- **Reference**: `rustrain-qwen3/src/lora.rs:QwenLoraRegistry`

#### 4.2 SFT dataset — `src/sft.rs`
- **What**: JSONL dataset loading, tokenization, response-only loss masking
- **Reference**: `rustrain-qwen3/src/sft.rs` (can largely copy structure)
- **Multimodal**: support image + text SFT data (image paths in JSONL)

#### 4.3 Training session — `src/session.rs`
- **What**: Single-GPU LoRA SFT training loop
- **Components**: AdamW optimizer, LR scheduler, gradient clipping, step logging, adapter save/load
- **Reference**: `rustrain-qwen3/src/session.rs`

#### 4.4 Config + CLI integration
- **What**: TOML config for Qwen3.6 LoRA SFT, CLI dispatch in `src/main.rs`
- **Config**: `configs/qwen3_6_lora_sft.toml` — model path, architecture="qwen3_6_lora_sft"
- **CLI**: add `qwen3_6_lora_sft` architecture dispatch in `src/main.rs`

### Phase 5 (Future): EP=8 Training
- EP sharding of 256 experts across 8 GPUs
- NCCL all-reduce for routed expert outputs
- Shared expert replication + all-reduce correction (reference: `rustrain-glm5/src/session_ep.rs:502-517`)
- FP8 GEMM integration (reference: `rustrain-deepseek-v4/src/fp8_kernel.rs`)

---

## Risks

| Risk | Mitigation |
|---|---|
| **Linear attention (Mamba2 SSM)** is new — no existing reference in codebase | Implement in pure tch-rs tensors first; if too slow, add C++ kernel later. Start with small seq_len for verification. |
| **Selective scan** may need sequential loop in Rust (no flash-attn equivalent) | Use tch-rs tensor ops; for training, process full sequence (no KV cache needed for training) |
| **256 experts** × 512 intermediate = large weight count (768 tensors per layer × 40 layers) | Load only needed shards via safetensors index; filter by layer range for LoRA |
| **Fused gate_up_proj** breaks existing LoRA and TP patterns | Add new `GateUpProj` LoRA target; split fused weight in forward, not at load time |
| **shared_expert_gate** is new (not in any existing crate) | Investigate HF transformers source for exact semantics; likely a learned scalar applied to shared expert output |
| **MRoPE interleaved** format differs from standard RoPE | Implement carefully; for text-only training, mrope_section may not matter (only affects temporal/spatial dimensions for video) |
| **Multimodal** adds significant scope | Phase the work: text-only forward first (Phase 2), vision encoder (Phase 3), multimodal SFT (Phase 4) |

---

## Definition of Done

### Phase 1-2: Text Forward
- [ ] `crates/rustrain-qwen3-6` compiles with `cargo build`
- [ ] Config parsing reads `config.json` → `Qwen36RuntimeConfig` correctly
- [ ] Weight loading from safetensors with `model.language_model.` prefix
- [ ] Full attention forward (with MRoPE + partial rotary + output gate)
- [ ] Linear attention forward (Mamba2 SSM with conv1d, A_log, dt_bias)
- [ ] Hybrid layer dispatch (40 layers, 3 linear + 1 full alternating)
- [ ] MoE forward (256 experts, fused gate_up_proj, shared expert + gate)
- [ ] MTP forward + loss
- [ ] `qwen36_forward_from_ids()` produces correct-shaped logits
- [ ] Forward pass verified: load real Qwen3.6-35B-A3B weights, run on CUDA, output logits match expected shape

### Phase 3: Vision Encoder
- [ ] Vision config parsing
- [ ] ViT forward (patch_embed → blocks → merger)
- [ ] Multimodal forward (vision tokens + text tokens)

### Phase 4: LoRA SFT
- [ ] LoRA registry with Qwen3.6 target modules
- [ ] SFT dataset loading (JSONL)
- [ ] Training loop: AdamW, LR schedule, gradient clip, response-only loss
- [ ] Adapter save/load
- [ ] `cargo run -- train --config configs/qwen3_6_lora_sft.toml` completes 2 steps on single GPU with decreasing loss

---

## Reference Implementation Notes

Based on HF transformers source (qwen3_next modeling, v4.57.0):

### shared_expert_gate (answered)

`shared_expert_gate` is a **per-token scalar gate**: `Linear(hidden_size, 1, bias=False)`.
In forward: `sigmoid(shared_expert_gate(hidden)) * shared_expert_output`, then added to
routed expert sum. This is a learned gating that controls the shared expert contribution
per token — NOT the SwiGLU gate inside the shared expert's MLP.

```python
# qwen3_next/modeling_qwen3_next.py:807-851
self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
# forward:
shared_out = self.shared_expert(hidden)
shared_out = F.sigmoid(self.shared_expert_gate(hidden)) * shared_out
final = routed_sum + shared_out
```

### Linear Attention — Gated Delta Rule (answered)

Qwen3.6's linear attention is the **Gated Delta Rule** (same as Qwen3-Next's
`Qwen3NextGatedDeltaNet`), NOT a pure Mamba2 SSM. Key differences from Qwen3-Next:
- Qwen3-Next: `in_proj_qkvz` (fused Q+K+V+Z), `in_proj_ba` (fused B+A)
- Qwen3.6: `in_proj_qkv` (Q+K+V), `in_proj_z` (Z gate), `in_proj_a` (A decay), `in_proj_b` (B beta) — all separate

Algorithm:
1. `qkv = in_proj_qkv(hidden)` → split Q, K, V (each [batch, seq, num_k_heads, head_k_dim] or [batch, seq, num_v_heads, head_v_dim])
2. `z = in_proj_z(hidden)` → gate [batch, seq, num_v_heads, head_v_dim]
3. `a = in_proj_a(hidden)` → decay [batch, seq, num_v_heads]
4. `b = in_proj_b(hidden)` → beta [batch, seq, num_v_heads]
5. `mixed_qkv = conv1d(cat(q,k,v)) + SiLU` (causal conv, kernel=4)
6. `beta = sigmoid(b)`
7. `g = -exp(A_log) * softplus(a + dt_bias)` — per-head decay rate
8. **Core**: `chunk_gated_delta_rule(query, key, value, g, beta, use_qk_l2norm=True)` — chunked recurrent delta rule with QK L2 normalization
9. `output = RMSNorm(core_attn_out, z)` — gated norm
10. `output = out_proj(output)`

For training (full sequence), we can use a simple recurrent formulation:
```
state_t = exp(g_t) * state_{t-1} + beta_t * (v_t - state_{t-1} @ k_t) ⊗ k_t   # gated delta update
output_t = q_t @ state_t
```
This is O(seq_len) sequential — acceptable for short seq_len (5–256). For longer sequences,
implement chunked formulation later.

### MRoPE for text-only (partially answered)

MRoPE uses 3D rotation (temporal, height, width) with `mrope_section = [11, 11, 10]` (sums
to partial_rotary_dim = head_dim * partial_rotary_factor = 256 * 0.25 = 64, split as 32+32).
For text-only training, all position IDs map to the temporal dimension — height/width positions
are zero. This effectively reduces to standard RoPE with the first 64 dims rotated, but with
interleaved format (`mrope_interleaved: true`).

For initial implementation: treat text-only MRoPE as standard RoPE with:
- `partial_rotary_factor = 0.25` (first 64 of 256 dims get RoPE)
- `mrope_interleaved = true` (interleave cos/sin pairs)
- `rope_theta = 10_000_000`

### MTP structure

MTP has its own separate expert weights (not shared with main model). The MTP layer uses full
attention (not hybrid) + MoE with shared expert + gate. Weight prefix: `mtp.` (no `model.language_model.` prefix).
