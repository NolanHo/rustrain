# Linear Attention (Gated Delta Rule) — Architecture & Pitfalls

## QKV Split Layout — CRITICAL

Transformers `Qwen3_5GatedDeltaNet` (used by Qwen3.5 and Qwen3.6 models) uses **flat QKV split**:
```
in_proj_qkv → [Q_all(2048) | K_all(2048) | V_all(4096)]
torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
```

NOT per-head interleaved like Qwen3Next:
```
# Qwen3Next (different model!) uses fix_query_key_value_ordering:
# [head0: Q(128) K(128) V(256) Z(256) | head1: ...]
```

### Weight Names
- `in_proj_qkv.weight` — QKV combined (flat: Q, K, V concatenated)
- `in_proj_z.weight` — Z gate projection
- `in_proj_a.weight` — A (for g computation)
- `in_proj_b.weight` — B (for beta computation)
- `conv1d.weight` — depthwise conv1d (groups=conv_dim)
- `A_log`, `dt_bias` — parameters
- `norm.weight`, `out_proj.weight`

### Bug History
- `linear_attention_batched()` in `qwen3_6_kernels.cpp` used per-head interleaved split (wrong)
- `linear_attention()` (non-batched, line ~517) used flat split (correct)
- Rust `model.rs` `linear_attention()` used flat split (correct)
- Fix: changed batched path to flat split — commit `58d5a7c`

## Delta Rule Recurrence

Both transformers and rustrain use the same recurrence:
```
S = S * exp(g_t)                    # decay
kv_mem = S @ k_t                    # key-value memory
delta = (v_t - kv_mem) * beta_t     # innovation
S = S + k_t ⊗ delta                 # state update
out_t = S @ q_t                     # output
```

## L2 Normalization

- Transformers: `x * rsqrt(sum(x²) + eps)` (eps=1e-6, in denominator)
- rustrain: `x / max(sqrt(sum(x²)), eps)` (eps=1e-6, as clamp)

Mathematically equivalent but numerically slightly different.

## Diagnostic Dumps

Set `QWEN36_DUMP_LAYERS=1` to enable per-layer hidden state dumps:
- `[dump] embedding:` — embedding output
- `[dump] layer N:` — hidden state after layer N
- `[diag-la]` — intermediate dumps in linear attention (layer 0 only)

## References
- Transformers source: `transformers/models/qwen3_5/modeling_qwen3_5.py`
- C++ kernel: `crates/rustrain-qwen3-6/kernels/qwen3_6_kernels.cpp`
- CUDA delta rule: `crates/rustrain-qwen3-6/kernels/delta_rule.cuh`
- FSDP reference: `scripts/fsdp_reference.py`
