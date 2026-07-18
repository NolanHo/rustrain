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

## Tensor Parallel Layout

GDN TP partitions K/V head groups while preserving `n_rep = V_heads / K_heads`:

- `in_proj_qkv.weight` and `conv1d.weight`: slice Q, K, and V segments independently, then repack each rank as `[Q_local | K_local | V_local]`.
- `in_proj_z.weight`: shard output rows by local V heads.
- `in_proj_a.weight`, `in_proj_b.weight`, `A_log`, `dt_bias`: shard by local V-head rows.
- `norm.weight`: replicate `[value_head_dim]`.
- `out_proj.weight`: shard input columns matching local V heads, then all-reduce the local output.
- Column-parallel LoRA replicates A and shards B; row-parallel LoRA shards A and replicates B. Replicated-factor gradients are summed once at the optimizer boundary.

The v4 flat-QKV checkpoint manifest records three local-to-global row segments for Q, K, and V. Merge or reshard must apply those segments instead of treating the packed rank-local rows as one contiguous global slice.

One `TpCopyToRegion` must wrap the shared input before the QKV/Z/A/B forks so backward sums their input-gradient contributions once. Do not all-reduce each fork separately.

## Backward Stability

The current fused CUDA backward reconstructs earlier states by dividing the decayed state by `g_exp`. This is fast and verified with realistic negative `dt_bias`, where decay stays near one, but it is ill-conditioned for synthetic long sequences with decay near `0.5`. A stable production replacement should checkpoint recurrent state by chunks and replay each chunk during backward instead of repeatedly inverting the decay.

## Right-Padding Fast Path

Strict-right-padding masks are reduced once to device-resident `lengths[B]`.
The persistent GDN forward and backward kernels stop at each sample's valid
length and explicitly zero output and gradient tails. Chunked eval converts the
global lengths to per-chunk offsets, and the batched LoRA path narrows lengths
with the same adapter sub-batch as Q/K/V. Dense batches use an empty sentinel
and retain the null-pointer fast path. Left padding and internal holes remain
rejected until packed `cu_seqlens` boundaries are implemented.

## Native GDN TP Verification

Use a Python environment with ABI-compatible prebuilt PyTorch, CUDA, and NCCL, then run:

```bash
PYTHON=/path/to/python scripts/run_qwen36_native_gdn_tp.sh smoke
PYTHON=/path/to/python scripts/run_qwen36_native_gdn_tp.sh bench-single
PYTHON=/path/to/python scripts/run_qwen36_native_gdn_tp.sh bench-tp2
```

The script builds only the repository kernel and native harnesses. It discovers and links the prebuilt dependency files from the selected Python environment; missing headers or libraries are reported instead of building third-party dependencies.

## L2 Normalization

- Transformers/Megatron fused pre-GDN: `x * rsqrt(sum(x²) + eps)` (eps=1e-6)
- rustrain C++ and Rust fallback: `x * rsqrt(sum(x²) + 1e-6)`

The epsilon is inside the squared norm. Keeping this form is important for
low-norm Q/K vectors; a post-sqrt clamp is not numerically equivalent.

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
