# Megakernel Implementation Plan

## Objective
Replace PyTorch autograd with hand-written forward+backward for all Qwen3.6 layers.
Eliminate checkpoint recompute. Target: 1M from 15min → <10min.

## Current State
- `backward.h` — Basic backward ops (rms_norm, silu, matmul, RoPE, L2, gated_norm) ✅
- `delta_rule.cuh` — Forward saves delta_buf + backward reverse pass ✅
- `megakernel.cu` — LinearAttnLayer framework (forward done, backward placeholder) ⚠️
- `qwen3_6_kernels.cpp` — Still uses checkpoint + manual_group_backward ❌

## Implementation Steps

### Step 1: Compile verify (delta_buf changes)
- Sync to remote, compile .so with nvcc
- Verify delta_rule.cuh backward kernel compiles
- Verify qwen3_6_kernels.cpp delta_buf parameter changes compile

### Step 2: Linear attention layer backward
File: `megakernel.cu` — `LinearAttnLayer::backward`

Forward saves:
- `hidden` [B,S,H] — for rms_norm backward + residual
- `attn_input` [B,S,H] — for qkv/a/b/z projection backward
- `qkv_conv` [B,S,qkv_dim] — for conv1d backward
- `q_normed, k_normed` [B,H,S,D_k] — for L2 backward
- `g, beta` [B,H,S] — decay/sigmoid backward
- `core_out` [B,S,V,D_v] — gated norm backward
- `gated` [B,S,V*D_v] — out_proj backward
- `delta_buf, final_state` — for delta_rule_backward kernel

Backward chain:
1. grad_result → grad_gated (matmul backward: grad_gated = grad_result @ out_proj)
2. grad_gated → grad_normed, grad_z (gated_norm backward: silu'(z))
3. grad_normed → grad_core (rms_norm backward, raw weight)
4. grad_core → grad_q, grad_k, grad_v, grad_g, grad_beta (delta_rule_backward kernel)
5. grad_q, grad_k → grad_q_pre_l2, grad_k_pre_l2 (L2 backward)
6. → undo scale, undo repeat_interleave
7. grad_v, grad_qkv_conv → grad_conv_out (silu backward + split)
8. grad_conv_out → grad_qkv (conv1d backward)
9. grad_qkv → grad_attn_input (matmul backward: grad_attn_input = grad_qkv @ in_proj_qkv)
10. + grad from a/b/z projections
11. grad_attn_input → grad_hidden_from_attn (rms_norm backward)
12. grad_hidden = grad_output (residual) + grad_hidden_from_attn

### Step 3: Full attention layer backward
File: `megakernel.cu` — `FullAttnLayer` (new autograd::Function)

Forward saves:
- `hidden, attn_input` — rms_norm backward
- `q, gate, k, v` — projection backward + SDPA backward + RoPE backward
- `k_expanded, v_expanded` — repeat_interleave backward
- `sdpa_output` — gate sigmoid backward

Backward chain:
1. grad_result → grad_sdpa_output (matmul backward: o_proj)
2. grad_sdpa_output → grad_attn_out (gate: grad * sigmoid'(gate))
3. grad_attn_out → grad_q, grad_k_expanded, grad_v_expanded (SDPA backward)
4. grad_k_expanded → grad_k (repeat_interleave: sum across repeated dims)
5. grad_q, grad_k → grad_q_pre_rope, grad_k_pre_rope (RoPE backward)
6. → grad_q_pre_norm, grad_k_pre_norm (rms_norm backward)
7. → grad_attn_input (matmul backward: q_proj, k_proj, v_proj)
8. → grad_hidden (rms_norm backward + residual)

SDPA backward: Use `at::scaled_dot_product_attention` with `is_causal=true` — 
PyTorch supports backward through SDPA natively via `torch::autograd::backward()`.

### Step 4: Dense MLP + MoE backward
File: `megakernel.cu` — `MoeLayer` (new autograd::Function)

Dense MLP backward:
1. grad_output → grad_activated (matmul backward: down_proj)
2. grad_activated → grad_gate_out, grad_up_out (silu backward + elementwise mul)
3. → grad_flat (matmul backward: gate_proj, up_proj)

MoE backward:
1. grad_output → grad_routed, grad_shared (split)
2. grad_routed → per-expert backward (matmul backward for each expert)
3. grad_shared → same as dense MLP backward
4. grad_routing → softmax backward
5. → grad_flat (sum of expert grads * routing weights)

### Step 5: Integrate into train_step
Replace `forward_full_checkpoint` + `manual_group_backward` with:
```cpp
// Forward: each layer via autograd::Function (builds graph, no recompute)
for (int i = 0; i < num_layers; i++) {
    if (layer_type == full_attn)
        hidden = FullAttnLayer::apply(hidden, weights...);
    else
        hidden = LinearAttnLayer::apply(hidden, weights...);
    // MoE/MLP integrated into each layer's Function
}
// CE backward (unchanged — detached, chunked)
// Main backward: hidden.backward(hidden.grad()) — ONE call, no recompute!
```

### Step 6: Compile + gradient verification
- Compile with nvcc + g++
- Run 1 training step on Qwen3.5-0.8B (fast, small)
- Compare loss + grad with checkpoint version (should match within BF16)
- If mismatch: debug per-layer

### Step 7: Performance test
- Run 1M on 35B EP8
- Compare with 15min baseline
- Target: <10min

## Key Design Decisions

1. **autograd::Function per layer** — not per segment. Each layer = 1 Function call.
   Forward builds minimal graph (just the Function node), backward uses saved tensors.
   
2. **Save only what's needed for backward** — not all intermediates.
   rms_norm backward only needs input + weight (recompute is cheap).
   matmul backward only needs input + weight (recompute is cheap).
   delta_rule backward needs delta_buf + final_state (can't recompute).
   SDPA backward needs Q/K/V (PyTorch handles internally).

3. **No recompute** — forward runs once, backward uses saved tensors.
   This is the key difference from checkpoint (which recomputes forward).

4. **BF16 rms_norm** — no FP32 conversion (eliminates 32GB OOM at 2M).

5. **LoRA grads** — accumulated into ctx->lora_a/b during backward.
   Weight grads are zero (frozen), only LoRA A/B have requires_grad=true.

## Risk Mitigation
- If megakernel has bugs: fall back to checkpoint (env var QWEN36_MEGAKERNEL=1 to enable)
- If OOM from saved tensors: use selective saving (only save delta_buf, recompute cheap ops)
- If gradient mismatch: compare per-layer with PyTorch autograd (torch::autograd::grad)
