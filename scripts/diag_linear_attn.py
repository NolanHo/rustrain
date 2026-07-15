#!/usr/bin/env python3
"""Diagnostic: compare transformers linear attention with rustrain C++ kernel.

Dumps intermediate values at each stage of the linear attention forward pass
for layer 0, so we can compare with rustrain's [diag-la] dumps.

Usage: python3 scripts/diag_linear_attn.py
"""
import os, sys, json
import torch
import torch.nn.functional as F

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/workspace/share/models/Qwen3.5-9B")

def l2norm(x, dim=-1, eps=1e-6):
    """FLA-compatible L2 norm."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def main():
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    text_config = config.text_config if hasattr(config, 'text_config') else config

    print(f"Model: {MODEL_PATH}")
    print(f"model_type: {text_config.model_type}")
    print(f"linear_num_key_heads: {text_config.linear_num_key_heads}")
    print(f"linear_key_head_dim: {text_config.linear_key_head_dim}")
    print(f"linear_num_value_heads: {text_config.linear_num_value_heads}")
    print(f"linear_value_head_dim: {text_config.linear_value_head_dim}")
    print(f"linear_conv_kernel_dim: {text_config.linear_conv_kernel_dim}")
    print(f"hidden_size: {text_config.hidden_size}")

    device = "cuda:0"
    dtype = torch.bfloat16

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, config=config, torch_dtype=dtype, trust_remote_code=True, device_map=device
    )
    model.eval()

    # Find first linear attention layer
    layer_types = text_config.layer_types
    linear_layer_idx = None
    for i, lt in enumerate(layer_types):
        if lt == "linear_attention":
            linear_layer_idx = i
            break
    print(f"\nFirst linear attention layer: {linear_layer_idx}")

    if linear_layer_idx is None:
        print("No linear attention layer found!")
        return

    layer = model.model.layers[linear_layer_idx]
    la = layer.linear_attn

    # Create dummy input matching rustrain ep-bench: input_ids all 1s
    batch, seq = 1, 512
    input_ids = torch.ones(batch, seq, dtype=torch.long, device=device)

    with torch.no_grad():
        # Get embedding output
        embed = model.model.embed_tokens(input_ids)  # [1, 512, hidden]
        hidden = embed

        # Apply input_layernorm (matching rustrain forward_single_layer)
        hidden = layer.input_layernorm(hidden)

        print(f"\n=== Input to linear attention ===")
        print(f"hidden: mean={hidden.float().mean().item():.6f} std={hidden.float().std().item():.6f} "
              f"[0,0,:3]={hidden[0,0,0].float().item():.6f},{hidden[0,0,1].float().item():.6f},{hidden[0,0,2].float().item():.6f}")

        # Now manually run the linear attention forward, step by step
        # following Qwen3NextGatedDeltaNet.forward()
        h = hidden  # [1, 512, hidden_size]

        # Step 1: in_proj_qkvz
        projected_qkvz = la.in_proj_qkvz(h)  # [1, 512, qkvz_dim]
        print(f"\n=== Step 1: in_proj_qkvz ===")
        print(f"qkvz_proj: shape={list(projected_qkvz.shape)} mean={projected_qkvz.float().mean().item():.6f} "
              f"std={projected_qkvz.float().std().item():.6f} "
              f"[0,0,:5]={projected_qkvz[0,0,0].float().item():.6f},{projected_qkvz[0,0,1].float().item():.6f},"
              f"{projected_qkvz[0,0,2].float().item():.6f},{projected_qkvz[0,0,3].float().item():.6f},"
              f"{projected_qkvz[0,0,4].float().item():.6f}")

        # Step 2: in_proj_ba
        projected_ba = la.in_proj_ba(h)  # [1, 512, num_v_heads*2]

        # Step 3: fix_query_key_value_ordering
        query, key, value, z, b, a = la.fix_query_key_value_ordering(projected_qkvz, projected_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        print(f"\n=== Step 3: after fix_query_key_value_ordering ===")
        print(f"Q: shape={list(query.shape)} mean={query.float().mean().item():.6f} std={query.float().std().item():.6f} "
              f"[0,0,0,:3]={query[0,0,0].float().item():.6f},{query[0,0,1].float().item():.6f},{query[0,0,2].float().item():.6f}")
        print(f"K: shape={list(key.shape)} mean={key.float().mean().item():.6f} std={key.float().std().item():.6f} "
              f"[0,0,0,:3]={key[0,0,0].float().item():.6f},{key[0,0,1].float().item():.6f},{key[0,0,2].float().item():.6f}")
        print(f"V: shape={list(value.shape)} mean={value.float().mean().item():.6f} std={value.float().std().item():.6f} "
              f"[0,0,0,:3]={value[0,0,0].float().item():.6f},{value[0,0,1].float().item():.6f},{value[0,0,2].float().item():.6f}")

        # Step 4: cat(Q,K,V) → conv1d
        mixed_qkv = torch.cat((query, key, value), dim=-1)  # [1, 512, 8192]
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [1, 8192, 512]

        # Conv1d
        conv_kernel = la.conv_kernel_size
        pad = conv_kernel - 1
        padding = torch.zeros(1, mixed_qkv.shape[1], pad, dtype=mixed_qkv.dtype, device=device)
        padded = torch.cat([padding, mixed_qkv], dim=2)
        # Use F.conv1d directly (depthwise: groups=channels)
        conv_w = la.conv1d.weight  # [conv_dim, 1, kernel_size]
        conv_dim = mixed_qkv.shape[1]
        conv_out = F.conv1d(padded, conv_w, bias=None, stride=1, padding=0, dilation=1, groups=conv_dim)
        conv_out = F.silu(conv_out[:, :, :seq])  # [1, 8192, 512]
        mixed_qkv_conv = conv_out.transpose(1, 2)  # [1, 512, 8192]

        print(f"\n=== Step 4: after conv1d+silu ===")
        print(f"mixed_qkv_conv: mean={mixed_qkv_conv.float().mean().item():.6f} "
              f"std={mixed_qkv_conv.float().std().item():.6f} "
              f"[0,0,:5]={mixed_qkv_conv[0,0,0].float().item():.6f},{mixed_qkv_conv[0,0,1].float().item():.6f},"
              f"{mixed_qkv_conv[0,0,2].float().item():.6f},{mixed_qkv_conv[0,0,3].float().item():.6f},"
              f"{mixed_qkv_conv[0,0,4].float().item():.6f}")

        # Step 5: flat split after conv1d (NOT per-head split)
        key_dim = la.key_dim  # = num_k_heads * head_k_dim = 16 * 128 = 2048
        value_dim = la.value_dim  # = num_v_heads * head_v_dim = 32 * 128 = 4096
        q_flat, k_flat, v_flat = torch.split(mixed_qkv_conv, [key_dim, key_dim, value_dim], dim=-1)

        q = q_flat.reshape(q_flat.shape[0], q_flat.shape[1], -1, la.head_k_dim)  # [1, 512, 16, 128]
        k = k_flat.reshape(k_flat.shape[0], k_flat.shape[1], -1, la.head_k_dim)  # [1, 512, 16, 128]
        v = v_flat.reshape(v_flat.shape[0], v_flat.shape[1], -1, la.head_v_dim)  # [1, 512, 32, 128]

        print(f"\n=== Step 5: after flat split+reshape ===")
        print(f"Q: shape={list(q.shape)} mean={q.float().mean().item():.6f} std={q.float().std().item():.6f} "
              f"[0,0,0,:3]={q[0,0,0,0].float().item():.6f},{q[0,0,0,1].float().item():.6f},{q[0,0,0,2].float().item():.6f}")
        print(f"K: shape={list(k.shape)} mean={k.float().mean().item():.6f} std={k.float().std().item():.6f} "
              f"[0,0,0,:3]={k[0,0,0,0].float().item():.6f},{k[0,0,0,1].float().item():.6f},{k[0,0,0,2].float().item():.6f}")
        print(f"V: shape={list(v.shape)} mean={v.float().mean().item():.6f} std={v.float().std().item():.6f} "
              f"[0,0,0,:3]={v[0,0,0,0].float().item():.6f},{v[0,0,0,1].float().item():.6f},{v[0,0,0,2].float().item():.6f}")

        # Step 6: beta, g
        beta = b.sigmoid()  # b: [1, 512, num_v_heads]
        g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)  # [1, 512, num_v_heads]

        print(f"\n=== Step 6: g, beta ===")
        print(f"g: mean={g.float().mean().item():.6f} std={g.float().std().item():.6f} "
              f"[0,0,:3]={g[0,0,0].float().item():.6f},{g[0,0,1].float().item():.6f},{g[0,0,2].float().item():.6f}")
        print(f"beta: mean={beta.float().mean().item():.6f} std={beta.float().std().item():.6f} "
              f"[0,0,:3]={beta[0,0,0].float().item():.6f},{beta[0,0,1].float().item():.6f},{beta[0,0,2].float().item():.6f}")

        # Step 7: repeat_interleave Q/K
        n_rep = la.num_v_heads // la.num_k_heads
        q = q.repeat_interleave(n_rep, dim=2)  # [1, 512, 32, 128]
        k = k.repeat_interleave(n_rep, dim=2)

        # Step 8: L2 norm
        q_norm = l2norm(q.float(), dim=-1, eps=1e-6)
        k_norm = l2norm(k.float(), dim=-1, eps=1e-6)

        print(f"\n=== Step 8: after L2 norm ===")
        print(f"Q: mean={q_norm.mean().item():.6f} std={q_norm.std().item():.6f} "
              f"[0,0,0,:3]={q_norm[0,0,0,0].item():.6f},{q_norm[0,0,0,1].item():.6f},{q_norm[0,0,0,2].item():.6f}")
        print(f"K: mean={k_norm.mean().item():.6f} std={k_norm.std().item():.6f} "
              f"[0,0,0,:3]={k_norm[0,0,0,0].item():.6f},{k_norm[0,0,0,1].item():.6f},{k_norm[0,0,0,2].item():.6f}")

        # Step 9: Scale Q
        scale = 1.0 / (q_norm.shape[-1] ** 0.5)
        q_scaled = q_norm * scale

        # Step 10: Delta rule (use torch_recurrent_gated_delta_rule)
        from transformers.models.qwen3_next.modeling_qwen3_next import torch_recurrent_gated_delta_rule
        q_t = q_scaled.transpose(1, 2).contiguous()  # [1, 32, 512, 128]
        k_t = k_norm.transpose(1, 2).contiguous()
        v_t = v.float().transpose(1, 2).contiguous()
        g_t = g.transpose(1, 2).contiguous()
        beta_t = beta.float().transpose(1, 2).contiguous()

        core_out, _ = torch_recurrent_gated_delta_rule(
            q_t, k_t, v_t, g=g_t, beta=beta_t,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False  # already normalized
        )
        # core_out: [1, 512, 32, 128]
        print(f"\n=== Step 10: after delta rule ===")
        print(f"core_out: mean={core_out.float().mean().item():.6f} std={core_out.float().std().item():.6f} "
              f"[0,0,0,:3]={core_out[0,0,0,0].float().item():.6f},{core_out[0,0,0,1].float().item():.6f},"
              f"{core_out[0,0,0,2].float().item():.6f}")

        # Step 11: RMSNorm + gate + out_proj
        core_flat = core_out.reshape(-1, core_out.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        variance = core_flat.float().pow(2).mean(-1, keepdim=True)
        normed = (core_flat.float() * torch.rsqrt(variance + la.norm.variance_epsilon) * la.norm.weight.float()).to(core_flat.dtype)
        gated = normed * F.silu(z_flat.float()).to(normed.dtype)
        gated = gated.reshape(batch, seq, -1)
        result = F.linear(gated, la.out_proj.weight)

        print(f"\n=== Step 11: after norm+gate+out_proj ===")
        print(f"result: mean={result.float().mean().item():.6f} std={result.float().std().item():.6f} "
              f"[0,0,:3]={result[0,0,0].float().item():.6f},{result[0,0,1].float().item():.6f},"
              f"{result[0,0,2].float().item():.6f}")

        # Also dump weight shapes
        print(f"\n=== Weight shapes ===")
        print(f"in_proj_qkvz.weight: {list(la.in_proj_qkvz.weight.shape)}")
        print(f"in_proj_ba.weight: {list(la.in_proj_ba.weight.shape)}")
        print(f"conv1d.weight: {list(la.conv1d.weight.shape)}")
        print(f"out_proj.weight: {list(la.out_proj.weight.shape)}")
        print(f"A_log: {list(la.A_log.shape)}")
        print(f"dt_bias: {list(la.dt_bias.shape)}")
        print(f"norm.weight: {list(la.norm.weight.shape)}")

        # Also check if checkpoint has in_proj_qkv or in_proj_qkvz
        print(f"\n=== Check checkpoint weight names ===")
        # List keys in the model state dict that match layer 0
        for name, param in model.named_parameters():
            if f"layers.{linear_layer_idx}.linear_attn" in name:
                print(f"  {name}: {list(param.shape)}")

if __name__ == "__main__":
    main()
