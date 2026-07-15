#!/usr/bin/env python3
"""Diagnostic: dump linear attention intermediate values using safetensors directly.

Only loads layer 0 weights (not the full model) to avoid OOM on CPU.
"""
import os, sys, json
import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/workspace/share/models/Qwen3.5-9B")

def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def main():
    # Load index to find shard files
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Find first linear attention layer (layer 0 for Qwen3.5)
    layer_idx = 0
    prefix = f"model.language_model.layers.{layer_idx}.linear_attn"

    # Load only needed tensors for layer 0
    needed = [
        f"{prefix}.in_proj_qkv.weight",
        f"{prefix}.in_proj_z.weight",
        f"{prefix}.in_proj_a.weight",
        f"{prefix}.in_proj_b.weight",
        f"{prefix}.A_log",
        f"{prefix}.dt_bias",
        f"{prefix}.conv1d.weight",
        f"{prefix}.norm.weight",
        f"{prefix}.out_proj.weight",
        f"model.language_model.layers.{layer_idx}.input_layernorm.weight",
        f"model.language_model.embed_tokens.weight",
    ]

    tensors = {}
    loaded_shards = set()
    for name in needed:
        shard = weight_map.get(name)
        if shard and shard not in loaded_shards:
            shard_path = os.path.join(MODEL_PATH, shard)
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if key in needed:
                        tensors[key] = f.get_tensor(key)
            loaded_shards.add(shard)

    # Config
    config_path = os.path.join(MODEL_PATH, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    tc = config.get("text_config", config)

    hidden_size = tc["hidden_size"]
    num_k_heads = tc["linear_num_key_heads"]
    head_k_dim = tc["linear_key_head_dim"]
    num_v_heads = tc["linear_num_value_heads"]
    head_v_dim = tc["linear_value_head_dim"]
    conv_kernel = tc["linear_conv_kernel_dim"]
    rms_eps = tc.get("rms_norm_eps", 1e-6)

    key_dim = num_k_heads * head_k_dim  # 16*128 = 2048
    value_dim = num_v_heads * head_v_dim  # 32*128 = 4096
    qkv_dim = key_dim * 2 + value_dim  # 8192

    print(f"hidden_size={hidden_size}, key_dim={key_dim}, value_dim={value_dim}")
    print(f"num_k_heads={num_k_heads}, head_k_dim={head_k_dim}")
    print(f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}")

    # Create dummy input (all 1s, matching ep-bench)
    batch, seq = 1, 512
    input_ids = torch.ones(batch, seq, dtype=torch.long)

    # Embedding
    embed_w = tensors[f"model.language_model.embed_tokens.weight"]
    hidden = F.embedding(input_ids, embed_w)  # [1, 512, hidden_size]

    # Apply input_layernorm
    layernorm_w = tensors[f"model.language_model.layers.{layer_idx}.input_layernorm.weight"]
    variance = hidden.float().pow(2).mean(-1, keepdim=True)
    hidden = (hidden.float() * torch.rsqrt(variance + rms_eps) * layernorm_w.float()).to(hidden.dtype)
    h = hidden.float()

    print(f"\n=== Input to linear attention (after layernorm) ===")
    print(f"hidden: mean={h.mean().item():.6f} std={h.std().item():.6f} "
          f"[0,0,:3]={h[0,0,0].item():.6f},{h[0,0,1].item():.6f},{h[0,0,2].item():.6f}")

    # Load weights as float32
    in_proj_qkv = tensors[f"{prefix}.in_proj_qkv.weight"].float()  # [qkv_dim, hidden]
    in_proj_z = tensors[f"{prefix}.in_proj_z.weight"].float()
    in_proj_a = tensors[f"{prefix}.in_proj_a.weight"].float()
    in_proj_b = tensors[f"{prefix}.in_proj_b.weight"].float()
    a_log = tensors[f"{prefix}.A_log"].float()
    dt_bias = tensors[f"{prefix}.dt_bias"].float()
    conv1d_w = tensors[f"{prefix}.conv1d.weight"].float()
    norm_w = tensors[f"{prefix}.norm.weight"].float()
    out_proj = tensors[f"{prefix}.out_proj.weight"].float()

    print(f"\n=== Weight shapes ===")
    print(f"in_proj_qkv: {list(in_proj_qkv.shape)}")  # [8192, 4096]
    print(f"in_proj_z: {list(in_proj_z.shape)}")
    print(f"conv1d_w: {list(conv1d_w.shape)}")
    print(f"A_log: {list(a_log.shape)}")
    print(f"dt_bias: {list(dt_bias.shape)}")

    # Step 1: in_proj_qkv
    qkv = F.linear(h, in_proj_qkv)  # [1, 512, 8192]
    print(f"\n=== Step 1: in_proj_qkv ===")
    print(f"qkv_proj: shape={list(qkv.shape)} mean={qkv.mean().item():.6f} "
          f"std={qkv.std().item():.6f} "
          f"[0,0,:5]={qkv[0,0,0].item():.6f},{qkv[0,0,1].item():.6f},"
          f"{qkv[0,0,2].item():.6f},{qkv[0,0,3].item():.6f},"
          f"{qkv[0,0,4].item():.6f}")

    # Step 2: Conv1d
    qkv_t = qkv.transpose(1, 2)  # [1, 8192, 512]
    pad = conv_kernel - 1
    padding = torch.zeros(1, qkv_dim, pad)
    padded = torch.cat([padding, qkv_t], dim=2)
    conv_out = F.conv1d(padded, conv1d_w, bias=None, stride=1, padding=0, dilation=1, groups=qkv_dim)
    conv_out = F.silu(conv_out[:, :, :seq])
    qkv_conv = conv_out.transpose(1, 2)  # [1, 512, 8192]

    print(f"\n=== Step 2: after conv1d+silu ===")
    print(f"qkv_conv: mean={qkv_conv.mean().item():.6f} "
          f"std={qkv_conv.std().item():.6f} "
          f"[0,0,:5]={qkv_conv[0,0,0].item():.6f},{qkv_conv[0,0,1].item():.6f},"
          f"{qkv_conv[0,0,2].item():.6f},{qkv_conv[0,0,3].item():.6f},"
          f"{qkv_conv[0,0,4].item():.6f}")

    # Step 3: FLAT split (matching transformers Qwen3_5GatedDeltaNet)
    q_flat = qkv_conv[:, :, 0:key_dim].reshape(batch, seq, num_k_heads, head_k_dim)
    k_flat = qkv_conv[:, :, key_dim:2*key_dim].reshape(batch, seq, num_k_heads, head_k_dim)
    v_flat = qkv_conv[:, :, 2*key_dim:].reshape(batch, seq, num_v_heads, head_v_dim)

    print(f"\n=== Step 3: after FLAT split (transformers compatible) ===")
    print(f"Q: mean={q_flat.mean().item():.6f} std={q_flat.std().item():.6f} "
          f"[0,0,0,:3]={q_flat[0,0,0,0].item():.6f},{q_flat[0,0,0,1].item():.6f},{q_flat[0,0,0,2].item():.6f}")
    print(f"K: mean={k_flat.mean().item():.6f} std={k_flat.std().item():.6f} "
          f"[0,0,0,:3]={k_flat[0,0,0,0].item():.6f},{k_flat[0,0,0,1].item():.6f},{k_flat[0,0,0,2].item():.6f}")
    print(f"V: mean={v_flat.mean().item():.6f} std={v_flat.std().item():.6f} "
          f"[0,0,0,:3]={v_flat[0,0,0,0].item():.6f},{v_flat[0,0,0,1].item():.6f},{v_flat[0,0,0,2].item():.6f}")

    # Step 4: Per-head interleaved split (old buggy rustrain)
    v_per_k = num_v_heads // num_k_heads
    per_head = head_k_dim + head_k_dim + head_v_dim * v_per_k  # 128+128+256 = 512
    qkv_reshaped = qkv_conv.reshape(batch, seq, num_k_heads, per_head)
    q_ph = qkv_reshaped[:, :, :, 0:head_k_dim]
    k_ph = qkv_reshaped[:, :, :, head_k_dim:2*head_k_dim]
    v_ph = qkv_reshaped[:, :, :, 2*head_k_dim:].reshape(batch, seq, num_v_heads, head_v_dim)

    print(f"\n=== Step 3b: PER-HEAD split (old buggy rustrain) ===")
    print(f"Q: mean={q_ph.mean().item():.6f} std={q_ph.std().item():.6f} "
          f"[0,0,0,:3]={q_ph[0,0,0,0].item():.6f},{q_ph[0,0,0,1].item():.6f},{q_ph[0,0,0,2].item():.6f}")
    print(f"K: mean={k_ph.mean().item():.6f} std={k_ph.std().item():.6f} "
          f"[0,0,0,:3]={k_ph[0,0,0,0].item():.6f},{k_ph[0,0,0,1].item():.6f},{k_ph[0,0,0,2].item():.6f}")
    print(f"V: mean={v_ph.mean().item():.6f} std={v_ph.std().item():.6f} "
          f"[0,0,0,:3]={v_ph[0,0,0,0].item():.6f},{v_ph[0,0,0,1].item():.6f},{v_ph[0,0,0,2].item():.6f}")

    print(f"\n=== Difference between split methods ===")
    print(f"Q diff: max={torch.abs(q_flat - q_ph).max().item():.6f}")
    print(f"K diff: max={torch.abs(k_flat - k_ph).max().item():.6f}")
    print(f"V diff: max={torch.abs(v_flat - v_ph).max().item():.6f}")

    # Step 5: Continue with FLAT split (correct path)
    # Project a, b, z
    a = F.linear(h, in_proj_a)  # [1, 512, num_v_heads]
    b = F.linear(h, in_proj_b)
    z = F.linear(h, in_proj_z).reshape(batch, seq, num_v_heads, head_v_dim)

    # g, beta
    g = -a_log.exp().unsqueeze(0).unsqueeze(0) * F.softplus(a + dt_bias.unsqueeze(0).unsqueeze(0))
    beta = b.sigmoid()

    print(f"\n=== Step 5: g, beta ===")
    print(f"g: mean={g.mean().item():.6f} std={g.std().item():.6f} "
          f"[0,0,:3]={g[0,0,0].item():.6f},{g[0,0,1].item():.6f},{g[0,0,2].item():.6f}")
    print(f"beta: mean={beta.mean().item():.6f} std={beta.std().item():.6f} "
          f"[0,0,:3]={beta[0,0,0].item():.6f},{beta[0,0,1].item():.6f},{beta[0,0,2].item():.6f}")

    # Expand Q/K to v_heads
    n_rep = num_v_heads // num_k_heads
    q = q_flat.repeat_interleave(n_rep, dim=2)
    k = k_flat.repeat_interleave(n_rep, dim=2)

    # L2 norm
    q_norm = l2norm(q, dim=-1, eps=1e-6)
    k_norm = l2norm(k, dim=-1, eps=1e-6)
    scale = 1.0 / (head_k_dim ** 0.5)
    q_scaled = q_norm * scale

    print(f"\n=== Step 6: after L2 norm ===")
    print(f"Q: mean={q_norm.mean().item():.6f} std={q_norm.std().item():.6f} "
          f"[0,0,0,:3]={q_norm[0,0,0,0].item():.6f},{q_norm[0,0,0,1].item():.6f},{q_norm[0,0,0,2].item():.6f}")
    print(f"K: mean={k_norm.mean().item():.6f} std={k_norm.std().item():.6f} "
          f"[0,0,0,:3]={k_norm[0,0,0,0].item():.6f},{k_norm[0,0,0,1].item():.6f},{k_norm[0,0,0,2].item():.6f}")

    # Delta rule (recurrent, matching torch_recurrent_gated_delta_rule)
    q_t = q_scaled.transpose(1, 2).contiguous()  # [1, 32, 512, 128]
    k_t = k_norm.transpose(1, 2).contiguous()
    v_t = v_flat.transpose(1, 2).contiguous()
    g_t = g.transpose(1, 2).contiguous()
    beta_t = beta.transpose(1, 2).contiguous()

    BH, S, D_K = q_t.shape[0] * q_t.shape[1], q_t.shape[2], q_t.shape[3]
    D_V = v_t.shape[3]
    state = torch.zeros(q_t.shape[0], q_t.shape[1], D_K, D_V)
    core_out = torch.zeros(q_t.shape[0], q_t.shape[1], S, D_V)

    g_exp = g_t.exp()
    for i in range(S):
        q_i = q_t[:, :, i]
        k_i = k_t[:, :, i]
        v_i = v_t[:, :, i]
        g_i = g_exp[:, :, i].unsqueeze(-1).unsqueeze(-1)
        beta_i = beta_t[:, :, i].unsqueeze(-1)

        state = state * g_i
        kv_mem = (state * k_i.unsqueeze(-1)).sum(dim=-2)
        delta = (v_i - kv_mem) * beta_i
        state = state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        core_out[:, :, i] = (state * q_i.unsqueeze(-1)).sum(dim=-2)

    # [1, 32, 512, 128] -> [1, 512, 32, 128]
    core_out = core_out.transpose(1, 2)

    print(f"\n=== Step 7: after delta rule ===")
    print(f"core_out: mean={core_out.mean().item():.6f} std={core_out.std().item():.6f} "
          f"[0,0,0,:3]={core_out[0,0,0,0].item():.6f},{core_out[0,0,0,1].item():.6f},"
          f"{core_out[0,0,0,2].item():.6f}")

    # RMSNorm + gate + out_proj
    core_flat = core_out.reshape(-1, head_v_dim)
    z_flat = z.reshape(-1, head_v_dim)
    variance = core_flat.pow(2).mean(-1, keepdim=True)
    normed = core_flat * torch.rsqrt(variance + rms_eps) * norm_w
    gated = normed * F.silu(z_flat)
    gated = gated.reshape(batch, seq, -1)
    result = F.linear(gated, out_proj)

    print(f"\n=== Step 8: after norm+gate+out_proj ===")
    print(f"result: mean={result.mean().item():.6f} std={result.std().item():.6f} "
          f"[0,0,:3]={result[0,0,0].item():.6f},{result[0,0,1].item():.6f},"
          f"{result[0,0,2].item():.6f}")

if __name__ == "__main__":
    main()
