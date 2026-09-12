#!/usr/bin/env python3
"""Minimal test: verify that flat QKV split vs per-head interleaved split
produces different results, confirming the bug we fixed.

This doesn't need transformers — just numpy.
"""
import numpy as np

def main():
    # Simulate: batch=1, seq=4, num_k_heads=2, key_dim=4, num_v_heads=4, val_dim=4
    # qkv_dim = 2*4*2 + 4*4 = 16 + 16 = 32
    num_k_heads = 2
    key_dim = 4
    num_v_heads = 4
    val_dim = 4
    seq = 4
    
    q_size = num_k_heads * key_dim  # 8
    v_size = num_v_heads * val_dim   # 16
    qkv_dim = q_size * 2 + v_size   # 32
    
    # Create random qkv_conv output
    np.random.seed(42)
    qkv_conv = np.random.randn(1, seq, qkv_dim).astype(np.float32)
    
    print(f"qkv_conv shape: {qkv_conv.shape}")
    print(f"qkv_conv[0,0,:8] = {qkv_conv[0,0,:8]}")
    
    # Method 1: FLAT split (correct, matches transformers Qwen3.5)
    # Q = [0:q_size], K = [q_size:2*q_size], V = [2*q_size:]
    q_flat = qkv_conv[:, :, 0:q_size].reshape(1, seq, num_k_heads, key_dim)
    k_flat = qkv_conv[:, :, q_size:2*q_size].reshape(1, seq, num_k_heads, key_dim)
    v_flat = qkv_conv[:, :, 2*q_size:].reshape(1, seq, num_v_heads, val_dim)
    
    print(f"\n=== FLAT split (correct) ===")
    print(f"Q[0,0,0,:4] = {q_flat[0,0,0,:]}")
    print(f"K[0,0,0,:4] = {k_flat[0,0,0,:]}")
    print(f"V[0,0,0,:4] = {v_flat[0,0,0,:]}")
    
    # Method 2: PER-HEAD interleaved split (old buggy code)
    # Reshape to [batch, seq, num_k_heads, per_head] then narrow
    v_per_k = num_v_heads // num_k_heads  # 2
    per_head = key_dim + key_dim + val_dim * v_per_k  # 4+4+8 = 16
    qkv_reshaped = qkv_conv.reshape(1, seq, num_k_heads, per_head)
    q_ph = qkv_reshaped[:, :, :, 0:key_dim]
    k_ph = qkv_reshaped[:, :, :, key_dim:2*key_dim]
    v_ph = qkv_reshaped[:, :, :, 2*key_dim:].reshape(1, seq, num_v_heads, val_dim)
    
    print(f"\n=== PER-HEAD interleaved split (old buggy) ===")
    print(f"Q[0,0,0,:4] = {q_ph[0,0,0,:]}")
    print(f"K[0,0,0,:4] = {k_ph[0,0,0,:]}")
    print(f"V[0,0,0,:4] = {v_ph[0,0,0,:]}")
    
    # Show the difference
    print(f"\n=== Difference ===")
    print(f"Q diff: {np.abs(q_flat - q_ph).max():.6f}")
    print(f"K diff: {np.abs(k_flat - k_ph).max():.6f}")
    print(f"V diff: {np.abs(v_flat - v_ph).max():.6f}")
    
    if np.abs(q_flat - q_ph).max() > 1e-6:
        print("\n✅ CONFIRMED: flat split and per-head split produce DIFFERENT results!")
        print("   The old per-head interleaved split was WRONG for Qwen3.5/3.6 models.")
        print("   Fix: use flat split (Q_all | K_all | V_all) to match transformers.")
    else:
        print("\n❓ No difference detected (may be model-specific)")

if __name__ == "__main__":
    main()
