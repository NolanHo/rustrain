#!/usr/bin/env python3
"""Diagnostic: dump weight names, shapes, and layouts for linear attention layer 0."""
import json, sys, os
import torch
from safetensors import safe_open

def find_model_dir():
    # Try common locations
    candidates = [
        "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0",
    ]
    # Also check env
    env_path = os.environ.get("MODEL_PATH", "")
    if env_path:
        candidates.insert(0, env_path)
    # Search for config.json with linear_num_key_heads
    for root, dirs, files in os.walk("/mnt/workspace", topdown=True):
        depth = root.count(os.sep) - "/mnt/workspace".count(os.sep)
        if depth > 6:
            dirs.clear()
            continue
        if "config.json" in files:
            try:
                c = json.load(open(os.path.join(root, "config.json")))
                if "linear_num_key_heads" in json.dumps(c) or "qwen3_next" in json.dumps(c):
                    candidates.insert(0, root)
                    break
            except:
                pass
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def main():
    model_dir = find_model_dir()
    if not model_dir:
        print("ERROR: Could not find model directory")
        sys.exit(1)
    print(f"Model dir: {model_dir}")

    # List all weight names
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path))
        weight_map = index["weight_map"]
        # Find layer 0 linear attention weights
        layer0_prefix = "model.layers.0.linear_attn."
        print("\n=== Layer 0 Linear Attention Weights ===")
        for name in sorted(weight_map.keys()):
            if "layers.0.linear_attn" in name:
                shard = weight_map[name]
                shard_path = os.path.join(model_dir, shard)
                with safe_open(shard_path, framework="pt") as f:
                    if name in f.keys():
                        t = f.get_tensor(name)
                        print(f"  {name}: shape={t.shape}, dtype={t.dtype}, "
                              f"mean={t.float().mean().item():.6f}, std={t.float().std().item():.6f}")
                        if t.dim() >= 1 and t.shape[0] >= 8:
                            print(f"    first 8 rows mean: {[t[i].float().mean().item() for i in range(min(8, t.shape[0]))]}")
    else:
        single = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single):
            with safe_open(single, framework="pt") as f:
                for name in f.keys():
                    if "layers.0.linear_attn" in name:
                        t = f.get_tensor(name)
                        print(f"  {name}: shape={t.shape}, dtype={t.dtype}")

    # Also check config
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        c = json.load(open(config_path))
        tc = c.get("text_config", c)
        print("\n=== Linear Attention Config ===")
        for k in ["linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
                   "linear_value_head_dim", "linear_conv_kernel_dim"]:
            print(f"  {k}: {tc.get(k, 'N/A')}")
        print(f"  layer_types: {tc.get('layer_types', 'N/A')[:10]}...")
        print(f"  model_type: {tc.get('model_type', 'N/A')}")

if __name__ == "__main__":
    main()
