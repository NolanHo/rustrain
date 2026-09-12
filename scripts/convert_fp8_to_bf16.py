#!/usr/bin/env python3
"""Convert FP8 safetensors shards to bf16 for tch-rs compatibility."""
import sys
import json
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def convert_shard(input_path, output_path, needed_names=None):
    """Convert a safetensors shard from FP8 to bf16."""
    tensors = {}
    with safe_open(input_path, framework="pt") as f:
        for key in f.keys():
            if needed_names is not None and key not in needed_names:
                continue
            t = f.get_tensor(key)
            if t.dtype == torch.float8_e4m3fn:
                t = t.to(torch.bfloat16)
            tensors[key] = t.cpu()
    if tensors:
        save_file(tensors, output_path)
        print(f"Converted {len(tensors)} tensors to {output_path}")
    return len(tensors)

if __name__ == "__main__":
    model_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else model_dir + "_bf16"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load index
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shards = sorted(set(weight_map.values()))
        for shard in shards:
            input_path = os.path.join(model_dir, shard)
            output_path = os.path.join(output_dir, shard)
            convert_shard(input_path, output_path)
        # Copy index.json
        import shutil
        shutil.copy(index_path, os.path.join(output_dir, "model.safetensors.index.json"))
    else:
        # Single file
        input_path = os.path.join(model_dir, "model.safetensors")
        output_path = os.path.join(output_dir, "model.safetensors")
        convert_shard(input_path, output_path)
    
    # Copy config and tokenizer
    for f in ["config.json", "tokenizer.json"]:
        src = os.path.join(model_dir, f)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(output_dir, f))
