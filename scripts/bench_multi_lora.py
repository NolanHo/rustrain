#!/usr/bin/env python3
"""Benchmark: multi-LoRA performance with 32 adapters × rank=16.

Tests:
1. Baseline: 1 adapter, rank=16
2. Multi-LoRA: 32 adapters, rank=16 (sum_ranks=512)
3. Compare train_step latency

Uses rustrain server API (HTTP).
"""
import requests
import json
import time
import sys
import base64
import struct

SERVER = "http://localhost:8080"
MODEL_PATH = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
SEQ_LEN = 512

def encode_tensor_i64(data):
    """Encode a list of int64 as base64 bytes."""
    return base64.b64encode(struct.pack(f'<{len(data)}q', *data)).decode()

def main():
    num_adapters = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    num_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    print(f"=== Multi-LoRA Benchmark: {num_adapters} adapters × rank={rank} ===")
    print(f"sum_ranks = {num_adapters * rank}")

    # 1. Create session
    r = requests.post(f"{SERVER}/v1/sessions", json={"session_id": "bench"})
    assert r.status_code == 200, f"create session: {r.text}"
    print("✓ Session created")

    # 2. Load model
    config_toml = f"""
[run]
name = "bench"
seed = 42

[model]
name = "bench"
architecture = "qwen3_6_lora_sft"
model_path = "{MODEL_PATH}"
vocab_size = 248320
hidden_size = 1024
num_layers = 24
num_attention_heads = 8
num_key_value_heads = 2
intermediate_size = 3584
seq_len = {SEQ_LEN}
norm = "rmsnorm"
activation = "swiglu"
rope = true
rms_norm_eps = 0.000001

[train]
max_steps = {num_steps}
backend = "tch"
micro_batch_size = 1
global_batch_size = 1
gradient_accumulation_steps = 1
learning_rate = 0.0001
weight_decay = 0.0
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_eps = 0.00000001
dtype = "bf16"
device = "cuda"
checkpoint_every = 0
eval_every = 0

[parallel]
tensor_model_parallel_size = 1
pipeline_model_parallel_size = 1
data_parallel_size = 1
expert_model_parallel_size = 1
context_parallel_size = 1

[lora]
rank = {rank}
alpha = {rank * 2}
target_layers = []
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"]

[data]
kind = "instruction_jsonl"
paths = ["/tmp/qwen3_6_test.jsonl"]
"""
    r = requests.post(f"{SERVER}/v1/sessions/bench/load_model",
                       json={"model_path": MODEL_PATH, "config_toml": config_toml})
    assert r.status_code == 200, f"load model: {r.text}"
    print("✓ Model loaded")

    # 3. Load dataset
    r = requests.post(f"{SERVER}/v1/sessions/bench/load_dataset",
                       json={"jsonl_path": "/tmp/qwen3_6_test.jsonl", "seq_len": SEQ_LEN})
    assert r.status_code == 200, f"load dataset: {r.text}"
    print("✓ Dataset loaded")

    # 4. Init LoRA (creates first adapter via init_lora)
    r = requests.post(f"{SERVER}/v1/sessions/bench/init_lora", json={
        "rank": rank,
        "alpha": rank * 2,
        "target_layers": [],
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"],
        "lr": 0.0001,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 0.00000001,
    })
    assert r.status_code == 200, f"init lora: {r.text}"
    count = r.json().get("lora_param_count", 0)
    print(f"✓ Initial LoRA created: {count} params")

    # 5. Add additional adapters
    for i in range(1, num_adapters):
        r = requests.post(f"{SERVER}/v1/sessions/bench/add_lora", json={
            "rank": rank,
            "alpha": rank * 2,
            "target_layers": [],
            "target_modules": "",
        })
        if r.status_code != 200:
            print(f"  add_lora {i} failed: {r.text}")
            break
    print(f"✓ Added {num_adapters - 1} additional adapters (total: {num_adapters})")

    # 6. List adapters
    r = requests.get(f"{SERVER}/v1/sessions/bench/list_lora")
    ids = r.json() if r.status_code == 200 else []
    print(f"✓ Active adapters: {len(ids)}")

    # 7. Prepare dummy input (all zeros, seq_len=512)
    input_ids = [1] * SEQ_LEN  # simple token IDs
    target_mask = [0.0] * 20 + [1.0] * (SEQ_LEN - 20)
    attention_mask = [1.0] * SEQ_LEN

    # 8. Warmup step
    print("\n--- Warmup ---")
    payload = {
        "input_ids": {"data": encode_tensor_i64(input_ids), "shape": [1, SEQ_LEN], "dtype": "int64"},
        "target_mask": {"data": encode_tensor_i64([int(m) for m in target_mask]), "shape": [1, SEQ_LEN], "dtype": "int64"},
        "attention_mask": {"data": encode_tensor_i64([int(m) for m in attention_mask]), "shape": [1, SEQ_LEN], "dtype": "int64"},
    }
    r = requests.post(f"{SERVER}/v1/sessions/bench/train_step", json=payload)
    if r.status_code != 200:
        print(f"  warmup failed: {r.text}")
        return
    print(f"  warmup loss: {r.json()['loss']:.6f}")

    # 9. Benchmark
    print(f"\n--- Benchmark: {num_steps} steps ---")
    times = []
    losses = []
    for step in range(num_steps):
        t0 = time.time()
        r = requests.post(f"{SERVER}/v1/sessions/bench/train_step", json=payload)
        t1 = time.time()
        if r.status_code != 200:
            print(f"  step {step} failed: {r.text}")
            break
        loss = r.json()["loss"]
        elapsed = t1 - t0
        times.append(elapsed)
        losses.append(loss)
        if step % 5 == 0 or step == num_steps - 1:
            print(f"  step {step:3d}: loss={loss:.6f} time={elapsed*1000:.1f}ms")

    # 10. Summary
    if times:
        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000
        print(f"\n=== Summary ===")
        print(f"Adapters: {num_adapters} × rank={rank} (sum_ranks={num_adapters * rank})")
        print(f"Steps: {len(times)}")
        print(f"Avg: {avg_ms:.1f}ms  Min: {min_ms:.1f}ms  Max: {max_ms:.1f}ms")
        print(f"Initial loss: {losses[0]:.6f}  Final loss: {losses[-1]:.6f}")

if __name__ == "__main__":
    main()
