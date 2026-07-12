#!/usr/bin/env python3
"""EP=4 benchmark — single HTTP request to EP server (no parallel requests)."""
import requests, base64, struct, time

SERVER = "http://localhost:8080"
MODEL = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
SEQ = 512

def enc(d):
    return base64.b64encode(struct.pack("<" + "q"*len(d), *d)).decode()

def make_cfg(sid):
    lt = '["linear_attention","linear_attention","linear_attention","full_attention"]'
    lt_full = ",".join([lt] * 10)
    return "\n".join([
        "[run]", f'name="{sid}"', "seed=42",
        "[model]", f'name="{sid}"', 'architecture="qwen3_6_lora_sft"',
        f'model_path="{MODEL}"', "vocab_size=248320",
        "hidden_size=2048", "num_layers=40", "num_attention_heads=16",
        "num_key_value_heads=2", "head_dim=256",
        f"seq_len={SEQ}", 'norm="rmsnorm"', 'activation="swiglu"',
        "rope=true", "rms_norm_eps=0.000001", "partial_rotary_factor=0.25",
        f"layer_types={lt_full}",
        "linear_num_key_heads=16", "linear_key_head_dim=128",
        "linear_num_value_heads=32", "linear_value_head_dim=128",
        "linear_conv_kernel_dim=4",
        "num_experts=256", "num_experts_per_tok=8", "moe_intermediate_size=512",
        "[train]", "max_steps=5", 'backend="tch"',
        "micro_batch_size=1", "global_batch_size=1",
        "gradient_accumulation_steps=1", "learning_rate=0.0001",
        'dtype="bf16"', 'device="cuda"',
        "checkpoint_every=0", "eval_every=0",
        "[parallel]", "tensor_model_parallel_size=1",
        "pipeline_model_parallel_size=1", "data_parallel_size=1",
        "expert_model_parallel_size=1", "context_parallel_size=1",
        "[lora]", "rank=8", "alpha=16", "target_layers=[]",
        'target_modules=["q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","out_proj"]',
        "[data]", 'kind="instruction_jsonl"',
        'paths=["/tmp/qwen3_6_test.jsonl"]',
    ])

def bench(n_adp, rank, n_steps):
    sid = f"ep_{n_adp}"
    requests.post(f"{SERVER}/v1/sessions", json={"session_id": sid})
    cfg = make_cfg(sid)
    r = requests.post(f"{SERVER}/v1/sessions/{sid}/load_model",
                      json={"model_path": MODEL, "config_toml": cfg})
    if r.status_code != 200:
        print(f"{n_adp} adp load FAIL: {r.text[:200]}")
        return None
    requests.post(f"{SERVER}/v1/sessions/{sid}/load_dataset",
                  json={"jsonl_path": "/tmp/qwen3_6_test.jsonl", "seq_len": SEQ})
    r = requests.post(f"{SERVER}/v1/sessions/{sid}/init_lora",
                      json={"rank": 8, "alpha": 16, "target_layers": [],
                            "target_modules": ["q_proj","k_proj","v_proj","o_proj",
                                               "in_proj_qkv","in_proj_z","out_proj"],
                            "lr": 0.0001, "beta1": 0.9, "beta2": 0.999, "eps": 0.00000001})
    if r.status_code != 200:
        print(f"{n_adp} adp init_lora FAIL: {r.text[:200]}")
        return None
    for i in range(n_adp):
        r = requests.post(f"{SERVER}/v1/sessions/{sid}/add_lora",
                          json={"rank": rank, "alpha": rank*2,
                                "target_layers": [], "target_modules": ""})
        if r.status_code != 200:
            print(f"{n_adp} adp add_lora {i} FAIL: {r.text[:200]}")
            return None
    ids = [1]*SEQ
    mask = [0]*20 + [1]*(SEQ-20)
    payload = {
        "input_ids": {"data": enc(ids), "shape": [1,SEQ], "dtype": "int64"},
        "target_mask": {"data": enc(mask), "shape": [1,SEQ], "dtype": "int64"},
        "attention_mask": {"data": enc([1]*SEQ), "shape": [1,SEQ], "dtype": "int64"},
    }
    # Warmup
    r = requests.post(f"{SERVER}/v1/sessions/{sid}/train_step", json=payload)
    if r.status_code != 200:
        print(f"{n_adp} adp warmup FAIL: {r.text[:200]}")
        return None
    loss = r.json().get("loss", "?")
    # Timed steps
    times = []
    for s in range(n_steps):
        t0 = time.time()
        r = requests.post(f"{SERVER}/v1/sessions/{sid}/train_step", json=payload)
        t1 = time.time()
        if r.status_code != 200:
            print(f"{n_adp} adp step {s} FAIL: {r.text[:200]}")
            break
        times.append((t1-t0)*1000)
    # Cleanup
    requests.delete(f"{SERVER}/v1/sessions/{sid}")
    if times:
        avg = sum(times)/len(times)
        print(f"{n_adp} adp: loss={str(loss)[:8]} avg={avg:.1f}ms min={min(times):.1f}ms max={max(times):.1f}ms")
        return avg
    return None

results = {}
for n in [1, 2, 4, 8, 16, 32]:
    t = bench(n, 16, 5)
    if t:
        results[n] = t
    else:
        print(f"Stopping at {n} adapters")

if results:
    print(f"\n=== Summary (rank=16, Qwen3.6-35B-A3B, EP=4, BF16 LoRA, seq={SEQ}, H20-3e) ===")
    base = results.get(1, 0)
    for n, t in sorted(results.items()):
        ovh = (t/base-1)*100 if base > 0 else 0
        print(f"  {n} adapters: {t:.1f} ms/step ({ovh:.1f}% overhead)")
