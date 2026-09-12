#!/usr/bin/env python3
"""Multi-LoRA benchmark with EP=4: sends train_step to all 4 servers in parallel."""
import requests, base64, struct, time, threading
from concurrent.futures import ThreadPoolExecutor

WORLD_SIZE = 4
SERVERS = [f"http://localhost:{8080+i}" for i in range(WORLD_SIZE)]
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

def send_to_all(method, path, json=None):
    """Send request to all servers, return list of responses."""
    def fetch(i):
        url = SERVERS[i] + path
        if method == "POST":
            return requests.post(url, json=json)
        return requests.get(url)
    with ThreadPoolExecutor(max_workers=WORLD_SIZE) as pool:
        results = list(pool.map(fetch, range(WORLD_SIZE)))
    return results

def send_to_all_parallel(path, json_data):
    """Send POST to all servers in parallel (for train_step sync)."""
    def fetch(i):
        return requests.post(SERVERS[i] + path, json=json_data)
    with ThreadPoolExecutor(max_workers=WORLD_SIZE) as pool:
        results = list(pool.map(fetch, range(WORLD_SIZE)))
    return results

def bench(n_adp, rank, n_steps):
    sid = f"ep_{n_adp}"

    # Create sessions on all servers
    send_to_all("POST", "/v1/sessions", json={"session_id": sid})

    # Load model on all servers (parallel)
    cfg = make_cfg(sid)
    results = send_to_all_parallel(f"/v1/sessions/{sid}/load_model",
                                   json={"model_path": MODEL, "config_toml": cfg})
    for i, r in enumerate(results):
        if r.status_code != 200:
            print(f"rank {i} load FAIL: {r.text[:200]}")
            return None

    # Load dataset on all servers
    send_to_all_parallel(f"/v1/sessions/{sid}/load_dataset",
                         json={"jsonl_path": "/tmp/qwen3_6_test.jsonl", "seq_len": SEQ})

    # Init LoRA on all servers
    results = send_to_all_parallel(f"/v1/sessions/{sid}/init_lora",
                                   json={"rank": 8, "alpha": 16, "target_layers": [],
                                         "target_modules": ["q_proj","k_proj","v_proj","o_proj",
                                                            "in_proj_qkv","in_proj_z","out_proj"],
                                         "lr": 0.0001, "beta1": 0.9, "beta2": 0.999, "eps": 0.00000001})
    for i, r in enumerate(results):
        if r.status_code != 200:
            print(f"rank {i} init_lora FAIL: {r.text[:200]}")
            return None

    # Add LoRA adapters on all servers
    for i in range(n_adp):
        results = send_to_all_parallel(f"/v1/sessions/{sid}/add_lora",
                                       json={"rank": rank, "alpha": rank*2,
                                             "target_layers": [], "target_modules": ""})
        for j, r in enumerate(results):
            if r.status_code != 200:
                print(f"rank {j} add_lora {i} FAIL: {r.text[:200]}")
                return None

    # Prepare payload
    ids = [1]*SEQ
    mask = [0]*20 + [1]*(SEQ-20)
    payload = {
        "input_ids": {"data": enc(ids), "shape": [1,SEQ], "dtype": "int64"},
        "target_mask": {"data": enc(mask), "shape": [1,SEQ], "dtype": "int64"},
        "attention_mask": {"data": enc([1]*SEQ), "shape": [1,SEQ], "dtype": "int64"},
    }

    # Warmup (all servers in parallel — NCCL syncs them)
    results = send_to_all_parallel(f"/v1/sessions/{sid}/train_step", payload)
    for i, r in enumerate(results):
        if r.status_code != 200:
            print(f"rank {i} warmup FAIL: {r.text[:200]}")
            return None
    loss = results[0].json().get("loss", "?")

    # Timed steps
    times = []
    for s in range(n_steps):
        t0 = time.time()
        results = send_to_all_parallel(f"/v1/sessions/{sid}/train_step", payload)
        t1 = time.time()
        for i, r in enumerate(results):
            if r.status_code != 200:
                print(f"rank {i} step {s} FAIL: {r.text[:200]}")
                break
        times.append((t1-t0)*1000)

    if times:
        avg = sum(times)/len(times)
        print(f"{n_adp} adp: loss={str(loss)[:8]} avg={avg:.1f}ms min={min(times):.1f}ms max={max(times):.1f}ms")
        return avg
    return None

def cleanup(sid):
    """Delete session on all servers to free GPU memory."""
    def fetch(i):
        return requests.delete(SERVERS[i] + f"/v1/sessions/{sid}")
    with ThreadPoolExecutor(max_workers=WORLD_SIZE) as pool:
        list(pool.map(fetch, range(WORLD_SIZE)))

results = {}
for n in [1, 2, 4, 8, 16, 32]:
    t = bench(n, 16, 5)
    if t:
        results[n] = t
    else:
        print(f"Stopping at {n} adapters")
    cleanup(f"ep_{n}")  # Free GPU memory after each test

if results:
    print(f"\n=== Summary (rank=16, Qwen3.6-35B-A3B, EP=4, seq={SEQ}, H20-3e) ===")
    base = results.get(1, 0)
    for n, t in sorted(results.items()):
        ovh = (t/base-1)*100 if base > 0 else 0
        print(f"  {n} adapters: {t:.1f} ms/step ({ovh:.1f}% overhead)")
