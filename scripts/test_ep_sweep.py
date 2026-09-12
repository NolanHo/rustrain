#!/usr/bin/env python3
"""Test EP=4 with varying adapter counts — find crash threshold."""
import requests, base64, struct, time
from concurrent.futures import ThreadPoolExecutor

SERVERS = [f"http://localhost:{9080+i}" for i in range(4)]
MODEL = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
SEQ = 512

def enc(d):
    return base64.b64encode(struct.pack("<" + "q"*len(d), *d)).decode()

def post_all(path, payload):
    def fetch(i):
        return requests.post(SERVERS[i] + path, json=payload)
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(fetch, range(4)))

def delete_all(path):
    def fetch(i):
        return requests.delete(SERVERS[i] + path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(fetch, range(4)))

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
        "[train]", "max_steps=2", 'backend="tch"',
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

def test_n_adapters(n_adp):
    sid = f"ep_{n_adp}"
    post_all("/v1/sessions", {"session_id": sid})
    cfg = make_cfg(sid)
    r = post_all(f"/v1/sessions/{sid}/load_model", {"model_path": MODEL, "config_toml": cfg})
    for i, resp in enumerate(r):
        if resp.status_code != 200:
            print(f"  {n_adp}adp: rank {i} load FAIL")
            delete_all(f"/v1/sessions/{sid}")
            return False
    post_all(f"/v1/sessions/{sid}/load_dataset", {"jsonl_path": "/tmp/qwen3_6_test.jsonl", "seq_len": SEQ})
    r = post_all(f"/v1/sessions/{sid}/init_lora", {
        "rank": 8, "alpha": 16, "target_layers": [],
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","out_proj"],
        "lr": 0.0001, "beta1": 0.9, "beta2": 0.999, "eps": 0.00000001
    })
    for i, resp in enumerate(r):
        if resp.status_code != 200:
            print(f"  {n_adp}adp: rank {i} init_lora FAIL")
            delete_all(f"/v1/sessions/{sid}")
            return False
    for i in range(n_adp):
        r = post_all(f"/v1/sessions/{sid}/add_lora", {"rank": 16, "alpha": 32, "target_layers": [], "target_modules": ""})
        for j, resp in enumerate(r):
            if resp.status_code != 200:
                print(f"  {n_adp}adp: rank {j} add_lora {i} FAIL")
                delete_all(f"/v1/sessions/{sid}")
                return False

    ids = [1]*SEQ
    mask = [0]*20 + [1]*(SEQ-20)
    payload = {
        "input_ids": {"data": enc(ids), "shape": [1,SEQ], "dtype": "int64"},
        "target_mask": {"data": enc(mask), "shape": [1,SEQ], "dtype": "int64"},
        "attention_mask": {"data": enc([1]*SEQ), "shape": [1,SEQ], "dtype": "int64"},
    }

    for step in range(2):
        t0 = time.time()
        r = post_all(f"/v1/sessions/{sid}/train_step", payload)
        t1 = time.time()
        all_ok = True
        for i, resp in enumerate(r):
            if resp.status_code == 200:
                loss = resp.json().get("loss", "?")
            else:
                print(f"  {n_adp}adp step {step}: rank {i} FAIL: {resp.text[:100]}")
                all_ok = False
        if not all_ok:
            delete_all(f"/v1/sessions/{sid}")
            return False
        print(f"  {n_adp}adp step {step}: OK (loss={str(loss)[:8]}, {(t1-t0)*1000:.0f}ms)")

    delete_all(f"/v1/sessions/{sid}")
    return True

for n in [1, 2, 4, 8]:
    print(f"\n=== Testing {n} adapters ===")
    ok = test_n_adapters(n)
    if ok:
        print(f"  {n} adapters: ALL STEPS PASSED")
    else:
        print(f"  {n} adapters: FAILED")
        break
