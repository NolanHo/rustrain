#!/usr/bin/env python3
"""Benchmark batched multi-LoRA: train_multi_lora endpoint.
Tests: 2, 8, 32, 128 adapters with rank=1, 8, 16.
Compares vs serial train_step (1 adapter at a time).
"""
import requests, base64, struct, time, json, sys

SERVER = "http://localhost:8080"
MODEL = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
SEQ = 512

def enc(d):
    return base64.b64encode(struct.pack("<" + "q"*len(d), *d)).decode()

def tensor(d, shape=None):
    shape = shape or [1, SEQ]
    return {"data": enc(d), "shape": shape, "dtype": "int64"}

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
        "[train]", "max_steps=10000", 'backend="tch"',
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

def setup(sid, n_adp, lora_rank=8):
    requests.post(f"{SERVER}/v1/sessions", json={"session_id": sid})
    cfg = make_cfg(sid)
    r = requests.post(f"{SERVER}/v1/sessions/{sid}/load_model",
                      json={"model_path": MODEL, "config_toml": cfg}, timeout=300)
    if r.status_code != 200:
        print(f"  load_model FAIL: {r.text[:300]}")
        return False
    requests.post(f"{SERVER}/v1/sessions/{sid}/load_dataset",
                  json={"jsonl_path": "/tmp/qwen3_6_test.jsonl", "seq_len": SEQ})
    r = requests.post(f"{SERVER}/v1/sessions/{sid}/init_lora",
                      json={"rank": 8, "alpha": 16, "target_layers": [],
                            "target_modules": ["q_proj","k_proj","v_proj","o_proj",
                                               "in_proj_qkv","in_proj_z","out_proj"],
                            "lr": 0.0001, "beta1": 0.9, "beta2": 0.999, "eps": 0.00000001})
    if r.status_code != 200:
        print(f"  init_lora FAIL: {r.text[:300]}")
        return False
    for i in range(n_adp):
        r = requests.post(f"{SERVER}/v1/sessions/{sid}/add_lora",
                          json={"rank": lora_rank, "alpha": lora_rank*2,
                                "target_layers": [], "target_modules": ""})
        if r.status_code != 200:
            print(f"  add_lora {i} FAIL: {r.text[:300]}")
            return False
    return True

def bench_multi(n_adp, lora_rank, n_steps=5):
    sid = f"ep_multi_{n_adp}_{lora_rank}"
    print(f"\n{'='*60}")
    print(f"  {n_adp} adapters, rank={lora_rank}, {n_steps} steps (batched)")
    print(f"{'='*60}")
    if not setup(sid, n_adp, lora_rank):
        return None

    ids = [1]*SEQ
    mask = [0]*20 + [1]*(SEQ-20)
    payload = {
        "input_ids": tensor(ids),
        "target_mask": tensor(mask),
        "attention_mask": tensor([1]*SEQ),
    }

    # Warmup
    t0 = time.time()
    r = requests.post(f"{SERVER}/v1/sessions/{sid}/train_multi", json=payload, timeout=600)
    t1 = time.time()
    if r.status_code != 200:
        print(f"  warmup FAIL: {r.text[:300]}")
        requests.delete(f"{SERVER}/v1/sessions/{sid}")
        return None
    warmup_loss = r.json().get("loss", -1)
    print(f"  warmup: loss={warmup_loss:.6f} time={1000*(t1-t0):.0f}ms")

    # Timed steps
    times = []
    losses = []
    for s in range(n_steps):
        t0 = time.time()
        r = requests.post(f"{SERVER}/v1/sessions/{sid}/train_multi", json=payload, timeout=600)
        t1 = time.time()
        if r.status_code != 200:
            print(f"  step {s} FAIL: {r.text[:300]}")
            break
        loss = r.json().get("loss", -1)
        losses.append(loss)
        times.append((t1-t0)*1000)
        print(f"  step {s}: loss={loss:.6f} time={times[-1]:.0f}ms")
        sys.stdout.flush()

    requests.delete(f"{SERVER}/v1/sessions/{sid}")
    if not times:
        return None
    avg = sum(times)/len(times)
    print(f"  avg={avg:.0f}ms min={min(times):.0f}ms max={max(times):.0f}ms")
    print(f"  loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"  per-adapter: {avg/n_adp:.1f}ms")
    return {"n": n_adp, "rank": lora_rank, "avg": avg, "min": min(times), "max": max(times),
            "losses": losses, "per_adp": avg/n_adp}

def bench_serial(n_adp, lora_rank, n_steps=3):
    """Serial: N separate train_step calls (baseline)."""
    sid = f"ep_serial_{n_adp}_{lora_rank}"
    print(f"\n  [serial baseline] {n_adp} adapters, rank={lora_rank}")
    if not setup(sid, n_adp, lora_rank):
        return None
    ids = [1]*SEQ
    mask = [0]*20 + [1]*(SEQ-20)
    payload = {
        "input_ids": tensor(ids),
        "target_mask": tensor(mask),
        "attention_mask": tensor([1]*SEQ),
    }
    # Warmup
    r = requests.post(f"{SERVER}/v1/sessions/{sid}/train_step", json=payload, timeout=300)
    if r.status_code != 200:
        print(f"  serial warmup FAIL: {r.text[:200]}")
        requests.delete(f"{SERVER}/v1/sessions/{sid}")
        return None
    times = []
    for s in range(n_steps):
        t0 = time.time()
        r = requests.post(f"{SERVER}/v1/sessions/{sid}/train_step", json=payload, timeout=300)
        t1 = time.time()
        if r.status_code != 200:
            break
        times.append((t1-t0)*1000)
    requests.delete(f"{SERVER}/v1/sessions/{sid}")
    if times:
        avg = sum(times)/len(times)
        serial_total = avg * n_adp
        print(f"  serial: {avg:.0f}ms/step × {n_adp} = {serial_total:.0f}ms total")
        return serial_total
    return None

# ─── Main ───
results = {}
for n in [2, 8, 32]:
    for rank in [1, 8]:
        r = bench_multi(n, rank, n_steps=5)
        if r:
            results[f"{n}_{rank}"] = r
        else:
            print(f"  {n} adapters rank={rank} FAILED")

# Serial baseline (1 adapter only, then extrapolate)
serial_1 = bench_serial(1, 8, n_steps=3)

if results:
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    if serial_1:
        print(f"  Serial (1 adp, rank=8): {serial_1:.0f}ms extrapolated for N adapters")
    for key, r in sorted(results.items()):
        n, rank = key.split("_")
        speedup_vs_serial = serial_1 / r["avg"] * int(n) if serial_1 else 0
        print(f"  {r['n']:3d} adp rank={r['rank']:2d}: avg={r['avg']:.0f}ms per-adp={r['per_adp']:.1f}ms"
              f"  loss={r['losses'][0]:.4f}->{r['losses'][-1]:.6f}"
              f"  {'speedup=' + f'{speedup_vs_serial:.1f}x' if serial_1 else ''}")
