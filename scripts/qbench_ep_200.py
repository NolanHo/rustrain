#!/usr/bin/env python3
"""EP=4 long-run benchmark — 200 steps, loss convergence, overfit, stability."""
import requests, base64, struct, time, sys, json

SERVER = "http://localhost:8080"
MODEL = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
SEQ = 512
STEPS = 200

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

def setup_session(sid, n_adp, lora_rank=16):
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

def run_steps(sid, n_steps):
    ids = [1]*SEQ
    mask = [0]*20 + [1]*(SEQ-20)
    payload = {
        "input_ids": {"data": enc(ids), "shape": [1,SEQ], "dtype": "int64"},
        "target_mask": {"data": enc(mask), "shape": [1,SEQ], "dtype": "int64"},
        "attention_mask": {"data": enc([1]*SEQ), "shape": [1,SEQ], "dtype": "int64"},
    }
    losses = []
    times = []
    for s in range(n_steps):
        t0 = time.time()
        r = requests.post(f"{SERVER}/v1/sessions/{sid}/train_step", json=payload, timeout=120)
        t1 = time.time()
        if r.status_code != 200:
            print(f"  step {s} FAIL: {r.text[:300]}")
            break
        loss = r.json().get("loss", -1)
        losses.append(loss)
        times.append((t1-t0)*1000)
        if s % 10 == 0 or s == n_steps - 1:
            recent_avg = sum(times[max(0,s-9):s+1])/min(10, s+1)
            print(f"  step {s:4d}: loss={loss:.6f}  time={times[-1]:.0f}ms  (avg10={recent_avg:.0f}ms)")
        sys.stdout.flush()
    return losses, times

def bench(n_adp, n_steps):
    sid = f"ep_long_{n_adp}"
    print(f"\n{'='*60}")
    print(f"  {n_adp} adapters x {n_steps} steps")
    print(f"{'='*60}")
    if not setup_session(sid, n_adp):
        print("  SETUP FAILED")
        return None
    # Warmup
    ids = [1]*SEQ
    mask = [0]*20 + [1]*(SEQ-20)
    payload = {
        "input_ids": {"data": enc(ids), "shape": [1,SEQ], "dtype": "int64"},
        "target_mask": {"data": enc(mask), "shape": [1,SEQ], "dtype": "int64"},
        "attention_mask": {"data": enc([1]*SEQ), "shape": [1,SEQ], "dtype": "int64"},
    }
    r = requests.post(f"{SERVER}/v1/sessions/{sid}/train_step", json=payload, timeout=120)
    if r.status_code != 200:
        print(f"  warmup FAIL: {r.text[:300]}")
        requests.delete(f"{SERVER}/v1/sessions/{sid}")
        return None
    warmup_loss = r.json().get("loss", -1)
    print(f"  warmup: loss={warmup_loss:.6f}")
    losses, times = run_steps(sid, n_steps)
    requests.delete(f"{SERVER}/v1/sessions/{sid}")
    if not times:
        print("  NO SUCCESSFUL STEPS")
        return None
    avg_t = sum(times)/len(times)
    min_t = min(times); max_t = max(times)
    sorted_t = sorted(times)
    p50 = sorted_t[len(sorted_t)//2]
    p95 = sorted_t[int(len(sorted_t)*0.95)]
    print(f"\n  Summary ({n_adp} adapters):")
    print(f"    steps: {len(losses)}")
    print(f"    loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"    time: avg={avg_t:.0f}ms  min={min_t:.0f}ms  max={max_t:.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms")
    print(f"    throughput: {1000/avg_t:.2f} steps/s")
    # Save raw data as JSON for plotting
    with open(f"/tmp/ep_loss_{n_adp}.json", "w") as f:
        json.dump({"warmup": warmup_loss, "losses": losses, "times": times}, f)
    return {"n": n_adp, "losses": losses, "times": times, "avg": avg_t,
            "min": min_t, "max": max_t, "p50": p50, "p95": p95, "warmup": warmup_loss}

results = {}
for n in [1, 8, 32]:
    r = bench(n, STEPS)
    if r:
        results[n] = r
    else:
        print(f"\n  {n} adapters FAILED, stopping")
        break

if results:
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON ({STEPS} steps each)")
    print(f"{'='*60}")
    base_avg = results.get(1, {}).get("avg", 0)
    for n, r in sorted(results.items()):
        ovh = (r["avg"]/base_avg - 1)*100 if base_avg > 0 else 0
        print(f"  {n:2d} adapters: avg={r['avg']:.0f}ms  p50={r['p50']:.0f}ms  p95={r['p95']:.0f}ms  "
              f"loss={r['losses'][0]:.4f}->{r['losses'][-1]:.4f}  ({ovh:+.1f}%)")
    # Save combined results
    with open("/tmp/ep_results.json", "w") as f:
        json.dump(results, f)
    print("\n  Raw data saved to /tmp/ep_loss_*.json")
