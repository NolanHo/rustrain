#!/usr/bin/env python3
"""Throughput benchmark: measure adapters/s over 5min (smoke) or 1h (sustained).
Usage:
  python3 qbench_throughput.py --n-adapters 100 --duration 300   # 5min smoke
  python3 qbench_throughput.py --n-adapters 4000 --duration 3600  # 1h sustained
"""
import requests, base64, struct, time, json, sys, argparse

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
        "[train]", "max_steps=100000", 'backend="tch"',
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

def setup(sid, n_adp, lora_rank=1):
    requests.post(f"{SERVER}/v1/sessions", json={"session_id": sid})
    cfg = make_cfg(sid)
    r = requests.post(f"{SERVER}/v1/sessions/{sid}/load_model",
                      json={"model_path": MODEL, "config_toml": cfg}, timeout=600)
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
    # Batch add all adapters in one request
    r = requests.post(f"{SERVER}/v1/sessions/{sid}/batch_add_lora",
                      json={"count": n_adp, "rank": lora_rank, "alpha": lora_rank*2,
                            "target_layers": [], "target_modules": ""})
    if r.status_code != 200:
        print(f"  batch_add_lora FAIL: {r.text[:300]}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-adapters", type=int, default=100)
    parser.add_argument("--duration", type=int, default=300, help="seconds (0 = single step)")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()

    sid = f"throughput_{args.n_adapters}_{args.rank}"
    print(f"Setting up {args.n_adapters} adapters (rank={args.rank})...")
    if not setup(sid, args.napters if hasattr(args,'napters') else args.n_adapters, args.rank):
        print("SETUP FAILED")
        return
    print("Setup OK")

    ids = [1]*SEQ
    mask = [0]*20 + [1]*(SEQ-20)
    payload = {
        "input_ids": tensor(ids),
        "target_mask": tensor(mask),
        "attention_mask": tensor([1]*SEQ),
        "n_total": args.n_adapters,
        "lora_rank": args.rank,
    }

    # Warmup
    print("Warmup...")
    t0 = time.time()
    r = requests.post(f"{SERVER}/v1/sessions/{sid}/train_multi", json=payload, timeout=600)
    t1 = time.time()
    if r.status_code != 200:
        print(f"Warmup FAIL: {r.text[:300]}")
        requests.delete(f"{SERVER}/v1/sessions/{sid}")
        return
    warmup_loss = r.json().get("loss", -1)
    print(f"Warmup: loss={warmup_loss:.6f} time={1000*(t1-t0):.0f}ms")

    if args.duration == 0:
        # Single step
        requests.delete(f"{SERVER}/v1/sessions/{sid}")
        return

    # Throughput measurement
    print(f"\nMeasuring throughput for {args.duration}s...")
    total_adapters = 0
    total_steps = 0
    losses = []
    start = time.time()
    while time.time() - start < args.duration:
        t0 = time.time()
        r = requests.post(f"{SERVER}/v1/sessions/{sid}/train_multi", json=payload, timeout=600)
        t1 = time.time()
        if r.status_code != 200:
            print(f"Step {total_steps} FAIL: {r.text[:200]}")
            break
        loss = r.json().get("loss", -1)
        losses.append(loss)
        total_adapters += args.n_adapters
        total_steps += 1
        if total_steps % 5 == 0 or total_steps == 1:
            elapsed = t1 - start
            rate = total_adapters / elapsed
            print(f"  step {total_steps}: loss={loss:.6f} step_time={1000*(t1-t0):.0f}ms "
                  f"total={total_adapters} adapters in {elapsed:.0f}s = {rate:.1f} adp/s")
        sys.stdout.flush()

    elapsed = time.time() - start
    rate = total_adapters / elapsed if elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"  RESULTS: {args.n_adapters} adapters, rank={args.rank}, EP={args.world_size}")
    print(f"{'='*60}")
    print(f"  Duration: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Steps: {total_steps}")
    print(f"  Total adapters processed: {total_adapters}")
    print(f"  Throughput: {rate:.2f} adapters/s ({rate*60:.0f} adapters/min)")
    if losses:
        print(f"  Loss: {losses[0]:.6f} → {losses[-1]:.6f}")
    print(f"  Failures: {total_steps - len(losses)}")

    # Save results
    with open(f"/tmp/throughput_{args.n_adapters}_{args.rank}.json", "w") as f:
        json.dump({"n_adapters": args.n_adapters, "rank": args.rank,
                   "duration": elapsed, "steps": total_steps,
                   "total_adapters": total_adapters, "throughput": rate,
                   "losses": losses}, f)

    requests.delete(f"{SERVER}/v1/sessions/{sid}")

if __name__ == "__main__":
    main()
