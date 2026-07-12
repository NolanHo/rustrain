#!/usr/bin/env python3
"""Multi-LoRA benchmark: 1 vs 2 vs 4 vs 8 vs 16 vs 32 adapters."""
import requests, base64, struct, time, sys

SERVER = "http://localhost:8080"
MODEL = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
SEQ = 512

def enc(d):
    return base64.b64encode(struct.pack("<" + "q"*len(d), *d)).decode()

def bench(n_adp, rank, n_steps):
    sid = "z%d" % n_adp
    requests.post(SERVER + "/v1/sessions", json={"session_id": sid})
    cfg_lines = [
        '[run]', 'name="%s"' % sid, 'seed=42',
        '[model]', 'name="%s"' % sid, 'architecture="qwen3_6_lora_sft"',
        'model_path="%s"' % MODEL, 'vocab_size=248320',
        'hidden_size=1024', 'num_layers=24', 'num_attention_heads=8',
        'num_key_value_heads=2', 'intermediate_size=3584',
        'seq_len=%d' % SEQ, 'norm="rmsnorm"', 'activation="swiglu"',
        'rope=true', 'rms_norm_eps=0.000001',
        '[train]', 'max_steps=%d' % n_steps, 'backend="tch"',
        'micro_batch_size=1', 'global_batch_size=1',
        'gradient_accumulation_steps=1', 'learning_rate=0.0001',
        'dtype="bf16"', 'device="cuda"',
        'checkpoint_every=0', 'eval_every=0',
        '[parallel]', 'tensor_model_parallel_size=1',
        'pipeline_model_parallel_size=1', 'data_parallel_size=1',
        'expert_model_parallel_size=1', 'context_parallel_size=1',
        '[lora]', 'rank=8', 'alpha=16', 'target_layers=[]',
        'target_modules=["q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","out_proj"]',
        '[data]', 'kind="instruction_jsonl"',
        'paths=["/tmp/qwen3_6_test.jsonl"]',
    ]
    cfg = "\n".join(cfg_lines)

    r = requests.post(SERVER + "/v1/sessions/" + sid + "/load_model",
                      json={"model_path": MODEL, "config_toml": cfg})
    if r.status_code != 200:
        print("%d load FAIL: %s" % (n_adp, r.text[:100]))
        return None
    requests.post(SERVER + "/v1/sessions/" + sid + "/load_dataset",
                  json={"jsonl_path": "/tmp/qwen3_6_test.jsonl", "seq_len": SEQ})
    requests.post(SERVER + "/v1/sessions/" + sid + "/init_lora",
                  json={"rank": 8, "alpha": 16, "target_layers": [],
                        "target_modules": ["q_proj","k_proj","v_proj","o_proj",
                                           "in_proj_qkv","in_proj_z","out_proj"],
                        "lr": 0.0001, "beta1": 0.9, "beta2": 0.999, "eps": 0.00000001})
    for i in range(n_adp):
        r = requests.post(SERVER + "/v1/sessions/" + sid + "/add_lora",
                          json={"rank": rank, "alpha": rank*2,
                                "target_layers": [], "target_modules": ""})
        if r.status_code != 200:
            print("%d add_lora %d FAIL: %s" % (n_adp, i, r.text[:100]))
            return None

    ids = [1]*SEQ
    mask = [0]*20 + [1]*(SEQ-20)
    payload = {
        "input_ids": {"data": enc(ids), "shape": [1,SEQ], "dtype": "int64"},
        "target_mask": {"data": enc(mask), "shape": [1,SEQ], "dtype": "int64"},
        "attention_mask": {"data": enc([1]*SEQ), "shape": [1,SEQ], "dtype": "int64"},
    }
    r = requests.post(SERVER + "/v1/sessions/" + sid + "/train_step", json=payload)
    if r.status_code != 200:
        print("%d warmup FAIL: %s" % (n_adp, r.text[:100]))
        return None
    loss = r.json().get("loss", "?")

    times = []
    for s in range(n_steps):
        t0 = time.time()
        r = requests.post(SERVER + "/v1/sessions/" + sid + "/train_step", json=payload)
        t1 = time.time()
        if r.status_code != 200:
            print("%d step %d FAIL: %s" % (n_adp, s, r.text[:100]))
            break
        times.append((t1-t0)*1000)

    if times:
        avg = sum(times)/len(times)
        print("%d adp: loss=%s avg=%.1fms min=%.1fms max=%.1fms" %
              (n_adp, str(loss)[:8], avg, min(times), max(times)))
        return avg
    return None

def cleanup(sid):
    requests.delete(SERVER + "/v1/sessions/" + sid)

results = {}
for n in [1, 2, 4, 8, 16, 32]:
    t = bench(n, 16, 5)
    if t:
        results[n] = t
    else:
        print("Stopping at %d adapters" % n)
    cleanup("z%d" % n)  # Free GPU memory after each test

if results:
    print("\n=== Summary (rank=16, Qwen3.5-0.8B, seq=512, H20-3e) ===")
    base = results.get(1, 0)
    for n, t in sorted(results.items()):
        ovh = (t/base-1)*100 if base > 0 else 0
        print("  %d adapters: %.1f ms/step (%.1f%% overhead)" % (n, t, ovh))
