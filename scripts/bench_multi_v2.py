#!/usr/bin/env python3
"""Multi-LoRA benchmark: compare 1 adapter vs 32 adapters."""
import requests, base64, struct, time, sys

SERVER = "http://localhost:8080"
MODEL = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
SEQ = 512

def enc(d):
    return base64.b64encode(struct.pack("<" + "q" * len(d), *d)).decode()

def bench(n_adp, rank, n_steps):
    sid = "b%d" % n_adp
    requests.post(SERVER + "/v1/sessions", json={"session_id": sid})
    
    cfg = ('[run]\nname="%s"\nseed=42\n[model]\nname="%s"\n'
           'architecture="qwen3_6_lora_sft"\n'
           'model_path="%s"\nvocab_size=248320\nhidden_size=1024\n'
           'num_layers=24\nnum_attention_heads=8\nnum_key_value_heads=2\n'
           'intermediate_size=3584\nseq_len=%d\nnorm="rmsnorm"\n'
           'activation="swiglu"\nrope=true\nrms_norm_eps=0.000001\n'
           '[train]\nmax_steps=%d\nbackend="tch"\nmicro_batch_size=1\n'
           'global_batch_size=1\ngradient_accumulation_steps=1\n'
           'learning_rate=0.0001\ndtype="bf16"\ndevice="cuda"\n'
           'checkpoint_every=0\neval_every=0\n'
           '[parallel]\ntensor_model_parallel_size=1\n'
           'pipeline_model_parallel_size=1\ndata_parallel_size=1\n'
           'expert_model_parallel_size=1\ncontext_parallel_size=1\n'
           '[lora]\nrank=8\nalpha=16\ntarget_layers=[]\n'
           'target_modules=["q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","out_proj"]\n'
           '[data]\nkind="instruction_jsonl"\n'
           'paths=["/tmp/qwen3_6_test.jsonl"]\n'
           ) % (sid, sid, MODEL, SEQ, n_steps)
    
    r = requests.post(SERVER + "/v1/sessions/" + sid + "/load_model",
                      json={"model_path": MODEL, "config_toml": cfg})
    if r.status_code != 200:
        print("load FAIL: " + r.text[:100])
        return []
    
    requests.post(SERVER + "/v1/sessions/" + sid + "/load_dataset",
                  json={"jsonl_path": "/tmp/qwen3_6_test.jsonl", "seq_len": SEQ})
    requests.post(SERVER + "/v1/sessions/" + sid + "/init_lora",
                  json={"rank": 8, "alpha": 16, "target_layers": [],
                        "target_modules": ["q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","out_proj"],
                        "lr": 0.0001, "beta1": 0.9, "beta2": 0.999, "eps": 0.00000001})
    
    for i in range(n_adp):
        requests.post(SERVER + "/v1/sessions/" + sid + "/add_lora",
                      json={"rank": rank, "alpha": rank * 2, "target_layers": [], "target_modules": ""})
    
    r = requests.get(SERVER + "/v1/sessions/" + sid + "/list_lora")
    print("adapters: " + str(r.json()))
    
    ids = [1] * SEQ
    mask = [0] * 20 + [1] * (SEQ - 20)
    payload = {
        "input_ids": {"data": enc(ids), "shape": [1, SEQ], "dtype": "int64"},
        "target_mask": {"data": enc(mask), "shape": [1, SEQ], "dtype": "int64"},
        "attention_mask": {"data": enc([1] * SEQ), "shape": [1, SEQ], "dtype": "int64"},
    }
    
    r = requests.post(SERVER + "/v1/sessions/" + sid + "/train_step", json=payload)
    loss = r.json().get("loss", "?")
    print("%d adp warmup: %d loss=%s" % (n_adp, r.status_code, loss))
    
    times = []
    for s in range(n_steps):
        t0 = time.time()
        r = requests.post(SERVER + "/v1/sessions/" + sid + "/train_step", json=payload)
        t1 = time.time()
        if r.status_code != 200:
            print("FAIL: " + r.text[:100])
            break
        times.append((t1 - t0) * 1000)
    return times

print("=== 1 adapter (baseline) ===")
t1 = bench(1, 16, 5)
print()
print("=== 32 adapters x rank=16 ===")
t32 = bench(32, 16, 5)

if t1 and t32:
    a1 = sum(t1) / len(t1)
    a32 = sum(t32) / len(t32)
    print()
    print("=== Summary ===")
    print("1 adapter:   %.1f ms/step" % a1)
    print("32 adapters: %.1f ms/step" % a32)
    print("Overhead: %.1f ms (%.1f%%)" % (a32 - a1, (a32 / a1 - 1) * 100))
