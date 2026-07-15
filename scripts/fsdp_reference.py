#!/usr/bin/env python3
"""FSDP reference benchmark for multi-LoRA training.

Loads Qwen3.6-35B-A3B with FSDP, applies rank-1 LoRA to attention modules,
trains N adapters simultaneously (batch dimension), measures throughput.

Usage: torchrun --nproc_per_node=8 scripts/fsdp_reference.py \
         --n_adapters 10 --seq_len 512 --duration 60
"""
import argparse, os, time, json
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from transformers import AutoModelForCausalLM, AutoConfig

MODEL_PATH = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"

def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

def create_lora_params(model, rank, num_adapters, device, layer_types):
    """Create N sets of LoRA A/B params for attention modules.
    Matches rustrain: A=randn*0.01, B=zeros, scaling=alpha/rank=2.0
    full_attention: q_proj, k_proj, v_proj, o_proj (4 modules)
    linear_attention: in_proj_qkvz, out_proj (2 modules)"""
    torch.manual_seed(42)  # match rustrain seed
    lora_params = []
    target_modules_full = ["q_proj", "k_proj", "v_proj", "o_proj"]
    target_modules_linear = ["in_proj_qkvz", "out_proj"]
    
    for adapter_idx in range(num_adapters):
        params_A = {}
        params_B = {}
        for layer_idx, layer in enumerate(model.model.layers):
            ltype = layer_types[layer_idx]
            if ltype == "full_attention":
                attn = layer.self_attn
                target = target_modules_full
            else:
                attn = layer.linear_attn
                target = target_modules_linear
            for mod_name in target:
                mod = getattr(attn, mod_name, None)
                if mod is None:
                    continue
                out_dim, in_dim = mod.weight.shape
                # A: [rank, in_dim], B: [out_dim, rank] — match rustrain init
                A = torch.nn.Parameter(torch.randn(rank, in_dim, dtype=torch.bfloat16, device=device) * 0.01)
                B = torch.nn.Parameter(torch.zeros(out_dim, rank, dtype=torch.bfloat16, device=device))
                A.requires_grad_(True)
                B.requires_grad_(True)
                params_A[(layer_idx, mod_name)] = A
                params_B[(layer_idx, mod_name)] = B
        lora_params.append((params_A, params_B))
    
    return lora_params

def apply_lora_forward(model, lora_params, input_ids, adapter_idx):
    """Forward with LoRA delta: output += B @ (A @ x) * scaling"""
    params_A, params_B = lora_params[adapter_idx]
    hooks = []
    def make_hook(A, B, scale):
        def hook(module, input, output):
            x = input[0]  # [batch, seq, in_dim]
            delta = (x @ A.t()) @ B.t() * scale
            return output + delta
        return hook
    
    for (layer_idx, mod_name), A in params_A.items():
        B = params_B[(layer_idx, mod_name)]
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn if hasattr(layer, 'self_attn') else layer.linear_attn
        mod = getattr(attn, mod_name)
        h = mod.register_forward_hook(make_hook(A, B, 2.0))  # alpha=2
        hooks.append(h)
    
    # attention_mask: all 1s (same as rustrain ep-bench)
    attn_mask = torch.ones(1, 1, input_ids.size(1), input_ids.size(1), dtype=torch.bool, device=input_ids.device)
    output = model(input_ids, attention_mask=attn_mask)
    for h in hooks:
        h.remove()
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_adapters", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    rank, world_size = setup_distributed()
    device = f"cuda:{rank}"
    
    if rank == 0:
        print(f"[fsdp_ref] Loading model from {MODEL_PATH}...")
    
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    text_config = config.text_config if hasattr(config, 'text_config') else config
    layer_types = text_config.layer_types
    
    # Load model with bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
    )
    model = model.to(device)
    
    # Freeze base model
    for p in model.parameters():
        p.requires_grad_(False)
    
    # Create LoRA params BEFORE FSDP (weight shapes are correct now)
    if rank == 0:
        print(f"[fsdp_ref] Creating {args.n_adapters} LoRA adapters...")
    lora_params = create_lora_params(
        model, rank=1, num_adapters=args.n_adapters, device=device, layer_types=layer_types
    )
    
    # Wrap with FSDP
    fsdp_config = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=fsdp_config,
        device_id=rank,
        use_orig_params=True,
    )
    model.eval()  # base model in eval mode (no dropout)
    
    # Adam optimizer for all LoRA params
    all_params = []
    for pa, pb in lora_params:
        all_params.extend(list(pa.values()))
        all_params.extend(list(pb.values()))
    optimizer = torch.optim.Adam(all_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    
    # Build input: same as ep-bench
    input_ids = torch.ones(args.n_adapters, args.seq_len, dtype=torch.long, device=device)
    # target_mask: first 20 tokens are prompt, rest are response
    target_mask = torch.zeros(args.n_adapters, args.seq_len, dtype=torch.long, device=device)
    target_mask[:, 20:] = 1
    
    if rank == 0:
        print(f"[fsdp_ref] Starting benchmark: N={args.n_adapters}, seq={args.seq_len}, duration={args.duration}s")
    
    # Warmup
    t0 = time.time()
    optimizer.zero_grad()
    
    # Process each adapter with its own LoRA
    total_loss = 0.0
    for i in range(args.n_adapters):
        # Forward: base model frozen, LoRA hooks add delta with grad tracking
        with torch.enable_grad():
            output = apply_lora_forward(model, lora_params, input_ids[i:i+1], i)
        logits = output.logits[:, :-1, :]  # [1, seq-1, vocab]
        targets = input_ids[i:i+1, 1:]     # [1, seq-1]
        mask = target_mask[i:i+1, 1:].float()
        
        # Diagnostic: print logits stats on step 1
        if rank == 0 and i == 0:
            print(f"[diag] logits shape: {logits.shape}, dtype: {logits.dtype}")
            print(f"[diag] logits[0,0,:5]: {logits[0,0,:5].float().tolist()}")
            print(f"[diag] logits mean: {logits.float().mean().item():.6f}")
            print(f"[diag] logits std: {logits.float().std().item():.6f}")
            print(f"[diag] target token: {targets[0,0].item()}")
            print(f"[diag] target logit: {logits[0,0,targets[0,0].item()].float().item():.6f}")
            # log_softmax for first token
            lsm = torch.log_softmax(logits[0,0].float(), dim=-1)
            print(f"[diag] log_softmax[target]: {lsm[targets[0,0].item()].item():.6f}")
            print(f"[diag] -log_softmax[target] (per-token loss): {-lsm[targets[0,0].item()].item():.6f}")
        
        loss_per_token = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none"
        ).reshape(1, -1)
        loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss / args.n_adapters  # scale for accumulation
        loss.backward()
        total_loss += loss.item()
    
    optimizer.step()
    if rank == 0:
        print(f"[fsdp_ref] Warmup: loss={total_loss:.6f} time={time.time()-t0:.1f}s")
    
    # Throughput measurement
    total_adapters = 0
    total_steps = 0
    losses = []
    start = time.time()
    
    while time.time() - start < args.duration:
        t0 = time.time()
        optimizer.zero_grad()
        total_loss = 0.0
        
        for i in range(args.n_adapters):
            with torch.enable_grad():
                output = apply_lora_forward(model, lora_params, input_ids[i:i+1], i)
            logits = output.logits[:, :-1, :]
            targets = input_ids[i:i+1, 1:]
            mask = target_mask[i:i+1, 1:].float()
            
            loss_per_token = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none"
            ).reshape(1, -1)
            loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss / args.n_adapters
            loss.backward()
            total_loss += loss.item()
        
        optimizer.step()
        total_adapters += args.n_adapters
        total_steps += 1
        losses.append(total_loss)
        
        if rank == 0 and (total_steps % 5 == 0 or total_steps == 1):
            elapsed = time.time() - start
            rate = total_adapters / elapsed
            print(f"  step {total_steps}: loss={total_loss:.6f} time={time.time()-t0:.1f}s "
                  f"total={total_adapters} adp in {elapsed:.0f}s = {rate:.2f} adp/s")
    
    elapsed = time.time() - start
    rate = total_adapters / elapsed if elapsed > 0 else 0
    
    if rank == 0:
        print()
        print("=" * 60)
        print(f"  FSDP REFERENCE RESULTS: {args.n_adapters} adapters, rank=1, EP={world_size}")
        print("=" * 60)
        print(f"  Duration: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"  Steps: {total_steps}")
        print(f"  Total adapters processed: {total_adapters}")
        print(f"  Throughput: {rate:.2f} adapters/s ({rate*60:.0f} adapters/min)")
        if losses:
            print(f"  Loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
