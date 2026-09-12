#!/usr/bin/env python3
"""HF vs rustrain precision comparison for Qwen3.6-35B-A3B.

Strategy:
1. Use the EXACT same token sequence as rustrain (with think/answer tokens)
2. Run HF forward pass, get logits
3. Compute loss the SAME way rustrain does (response-only, masked CE)
4. Compare with rustrain's reported loss

This isolates forward-pass correctness from loss computation differences.
"""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    print("Loading model (bf16, single GPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    
    # ── Build the EXACT same token sequence as rustrain ──
    # rustrain format (from sft.rs encode):
    #   <|im_start|>user\n{instruction} {input}<|im_end|>\n<|im_start|>assistant\n
    #   + think(248068) + newline(271) + answer(248069) + newline(271)
    #   + response_tokens + eos(248046)
    
    instruction = "What is 2+2?"
    response = "4"
    
    # Prompt part (same as rustrain)
    prompt_text = f"<|im_start|>user\n{instruction} <|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # think + \n\n + answer + \n\n (rustrain inserts these as raw token IDs)
    think_id = 248068
    answer_id = 248069
    newline_id = 271  # \n\n
    
    # Response tokens
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    
    # EOS
    eos_id = 248046  # <|im_end|>
    
    # Full sequence (same as rustrain)
    token_ids = prompt_ids + [think_id, newline_id, answer_id, newline_id] + response_ids + [eos_id]
    
    # Target mask (same as rustrain): 0 for prompt+think+\n\n, 1 for answer+\n\n+response+eos
    target_mask = [0.0] * len(prompt_ids) + [0.0, 0.0, 1.0, 1.0] + [1.0] * len(response_ids) + [1.0]
    
    print(f"Token sequence ({len(token_ids)} tokens):")
    for i, (tid, mask) in enumerate(zip(token_ids, target_mask)):
        decoded = tokenizer.decode([tid])
        print(f"  {i:3d}: id={tid:6d} mask={mask:.0f} text={decoded!r}")
    
    input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
    
    # ── HF forward pass ──
    print("\nRunning HF forward pass...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    print(f"Logits shape: {logits.shape}")
    
    # ── Compute loss the SAME way rustrain does ──
    # rustrain (from qwen3_6_kernels.cpp compute_loss):
    #   shifted_logits = logits[:, :-1, :].reshape(-1, vocab)
    #   shifted_targets = input_ids[:, 1:].reshape(-1)
    #   shifted_mask = target_mask[1:].reshape(-1)
    #   log_probs = log_softmax(shifted_logits)
    #   per_token_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    #   masked = per_token_loss * shifted_mask
    #   loss = masked.sum() / shifted_mask.sum()
    
    seq_len = logits.size(1)
    vocab_size = logits.size(-1)
    
    shifted_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    shifted_targets = input_ids[:, 1:].reshape(-1)
    shifted_mask = torch.tensor(target_mask[1:], dtype=torch.float32, device="cuda:0")
    
    log_probs = torch.log_softmax(shifted_logits.float(), dim=-1)
    per_token_loss = -log_probs.gather(1, shifted_targets.unsqueeze(1)).squeeze(1)
    masked = per_token_loss * shifted_mask
    loss = masked.sum() / shifted_mask.sum().clamp_min(1.0)
    
    print(f"\n=== HF Reference (rustrain-compatible loss) ===")
    print(f"Loss: {loss.item():.6f}")
    print(f"Response tokens: {int(shifted_mask.sum().item())}")
    print(f"Total tokens: {len(shifted_targets)}")
    
    # ── Also compute HF standard loss (all tokens) for reference ──
    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
    hf_loss = loss_fct(shifted_logits, shifted_targets)
    print(f"\n=== HF standard loss (all tokens) ===")
    print(f"Loss: {hf_loss.item():.6f}")
    
    # ── Per-token loss breakdown ──
    print(f"\n=== Per-token loss (response tokens only) ===")
    for i in range(len(shifted_targets)):
        if shifted_mask[i] > 0:
            tid = shifted_targets[i].item()
            decoded = tokenizer.decode([tid])
            tl = per_token_loss[i].item()
            print(f"  pos {i}: target={tid} ({decoded!r}) loss={tl:.4f}")

if __name__ == "__main__":
    main()
