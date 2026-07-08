#!/usr/bin/env python3
"""HF layer-by-layer hidden states dump for Qwen3.6-35B-A3B.

Dumps mean/std/max of hidden states after each layer for comparison with rustrain.
Uses the EXACT same token sequence as rustrain (with think/answer tokens).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    # Exact same token sequence as rustrain
    instruction = "What is 2+2?"
    response = "4"
    prompt_text = f"<|im_start|>user\n{instruction} <|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    token_ids = prompt_ids + [248068, 271, 248069, 271] + tokenizer.encode(response, add_special_tokens=False) + [248046]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
    
    print(f"Token sequence ({len(token_ids)} tokens): {token_ids}")
    
    # Forward with hidden states
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    
    # hidden_states[0] = embedding output, hidden_states[i] = after layer i-1
    print(f"\n=== HF Hidden States (per layer) ===")
    print(f"Layer  0 (embed): mean={outputs.hidden_states[0].float().mean().item():.6f} std={outputs.hidden_states[0].float().std().item():.6f} max_abs={outputs.hidden_states[0].float().abs().max().item():.6f}")
    for i in range(1, len(outputs.hidden_states)):
        h = outputs.hidden_states[i].float()
        print(f"Layer {i:2d} (L{i-1}):  mean={h.mean().item():.6f} std={h.std().item():.6f} max_abs={h.abs().max().item():.6f}")
    
    # Final logits
    logits = outputs.logits.float()
    print(f"\nLogits: mean={logits.mean().item():.6f} std={logits.std().item():.6f} max_abs={logits.abs().max().item():.6f}")
    
    # Compute loss (same as rustrain: response-only masked CE)
    target_mask = [0.0] * len(prompt_ids) + [0.0, 0.0, 1.0, 1.0] + [1.0] * len(tokenizer.encode(response, add_special_tokens=False)) + [1.0]
    shifted_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shifted_targets = input_ids[:, 1:].reshape(-1)
    shifted_mask = torch.tensor(target_mask[1:], dtype=torch.float32, device="cuda:0")
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    per_token_loss = -log_probs.gather(1, shifted_targets.unsqueeze(1)).squeeze(1)
    masked = per_token_loss * shifted_mask
    loss = masked.sum() / shifted_mask.sum().clamp_min(1.0)
    print(f"\nLoss (response-only): {loss.item():.6f}")

    # Save hidden states for comparison
    torch.save({"hidden_states": [h.cpu() for h in outputs.hidden_states],
                "logits": logits.cpu(),
                "token_ids": token_ids}, "/tmp/hf_hidden_states.pt")
    print("\nSaved hidden states to /tmp/hf_hidden_states.pt")

if __name__ == "__main__":
    main()
