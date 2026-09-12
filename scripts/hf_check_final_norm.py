#!/usr/bin/env python3
"""Check if HF hidden_states[-1] includes final norm or not.

Compare:
1. hidden_states[-1] (what HF reports)
2. model.norm(hidden_states[-1]) (manual final norm)
3. If #1 == #2, then HF includes final norm in hidden_states
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"

def stats(t, name):
    tf = t.float()
    print(f"{name}: mean={tf.mean().item():.6f} std={tf.std().item():.6f} max_abs={tf.abs().max().item():.6f}")

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()

    # Same token sequence as rustrain (padded to 512)
    prompt_text = "<|im_start|>user\nWhat is 2+2? <|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    token_ids = prompt_ids + [248068, 271, 248069, 271] + tokenizer.encode("4", add_special_tokens=False) + [248046]
    token_ids += [248044] * (512 - len(token_ids))  # pad
    input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]  # What HF reports as last hidden state
    
    # Apply final norm manually
    with torch.no_grad():
        normed = model.model.norm(last_hidden)
    
    stats(last_hidden, "HF hidden_states[-1] (pre-norm?)")
    stats(normed, "After model.norm (post-norm)")
    
    # rustrain reported: mean=-0.000865 std=0.367881 max_abs=57.500000
    print("\nrustrain Layer 39: mean=-0.000865 std=0.367881 max_abs=57.500000")
    
    # Check which one matches rustrain
    last_f = last_hidden.float()
    normed_f = normed.float()
    
    # rustrain's value
    rustrain_std = 0.367881
    
    print(f"\nHF pre-norm std:  {last_f.std().item():.6f}  vs rustrain: {rustrain_std:.6f}  match={abs(last_f.std().item() - rustrain_std) < 0.01}")
    print(f"HF post-norm std: {normed_f.std().item():.6f}  vs rustrain: {rustrain_std:.6f}  match={abs(normed_f.std().item() - rustrain_std) < 0.01}")

if __name__ == "__main__":
    main()
