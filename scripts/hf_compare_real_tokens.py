#!/usr/bin/env python3
"""Compare HF vs rustrain hidden states — ONLY real tokens (first 22), not padding."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"

def stats_real(t, name, n_real=22):
    """Stats over only the first n_real tokens."""
    tf = t[:, :n_real, :].float()
    print(f"{name}: mean={tf.mean().item():.6f} std={tf.std().item():.6f} max_abs={tf.abs().max().item():.6f}")

def stats_all(t, name):
    tf = t.float()
    print(f"{name}: mean={tf.mean().item():.6f} std={tf.std().item():.6f} max_abs={tf.abs().max().item():.6f}")

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()

    prompt_text = "<|im_start|>user\nWhat is 2+2? <|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    token_ids = prompt_ids + [248068, 271, 248069, 271] + tokenizer.encode("4", add_special_tokens=False) + [248046]
    n_real = len(token_ids)
    print(f"Real tokens: {n_real}")

    # Padded to 512 (same as rustrain)
    padded = token_ids + [248044] * (512 - n_real)
    input_ids = torch.tensor([padded], dtype=torch.long, device="cuda:0")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    print(f"\n=== Layer 0 (embed) ===")
    stats_all(outputs.hidden_states[0], "HF all 512")
    stats_real(outputs.hidden_states[0], "HF real 22", n_real)
    print(f"rustrain all 512: mean=-0.000205 std=0.011426 max_abs=0.255859")

    print(f"\n=== Layer 40 (L39, last) ===")
    stats_all(outputs.hidden_states[-1], "HF all 512")
    stats_real(outputs.hidden_states[-1], "HF real 22", n_real)
    print(f"rustrain all 512: mean=-0.000865 std=0.367881 max_abs=57.500000")

    # Also check a mid layer
    print(f"\n=== Layer 11 (L10) ===")
    stats_all(outputs.hidden_states[11], "HF all 512")
    stats_real(outputs.hidden_states[11], "HF real 22", n_real)
    print(f"rustrain all 512: mean=-0.000326 std=0.066966 max_abs=41.500000")

if __name__ == "__main__":
    main()
