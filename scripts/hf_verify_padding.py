#!/usr/bin/env python3
"""Compare HF with/without padding to verify if attention_mask is the issue.

If rustrain matches HF-no-padding but not HF-with-padding,
then the issue is rustrain not applying attention_mask.
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

    prompt_text = "<|im_start|>user\nWhat is 2+2? <|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    token_ids = prompt_ids + [248068, 271, 248069, 271] + tokenizer.encode("4", add_special_tokens=False) + [248046]

    # Run WITHOUT padding (22 tokens, no attention mask needed)
    input_ids_nopad = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
    with torch.no_grad():
        out_nopad = model(input_ids=input_ids_nopad, output_hidden_states=True)
    
    # Run WITH padding (512 tokens, with attention_mask)
    pad_id = 248044
    padded = token_ids + [pad_id] * (512 - len(token_ids))
    input_ids_pad = torch.tensor([padded], dtype=torch.long, device="cuda:0")
    attn_mask = torch.zeros(1, 512, dtype=torch.long, device="cuda:0")
    attn_mask[0, :len(token_ids)] = 1  # Only attend to real tokens
    with torch.no_grad():
        out_pad_masked = model(input_ids=input_ids_pad, attention_mask=attn_mask, output_hidden_states=True)
    
    # Run WITH padding but NO attention_mask (simulates rustrain behavior)
    with torch.no_grad():
        out_pad_nomask = model(input_ids=input_ids_pad, output_hidden_states=True)

    # Compare last layer
    print("=== Last Layer (L39) comparison ===")
    stats(out_nopad.hidden_states[-1], "HF no-pad (22 tok)  ")
    stats(out_pad_masked.hidden_states[-1], "HF pad+mask (512)   ")
    stats(out_pad_nomask.hidden_states[-1], "HF pad+nomask (512)")
    print("rustrain Layer 39:                 mean=-0.000865 std=0.367881 max_abs=57.500000")
    
    # Compare a few earlier layers too
    print("\n=== Layer 10 (L9) comparison ===")
    stats(out_nopad.hidden_states[10], "HF no-pad (22 tok)  ")
    stats(out_pad_masked.hidden_states[10], "HF pad+mask (512)   ")
    stats(out_pad_nomask.hidden_states[10], "HF pad+nomask (512) ")
    print("rustrain Layer 9:  mean=-0.000380 std=0.042763 max_abs=7.968750")

    # Compute loss for each variant
    target_mask = [0.0] * len(prompt_ids) + [0.0, 0.0, 1.0, 1.0] + [1.0, 1.0]
    
    for name, out, ids in [("no-pad", out_nopad, input_ids_nopad),
                            ("pad+mask", out_pad_masked, input_ids_pad),
                            ("pad+nomask", out_pad_nomask, input_ids_pad)]:
        logits = out.logits.float()
        seq = logits.size(1)
        tm = target_mask[:seq] + [0.0] * (seq - len(target_mask))
        sl = logits[:, :-1, :].reshape(-1, logits.size(-1))
        st = ids[:, 1:].reshape(-1)
        sm = torch.tensor(tm[1:], dtype=torch.float32, device="cuda:0")
        lp = torch.log_softmax(sl, dim=-1)
        pl = -lp.gather(1, st.unsqueeze(1)).squeeze(1)
        loss = (pl * sm).sum() / sm.sum().clamp_min(1.0)
        print(f"\nLoss ({name}): {loss.item():.6f}")

if __name__ == "__main__":
    main()
