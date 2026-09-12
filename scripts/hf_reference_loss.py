#!/usr/bin/env python3
"""HF reference loss for Qwen3.6-35B-A3B — compare with rustrain C++ kernel.

Loads the model with HuggingFace transformers, runs forward on the same SFT data,
and reports the cross-entropy loss. This should match rustrain's step 0 loss
(when LoRA B=0, so model output is identical to base model).
"""
import torch
import json
import sys
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
    
    # Same SFT data as rustrain: "What is 2+2?" -> "4"
    # Use the same chat template format
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print(f"Chat text: {text[:200]}...")
    
    # Tokenize
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to("cuda:0")
    attention_mask = enc["attention_mask"].to("cuda:0")
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Tokens: {input_ids[0][:20].tolist()}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Compute cross-entropy loss (shifted, like standard LM training)
    # rustrain uses: shifted_logits = logits[:, :-1], shifted_targets = input_ids[:, 1:]
    # loss = -log_softmax(shifted_logits).gather(targets).mean()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    print(f"\n=== HF Reference Loss ===")
    print(f"Loss: {loss.item():.6f}")
    print(f"Logits shape: {logits.shape}")
    print(f"Vocab size: {logits.size(-1)}")
    
    # Also compute per-token loss for response tokens only
    # Find where the assistant response starts
    # For now, just report the full sequence loss
    
    # Also try with the exact same format rustrain uses
    # rustrain uses: instruction + response, with target_mask on response tokens
    # Let's also compute loss on just the response token "4"
    response_start = text.rfind("4")  # Find the last "4" which is the response
    print(f"\n=== Per-token analysis ===")
    for i in range(min(20, input_ids.size(1))):
        token = tokenizer.decode([input_ids[0, i].item()])
        print(f"  token {i}: id={input_ids[0, i].item()}, text='{token}'")

if __name__ == "__main__":
    main()
