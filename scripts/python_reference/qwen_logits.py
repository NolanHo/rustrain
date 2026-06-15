#!/usr/bin/env python3
"""Emit a deterministic logits fixture from a local Qwen-family HF checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = Path("/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct")
DEFAULT_PROMPT = Path("data/parity/qwen_prompt.txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=Path("runs/parity/qwen_logits.pt"))
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    logits = model(**inputs).logits.cpu().contiguous()
    last_logits = logits[0, -1]
    values, indices = torch.topk(last_logits, k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_path": str(args.model_path),
            "prompt": prompt,
            "input_ids": inputs["input_ids"].cpu(),
            "logits": logits,
        },
        args.output,
    )

    summary = {
        "model_path": str(args.model_path),
        "prompt_file": str(args.prompt_file),
        "prompt": prompt,
        "input_ids": inputs["input_ids"][0].tolist(),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "last_token_topk": [
            {"token_id": int(token_id), "logit": float(value)}
            for token_id, value in zip(indices.tolist(), values.tolist())
        ],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
