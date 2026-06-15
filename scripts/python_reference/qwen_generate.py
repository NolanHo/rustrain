#!/usr/bin/env python3
"""Emit a deterministic greedy generation fixture from a local Qwen checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = Path("/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct")
DEFAULT_PROMPT = Path("data/parity/qwen_prompt.txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/parity/qwen2_5_0_5b_generate.safetensors"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/parity/qwen2_5_0_5b_generate_summary.json"),
    )
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
    generated = inputs["input_ids"]
    for _ in range(args.max_new_tokens):
        logits = model(input_ids=generated, use_cache=False).logits
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    generated = generated.cpu().contiguous()
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "input_ids": inputs["input_ids"].cpu().contiguous(),
            "generated_ids": generated,
        },
        args.output,
    )

    summary = {
        "model_path": str(args.model_path),
        "prompt_file": str(args.prompt_file),
        "prompt": prompt,
        "input_ids": inputs["input_ids"][0].tolist(),
        "generated_ids": generated[0].tolist(),
        "new_token_ids": generated[0, inputs["input_ids"].shape[1] :].tolist(),
        "max_new_tokens": args.max_new_tokens,
        "generated_text": generated_text,
        "output": str(args.output),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
