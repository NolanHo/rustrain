#!/usr/bin/env python3
"""Write module-level Qwen parity fixtures for the Rust tch implementation."""

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
    parser.add_argument("--output", type=Path, default=Path("runs/parity/qwen_layer0_mlp.safetensors"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/parity/qwen_layer0_mlp_summary.json"),
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

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    layer0 = model.model.layers[0]
    hidden = model.model.embed_tokens(input_ids)
    normed = layer0.post_attention_layernorm(hidden)
    mlp_output = layer0.mlp(normed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "input_ids": input_ids.to(torch.int64).cpu().contiguous(),
            "embedded_hidden": hidden.cpu().contiguous(),
            "post_attention_normed": normed.cpu().contiguous(),
            "mlp_output": mlp_output.cpu().contiguous(),
        },
        args.output,
    )

    summary = {
        "model_path": str(args.model_path),
        "prompt_file": str(args.prompt_file),
        "fixture": str(args.output),
        "input_ids": input_ids[0].tolist(),
        "embedded_hidden_shape": list(hidden.shape),
        "post_attention_normed_shape": list(normed.shape),
        "mlp_output_shape": list(mlp_output.shape),
        "mlp_output_checksum": float(mlp_output.float().sum().item()),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
