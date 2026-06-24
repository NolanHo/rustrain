#!/usr/bin/env python3
"""Write module-level Qwen parity fixtures for the Rust tch implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_model_path import resolve_qwen_model_path


DEFAULT_MODEL = Path("/path/to/huggingface/Qwen2.5-0.5B-Instruct")
DEFAULT_PROMPT = Path("data/parity/qwen_prompt.txt")


def capture_tensor(captures: dict[str, torch.Tensor], name: str, tuple_index: int | None = None):
    def hook(_module, _inputs, output):
        tensor = output[tuple_index] if tuple_index is not None else output
        captures[name] = tensor.detach()

    return hook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/parity/qwen_layer0_modules.safetensors"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/parity/qwen_layer0_modules_summary.json"),
    )
    args = parser.parse_args()
    model_path = resolve_qwen_model_path(args.model_path)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    layer0 = model.model.layers[0]
    layer1 = model.model.layers[1]
    captures: dict[str, torch.Tensor] = {}
    hooks = [
        model.model.embed_tokens.register_forward_hook(capture_tensor(captures, "hidden")),
        layer0.input_layernorm.register_forward_hook(
            capture_tensor(captures, "attention_normed")
        ),
        layer0.self_attn.register_forward_hook(
            capture_tensor(captures, "attention_output", 0)
        ),
        layer0.register_forward_hook(capture_tensor(captures, "layer0_output")),
        layer1.register_forward_hook(capture_tensor(captures, "layer1_output")),
    ]
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    hidden = captures["hidden"]
    attention_normed = captures["attention_normed"]
    attention_output = captures["attention_output"]
    layer_output = captures["layer0_output"]
    layer1_output = captures["layer1_output"]
    normed = layer0.post_attention_layernorm(hidden)
    mlp_output = layer0.mlp(normed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "input_ids": input_ids.to(torch.int64).cpu().contiguous(),
            "embedded_hidden": hidden.cpu().contiguous(),
            "input_attention_normed": attention_normed.cpu().contiguous(),
            "attention_output": attention_output.cpu().contiguous(),
            "post_attention_normed": normed.cpu().contiguous(),
            "mlp_output": mlp_output.cpu().contiguous(),
            "layer0_output": layer_output.cpu().contiguous(),
            "layer1_output": layer1_output.cpu().contiguous(),
        },
        args.output,
    )

    summary = {
        "model_path": str(model_path),
        "prompt_file": str(args.prompt_file),
        "fixture": str(args.output),
        "input_ids": input_ids[0].tolist(),
        "embedded_hidden_shape": list(hidden.shape),
        "input_attention_normed_shape": list(attention_normed.shape),
        "attention_output_shape": list(attention_output.shape),
        "attention_output_checksum": float(attention_output.float().sum().item()),
        "post_attention_normed_shape": list(normed.shape),
        "mlp_output_shape": list(mlp_output.shape),
        "mlp_output_checksum": float(mlp_output.float().sum().item()),
        "layer0_output_shape": list(layer_output.shape),
        "layer0_output_checksum": float(layer_output.float().sum().item()),
        "layer1_output_shape": list(layer1_output.shape),
        "layer1_output_checksum": float(layer1_output.float().sum().item()),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
