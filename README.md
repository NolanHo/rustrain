# rustrain

rustrain is an educational Rust LLM training engine. It starts with a small
Qwen-like toy trainer and incrementally adds real Qwen checkpoint loading,
generation, LoRA/SFT building blocks, Megatron-style parallel module concepts,
MoE toy modules, and CUDA smoke tests through `tch-rs`.

This repository is not yet a production trainer. The current code is organized
as milestone-sized, verifiable smokes.

## Local CPU Verification

Source the local PyTorch/tch environment first:

```sh
source scripts/tch_env.sh
```

Run the default CPU verification set:

```sh
scripts/verify_cpu.sh
```

The script runs:

- `cargo test`
- `cargo run -- train --config configs/debug.toml`
- `cargo run -- train --config configs/qwen3_mini.toml`
- `cargo run -- train --config configs/tch_smoke.toml`
- `cargo run -- train --config configs/text_debug.toml`
- `cargo run -- train --config configs/gsm8k_toy.toml`
- `cargo run -- train --config configs/sft_debug.toml`
- `cargo run -- qwen-lora-smoke`
- `cargo run -- qwen-kv-cache-parity`
- `cargo run -- parallel-dp-smoke --output-dir runs/parallel-dp-smoke`
- `cargo run -- parallel-tp-smoke`
- `cargo run -- parallel-ep-smoke`

## A800 CUDA Verification

On the A800 worker, source the CUDA PyTorch/tch environment:

```sh
source scripts/tch_a800_env.sh
```

Then run:

```sh
cargo run -- tch-cuda-probe
cargo run -- train --config configs/tch_smoke_cuda.toml
```

The A800 config writes runs to `/tmp/rustrain-runs` because the shared project
checkout may be read-only from the worker.

## Real Qwen Smokes

The current real-model path targets the local Qwen2.5 checkpoint:

```sh
source scripts/tch_env.sh
cargo run -- inspect --model-path /vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct
cargo run -- qwen-logits-parity
cargo run -- qwen-generate-parity
cargo run -- qwen-sampling-smoke
cargo run -- qwen-kv-cache-parity
cargo run -- qwen-lora-smoke
```

The full-train smoke command is present, but A800 execution must be used for
the real checkpoint:

```sh
source scripts/tch_a800_env.sh
cargo run -- qwen-full-train-smoke
```

## Current Major Gaps

- Real full-Qwen training is only partially done: the command and representative
  trainable tensor path exist, but the A800 full-checkpoint smoke still needs to
  be run when the worker is reachable.
- Real Qwen module-level LoRA for layer0 attention `q_proj`/`v_proj` is
  implemented; trainer-integrated LoRA SFT is not done yet.
- KV-cache greedy parity is implemented; cached sampling and Python cached
  generation parity are future work.
- Multi-GPU DP/TP/EP are toy smokes, not real NCCL-backed distributed training.
- Real Qwen checkpoint manifest support has started with delta metadata, but
  full optimizer state and distributed checkpoint layout are not done.
- Trainer production basics such as scheduler and grad clipping are implemented
  for local toy/tch paths; memory metrics and real tokenizer-backed batching
  remain partial.

Internal planning details live in `_internal_docs/TODO.md`; that directory is
ignored by git.
