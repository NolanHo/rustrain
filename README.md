# rustrain

rustrain is an educational Rust LLM training engine. It starts with a small
Qwen-like toy trainer and incrementally adds real Qwen checkpoint loading,
generation, LoRA/SFT building blocks, Megatron-style parallel module concepts,
MoE toy modules, and CUDA smoke tests through `tch-rs`.

This repository is not yet a production trainer. The current code is organized
as milestone-sized, verifiable smokes.

## GPU-Only Verification

All smoke, test, parity, and training verification runs on the GPU host through
Ray `num_gpus=1` workers. The local machine is only for editing and git
operations; local CPU runs are not accepted as development smoke or milestone
evidence.

Run one command on a Ray GPU worker:

```sh
scripts/gpu_run.sh cargo run -- tch-cuda-probe
```

Run the focused verification suite:

```sh
scripts/verify_gpu.sh
```

Both scripts SSH to `root@192.168.42.106:2222`, submit work to Ray with
`num_gpus=1`, enter the remote checkout, and source `scripts/tch_a800_env.sh`
before running project commands. Do not run `cargo test`, train smokes, or
parity commands directly in the plain SSH shell; that shell does not expose the
GPU devices.

The required remote checkout is:

```sh
/vePFS-Mindverse/user/nolanho/code/rustrain
```

Current bootstrap fallback while the shared checkout is being prepared:

```sh
tar --exclude .git --exclude target --exclude runs -cf - . \
  | ssh -p 2222 root@192.168.42.106 'rm -rf /root/rustrain && mkdir -p /root/rustrain && cd /root/rustrain && tar -xf -'
RUSTRAIN_REMOTE_DIR=/root/rustrain scripts/verify_gpu.sh
```

The A800 config writes runs to `/tmp/rustrain-runs` because the shared project
checkout may be read-only from the worker. If `tch-cuda-probe` reports
`device_count: 0`, the SSH target has not exposed GPU devices and milestone
acceptance is blocked until the runtime is fixed.

## Real Qwen Smokes

The current real-model path targets the Qwen2.5 checkpoint under
`/vePFS-Mindverse/share/huggingface` on the GPU host:

```sh
scripts/gpu_run.sh cargo run -- qwen-logits-parity
```

The full-train smoke command is present, but GPU execution must be used for the
real checkpoint:

```sh
scripts/gpu_run.sh cargo run -- qwen-full-train-smoke
```

## Current Major Gaps

- Real full-Qwen training is partially reusable: representative Qwen parameters
  now go through a trainable registry, but this is not yet a trainer-owned full
  model module.
- Real Qwen module-level LoRA now uses a target-layer/module registry for layer0
  attention `q_proj`/`v_proj`; trainer-owned full-model LoRA injection is not
  done yet.
- KV-cache greedy parity and cached sampling parity are implemented; Python
  cached-generation parity is future work.
- Multi-GPU DP/TP/EP are toy smokes, not real NCCL-backed distributed training.
- Real Qwen checkpoint manifest support has delta metadata and representative
  Adam slot persistence, but full train resume and distributed checkpoint layout
  are not done.
- Trainer production basics such as scheduler, grad clipping, RSS memory
  metrics, and Ray-worker GPU memory reporting are implemented for toy/tch
  paths; real tokenizer-backed padded LoRA SFT batching exists, and the tiny
  `tch` CUDA path has an explicit bf16 compute policy smoke.

Internal planning details live in `_internal_docs/TODO.md`; that directory is
ignored by git.
