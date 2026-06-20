# rustrain

rustrain is an educational Rust LLM training engine. It starts with a small
Qwen-like toy trainer and incrementally adds real Qwen checkpoint loading,
generation, LoRA/SFT building blocks, Megatron-style parallel module concepts,
MoE toy modules, and CUDA smoke tests through `tch-rs`.

This repository is not yet a production trainer. The current code is organized
as milestone-sized, verifiable smokes.

## GPU-Only Verification

All smoke, test, parity, and training verification runs on the GPU host through
Ray `num_gpus=1` workers. The local machine is only for editing, git
operations, and launching remote jobs. Local CPU smoke is forbidden: do not use
it for development feedback, debugging evidence, or final acceptance.

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
before running project commands. Do not run `cargo test`, train smokes, parity
commands, or quick local CPU checks on the local machine. Do not run them
directly in the plain SSH shell either; that shell does not expose the GPU
devices. If a check is worth running, run it on a Ray GPU worker.

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

## Distributed Launch

A minimal local-rank launcher is available for rank process management:

```sh
scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 print-launch-env
scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 parallel-dp-rank-smoke --output-dir /tmp/rustrain-runs/launch-dp-rank-smoke/dp
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/nccl-all-reduce-smoke nccl-all-reduce-rank-smoke --output-dir /tmp/rustrain-runs/nccl-all-reduce-smoke/ranks
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/nccl-dp-gradient-smoke nccl-dp-gradient-rank-smoke --output-dir /tmp/rustrain-runs/nccl-dp-gradient-smoke/ranks
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/tch-dp-gradient-smoke tch-dp-gradient-rank-smoke --output-dir /tmp/rustrain-runs/tch-dp-gradient-smoke/ranks
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/tch-trainer-dp2-launch train --config configs/tch_smoke_cuda_dp2.toml
```

The launcher sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`,
`MASTER_ADDR`, and `MASTER_PORT`, captures rank-local logs, and writes a
`launch-summary.json`. `scripts/gpu_run.sh` defaults to one Ray GPU; set
`RUSTRAIN_RAY_NUM_GPUS=2` for two-rank GPU collective smokes. Minimal NCCL f32
all-reduce, toy DP gradient all-reduce, and `tch` autograd DP gradient smokes
exist, including a `trainer::train` config path for tiny `tch` DP=2. Real
multi-GPU Qwen training is not implemented yet.

## Current Major Gaps

- G5 representative checkpoint resume is implemented for the Qwen full-train
  smoke: manifest-driven reload restores delta tensors plus Adam optimizer
  slots and proves next-step parity. Distributed checkpoint layout is still
  open.
- C4/G1 trainer-owned Qwen training is still incomplete, but the full-train
  smoke now uses a reusable `QwenTrainableSession` surface for train steps,
  loss evaluation, and manifest resume. Wiring that surface into the general
  trainer remains open.
- G6 trainer-level real SFT data is incomplete: tokenizer-backed
  `QwenSftDataset` padded response-only batches, including optional
  instruction JSONL input, now feed the focused LoRA SFT smoke. The general
  trainer loop does not own that dataset path yet.
- Real Qwen module-level LoRA now uses a target-layer/module registry for
  layer0 attention `q_proj`/`v_proj`; trainer-owned full-model LoRA injection
  is not done yet.
- KV-cache greedy parity and cached sampling parity are implemented; Python
  cached-generation parity is future work.
- G4 launcher process management plus NCCL scalar, toy DP gradient, and `tch`
  autograd DP=2 trainer smokes exist, but real distributed training is still
  missing: Qwen DP/TP/EP trainer/model paths are not NCCL-backed rank-local
  training yet.
- Distributed checkpoint layout is not defined.
- Trainer production basics such as scheduler, grad clipping, RSS memory
  metrics, and Ray-worker GPU memory reporting are implemented for toy/tch
  paths; real tokenizer-backed padded LoRA SFT batching exists, and the tiny
  `tch` CUDA path plus the representative Qwen full-train smoke have explicit
  bf16 compute policy smokes. General trainer mixed-precision ownership is
  still future work.

Internal planning details live in `_internal_docs/TODO.md`; that directory is
ignored by git.
