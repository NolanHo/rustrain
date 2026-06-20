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
operations, and launching remote jobs. Local CPU smoke is not allowed, including
as temporary development evidence.

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
commands, or quick local CPU checks on the local machine or directly in the
plain SSH shell; the plain shell does not expose the GPU devices. If a check is
worth running, run it on a Ray GPU worker.

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
- G4 real distributed training is missing: multi-GPU DP/TP/EP are toy or
  simulated smokes, not NCCL-backed rank-local Qwen training.
- Distributed checkpoint layout is not defined.
- Trainer production basics such as scheduler, grad clipping, RSS memory
  metrics, and Ray-worker GPU memory reporting are implemented for toy/tch
  paths; real tokenizer-backed padded LoRA SFT batching exists, and the tiny
  `tch` CUDA path plus the representative Qwen full-train smoke have explicit
  bf16 compute policy smokes. General trainer mixed-precision ownership is
  still future work.

Internal planning details live in `_internal_docs/TODO.md`; that directory is
ignored by git.
