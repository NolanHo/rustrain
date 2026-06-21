# Checkpoint Contract

This document defines the checkpoint behavior that current rustrain smokes
must preserve, and the production distributed checkpoint behavior still to be
implemented.

## Current Formats

### Toy Trainer Checkpoints

Toy checkpoints are single-process artifacts owned by the trainer run
directory. They store model state and AdamW optimizer state so the toy trainer
can resume deterministically on the same small batch.

Acceptance:

- The checkpoint records the model tensors needed by the toy backend.
- AdamW first and second moment state is persisted.
- Reloading the checkpoint reproduces the same one-batch loss within tolerance.
- Resuming training continues from the expected step.

### Qwen Delta Checkpoints

Representative Qwen smokes do not write a full copy of the base model. They
write a delta checkpoint against a local HuggingFace checkpoint plus an
optimizer sidecar.

Files:

- `<name>.safetensors`: trainable tensor deltas.
- `<name>.safetensors.optimizer.safetensors`: AdamW first and second moments.
- `<name>.safetensors.json`: manifest.

The manifest must record:

- `format`: stable format identifier.
- `base_model_path`: local base checkpoint path.
- `reference_fixture`: input/fixture identity used by the smoke.
- `delta_safetensors`: delta safetensors path.
- `optimizer_safetensors`: optimizer safetensors path when optimizer state is
  available.
- `train_step`: completed step count represented by the checkpoint.
- `learning_rate`: learning rate used for the stored update.
- `initial_loss` and `final_loss`: smoke evidence around the stored update.
- Tensor entries with base tensor name, delta tensor name, Adam m slot name,
  Adam v slot name, shape, dtype, and gradient norm evidence.
- When the run is backed by tokenizer JSONL data, dataset provenance and data
  progress fields:
  - `dataset_source_files` and `dataset_source_sample_counts`.
  - `dataset_fingerprint`.
  - `dataset_shuffle`.
  - `data_cursor_start`, `data_cursor_end`, and `data_cursor_next`.
  - `data_epoch_*` and `data_sample_offset_*`, derived from
    `cursor / train_samples` and `cursor % train_samples`.

Acceptance:

- Applying the manifest to the base model reproduces the post-step loss within
  tolerance.
- Reloading both deltas and optimizer slots reproduces the next AdamW step
  within tolerance against a continuous in-memory run.
- Missing tensor names, malformed optimizer slot names, and unsupported format
  identifiers fail clearly.
- Resume rejects changed JSONL provenance or shuffle semantics when the manifest
  contains dataset metadata; legacy manifests without provenance remain
  loadable.

## Distributed Checkpoint Semantics

### Rank0 Replicated DP Checkpoints

The current representative DP Qwen session path uses replicated data-parallel
weights. Rank0 writes the checkpoint artifacts after gradient synchronization;
non-rank0 ranks do not write checkpoint files.

Rank0 artifacts:

- Rank0 delta safetensors.
- Rank0 AdamW optimizer safetensors.
- Rank0 manifest JSON.
- Rank-local summary JSON for every rank.

The rank0 manifest must record:

- `format = "rustrain.qwen_session_dp_rank0.v1"`.
- `writer_rank = 0`.
- `world_size`.
- `train_step`.
- `tensor_count`.
- Delta and optimizer safetensors paths.
- Tensor entries with AdamW slot names for every trainable tensor.
- For JSONL-backed representative runs, the same dataset provenance,
  cursor/epoch/offset, and shuffle fields used by single-rank Qwen delta
  manifests.

Acceptance:

- Rank0 writes delta, optimizer, and manifest artifacts.
- Non-rank0 ranks report the checkpoint path but do not write checkpoint
  artifacts.
- Every rank can reload from the rank0 manifest and reproduce the next step
  within tolerance.
- Rank-local summaries agree on world size, train step count, tensor count,
  global loss improvement, reload delta, and next-step delta.
- JSONL-backed DP resume starts from manifest `data_cursor_next`, advances the
  cursor by `steps * local_batch_size * world_size`, and preserves dataset
  provenance in the next rank0 manifest.

### Representative Sharded Checkpoints and Future Production

Production distributed training must not overload the rank0 replicated format.
It needs a separate sharded format with explicit rank-local ownership.

The reserved manifest identifier is `rustrain.qwen_sharded.v1`. The current code
defines and validates this schema, and the representative Qwen session DP smoke
writes rank-owned shard manifests plus a global manifest. The representative
2-rank trainer verification restores each rank from its rank-owned model and
optimizer safetensors through the global manifest, verifies reload loss parity,
and verifies next-step resume parity against a continuous rank0-manifest run.
The focused Qwen TP=2 layer0 smoke also writes rank-owned model shard files and
a global `rustrain.qwen_sharded.v1` manifest for its layer0 attention/MLP tensor
partitions. That TP manifest is checkpoint-contract evidence only: optimizer
slots are zero smoke placeholders. The same focused smoke restores each rank's
model shards through the global manifest and verifies the fused layer0 output
against the full layer0 reference within tolerance. TP next-step resume parity
and full production sharded restore over external streaming real data remain
open.

Required manifest structure:

- A global manifest at the checkpoint root.
- One rank manifest per data/model-parallel rank that writes shard artifacts.
- Explicit axes for DP, TP, PP, EP, and CP ranks.
- A base model identity and tokenizer identity.
- Global train state: global step, consumed samples/tokens, RNG seeds, dtype,
  optimizer, scheduler, and parallel config.
- JSONL dataset provenance: source files, per-source sample counts, content/path
  fingerprint, shuffle flag, and train-sample count.
- JSONL data progress: `data_cursor_next`, `data_epoch_next`, and
  `data_sample_offset_next`, consistent with `consumed_samples`.
- Shard entries mapping logical parameter names to rank-owned safetensors
  shards.
- Optimizer shard entries for AdamW first and second moments.
- Restore policy for replicated tensors, partitioned tensors, and tied weights.

Minimum acceptance before calling production sharded checkpointing implemented:

- A DP=2 or TP=2 Qwen training path writes distinct rank-owned shard files.
- A fresh launched production run restores those shards without reading
  rank0-only model deltas as the source of truth. Representative DP session
  smoke coverage exists, and focused TP layer0 smoke coverage exists for
  output restore parity; production run ownership remains open.
- The restored production run reproduces loss before the next step within
  tolerance. Done for the representative DP session smoke; focused TP layer0
  restore reproduces fused output parity, not a production loss resume.
- The next step after restore matches a continuous run within tolerance. Done
  for the representative DP session smoke.
- Manifest validation rejects missing rank shards, wrong world size, wrong
  parallel config, and missing optimizer slots.
- Manifest validation rejects partial dataset provenance, non-JSONL dataset
  source paths, inconsistent data cursor/epoch/offset fields, and mismatched
  dataset provenance during resume. Done for the representative DP session
  smoke and schema tests.
- The standard distributed verifier covers rank0-manifest external resume for
  `configs/qwen_session_dp2_sft.toml`: a base DP launch writes a rank0
  checkpoint, a second DP launch resumes from it on both ranks, and the resumed
  run verifies rank0 plus sharded reload/next-step parity.

Until that production path exists, production checkpoint/resume remains open.
