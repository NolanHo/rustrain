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
`configs/qwen_session_single_sft.toml` exercises the tokenizer-backed
single-GPU JSONL resume contract for the default representative layer0 trainable
set. `configs/qwen_session_single_layers07_sft.toml` and
`configs/qwen_session_single_layers07_sft_bf16.toml` extend that same contract
to layer0-layer7 in fp32 and bf16 compute. The layer0-layer7 single-rank paths
expect 98 trainable tensors, including the tied embedding, and must preserve
manifest-backed dataset provenance plus absolute data cursor continuity across
external `--resume-from` launches. The bf16 layer0-layer7 path also asserts
`compute_kind = bf16`.

Files:

- `<name>.safetensors`: trainable tensor deltas.
- `<name>.safetensors.optimizer.safetensors`: AdamW first and second moments.
- `<name>.safetensors.json`: manifest.

The manifest must record:

- `format`: stable format identifier.
- `base_model_path`: resolved local base checkpoint path. Qwen paths may start
  as a configured legacy directory such as
  `/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct`, but manifests and
  verifier summaries should record the resolved complete checkpoint directory,
  including HuggingFace `hub/` snapshots when that is the only worker-visible
  layout.
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
  - `streaming_train_batches` when the train batches are built from the JSONL
    streaming window instead of the materialized dataset path.
  - `data_cursor_start`, `data_cursor_end`, and `data_cursor_next`.
  - `data_epoch_*` and `data_sample_offset_*`, derived from
    `cursor / train_samples` and `cursor % train_samples`.

Acceptance:

- Applying the manifest to the base model reproduces the post-step loss within
  tolerance.
- Verifiers reject Qwen delta manifests unless `base_model_path` points to a
  resolved complete Qwen checkpoint containing `config.json`, `tokenizer.json`,
  and `model.safetensors`.
- Reloading both deltas and optimizer slots reproduces the next AdamW step
  within tolerance against a continuous in-memory run.
- Missing tensor names, malformed optimizer slot names, and unsupported format
  identifiers fail clearly.
- Resume rejects changed JSONL provenance or shuffle semantics when the manifest
  contains dataset metadata; legacy manifests without provenance remain
  loadable.
- The single-GPU JSONL external resume paths verify cursor continuity,
  provenance preservation, reload parity, next-step resume parity, and the
  expected trainable registry. The layer0-layer7 fp32 and bf16 variants both
  require 98 trainable tensors, including the tied embedding. The bf16 variant
  also asserts `compute_kind = bf16`.

### Qwen LoRA SFT Adapter Checkpoints

The focused Qwen LoRA SFT trainer writes adapter-only checkpoints for
configured projection targets. These checkpoints are not full model
checkpoints; they restore LoRA adapter tensors against the local base
HuggingFace checkpoint and the current `[lora]` configuration.
`configs/qwen_lora_sft.toml` and `configs/qwen_lora_sft_bf16.toml` both use
target layers `[0, 1]` and target modules `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

Files:

- `<name>.safetensors`: LoRA adapter tensors.
- `<name>.safetensors.json`: adapter manifest.

The adapter manifest must record:

- `format = "rustrain.qwen_lora_sft_adapter.v1"`.
- `base_model_path` and `adapter_safetensors`. `base_model_path` must point to
  the resolved complete Qwen checkpoint directory, not an incomplete legacy
  top-level path.
- `compute_kind`, `steps`, and `train_step`.
- Dataset provenance: `dataset_source_files`,
  `dataset_source_sample_counts`, `dataset_fingerprint`,
  `dataset_order_seed`, `dataset_shuffle`, total/train/eval sample counts,
  `streaming_train_batches`, and data cursor/epoch/offset fields.
- Batch and accumulation metadata: `batch_size` and
  `gradient_accumulation_steps`.
- LoRA identity: `target_layers` and `target_modules`.

Acceptance:

- Resuming from the adapter manifest rejects changed LoRA target layers/modules
  and changed JSONL provenance.
- Adapter verifiers reject manifests unless `base_model_path` contains the
  complete Qwen checkpoint files: `config.json`, `tokenizer.json`, and
  `model.safetensors`.
- Manifest-backed resume rejects `compute_kind` drift between the saved adapter
  manifest and the current train dtype. Direct `.safetensors` adapter resume
  without manifest metadata remains a compatibility path and cannot enforce
  this dtype identity check.
- Direct `.safetensors` adapter resume restores adapter weights but does not
  claim manifest cursor continuity; the run starts from the configured data
  cursor origin and writes a fresh adapter manifest.
- Resume starts from manifest `data_cursor_next`, advances by
  `steps * global_batch_size`, and preserves cursor/epoch/offset consistency.
- JSONL SFT train summaries must report `streaming_train_batches = true`; the
  LoRA resume verifier checks this for both manifest-backed and direct adapter
  resume runs.
- JSONL SFT trainer summaries/logs expose `streaming_index_cache_path`,
  `streaming_index_cache_hit`, and `streaming_index_cache_written` when
  `data.index_cache` is configured. The trainer cache verifier runs the same
  config twice and requires the first run to write the offset-index cache and
  the second run to hit it. The GPU verifier also exports a JSONL subset from
  the cached HuggingFace Arrow dataset
  `/vePFS-Mindverse/share/huggingface/datasets/iamtarun___code_instructions_120k_alpaca`,
  using `scripts/export_instruction_arrow_jsonl.py`, which accepts Arrow IPC
  stream or file caches, scans Arrow record batches without materializing the
  full table, can skip exact full-source row counting with
  `--no-full-row-count`, maps the dataset's `output` column into the normalized
  JSONL `response` field, writes two JSONL shards, attaches the export metadata
  through `data.external_metadata_paths`, and checks tokenizer-free streaming
  metadata, tokenizer-backed batch parity, cursor wrap, and cache write/hit
  behavior on that
  external-cache-derived source. The offset-index cache is keyed by source paths,
  source JSONL file size/mtime metadata, `max_samples`, and the JSONL field
  mapping from `data.instruction_field`, `data.input_field`,
  `data.response_field`, optional
  `data.system_field`, and optional `data.chat_messages_field`, plus
  `data.prompt_template`, `data.prompt_with_input_template`,
  `data.trim_fields`, optional `data.field_defaults`, optional
  `data.field_replacements`, optional `data.field_regex_replacements`, optional
  `data.field_case_transforms`, optional
  `data.field_affixes`, optional `data.field_strips`, optional
  `data.field_splits`, optional
  `data.field_truncations`, optional `data.normalize_whitespace`,
  optional `data.field_regex_contains_any`, optional
  `data.field_regex_excludes_any`,
  `data.min_response_chars`, optional
  `data.max_response_chars`, optional `data.instruction_contains_any`,
  optional `data.response_contains_any`, optional
  `data.response_excludes_any`, optional `data.min_instruction_chars`,
  optional `data.max_instruction_chars`, optional `data.min_input_chars`,
  optional `data.max_input_chars`, optional `data.min_system_chars`, optional
  `data.max_system_chars`, optional `data.min_prompt_chars`, optional
  `data.max_prompt_chars`, optional `data.min_sample_chars`, optional
  `data.max_sample_chars`, optional `data.dedupe_samples`, and training
  `data.source_weights` plus optional `data.source_max_samples`, optional
  `data.skip_invalid_records`, and optional `data.external_metadata_paths`, so stale
  caches from a different external file state, external schema, prompt format,
  default/replacement/case/affix/strip/split/truncation/normalization policy,
  field-regex filtering policy,
  instruction/input/system/prompt/sample/response filtering or
  instruction/response substring/dedupe policy, source weighting policy,
  per-source sample-limit policy, or invalid-record handling policy are
  rejected. Qwen JSONL SFT loaders compile regex replacement and filter entries
  once per resolved field map, then reuse that compiled plan for materialized
  scans, tokenizer-free streaming summaries, offset-index construction, and
  raw-offset replay; the manifest and cache identity still store and compare
  the raw serializable regex configuration.
  The field mapping entries can be flat JSON keys or dotted JSON paths; exact
  top-level keys are resolved before path traversal so flat columns containing
  dots keep their existing meaning.
  When `data.system_field` is unset, the system field is omitted from dataset
  hashing so existing default fingerprints remain stable; when set, the
  normalized system value participates in prompt rendering, deduplication, and
  fingerprint/cache identity.
- Explicit held-out `data.eval_paths` can be capped with
  `data.max_eval_samples`. This limit participates in combined dataset
  provenance and resume fingerprints through the selected eval records, but it
  is not part of the training offset-index cache because that cache indexes
  training sources only.
- Adapter reload preserves SFT train/eval loss, full-Qwen forward logits, and
  greedy generation output.
- Merge/unmerge parity is checked for the focused full-Qwen adapter path, with
  dtype-aware bf16 tolerances.
- The bf16 adapter resume verifier asserts `compute_kind = bf16`.

### Single-GPU `tch-rs` MoE Smoke Checkpoints

The CUDA `tch-moe-smoke` path writes a tiny single-process MoE checkpoint as
checkpoint-contract evidence for autograd parameters and AdamW state. It is not
a production expert-parallel checkpoint format.

Files:

- `<name>.safetensors`: router and expert model tensors.
- `<name>.optimizer.safetensors`: AdamW first and second moments plus
  `optimizer.step`.
- `<name>.safetensors.json`: manifest.

The manifest must record:

- `format = "rustrain.tch_moe.v1"`.
- `global_step`: completed AdamW step count.
- `model_safetensors` and `optimizer_safetensors` paths.
- Model tensor entries for `router.weight`, `experts.up.weight`, and
  `experts.down.weight`, including shape and dtype.
- AdamW slot entries for each model tensor's `adam_m` and `adam_v` state,
  including shape and dtype.
- `optimizer_step_tensor` metadata for the stored scalar step tensor.

Acceptance:

- Model and optimizer safetensors exist and are non-empty.
- The manifest paths match the written safetensors files.
- Manifest tensor names, AdamW slot names, shapes, dtypes, and `global_step`
  match the smoke summary.
- Reloading model tensors plus optimizer slots reproduces post-train loss and
  the next AdamW step within tolerance.

## Distributed Checkpoint Semantics

### Rank0 Replicated DP Checkpoints

The current representative DP Qwen session path uses replicated data-parallel
weights. Rank0 writes the checkpoint artifacts after gradient synchronization;
non-rank0 ranks do not write checkpoint files.
`configs/qwen_session_dp2_layers01.toml` exercises the same contract with
configured trainable transformer layers `[0, 1]`. That DP path has 25 trainable
tensors because it excludes the tied embedding and owns the configured layer
tensors plus final norm.
`configs/qwen_session_dp2_layers01_sft.toml` and
`configs/qwen_session_dp2_layers01_sft_bf16.toml` combine that multi-layer DP
training surface with tokenizer-backed JSONL batches, so the rank0 manifest and
global sharded manifest must also preserve dataset provenance and cursor fields
for the layer0+layer1 trainable set. The external DP resume verifier runs each
config twice: a base DP=2 JSONL run writes the rank0 manifest, then a second
DP=2 run resumes from that manifest and must start at the base
`data_cursor_next`, advance by `steps * local_batch_size * world_size`, keep the
same dataset provenance, and keep the expected 25 trainable tensors for layers
0 and 1 plus final norm. The bf16 path also asserts `compute_kind = bf16`.
`configs/qwen_session_dp2_layers03_sft.toml` and
`configs/qwen_session_dp2_layers03_sft_bf16.toml` exercise the same external
resume contract for the layer0-layer3 representative JSONL DP path. Those
configs must preserve the same cursor/provenance fields, but the expected
trainable registry expands to 49 tensors for layers 0 through 3 plus final norm.
Both the fp32 and bf16 layer0-layer3 SFT resume paths still exclude the tied
embedding by design.
`configs/qwen_session_dp2_layers07_sft.toml` and
`configs/qwen_session_dp2_layers07_sft_bf16.toml` extend that same external
resume contract to the layer0-layer7 representative JSONL DP path. These paths
expect 97 trainable tensors for layers 0 through 7 plus final norm, keep the
same dataset cursor/provenance requirements, and still exclude the tied
embedding by design.

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
  cursor/epoch/offset, shuffle, and `streaming_train_batches` fields used by
  single-rank Qwen delta manifests.

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
- JSONL-backed DP rank summaries must report `streaming_train_batches = true`;
  the DP worker and external resume verifiers reject summaries that omit it or
  report `false`.
- JSONL-backed DP rank summaries expose the same
  `streaming_index_cache_path`, `streaming_index_cache_hit`, and
  `streaming_index_cache_written` fields when `data.index_cache` is configured.
  The DP trainer cache verifier derives rank-local offset-index cache paths,
  then requires both ranks to write their caches on the first launch and hit
  those caches on the second launch.
- The layer0+layer1 JSONL external resume paths verify both the cursor
  continuity contract and the 25-tensor trainable registry for both fp32 and
  bf16 resumed summaries and manifests. The bf16 path also asserts
  `compute_kind = bf16`.
- The layer0-layer3 JSONL external resume paths verify the same cursor
  continuity contract and the 49-tensor trainable registry for both fp32 and
  bf16 resumed summaries and manifests. The bf16 path also asserts
  `compute_kind = bf16`.
- The layer0-layer7 JSONL external resume paths verify the same cursor
  continuity contract and the 97-tensor trainable registry for both fp32 and
  bf16 resumed summaries and manifests. The bf16 path also asserts
  `compute_kind = bf16`.

### Representative Sharded Checkpoints and Future Production

Production distributed training must not overload the rank0 replicated format.
It needs a separate sharded format with explicit rank-local ownership.

The reserved manifest identifier is `rustrain.qwen_sharded.v1`. The current code
defines and validates this schema, and the representative Qwen session DP smoke
writes rank-owned shard manifests plus a global manifest. The representative
2-rank trainer verification restores each rank from its rank-owned model and
optimizer safetensors through the global manifest, verifies reload loss parity,
and verifies next-step resume parity against a continuous rank0-manifest run.
The layer0+layer1 DP verifier uses the same global manifest path and checks
that both rank-local sharded restore and the following sharded train step match
the continuous rank0-manifest run. Its JSONL variant additionally checks that
the global sharded manifest's `consumed_samples`, `data_cursor_next`,
`data_epoch_next`, `data_sample_offset_next`, `data_train_samples`,
`dataset_source_files`, and `dataset_fingerprint` match the rank-local
summaries. DP verifiers also reject Qwen sharded manifests unless
`base_model_path` points to a resolved complete Qwen checkpoint containing
`config.json`, `tokenizer.json`, and `model.safetensors`, and `tokenizer_path`
belongs to that same checkpoint directory.
The focused Qwen TP=2 layer0 smoke also writes rank-owned model shard files and
a global `rustrain.qwen_sharded.v1` manifest for its layer0 attention/MLP tensor
partitions. The global manifest records `base_model_path` and `tokenizer_path`
from the resolved complete Qwen checkpoint, and TP verifiers apply the same
complete-checkpoint requirement to manifests and rank summaries. That TP
manifest is checkpoint-contract evidence only:
optimizer slots for TP row/column shards are first-step smoke AdamW m/v tensors, and
replicated norm slots remain zero because the focused TP train step does not
update them. The same focused smoke restores each rank's model shards through
the global manifest and verifies the fused layer0 output against the full
layer0 reference within tolerance, then applies the same focused shard SGD
update from the restored shards and verifies next-update output parity against
the continuous focused path. The trainer-entry TP path also verifies a focused
causal-LM train-step over a real token batch, but the checkpoint artifacts still
only cover focused layer0 TP shard state with smoke optimizer slots. The
focused TP verifier now also checks the global manifest's base model identity,
tokenizer identity, global step, consumed sample/token counts, seed, dtype,
optimizer, scheduler, explicit lack of JSONL provenance for the focused smoke,
exact parallel topology, embedded rank manifests, every declared rank-owned
model shard plus AdamW slot shape, and non-zero optimizer slots for TP-owned
trainable shards in the written safetensors files. The TP verifiers also check
the focused AdamW first-step slot formulas against causal-LM shard gradient
evidence: `adam_m.sum == (1 - beta1) * grad.sum` and
`adam_v.sum == (1 - beta2) * grad_norm^2`. The focused external TP resume
verifier repeats the same manifest, artifact, and optimizer slot formula checks
for the base checkpoint and the resumed launch's newly written global manifest
while verifying restore and next-update parity.
Production full-parameter TP checkpoint resume and full production sharded
restore over external streaming real data remain open.
The focused EP `parallel-ep-tch-moe-rank-smoke` also writes
`rustrain.ep_sharded.v1` rank manifests. Each rank owns a contiguous expert
range and writes two model shards, `experts.up.weight` and
`experts.down.weight`, plus AdamW `adam_m`/`adam_v` optimizer slots for each
shard. This is checkpoint-contract evidence for rank-owned `tch-rs` expert MLP
parameters after sparse send/recv dispatch and gradient return; it is not a
production MoE trainer checkpoint. The same focused EP path is wired through
`train --config configs/tch_moe_ep2.toml`, so the trainer entrypoint produces
the same rank-owned checkpoint artifacts under a launch output directory and a
shared `rustrain.ep_sharded.v1` global manifest. The global manifest is marked
with `manifest_kind = "global"` and embeds the two rank manifests plus the EP
parallel topology, global step, consumed sample/token counts, dtype, optimizer,
and scheduler metadata. It also records a focused smoke base-model identity,
the tokenizer identity slot, seed, empty dataset provenance, and optional data
progress fields so the EP schema stays aligned with the broader sharded
checkpoint contract. Rank manifests are marked with `manifest_kind = "rank"`;
the field defaults to `rank` for backward-compatible readers. The focused
trainer-entry EP path also accepts that global manifest through `--resume-from`
and restores each rank's owned expert MLP shards plus AdamW slots to verify
reload and next-step parity. This remains focused checkpoint-contract evidence,
not a production MoE trainer checkpoint.
The focused EP global manifest validator also opens each rank-owned model and
optimizer safetensors, requires declared expert shards and AdamW slots to
exist, and rejects artifact shapes that do not match the manifest shard shape.

Required manifest structure:

- A global manifest at the checkpoint root.
- One rank manifest per data/model-parallel rank that writes shard artifacts.
- Explicit axes for DP, TP, PP, EP, and CP ranks.
- A base model identity and tokenizer identity.
- Global train state: global step, consumed samples/tokens, RNG seeds, dtype,
  optimizer, scheduler, and parallel config.
- JSONL dataset provenance: source files, per-source sample counts,
  content/path/field-map/prompt-template/normalization fingerprint, shuffle
  flag, and train-sample count.
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
  output restore plus next-update parity; production run ownership remains
  open.
- The restored production run reproduces loss before the next step within
  tolerance. Done for the representative DP session smoke; focused TP layer0
  restore reproduces fused output and next-update output parity, not a
  production loss resume.
- The next step after restore matches a continuous run within tolerance. Done
  for the representative DP session smoke.
- Manifest validation rejects missing rank shards, wrong world size, wrong
  parallel config, and missing optimizer slots.
- Manifest validation rejects duplicate rank-axis ownership, rank ids that do
  not match the linearized parallel axes, empty scheduler/global-step metadata,
  invalid shard shape metadata, unsupported shard dtype or partition policies,
  duplicate rank-local tensor shard names, and duplicate or colliding optimizer
  slot names. Done for the representative Qwen sharded schema tests.
- Manifest validation can also open rank-owned model and optimizer
  safetensors, require every declared shard and AdamW slot to exist, and reject
  artifact shapes that do not match manifest `shard_shape`. Done for the
  representative Qwen sharded artifact tests and DP/TP resume verifiers.
- Manifest validation rejects partial dataset provenance, non-JSONL dataset
  source paths, inconsistent data cursor/epoch/offset fields, and mismatched
  dataset provenance during resume. Done for the representative DP session
  smoke and schema tests.
- The standard distributed verifier covers rank0-manifest external resume for
  `configs/qwen_session_dp2_sft.toml`: a base DP launch writes a rank0
  checkpoint, a second DP launch resumes from it on both ranks, and the resumed
  run verifies rank0 plus sharded reload/next-step parity.

Until that production path exists, production checkpoint/resume remains open.
