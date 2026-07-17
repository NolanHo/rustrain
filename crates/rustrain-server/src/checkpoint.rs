//! Checkpoint save/load: adapter (LoRA A/B) + optimizer state (Adam m/v) + step count.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use tch::Tensor;

const TP_CHECKPOINT_FORMAT: &str = "rustrain-checkpoint-v4-tp";
const LEGACY_TP_CHECKPOINT_FORMAT: &str = "rustrain-checkpoint-v3-tp";

pub fn is_legacy_tensor_parallel_checkpoint(manifest: &CheckpointManifest) -> bool {
    manifest.format == LEGACY_TP_CHECKPOINT_FORMAT
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoraTpShardLayout {
    #[default]
    LatentRank,
    ColumnParallel,
    RowParallel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraSlotIdentity {
    pub index: usize,
    pub layer: usize,
    pub module: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelCheckpointManifest {
    pub world_size: usize,
    pub tensor_model_parallel_size: usize,
    pub pipeline_model_parallel_size: usize,
    pub data_parallel_size: usize,
    pub expert_model_parallel_size: usize,
    pub context_parallel_size: usize,
    pub global_rank: usize,
    pub tensor_model_parallel_rank: usize,
}

impl ParallelCheckpointManifest {
    pub fn new(
        world_size: usize,
        global_rank: usize,
        tensor_model_parallel_size: usize,
        pipeline_model_parallel_size: usize,
        data_parallel_size: usize,
        expert_model_parallel_size: usize,
        context_parallel_size: usize,
    ) -> Result<Self> {
        let sizes = [
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            data_parallel_size,
            expert_model_parallel_size,
            context_parallel_size,
        ];
        if sizes.contains(&0) {
            bail!("parallel checkpoint sizes must be positive");
        }
        let expected_world_size = sizes
            .into_iter()
            .try_fold(1usize, |product, size| product.checked_mul(size))
            .context("parallel checkpoint world size overflow")?;
        if world_size != expected_world_size {
            bail!(
                "parallel checkpoint topology product is {expected_world_size}, but WORLD_SIZE is {world_size}"
            );
        }
        if global_rank >= world_size {
            bail!("global rank {global_rank} is outside WORLD_SIZE={world_size}");
        }
        Ok(Self {
            world_size,
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            data_parallel_size,
            expert_model_parallel_size,
            context_parallel_size,
            global_rank,
            tensor_model_parallel_rank: global_rank % tensor_model_parallel_size,
        })
    }

    pub fn from_env() -> Result<Self> {
        let world_size = env_usize(&["WORLD_SIZE"], 1)?;
        let global_rank = env_usize(&["RANK"], 0)?;
        let tp = env_usize(&["TP_SIZE", "RUSTRAIN_TP_SIZE"], 1)?;
        let pp = env_usize(&["PP_SIZE", "RUSTRAIN_PP_SIZE"], 1)?;
        let ep = env_usize(&["EP_SIZE", "RUSTRAIN_EP_SIZE"], 1)?;
        let cp = env_usize(&["CP_SIZE", "RUSTRAIN_CP_SIZE"], 1)?;
        let non_dp = tp
            .checked_mul(pp)
            .and_then(|size| size.checked_mul(ep))
            .and_then(|size| size.checked_mul(cp))
            .context("model-parallel topology product overflow")?;
        let dp = match env_usize_optional(&["DP_SIZE", "RUSTRAIN_DP_SIZE"])? {
            Some(dp) => dp,
            None if world_size % non_dp == 0 => world_size / non_dp,
            None => {
                bail!("WORLD_SIZE={world_size} is not divisible by model-parallel product {non_dp}")
            }
        };
        Self::new(world_size, global_rank, tp, pp, dp, ep, cp)
    }

    fn is_tensor_parallel(&self) -> bool {
        self.tensor_model_parallel_size > 1
    }

    fn replica_identity(&self) -> String {
        format!(
            "global-rank-{}-tp-rank-{}",
            self.global_rank, self.tensor_model_parallel_rank
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShardManifest {
    pub file: String,
    pub tensor_name: String,
    pub state: String,
    #[serde(default)]
    pub adapter_id: Option<i64>,
    pub global_lora_rank: i64,
    pub global_shape: Vec<i64>,
    pub local_shape: Vec<i64>,
    pub partition_axis: usize,
    #[serde(default)]
    pub layout: LoraTpShardLayout,
    #[serde(default)]
    pub replicated: bool,
    pub global_offset: Vec<i64>,
    pub replica_identity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub format: String,
    pub step: u64,
    pub loss: f64,
    pub model_path: String,
    pub lora_rank: i64,
    pub lora_alpha: f64,
    pub files: Vec<String>,
    #[serde(default)]
    pub dynamic_adapters: Vec<DynamicAdapterManifest>,
    #[serde(default)]
    pub parallel: Option<ParallelCheckpointManifest>,
    #[serde(default)]
    pub tensor_shards: Vec<TensorShardManifest>,
    #[serde(default)]
    pub fixed_shard_layouts: Vec<LoraTpShardLayout>,
    #[serde(default)]
    pub fixed_slot_identities: Vec<LoraSlotIdentity>,
}

pub fn validate_fixed_tp_resume(
    manifest: &CheckpointManifest,
    expected_layouts: &[LoraTpShardLayout],
    expected_identities: &[LoraSlotIdentity],
) -> Result<()> {
    if is_legacy_tensor_parallel_checkpoint(manifest) {
        if expected_layouts
            .iter()
            .any(|layout| *layout != LoraTpShardLayout::LatentRank)
        {
            bail!(
                "legacy tensor-parallel v3 checkpoints cannot restore fixed Q/K/V/O LoRA into the projection-aware layout; use a v4 checkpoint or migrate the adapter from a merged artifact"
            );
        }
        return Ok(());
    }
    if manifest.fixed_shard_layouts != expected_layouts {
        bail!(
            "fixed LoRA shard layouts do not match the current runtime slots: checkpoint={:?}, runtime={expected_layouts:?}",
            manifest.fixed_shard_layouts
        );
    }
    if manifest.fixed_slot_identities != expected_identities {
        bail!(
            "fixed LoRA slot identities do not match the current runtime slots: checkpoint={:?}, runtime={expected_identities:?}",
            manifest.fixed_slot_identities
        );
    }
    Ok(())
}

pub fn validate_dynamic_tp_resume(
    manifest: &CheckpointManifest,
    adapter_id: i64,
    saved_layouts: &[LoraTpShardLayout],
    expected_layouts: &[LoraTpShardLayout],
) -> Result<()> {
    if is_legacy_tensor_parallel_checkpoint(manifest) {
        if expected_layouts
            .iter()
            .any(|layout| *layout != LoraTpShardLayout::LatentRank)
        {
            bail!(
                "legacy tensor-parallel v3 checkpoint adapter {adapter_id} contains Q/K/V/O LoRA that cannot be restored into the projection-aware layout; use a v4 checkpoint or migrate the adapter from a merged artifact"
            );
        }
        return Ok(());
    }
    if saved_layouts != expected_layouts {
        bail!(
            "dynamic adapter {adapter_id} shard layouts do not match the current runtime slots: checkpoint={saved_layouts:?}, runtime={expected_layouts:?}"
        );
    }
    Ok(())
}

pub fn fixed_restore_slot_indices(
    saved_a_count: usize,
    saved_b_count: usize,
    active_slot_indices: &[usize],
    native_slot_count: usize,
) -> Result<Vec<usize>> {
    if saved_a_count != saved_b_count {
        bail!("checkpoint fixed LoRA A/B count mismatch: {saved_a_count}/{saved_b_count}");
    }
    if saved_a_count == active_slot_indices.len() {
        return Ok(active_slot_indices.to_vec());
    }
    if saved_a_count == native_slot_count {
        // v1/v2 checkpoints stored inactive positional placeholders.
        return Ok((0..native_slot_count).collect());
    }
    bail!(
        "checkpoint LoRA slot count mismatch: checkpoint A/B={saved_a_count}/{saved_b_count}, active={}, native={native_slot_count}",
        active_slot_indices.len()
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicAdapterManifest {
    pub id: i64,
    pub rank: i64,
    pub alpha: f64,
    /// Adam bias-correction clock for this tenant. It is independent from
    /// the session-wide `CheckpointManifest::step` and other adapters.
    /// Missing values in older v2 manifests default to zero.
    #[serde(default)]
    pub optimizer_step: u64,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<String>,
    #[serde(default)]
    pub shard_layouts: Vec<LoraTpShardLayout>,
    pub parameter_count: usize,
    pub optimizer_count: usize,
}

pub struct DynamicAdapterCheckpoint {
    pub manifest: DynamicAdapterManifest,
    pub lora_a: Vec<Tensor>,
    pub lora_b: Vec<Tensor>,
    pub adam_m: Vec<Tensor>,
    pub adam_v: Vec<Tensor>,
}

pub struct CheckpointData {
    pub manifest: CheckpointManifest,
    pub lora_a: Vec<Tensor>,
    pub lora_b: Vec<Tensor>,
    pub adam_m: Vec<Tensor>,
    pub adam_v: Vec<Tensor>,
    pub dynamic_adapters: Vec<DynamicAdapterCheckpoint>,
}

/// Save checkpoint to a directory.
/// Creates: manifest.json, adapter.safetensors, optimizer.safetensors
pub fn save_checkpoint(
    dir: &Path,
    step: u64,
    loss: f64,
    model_path: &str,
    lora_rank: i64,
    lora_alpha: f64,
    lora_a: &[Tensor],
    lora_b: &[Tensor],
    adam_m: &[Tensor],
    adam_v: &[Tensor],
) -> Result<()> {
    save_checkpoint_with_dynamic(
        dir,
        step,
        loss,
        model_path,
        lora_rank,
        lora_alpha,
        lora_a,
        lora_b,
        adam_m,
        adam_v,
        &[],
    )
}

pub fn save_checkpoint_with_dynamic(
    dir: &Path,
    step: u64,
    loss: f64,
    model_path: &str,
    lora_rank: i64,
    lora_alpha: f64,
    lora_a: &[Tensor],
    lora_b: &[Tensor],
    adam_m: &[Tensor],
    adam_v: &[Tensor],
    dynamic_adapters: &[DynamicAdapterCheckpoint],
) -> Result<()> {
    save_checkpoint_with_dynamic_at(
        dir,
        step,
        loss,
        model_path,
        lora_rank,
        lora_alpha,
        lora_a,
        lora_b,
        adam_m,
        adam_v,
        dynamic_adapters,
        &[],
        &[],
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn save_checkpoint_with_dynamic_for_topology(
    dir: &Path,
    step: u64,
    loss: f64,
    model_path: &str,
    lora_rank: i64,
    lora_alpha: f64,
    lora_a: &[Tensor],
    lora_b: &[Tensor],
    adam_m: &[Tensor],
    adam_v: &[Tensor],
    dynamic_adapters: &[DynamicAdapterCheckpoint],
    fixed_shard_layouts: &[LoraTpShardLayout],
    fixed_slot_identities: &[LoraSlotIdentity],
    parallel: &ParallelCheckpointManifest,
) -> Result<()> {
    if !parallel.is_tensor_parallel() {
        return save_checkpoint_with_dynamic(
            dir,
            step,
            loss,
            model_path,
            lora_rank,
            lora_alpha,
            lora_a,
            lora_b,
            adam_m,
            adam_v,
            dynamic_adapters,
        );
    }
    let rank_dir = rank_checkpoint_dir(dir, parallel.global_rank);
    save_checkpoint_with_dynamic_at(
        &rank_dir,
        step,
        loss,
        model_path,
        lora_rank,
        lora_alpha,
        lora_a,
        lora_b,
        adam_m,
        adam_v,
        dynamic_adapters,
        fixed_shard_layouts,
        fixed_slot_identities,
        Some(parallel),
    )
}

#[allow(clippy::too_many_arguments)]
fn save_checkpoint_with_dynamic_at(
    dir: &Path,
    step: u64,
    loss: f64,
    model_path: &str,
    lora_rank: i64,
    lora_alpha: f64,
    lora_a: &[Tensor],
    lora_b: &[Tensor],
    adam_m: &[Tensor],
    adam_v: &[Tensor],
    dynamic_adapters: &[DynamicAdapterCheckpoint],
    fixed_shard_layouts: &[LoraTpShardLayout],
    fixed_slot_identities: &[LoraSlotIdentity],
    parallel: Option<&ParallelCheckpointManifest>,
) -> Result<()> {
    validate_tensor_counts(
        lora_a,
        lora_b,
        adam_m,
        adam_v,
        dynamic_adapters,
        parallel.is_some(),
    )?;
    if parallel.is_some() && fixed_slot_identities.len() != lora_a.len() {
        bail!(
            "fixed LoRA slot identity count {} does not match parameter count {}",
            fixed_slot_identities.len(),
            lora_a.len()
        );
    }
    let unique_fixed_slots = fixed_slot_identities
        .iter()
        .map(|identity| (identity.index, identity.layer, identity.module.as_str()))
        .collect::<std::collections::BTreeSet<_>>();
    if unique_fixed_slots.len() != fixed_slot_identities.len() {
        bail!("fixed LoRA slot identities must be unique");
    }
    let tensor_shards = match parallel {
        Some(parallel) => build_tensor_shard_manifest(
            parallel,
            lora_rank,
            lora_a,
            lora_b,
            adam_m,
            adam_v,
            dynamic_adapters,
            fixed_shard_layouts,
        )?,
        None => Vec::new(),
    };
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create checkpoint dir {}", dir.display()))?;

    // Save adapter (LoRA A/B) as safetensors
    let adapter_path = dir.join("adapter.safetensors");
    let mut adapter_tensors = named_tensors(lora_a, lora_b, "a_", "b_");
    let mut dynamic_manifests = Vec::with_capacity(dynamic_adapters.len());
    for adapter in dynamic_adapters {
        let id = adapter.manifest.id;
        adapter_tensors.extend(named_tensors(
            &adapter.lora_a,
            &adapter.lora_b,
            &format!("dynamic_{id}_a_"),
            &format!("dynamic_{id}_b_"),
        ));
        dynamic_manifests.push(adapter.manifest.clone());
    }
    save_named_tensors(&adapter_path, adapter_tensors)?;

    // Save optimizer state (Adam m/v) as safetensors
    let optimizer_path = dir.join("optimizer.safetensors");
    let mut optimizer_tensors = named_tensors(adam_m, adam_v, "a_", "b_");
    for adapter in dynamic_adapters {
        let id = adapter.manifest.id;
        optimizer_tensors.extend(named_tensors(
            &adapter.adam_m,
            &adapter.adam_v,
            &format!("dynamic_{id}_a_"),
            &format!("dynamic_{id}_b_"),
        ));
    }
    save_named_tensors(&optimizer_path, optimizer_tensors)?;

    // Write manifest
    let manifest = CheckpointManifest {
        format: if parallel.is_some() {
            TP_CHECKPOINT_FORMAT.to_string()
        } else if dynamic_manifests.is_empty() {
            "rustrain-checkpoint-v1".to_string()
        } else {
            "rustrain-checkpoint-v2".to_string()
        },
        step,
        loss,
        model_path: model_path.to_string(),
        lora_rank,
        lora_alpha,
        files: vec!["adapter.safetensors".into(), "optimizer.safetensors".into()],
        dynamic_adapters: dynamic_manifests,
        parallel: parallel.cloned(),
        tensor_shards,
        fixed_shard_layouts: fixed_shard_layouts.to_vec(),
        fixed_slot_identities: fixed_slot_identities.to_vec(),
    };
    let manifest_path = dir.join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| "write manifest.json")?;

    tracing::info!(
        step,
        loss,
        path = dir.display().to_string(),
        "checkpoint saved"
    );
    Ok(())
}

/// Load checkpoint from a directory.
pub fn load_checkpoint(dir: &Path) -> Result<CheckpointData> {
    load_checkpoint_at(dir, None)
}

pub fn load_checkpoint_for_topology(
    dir: &Path,
    parallel: &ParallelCheckpointManifest,
) -> Result<CheckpointData> {
    if !parallel.is_tensor_parallel() {
        return load_checkpoint(dir);
    }
    let rank_dir = rank_checkpoint_dir(dir, parallel.global_rank);
    load_checkpoint_at(&rank_dir, Some(parallel))
}

fn load_checkpoint_at(
    dir: &Path,
    expected_parallel: Option<&ParallelCheckpointManifest>,
) -> Result<CheckpointData> {
    let manifest_path = dir.join("manifest.json");
    let manifest: CheckpointManifest = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?,
    )
    .with_context(|| "parse manifest.json")?;
    match expected_parallel {
        Some(expected) => {
            if manifest.format != TP_CHECKPOINT_FORMAT
                && manifest.format != LEGACY_TP_CHECKPOINT_FORMAT
            {
                bail!(
                    "tensor-parallel resume requires {TP_CHECKPOINT_FORMAT} or {LEGACY_TP_CHECKPOINT_FORMAT}, found {}",
                    manifest.format
                );
            }
            let saved = manifest
                .parallel
                .as_ref()
                .context("tensor-parallel checkpoint is missing topology metadata")?;
            if saved != expected {
                bail!(
                    "tensor-parallel checkpoint topology mismatch: saved={saved:?}, current={expected:?}"
                );
            }
        }
        None if manifest.format == TP_CHECKPOINT_FORMAT
            || manifest.format == LEGACY_TP_CHECKPOINT_FORMAT =>
        {
            bail!("tensor-parallel checkpoint must be loaded with rank topology");
        }
        None => {}
    }

    let adapter_path = dir.join("adapter.safetensors");
    let adapter_named = read_named_tensors(&adapter_path)?;
    let lora_a = collect_side(&adapter_named, "a_")?;
    let lora_b = collect_side(&adapter_named, "b_")?;

    let optimizer_path = dir.join("optimizer.safetensors");
    let optimizer_named = read_named_tensors(&optimizer_path)?;
    let adam_m = collect_side(&optimizer_named, "a_")?;
    let adam_v = collect_side(&optimizer_named, "b_")?;
    let mut dynamic_adapters = Vec::with_capacity(manifest.dynamic_adapters.len());
    for dynamic_manifest in &manifest.dynamic_adapters {
        let id = dynamic_manifest.id;
        let dynamic_lora_a = collect_side(&adapter_named, &format!("dynamic_{id}_a_"))?;
        let dynamic_lora_b = collect_side(&adapter_named, &format!("dynamic_{id}_b_"))?;
        let dynamic_adam_m = collect_side(&optimizer_named, &format!("dynamic_{id}_a_"))?;
        let dynamic_adam_v = collect_side(&optimizer_named, &format!("dynamic_{id}_b_"))?;
        if dynamic_lora_a.len() != dynamic_manifest.parameter_count
            || dynamic_lora_b.len() != dynamic_manifest.parameter_count
            || dynamic_adam_m.len() != dynamic_manifest.optimizer_count
            || dynamic_adam_v.len() != dynamic_manifest.optimizer_count
        {
            anyhow::bail!("dynamic adapter {id} checkpoint tensor count mismatch");
        }
        dynamic_adapters.push(DynamicAdapterCheckpoint {
            manifest: dynamic_manifest.clone(),
            lora_a: dynamic_lora_a,
            lora_b: dynamic_lora_b,
            adam_m: dynamic_adam_m,
            adam_v: dynamic_adam_v,
        });
    }

    if let Some(parallel) = expected_parallel {
        let expected_shards = build_tensor_shard_manifest(
            parallel,
            manifest.lora_rank,
            &lora_a,
            &lora_b,
            &adam_m,
            &adam_v,
            &dynamic_adapters,
            &manifest.fixed_shard_layouts,
        )?;
        validate_saved_shards(&manifest.tensor_shards, &expected_shards)?;
    }

    tracing::info!(
        step = manifest.step,
        loss = manifest.loss,
        "checkpoint loaded"
    );

    Ok(CheckpointData {
        manifest,
        lora_a,
        lora_b,
        adam_m,
        adam_v,
        dynamic_adapters,
    })
}

#[derive(Clone, Copy)]
enum LoraSide {
    A,
    B,
}

fn validate_tensor_counts(
    lora_a: &[Tensor],
    lora_b: &[Tensor],
    adam_m: &[Tensor],
    adam_v: &[Tensor],
    dynamic_adapters: &[DynamicAdapterCheckpoint],
    strict_optimizer_layout: bool,
) -> Result<()> {
    if lora_a.len() != lora_b.len() || adam_m.len() != adam_v.len() {
        bail!("fixed adapter checkpoint tensor count mismatch");
    }
    if strict_optimizer_layout
        && !adam_m.is_empty()
        && adam_m.len() != lora_a.len().saturating_mul(2)
    {
        bail!("tensor-parallel fixed optimizer state must contain A/B entries for every LoRA slot");
    }
    for adapter in dynamic_adapters {
        if adapter.lora_a.len() != adapter.lora_b.len()
            || adapter.adam_m.len() != adapter.adam_v.len()
            || adapter.manifest.parameter_count != adapter.lora_a.len()
            || adapter.manifest.optimizer_count != adapter.adam_m.len()
        {
            bail!(
                "dynamic adapter {} checkpoint count mismatch",
                adapter.manifest.id
            );
        }
        if strict_optimizer_layout
            && !adapter.adam_m.is_empty()
            && adapter.adam_m.len() != adapter.lora_a.len().saturating_mul(2)
        {
            bail!(
                "tensor-parallel dynamic adapter {} optimizer state must contain A/B entries for every LoRA slot",
                adapter.manifest.id
            );
        }
    }
    Ok(())
}

fn build_tensor_shard_manifest(
    parallel: &ParallelCheckpointManifest,
    fixed_lora_rank: i64,
    lora_a: &[Tensor],
    lora_b: &[Tensor],
    adam_m: &[Tensor],
    adam_v: &[Tensor],
    dynamic_adapters: &[DynamicAdapterCheckpoint],
    fixed_shard_layouts: &[LoraTpShardLayout],
) -> Result<Vec<TensorShardManifest>> {
    validate_tensor_counts(lora_a, lora_b, adam_m, adam_v, dynamic_adapters, true)?;
    let mut shards = Vec::new();
    append_adapter_shards(
        &mut shards,
        parallel,
        None,
        fixed_lora_rank,
        lora_a,
        lora_b,
        adam_m,
        adam_v,
        fixed_shard_layouts,
    )?;
    for adapter in dynamic_adapters {
        append_adapter_shards(
            &mut shards,
            parallel,
            Some(adapter.manifest.id),
            adapter.manifest.rank,
            &adapter.lora_a,
            &adapter.lora_b,
            &adapter.adam_m,
            &adapter.adam_v,
            &adapter.manifest.shard_layouts,
        )?;
    }
    Ok(shards)
}

#[allow(clippy::too_many_arguments)]
fn append_adapter_shards(
    shards: &mut Vec<TensorShardManifest>,
    parallel: &ParallelCheckpointManifest,
    adapter_id: Option<i64>,
    global_lora_rank: i64,
    lora_a: &[Tensor],
    lora_b: &[Tensor],
    adam_m: &[Tensor],
    adam_v: &[Tensor],
    shard_layouts: &[LoraTpShardLayout],
) -> Result<()> {
    if !shard_layouts.is_empty() && shard_layouts.len() != lora_a.len() {
        bail!(
            "LoRA shard layout count {} does not match parameter count {}",
            shard_layouts.len(),
            lora_a.len()
        );
    }
    let layout = |index: usize| {
        shard_layouts
            .get(index)
            .copied()
            .unwrap_or(LoraTpShardLayout::LatentRank)
    };
    let adapter_prefix = adapter_id
        .map(|id| format!("dynamic_{id}_"))
        .unwrap_or_default();
    for (index, tensor) in lora_a.iter().enumerate() {
        shards.push(tensor_shard(
            parallel,
            adapter_id,
            global_lora_rank,
            "adapter.safetensors",
            format!("{adapter_prefix}a_{index}"),
            "lora_a",
            LoraSide::A,
            layout(index),
            tensor,
        )?);
    }
    for (index, tensor) in lora_b.iter().enumerate() {
        shards.push(tensor_shard(
            parallel,
            adapter_id,
            global_lora_rank,
            "adapter.safetensors",
            format!("{adapter_prefix}b_{index}"),
            "lora_b",
            LoraSide::B,
            layout(index),
            tensor,
        )?);
    }
    for (index, tensor) in adam_m.iter().enumerate() {
        shards.push(tensor_shard(
            parallel,
            adapter_id,
            global_lora_rank,
            "optimizer.safetensors",
            format!("{adapter_prefix}a_{index}"),
            "adam_m",
            if index % 2 == 0 {
                LoraSide::A
            } else {
                LoraSide::B
            },
            layout(index / 2),
            tensor,
        )?);
    }
    for (index, tensor) in adam_v.iter().enumerate() {
        shards.push(tensor_shard(
            parallel,
            adapter_id,
            global_lora_rank,
            "optimizer.safetensors",
            format!("{adapter_prefix}b_{index}"),
            "adam_v",
            if index % 2 == 0 {
                LoraSide::A
            } else {
                LoraSide::B
            },
            layout(index / 2),
            tensor,
        )?);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tensor_shard(
    parallel: &ParallelCheckpointManifest,
    adapter_id: Option<i64>,
    global_lora_rank: i64,
    file: &str,
    tensor_name: String,
    state: &str,
    side: LoraSide,
    layout: LoraTpShardLayout,
    tensor: &Tensor,
) -> Result<TensorShardManifest> {
    let tp_size = i64::try_from(parallel.tensor_model_parallel_size)
        .context("TP size exceeds checkpoint tensor shape range")?;
    let tp_rank = i64::try_from(parallel.tensor_model_parallel_rank)
        .context("TP rank exceeds checkpoint tensor shape range")?;
    if global_lora_rank <= 0 || global_lora_rank % tp_size != 0 {
        bail!(
            "global LoRA rank {global_lora_rank} must be positive and divisible by TP size {tp_size}"
        );
    }
    let local_shape = tensor.size();
    if local_shape.len() < 2 {
        bail!("checkpoint tensor {file}:{tensor_name} must have at least two dimensions");
    }
    let rank_axis = match side {
        LoraSide::A => local_shape.len() - 2,
        LoraSide::B => local_shape.len() - 1,
    };
    let local_lora_rank = global_lora_rank / tp_size;
    let mut global_shape = local_shape.clone();
    let mut global_offset = vec![0; local_shape.len()];
    let (partition_axis, replicated) = match (layout, side) {
        (LoraTpShardLayout::LatentRank, _) => {
            if local_shape[rank_axis] != local_lora_rank {
                bail!(
                    "checkpoint tensor {file}:{tensor_name} has local rank {} on axis {rank_axis}, expected {local_lora_rank}",
                    local_shape[rank_axis]
                );
            }
            global_shape[rank_axis] = global_lora_rank;
            global_offset[rank_axis] = tp_rank * local_lora_rank;
            (rank_axis, false)
        }
        (LoraTpShardLayout::ColumnParallel, LoraSide::A)
        | (LoraTpShardLayout::RowParallel, LoraSide::B) => {
            if local_shape[rank_axis] != global_lora_rank {
                bail!(
                    "replicated checkpoint tensor {file}:{tensor_name} has rank {} on axis {rank_axis}, expected {global_lora_rank}",
                    local_shape[rank_axis]
                );
            }
            (rank_axis, true)
        }
        (LoraTpShardLayout::ColumnParallel, LoraSide::B) => {
            if local_shape[rank_axis] != global_lora_rank {
                bail!(
                    "column-parallel checkpoint tensor {file}:{tensor_name} has rank {} on axis {rank_axis}, expected {global_lora_rank}",
                    local_shape[rank_axis]
                );
            }
            let axis = local_shape.len() - 2;
            global_shape[axis] *= tp_size;
            global_offset[axis] = tp_rank * local_shape[axis];
            (axis, false)
        }
        (LoraTpShardLayout::RowParallel, LoraSide::A) => {
            if local_shape[rank_axis] != global_lora_rank {
                bail!(
                    "row-parallel checkpoint tensor {file}:{tensor_name} has rank {} on axis {rank_axis}, expected {global_lora_rank}",
                    local_shape[rank_axis]
                );
            }
            let axis = local_shape.len() - 1;
            global_shape[axis] *= tp_size;
            global_offset[axis] = tp_rank * local_shape[axis];
            (axis, false)
        }
    };
    Ok(TensorShardManifest {
        file: file.to_string(),
        tensor_name,
        state: state.to_string(),
        adapter_id,
        global_lora_rank,
        global_shape,
        local_shape,
        partition_axis,
        layout,
        replicated,
        global_offset,
        replica_identity: if replicated {
            "tp-replicated".to_string()
        } else {
            parallel.replica_identity()
        },
    })
}

fn validate_saved_shards(
    saved: &[TensorShardManifest],
    expected: &[TensorShardManifest],
) -> Result<()> {
    let keyed = |shards: &[TensorShardManifest]| -> Result<BTreeMap<_, _>> {
        let mut by_key = BTreeMap::new();
        for shard in shards {
            let key = (
                shard.file.clone(),
                shard.tensor_name.clone(),
                shard.state.clone(),
            );
            if by_key.insert(key, shard.clone()).is_some() {
                bail!(
                    "duplicate tensor shard metadata for {}:{} ({})",
                    shard.file,
                    shard.tensor_name,
                    shard.state
                );
            }
        }
        Ok(by_key)
    };
    let saved = keyed(saved)?;
    let expected = keyed(expected)?;
    if saved.len() != expected.len() {
        bail!(
            "tensor shard metadata count mismatch: saved={}, expected={}",
            saved.len(),
            expected.len()
        );
    }
    for (key, expected_shard) in expected {
        let saved_shard = saved
            .get(&key)
            .with_context(|| format!("missing tensor shard metadata for {}:{}", key.0, key.1))?;
        if saved_shard != &expected_shard {
            bail!(
                "tensor shard metadata mismatch for {}:{}: saved={saved_shard:?}, expected={expected_shard:?}",
                key.0,
                key.1
            );
        }
    }
    Ok(())
}

fn rank_checkpoint_dir(root: &Path, global_rank: usize) -> PathBuf {
    root.join(format!("rank-{global_rank:05}"))
}

fn env_usize(names: &[&str], default: usize) -> Result<usize> {
    Ok(env_usize_optional(names)?.unwrap_or(default))
}

fn env_usize_optional(names: &[&str]) -> Result<Option<usize>> {
    let Some((name, value)) = names
        .iter()
        .find_map(|name| env::var(name).ok().map(|value| (*name, value)))
    else {
        return Ok(None);
    };
    let value = value
        .parse::<usize>()
        .with_context(|| format!("{name} must be a non-negative integer"))?;
    if value == 0 && name != "RANK" {
        bail!("{name} must be positive");
    }
    Ok(Some(value))
}

fn named_tensors(
    a: &[Tensor],
    b: &[Tensor],
    a_prefix: &str,
    b_prefix: &str,
) -> Vec<(String, Tensor)> {
    let mut named: Vec<(String, Tensor)> = Vec::new();
    for (i, t) in a.iter().enumerate() {
        named.push((
            format!("{a_prefix}{i}"),
            t.to_kind(tch::Kind::Float).to_device(tch::Device::Cpu),
        ));
    }
    for (i, t) in b.iter().enumerate() {
        named.push((
            format!("{b_prefix}{i}"),
            t.to_kind(tch::Kind::Float).to_device(tch::Device::Cpu),
        ));
    }

    named
}

fn save_named_tensors(path: &Path, named: Vec<(String, Tensor)>) -> Result<()> {
    Tensor::write_safetensors(&named, path).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn read_named_tensors(path: &Path) -> Result<std::collections::BTreeMap<String, Tensor>> {
    let named =
        Tensor::read_safetensors(path).with_context(|| format!("read {}", path.display()))?;
    let mut by_name = std::collections::BTreeMap::new();
    for (name, tensor) in named {
        by_name.insert(name, tensor);
    }
    Ok(by_name)
}

fn collect_side(
    tensors: &std::collections::BTreeMap<String, Tensor>,
    prefix: &str,
) -> Result<Vec<Tensor>> {
    let mut indices = tensors
        .keys()
        .filter_map(|name| name.strip_prefix(prefix)?.parse::<usize>().ok())
        .collect::<Vec<_>>();
    indices.sort_unstable();
    let mut result = Vec::with_capacity(indices.len());
    for (expected, index) in indices.into_iter().enumerate() {
        if index != expected {
            anyhow::bail!("checkpoint tensor indices for {prefix} are not contiguous");
        }
        result.push(
            tensors
                .get(&format!("{prefix}{index}"))
                .expect("index collected from map")
                .shallow_clone(),
        );
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_adapter_and_optimizer_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let a = Tensor::arange(6, (tch::Kind::Float, tch::Device::Cpu)).reshape([2, 3]);
        let b = Tensor::arange(8, (tch::Kind::Float, tch::Device::Cpu)).reshape([4, 2]);
        let m = Tensor::ones([2, 3], (tch::Kind::Float, tch::Device::Cpu));
        let v = Tensor::full([4, 2], 2.0, (tch::Kind::Float, tch::Device::Cpu));
        save_checkpoint(
            dir.path(),
            7,
            1.25,
            "Qwen/test",
            2,
            4.0,
            &[a.shallow_clone()],
            &[b.shallow_clone()],
            &[m.shallow_clone()],
            &[v.shallow_clone()],
        )
        .unwrap();
        let loaded = load_checkpoint(dir.path()).unwrap();
        assert!(loaded.manifest.parallel.is_none());
        assert!(loaded.manifest.tensor_shards.is_empty());
        assert_eq!(loaded.manifest.lora_alpha, 4.0);
        assert_eq!(loaded.manifest.step, 7);
        assert_eq!(loaded.lora_a[0].size(), [2, 3]);
        assert_eq!(loaded.lora_b[0].size(), [4, 2]);
        assert!(loaded.lora_a[0].allclose(&a, 1e-6, 1e-6, false));
        assert!(loaded.lora_b[0].allclose(&b, 1e-6, 1e-6, false));
        assert!(loaded.adam_m[0].allclose(&m, 1e-6, 1e-6, false));
        assert!(loaded.adam_v[0].allclose(&v, 1e-6, 1e-6, false));
    }

    #[test]
    fn dynamic_checkpoint_roundtrip_preserves_metadata_and_state() {
        let dir = tempfile::tempdir().unwrap();
        let dynamic = DynamicAdapterCheckpoint {
            manifest: DynamicAdapterManifest {
                id: 7,
                rank: 3,
                alpha: 6.0,
                optimizer_step: 19,
                target_layers: vec![1, 3],
                target_modules: vec!["q_proj".into(), "down_proj".into()],
                shard_layouts: Vec::new(),
                parameter_count: 2,
                optimizer_count: 4,
            },
            lora_a: (0..2)
                .map(|_| Tensor::ones([3, 8], (tch::Kind::Float, tch::Device::Cpu)))
                .collect(),
            lora_b: (0..2)
                .map(|_| Tensor::full([16, 3], 2.0, (tch::Kind::Float, tch::Device::Cpu)))
                .collect(),
            adam_m: (0..4)
                .map(|_| Tensor::full([3, 8], 3.0, (tch::Kind::Float, tch::Device::Cpu)))
                .collect(),
            adam_v: (0..4)
                .map(|_| Tensor::full([3, 8], 4.0, (tch::Kind::Float, tch::Device::Cpu)))
                .collect(),
        };
        save_checkpoint_with_dynamic(
            dir.path(),
            11,
            0.5,
            "Qwen/test",
            2,
            4.0,
            &[],
            &[],
            &[],
            &[],
            &[dynamic],
        )
        .unwrap();
        let loaded = load_checkpoint(dir.path()).unwrap();
        assert_eq!(loaded.manifest.format, "rustrain-checkpoint-v2");
        assert!(loaded.manifest.parallel.is_none());
        assert!(loaded.manifest.tensor_shards.is_empty());
        assert_eq!(loaded.dynamic_adapters.len(), 1);
        let loaded_dynamic = &loaded.dynamic_adapters[0];
        assert_eq!(loaded_dynamic.manifest.id, 7);
        assert_eq!(loaded_dynamic.manifest.rank, 3);
        assert_eq!(loaded_dynamic.manifest.optimizer_step, 19);
        assert_eq!(loaded_dynamic.manifest.target_layers, vec![1, 3]);
        assert_eq!(loaded_dynamic.lora_a.len(), 2);
        assert_eq!(loaded_dynamic.adam_m.len(), 4);
        assert!(loaded_dynamic.adam_m[0].allclose(
            &Tensor::full([3, 8], 3.0, (tch::Kind::Float, tch::Device::Cpu)),
            1e-6,
            1e-6,
            false
        ));
    }

    #[test]
    fn old_dynamic_manifest_defaults_optimizer_step_to_zero() {
        let json = r#"{
            "id": 7,
            "rank": 3,
            "alpha": 6.0,
            "target_layers": [1],
            "target_modules": ["q_proj"],
            "parameter_count": 1,
            "optimizer_count": 2
        }"#;
        let manifest: DynamicAdapterManifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.optimizer_step, 0);
        assert!(manifest.shard_layouts.is_empty());
    }

    fn tp_topology(global_rank: usize, tp_size: usize) -> ParallelCheckpointManifest {
        ParallelCheckpointManifest::new(tp_size, global_rank, tp_size, 1, 1, 1, 1).unwrap()
    }

    fn tp_state(value: f64) -> (Vec<Tensor>, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>) {
        let a = Tensor::full([2, 3], value, (tch::Kind::Float, tch::Device::Cpu));
        let b = Tensor::full([4, 2], value + 1.0, (tch::Kind::Float, tch::Device::Cpu));
        let m = vec![a.zeros_like(), b.zeros_like()];
        let v = vec![a.ones_like(), b.ones_like()];
        (vec![a], vec![b], m, v)
    }

    fn tp_fixed_identities() -> [LoraSlotIdentity; 1] {
        [LoraSlotIdentity {
            index: 0,
            layer: 0,
            module: "in_proj_qkv".to_string(),
        }]
    }

    fn tp_dynamic_adapter(value: f64) -> DynamicAdapterCheckpoint {
        let a = Tensor::full([3, 5], value, (tch::Kind::Float, tch::Device::Cpu));
        let b = Tensor::full([7, 3], value + 1.0, (tch::Kind::Float, tch::Device::Cpu));
        DynamicAdapterCheckpoint {
            manifest: DynamicAdapterManifest {
                id: 9,
                rank: 6,
                alpha: 12.0,
                optimizer_step: 4,
                target_layers: vec![1],
                target_modules: vec!["q_proj".into()],
                shard_layouts: Vec::new(),
                parameter_count: 1,
                optimizer_count: 2,
            },
            lora_a: vec![a.shallow_clone()],
            lora_b: vec![b.shallow_clone()],
            adam_m: vec![a.zeros_like(), b.zeros_like()],
            adam_v: vec![a.ones_like(), b.ones_like()],
        }
    }

    #[test]
    fn tensor_parallel_ranks_use_distinct_paths_and_resume_same_topology() {
        let dir = tempfile::tempdir().unwrap();
        for global_rank in 0..2 {
            let topology = tp_topology(global_rank, 2);
            let (a, b, m, v) = tp_state(global_rank as f64 + 1.0);
            let dynamic = tp_dynamic_adapter(global_rank as f64 + 10.0);
            save_checkpoint_with_dynamic_for_topology(
                dir.path(),
                7,
                0.25,
                "Qwen/test",
                4,
                8.0,
                &a,
                &b,
                &m,
                &v,
                &[dynamic],
                &[],
                &tp_fixed_identities(),
                &topology,
            )
            .unwrap();
        }

        let rank0_dir = dir.path().join("rank-00000");
        let rank1_dir = dir.path().join("rank-00001");
        assert!(rank0_dir.join("manifest.json").is_file());
        assert!(rank1_dir.join("manifest.json").is_file());

        let topology = tp_topology(1, 2);
        let loaded = load_checkpoint_for_topology(dir.path(), &topology).unwrap();
        assert_eq!(loaded.manifest.format, TP_CHECKPOINT_FORMAT);
        assert_eq!(loaded.manifest.parallel.as_ref(), Some(&topology));
        assert_eq!(loaded.manifest.tensor_shards.len(), 12);
        assert_eq!(loaded.lora_a[0].double_value(&[0, 0]), 2.0);
        assert_eq!(loaded.dynamic_adapters.len(), 1);
        assert_eq!(
            loaded.dynamic_adapters[0].lora_a[0].double_value(&[0, 0]),
            11.0
        );

        let a_shard = loaded
            .manifest
            .tensor_shards
            .iter()
            .find(|shard| shard.state == "lora_a")
            .unwrap();
        assert_eq!(a_shard.global_lora_rank, 4);
        assert_eq!(a_shard.local_shape, vec![2, 3]);
        assert_eq!(a_shard.global_shape, vec![4, 3]);
        assert_eq!(a_shard.partition_axis, 0);
        assert_eq!(a_shard.global_offset, vec![2, 0]);
        assert_eq!(a_shard.replica_identity, "global-rank-1-tp-rank-1");

        let b_shard = loaded
            .manifest
            .tensor_shards
            .iter()
            .find(|shard| shard.state == "lora_b")
            .unwrap();
        assert_eq!(b_shard.global_shape, vec![4, 4]);
        assert_eq!(b_shard.partition_axis, 1);
        assert_eq!(b_shard.global_offset, vec![0, 2]);

        let dynamic_a_shard = loaded
            .manifest
            .tensor_shards
            .iter()
            .find(|shard| shard.state == "lora_a" && shard.adapter_id == Some(9))
            .unwrap();
        assert_eq!(dynamic_a_shard.global_lora_rank, 6);
        assert_eq!(dynamic_a_shard.global_shape, vec![6, 5]);
        assert_eq!(dynamic_a_shard.global_offset, vec![3, 0]);
    }

    #[test]
    fn tensor_parallel_resume_rejects_different_tp_size() {
        let dir = tempfile::tempdir().unwrap();
        let topology = tp_topology(0, 2);
        let (a, b, m, v) = tp_state(1.0);
        save_checkpoint_with_dynamic_for_topology(
            dir.path(),
            7,
            0.25,
            "Qwen/test",
            4,
            8.0,
            &a,
            &b,
            &m,
            &v,
            &[],
            &[],
            &tp_fixed_identities(),
            &topology,
        )
        .unwrap();

        let different_tp = tp_topology(0, 4);
        let error = load_checkpoint_for_topology(dir.path(), &different_tp)
            .err()
            .expect("different TP size must fail");
        assert!(error.to_string().contains("topology mismatch"));
    }

    #[test]
    fn tensor_parallel_resume_rejects_another_ranks_shard() {
        let dir = tempfile::tempdir().unwrap();
        let rank0 = tp_topology(0, 2);
        let (a, b, m, v) = tp_state(1.0);
        save_checkpoint_with_dynamic_for_topology(
            dir.path(),
            7,
            0.25,
            "Qwen/test",
            4,
            8.0,
            &a,
            &b,
            &m,
            &v,
            &[],
            &[],
            &tp_fixed_identities(),
            &rank0,
        )
        .unwrap();

        let rank1_dir = dir.path().join("rank-00001");
        std::fs::create_dir(&rank1_dir).unwrap();
        for file in [
            "manifest.json",
            "adapter.safetensors",
            "optimizer.safetensors",
        ] {
            std::fs::copy(
                dir.path().join("rank-00000").join(file),
                rank1_dir.join(file),
            )
            .unwrap();
        }

        let rank1 = tp_topology(1, 2);
        let error = load_checkpoint_for_topology(dir.path(), &rank1)
            .err()
            .expect("loading another rank's shard must fail");
        assert!(error.to_string().contains("topology mismatch"));
    }

    #[test]
    fn tensor_parallel_projection_layouts_record_global_tensor_geometry() {
        let dir = tempfile::tempdir().unwrap();
        let topology = tp_topology(1, 2);
        let column_a = Tensor::zeros([4, 8], (tch::Kind::Float, tch::Device::Cpu));
        let column_b = Tensor::zeros([8, 4], (tch::Kind::Float, tch::Device::Cpu));
        let row_a = Tensor::zeros([4, 4], (tch::Kind::Float, tch::Device::Cpu));
        let row_b = Tensor::zeros([8, 4], (tch::Kind::Float, tch::Device::Cpu));
        let lora_a = vec![column_a.shallow_clone(), row_a.shallow_clone()];
        let lora_b = vec![column_b.shallow_clone(), row_b.shallow_clone()];
        let adam_m = vec![
            column_a.zeros_like(),
            column_b.zeros_like(),
            row_a.zeros_like(),
            row_b.zeros_like(),
        ];
        let adam_v = vec![
            column_a.ones_like(),
            column_b.ones_like(),
            row_a.ones_like(),
            row_b.ones_like(),
        ];
        let layouts = [
            LoraTpShardLayout::ColumnParallel,
            LoraTpShardLayout::RowParallel,
        ];
        let identities = [
            LoraSlotIdentity {
                index: 0,
                layer: 0,
                module: "q_proj".to_string(),
            },
            LoraSlotIdentity {
                index: 3,
                layer: 0,
                module: "o_proj".to_string(),
            },
        ];

        save_checkpoint_with_dynamic_for_topology(
            dir.path(),
            3,
            0.5,
            "Qwen/test",
            4,
            8.0,
            &lora_a,
            &lora_b,
            &adam_m,
            &adam_v,
            &[],
            &layouts,
            &identities,
            &topology,
        )
        .unwrap();

        let loaded = load_checkpoint_for_topology(dir.path(), &topology).unwrap();
        assert_eq!(loaded.manifest.fixed_shard_layouts, layouts);
        assert_eq!(loaded.manifest.fixed_slot_identities, identities);
        let shard = |name: &str| {
            loaded
                .manifest
                .tensor_shards
                .iter()
                .find(|shard| shard.file == "adapter.safetensors" && shard.tensor_name == name)
                .unwrap()
        };

        let column_a_shard = shard("a_0");
        assert_eq!(column_a_shard.layout, LoraTpShardLayout::ColumnParallel);
        assert!(column_a_shard.replicated);
        assert_eq!(column_a_shard.global_shape, vec![4, 8]);
        assert_eq!(column_a_shard.global_offset, vec![0, 0]);
        assert_eq!(column_a_shard.replica_identity, "tp-replicated");

        let column_b_shard = shard("b_0");
        assert!(!column_b_shard.replicated);
        assert_eq!(column_b_shard.partition_axis, 0);
        assert_eq!(column_b_shard.global_shape, vec![16, 4]);
        assert_eq!(column_b_shard.global_offset, vec![8, 0]);

        let row_a_shard = shard("a_1");
        assert_eq!(row_a_shard.layout, LoraTpShardLayout::RowParallel);
        assert!(!row_a_shard.replicated);
        assert_eq!(row_a_shard.partition_axis, 1);
        assert_eq!(row_a_shard.global_shape, vec![4, 8]);
        assert_eq!(row_a_shard.global_offset, vec![0, 4]);

        let row_b_shard = shard("b_1");
        assert!(row_b_shard.replicated);
        assert_eq!(row_b_shard.global_shape, vec![8, 4]);
        assert_eq!(row_b_shard.global_offset, vec![0, 0]);
        assert_eq!(row_b_shard.replica_identity, "tp-replicated");
    }

    #[test]
    fn tensor_parallel_loader_accepts_legacy_latent_rank_v3_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let topology = tp_topology(0, 2);
        let (a, b, m, v) = tp_state(1.0);
        save_checkpoint_with_dynamic_for_topology(
            dir.path(),
            7,
            0.25,
            "Qwen/test",
            4,
            8.0,
            &a,
            &b,
            &m,
            &v,
            &[],
            &[],
            &tp_fixed_identities(),
            &topology,
        )
        .unwrap();

        let manifest_path = dir.path().join("rank-00000/manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
        let object = manifest.as_object_mut().unwrap();
        object.insert(
            "format".to_string(),
            serde_json::Value::String(LEGACY_TP_CHECKPOINT_FORMAT.to_string()),
        );
        object.remove("fixed_shard_layouts");
        object.remove("fixed_slot_identities");
        for shard in object
            .get_mut("tensor_shards")
            .unwrap()
            .as_array_mut()
            .unwrap()
        {
            let shard = shard.as_object_mut().unwrap();
            shard.remove("layout");
            shard.remove("replicated");
        }
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let loaded = load_checkpoint_for_topology(dir.path(), &topology).unwrap();
        assert_eq!(loaded.manifest.format, LEGACY_TP_CHECKPOINT_FORMAT);
        assert!(loaded.manifest.fixed_shard_layouts.is_empty());
        assert!(loaded
            .manifest
            .tensor_shards
            .iter()
            .all(|shard| shard.layout == LoraTpShardLayout::LatentRank && !shard.replicated));
    }

    fn resume_validation_manifest(format: &str) -> CheckpointManifest {
        CheckpointManifest {
            format: format.to_string(),
            step: 0,
            loss: 0.0,
            model_path: "Qwen/test".to_string(),
            lora_rank: 4,
            lora_alpha: 8.0,
            files: Vec::new(),
            dynamic_adapters: Vec::new(),
            parallel: None,
            tensor_shards: Vec::new(),
            fixed_shard_layouts: vec![LoraTpShardLayout::ColumnParallel],
            fixed_slot_identities: vec![LoraSlotIdentity {
                index: 0,
                layer: 0,
                module: "q_proj".to_string(),
            }],
        }
    }

    #[test]
    fn tensor_parallel_resume_rejects_fixed_slot_identity_mismatch() {
        let manifest = resume_validation_manifest(TP_CHECKPOINT_FORMAT);
        let expected = [LoraSlotIdentity {
            index: 1,
            layer: 0,
            module: "k_proj".to_string(),
        }];
        let error =
            validate_fixed_tp_resume(&manifest, &[LoraTpShardLayout::ColumnParallel], &expected)
                .unwrap_err();
        assert!(error.to_string().contains("slot identities"));
    }

    #[test]
    fn legacy_tensor_parallel_resume_rejects_projection_aware_layouts() {
        let manifest = resume_validation_manifest(LEGACY_TP_CHECKPOINT_FORMAT);
        let fixed_error =
            validate_fixed_tp_resume(&manifest, &[LoraTpShardLayout::ColumnParallel], &[])
                .unwrap_err();
        assert!(fixed_error.to_string().contains("fixed Q/K/V/O"));
        let dynamic_error =
            validate_dynamic_tp_resume(&manifest, 17, &[], &[LoraTpShardLayout::RowParallel])
                .unwrap_err();
        assert!(dynamic_error.to_string().contains("adapter 17"));
        validate_fixed_tp_resume(&manifest, &[LoraTpShardLayout::LatentRank], &[]).unwrap();
    }

    #[test]
    fn fixed_restore_mapping_supports_compact_and_legacy_positional_slots() {
        assert_eq!(
            fixed_restore_slot_indices(2, 2, &[1, 4], 7).unwrap(),
            vec![1, 4]
        );
        assert_eq!(
            fixed_restore_slot_indices(7, 7, &[1, 4], 7).unwrap(),
            (0..7).collect::<Vec<_>>()
        );
        assert!(fixed_restore_slot_indices(2, 1, &[1, 4], 7).is_err());
        assert!(fixed_restore_slot_indices(3, 3, &[1, 4], 7).is_err());
    }
}
