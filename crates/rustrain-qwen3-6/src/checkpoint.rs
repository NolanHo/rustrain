//! Qwen checkpoint save/load: LoRA parameters, Adam state, and parallel metadata.

use anyhow::{Context, Result, bail};
use rustrain_parallel::topology::{ParallelAxis, ParallelTopology, RankCoordinates};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::io::Read;
use std::path::{Path, PathBuf};
use tch::Tensor;

use crate::config::Qwen36RuntimeConfig;
use crate::lora::{
    Qwen36AdapterArtifact, Qwen36LoraConfig, Qwen36LoraTargetModule, native_lora_slots,
};

const TP_CHECKPOINT_FORMAT: &str = "rustrain-checkpoint-v5-parallel";
const PROJECTION_AWARE_TP_CHECKPOINT_FORMAT: &str = "rustrain-checkpoint-v4-tp";
const LEGACY_TP_CHECKPOINT_FORMAT: &str = "rustrain-checkpoint-v3-tp";
const RANK_RECEIPT_FORMAT: &str = "rustrain-checkpoint-rank-receipt-v1";
const RANK_RECEIPT_FILE: &str = "rank-receipt.json";
const RANK_RECEIPT_VERSION: u32 = 1;

pub fn is_legacy_tensor_parallel_checkpoint(manifest: &CheckpointManifest) -> bool {
    manifest.format == LEGACY_TP_CHECKPOINT_FORMAT
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoraTpShardLayout {
    #[default]
    LatentRank,
    ColumnParallel,
    /// Column-parallel projection whose global rows use flat
    /// `[Q_all | K_all | V_all]` storage while each rank stores packed
    /// `[Q_local | K_local | V_local]` rows.
    FlatQkvColumnParallel {
        q_rows: i64,
        k_rows: i64,
        v_rows: i64,
    },
    RowParallel,
    /// EP shards experts on axis 0 while TP stores packed local
    /// `[gate_local | up_local]` rows on the projection axis.
    RoutedExpertFusedGateUp,
    /// EP shards experts on axis 0 while TP shards the input projection axis.
    RoutedExpertDown,
}

pub fn lora_tp_shard_layout(
    module: Qwen36LoraTargetModule,
    config: &Qwen36RuntimeConfig,
) -> LoraTpShardLayout {
    match module {
        Qwen36LoraTargetModule::QProj
        | Qwen36LoraTargetModule::KProj
        | Qwen36LoraTargetModule::VProj
        | Qwen36LoraTargetModule::InProjZ
        | Qwen36LoraTargetModule::InProjA
        | Qwen36LoraTargetModule::InProjB
        | Qwen36LoraTargetModule::GateProj
        | Qwen36LoraTargetModule::UpProj
        | Qwen36LoraTargetModule::SharedGateProj
        | Qwen36LoraTargetModule::SharedUpProj => LoraTpShardLayout::ColumnParallel,
        Qwen36LoraTargetModule::InProjQkv => LoraTpShardLayout::FlatQkvColumnParallel {
            q_rows: config.linear_num_key_heads * config.linear_key_head_dim,
            k_rows: config.linear_num_key_heads * config.linear_key_head_dim,
            v_rows: config.linear_num_value_heads * config.linear_value_head_dim,
        },
        Qwen36LoraTargetModule::OProj
        | Qwen36LoraTargetModule::OutProj
        | Qwen36LoraTargetModule::DownProj
        | Qwen36LoraTargetModule::SharedDownProj => LoraTpShardLayout::RowParallel,
        Qwen36LoraTargetModule::ExpertsGateUpProj => LoraTpShardLayout::RoutedExpertFusedGateUp,
        Qwen36LoraTargetModule::ExpertsDownProj => LoraTpShardLayout::RoutedExpertDown,
    }
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
    #[serde(default = "default_rank_order")]
    pub rank_order: [ParallelAxis; 5],
    #[serde(default = "default_rank_coordinates")]
    pub coordinates: RankCoordinates,
}

fn default_rank_order() -> [ParallelAxis; 5] {
    ParallelTopology::new(1, 1, 1, 1, 1)
        .expect("singleton parallel topology is valid")
        .order()
}

const fn default_rank_coordinates() -> RankCoordinates {
    RankCoordinates::ZERO
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
        let topology = ParallelTopology::new(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            data_parallel_size,
            expert_model_parallel_size,
            context_parallel_size,
        )?;
        Self::from_topology(world_size, global_rank, &topology)
    }

    pub fn from_topology(
        world_size: usize,
        global_rank: usize,
        topology: &ParallelTopology,
    ) -> Result<Self> {
        topology.validate_world_size(world_size)?;
        let coordinates = topology.coordinates(global_rank)?;
        Ok(Self {
            world_size,
            tensor_model_parallel_size: topology.tensor_model_parallel_size(),
            pipeline_model_parallel_size: topology.pipeline_model_parallel_size(),
            data_parallel_size: topology.data_parallel_size(),
            expert_model_parallel_size: topology.expert_model_parallel_size(),
            context_parallel_size: topology.context_parallel_size(),
            global_rank,
            tensor_model_parallel_rank: coordinates.tensor,
            rank_order: topology.order(),
            coordinates,
        })
    }

    pub fn from_env() -> Result<Self> {
        let world_size = env_usize(&["WORLD_SIZE"], 1)?;
        let global_rank = env_usize(&["RANK"], 0)?;
        let topology = ParallelTopology::from_env_with_world_size(world_size)?;
        Self::from_topology(world_size, global_rank, &topology)
    }

    fn is_distributed(&self) -> bool {
        self.world_size > 1
    }

    fn legacy_fields_match(&self, other: &Self) -> bool {
        self.world_size == other.world_size
            && self.tensor_model_parallel_size == other.tensor_model_parallel_size
            && self.pipeline_model_parallel_size == other.pipeline_model_parallel_size
            && self.data_parallel_size == other.data_parallel_size
            && self.expert_model_parallel_size == other.expert_model_parallel_size
            && self.context_parallel_size == other.context_parallel_size
            && self.global_rank == other.global_rank
            && self.tensor_model_parallel_rank == other.tensor_model_parallel_rank
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
    /// Non-contiguous mappings from rank-local storage into the original
    /// global tensor. Empty for ordinary contiguous shards.
    #[serde(default)]
    pub segments: Vec<TensorShardSegmentManifest>,
    /// V5 supports independent placements on multiple parallel axes.
    #[serde(default)]
    pub placements: Vec<TensorShardPlacementManifest>,
    #[serde(default)]
    pub replicated_axes: Vec<ParallelAxis>,
    pub replica_identity: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShardPlacementManifest {
    pub parallel_axis: ParallelAxis,
    pub tensor_axis: usize,
    pub global_size: i64,
    pub local_size: i64,
    pub global_offset: i64,
    #[serde(default)]
    pub segments: Vec<TensorShardSegmentManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShardSegmentManifest {
    pub local_offset: i64,
    pub global_offset: i64,
    pub length: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub format: String,
    /// New v5 writers publish a compact receipt after this manifest. Readers
    /// use the receipt set for O(world-size) preflight metadata.
    #[serde(default)]
    pub rank_receipt_version: Option<u32>,
    /// Shared logical generation across every rank in one distributed save.
    #[serde(default)]
    pub checkpoint_generation: Option<String>,
    /// Session/transport progress. Dynamic-only requests advance this clock.
    pub step: u64,
    /// Fixed-adapter Adam bias-correction clock. Older checkpoints used
    /// `step` for both meanings, so a missing value falls back to `step`.
    #[serde(default)]
    pub fixed_optimizer_step: Option<u64>,
    pub loss: f64,
    pub model_path: String,
    pub lora_rank: i64,
    pub lora_alpha: f64,
    pub files: Vec<String>,
    /// Stable content digests bind an atomically published manifest to the
    /// tensor files it describes. Older checkpoints omit this field.
    #[serde(default)]
    pub file_digests: BTreeMap<String, String>,
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

impl CheckpointManifest {
    pub fn effective_fixed_optimizer_step(&self) -> u64 {
        self.fixed_optimizer_step.unwrap_or(self.step)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CheckpointRankReceipt {
    format: String,
    checkpoint_format: String,
    checkpoint_generation: String,
    global_rank: usize,
    parallel: ParallelCheckpointManifest,
    manifest_digest: String,
    generation_digest: String,
    shard_identity_digest: String,
    shard_count: usize,
    shard_identities_unique: bool,
    files: Vec<String>,
    file_digests: BTreeMap<String, String>,
    all_files_declared: bool,
    data_replica_metadata_complete: bool,
}

#[derive(Serialize)]
struct CheckpointGenerationIdentity<'a> {
    checkpoint_generation: &'a Option<String>,
    step: u64,
    fixed_optimizer_step: u64,
    model_path: &'a str,
    lora_rank: i64,
    lora_alpha_bits: u64,
    files: &'a [String],
    dynamic_adapters: &'a [DynamicAdapterManifest],
    fixed_shard_layouts: &'a [LoraTpShardLayout],
    fixed_slot_identities: &'a [LoraSlotIdentity],
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
                "legacy tensor-parallel v3 checkpoints cannot restore fixed projection-aware attention LoRA; use a v4 checkpoint or migrate the adapter from a merged artifact"
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
    expected_identities: &[LoraSlotIdentity],
) -> Result<()> {
    if is_legacy_tensor_parallel_checkpoint(manifest) {
        if expected_layouts
            .iter()
            .any(|layout| *layout != LoraTpShardLayout::LatentRank)
        {
            bail!(
                "legacy tensor-parallel v3 checkpoint adapter {adapter_id} contains projection-aware attention LoRA that cannot be restored; use a v4 checkpoint or migrate the adapter from a merged artifact"
            );
        }
        return Ok(());
    }
    if saved_layouts != expected_layouts {
        bail!(
            "dynamic adapter {adapter_id} shard layouts do not match the current runtime slots: checkpoint={saved_layouts:?}, runtime={expected_layouts:?}"
        );
    }
    if manifest.format == TP_CHECKPOINT_FORMAT {
        let saved = manifest
            .dynamic_adapters
            .iter()
            .find(|adapter| adapter.id == adapter_id)
            .with_context(|| format!("dynamic adapter {adapter_id} is missing from manifest"))?;
        if saved.slot_identities != expected_identities {
            bail!(
                "dynamic adapter {adapter_id} slot identities do not match the current runtime slots: checkpoint={:?}, runtime={expected_identities:?}",
                saved.slot_identities
            );
        }
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    #[serde(default)]
    pub slot_identities: Vec<LoraSlotIdentity>,
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

pub struct MergedLoraSlot {
    pub identity: LoraSlotIdentity,
    pub lora_a: Tensor,
    pub lora_b: Tensor,
}

pub struct MergedAdapterCheckpoint {
    pub model_path: String,
    pub step: u64,
    pub rank: i64,
    pub alpha: f64,
    pub optimizer_step: u64,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<String>,
    pub slots: Vec<MergedLoraSlot>,
}

struct TensorFragment {
    global_ranges: Vec<(i64, i64)>,
    tensor: Tensor,
}

/// Merge one fixed (`None`) or dynamic adapter from a complete v5 rank set.
/// This is an offline artifact operation, not part of the training hot path.
pub fn merge_distributed_adapter_checkpoint(
    root: &Path,
    adapter_id: Option<i64>,
) -> Result<MergedAdapterCheckpoint> {
    let rank_zero_path = rank_checkpoint_dir(root, 0).join("manifest.json");
    let rank_zero: CheckpointManifest = serde_json::from_str(
        &std::fs::read_to_string(&rank_zero_path)
            .with_context(|| format!("read {}", rank_zero_path.display()))?,
    )
    .with_context(|| format!("parse {}", rank_zero_path.display()))?;
    if rank_zero.format != TP_CHECKPOINT_FORMAT {
        bail!("distributed adapter merge requires a v5 checkpoint");
    }
    let topology = rank_zero
        .parallel
        .as_ref()
        .context("rank 0 checkpoint is missing parallel topology")?;
    let world_size = topology.world_size;
    let rank_order = topology
        .rank_order
        .iter()
        .map(|axis| axis.name())
        .collect::<Vec<_>>()
        .join("-");
    let expected_topology = ParallelTopology::with_order(
        topology.tensor_model_parallel_size,
        topology.pipeline_model_parallel_size,
        topology.data_parallel_size,
        topology.expert_model_parallel_size,
        topology.context_parallel_size,
        &rank_order,
    )?;
    expected_topology.validate_world_size(world_size)?;
    let (rank, alpha, optimizer_step, target_layers, target_modules, identities) = match adapter_id
    {
        None | Some(0) => (
            rank_zero.lora_rank,
            rank_zero.lora_alpha,
            rank_zero.effective_fixed_optimizer_step(),
            rank_zero
                .fixed_slot_identities
                .iter()
                .map(|identity| identity.layer)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            rank_zero
                .fixed_slot_identities
                .iter()
                .map(|identity| identity.module.clone())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            rank_zero.fixed_slot_identities.clone(),
        ),
        Some(id) => {
            let adapter = rank_zero
                .dynamic_adapters
                .iter()
                .find(|adapter| adapter.id == id)
                .with_context(|| format!("rank 0 checkpoint has no dynamic adapter {id}"))?;
            (
                adapter.rank,
                adapter.alpha,
                adapter.optimizer_step,
                adapter.target_layers.clone(),
                adapter.target_modules.clone(),
                adapter.slot_identities.clone(),
            )
        }
    };
    if identities.is_empty() {
        bail!("distributed adapter merge requires explicit slot identities");
    }

    let selected_id = adapter_id.filter(|id| *id != 0);
    let mut fragments: BTreeMap<(String, String), (Vec<i64>, Vec<TensorFragment>)> =
        BTreeMap::new();
    let mut seen_ranks = std::collections::BTreeSet::new();
    for global_rank in 0..world_size {
        let rank_dir = rank_checkpoint_dir(root, global_rank);
        let manifest_path = rank_dir.join("manifest.json");
        let manifest: CheckpointManifest = serde_json::from_str(
            &std::fs::read_to_string(&manifest_path)
                .with_context(|| format!("read {}", manifest_path.display()))?,
        )
        .with_context(|| format!("parse {}", manifest_path.display()))?;
        validate_manifest_file_digests(&rank_dir, &manifest)?;
        let parallel = manifest
            .parallel
            .as_ref()
            .with_context(|| format!("rank {global_rank} checkpoint is missing topology"))?;
        if manifest.format != TP_CHECKPOINT_FORMAT
            || parallel.world_size != world_size
            || parallel.global_rank != global_rank
            || parallel.rank_order != topology.rank_order
            || parallel.tensor_model_parallel_size != topology.tensor_model_parallel_size
            || parallel.pipeline_model_parallel_size != topology.pipeline_model_parallel_size
            || parallel.data_parallel_size != topology.data_parallel_size
            || parallel.expert_model_parallel_size != topology.expert_model_parallel_size
            || parallel.context_parallel_size != topology.context_parallel_size
            || parallel.coordinates != expected_topology.coordinates(global_rank)?
            || manifest.step != rank_zero.step
            || manifest.effective_fixed_optimizer_step()
                != rank_zero.effective_fixed_optimizer_step()
            || manifest.model_path != rank_zero.model_path
        {
            bail!("rank {global_rank} checkpoint metadata is inconsistent with rank 0");
        }
        validate_checkpoint_generation(&rank_zero, &manifest, global_rank)?;
        if !seen_ranks.insert((
            parallel.global_rank,
            parallel.coordinates.tensor,
            parallel.coordinates.pipeline,
            parallel.coordinates.data,
            parallel.coordinates.expert,
            parallel.coordinates.context,
        )) {
            bail!("duplicate distributed checkpoint coordinates at rank {global_rank}");
        }
        match selected_id {
            None => {
                if manifest.lora_rank != rank_zero.lora_rank
                    || manifest.lora_alpha != rank_zero.lora_alpha
                    || manifest.fixed_shard_layouts != rank_zero.fixed_shard_layouts
                    || manifest.fixed_slot_identities != rank_zero.fixed_slot_identities
                {
                    bail!("rank {global_rank} fixed adapter metadata differs from rank 0");
                }
            }
            Some(id) => {
                let expected_adapter = rank_zero
                    .dynamic_adapters
                    .iter()
                    .find(|adapter| adapter.id == id)
                    .expect("selected dynamic adapter was validated on rank 0");
                let adapter = manifest
                    .dynamic_adapters
                    .iter()
                    .find(|adapter| adapter.id == id)
                    .with_context(|| format!("rank {global_rank} has no dynamic adapter {id}"))?;
                if adapter != expected_adapter {
                    bail!("rank {global_rank} dynamic adapter {id} metadata differs from rank 0");
                }
            }
        }
        let tensors = read_named_tensors(&rank_dir.join("adapter.safetensors"))?;
        for shard in manifest.tensor_shards.iter().filter(|shard| {
            shard.file == "adapter.safetensors"
                && (shard.state == "lora_a" || shard.state == "lora_b")
                && shard.adapter_id == selected_id
        }) {
            let tensor = tensors.get(&shard.tensor_name).with_context(|| {
                format!(
                    "rank {global_rank} adapter is missing tensor {}",
                    shard.tensor_name
                )
            })?;
            if tensor.size() != shard.local_shape {
                bail!(
                    "rank {global_rank} tensor {} shape {:?} does not match manifest {:?}",
                    shard.tensor_name,
                    tensor.size(),
                    shard.local_shape
                );
            }
            let key = (shard.state.clone(), shard.tensor_name.clone());
            let entry = fragments
                .entry(key)
                .or_insert_with(|| (shard.global_shape.clone(), Vec::new()));
            if entry.0 != shard.global_shape {
                bail!("global shape mismatch for tensor {}", shard.tensor_name);
            }
            entry.1.extend(tensor_fragments(shard, tensor)?);
        }
    }
    if seen_ranks.len() != world_size {
        bail!("distributed checkpoint rank set is incomplete");
    }

    let merged = fragments
        .into_iter()
        .map(|(key, (shape, fragments))| Ok((key, merge_tensor_fragments(&shape, fragments)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let prefix = selected_id
        .map(|id| format!("dynamic_{id}_"))
        .unwrap_or_default();
    let mut slots = Vec::with_capacity(identities.len());
    for (index, identity) in identities.into_iter().enumerate() {
        let a_name = format!("{prefix}a_{index}");
        let b_name = format!("{prefix}b_{index}");
        let lora_a = merged
            .get(&("lora_a".to_string(), a_name.clone()))
            .with_context(|| format!("merged adapter is missing {a_name}"))?
            .shallow_clone();
        let lora_b = merged
            .get(&("lora_b".to_string(), b_name.clone()))
            .with_context(|| format!("merged adapter is missing {b_name}"))?
            .shallow_clone();
        slots.push(MergedLoraSlot {
            identity,
            lora_a,
            lora_b,
        });
    }
    Ok(MergedAdapterCheckpoint {
        model_path: rank_zero.model_path,
        step: rank_zero.step,
        rank,
        alpha,
        optimizer_step,
        target_layers,
        target_modules,
        slots,
    })
}

fn tensor_fragments(shard: &TensorShardManifest, tensor: &Tensor) -> Result<Vec<TensorFragment>> {
    let dimensions = shard.local_shape.len();
    let mut mappings = shard
        .local_shape
        .iter()
        .map(|size| vec![(0, 0, *size)])
        .collect::<Vec<_>>();
    let mut placed_axes = std::collections::BTreeSet::new();
    for placement in &shard.placements {
        if placement.tensor_axis >= dimensions || !placed_axes.insert(placement.tensor_axis) {
            bail!(
                "invalid or duplicate placement axis for {}",
                shard.tensor_name
            );
        }
        mappings[placement.tensor_axis] = if placement.segments.is_empty() {
            vec![(0, placement.global_offset, placement.local_size)]
        } else {
            placement
                .segments
                .iter()
                .map(|segment| (segment.local_offset, segment.global_offset, segment.length))
                .collect()
        };
    }
    let mut selections = Vec::new();
    cartesian_axis_mappings(&mappings, 0, &mut Vec::new(), &mut selections);
    let mut fragments = Vec::with_capacity(selections.len());
    for selection in selections {
        let mut local = tensor.shallow_clone();
        for (axis, (local_offset, _, length)) in selection.iter().copied().enumerate() {
            if local_offset < 0 || length <= 0 || local_offset + length > shard.local_shape[axis] {
                bail!("local placement is out of bounds for {}", shard.tensor_name);
            }
            local = local.narrow(axis as i64, local_offset, length);
        }
        let global_ranges = selection
            .iter()
            .enumerate()
            .map(|(axis, (_, global_offset, length))| {
                if *global_offset < 0
                    || *length <= 0
                    || *global_offset + *length > shard.global_shape[axis]
                {
                    bail!(
                        "global placement is out of bounds for {}",
                        shard.tensor_name
                    );
                }
                Ok((*global_offset, *length))
            })
            .collect::<Result<Vec<_>>>()?;
        fragments.push(TensorFragment {
            global_ranges,
            tensor: local,
        });
    }
    Ok(fragments)
}

fn cartesian_axis_mappings(
    mappings: &[Vec<(i64, i64, i64)>],
    axis: usize,
    current: &mut Vec<(i64, i64, i64)>,
    output: &mut Vec<Vec<(i64, i64, i64)>>,
) {
    if axis == mappings.len() {
        output.push(current.clone());
        return;
    }
    for mapping in &mappings[axis] {
        current.push(*mapping);
        cartesian_axis_mappings(mappings, axis + 1, current, output);
        current.pop();
    }
}

fn merge_tensor_fragments(shape: &[i64], fragments: Vec<TensorFragment>) -> Result<Tensor> {
    let mut unique: BTreeMap<Vec<(i64, i64)>, Tensor> = BTreeMap::new();
    for fragment in fragments {
        if let Some(existing) = unique.get(&fragment.global_ranges) {
            if !existing.allclose(&fragment.tensor, 0.0, 0.0, false) {
                bail!("replicated distributed checkpoint fragments differ");
            }
        } else {
            unique.insert(fragment.global_ranges, fragment.tensor);
        }
    }
    let regions = unique.keys().collect::<Vec<_>>();
    for left in 0..regions.len() {
        for right in (left + 1)..regions.len() {
            let overlaps = regions[left].iter().zip(regions[right].iter()).all(
                |(&(left_start, left_len), &(right_start, right_len))| {
                    left_start < right_start + right_len && right_start < left_start + left_len
                },
            );
            if overlaps {
                bail!("distributed checkpoint fragments overlap");
            }
        }
    }
    let covered = unique
        .keys()
        .map(|ranges| ranges.iter().map(|(_, length)| *length).product::<i64>())
        .sum::<i64>();
    let expected = shape.iter().product::<i64>();
    if covered != expected {
        bail!("distributed checkpoint fragments cover {covered} elements, expected {expected}");
    }
    let output = Tensor::zeros(shape, (tch::Kind::Float, tch::Device::Cpu));
    for (ranges, source) in unique {
        let mut target = output.shallow_clone();
        for (axis, (offset, length)) in ranges.into_iter().enumerate() {
            target = target.narrow(axis as i64, offset, length);
        }
        target.copy_(&source.to_kind(tch::Kind::Float));
    }
    Ok(output)
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
        if lora_a.is_empty() { 0 } else { step },
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
    save_checkpoint_with_dynamic_and_fixed_step_for_topology(
        dir,
        step,
        if lora_a.is_empty() { 0 } else { step },
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
        parallel,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn save_checkpoint_with_dynamic_and_fixed_step_for_topology(
    dir: &Path,
    step: u64,
    fixed_optimizer_step: u64,
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
    if parallel.is_distributed() {
        bail!(
            "distributed checkpoint save requires an explicit unique generation from its coordinator"
        );
    }
    save_checkpoint_with_dynamic_and_fixed_step_for_topology_generation(
        dir,
        step,
        fixed_optimizer_step,
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
        parallel,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn save_checkpoint_with_dynamic_and_fixed_step_for_topology_generation(
    dir: &Path,
    step: u64,
    fixed_optimizer_step: u64,
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
    checkpoint_generation: Option<&str>,
) -> Result<()> {
    if !parallel.is_distributed() {
        return save_checkpoint_with_dynamic_at(
            dir,
            step,
            fixed_optimizer_step,
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
            None,
        );
    }
    let checkpoint_generation = checkpoint_generation
        .filter(|generation| !generation.is_empty())
        .context("distributed checkpoint generation must be non-empty")?;
    validate_checkpoint_generation_value(checkpoint_generation)?;
    let rank_dir = rank_checkpoint_dir(dir, parallel.global_rank);
    let manifest_path = rank_dir.join("manifest.json");
    if manifest_path.is_file() {
        let previous = read_checkpoint_manifest(&manifest_path)?;
        if previous.checkpoint_generation.as_deref() == Some(checkpoint_generation) {
            bail!(
                "distributed checkpoint generation {checkpoint_generation} has already been used for rank {}",
                parallel.global_rank
            );
        }
    }
    save_checkpoint_with_dynamic_at(
        &rank_dir,
        step,
        fixed_optimizer_step,
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
        Some(checkpoint_generation),
    )
}

fn validate_checkpoint_generation_value(generation: &str) -> Result<()> {
    if generation.len() > 256
        || !generation
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!(
            "distributed checkpoint generation must be at most 256 path-safe ASCII characters"
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn save_checkpoint_with_dynamic_at(
    dir: &Path,
    step: u64,
    fixed_optimizer_step: u64,
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
    checkpoint_generation: Option<&str>,
) -> Result<()> {
    validate_tensor_counts(
        lora_a,
        lora_b,
        adam_m,
        adam_v,
        dynamic_adapters,
        parallel.is_some(),
    )?;
    if fixed_optimizer_step > 0 && (adam_m.is_empty() || adam_v.is_empty()) {
        bail!("fixed optimizer step {fixed_optimizer_step} requires Adam state");
    }
    for adapter in dynamic_adapters {
        if adapter.manifest.optimizer_step > 0
            && (adapter.adam_m.is_empty() || adapter.adam_v.is_empty())
        {
            bail!(
                "dynamic adapter {} optimizer step {} requires Adam state",
                adapter.manifest.id,
                adapter.manifest.optimizer_step
            );
        }
    }
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

    let file_digests = [
        ("adapter.safetensors".to_string(), stable_file_digest(&adapter_path)?),
        (
            "optimizer.safetensors".to_string(),
            stable_file_digest(&optimizer_path)?,
        ),
    ]
    .into_iter()
    .collect();

    // Write manifest
    let manifest = CheckpointManifest {
        format: if parallel.is_some() {
            TP_CHECKPOINT_FORMAT.to_string()
        } else if dynamic_manifests.is_empty() {
            "rustrain-checkpoint-v1".to_string()
        } else {
            "rustrain-checkpoint-v2".to_string()
        },
        rank_receipt_version: parallel.map(|_| RANK_RECEIPT_VERSION),
        checkpoint_generation: checkpoint_generation.map(str::to_string),
        step,
        fixed_optimizer_step: Some(fixed_optimizer_step),
        loss,
        model_path: model_path.to_string(),
        lora_rank,
        lora_alpha,
        files: vec!["adapter.safetensors".into(), "optimizer.safetensors".into()],
        file_digests,
        dynamic_adapters: dynamic_manifests,
        parallel: parallel.cloned(),
        tensor_shards,
        fixed_shard_layouts: fixed_shard_layouts.to_vec(),
        fixed_slot_identities: fixed_slot_identities.to_vec(),
    };
    let manifest_path = dir.join("manifest.json");
    let manifest_contents = serde_json::to_vec_pretty(&manifest)?;
    write_atomic(
        &manifest_path,
        &manifest_contents,
    )
    .with_context(|| "write manifest.json")?;
    if parallel.is_some() {
        let receipt = checkpoint_rank_receipt(&manifest, &manifest_contents)?;
        write_atomic(
            &dir.join(RANK_RECEIPT_FILE),
            &serde_json::to_vec_pretty(&receipt)?,
        )
        .with_context(|| format!("write {RANK_RECEIPT_FILE}"))?;
    }

    tracing::info!(
        step,
        loss,
        path = dir.display().to_string(),
        "checkpoint saved"
    );
    Ok(())
}

pub fn write_atomic(path: &Path, contents: &[u8]) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("atomic output path has no UTF-8 file name")?;
    let partial_path = path.with_file_name(format!(".{file_name}.partial"));
    std::fs::write(&partial_path, contents)
        .with_context(|| format!("write partial file {}", partial_path.display()))?;
    std::fs::rename(&partial_path, path).with_context(|| {
        format!(
            "publish atomic file {} as {}",
            partial_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn stable_file_digest(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("open checkpoint content {}", path.display()))?;
    let mut first = 0xcbf29ce484222325_u64;
    let mut second = 0x84222325cbf29ce4_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("hash checkpoint content {}", path.display()))?;
        if count == 0 {
            break;
        }
        for &byte in &buffer[..count] {
            first ^= u64::from(byte);
            first = first.wrapping_mul(0x100000001b3);
            second ^= u64::from(byte).wrapping_add(0x9d);
            second = second.wrapping_mul(0x100000001b3 ^ 0x517cc1b727220a95);
        }
    }
    Ok(format!("fnv128-v1:{first:016x}{second:016x}"))
}

fn stable_bytes_digest(contents: &[u8]) -> String {
    let mut first = 0xcbf29ce484222325_u64;
    let mut second = 0x84222325cbf29ce4_u64;
    for &byte in contents {
        first ^= u64::from(byte);
        first = first.wrapping_mul(0x100000001b3);
        second ^= u64::from(byte).wrapping_add(0x9d);
        second = second.wrapping_mul(0x100000001b3 ^ 0x517cc1b727220a95);
    }
    format!("fnv128-v1:{first:016x}{second:016x}")
}

fn checkpoint_generation_digest(manifest: &CheckpointManifest) -> Result<String> {
    Ok(stable_bytes_digest(&serde_json::to_vec(
        &CheckpointGenerationIdentity {
            checkpoint_generation: &manifest.checkpoint_generation,
            step: manifest.step,
            fixed_optimizer_step: manifest.effective_fixed_optimizer_step(),
            model_path: &manifest.model_path,
            lora_rank: manifest.lora_rank,
            lora_alpha_bits: manifest.lora_alpha.to_bits(),
            files: &manifest.files,
            dynamic_adapters: &manifest.dynamic_adapters,
            fixed_shard_layouts: &manifest.fixed_shard_layouts,
            fixed_slot_identities: &manifest.fixed_slot_identities,
        },
    )?))
}

fn checkpoint_rank_receipt(
    manifest: &CheckpointManifest,
    manifest_contents: &[u8],
) -> Result<CheckpointRankReceipt> {
    let parallel = manifest
        .parallel
        .clone()
        .context("distributed checkpoint receipt requires topology metadata")?;
    let checkpoint_generation = manifest
        .checkpoint_generation
        .clone()
        .filter(|generation| !generation.is_empty())
        .context("distributed checkpoint receipt requires a non-empty generation")?;
    let shard_identities = manifest
        .tensor_shards
        .iter()
        .map(tensor_shard_identity)
        .collect::<std::collections::BTreeSet<_>>();
    let shard_identity_digest = stable_bytes_digest(&serde_json::to_vec(&shard_identities)?);
    let all_files_declared = manifest.file_digests.len() == manifest.files.len()
        && manifest
            .files
            .iter()
            .all(|file| manifest.file_digests.contains_key(file))
        && manifest
            .tensor_shards
            .iter()
            .all(|shard| manifest.files.iter().any(|file| file == &shard.file));
    let data_replica_metadata_complete = parallel.data_parallel_size <= 1
        || manifest
            .tensor_shards
            .iter()
            .all(|shard| shard.replicated_axes.contains(&ParallelAxis::Data));
    Ok(CheckpointRankReceipt {
        format: RANK_RECEIPT_FORMAT.to_string(),
        checkpoint_format: manifest.format.clone(),
        checkpoint_generation,
        global_rank: parallel.global_rank,
        parallel,
        manifest_digest: stable_bytes_digest(manifest_contents),
        generation_digest: checkpoint_generation_digest(manifest)?,
        shard_identity_digest,
        shard_count: manifest.tensor_shards.len(),
        shard_identities_unique: shard_identities.len() == manifest.tensor_shards.len(),
        files: manifest.files.clone(),
        file_digests: manifest.file_digests.clone(),
        all_files_declared,
        data_replica_metadata_complete,
    })
}

pub fn export_distributed_adapter_checkpoint(
    final_path: &Path,
    generation: &str,
    parallel: &ParallelCheckpointManifest,
    adapter_id: Option<i64>,
    save_rank: impl FnOnce(&Path) -> Result<()>,
) -> Result<usize> {
    if parallel.world_size <= 1 {
        bail!("distributed adapter export requires more than one rank");
    }
    if final_path.extension().is_some() {
        bail!(
            "distributed adapter export path must be a directory without a file extension: {}",
            final_path.display()
        );
    }
    if generation.is_empty()
        || !generation
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!("distributed adapter generation contains invalid path characters");
    }
    if final_path.exists() {
        bail!(
            "adapter export destination already exists: {}",
            final_path.display()
        );
    }
    let attempt = generation.to_string();
    let staging = coordinate_export_attempt(final_path, generation, parallel)?;
    let result = (|| -> Result<usize> {
        save_rank(&staging)?;
        wait_for_rank_manifests(&staging, parallel.world_size)?;
        if parallel.global_rank == 0 {
            if final_path.exists() {
                bail!(
                    "adapter export destination already exists: {}",
                    final_path.display()
                );
            }
            let merged = merge_distributed_adapter_checkpoint(&staging, adapter_id)?;
            let artifact = merged_adapter_artifact(merged)?;
            let count = artifact.tensors.len();
            let partial_path = sibling_with_suffix(final_path, &format!(".partial-{generation}"))?;
            if partial_path.exists() {
                bail!(
                    "adapter export partial destination already exists: {}",
                    partial_path.display()
                );
            }
            artifact.save(&partial_path)?;
            write_atomic(
                &partial_path.join(".rustrain-export-completed.json"),
                &serde_json::to_vec_pretty(&AdapterExportCompletion {
                    attempt: attempt.clone(),
                    tensor_count: count,
                })?,
            )?;
            std::fs::rename(&partial_path, final_path).with_context(|| {
                format!(
                    "publish adapter {} to {}",
                    partial_path.display(),
                    final_path.display()
                )
            })?;
            Ok(count)
        } else {
            wait_for_export_completion(final_path, &attempt, &staging)
        }
    })();
    if let Err(error) = &result {
        let errors = staging.join("errors");
        let _ = std::fs::create_dir_all(&errors);
        let _ = write_atomic(
            &errors.join(format!("rank-{:05}.txt", parallel.global_rank)),
            error.to_string().as_bytes(),
        );
    }
    result
}

#[derive(Debug, Serialize, Deserialize)]
struct AdapterExportCompletion {
    attempt: String,
    tensor_count: usize,
}

fn coordinate_export_attempt(
    final_path: &Path,
    generation: &str,
    parallel: &ParallelCheckpointManifest,
) -> Result<PathBuf> {
    let staging = sibling_with_suffix(final_path, &format!(".rustrain-shards-{generation}"))?;
    if parallel.global_rank == 0 {
        std::fs::create_dir(&staging).with_context(|| {
            format!(
                "create unique distributed export staging {}; attempt IDs cannot be reused",
                staging.display()
            )
        })?;
        let result = write_atomic(&staging.join("READY"), generation.as_bytes());
        if let Err(error) = &result {
            let errors = staging.join("errors");
            let _ = std::fs::create_dir_all(&errors);
            let _ = write_atomic(&errors.join("rank-00000.txt"), error.to_string().as_bytes());
        }
        result?;
    } else {
        wait_for_distributed_export(&staging, || {
            read_marker(&staging.join("READY")).as_deref() == Some(generation)
        })?;
    }
    Ok(staging)
}

fn read_marker(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub fn merged_adapter_artifact(merged: MergedAdapterCheckpoint) -> Result<Qwen36AdapterArtifact> {
    let runtime_config = crate::config::read_qwen36_runtime_config(Path::new(&merged.model_path))?;
    let target_modules = merged
        .target_modules
        .iter()
        .map(|module| Qwen36LoraTargetModule::parse(module))
        .collect::<Result<Vec<_>>>()?;
    let lora_config = Qwen36LoraConfig {
        rank: merged.rank,
        alpha: merged.alpha,
        target_layers: merged.target_layers.clone(),
        target_modules,
    };
    let mut by_index = merged
        .slots
        .into_iter()
        .map(|slot| (slot.identity.index, slot))
        .collect::<BTreeMap<_, _>>();
    let mut exported = Vec::new();
    for slot in native_lora_slots(&runtime_config, &lora_config) {
        if !slot.active {
            let placeholder = Tensor::zeros([], (tch::Kind::Float, tch::Device::Cpu));
            exported.push((placeholder.shallow_clone(), placeholder));
            continue;
        }
        let merged_slot = by_index
            .remove(&slot.index)
            .with_context(|| format!("merged adapter is missing native slot {}", slot.index))?;
        if merged_slot.identity.layer != slot.layer
            || merged_slot.identity.module != slot.module.cpp_name()
        {
            bail!(
                "merged adapter slot {} identity is inconsistent",
                slot.index
            );
        }
        exported.push((merged_slot.lora_a, merged_slot.lora_b));
    }
    if !by_index.is_empty() {
        bail!("merged adapter contains slots not present in the runtime model");
    }
    Qwen36AdapterArtifact::from_native_exports(
        &merged.model_path,
        "qwen3_hybrid_lora_sft",
        Some(Path::new(&merged.model_path)),
        &runtime_config,
        &lora_config,
        exported,
    )
}

fn sibling_with_suffix(path: &Path, suffix: &str) -> Result<PathBuf> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("adapter export path has no UTF-8 file name")?;
    Ok(path.with_file_name(format!("{file_name}{suffix}")))
}

fn wait_for_rank_manifests(root: &Path, world_size: usize) -> Result<()> {
    wait_for_distributed_export(root, || rank_manifests_ready(root, world_size))
}

fn rank_manifests_ready(root: &Path, world_size: usize) -> bool {
    (0..world_size).all(|rank| {
        let rank_dir = root.join(format!("rank-{rank:05}"));
        rank_dir.join("manifest.json").is_file() && rank_dir.join(RANK_RECEIPT_FILE).is_file()
    })
}

fn wait_for_export_completion(final_path: &Path, attempt: &str, staging: &Path) -> Result<usize> {
    let completion_path = final_path.join(".rustrain-export-completed.json");
    wait_for_distributed_export(staging, || {
        std::fs::read(&completion_path)
            .ok()
            .and_then(|contents| serde_json::from_slice::<AdapterExportCompletion>(&contents).ok())
            .is_some_and(|completion| completion.attempt == attempt)
    })?;
    let completion: AdapterExportCompletion = serde_json::from_slice(
        &std::fs::read(&completion_path)
            .with_context(|| format!("read {}", completion_path.display()))?,
    )
    .with_context(|| format!("parse {}", completion_path.display()))?;
    if completion.attempt != attempt {
        bail!("distributed adapter completion belongs to another attempt");
    }
    Ok(completion.tensor_count)
}

fn wait_for_distributed_export(root: &Path, ready: impl Fn() -> bool) -> Result<()> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        let errors = root.join("errors");
        if errors.is_dir() {
            let mut entries = std::fs::read_dir(&errors)?.collect::<std::io::Result<Vec<_>>>()?;
            entries.sort_by_key(|entry| entry.file_name());
            if let Some(error) = entries.first() {
                let message = std::fs::read_to_string(error.path()).unwrap_or_else(|read_error| {
                    format!("failed to read distributed export error: {read_error}")
                });
                bail!("distributed adapter export failed: {message}");
            }
        }
        if ready() {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            bail!(
                "timed out waiting for distributed adapter export at {}",
                root.display()
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// Load checkpoint from a directory.
pub fn load_checkpoint(dir: &Path) -> Result<CheckpointData> {
    load_checkpoint_at(dir, None)
}

pub fn load_checkpoint_for_topology(
    dir: &Path,
    parallel: &ParallelCheckpointManifest,
) -> Result<CheckpointData> {
    if !parallel.is_distributed() {
        return load_checkpoint(dir);
    }
    let rank_dir = rank_checkpoint_dir(dir, parallel.global_rank);
    preflight_distributed_checkpoint_set(dir, &rank_dir, parallel)?;
    load_checkpoint_at(&rank_dir, Some(parallel))
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DataReplicaKey {
    file: String,
    tensor_name: String,
    state: String,
    adapter_id: Option<i64>,
    tensor_rank: usize,
    pipeline_rank: usize,
    expert_rank: usize,
    context_rank: usize,
}

struct DataReplica {
    global_rank: usize,
    shard: TensorShardManifest,
    tensor: Tensor,
}

struct DataReplicaGroup {
    data_ranks: std::collections::BTreeSet<usize>,
    reference: DataReplica,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DataReplicaFileKey {
    file: String,
    tensor_rank: usize,
    pipeline_rank: usize,
    expert_rank: usize,
    context_rank: usize,
}

struct DataReplicaFileGroup {
    data_ranks: std::collections::BTreeSet<usize>,
    global_rank: usize,
    digest: String,
}

fn read_checkpoint_manifest(path: &Path) -> Result<CheckpointManifest> {
    serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))
}

fn read_checkpoint_rank_receipt(path: &Path) -> Result<CheckpointRankReceipt> {
    serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))
}

fn preflight_distributed_receipts(
    root: &Path,
    requested_rank_dir: &Path,
    requested_manifest: &CheckpointManifest,
    expected: &ParallelCheckpointManifest,
    topology: &ParallelTopology,
) -> Result<()> {
    let requested_manifest_path = requested_rank_dir.join("manifest.json");
    let requested_manifest_contents = std::fs::read(&requested_manifest_path)
        .with_context(|| format!("read {}", requested_manifest_path.display()))?;
    let expected_local_receipt =
        checkpoint_rank_receipt(requested_manifest, &requested_manifest_contents)?;
    let mut baseline: Option<CheckpointRankReceipt> = None;
    let mut replica_files: BTreeMap<DataReplicaFileKey, DataReplicaFileGroup> = BTreeMap::new();

    for global_rank in 0..expected.world_size {
        let rank_dir = rank_checkpoint_dir(root, global_rank);
        let manifest_path = rank_dir.join("manifest.json");
        if !manifest_path.is_file() {
            bail!(
                "distributed checkpoint is missing rank {global_rank} manifest {}",
                manifest_path.display()
            );
        }
        let receipt_path = rank_dir.join(RANK_RECEIPT_FILE);
        let receipt = read_checkpoint_rank_receipt(&receipt_path)?;
        let expected_parallel =
            ParallelCheckpointManifest::from_topology(expected.world_size, global_rank, topology)?;
        if receipt.format != RANK_RECEIPT_FORMAT
            || receipt.checkpoint_format != TP_CHECKPOINT_FORMAT
            || receipt.global_rank != global_rank
            || receipt.parallel != expected_parallel
        {
            bail!(
                "distributed checkpoint topology mismatch: rank {global_rank} receipt format or topology is inconsistent"
            );
        }
        if !receipt.shard_identities_unique {
            bail!("rank {global_rank} checkpoint contains duplicate tensor shard identities");
        }
        if !receipt.all_files_declared {
            bail!("rank {global_rank} checkpoint receipt has incomplete file declarations");
        }
        if !receipt.data_replica_metadata_complete {
            bail!(
                "rank {global_rank} checkpoint is missing data-parallel replica metadata"
            );
        }
        if receipt.file_digests.len() != receipt.files.len()
            || receipt
                .files
                .iter()
                .any(|file| !receipt.file_digests.contains_key(file))
        {
            bail!("rank {global_rank} checkpoint receipt is missing file content digests");
        }
        if global_rank == expected.global_rank && receipt != expected_local_receipt {
            bail!("local checkpoint manifest does not match its compact rank receipt");
        }

        if let Some(baseline) = &baseline {
            if receipt.checkpoint_generation != baseline.checkpoint_generation
                || receipt.generation_digest != baseline.generation_digest
                || receipt.shard_identity_digest != baseline.shard_identity_digest
                || receipt.shard_count != baseline.shard_count
                || receipt.files != baseline.files
            {
                bail!("rank {global_rank} receipt belongs to a different checkpoint generation");
            }
        } else {
            baseline = Some(receipt.clone());
        }

        if expected.data_parallel_size <= 1 {
            continue;
        }
        for file in &receipt.files {
            let digest = receipt
                .file_digests
                .get(file)
                .context("validated checkpoint receipt digest disappeared")?;
            let coordinates = &receipt.parallel.coordinates;
            let key = DataReplicaFileKey {
                file: file.clone(),
                tensor_rank: coordinates.tensor,
                pipeline_rank: coordinates.pipeline,
                expert_rank: coordinates.expert,
                context_rank: coordinates.context,
            };
            match replica_files.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(DataReplicaFileGroup {
                        data_ranks: [coordinates.data].into_iter().collect(),
                        global_rank,
                        digest: digest.clone(),
                    });
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let key = entry.key().clone();
                    let group = entry.get_mut();
                    if !group.data_ranks.insert(coordinates.data) {
                        bail!("duplicate data-parallel checkpoint replica at rank {global_rank}");
                    }
                    if group.digest != *digest {
                        bail!(
                            "data-parallel replica content digest differs for {} between ranks {} and {}",
                            key.file,
                            group.global_rank,
                            global_rank
                        );
                    }
                }
            }
        }
    }

    for (key, group) in replica_files {
        if group.data_ranks.len() != expected.data_parallel_size {
            bail!(
                "data-parallel checkpoint replica set for {} has {} ranks, expected {}",
                key.file,
                group.data_ranks.len(),
                expected.data_parallel_size
            );
        }
    }
    Ok(())
}

/// Validate a complete v5 rank set before restoring rank-local state.
///
/// New v5 manifests bind tensor files to content digests. Receipt-aware
/// checkpoints read compact metadata for every rank and validate only local
/// full state. V5 checkpoints that predate receipts retain the full-manifest
/// compatibility preflight; digestless v5 also reads tensors for DP parity.
fn preflight_distributed_checkpoint_set(
    root: &Path,
    requested_rank_dir: &Path,
    expected: &ParallelCheckpointManifest,
) -> Result<()> {
    let requested_manifest = read_checkpoint_manifest(&requested_rank_dir.join("manifest.json"))?;
    if requested_manifest.format != TP_CHECKPOINT_FORMAT {
        // V3/V4 rank shards predate complete-set and DP replica metadata.
        return Ok(());
    }

    let rank_order = expected
        .rank_order
        .iter()
        .map(|axis| axis.name())
        .collect::<Vec<_>>()
        .join("-");
    let topology = ParallelTopology::with_order(
        expected.tensor_model_parallel_size,
        expected.pipeline_model_parallel_size,
        expected.data_parallel_size,
        expected.expert_model_parallel_size,
        expected.context_parallel_size,
        &rank_order,
    )?;
    topology.validate_world_size(expected.world_size)?;
    let expected_requested = ParallelCheckpointManifest::from_topology(
        expected.world_size,
        expected.global_rank,
        &topology,
    )?;
    if &expected_requested != expected {
        bail!(
            "distributed checkpoint topology mismatch: current topology metadata is internally inconsistent"
        );
    }

    match requested_manifest.rank_receipt_version {
        Some(RANK_RECEIPT_VERSION) => {
            return preflight_distributed_receipts(
                root,
                requested_rank_dir,
                &requested_manifest,
                expected,
                &topology,
            );
        }
        Some(version) => {
            bail!("unsupported distributed checkpoint rank receipt version {version}");
        }
        None => {}
    }

    let rank_zero_path = rank_checkpoint_dir(root, 0).join("manifest.json");
    let rank_zero = read_checkpoint_manifest(&rank_zero_path)?;
    if rank_zero.format != TP_CHECKPOINT_FORMAT {
        bail!("distributed checkpoint rank set mixes v5 and legacy checkpoint formats");
    }
    let rank_zero_parallel = rank_zero
        .parallel
        .as_ref()
        .context("rank 0 v5 checkpoint is missing topology metadata")?;
    let expected_rank_zero =
        ParallelCheckpointManifest::from_topology(expected.world_size, 0, &topology)?;
    if rank_zero_parallel != &expected_rank_zero {
        bail!(
            "distributed checkpoint topology mismatch: rank 0 does not match the current topology"
        );
    }

    let baseline_shards = rank_zero
        .tensor_shards
        .iter()
        .map(tensor_shard_identity)
        .collect::<std::collections::BTreeSet<_>>();
    if baseline_shards.len() != rank_zero.tensor_shards.len() {
        bail!("rank 0 checkpoint contains duplicate tensor shard identities");
    }

    let digest_preflight = !rank_zero.file_digests.is_empty();
    let mut replicas: BTreeMap<DataReplicaKey, DataReplicaGroup> = BTreeMap::new();
    let mut replica_files: BTreeMap<DataReplicaFileKey, DataReplicaFileGroup> = BTreeMap::new();
    for global_rank in 0..expected.world_size {
        let rank_dir = rank_checkpoint_dir(root, global_rank);
        let manifest_path = rank_dir.join("manifest.json");
        let manifest = if global_rank == expected.global_rank {
            requested_manifest.clone()
        } else if global_rank == 0 {
            rank_zero.clone()
        } else {
            read_checkpoint_manifest(&manifest_path)?
        };
        let saved_parallel = manifest
            .parallel
            .as_ref()
            .with_context(|| format!("rank {global_rank} v5 checkpoint is missing topology"))?;
        let expected_parallel =
            ParallelCheckpointManifest::from_topology(expected.world_size, global_rank, &topology)?;
        if manifest.format != TP_CHECKPOINT_FORMAT || saved_parallel != &expected_parallel {
            bail!(
                "distributed checkpoint topology mismatch: rank {global_rank} topology or format is inconsistent with the complete v5 rank set"
            );
        }
        validate_checkpoint_generation(&rank_zero, &manifest, global_rank)?;

        let shard_identities = manifest
            .tensor_shards
            .iter()
            .map(tensor_shard_identity)
            .collect::<std::collections::BTreeSet<_>>();
        if shard_identities.len() != manifest.tensor_shards.len()
            || shard_identities != baseline_shards
        {
            bail!("rank {global_rank} checkpoint tensor shard set differs from rank 0");
        }

        if digest_preflight {
            if manifest.file_digests.len() != manifest.files.len()
                || manifest
                    .files
                    .iter()
                    .any(|file| !manifest.file_digests.contains_key(file))
            {
                bail!(
                    "rank {global_rank} checkpoint is missing tensor file content digests"
                );
            }
            for shard in &manifest.tensor_shards {
                if !manifest.files.iter().any(|saved| saved == &shard.file) {
                    bail!(
                        "rank {global_rank} tensor shard references undeclared file {}",
                        shard.file
                    );
                }
                if expected.data_parallel_size > 1
                    && !shard.replicated_axes.contains(&ParallelAxis::Data)
                {
                    bail!(
                        "rank {global_rank} tensor {} is missing data-parallel replica metadata",
                        shard.tensor_name
                    );
                }
            }
            for file in &manifest.files {
                let digest = manifest
                    .file_digests
                    .get(file)
                    .context("validated checkpoint file digest disappeared")?;
                if expected.data_parallel_size <= 1 {
                    continue;
                }
                let key = DataReplicaFileKey {
                    file: file.clone(),
                    tensor_rank: saved_parallel.coordinates.tensor,
                    pipeline_rank: saved_parallel.coordinates.pipeline,
                    expert_rank: saved_parallel.coordinates.expert,
                    context_rank: saved_parallel.coordinates.context,
                };
                match replica_files.entry(key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(DataReplicaFileGroup {
                            data_ranks: [saved_parallel.coordinates.data]
                                .into_iter()
                                .collect(),
                            global_rank,
                            digest: digest.clone(),
                        });
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        let key = entry.key().clone();
                        let group = entry.get_mut();
                        if !group.data_ranks.insert(saved_parallel.coordinates.data) {
                            bail!(
                                "duplicate data-parallel checkpoint replica at rank {global_rank}"
                            );
                        }
                        if group.digest != *digest {
                            bail!(
                                "data-parallel replica content digest differs for {} between ranks {} and {}",
                                key.file,
                                group.global_rank,
                                global_rank
                            );
                        }
                    }
                }
            }
            continue;
        }

        let mut tensors_by_file = BTreeMap::new();
        for file in manifest
            .tensor_shards
            .iter()
            .map(|shard| shard.file.as_str())
            .collect::<std::collections::BTreeSet<_>>()
        {
            if !manifest.files.iter().any(|saved| saved == file) {
                bail!("rank {global_rank} tensor shard references undeclared file {file}");
            }
            tensors_by_file.insert(file.to_string(), read_named_tensors(&rank_dir.join(file))?);
        }

        for shard in &manifest.tensor_shards {
            let tensor = tensors_by_file
                .get(&shard.file)
                .and_then(|tensors| tensors.get(&shard.tensor_name))
                .with_context(|| {
                    format!(
                        "rank {global_rank} checkpoint is missing tensor {}:{}",
                        shard.file, shard.tensor_name
                    )
                })?;
            if tensor.size() != shard.local_shape {
                bail!(
                    "rank {global_rank} tensor {}:{} shape {:?} does not match manifest {:?}",
                    shard.file,
                    shard.tensor_name,
                    tensor.size(),
                    shard.local_shape
                );
            }
            if expected.data_parallel_size > 1
                && !shard.replicated_axes.contains(&ParallelAxis::Data)
            {
                bail!(
                    "rank {global_rank} tensor {} is missing data-parallel replica metadata",
                    shard.tensor_name
                );
            }
            if !shard.replicated_axes.contains(&ParallelAxis::Data) {
                continue;
            }
            let key = DataReplicaKey {
                file: shard.file.clone(),
                tensor_name: shard.tensor_name.clone(),
                state: shard.state.clone(),
                adapter_id: shard.adapter_id,
                tensor_rank: saved_parallel.coordinates.tensor,
                pipeline_rank: saved_parallel.coordinates.pipeline,
                expert_rank: saved_parallel.coordinates.expert,
                context_rank: saved_parallel.coordinates.context,
            };
            let mut replica_shard = shard.clone();
            replica_shard.replica_identity.clear();
            let replica = DataReplica {
                global_rank,
                shard: replica_shard,
                tensor: tensor.shallow_clone(),
            };
            match replicas.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(DataReplicaGroup {
                        data_ranks: [saved_parallel.coordinates.data].into_iter().collect(),
                        reference: replica,
                    });
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let key = entry.key().clone();
                    let group = entry.get_mut();
                    if !group.data_ranks.insert(saved_parallel.coordinates.data) {
                        bail!("duplicate data-parallel checkpoint replica at rank {global_rank}");
                    }
                    if replica.shard != group.reference.shard {
                        bail!(
                            "data-parallel replica metadata differs for {}:{} between ranks {} and {}",
                            key.file,
                            key.tensor_name,
                            group.reference.global_rank,
                            replica.global_rank
                        );
                    }
                    if !group
                        .reference
                        .tensor
                        .allclose(&replica.tensor, 0.0, 0.0, false)
                    {
                        bail!(
                            "data-parallel replica tensor differs for {}:{} ({}) between ranks {} and {}",
                            key.file,
                            key.tensor_name,
                            key.state,
                            group.reference.global_rank,
                            replica.global_rank
                        );
                    }
                }
            }
        }
    }

    for (key, group) in replicas {
        if group.data_ranks.len() != expected.data_parallel_size {
            bail!(
                "data-parallel checkpoint replica set for {}:{} has {} ranks, expected {}",
                key.file,
                key.tensor_name,
                group.data_ranks.len(),
                expected.data_parallel_size
            );
        }
    }
    for (key, group) in replica_files {
        if group.data_ranks.len() != expected.data_parallel_size {
            bail!(
                "data-parallel checkpoint replica set for {} has {} ranks, expected {}",
                key.file,
                group.data_ranks.len(),
                expected.data_parallel_size
            );
        }
    }
    Ok(())
}

fn tensor_shard_identity(shard: &TensorShardManifest) -> (String, String, String, Option<i64>) {
    (
        shard.file.clone(),
        shard.tensor_name.clone(),
        shard.state.clone(),
        shard.adapter_id,
    )
}

fn validate_checkpoint_generation(
    baseline: &CheckpointManifest,
    manifest: &CheckpointManifest,
    global_rank: usize,
) -> Result<()> {
    // Distributed ranks may report different local losses for the same step.
    if manifest.checkpoint_generation != baseline.checkpoint_generation
        || manifest.step != baseline.step
        || manifest.effective_fixed_optimizer_step()
            != baseline.effective_fixed_optimizer_step()
        || manifest.model_path != baseline.model_path
        || manifest.lora_rank != baseline.lora_rank
        || manifest.lora_alpha.to_bits() != baseline.lora_alpha.to_bits()
        || manifest.files != baseline.files
        || manifest.dynamic_adapters != baseline.dynamic_adapters
        || manifest.fixed_shard_layouts != baseline.fixed_shard_layouts
        || manifest.fixed_slot_identities != baseline.fixed_slot_identities
    {
        bail!("rank {global_rank} checkpoint belongs to a different checkpoint generation");
    }
    Ok(())
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
    validate_manifest_file_digests(dir, &manifest)?;
    match expected_parallel {
        Some(expected) => {
            if manifest.format != TP_CHECKPOINT_FORMAT
                && manifest.format != PROJECTION_AWARE_TP_CHECKPOINT_FORMAT
                && manifest.format != LEGACY_TP_CHECKPOINT_FORMAT
            {
                bail!(
                    "distributed resume requires {TP_CHECKPOINT_FORMAT}, {PROJECTION_AWARE_TP_CHECKPOINT_FORMAT}, or {LEGACY_TP_CHECKPOINT_FORMAT}, found {}",
                    manifest.format
                );
            }
            if manifest.format != TP_CHECKPOINT_FORMAT
                && (expected.pipeline_model_parallel_size > 1
                    || expected.data_parallel_size > 1
                    || expected.expert_model_parallel_size > 1
                    || expected.context_parallel_size > 1)
            {
                bail!(
                    "legacy v3/v4 checkpoints only support tensor-parallel rank shards; save a v5 checkpoint for this distributed topology"
                );
            }
            let saved = manifest
                .parallel
                .as_ref()
                .context("tensor-parallel checkpoint is missing topology metadata")?;
            let topology_matches = if manifest.format == TP_CHECKPOINT_FORMAT {
                saved == expected
            } else {
                saved.legacy_fields_match(expected)
            };
            if !topology_matches {
                bail!(
                    "distributed checkpoint topology mismatch: saved={saved:?}, current={expected:?}"
                );
            }
        }
        None if manifest.format == TP_CHECKPOINT_FORMAT
            || manifest.format == PROJECTION_AWARE_TP_CHECKPOINT_FORMAT
            || manifest.format == LEGACY_TP_CHECKPOINT_FORMAT =>
        {
            bail!("distributed checkpoint must be loaded with rank topology");
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
    validate_optimizer_clock_state(
        &manifest,
        &adam_m,
        &adam_v,
        &dynamic_adapters,
    )?;

    if let Some(parallel) = expected_parallel {
        let mut expected_shards = build_tensor_shard_manifest(
            parallel,
            manifest.lora_rank,
            &lora_a,
            &lora_b,
            &adam_m,
            &adam_v,
            &dynamic_adapters,
            &manifest.fixed_shard_layouts,
        )?;
        if manifest.format != TP_CHECKPOINT_FORMAT {
            for shard in &mut expected_shards {
                shard.placements.clear();
                shard.replicated_axes.clear();
            }
        }
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

fn validate_optimizer_clock_state(
    manifest: &CheckpointManifest,
    adam_m: &[Tensor],
    adam_v: &[Tensor],
    dynamic_adapters: &[DynamicAdapterCheckpoint],
) -> Result<()> {
    if manifest.effective_fixed_optimizer_step() > 0
        && (adam_m.is_empty() || adam_v.is_empty())
    {
        bail!(
            "checkpoint fixed optimizer step {} has no Adam state",
            manifest.effective_fixed_optimizer_step()
        );
    }
    for adapter in dynamic_adapters {
        if adapter.manifest.optimizer_step > 0
            && (adapter.adam_m.is_empty() || adapter.adam_v.is_empty())
        {
            bail!(
                "dynamic adapter {} optimizer step {} has no Adam state",
                adapter.manifest.id,
                adapter.manifest.optimizer_step
            );
        }
    }
    Ok(())
}

fn validate_manifest_file_digests(dir: &Path, manifest: &CheckpointManifest) -> Result<()> {
    if manifest.file_digests.is_empty() {
        return Ok(());
    }
    if manifest.file_digests.len() != manifest.files.len()
        || manifest
            .files
            .iter()
            .any(|file| !manifest.file_digests.contains_key(file))
    {
        bail!("checkpoint manifest has an incomplete tensor file digest set");
    }
    for file in &manifest.files {
        let expected = manifest
            .file_digests
            .get(file)
            .context("validated checkpoint file digest disappeared")?;
        let actual = stable_file_digest(&dir.join(file))?;
        if actual != *expected {
            bail!("checkpoint content digest differs from manifest for {file}");
        }
    }
    Ok(())
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
    if parallel.pipeline_model_parallel_size > 1 || parallel.context_parallel_size > 1 {
        bail!(
            "checkpoint v5 does not yet support pipeline/context sharded LoRA state (pp={}, cp={})",
            parallel.pipeline_model_parallel_size,
            parallel.context_parallel_size
        );
    }
    let tp_size = i64::try_from(parallel.tensor_model_parallel_size)
        .context("TP size exceeds checkpoint tensor shape range")?;
    let tp_rank = i64::try_from(parallel.tensor_model_parallel_rank)
        .context("TP rank exceeds checkpoint tensor shape range")?;
    if global_lora_rank <= 0 {
        bail!("global LoRA rank must be positive");
    }
    if layout == LoraTpShardLayout::LatentRank && global_lora_rank % tp_size != 0 {
        bail!(
            "latent-rank sharded LoRA rank {global_lora_rank} must be divisible by TP size {tp_size}"
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
        | (LoraTpShardLayout::FlatQkvColumnParallel { .. }, LoraSide::A)
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
        (
            LoraTpShardLayout::FlatQkvColumnParallel {
                q_rows,
                k_rows,
                v_rows,
            },
            LoraSide::B,
        ) => {
            if local_shape[rank_axis] != global_lora_rank {
                bail!(
                    "flat-QKV column-parallel checkpoint tensor {file}:{tensor_name} has rank {} on axis {rank_axis}, expected {global_lora_rank}",
                    local_shape[rank_axis]
                );
            }
            if q_rows <= 0
                || k_rows <= 0
                || v_rows <= 0
                || q_rows % tp_size != 0
                || k_rows % tp_size != 0
                || v_rows % tp_size != 0
            {
                bail!(
                    "flat-QKV global row segments [{q_rows}, {k_rows}, {v_rows}] must be positive and divisible by TP size {tp_size}"
                );
            }
            let axis = local_shape.len() - 2;
            let local_q = q_rows / tp_size;
            let local_k = k_rows / tp_size;
            let local_v = v_rows / tp_size;
            if local_shape[axis] != local_q + local_k + local_v {
                bail!(
                    "flat-QKV checkpoint tensor {file}:{tensor_name} has {} local rows, expected {} from Q/K/V segments",
                    local_shape[axis],
                    local_q + local_k + local_v
                );
            }
            global_shape[axis] = q_rows + k_rows + v_rows;
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
        (LoraTpShardLayout::RoutedExpertFusedGateUp, LoraSide::A)
        | (LoraTpShardLayout::RoutedExpertDown, LoraSide::B) => {
            if local_shape.len() != 3 || local_shape[rank_axis] != global_lora_rank {
                bail!(
                    "routed expert checkpoint tensor {file}:{tensor_name} must be rank 3 with LoRA rank {global_lora_rank}"
                );
            }
            (rank_axis, true)
        }
        (LoraTpShardLayout::RoutedExpertFusedGateUp, LoraSide::B) => {
            if local_shape.len() != 3 || local_shape[rank_axis] != global_lora_rank {
                bail!(
                    "routed fused gate/up checkpoint tensor {file}:{tensor_name} must be rank 3 with LoRA rank {global_lora_rank}"
                );
            }
            let axis = local_shape.len() - 2;
            if local_shape[axis] <= 0 || local_shape[axis] % 2 != 0 {
                bail!(
                    "routed fused gate/up checkpoint tensor {file}:{tensor_name} must have an even positive local projection size"
                );
            }
            global_shape[axis] *= tp_size;
            (axis, false)
        }
        (LoraTpShardLayout::RoutedExpertDown, LoraSide::A) => {
            if local_shape.len() != 3 || local_shape[rank_axis] != global_lora_rank {
                bail!(
                    "routed expert down checkpoint tensor {file}:{tensor_name} must be rank 3 with LoRA rank {global_lora_rank}"
                );
            }
            let axis = local_shape.len() - 1;
            global_shape[axis] *= tp_size;
            global_offset[axis] = tp_rank * local_shape[axis];
            (axis, false)
        }
    };
    let segments = match (layout, side) {
        (
            LoraTpShardLayout::FlatQkvColumnParallel {
                q_rows,
                k_rows,
                v_rows,
            },
            LoraSide::B,
        ) => {
            let local_q = q_rows / tp_size;
            let local_k = k_rows / tp_size;
            let local_v = v_rows / tp_size;
            vec![
                TensorShardSegmentManifest {
                    local_offset: 0,
                    global_offset: tp_rank * local_q,
                    length: local_q,
                },
                TensorShardSegmentManifest {
                    local_offset: local_q,
                    global_offset: q_rows + tp_rank * local_k,
                    length: local_k,
                },
                TensorShardSegmentManifest {
                    local_offset: local_q + local_k,
                    global_offset: q_rows + k_rows + tp_rank * local_v,
                    length: local_v,
                },
            ]
        }
        (LoraTpShardLayout::RoutedExpertFusedGateUp, LoraSide::B) => {
            let axis = local_shape.len() - 2;
            let local_half = local_shape[axis] / 2;
            let global_half = local_half * tp_size;
            vec![
                TensorShardSegmentManifest {
                    local_offset: 0,
                    global_offset: tp_rank * local_half,
                    length: local_half,
                },
                TensorShardSegmentManifest {
                    local_offset: local_half,
                    global_offset: global_half + tp_rank * local_half,
                    length: local_half,
                },
            ]
        }
        _ => Vec::new(),
    };
    let routed_expert = matches!(
        layout,
        LoraTpShardLayout::RoutedExpertFusedGateUp | LoraTpShardLayout::RoutedExpertDown
    );
    let ep_size = i64::try_from(parallel.expert_model_parallel_size)
        .context("EP size exceeds checkpoint tensor shape range")?;
    let ep_rank = i64::try_from(parallel.coordinates.expert)
        .context("EP rank exceeds checkpoint tensor shape range")?;
    let mut placements = Vec::new();
    let mut replicated_axes = Vec::new();
    if routed_expert && ep_size > 1 {
        global_shape[0] *= ep_size;
        global_offset[0] = ep_rank * local_shape[0];
        placements.push(TensorShardPlacementManifest {
            parallel_axis: ParallelAxis::Expert,
            tensor_axis: 0,
            global_size: global_shape[0],
            local_size: local_shape[0],
            global_offset: global_offset[0],
            segments: Vec::new(),
        });
    } else if parallel.expert_model_parallel_size > 1 {
        replicated_axes.push(ParallelAxis::Expert);
    }
    if parallel.tensor_model_parallel_size > 1 {
        if replicated {
            replicated_axes.push(ParallelAxis::Tensor);
        } else {
            placements.push(TensorShardPlacementManifest {
                parallel_axis: ParallelAxis::Tensor,
                tensor_axis: partition_axis,
                global_size: global_shape[partition_axis],
                local_size: local_shape[partition_axis],
                global_offset: global_offset[partition_axis],
                segments: segments.clone(),
            });
        }
    }
    if parallel.data_parallel_size > 1 {
        replicated_axes.push(ParallelAxis::Data);
    }
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
        segments,
        placements,
        replicated_axes,
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

    #[allow(clippy::too_many_arguments)]
    fn save_checkpoint_with_dynamic_for_topology(
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
        super::save_checkpoint_with_dynamic_and_fixed_step_for_topology_generation(
            dir,
            step,
            if lora_a.is_empty() { 0 } else { step },
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
            parallel,
            Some("checkpoint-test-generation"),
        )
    }

    #[test]
    fn partial_rank_manifest_is_not_ready() {
        let root = tempfile::tempdir().unwrap();
        let rank_dir = root.path().join("rank-00000");
        std::fs::create_dir(&rank_dir).unwrap();
        let partial = rank_dir.join(".manifest.json.partial");
        std::fs::write(&partial, b"{\"format\":").unwrap();

        assert!(!rank_manifests_ready(root.path(), 1));

        std::fs::rename(partial, rank_dir.join("manifest.json")).unwrap();
        std::fs::write(rank_dir.join(RANK_RECEIPT_FILE), b"{}").unwrap();
        assert!(rank_manifests_ready(root.path(), 1));
    }

    #[test]
    fn distributed_checkpoint_requires_nonempty_generation() {
        let root = tempfile::tempdir().unwrap();
        let parallel = ep_topology(0, 2);

        let error = super::save_checkpoint_with_dynamic_and_fixed_step_for_topology(
            root.path(),
            0,
            0,
            0.0,
            "Qwen/test",
            2,
            4.0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &parallel,
        )
        .unwrap_err();
        assert!(error.to_string().contains("explicit unique generation"));

        for generation in [None, Some("")] {
            let error = super::save_checkpoint_with_dynamic_and_fixed_step_for_topology_generation(
                root.path(),
                0,
                0,
                0.0,
                "Qwen/test",
                2,
                4.0,
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
                &parallel,
                generation,
            )
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("distributed checkpoint generation must be non-empty")
            );
        }

        let too_long = "x".repeat(257);
        for generation in ["bad/generation", too_long.as_str()] {
            let error =
                super::save_checkpoint_with_dynamic_and_fixed_step_for_topology_generation(
                    root.path(),
                    0,
                    0,
                    0.0,
                    "Qwen/test",
                    2,
                    4.0,
                    &[],
                    &[],
                    &[],
                    &[],
                    &[],
                    &[],
                    &[],
                    &parallel,
                    Some(generation),
                )
                .unwrap_err();
            assert!(error.to_string().contains("path-safe ASCII"));
        }

        super::save_checkpoint_with_dynamic_and_fixed_step_for_topology_generation(
            root.path(),
            0,
            0,
            0.0,
            "Qwen/test",
            2,
            4.0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &parallel,
            Some("save-transaction-1"),
        )
        .unwrap();
        let error = super::save_checkpoint_with_dynamic_and_fixed_step_for_topology_generation(
            root.path(),
            0,
            0,
            0.0,
            "Qwen/test",
            2,
            4.0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &parallel,
            Some("save-transaction-1"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("has already been used"));
    }

    #[test]
    fn distributed_export_publishes_one_peft_artifact() {
        let root = tempfile::tempdir().unwrap();
        let model_dir = root.path().join("model");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "vocab_size": 16,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "intermediate_size": 8
            }"#,
        )
        .unwrap();
        let final_path = root.path().join("adapter");
        for rank in 0..2 {
            let stale_rank = root.path().join(format!(
                "adapter.rustrain-shards-old-generation/rank-{rank:05}"
            ));
            std::fs::create_dir_all(&stale_rank).unwrap();
            std::fs::write(stale_rank.join("manifest.json"), "stale").unwrap();
        }
        let model_path = model_dir.to_string_lossy().into_owned();
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for global_rank in 0..2 {
                let final_path = final_path.clone();
                let model_path = model_path.clone();
                handles.push(scope.spawn(move || {
                    let parallel = tp_topology(global_rank, 2);
                    let a = Tensor::full([2, 4], 3.0, (tch::Kind::Float, tch::Device::Cpu));
                    let b = Tensor::full(
                        [2, 2],
                        global_rank as f64 + 1.0,
                        (tch::Kind::Float, tch::Device::Cpu),
                    );
                    let adam_m = vec![a.zeros_like(), b.zeros_like()];
                    let adam_v = vec![a.ones_like(), b.ones_like()];
                    export_distributed_adapter_checkpoint(
                        &final_path,
                        "test-generation",
                        &parallel,
                        None,
                        |staging| {
                            save_checkpoint_with_dynamic_for_topology(
                                staging,
                                7,
                                0.25,
                                &model_path,
                                2,
                                4.0,
                                &[a],
                                &[b],
                                &adam_m,
                                &adam_v,
                                &[],
                                &[LoraTpShardLayout::ColumnParallel],
                                &[LoraSlotIdentity {
                                    index: 0,
                                    layer: 0,
                                    module: "q_proj".to_string(),
                                }],
                                &parallel,
                            )
                        },
                    )
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap().unwrap())
                .collect::<Vec<_>>()
        });

        assert_eq!(results, vec![2, 2]);
        let artifact = Qwen36AdapterArtifact::load(&final_path).unwrap();
        assert_eq!(artifact.tensors.len(), 2);
        let a = artifact
            .tensors
            .get("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight")
            .unwrap();
        let b = artifact
            .tensors
            .get("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight")
            .unwrap();
        assert_eq!(a.size(), [2, 4]);
        assert_eq!(b.size(), [4, 2]);
        assert_eq!(b.double_value(&[0, 0]), 1.0);
        assert_eq!(b.double_value(&[3, 0]), 2.0);
        assert!(final_path.join(".rustrain-export-completed.json").is_file());
    }

    #[test]
    fn distributed_export_rejects_reused_attempt_id() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("adapter.rustrain-shards-reused")).unwrap();
        let error = export_distributed_adapter_checkpoint(
            &root.path().join("adapter"),
            "reused",
            &tp_topology(0, 2),
            None,
            |_| panic!("reused attempt must fail before writing rank state"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("attempt IDs cannot be reused"));
        assert!(
            !root
                .path()
                .join("adapter.rustrain-shards-reused/errors")
                .exists()
        );
    }

    #[test]
    fn distributed_export_rejects_file_paths() {
        let root = tempfile::tempdir().unwrap();
        let error = export_distributed_adapter_checkpoint(
            &root.path().join("adapter.safetensors"),
            "test-generation",
            &tp_topology(0, 2),
            None,
            |_| panic!("invalid export path must fail before writing rank state"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("must be a directory"));
    }

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
        assert_eq!(loaded.manifest.effective_fixed_optimizer_step(), 7);
        assert_eq!(loaded.lora_a[0].size(), [2, 3]);
        assert_eq!(loaded.lora_b[0].size(), [4, 2]);
        assert!(loaded.lora_a[0].allclose(&a, 1e-6, 1e-6, false));
        assert!(loaded.lora_b[0].allclose(&b, 1e-6, 1e-6, false));
        assert!(loaded.adam_m[0].allclose(&m, 1e-6, 1e-6, false));
        assert!(loaded.adam_v[0].allclose(&v, 1e-6, 1e-6, false));

        let manifest_path = dir.path().join("manifest.json");
        let mut legacy: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
        legacy.as_object_mut().unwrap().remove("fixed_optimizer_step");
        std::fs::write(&manifest_path, serde_json::to_string_pretty(&legacy).unwrap()).unwrap();
        let legacy = load_checkpoint(dir.path()).unwrap();
        assert_eq!(legacy.manifest.fixed_optimizer_step, None);
        assert_eq!(legacy.manifest.effective_fixed_optimizer_step(), 7);

        let split_clock_dir = tempfile::tempdir().unwrap();
        let single_rank = ParallelCheckpointManifest::new(1, 0, 1, 1, 1, 1, 1).unwrap();
        save_checkpoint_with_dynamic_and_fixed_step_for_topology(
            split_clock_dir.path(),
            9,
            3,
            1.0,
            "Qwen/test",
            2,
            4.0,
            &[a],
            &[b],
            &[m],
            &[v],
            &[],
            &[],
            &[],
            &single_rank,
        )
        .unwrap();
        let split_clock = load_checkpoint(split_clock_dir.path()).unwrap();
        assert_eq!(split_clock.manifest.step, 9);
        assert_eq!(split_clock.manifest.effective_fixed_optimizer_step(), 3);
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
                slot_identities: Vec::new(),
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

    #[test]
    fn optimizer_clocks_require_matching_adam_state() {
        let mut manifest = resume_validation_manifest("rustrain-checkpoint-v2");
        manifest.fixed_optimizer_step = Some(2);
        let error = validate_optimizer_clock_state(&manifest, &[], &[], &[]).unwrap_err();
        assert!(error.to_string().contains("fixed optimizer step 2"));

        manifest.fixed_optimizer_step = Some(0);
        let dynamic = DynamicAdapterCheckpoint {
            manifest: DynamicAdapterManifest {
                id: 7,
                rank: 4,
                alpha: 8.0,
                optimizer_step: 3,
                target_layers: vec![0],
                target_modules: vec!["q_proj".to_string()],
                shard_layouts: Vec::new(),
                slot_identities: Vec::new(),
                parameter_count: 0,
                optimizer_count: 0,
            },
            lora_a: Vec::new(),
            lora_b: Vec::new(),
            adam_m: Vec::new(),
            adam_v: Vec::new(),
        };
        let error = validate_optimizer_clock_state(&manifest, &[], &[], &[dynamic]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("dynamic adapter 7 optimizer step 3")
        );
    }

    fn tp_topology(global_rank: usize, tp_size: usize) -> ParallelCheckpointManifest {
        ParallelCheckpointManifest::new(tp_size, global_rank, tp_size, 1, 1, 1, 1).unwrap()
    }

    fn ep_topology(global_rank: usize, ep_size: usize) -> ParallelCheckpointManifest {
        ParallelCheckpointManifest::new(ep_size, global_rank, 1, 1, 1, ep_size, 1).unwrap()
    }

    fn tp_ep_topology(global_rank: usize) -> ParallelCheckpointManifest {
        ParallelCheckpointManifest::new(4, global_rank, 2, 1, 1, 2, 1).unwrap()
    }

    fn tp_ep_dp_topology(global_rank: usize) -> ParallelCheckpointManifest {
        ParallelCheckpointManifest::new(8, global_rank, 2, 1, 2, 2, 1).unwrap()
    }

    fn save_tp_ep_dp_checkpoint(dir: &Path) {
        let identity = LoraSlotIdentity {
            index: 0,
            layer: 0,
            module: "experts_gate_up_proj".to_string(),
        };
        for global_rank in 0..8 {
            let topology = tp_ep_dp_topology(global_rank);
            let value = (topology.coordinates.expert * 10 + topology.coordinates.tensor) as f64;
            let a = Tensor::full([2, 4, 2], value + 1.0, (tch::Kind::Float, tch::Device::Cpu));
            let b = Tensor::full([2, 4, 4], value + 2.0, (tch::Kind::Float, tch::Device::Cpu));
            let adam_m = vec![
                Tensor::full(a.size(), value + 3.0, (tch::Kind::Float, tch::Device::Cpu)),
                Tensor::full(b.size(), value + 4.0, (tch::Kind::Float, tch::Device::Cpu)),
            ];
            let adam_v = vec![
                Tensor::full(a.size(), value + 5.0, (tch::Kind::Float, tch::Device::Cpu)),
                Tensor::full(b.size(), value + 6.0, (tch::Kind::Float, tch::Device::Cpu)),
            ];
            let dynamic = DynamicAdapterCheckpoint {
                manifest: DynamicAdapterManifest {
                    id: 17,
                    rank: 4,
                    alpha: 8.0,
                    optimizer_step: 29,
                    target_layers: vec![0],
                    target_modules: vec!["experts_gate_up_proj".to_string()],
                    shard_layouts: vec![LoraTpShardLayout::RoutedExpertFusedGateUp],
                    slot_identities: vec![identity.clone()],
                    parameter_count: 1,
                    optimizer_count: 2,
                },
                lora_a: vec![Tensor::full(
                    [2, 4, 2],
                    value + 11.0,
                    (tch::Kind::Float, tch::Device::Cpu),
                )],
                lora_b: vec![Tensor::full(
                    [2, 4, 4],
                    value + 12.0,
                    (tch::Kind::Float, tch::Device::Cpu),
                )],
                adam_m: vec![
                    Tensor::full(
                        [2, 4, 2],
                        value + 13.0,
                        (tch::Kind::Float, tch::Device::Cpu),
                    ),
                    Tensor::full(
                        [2, 4, 4],
                        value + 14.0,
                        (tch::Kind::Float, tch::Device::Cpu),
                    ),
                ],
                adam_v: vec![
                    Tensor::full(
                        [2, 4, 2],
                        value + 15.0,
                        (tch::Kind::Float, tch::Device::Cpu),
                    ),
                    Tensor::full(
                        [2, 4, 4],
                        value + 16.0,
                        (tch::Kind::Float, tch::Device::Cpu),
                    ),
                ],
            };
            save_checkpoint_with_dynamic_for_topology(
                dir,
                31,
                0.125,
                "Qwen/tri-axis-test",
                4,
                8.0,
                &[a],
                &[b],
                &adam_m,
                &adam_v,
                &[dynamic],
                &[LoraTpShardLayout::RoutedExpertFusedGateUp],
                std::slice::from_ref(&identity),
                &topology,
            )
            .unwrap();
        }
    }

    fn rank_for_coordinates(tensor: usize, expert: usize, data: usize) -> usize {
        (0..8)
            .find(|global_rank| {
                let coordinates = tp_ep_dp_topology(*global_rank).coordinates;
                coordinates.tensor == tensor
                    && coordinates.expert == expert
                    && coordinates.data == data
            })
            .expect("tri-axis coordinates must map to one global rank")
    }

    fn replace_checkpoint_tensor(path: &Path, name: &str, value: f64) {
        let mut tensors = read_named_tensors(path).unwrap();
        let original = tensors.get(name).unwrap();
        tensors.insert(
            name.to_string(),
            Tensor::full(original.size(), value, (original.kind(), original.device())),
        );
        save_named_tensors(path, tensors.into_iter().collect()).unwrap();
    }

    fn write_test_manifest_and_receipt(
        rank_dir: &Path,
        manifest: &CheckpointManifest,
    ) {
        let manifest_contents = serde_json::to_vec_pretty(manifest).unwrap();
        write_atomic(&rank_dir.join("manifest.json"), &manifest_contents).unwrap();
        let receipt = checkpoint_rank_receipt(manifest, &manifest_contents).unwrap();
        write_atomic(
            &rank_dir.join(RANK_RECEIPT_FILE),
            &serde_json::to_vec_pretty(&receipt).unwrap(),
        )
        .unwrap();
    }

    fn refresh_checkpoint_file_digest(path: &Path) {
        let rank_dir = path.parent().unwrap();
        let file = path.file_name().unwrap().to_str().unwrap();
        let manifest_path = rank_dir.join("manifest.json");
        let mut manifest = read_checkpoint_manifest(&manifest_path).unwrap();
        manifest
            .file_digests
            .insert(file.to_string(), stable_file_digest(path).unwrap());
        write_test_manifest_and_receipt(rank_dir, &manifest);
    }

    #[test]
    fn tp_ep_dp_v5_full_rank_set_resumes_every_rank() {
        let dir = tempfile::tempdir().unwrap();
        save_tp_ep_dp_checkpoint(dir.path());

        for global_rank in 0..8 {
            let topology = tp_ep_dp_topology(global_rank);
            let loaded = load_checkpoint_for_topology(dir.path(), &topology).unwrap();
            let value = (topology.coordinates.expert * 10 + topology.coordinates.tensor) as f64;
            assert_eq!(loaded.manifest.step, 31);
            assert_eq!(loaded.lora_a[0].double_value(&[0, 0, 0]), value + 1.0);
            assert_eq!(loaded.adam_m[1].double_value(&[0, 0, 0]), value + 4.0);
            assert_eq!(
                loaded.dynamic_adapters[0].adam_v[1].double_value(&[0, 0, 0]),
                value + 16.0
            );
            assert!(
                loaded
                    .manifest
                    .tensor_shards
                    .iter()
                    .all(|shard| { shard.replicated_axes.contains(&ParallelAxis::Data) })
            );
        }
    }

    #[test]
    fn compact_receipts_avoid_remote_manifest_parsing_and_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        save_tp_ep_dp_checkpoint(dir.path());
        let remote_manifest = rank_checkpoint_dir(dir.path(), 1).join("manifest.json");
        std::fs::write(&remote_manifest, b"not-json").unwrap();

        load_checkpoint_for_topology(dir.path(), &tp_ep_dp_topology(0))
            .expect("rank 0 preflight should consume only the remote compact receipt");
        let error = load_checkpoint_for_topology(dir.path(), &tp_ep_dp_topology(1))
            .err()
            .expect("the owner rank must validate its full local manifest");
        assert!(error.to_string().contains("parse"));

        let missing_receipt = tempfile::tempdir().unwrap();
        save_tp_ep_dp_checkpoint(missing_receipt.path());
        let receipt = rank_checkpoint_dir(missing_receipt.path(), 7).join(RANK_RECEIPT_FILE);
        std::fs::rename(&receipt, receipt.with_extension("missing")).unwrap();
        let error = load_checkpoint_for_topology(
            missing_receipt.path(),
            &tp_ep_dp_topology(0),
        )
        .err()
        .expect("new v5 checkpoints must not fall back when a receipt is missing");
        assert!(error.to_string().contains(RANK_RECEIPT_FILE));
    }

    #[test]
    fn digest_v5_without_receipts_retains_full_manifest_compatibility() {
        let dir = tempfile::tempdir().unwrap();
        save_tp_ep_dp_checkpoint(dir.path());
        for global_rank in 0..8 {
            let manifest_path = rank_checkpoint_dir(dir.path(), global_rank).join("manifest.json");
            let mut manifest = read_checkpoint_manifest(&manifest_path).unwrap();
            manifest.rank_receipt_version = None;
            write_atomic(
                &manifest_path,
                &serde_json::to_vec_pretty(&manifest).unwrap(),
            )
            .unwrap();
        }

        load_checkpoint_for_topology(dir.path(), &tp_ep_dp_topology(0))
            .expect("digest v5 checkpoints written before compact receipts must remain readable");
    }

    #[test]
    fn tp_ep_dp_v5_resume_rejects_divergent_adapter_replica() {
        let dir = tempfile::tempdir().unwrap();
        save_tp_ep_dp_checkpoint(dir.path());
        let corrupt_rank = rank_for_coordinates(0, 0, 1);
        let corrupt_path =
            rank_checkpoint_dir(dir.path(), corrupt_rank).join("adapter.safetensors");
        replace_checkpoint_tensor(&corrupt_path, "a_0", 999.0);
        refresh_checkpoint_file_digest(&corrupt_path);

        let error = load_checkpoint_for_topology(dir.path(), &tp_ep_dp_topology(0))
            .err()
            .expect("divergent DP adapter replica must fail preflight");
        assert!(
            error
                .to_string()
                .contains("data-parallel replica content digest differs")
        );
        assert!(error.to_string().contains("adapter.safetensors"));
    }

    #[test]
    fn tp_ep_dp_v5_resume_rejects_divergent_optimizer_replica() {
        let dir = tempfile::tempdir().unwrap();
        save_tp_ep_dp_checkpoint(dir.path());
        let corrupt_rank = rank_for_coordinates(1, 1, 1);
        let corrupt_path =
            rank_checkpoint_dir(dir.path(), corrupt_rank).join("optimizer.safetensors");
        replace_checkpoint_tensor(&corrupt_path, "b_1", 777.0);
        refresh_checkpoint_file_digest(&corrupt_path);

        let error = load_checkpoint_for_topology(dir.path(), &tp_ep_dp_topology(0))
            .err()
            .expect("divergent DP optimizer replica must fail preflight");
        assert!(
            error
                .to_string()
                .contains("data-parallel replica content digest differs")
        );
        assert!(error.to_string().contains("optimizer.safetensors"));
    }

    #[test]
    fn tp_ep_dp_v5_resume_rejects_tensor_file_newer_than_manifest() {
        let dir = tempfile::tempdir().unwrap();
        save_tp_ep_dp_checkpoint(dir.path());
        let corrupt_rank = rank_for_coordinates(0, 1, 1);
        let corrupt_path =
            rank_checkpoint_dir(dir.path(), corrupt_rank).join("adapter.safetensors");
        replace_checkpoint_tensor(&corrupt_path, "a_0", 555.0);

        let error = load_checkpoint_for_topology(
            dir.path(),
            &tp_ep_dp_topology(corrupt_rank),
        )
        .err()
        .expect("stale manifest must not accept newly overwritten tensor content");
        assert!(
            error
                .to_string()
                .contains("checkpoint content digest differs from manifest")
        );
    }

    #[test]
    fn tp_ep_dp_v5_resume_rejects_missing_rank_and_mixed_generation() {
        let missing = tempfile::tempdir().unwrap();
        save_tp_ep_dp_checkpoint(missing.path());
        let missing_manifest = rank_checkpoint_dir(missing.path(), 7).join("manifest.json");
        std::fs::rename(
            &missing_manifest,
            missing_manifest.with_extension("missing"),
        )
        .unwrap();
        let error = load_checkpoint_for_topology(missing.path(), &tp_ep_dp_topology(0))
            .err()
            .expect("missing rank manifest must fail preflight");
        assert!(error.to_string().contains("rank-00007/manifest.json"));

        let mixed = tempfile::tempdir().unwrap();
        save_tp_ep_dp_checkpoint(mixed.path());
        let mixed_manifest_path = rank_checkpoint_dir(mixed.path(), 6).join("manifest.json");
        let mut mixed_manifest = read_checkpoint_manifest(&mixed_manifest_path).unwrap();
        mixed_manifest.step += 1;
        write_test_manifest_and_receipt(mixed_manifest_path.parent().unwrap(), &mixed_manifest);
        let error = load_checkpoint_for_topology(mixed.path(), &tp_ep_dp_topology(0))
            .err()
            .expect("mixed checkpoint generation must fail preflight");
        assert!(
            error
                .to_string()
                .contains("different checkpoint generation")
        );
    }

    fn tp_state(value: f64) -> (Vec<Tensor>, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>) {
        let a = Tensor::full([2, 3], value, (tch::Kind::Float, tch::Device::Cpu));
        let b = Tensor::full([4, 2], value + 1.0, (tch::Kind::Float, tch::Device::Cpu));
        let m = vec![a.zeros_like(), b.zeros_like()];
        let v = vec![a.ones_like(), b.ones_like()];
        (vec![a], vec![b], m, v)
    }

    #[test]
    fn expert_parallel_checkpoint_uses_rank_scoped_directories() {
        let dir = tempfile::tempdir().unwrap();
        for rank in 0..2 {
            let topology = ep_topology(rank, 2);
            let (a, b, m, v) = tp_state(rank as f64 + 1.0);
            save_checkpoint_with_dynamic_for_topology(
                dir.path(),
                7,
                0.25,
                "Qwen/test",
                2,
                4.0,
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
        }

        assert!(dir.path().join("rank-00000/manifest.json").is_file());
        assert!(dir.path().join("rank-00001/manifest.json").is_file());
        assert!(!dir.path().join("manifest.json").exists());
        let rank1 = load_checkpoint_for_topology(dir.path(), &ep_topology(1, 2)).unwrap();
        assert_eq!(rank1.lora_a[0].double_value(&[0, 0]), 2.0);
    }

    #[test]
    fn checkpoint_topology_preserves_custom_rank_order_and_coordinates() {
        let topology = ParallelTopology::with_order(2, 1, 1, 2, 1, "ep-tp").unwrap();
        let manifest = ParallelCheckpointManifest::from_topology(4, 1, &topology).unwrap();
        assert_eq!(manifest.rank_order, topology.order());
        assert_eq!(manifest.coordinates.tensor, 0);
        assert_eq!(manifest.coordinates.expert, 1);
        assert_eq!(manifest.tensor_model_parallel_rank, 0);
    }

    #[test]
    fn routed_expert_shards_record_ep_and_tp_placements() {
        let topology = tp_ep_topology(3);
        let gate_up_a = Tensor::zeros([2, 4, 8], (tch::Kind::Float, tch::Device::Cpu));
        let gate_up_b = Tensor::zeros([2, 6, 4], (tch::Kind::Float, tch::Device::Cpu));
        let down_a = Tensor::zeros([2, 4, 5], (tch::Kind::Float, tch::Device::Cpu));
        let down_b = Tensor::zeros([2, 8, 4], (tch::Kind::Float, tch::Device::Cpu));

        let gate_up_a = tensor_shard(
            &topology,
            None,
            4,
            "adapter.safetensors",
            "a_0".to_string(),
            "lora_a",
            LoraSide::A,
            LoraTpShardLayout::RoutedExpertFusedGateUp,
            &gate_up_a,
        )
        .unwrap();
        let gate_up_b = tensor_shard(
            &topology,
            None,
            4,
            "adapter.safetensors",
            "b_0".to_string(),
            "lora_b",
            LoraSide::B,
            LoraTpShardLayout::RoutedExpertFusedGateUp,
            &gate_up_b,
        )
        .unwrap();
        let down_a = tensor_shard(
            &topology,
            None,
            4,
            "adapter.safetensors",
            "a_1".to_string(),
            "lora_a",
            LoraSide::A,
            LoraTpShardLayout::RoutedExpertDown,
            &down_a,
        )
        .unwrap();
        let down_b = tensor_shard(
            &topology,
            None,
            4,
            "adapter.safetensors",
            "b_1".to_string(),
            "lora_b",
            LoraSide::B,
            LoraTpShardLayout::RoutedExpertDown,
            &down_b,
        )
        .unwrap();

        assert_eq!(gate_up_a.global_shape, vec![4, 4, 8]);
        assert_eq!(gate_up_a.placements.len(), 1);
        assert_eq!(gate_up_a.placements[0].parallel_axis, ParallelAxis::Expert);
        assert!(gate_up_a.replicated_axes.contains(&ParallelAxis::Tensor));

        assert_eq!(gate_up_b.global_shape, vec![4, 12, 4]);
        assert_eq!(gate_up_b.global_offset, vec![2, 0, 0]);
        assert_eq!(gate_up_b.placements.len(), 2);
        let tp = gate_up_b
            .placements
            .iter()
            .find(|placement| placement.parallel_axis == ParallelAxis::Tensor)
            .unwrap();
        assert_eq!(tp.tensor_axis, 1);
        assert_eq!(tp.segments.len(), 2);
        assert_eq!(tp.segments[0].global_offset, 3);
        assert_eq!(tp.segments[1].global_offset, 9);

        assert_eq!(down_a.global_shape, vec![4, 4, 10]);
        assert_eq!(down_a.global_offset, vec![2, 0, 5]);
        assert_eq!(down_a.placements.len(), 2);
        assert_eq!(down_b.global_shape, vec![4, 8, 4]);
        assert_eq!(down_b.placements.len(), 1);
        assert!(down_b.replicated_axes.contains(&ParallelAxis::Tensor));
    }

    #[test]
    fn routed_expert_projection_layout_does_not_shard_lora_rank() {
        let topology = tp_ep_topology(0);
        let tensor = Tensor::zeros([2, 3, 8], (tch::Kind::Float, tch::Device::Cpu));
        let shard = tensor_shard(
            &topology,
            None,
            3,
            "adapter.safetensors",
            "a_0".to_string(),
            "lora_a",
            LoraSide::A,
            LoraTpShardLayout::RoutedExpertFusedGateUp,
            &tensor,
        )
        .unwrap();
        assert_eq!(shard.global_lora_rank, 3);
        assert_eq!(shard.global_shape, vec![4, 3, 8]);
    }

    #[test]
    fn distributed_merge_reconstructs_fused_expert_gate_up_order() {
        let dir = tempfile::tempdir().unwrap();
        let identity = LoraSlotIdentity {
            index: 0,
            layer: 0,
            module: "experts_gate_up_proj".to_string(),
        };
        for global_rank in 0..4 {
            let topology = tp_ep_topology(global_rank);
            let tp_rank = topology.coordinates.tensor as i64;
            let ep_rank = topology.coordinates.expert as i64;
            let a_values = (0..2)
                .flat_map(|local_expert| {
                    let expert = ep_rank * 2 + local_expert;
                    (0..4).flat_map(move |rank| {
                        (0..2).map(move |hidden| (expert * 100 + rank * 10 + hidden) as f32)
                    })
                })
                .collect::<Vec<_>>();
            let local_rows = [
                tp_rank * 2,
                tp_rank * 2 + 1,
                4 + tp_rank * 2,
                5 + tp_rank * 2,
            ];
            let b_values = (0..2)
                .flat_map(|local_expert| {
                    let expert = ep_rank * 2 + local_expert;
                    local_rows.into_iter().flat_map(move |row| {
                        (0..4).map(move |rank| (expert * 1000 + row * 10 + rank) as f32)
                    })
                })
                .collect::<Vec<_>>();
            let a = Tensor::from_slice(&a_values).reshape([2, 4, 2]);
            let b = Tensor::from_slice(&b_values).reshape([2, 4, 4]);
            let adam_m = vec![a.zeros_like(), b.zeros_like()];
            let adam_v = vec![a.zeros_like(), b.zeros_like()];
            save_checkpoint_with_dynamic_for_topology(
                dir.path(),
                11,
                0.5,
                "Qwen/test",
                4,
                8.0,
                &[a],
                &[b],
                &adam_m,
                &adam_v,
                &[],
                &[LoraTpShardLayout::RoutedExpertFusedGateUp],
                std::slice::from_ref(&identity),
                &topology,
            )
            .unwrap();
        }

        for global_rank in 0..4 {
            let topology = tp_ep_topology(global_rank);
            let restored = load_checkpoint_for_topology(dir.path(), &topology).unwrap();
            assert_eq!(restored.manifest.step, 11);
            assert_eq!(restored.lora_a[0].size(), [2, 4, 2]);
            assert_eq!(restored.lora_b[0].size(), [2, 4, 4]);
        }

        let merged = merge_distributed_adapter_checkpoint(dir.path(), None).unwrap();
        assert_eq!(merged.step, 11);
        assert_eq!(merged.slots.len(), 1);
        assert_eq!(merged.slots[0].lora_a.size(), [4, 4, 2]);
        assert_eq!(merged.slots[0].lora_b.size(), [4, 8, 4]);
        for expert in 0..4 {
            for row in 0..8 {
                for rank in 0..4 {
                    assert_eq!(
                        merged.slots[0].lora_b.double_value(&[expert, row, rank]),
                        (expert * 1000 + row * 10 + rank) as f64
                    );
                }
            }
        }
    }

    #[test]
    fn distributed_merge_preserves_dynamic_tenant_identity_and_clock() {
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

        let merged = merge_distributed_adapter_checkpoint(dir.path(), Some(9)).unwrap();
        assert_eq!(merged.rank, 6);
        assert_eq!(merged.optimizer_step, 4);
        assert_eq!(merged.slots.len(), 1);
        assert_eq!(merged.slots[0].identity.module, "q_proj");
        assert_eq!(merged.slots[0].lora_a.size(), [6, 5]);
        assert_eq!(merged.slots[0].lora_b.size(), [7, 6]);
        assert_eq!(merged.slots[0].lora_a.double_value(&[0, 0]), 10.0);
        assert_eq!(merged.slots[0].lora_a.double_value(&[3, 0]), 11.0);
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
                slot_identities: vec![LoraSlotIdentity {
                    index: 0,
                    layer: 1,
                    module: "q_proj".to_string(),
                }],
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
            RANK_RECEIPT_FILE,
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
        let column_a = Tensor::zeros([4, 8], (tch::Kind::Float, tch::Device::Cpu));
        let column_b = Tensor::zeros([8, 4], (tch::Kind::Float, tch::Device::Cpu));
        let flat_qkv_a = Tensor::zeros([4, 8], (tch::Kind::Float, tch::Device::Cpu));
        let flat_qkv_b = Tensor::zeros([12, 4], (tch::Kind::Float, tch::Device::Cpu));
        let row_a = Tensor::zeros([4, 4], (tch::Kind::Float, tch::Device::Cpu));
        let row_b = Tensor::zeros([8, 4], (tch::Kind::Float, tch::Device::Cpu));
        let lora_a = vec![
            column_a.shallow_clone(),
            flat_qkv_a.shallow_clone(),
            row_a.shallow_clone(),
        ];
        let lora_b = vec![
            column_b.shallow_clone(),
            flat_qkv_b.shallow_clone(),
            row_b.shallow_clone(),
        ];
        let adam_m = vec![
            column_a.zeros_like(),
            column_b.zeros_like(),
            flat_qkv_a.zeros_like(),
            flat_qkv_b.zeros_like(),
            row_a.zeros_like(),
            row_b.zeros_like(),
        ];
        let adam_v = vec![
            column_a.ones_like(),
            column_b.ones_like(),
            flat_qkv_a.ones_like(),
            flat_qkv_b.ones_like(),
            row_a.ones_like(),
            row_b.ones_like(),
        ];
        let layouts = [
            LoraTpShardLayout::ColumnParallel,
            LoraTpShardLayout::FlatQkvColumnParallel {
                q_rows: 4,
                k_rows: 4,
                v_rows: 16,
            },
            LoraTpShardLayout::RowParallel,
        ];
        let identities = [
            LoraSlotIdentity {
                index: 0,
                layer: 0,
                module: "q_proj".to_string(),
            },
            LoraSlotIdentity {
                index: 4,
                layer: 1,
                module: "in_proj_qkv".to_string(),
            },
            LoraSlotIdentity {
                index: 3,
                layer: 0,
                module: "o_proj".to_string(),
            },
        ];

        for global_rank in 0..2 {
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
                &tp_topology(global_rank, 2),
            )
            .unwrap();
        }

        let topology = tp_topology(1, 2);
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

        let flat_qkv_a_shard = shard("a_1");
        assert_eq!(
            flat_qkv_a_shard.layout,
            LoraTpShardLayout::FlatQkvColumnParallel {
                q_rows: 4,
                k_rows: 4,
                v_rows: 16,
            }
        );
        assert!(flat_qkv_a_shard.replicated);
        assert_eq!(flat_qkv_a_shard.global_shape, vec![4, 8]);
        assert_eq!(flat_qkv_a_shard.global_offset, vec![0, 0]);

        let flat_qkv_b_shard = shard("b_1");
        assert_eq!(
            flat_qkv_b_shard.layout,
            LoraTpShardLayout::FlatQkvColumnParallel {
                q_rows: 4,
                k_rows: 4,
                v_rows: 16,
            }
        );
        assert!(!flat_qkv_b_shard.replicated);
        assert_eq!(flat_qkv_b_shard.partition_axis, 0);
        assert_eq!(flat_qkv_b_shard.global_shape, vec![24, 4]);
        assert_eq!(flat_qkv_b_shard.global_offset, vec![0, 0]);
        assert_eq!(
            flat_qkv_b_shard.segments,
            vec![
                TensorShardSegmentManifest {
                    local_offset: 0,
                    global_offset: 2,
                    length: 2,
                },
                TensorShardSegmentManifest {
                    local_offset: 2,
                    global_offset: 6,
                    length: 2,
                },
                TensorShardSegmentManifest {
                    local_offset: 4,
                    global_offset: 16,
                    length: 8,
                },
            ]
        );

        let row_a_shard = shard("a_2");
        assert_eq!(row_a_shard.layout, LoraTpShardLayout::RowParallel);
        assert!(!row_a_shard.replicated);
        assert_eq!(row_a_shard.partition_axis, 1);
        assert_eq!(row_a_shard.global_shape, vec![4, 8]);
        assert_eq!(row_a_shard.global_offset, vec![0, 4]);

        let row_b_shard = shard("b_2");
        assert!(row_b_shard.replicated);
        assert_eq!(row_b_shard.global_shape, vec![8, 4]);
        assert_eq!(row_b_shard.global_offset, vec![0, 0]);
        assert_eq!(row_b_shard.replica_identity, "tp-replicated");
    }

    #[test]
    fn flat_qkv_segments_reconstruct_global_lora_and_optimizer_rows() {
        let layout = LoraTpShardLayout::FlatQkvColumnParallel {
            q_rows: 4,
            k_rows: 4,
            v_rows: 16,
        };
        let mut reconstructed = [vec![f64::NAN; 24], vec![f64::NAN; 24], vec![f64::NAN; 24]];

        for rank in 0..2 {
            let topology = tp_topology(rank, 2);
            let global_rows = (0..24).map(f64::from).collect::<Vec<_>>();
            let local_rows = [
                &global_rows[rank * 2..rank * 2 + 2],
                &global_rows[4 + rank * 2..4 + rank * 2 + 2],
                &global_rows[8 + rank * 8..8 + rank * 8 + 8],
            ]
            .concat();

            for (state_index, (state, delta)) in
                [("lora_b", 0.0), ("adam_m", 100.0), ("adam_v", 200.0)]
                    .into_iter()
                    .enumerate()
            {
                let values = local_rows
                    .iter()
                    .map(|value| value + delta)
                    .collect::<Vec<_>>();
                let tensor = Tensor::from_slice(&values).reshape([12, 1]).repeat([1, 4]);
                let shard = tensor_shard(
                    &topology,
                    None,
                    4,
                    "state.safetensors",
                    state.to_string(),
                    state,
                    LoraSide::B,
                    layout,
                    &tensor,
                )
                .unwrap();
                for segment in shard.segments {
                    for offset in 0..segment.length {
                        reconstructed[state_index][(segment.global_offset + offset) as usize] =
                            tensor.double_value(&[segment.local_offset + offset, 0]);
                    }
                }
            }
        }

        for (state_index, delta) in [0.0, 100.0, 200.0].into_iter().enumerate() {
            let expected = (0..24)
                .map(|row| f64::from(row) + delta)
                .collect::<Vec<_>>();
            assert_eq!(reconstructed[state_index], expected);
        }
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
        let parallel = object.get_mut("parallel").unwrap().as_object_mut().unwrap();
        parallel.remove("rank_order");
        parallel.remove("coordinates");
        for shard in object
            .get_mut("tensor_shards")
            .unwrap()
            .as_array_mut()
            .unwrap()
        {
            let shard = shard.as_object_mut().unwrap();
            shard.remove("layout");
            shard.remove("replicated");
            shard.remove("placements");
            shard.remove("replicated_axes");
        }
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let loaded = load_checkpoint_for_topology(dir.path(), &topology).unwrap();
        assert_eq!(loaded.manifest.format, LEGACY_TP_CHECKPOINT_FORMAT);
        assert!(loaded.manifest.fixed_shard_layouts.is_empty());
        assert!(
            loaded
                .manifest
                .tensor_shards
                .iter()
                .all(|shard| shard.layout == LoraTpShardLayout::LatentRank && !shard.replicated)
        );
    }

    #[test]
    fn tensor_parallel_loader_accepts_projection_aware_v4_without_v5_metadata() {
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
            serde_json::Value::String(PROJECTION_AWARE_TP_CHECKPOINT_FORMAT.to_string()),
        );
        let parallel = object.get_mut("parallel").unwrap().as_object_mut().unwrap();
        parallel.remove("rank_order");
        parallel.remove("coordinates");
        for shard in object
            .get_mut("tensor_shards")
            .unwrap()
            .as_array_mut()
            .unwrap()
        {
            let shard = shard.as_object_mut().unwrap();
            shard.remove("placements");
            shard.remove("replicated_axes");
        }
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let loaded = load_checkpoint_for_topology(dir.path(), &topology).unwrap();
        assert_eq!(
            loaded.manifest.format,
            PROJECTION_AWARE_TP_CHECKPOINT_FORMAT
        );
        assert!(
            loaded
                .manifest
                .tensor_shards
                .iter()
                .all(|shard| shard.placements.is_empty() && shard.replicated_axes.is_empty())
        );
    }

    fn resume_validation_manifest(format: &str) -> CheckpointManifest {
        CheckpointManifest {
            format: format.to_string(),
            rank_receipt_version: None,
            checkpoint_generation: None,
            step: 0,
            fixed_optimizer_step: None,
            loss: 0.0,
            model_path: "Qwen/test".to_string(),
            lora_rank: 4,
            lora_alpha: 8.0,
            files: Vec::new(),
            file_digests: BTreeMap::new(),
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
        assert!(
            fixed_error
                .to_string()
                .contains("fixed projection-aware attention LoRA")
        );
        let dynamic_error =
            validate_dynamic_tp_resume(&manifest, 17, &[], &[LoraTpShardLayout::RowParallel], &[])
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
