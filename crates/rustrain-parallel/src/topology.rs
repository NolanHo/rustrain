//! Rank topology for Megatron-style orthogonal parallel groups.
//!
//! This module only describes process topology. It deliberately does not
//! create NCCL process groups or shard model weights; those operations belong
//! to the model/runtime that owns each collective. Keeping the rank mapping in
//! one place prevents TP/PP/DP/EP/CP launchers from silently disagreeing.

use std::{convert::TryInto, env, ops::Range};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use rustrain_core::runtime::ParallelConfig;

/// Megatron's current default rank order for decoder groups.
pub const DEFAULT_RANK_ORDER: &str = "tp-cp-ep-dp-pp";

/// One orthogonal parallelism axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParallelAxis {
    Tensor,
    Pipeline,
    Data,
    Expert,
    Context,
}

impl ParallelAxis {
    pub const ALL: [Self; 5] = [
        Self::Tensor,
        Self::Context,
        Self::Expert,
        Self::Data,
        Self::Pipeline,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Tensor => "tp",
            Self::Pipeline => "pp",
            Self::Data => "dp",
            Self::Expert => "ep",
            Self::Context => "cp",
        }
    }

    fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "tp" | "tensor" => Ok(Self::Tensor),
            "pp" | "pipeline" => Ok(Self::Pipeline),
            "dp" | "data" => Ok(Self::Data),
            "ep" | "expert" => Ok(Self::Expert),
            "cp" | "context" => Ok(Self::Context),
            other => bail!("unknown parallel axis '{other}' (expected tp/pp/dp/ep/cp)"),
        }
    }
}

/// Coordinates of one global rank in the five-dimensional topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankCoordinates {
    pub tensor: usize,
    pub pipeline: usize,
    pub data: usize,
    pub expert: usize,
    pub context: usize,
}

impl RankCoordinates {
    pub const ZERO: Self = Self {
        tensor: 0,
        pipeline: 0,
        data: 0,
        expert: 0,
        context: 0,
    };

    pub const fn get(self, axis: ParallelAxis) -> usize {
        match axis {
            ParallelAxis::Tensor => self.tensor,
            ParallelAxis::Pipeline => self.pipeline,
            ParallelAxis::Data => self.data,
            ParallelAxis::Expert => self.expert,
            ParallelAxis::Context => self.context,
        }
    }

    pub fn set(&mut self, axis: ParallelAxis, value: usize) {
        match axis {
            ParallelAxis::Tensor => self.tensor = value,
            ParallelAxis::Pipeline => self.pipeline = value,
            ParallelAxis::Data => self.data = value,
            ParallelAxis::Expert => self.expert = value,
            ParallelAxis::Context => self.context = value,
        }
    }
}

/// A validated orthogonal TP/PP/DP/EP/CP rank topology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelTopology {
    sizes: RankCoordinates,
    /// The first axis is the least-significant rank digit, matching
    /// Megatron's `generate_masked_orthogonal_rank_groups` convention.
    order: [ParallelAxis; 5],
}

impl ParallelTopology {
    /// Construct a topology using the Megatron default order.
    pub fn new(
        tensor: usize,
        pipeline: usize,
        data: usize,
        expert: usize,
        context: usize,
    ) -> Result<Self> {
        Self::with_order(tensor, pipeline, data, expert, context, DEFAULT_RANK_ORDER)
    }

    /// Construct a topology with an explicit least-significant-first order.
    pub fn with_order(
        tensor: usize,
        pipeline: usize,
        data: usize,
        expert: usize,
        context: usize,
        order: &str,
    ) -> Result<Self> {
        let sizes = RankCoordinates {
            tensor,
            pipeline,
            data,
            expert,
            context,
        };
        for axis in ParallelAxis::ALL {
            if sizes.get(axis) == 0 {
                bail!("{} parallel size must be greater than zero", axis.name());
            }
        }

        let parsed: Vec<ParallelAxis> = order
            .split('-')
            .filter(|token| !token.trim().is_empty())
            .map(ParallelAxis::parse)
            .collect::<Result<_>>()?;
        if parsed.is_empty() || parsed.len() > ParallelAxis::ALL.len() {
            bail!("parallel rank order must list each axis at most once (got '{order}')");
        }
        for axis in ParallelAxis::ALL {
            let count = parsed
                .iter()
                .filter(|candidate| **candidate == axis)
                .count();
            if count > 1 {
                bail!(
                    "parallel rank order must contain {} at most once",
                    axis.name()
                );
            }
            if count == 0 && sizes.get(axis) != 1 {
                bail!(
                    "parallel rank order omits non-singleton {} axis (size {})",
                    axis.name(),
                    sizes.get(axis)
                );
            }
        }
        let mut order = parsed;
        for axis in ParallelAxis::ALL {
            if !order.contains(&axis) {
                order.push(axis);
            }
        }
        let order: [ParallelAxis; 5] = order
            .try_into()
            .map_err(|_| anyhow!("parallel rank order must contain five axes"))?;
        Ok(Self { sizes, order })
    }

    /// Construct from the public runtime configuration and expected world size.
    pub fn from_config(config: &ParallelConfig, world_size: usize) -> Result<Self> {
        let topology = Self::new(
            config.tensor_model_parallel_size,
            config.pipeline_model_parallel_size,
            config.data_parallel_size,
            config.expert_model_parallel_size,
            config.context_parallel_size,
        )?;
        topology.validate_world_size(world_size)?;
        Ok(topology)
    }

    /// Read axis sizes from launcher environment variables.
    ///
    /// `TP_SIZE`, `PP_SIZE`, `DP_SIZE`, `EP_SIZE`, and `CP_SIZE` are accepted;
    /// each also has a `RUSTRAIN_`-prefixed alias. When no axis is specified,
    /// all ranks are treated as data-parallel replicas. If DP is omitted while
    /// other axes are specified, it is inferred from `WORLD_SIZE`.
    pub fn from_env() -> Result<Self> {
        let world_size = parse_env_usize("WORLD_SIZE")?;
        Self::from_env_with_world_size(world_size)
    }

    /// Read axis sizes from the launcher environment while taking world size
    /// from the caller. This is used by the launcher before it spawns ranks,
    /// when `WORLD_SIZE` is not yet present in the parent process.
    pub fn from_env_with_world_size(world_size: usize) -> Result<Self> {
        if world_size == 0 {
            bail!("WORLD_SIZE must be greater than zero");
        }
        let mut values = [None; 5];
        for (index, names) in [
            ["TP_SIZE", "RUSTRAIN_TP_SIZE", "TENSOR_MODEL_PARALLEL_SIZE"],
            [
                "PP_SIZE",
                "RUSTRAIN_PP_SIZE",
                "PIPELINE_MODEL_PARALLEL_SIZE",
            ],
            ["DP_SIZE", "RUSTRAIN_DP_SIZE", "DATA_PARALLEL_SIZE"],
            ["EP_SIZE", "RUSTRAIN_EP_SIZE", "EXPERT_MODEL_PARALLEL_SIZE"],
            ["CP_SIZE", "RUSTRAIN_CP_SIZE", "CONTEXT_PARALLEL_SIZE"],
        ]
        .into_iter()
        .enumerate()
        {
            values[index] = first_env_usize(&names)?;
        }

        let any_explicit = values.iter().any(Option::is_some);
        let tensor = values[0].unwrap_or(1);
        let pipeline = values[1].unwrap_or(1);
        let expert = values[3].unwrap_or(1);
        let context = values[4].unwrap_or(1);
        let data = match values[2] {
            Some(value) => value,
            None if !any_explicit => world_size,
            None => {
                let model_parallel = tensor
                    .checked_mul(pipeline)
                    .and_then(|value| value.checked_mul(expert))
                    .and_then(|value| value.checked_mul(context))
                    .ok_or_else(|| anyhow!("parallel axis product overflowed usize"))?;
                if model_parallel == 0 || world_size % model_parallel != 0 {
                    bail!(
                        "WORLD_SIZE={world_size} is not divisible by specified model-parallel product {model_parallel}; set DP_SIZE explicitly"
                    );
                }
                world_size / model_parallel
            }
        };
        let order = env::var("RUSTRAIN_PARALLEL_ORDER")
            .or_else(|_| env::var("PARALLEL_ORDER"))
            .unwrap_or_else(|_| DEFAULT_RANK_ORDER.to_string());
        let topology = Self::with_order(tensor, pipeline, data, expert, context, &order)?;
        topology.validate_world_size(world_size)?;
        Ok(topology)
    }

    pub const fn order(&self) -> [ParallelAxis; 5] {
        self.order
    }

    pub const fn sizes(&self) -> RankCoordinates {
        self.sizes
    }

    pub const fn tensor_model_parallel_size(&self) -> usize {
        self.sizes.tensor
    }

    pub const fn pipeline_model_parallel_size(&self) -> usize {
        self.sizes.pipeline
    }

    pub const fn data_parallel_size(&self) -> usize {
        self.sizes.data
    }

    pub const fn expert_model_parallel_size(&self) -> usize {
        self.sizes.expert
    }

    pub const fn context_parallel_size(&self) -> usize {
        self.sizes.context
    }

    pub fn world_size(&self) -> usize {
        self.order
            .into_iter()
            .map(|axis| self.sizes.get(axis))
            .product()
    }

    pub fn validate_world_size(&self, world_size: usize) -> Result<()> {
        let expected = self.world_size();
        if world_size != expected {
            bail!(
                "parallel topology expects world_size={expected} (tp={} pp={} dp={} ep={} cp={}), got {world_size}",
                self.sizes.tensor,
                self.sizes.pipeline,
                self.sizes.data,
                self.sizes.expert,
                self.sizes.context,
            );
        }
        Ok(())
    }

    /// Convert a global rank to its five local coordinates.
    pub fn coordinates(&self, rank: usize) -> Result<RankCoordinates> {
        if rank >= self.world_size() {
            bail!(
                "global rank {rank} is outside world_size {}",
                self.world_size()
            );
        }
        let mut remainder = rank;
        let mut coordinates = RankCoordinates::ZERO;
        for axis in self.order {
            let size = self.sizes.get(axis);
            coordinates.set(axis, remainder % size);
            remainder /= size;
        }
        debug_assert_eq!(remainder, 0);
        Ok(coordinates)
    }

    /// Convert local coordinates to a global rank.
    pub fn rank(&self, coordinates: RankCoordinates) -> Result<usize> {
        for axis in ParallelAxis::ALL {
            if coordinates.get(axis) >= self.sizes.get(axis) {
                bail!(
                    "{} coordinate {} is outside size {}",
                    axis.name(),
                    coordinates.get(axis),
                    self.sizes.get(axis)
                );
            }
        }
        let mut rank = 0usize;
        let mut stride = 1usize;
        for axis in self.order {
            rank = rank
                .checked_add(
                    coordinates
                        .get(axis)
                        .checked_mul(stride)
                        .ok_or_else(|| anyhow!("parallel rank calculation overflowed usize"))?,
                )
                .ok_or_else(|| anyhow!("parallel rank calculation overflowed usize"))?;
            stride = stride
                .checked_mul(self.sizes.get(axis))
                .ok_or_else(|| anyhow!("parallel rank calculation overflowed usize"))?;
        }
        Ok(rank)
    }

    /// Return the global ranks in the process group for one axis.
    pub fn group(&self, rank: usize, axis: ParallelAxis) -> Result<Vec<usize>> {
        let coordinates = self.coordinates(rank)?;
        let mut ranks = Vec::with_capacity(self.sizes.get(axis));
        for local_rank in 0..self.sizes.get(axis) {
            let mut member = coordinates;
            member.set(axis, local_rank);
            ranks.push(self.rank(member)?);
        }
        Ok(ranks)
    }

    pub fn tensor_group(&self, rank: usize) -> Result<Vec<usize>> {
        self.group(rank, ParallelAxis::Tensor)
    }

    pub fn pipeline_group(&self, rank: usize) -> Result<Vec<usize>> {
        self.group(rank, ParallelAxis::Pipeline)
    }

    pub fn data_group(&self, rank: usize) -> Result<Vec<usize>> {
        self.group(rank, ParallelAxis::Data)
    }

    pub fn expert_group(&self, rank: usize) -> Result<Vec<usize>> {
        self.group(rank, ParallelAxis::Expert)
    }

    pub fn context_group(&self, rank: usize) -> Result<Vec<usize>> {
        self.group(rank, ParallelAxis::Context)
    }

    pub fn tensor_rank(&self, rank: usize) -> Result<usize> {
        Ok(self.coordinates(rank)?.tensor)
    }

    pub fn pipeline_rank(&self, rank: usize) -> Result<usize> {
        Ok(self.coordinates(rank)?.pipeline)
    }

    pub fn data_rank(&self, rank: usize) -> Result<usize> {
        Ok(self.coordinates(rank)?.data)
    }

    pub fn expert_rank(&self, rank: usize) -> Result<usize> {
        Ok(self.coordinates(rank)?.expert)
    }

    pub fn context_rank(&self, rank: usize) -> Result<usize> {
        Ok(self.coordinates(rank)?.context)
    }

    pub fn is_first_pipeline_stage(&self, rank: usize) -> Result<bool> {
        Ok(self.pipeline_rank(rank)? == 0)
    }

    pub fn is_last_pipeline_stage(&self, rank: usize) -> Result<bool> {
        Ok(self.pipeline_rank(rank)? + 1 == self.pipeline_model_parallel_size())
    }

    /// Return the contiguous layer range assigned to a pipeline stage.
    pub fn layer_range(&self, rank: usize, num_layers: usize) -> Result<Range<usize>> {
        let stage = self.pipeline_rank(rank)?;
        let stages = self.pipeline_model_parallel_size();
        if num_layers < stages {
            bail!("num_layers={num_layers} must be >= pipeline size {stages}");
        }
        let start = num_layers * stage / stages;
        let end = num_layers * (stage + 1) / stages;
        Ok(start..end)
    }
}

fn parse_env_usize(name: &str) -> Result<usize> {
    env::var(name)
        .with_context(|| format!("{name} is not set"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a positive integer"))
}

fn first_env_usize(names: &[&str]) -> Result<Option<usize>> {
    let Some((name, raw)) = names
        .iter()
        .find_map(|name| env::var(name).ok().map(|raw| (*name, raw)))
    else {
        return Ok(None);
    };
    let value = raw
        .parse::<usize>()
        .with_context(|| format!("{name} must be a positive integer"))?;
    if value == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn megatron_like_tp_dp_pp_order_round_trips_and_builds_groups() {
        let topology = ParallelTopology::with_order(2, 4, 3, 1, 1, "tp-dp-pp").unwrap();
        assert_eq!(topology.world_size(), 24);
        let rank = topology
            .rank(RankCoordinates {
                tensor: 1,
                pipeline: 2,
                data: 1,
                expert: 0,
                context: 0,
            })
            .unwrap();
        // tp is the least-significant digit: 1 + 1*2 + 2*2*3 = 15.
        assert_eq!(rank, 15);
        assert_eq!(topology.coordinates(rank).unwrap().tensor, 1);
        assert_eq!(topology.coordinates(rank).unwrap().data, 1);
        assert_eq!(topology.coordinates(rank).unwrap().pipeline, 2);
        assert_eq!(topology.tensor_group(rank).unwrap(), vec![14, 15]);
        assert_eq!(topology.data_group(rank).unwrap(), vec![13, 15, 17]);
        assert_eq!(topology.pipeline_group(rank).unwrap(), vec![3, 9, 15, 21]);
    }

    #[test]
    fn tp2_dp2_builds_orthogonal_groups_for_every_rank() {
        let topology = ParallelTopology::new(2, 1, 2, 1, 1).unwrap();
        let expected = [
            (0, 0, vec![0, 1], vec![0, 2]),
            (1, 0, vec![0, 1], vec![1, 3]),
            (0, 1, vec![2, 3], vec![0, 2]),
            (1, 1, vec![2, 3], vec![1, 3]),
        ];
        for (rank, (tp_rank, dp_rank, tp_group, dp_group)) in expected.into_iter().enumerate() {
            assert_eq!(topology.tensor_rank(rank).unwrap(), tp_rank);
            assert_eq!(topology.data_rank(rank).unwrap(), dp_rank);
            assert_eq!(topology.tensor_group(rank).unwrap(), tp_group);
            assert_eq!(topology.data_group(rank).unwrap(), dp_group);
        }
    }

    #[test]
    fn five_dimensional_default_order_matches_orthogonal_mapping() {
        let topology = ParallelTopology::new(2, 2, 2, 2, 2).unwrap();
        let coordinates = RankCoordinates {
            tensor: 1,
            pipeline: 0,
            data: 1,
            expert: 1,
            context: 0,
        };
        // tp-cp-ep-dp-pp: 1 + 1*(2*2) + 1*(2*2*2) = 13.
        assert_eq!(topology.rank(coordinates).unwrap(), 13);
        assert_eq!(topology.coordinates(13).unwrap(), coordinates);
        assert_eq!(topology.expert_group(13).unwrap(), vec![9, 13]);
        assert_eq!(topology.context_group(13).unwrap(), vec![13, 15]);
    }

    #[test]
    fn cp2_pp2_groups_match_native_communicator_colors() {
        let topology = ParallelTopology::new(1, 2, 1, 1, 2).unwrap();
        let expected = [
            (0, 0, vec![0, 1], vec![0, 2]),
            (1, 0, vec![0, 1], vec![1, 3]),
            (0, 1, vec![2, 3], vec![0, 2]),
            (1, 1, vec![2, 3], vec![1, 3]),
        ];
        for (rank, (cp_rank, pp_rank, cp_group, pp_group)) in
            expected.into_iter().enumerate()
        {
            assert_eq!(topology.context_rank(rank).unwrap(), cp_rank);
            assert_eq!(topology.pipeline_rank(rank).unwrap(), pp_rank);
            assert_eq!(topology.context_group(rank).unwrap(), cp_group);
            assert_eq!(topology.pipeline_group(rank).unwrap(), pp_group);
            assert_eq!(
                topology.context_group(rank).unwrap().into_iter().min(),
                Some(if pp_rank == 0 { 0 } else { 2 })
            );
            assert_eq!(
                topology.pipeline_group(rank).unwrap().into_iter().min(),
                Some(cp_rank)
            );
        }
    }

    #[test]
    fn rejects_invalid_order_and_world_size() {
        assert!(ParallelTopology::with_order(1, 1, 1, 1, 1, "tp-unknown").is_err());
        assert!(ParallelTopology::with_order(1, 1, 1, 1, 1, "tp-dp-dp-ep-cp").is_err());
        let topology = ParallelTopology::new(2, 2, 1, 1, 1).unwrap();
        assert!(topology.validate_world_size(3).is_err());
        assert!(topology.coordinates(4).is_err());
    }

    #[test]
    fn layer_ranges_are_contiguous_and_cover_the_model() {
        let topology = ParallelTopology::new(1, 3, 1, 1, 1).unwrap();
        assert_eq!(topology.layer_range(0, 10).unwrap(), 0..3);
        assert_eq!(topology.layer_range(1, 10).unwrap(), 3..6);
        assert_eq!(topology.layer_range(2, 10).unwrap(), 6..10);
        assert!(topology.layer_range(0, 2).is_err());
    }
}
