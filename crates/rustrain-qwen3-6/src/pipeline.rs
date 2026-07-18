use anyhow::{Result, bail};
use std::ops::Range;

/// Contiguous pipeline ownership for one physical PP rank.
///
/// Layer IDs remain global at every external boundary. `layer_range` is the
/// only place where they become a stage-local contiguous slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStageLayout {
    pub pipeline_rank: usize,
    pub pipeline_size: usize,
    pub global_num_layers: usize,
    pub layer_range: Range<usize>,
}

impl PipelineStageLayout {
    pub fn new(
        global_num_layers: usize,
        pipeline_rank: usize,
        pipeline_size: usize,
    ) -> Result<Self> {
        if pipeline_size == 0 {
            bail!("pipeline_size must be positive");
        }
        if pipeline_rank >= pipeline_size {
            bail!("pipeline rank {pipeline_rank} is outside pipeline_size={pipeline_size}");
        }
        if global_num_layers < pipeline_size {
            bail!(
                "global_num_layers={global_num_layers} must be at least pipeline_size={pipeline_size}"
            );
        }
        let start = global_num_layers * pipeline_rank / pipeline_size;
        let end = global_num_layers * (pipeline_rank + 1) / pipeline_size;
        Ok(Self {
            pipeline_rank,
            pipeline_size,
            global_num_layers,
            layer_range: start..end,
        })
    }

    pub fn full(global_num_layers: usize) -> Result<Self> {
        Self::new(global_num_layers, 0, 1)
    }

    pub fn is_first(&self) -> bool {
        self.pipeline_rank == 0
    }

    pub fn is_last(&self) -> bool {
        self.pipeline_rank + 1 == self.pipeline_size
    }

    pub fn local_num_layers(&self) -> usize {
        self.layer_range.len()
    }

    pub fn owns_layer(&self, global_layer: usize) -> bool {
        self.layer_range.contains(&global_layer)
    }

    pub fn local_target_layers(&self, global_targets: &[usize]) -> Vec<usize> {
        global_targets
            .iter()
            .copied()
            .filter(|layer| self.owns_layer(*layer))
            .collect()
    }

    pub(crate) fn native_flags(&self) -> i32 {
        i32::from(self.is_first()) | (i32::from(self.is_last()) << 1)
    }
}

#[cfg(test)]
mod tests {
    use super::PipelineStageLayout;

    #[test]
    fn uneven_layers_are_contiguous_and_cover_the_model() {
        let stages = (0..2)
            .map(|rank| PipelineStageLayout::new(5, rank, 2).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(stages[0].layer_range, 0..2);
        assert_eq!(stages[1].layer_range, 2..5);
        assert!(stages[0].is_first());
        assert!(stages[1].is_last());
        assert_eq!(stages[0].layer_range.end, stages[1].layer_range.start);
        assert_eq!(stages[1].layer_range.end, 5);
    }

    #[test]
    fn target_layers_keep_global_identity() {
        let first = PipelineStageLayout::new(4, 0, 2).unwrap();
        let last = PipelineStageLayout::new(4, 1, 2).unwrap();
        assert_eq!(first.local_target_layers(&[0, 3]), vec![0]);
        assert_eq!(last.local_target_layers(&[0, 3]), vec![3]);
    }

    #[test]
    fn rejects_empty_stages_and_invalid_rank() {
        assert!(PipelineStageLayout::new(1, 0, 2).is_err());
        assert!(PipelineStageLayout::new(4, 2, 2).is_err());
        assert!(PipelineStageLayout::new(4, 0, 0).is_err());
    }
}
