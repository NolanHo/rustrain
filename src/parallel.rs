use ndarray::Array2;

use crate::runtime::ParallelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankInfo {
    pub rank: usize,
    pub world_size: usize,
}

pub trait ProcessGroup {
    fn rank_info(&self) -> RankInfo;
    fn all_reduce_sum(&self, values: &mut Array2<f32>);
    fn all_gather(&self, values: &Array2<f32>) -> Vec<Array2<f32>>;
    fn barrier(&self);
}

#[derive(Debug, Clone)]
pub struct SingleRankProcessGroup {
    rank_info: RankInfo,
}

impl SingleRankProcessGroup {
    pub fn new(config: &ParallelConfig) -> Self {
        debug_assert_eq!(config.tensor_model_parallel_size, 1);
        debug_assert_eq!(config.pipeline_model_parallel_size, 1);
        debug_assert_eq!(config.data_parallel_size, 1);
        debug_assert_eq!(config.expert_model_parallel_size, 1);
        debug_assert_eq!(config.context_parallel_size, 1);

        Self {
            rank_info: RankInfo {
                rank: 0,
                world_size: 1,
            },
        }
    }
}

impl ProcessGroup for SingleRankProcessGroup {
    fn rank_info(&self) -> RankInfo {
        self.rank_info
    }

    fn all_reduce_sum(&self, _values: &mut Array2<f32>) {}

    fn all_gather(&self, values: &Array2<f32>) -> Vec<Array2<f32>> {
        vec![values.clone()]
    }

    fn barrier(&self) {}
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::runtime::ParallelConfig;

    fn single_rank_config() -> ParallelConfig {
        ParallelConfig {
            tensor_model_parallel_size: 1,
            pipeline_model_parallel_size: 1,
            data_parallel_size: 1,
            expert_model_parallel_size: 1,
            context_parallel_size: 1,
        }
    }

    #[test]
    fn single_rank_collectives_are_noops() {
        let group = SingleRankProcessGroup::new(&single_rank_config());
        let mut values = array![[1.0, 2.0], [3.0, 4.0]];

        group.all_reduce_sum(&mut values);
        let gathered = group.all_gather(&values);
        group.barrier();

        assert_eq!(group.rank_info().rank, 0);
        assert_eq!(group.rank_info().world_size, 1);
        assert_eq!(values, array![[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(gathered, vec![array![[1.0, 2.0], [3.0, 4.0]]]);
    }
}
