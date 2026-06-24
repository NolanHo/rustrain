use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnParallelLinear {
    weight: Array2<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowParallelLinear {
    weight: Array2<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabParallelEmbedding {
    weight: Array2<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelLMHead {
    weight: Array2<f32>,
}

impl ColumnParallelLinear {
    pub fn new_tp1(weight: Array2<f32>) -> Self {
        Self { weight }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        input.dot(&self.weight)
    }
}

impl RowParallelLinear {
    pub fn new_tp1(weight: Array2<f32>) -> Self {
        Self { weight }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        input.dot(&self.weight)
    }
}

impl VocabParallelEmbedding {
    pub fn new_tp1(weight: Array2<f32>) -> Self {
        Self { weight }
    }

    pub fn forward(&self, input_ids: &[usize]) -> Array2<f32> {
        self.weight.select(Axis(0), input_ids)
    }
}

impl ParallelLMHead {
    pub fn new_tp1(weight: Array2<f32>) -> Self {
        Self { weight }
    }

    pub fn forward(&self, hidden: &Array2<f32>) -> Array2<f32> {
        hidden.dot(&self.weight)
    }
}

pub fn tp1_module_check() -> f32 {
    let input = Array2::ones((1, 2));
    let linear_weight = Array2::eye(2);
    let column = ColumnParallelLinear::new_tp1(linear_weight.clone());
    let row = RowParallelLinear::new_tp1(linear_weight.clone());
    let embedding = VocabParallelEmbedding::new_tp1(linear_weight.clone());
    let head = ParallelLMHead::new_tp1(linear_weight);

    let hidden = row.forward(&column.forward(&input));
    let embedded = embedding.forward(&[0]);
    let logits = head.forward(&hidden);

    logits.sum() + embedded.sum()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn column_parallel_linear_tp1_matches_matmul() {
        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let weight = array![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]];
        let layer = ColumnParallelLinear::new_tp1(weight.clone());

        assert_eq!(layer.forward(&input), input.dot(&weight));
    }

    #[test]
    fn row_parallel_linear_tp1_matches_matmul() {
        let input = array![[1.0, 2.0, 3.0]];
        let weight = array![[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]];
        let layer = RowParallelLinear::new_tp1(weight.clone());

        assert_eq!(layer.forward(&input), input.dot(&weight));
    }

    #[test]
    fn vocab_parallel_embedding_tp1_matches_row_select() {
        let weight = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let embedding = VocabParallelEmbedding::new_tp1(weight);
        let output = embedding.forward(&[2, 0]);

        assert_eq!(output, array![[5.0, 6.0], [1.0, 2.0]]);
    }

    #[test]
    fn parallel_lm_head_tp1_matches_matmul() {
        let hidden = array![[1.0, 2.0]];
        let weight = array![[1.0, 3.0], [2.0, 4.0]];
        let head = ParallelLMHead::new_tp1(weight.clone());

        assert_eq!(head.forward(&hidden), hidden.dot(&weight));
    }
}
