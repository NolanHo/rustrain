use anyhow::{Context, Result};
use ndarray::{Array2, array};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraLinear {
    base_weight: Array2<f32>,
    lora_a: Array2<f32>,
    lora_b: Array2<f32>,
    rank: usize,
    alpha: f32,
    merged: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainableParam {
    pub name: String,
    pub shape: Vec<usize>,
}

impl LoraLinear {
    pub fn zero_init(base_weight: Array2<f32>, rank: usize, alpha: f32) -> Self {
        let in_features = base_weight.nrows();
        let out_features = base_weight.ncols();

        Self {
            base_weight,
            lora_a: Array2::zeros((in_features, rank)),
            lora_b: Array2::zeros((rank, out_features)),
            rank,
            alpha,
            merged: false,
        }
    }

    pub fn with_adapter(
        base_weight: Array2<f32>,
        lora_a: Array2<f32>,
        lora_b: Array2<f32>,
        alpha: f32,
    ) -> Self {
        let rank = lora_a.ncols();
        Self {
            base_weight,
            lora_a,
            lora_b,
            rank,
            alpha,
            merged: false,
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        input.dot(&self.effective_weight())
    }

    pub fn merge(&mut self) {
        if !self.merged {
            self.base_weight += &self.delta_weight();
            self.merged = true;
        }
    }

    pub fn unmerge(&mut self) {
        if self.merged {
            self.base_weight -= &self.delta_weight();
            self.merged = false;
        }
    }

    pub fn trainable_params(&self, prefix: &str) -> Vec<TrainableParam> {
        vec![
            TrainableParam {
                name: format!("{prefix}.lora_a"),
                shape: vec![self.lora_a.nrows(), self.lora_a.ncols()],
            },
            TrainableParam {
                name: format!("{prefix}.lora_b"),
                shape: vec![self.lora_b.nrows(), self.lora_b.ncols()],
            },
        ]
    }

    pub fn save_adapter(&self, path: &std::path::Path) -> Result<()> {
        let contents = self.adapter_toml()?;
        std::fs::write(path, contents)
            .with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn load_adapter(&mut self, path: &std::path::Path) -> Result<()> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read adapter {}", path.display()))?;
        let adapter: LoraAdapter = toml::from_str(&contents)
            .with_context(|| format!("failed to parse adapter {}", path.display()))?;
        self.lora_a = adapter.lora_a;
        self.lora_b = adapter.lora_b;
        self.rank = adapter.rank;
        self.alpha = adapter.alpha;
        self.merged = false;
        Ok(())
    }

    pub fn step_adapter(&mut self, input: &Array2<f32>, grad_output: &Array2<f32>, lr: f32) {
        assert!(!self.merged, "cannot train a merged LoRA adapter");
        let scale = self.alpha / self.rank as f32;
        let grad_delta = input.t().dot(grad_output) * scale;
        let grad_a = grad_delta.dot(&self.lora_b.t());
        let grad_b = self.lora_a.t().dot(&grad_delta);

        self.lora_a -= &(grad_a * lr);
        self.lora_b -= &(grad_b * lr);
    }

    pub fn adapter_param_count(&self) -> usize {
        self.lora_a.len() + self.lora_b.len()
    }

    pub fn adapter_toml(&self) -> Result<String> {
        let adapter = LoraAdapter {
            lora_a: self.lora_a.clone(),
            lora_b: self.lora_b.clone(),
            rank: self.rank,
            alpha: self.alpha,
        };
        toml::to_string(&adapter).context("failed to serialize LoRA adapter")
    }

    pub fn load_adapter_toml(&mut self, contents: &str) -> Result<()> {
        let adapter: LoraAdapter = toml::from_str(contents).context("failed to parse adapter")?;
        self.lora_a = adapter.lora_a;
        self.lora_b = adapter.lora_b;
        self.rank = adapter.rank;
        self.alpha = adapter.alpha;
        self.merged = false;
        Ok(())
    }

    fn effective_weight(&self) -> Array2<f32> {
        if self.merged {
            self.base_weight.clone()
        } else {
            &self.base_weight + &self.delta_weight()
        }
    }

    fn delta_weight(&self) -> Array2<f32> {
        self.lora_a.dot(&self.lora_b) * (self.alpha / self.rank as f32)
    }
}

pub fn lora_trainable_params_check() -> usize {
    let base = array![[1.0, 0.0], [0.0, 1.0]];
    let input = array![[1.0, 2.0]];
    let lora_a = array![[0.25], [0.5]];
    let lora_b = array![[1.0, -1.0]];
    let layer = LoraLinear::with_adapter(base.clone(), lora_a, lora_b, 1.0);
    let _ = layer.forward(&input);
    let params = layer.trainable_params("check");
    let adapter = layer.adapter_toml().expect("adapter should serialize");
    let mut layer = LoraLinear::zero_init(base, 1, 1.0);
    layer
        .load_adapter_toml(&adapter)
        .expect("adapter should load");
    layer.merge();
    layer.unmerge();
    params.len()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LoraAdapter {
    lora_a: Array2<f32>,
    lora_b: Array2<f32>,
    rank: usize,
    alpha: f32,
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn zero_lora_output_equals_base() {
        let base = array![[1.0, 2.0], [3.0, 4.0]];
        let input = array![[2.0, 3.0]];
        let layer = LoraLinear::zero_init(base.clone(), 1, 1.0);

        assert_eq!(layer.forward(&input), input.dot(&base));
    }

    #[test]
    fn non_zero_lora_changes_output() {
        let base = array![[1.0, 2.0], [3.0, 4.0]];
        let input = array![[2.0, 3.0]];
        let lora_a = array![[1.0], [0.5]];
        let lora_b = array![[0.25, -0.5]];
        let layer = LoraLinear::with_adapter(base.clone(), lora_a, lora_b, 1.0);

        assert_ne!(layer.forward(&input), input.dot(&base));
    }

    #[test]
    fn only_lora_params_are_marked_trainable() {
        let base = array![[1.0, 2.0], [3.0, 4.0]];
        let layer = LoraLinear::zero_init(base, 2, 4.0);
        let params = layer.trainable_params("q_proj");

        assert_eq!(
            params,
            vec![
                TrainableParam {
                    name: "q_proj.lora_a".to_string(),
                    shape: vec![2, 2],
                },
                TrainableParam {
                    name: "q_proj.lora_b".to_string(),
                    shape: vec![2, 2],
                },
            ]
        );
    }

    #[test]
    fn adapter_save_load_preserves_output() {
        let base = array![[1.0, 2.0], [3.0, 4.0]];
        let input = array![[2.0, 3.0]];
        let lora_a = array![[1.0], [0.5]];
        let lora_b = array![[0.25, -0.5]];
        let layer = LoraLinear::with_adapter(base.clone(), lora_a, lora_b, 2.0);
        let before = layer.forward(&input);
        let mut loaded = LoraLinear::zero_init(base, 1, 1.0);
        let file = NamedTempFile::new().expect("temp adapter should be created");

        layer
            .save_adapter(file.path())
            .expect("adapter should save");
        loaded
            .load_adapter(file.path())
            .expect("adapter should load");

        assert_eq!(before, loaded.forward(&input));
    }

    #[test]
    fn merge_unmerge_preserves_output() {
        let base = array![[1.0, 2.0], [3.0, 4.0]];
        let input = array![[2.0, 3.0]];
        let lora_a = array![[1.0], [0.5]];
        let lora_b = array![[0.25, -0.5]];
        let mut layer = LoraLinear::with_adapter(base, lora_a, lora_b, 2.0);
        let before = layer.forward(&input);

        layer.merge();
        let merged = layer.forward(&input);
        layer.unmerge();
        let unmerged = layer.forward(&input);

        assert_eq!(before, merged);
        assert_eq!(before, unmerged);
    }

    #[test]
    fn adapter_step_changes_only_lora_path() {
        let base = array![[1.0, 0.0], [0.0, 1.0]];
        let input = array![[1.0, 2.0]];
        let grad = array![[0.5, -0.25]];
        let mut layer =
            LoraLinear::with_adapter(base.clone(), array![[0.1], [0.2]], array![[0.3, -0.1]], 1.0);
        let before = layer.forward(&input);

        layer.step_adapter(&input, &grad, 0.1);

        assert_ne!(before, layer.forward(&input));
        assert_eq!(layer.base_weight, base);
        assert_eq!(layer.adapter_param_count(), 4);
    }
}
