use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use tch::Tensor;

pub fn read_safetensors_map(path: &Path) -> Result<BTreeMap<String, Tensor>> {
    let tensors = Tensor::read_safetensors(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(tensors.into_iter().collect())
}

pub fn read_safetensors_dir(model_dir: &Path) -> Result<BTreeMap<String, Tensor>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return read_safetensors_map(&single);
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        anyhow::bail!(
            "no model.safetensors or model.safetensors.index.json in {}",
            model_dir.display()
        );
    }
    let index_text = std::fs::read_to_string(&index_path)
        .with_context(|| format!("failed to read {}", index_path.display()))?;
    #[derive(serde::Deserialize)]
    struct SafetensorsIndex {
        weight_map: std::collections::HashMap<String, String>,
    }
    let index: SafetensorsIndex = serde_json::from_str(&index_text)
        .with_context(|| format!("failed to parse {}", index_path.display()))?;
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();
    let mut weights = BTreeMap::new();
    for shard_file in &shard_files {
        let shard_path = model_dir.join(shard_file);
        let shard_tensors = Tensor::read_safetensors(&shard_path)
            .with_context(|| format!("failed to read {}", shard_path.display()))?;
        for (name, tensor) in shard_tensors {
            weights.insert(name, tensor);
        }
    }
    Ok(weights)
}

pub fn tensor<'a>(tensors: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| anyhow!("missing tensor {name}"))
}
