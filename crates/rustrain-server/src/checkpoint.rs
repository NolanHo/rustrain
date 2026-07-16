//! Checkpoint save/load: adapter (LoRA A/B) + optimizer state (Adam m/v) + step count.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tch::Tensor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub format: String,
    pub step: u64,
    pub loss: f64,
    pub model_path: String,
    pub lora_rank: i64,
    pub lora_alpha: f64,
    pub files: Vec<String>,
}

pub struct CheckpointData {
    pub manifest: CheckpointManifest,
    pub lora_a: Vec<Tensor>,
    pub lora_b: Vec<Tensor>,
    pub adam_m: Vec<Tensor>,
    pub adam_v: Vec<Tensor>,
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
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create checkpoint dir {}", dir.display()))?;

    // Save adapter (LoRA A/B) as safetensors
    let adapter_path = dir.join("adapter.safetensors");
    save_tensors(&adapter_path, &lora_a, &lora_b)?;

    // Save optimizer state (Adam m/v) as safetensors
    let optimizer_path = dir.join("optimizer.safetensors");
    save_tensors(&optimizer_path, &adam_m, &adam_v)?;

    // Write manifest
    let manifest = CheckpointManifest {
        format: "rustrain-checkpoint-v1".to_string(),
        step,
        loss,
        model_path: model_path.to_string(),
        lora_rank,
        lora_alpha,
        files: vec!["adapter.safetensors".into(), "optimizer.safetensors".into()],
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
    let manifest_path = dir.join("manifest.json");
    let manifest: CheckpointManifest = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?,
    )
    .with_context(|| "parse manifest.json")?;

    let adapter_path = dir.join("adapter.safetensors");
    let (lora_a, lora_b) = load_tensors(&adapter_path)?;

    let optimizer_path = dir.join("optimizer.safetensors");
    let (adam_m, adam_v) = load_tensors(&optimizer_path)?;

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
    })
}

fn save_tensors(path: &Path, a: &[Tensor], b: &[Tensor]) -> Result<()> {
    let mut named: Vec<(String, Tensor)> = Vec::new();
    for (i, t) in a.iter().enumerate() {
        named.push((
            format!("a_{i}"),
            t.to_kind(tch::Kind::Float).to_device(tch::Device::Cpu),
        ));
    }
    for (i, t) in b.iter().enumerate() {
        named.push((
            format!("b_{i}"),
            t.to_kind(tch::Kind::Float).to_device(tch::Device::Cpu),
        ));
    }

    Tensor::write_safetensors(&named, path).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn load_tensors(path: &Path) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let named =
        Tensor::read_safetensors(path).with_context(|| format!("read {}", path.display()))?;
    let mut by_name = std::collections::BTreeMap::new();
    for (name, tensor) in named {
        by_name.insert(name, tensor);
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
    Ok((collect_side(&by_name, "a_")?, collect_side(&by_name, "b_")?))
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
        assert_eq!(loaded.manifest.lora_alpha, 4.0);
        assert_eq!(loaded.manifest.step, 7);
        assert_eq!(loaded.lora_a[0].size(), [2, 3]);
        assert_eq!(loaded.lora_b[0].size(), [4, 2]);
        assert!(loaded.lora_a[0].allclose(&a, 1e-6, 1e-6, false));
        assert!(loaded.lora_b[0].allclose(&b, 1e-6, 1e-6, false));
        assert!(loaded.adam_m[0].allclose(&m, 1e-6, 1e-6, false));
        assert!(loaded.adam_v[0].allclose(&v, 1e-6, 1e-6, false));
    }
}
