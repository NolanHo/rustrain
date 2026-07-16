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
    #[serde(default)]
    pub dynamic_adapters: Vec<DynamicAdapterManifest>,
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
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create checkpoint dir {}", dir.display()))?;

    // Save adapter (LoRA A/B) as safetensors
    let adapter_path = dir.join("adapter.safetensors");
    let mut adapter_tensors = named_tensors(lora_a, lora_b, "a_", "b_");
    let mut dynamic_manifests = Vec::with_capacity(dynamic_adapters.len());
    for adapter in dynamic_adapters {
        if adapter.lora_a.len() != adapter.lora_b.len()
            || adapter.adam_m.len() != adapter.adam_v.len()
            || adapter.manifest.parameter_count != adapter.lora_a.len()
            || adapter.manifest.optimizer_count != adapter.adam_m.len()
        {
            anyhow::bail!(
                "dynamic adapter {} checkpoint count mismatch",
                adapter.manifest.id
            );
        }
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
        format: if dynamic_manifests.is_empty() {
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
    }
}
