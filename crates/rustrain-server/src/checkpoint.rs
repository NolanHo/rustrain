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
    pub lora_alpha: i64,
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
    lora_alpha: i64,
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
        step, loss,
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
    use std::io::Write;
    let mut named: Vec<(String, Tensor)> = Vec::new();
    for (i, t) in a.iter().enumerate() {
        named.push((format!("a_{i}"), t.to_kind(tch::Kind::Float).to_device(tch::Device::Cpu)));
    }
    for (i, t) in b.iter().enumerate() {
        named.push((format!("b_{i}"), t.to_kind(tch::Kind::Float).to_device(tch::Device::Cpu)));
    }

    // Build safetensors manually (header + data)
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    let mut all_bytes: Vec<u8> = Vec::new();
    for (name, t) in &named {
        let t = t.contiguous().to_kind(tch::Kind::Float);
        let shape: Vec<i64> = t.size().iter().copied().collect();
        let data: Vec<f32> = Vec::<f32>::try_from(&t.reshape([-1]))?;
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        header.insert(
            name.clone(),
            serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[offset, offset + bytes.len() as u64]}),
        );
        offset += bytes.len() as u64;
        all_bytes.extend_from_slice(&bytes);
    }
    let header_str = serde_json::to_string(&serde_json::Value::Object(header))?;
    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    writer.write_all(&(header_str.len() as u64).to_le_bytes())?;
    writer.write_all(header_str.as_bytes())?;
    writer.write_all(&all_bytes)?;
    Ok(())
}

fn load_tensors(path: &Path) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if data.len() < 8 {
        anyhow::bail!("safetensors file too small");
    }
    let header_len = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let header_str = std::str::from_utf8(&data[8..8 + header_len])?;
    let header: serde_json::Value = serde_json::from_str(header_str)?;

    // Count a_N and b_N entries
    let mut a_count = 0;
    let mut b_count = 0;
    if let serde_json::Value::Object(map) = &header {
        for key in map.keys() {
            if key.starts_with("a_") {
                a_count += 1;
            } else if key.starts_with("b_") {
                b_count += 1;
            }
        }
    }

    let mut a_tensors = Vec::new();
    let mut b_tensors = Vec::new();

    for i in 0..a_count {
        let key = format!("a_{i}");
        let (tensor, shape) = load_one_tensor(&header, &key, &data, 8 + header_len)?;
        a_tensors.push(tensor.reshape(&shape));
    }
    for i in 0..b_count {
        let key = format!("b_{i}");
        let (tensor, shape) = load_one_tensor(&header, &key, &data, 8 + header_len)?;
        b_tensors.push(tensor.reshape(&shape));
    }

    Ok((a_tensors, b_tensors))
}

fn load_one_tensor(
    header: &serde_json::Value,
    key: &str,
    data: &[u8],
    data_offset: usize,
) -> Result<(Tensor, Vec<i64>)> {
    let entry = header
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("key {key} not found in safetensors"))?;
    let shape: Vec<i64> = entry["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let offsets = entry["data_offsets"].as_array().unwrap();
    let start = data_offset + offsets[0].as_u64().unwrap() as usize;
    let end = data_offset + offsets[1].as_u64().unwrap() as usize;
    let bytes = &data[start..end];
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let tensor = Tensor::from_slice(&floats);
    Ok((tensor, shape))
}
