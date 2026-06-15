use std::{collections::BTreeMap, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use tch::{Device, Kind, Tensor};

#[derive(Debug, Serialize)]
struct DiffStats {
    max_abs: f64,
    mean_abs: f64,
}

#[derive(Debug, Serialize)]
struct QwenModuleParitySummary {
    model_safetensors: String,
    fixture: String,
    rms_norm_diff: DiffStats,
    mlp_diff: DiffStats,
}

pub fn qwen_module_parity(model_safetensors: &Path, fixture: &Path) -> Result<()> {
    let weights = read_safetensors_map(model_safetensors)?;
    let fixture_tensors = read_safetensors_map(fixture)?;
    let input = tensor(&fixture_tensors, "embedded_hidden")?.to_kind(Kind::Float);
    let expected_norm = tensor(&fixture_tensors, "post_attention_normed")?.to_kind(Kind::Float);
    let expected_mlp = tensor(&fixture_tensors, "mlp_output")?.to_kind(Kind::Float);

    let norm_weight =
        tensor(&weights, "model.layers.0.post_attention_layernorm.weight")?.to_kind(Kind::Float);
    let gate_proj = tensor(&weights, "model.layers.0.mlp.gate_proj.weight")?.to_kind(Kind::Float);
    let up_proj = tensor(&weights, "model.layers.0.mlp.up_proj.weight")?.to_kind(Kind::Float);
    let down_proj = tensor(&weights, "model.layers.0.mlp.down_proj.weight")?.to_kind(Kind::Float);

    let actual_norm = rms_norm(&input, &norm_weight, 1e-6);
    let actual_mlp = qwen_mlp(&actual_norm, &gate_proj, &up_proj, &down_proj);
    let rms_norm_diff = diff_stats(&actual_norm, &expected_norm)?;
    let mlp_diff = diff_stats(&actual_mlp, &expected_mlp)?;

    if rms_norm_diff.max_abs > 1e-5 {
        bail!("RMSNorm parity failed: max_abs={}", rms_norm_diff.max_abs);
    }
    if mlp_diff.max_abs > 1e-4 {
        bail!("MLP parity failed: max_abs={}", mlp_diff.max_abs);
    }

    let summary = QwenModuleParitySummary {
        model_safetensors: model_safetensors.display().to_string(),
        fixture: fixture.display().to_string(),
        rms_norm_diff,
        mlp_diff,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

fn read_safetensors_map(path: &Path) -> Result<BTreeMap<String, Tensor>> {
    let tensors = Tensor::read_safetensors(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(tensors.into_iter().collect())
}

fn tensor<'a>(tensors: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| anyhow!("missing tensor {name}"))
}

fn rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let variance = input
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    input * (variance + eps).rsqrt() * weight
}

fn qwen_mlp(input: &Tensor, gate_proj: &Tensor, up_proj: &Tensor, down_proj: &Tensor) -> Tensor {
    let gate = input.linear::<&Tensor>(gate_proj, None);
    let up = input.linear::<&Tensor>(up_proj, None);
    (gate.silu() * up).linear::<&Tensor>(down_proj, None)
}

fn diff_stats(actual: &Tensor, expected: &Tensor) -> Result<DiffStats> {
    if actual.size() != expected.size() {
        bail!(
            "shape mismatch: actual {:?}, expected {:?}",
            actual.size(),
            expected.size()
        );
    }
    let diff = (actual - expected).abs().to_device(Device::Cpu);
    Ok(DiffStats {
        max_abs: diff.max().double_value(&[]),
        mean_abs: diff.mean(Kind::Float).double_value(&[]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_matches_manual_formula() {
        let input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([1, 2, 2]);
        let weight = Tensor::from_slice(&[0.5_f32, 2.0]);
        let output = rms_norm(&input, &weight, 1e-6);

        assert_eq!(output.size(), vec![1, 2, 2]);
        assert!(output.isfinite().all().int64_value(&[]) == 1);
    }
}
