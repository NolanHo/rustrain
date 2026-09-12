use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};

use crate::manifest::{QwenDeltaCheckpointManifest, QwenLoraSftAdapterManifest};

pub fn delta_manifest_path(delta_output: &Path) -> PathBuf {
    let mut path = delta_output.as_os_str().to_os_string();
    path.push(".json");
    path.into()
}

pub fn optimizer_state_path(delta_output: &Path) -> PathBuf {
    let mut path = delta_output.as_os_str().to_os_string();
    path.push(".optimizer.safetensors");
    path.into()
}

pub fn qwen_lora_sft_adapter_manifest_path(adapter_output: &Path) -> PathBuf {
    PathBuf::from(format!("{}.json", adapter_output.display()))
}

pub fn write_qwen_delta_manifest(
    manifest_output: &Path,
    manifest: &QwenDeltaCheckpointManifest,
) -> Result<()> {
    if let Some(parent) = manifest_output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        manifest_output,
        serde_json::to_string_pretty(manifest).context("failed to serialize manifest")? + "\n",
    )
    .with_context(|| format!("failed to write {}", manifest_output.display()))
}

pub fn read_qwen_lora_sft_resume_manifest(
    resume_from: &Path,
) -> Result<Option<QwenLoraSftAdapterManifest>> {
    if resume_from.extension().and_then(|value| value.to_str()) != Some("json") {
        return Ok(None);
    }
    let text = fs::read_to_string(resume_from)
        .with_context(|| format!("failed to read {}", resume_from.display()))?;
    let manifest: QwenLoraSftAdapterManifest = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", resume_from.display()))?;
    if manifest.format != "rustrain.qwen_lora_sft_adapter.v1" {
        bail!(
            "unsupported Qwen LoRA SFT adapter manifest format {}",
            manifest.format
        );
    }
    Ok(Some(manifest))
}

pub fn write_qwen_lora_sft_adapter_manifest(
    manifest_output: &Path,
    manifest: &QwenLoraSftAdapterManifest,
) -> Result<()> {
    if let Some(parent) = manifest_output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        manifest_output,
        serde_json::to_string_pretty(manifest)
            .context("failed to serialize Qwen LoRA SFT adapter manifest")?
            + "\n",
    )
    .with_context(|| format!("failed to write {}", manifest_output.display()))
}
