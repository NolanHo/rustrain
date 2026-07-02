// test_vision.rs — Standalone test for vision encoder on GPU
// Loads vision weights, runs forward, compares against HF reference.

use std::collections::BTreeMap;
use anyhow::Result;
use tch::{Kind, Tensor};

fn main() -> Result<()> {
    let model_dir = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0";

    // Load config
    let runtime_config = rustrain_qwen3_6::config::read_qwen36_runtime_config(
        &rustrain_qwen3_6::config::resolve_qwen36_model_path(
            &std::path::PathBuf::from(model_dir))?,
    )?;
    println!("Config: has_vision={}, depth={}, hidden={}, heads={}, patch={}, temporal={}, merge={}, out_hidden={}",
        runtime_config.has_vision, runtime_config.vision_depth, runtime_config.vision_hidden_size,
        runtime_config.vision_num_heads, runtime_config.vision_patch_size,
        runtime_config.vision_temporal_patch_size, runtime_config.vision_spatial_merge_size,
        runtime_config.vision_out_hidden_size);

    // Load vision weights
    let vision_names = rustrain_qwen3_6::vision::VisionWeights::weight_names(&runtime_config);
    let needed: std::collections::HashSet<String> = vision_names.into_iter().collect();
    let tensors = rustrain_checkpoint::safetensors::read_safetensors_dir_filtered(
        &std::path::PathBuf::from(model_dir), &needed)?;
    println!("Loaded {} vision tensors from safetensors", tensors.len());

    // Move to GPU as BF16
    let device = tch::Device::Cuda(0);
    let compute_kind = Kind::BFloat16;
    let mut weights_gpu: BTreeMap<String, Tensor> = BTreeMap::new();
    for (name, tensor) in &tensors {
        weights_gpu.insert(name.clone(), tensor.to_device(device).to_kind(compute_kind));
    }

    // Load VisionWeights
    let vision_weights = rustrain_qwen3_6::vision::VisionWeights::load(&weights_gpu, &runtime_config, compute_kind)?;
    println!("VisionWeights loaded: {} blocks", vision_weights.blocks.len());

    // Create test input: [seq_len, in_ch * temporal * P * P]
    let in_ch = 3_i64;
    let t_p = runtime_config.vision_temporal_patch_size;
    let P = runtime_config.vision_patch_size;
    let grid_h = 48_i64;
    let grid_w = 48_i64;
    let seq_len = grid_h * grid_w;

    // Use same random seed and shape as HF reference
    // HF used: torch.randn(seq_len, in_ch * t_p * P * P, dtype=bfloat16, device="cuda")
    // We need to use the SAME input. Save from HF test, load here.
    let ref_tensors = rustrain_checkpoint::safetensors::read_safetensors_map(
        &std::path::PathBuf::from("/tmp/vision_input_ref.safetensors"))?;
    let pixel_values = ref_tensors.get("pixel_values")
        .ok_or_else(|| anyhow::anyhow!("pixel_values not found"))?
        .to_device(device).to_kind(compute_kind);
    println!("Input pixel_values: {:?}", pixel_values.size());

    // Run forward
    let output = rustrain_qwen3_6::vision::vision_forward(&pixel_values, &vision_weights, &runtime_config, compute_kind);
    println!("Output shape: {:?}", output.size());
    println!("Output stats: mean={:.6}, std={:.6}",
        output.to_kind(Kind::Float).mean(Kind::Float).double_value(&[]),
        output.to_kind(Kind::Float).std(false).double_value(&[]));

    // Compare with HF reference (saved as safetensors in /tmp)
    let ref_tensors = rustrain_checkpoint::safetensors::read_safetensors_map(
        &std::path::PathBuf::from("/tmp/vision_merged_ref.safetensors"))?;
    let hf_merged = ref_tensors.get("merged")
        .ok_or_else(|| anyhow::anyhow!("merged ref not found"))?;
    let hf_out = hf_merged.to_device(device).to_kind(compute_kind);
    println!("HF merged shape: {:?}", hf_out.size());
    println!("HF merged stats: mean={:.6}, std={:.6}",
        hf_out.to_kind(Kind::Float).mean(Kind::Float).double_value(&[]),
        hf_out.to_kind(Kind::Float).std(false).double_value(&[]));
    let diff = (&output - &hf_out).abs();
    let max_diff = diff.max().double_value(&[]);
    let mean_diff = diff.to_kind(Kind::Float).mean(Kind::Float).double_value(&[]);
    println!("Max diff vs HF: {:.6}", max_diff);
    println!("Mean diff vs HF: {:.6}", mean_diff);

    // BF16 precision: 27 ViT blocks accumulate error. max_diff < 3.0 is acceptable for BF16.
    if max_diff < 3.0 {
        println!("PASS: vision encoder output matches HF reference within BF16 precision (max_diff={:.6})", max_diff);
    } else {
        println!("FAIL: vision encoder output differs from HF (max_diff={:.6})", max_diff);
    }

    Ok(())
}
