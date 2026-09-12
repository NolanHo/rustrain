// test_forward.rs — Compare our forward against HF reference, layer by layer.

use std::collections::BTreeMap;
use anyhow::Result;
use tch::{Kind, Tensor};

fn main() -> Result<()> {
    let model_dir = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0";

    let runtime_config = rustrain_qwen3_6::config::read_qwen36_runtime_config(
        &rustrain_qwen3_6::config::resolve_qwen36_model_path(
            &std::path::PathBuf::from(model_dir))?,
    )?;
    println!("Config: {} layers, hidden={}, prefix={}", 
        runtime_config.num_hidden_layers, runtime_config.hidden_size, runtime_config.weight_prefix);

    let device = tch::Device::Cuda(0);
    let compute_kind = Kind::BFloat16;

    // Load ALL weight tensors
    let index_path = std::path::PathBuf::from(model_dir).join("model.safetensors.index.json");
    let index_text = std::fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_text)?;
    let weight_map = index["weight_map"].as_object().unwrap();
    let mut shards: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_name, shard) in weight_map {
        if let Some(s) = shard.as_str() { shards.insert(s.to_string()); }
    }
    let mut weights_cpu: BTreeMap<String, Tensor> = BTreeMap::new();
    for shard in &shards {
        let shard_path = std::path::PathBuf::from(model_dir).join(shard);
        let shard_tensors = rustrain_checkpoint::safetensors::read_safetensors_map(&shard_path)?;
        for (name, tensor) in shard_tensors { weights_cpu.insert(name, tensor); }
    }
    println!("Loaded {} weight tensors from {} shards", weights_cpu.len(), shards.len());

    // Move to GPU
    let mut weights_gpu: BTreeMap<String, Tensor> = BTreeMap::new();
    for (name, tensor) in &weights_cpu {
        weights_gpu.insert(name.clone(), tensor.to_device(device).to_kind(compute_kind));
    }

    // Create the same input as HF test
    let input_ids_vec: Vec<i64> = vec![3710, 369, 220, 17, 10, 17, 30, 198, 760, 4087, 369, 220, 19, 13, 248046]
        .into_iter().chain(std::iter::repeat(91).take(17)).collect();
    let input_ids = Tensor::from_slice(&input_ids_vec).view([1, 32]).to_device(device).to_kind(Kind::Int64);

    // Load HF reference
    let ref_tensors = rustrain_checkpoint::safetensors::read_safetensors_map(
        &std::path::PathBuf::from("/tmp/hf_intermediate_ref.safetensors"))?;

    // Run our forward, saving intermediate hidden states
    let embed_prefix = format!("{}embed_tokens.weight", runtime_config.weight_prefix);
    let embed_tokens = weights_gpu.get(&embed_prefix).unwrap();
    let mut hidden = Tensor::embedding(embed_tokens, &input_ids, -1, false, false);

    // Compare embedding output (hidden_0 in HF = after embedding, before any layer)
    compare_hidden("embedding (hidden_0)", &hidden, &ref_tensors, "hidden_0", device);

    // Load HF attention reference
    let attn_ref = rustrain_checkpoint::safetensors::read_safetensors_map(
        &std::path::PathBuf::from("/tmp/hf_attn_ref.safetensors"))?;

    for layer_index in 0..runtime_config.num_hidden_layers {
        let layer = rustrain_qwen3_6::model::Qwen36LayerWeights::load(
            &weights_gpu, &runtime_config, layer_index, compute_kind)?;

        // For layer 0 and 10, compare attention output
        if layer_index == 0 || layer_index == 10 {
            let input = hidden.to_kind(compute_kind);
            let attn_input = rustrain_qwen3_6::model::rms_norm(&input, &layer.input_norm, runtime_config.rms_norm_eps);
            let attn_ref_key = if layer_index == 0 { "attn_input" } else { "attn_input_10" };
            let attn_out_ref_key = if layer_index == 0 { "attn_output" } else { "attn_output_10" };
            if let Some(hf_attn_in) = attn_ref.get(attn_ref_key) {
                let diff = (attn_input.to_kind(Kind::Float) - hf_attn_in.to_device(device).to_kind(Kind::Float)).abs();
                println!("Layer {layer_index} attn_input:  max_diff={:.6} mean_diff={:.6}", diff.max().double_value(&[]), diff.mean(Kind::Float).double_value(&[]));
            }

            let attn_output = match &layer.attn {
                rustrain_qwen3_6::model::LayerAttnWeights::Full(w) => 
                    rustrain_qwen3_6::model::full_attention(&attn_input, w, &runtime_config, compute_kind),
                rustrain_qwen3_6::model::LayerAttnWeights::Linear(w) => 
                    rustrain_qwen3_6::model::linear_attention(&attn_input, w, &runtime_config, compute_kind),
            };

            if let Some(hf_attn_out) = attn_ref.get(attn_out_ref_key) {
                let diff = (attn_output.to_kind(Kind::Float) - hf_attn_out.to_device(device).to_kind(Kind::Float)).abs();
                let our_std = attn_output.to_kind(Kind::Float).std(false).double_value(&[]);
                let hf_std = hf_attn_out.to_kind(Kind::Float).std(false).double_value(&[]);
                println!("Layer {layer_index} attn_output: max_diff={:.6} mean_diff={:.6} our_std={:.6} hf_std={:.6}", 
                    diff.max().double_value(&[]), diff.mean(Kind::Float).double_value(&[]), our_std, hf_std);
            }
        }

        hidden = rustrain_qwen3_6::model::qwen36_layer(&hidden, &layer, &runtime_config, compute_kind);

        // Compare at specific layers
        let hf_key = format!("hidden_{}", layer_index + 1);
        if ref_tensors.contains_key(&hf_key) {
            compare_hidden(&format!("layer {layer_index} output ({hf_key})"), &hidden, &ref_tensors, &hf_key, device);
        }
    }

    // Final norm + logits
    let final_norm = weights_gpu.get(&format!("{}norm.weight", runtime_config.weight_prefix)).unwrap();
    let hidden_normed = rustrain_qwen3_6::model::rms_norm(&hidden, final_norm, runtime_config.rms_norm_eps);
    let lm_head = weights_gpu.get("lm_head.weight").unwrap();
    let logits = hidden_normed.linear::<&Tensor>(lm_head, None);
    println!("\nOur logits: {:?}", logits.size());

    // Compare logits
    let hf_logits = ref_tensors.get("logits").unwrap().to_device(device).to_kind(Kind::Float);
    let our_logits_f32 = logits.to_kind(Kind::Float);
    let diff = (&our_logits_f32 - &hf_logits).abs();
    println!("Logits max diff: {:.6}", diff.max().double_value(&[]));
    println!("Logits mean diff: {:.6}", diff.mean(Kind::Float).double_value(&[]));

    // Our loss
    let seq_len = logits.size()[1];
    let vocab_size = logits.size()[2];
    let shifted_logits = logits.narrow(1, 0, seq_len - 1).reshape([-1, vocab_size]);
    let shifted_targets = input_ids.narrow(1, 1, seq_len - 1).reshape([-1]);
    let log_probs = shifted_logits.log_softmax(-1, Kind::Float);
    let per_token_loss = log_probs.gather(1, &shifted_targets.unsqueeze(1), false).squeeze().neg();
    println!("Our loss (all tokens): {:.4}", per_token_loss.mean(Kind::Float).double_value(&[]));

    Ok(())
}

fn compare_hidden(label: &str, our: &Tensor, ref_tensors: &BTreeMap<String, Tensor>, key: &str, device: tch::Device) {
    let hf = ref_tensors.get(key).unwrap().to_device(device).to_kind(Kind::Float);
    let our_f32 = our.to_kind(Kind::Float);
    let diff = (&our_f32 - &hf).abs();
    let max_diff = diff.max().double_value(&[]);
    let mean_diff = diff.mean(Kind::Float).double_value(&[]);
    let our_std = our_f32.std(false).double_value(&[]);
    let hf_std = hf.std(false).double_value(&[]);
    println!("{label:<40} max_diff={max_diff:.6} mean_diff={mean_diff:.6} our_std={our_std:.6} hf_std={hf_std:.6}");
}
