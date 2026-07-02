// test_real_forward.rs — Compare our forward against HF on REAL SFT data.

use std::collections::BTreeMap;
use anyhow::Result;
use tch::{Kind, Tensor};

fn main() -> Result<()> {
    let model_dir = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0";

    let runtime_config = rustrain_qwen3_6::config::read_qwen36_runtime_config(
        &rustrain_qwen3_6::config::resolve_qwen36_model_path(
            &std::path::PathBuf::from(model_dir))?,
    )?;
    println!("Config: {} layers, hidden={}", runtime_config.num_hidden_layers, runtime_config.hidden_size);

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
    println!("Loaded {} weight tensors", weights_cpu.len());

    let mut weights_gpu: BTreeMap<String, Tensor> = BTreeMap::new();
    for (name, tensor) in &weights_cpu {
        weights_gpu.insert(name.clone(), tensor.to_device(device).to_kind(compute_kind));
    }

    // Load real input IDs
    let ref_input = rustrain_checkpoint::safetensors::read_safetensors_map(
        &std::path::PathBuf::from("/tmp/real_input_ids.safetensors"))?;
    let input_ids = ref_input.get("input_ids").unwrap().to_device(device).to_kind(Kind::Int64);
    println!("Input IDs: {:?}", input_ids.size());

    // Load HF reference
    let hf_ref = rustrain_checkpoint::safetensors::read_safetensors_map(
        &std::path::PathBuf::from("/tmp/hf_real_ref.safetensors"))?;

    // Run our forward, saving intermediate hidden states
    let embed_prefix = format!("{}embed_tokens.weight", runtime_config.weight_prefix);
    let embed_tokens = weights_gpu.get(&embed_prefix).unwrap();
    let mut hidden = Tensor::embedding(embed_tokens, &input_ids, -1, false, false);

    compare_hidden("embedding (hidden_0)", &hidden, &hf_ref, "hidden_0", device);

    // Load layer 3 attention reference
    let l3_ref = rustrain_checkpoint::safetensors::read_safetensors_map(
        &std::path::PathBuf::from("/tmp/hf_l3_attn.safetensors"))?;

    for layer_index in 0..runtime_config.num_hidden_layers {
        let layer = rustrain_qwen3_6::model::Qwen36LayerWeights::load(
            &weights_gpu, &runtime_config, layer_index, compute_kind)?;

        // For layer 3 (first full attention), compare attention details
        if layer_index == 3 {
            // Compare hidden input (before norm)
            compare_hidden("L3 hidden_in", &hidden, &l3_ref, "l3_hidden_in", device);

            let input = hidden.to_kind(compute_kind);
            let attn_input = rustrain_qwen3_6::model::rms_norm(&input, &layer.input_norm, runtime_config.rms_norm_eps);

            // Run full attention
            if let rustrain_qwen3_6::model::LayerAttnWeights::Full(w) = &layer.attn {
                let attn_out = rustrain_qwen3_6::model::full_attention(&attn_input, w, &runtime_config, compute_kind);
                let after_attn = &input + &attn_out;
                // Compare just the attention output (not MoE)
                let our_attn_std = attn_out.to_kind(Kind::Float).std(false).double_value(&[]);
                println!("L3 attn_out: our_std={:.6}", our_attn_std);
                compare_hidden("L3 after_attn", &after_attn, &l3_ref, "l3_hidden_out", device);
            }
        }

        hidden = rustrain_qwen3_6::model::qwen36_layer(&hidden, &layer, &runtime_config, compute_kind);

        let hf_key = format!("hidden_{}", layer_index + 1);
        if hf_ref.contains_key(&hf_key) {
            compare_hidden(&format!("layer {layer_index}"), &hidden, &hf_ref, &hf_key, device);
        }
    }

    // Final norm + logits
    let final_norm = weights_gpu.get(&format!("{}norm.weight", runtime_config.weight_prefix)).unwrap();
    let hidden_normed = rustrain_qwen3_6::model::rms_norm(&hidden, final_norm, runtime_config.rms_norm_eps);
    let lm_head = weights_gpu.get("lm_head.weight").unwrap();
    let logits = hidden_normed.linear::<&Tensor>(lm_head, None);

    let hf_logits = hf_ref.get("logits").unwrap().to_device(device).to_kind(Kind::Float);
    let our_f32 = logits.to_kind(Kind::Float);
    let diff = (&our_f32 - &hf_logits).abs();
    println!("\nLogits max diff: {:.6}", diff.max().double_value(&[]));
    println!("Logits mean diff: {:.6}", diff.mean(Kind::Float).double_value(&[]));

    // Compute masked loss (same as C++ compute_loss)
    let seq_len = logits.size()[1];
    let vocab_size = logits.size()[2];
    let shifted_logits = logits.narrow(1, 0, seq_len - 1).reshape([-1, vocab_size]);
    let shifted_targets = input_ids.narrow(1, 1, seq_len - 1).reshape([-1]);

    // Mask: 0 for first 109 tokens, 1 for next 66, 0 for rest
    let prompt_len = 109_i64;
    let response_len = 65_i64;
    let mut mask_vec = vec![0.0_f32; 256];
    for i in prompt_len..(prompt_len + response_len + 1) {
        if i < 256 { mask_vec[i as usize] = 1.0; }
    }
    let mask = Tensor::from_slice(&mask_vec).view([1, 256]).to_device(device).to_kind(Kind::Float);
    let shifted_mask = mask.narrow(1, 1, seq_len - 1).reshape([-1]);

    let log_probs = shifted_logits.log_softmax(-1, Kind::Float);
    let per_token_loss = log_probs.gather(1, &shifted_targets.unsqueeze(1), false).squeeze().neg();
    let masked = &per_token_loss * &shifted_mask;
    let our_loss = masked.sum(Kind::Float).double_value(&[]) / shifted_mask.sum(Kind::Float).double_value(&[]);
    let num_masked = shifted_mask.sum(Kind::Float).double_value(&[]);

    println!("\nOur masked loss: {:.4} ({} tokens)", our_loss, num_masked as i64);
    println!("HF masked loss:  2.3452 (66 tokens)");

    Ok(())
}

fn compare_hidden(label: &str, our: &Tensor, ref_tensors: &BTreeMap<String, Tensor>, key: &str, device: tch::Device) {
    if let Some(hf) = ref_tensors.get(key) {
        let hf = hf.to_device(device).to_kind(Kind::Float);
        let our_f32 = our.to_kind(Kind::Float);
        let diff = (&our_f32 - &hf).abs();
        let max_diff = diff.max().double_value(&[]);
        let mean_diff = diff.mean(Kind::Float).double_value(&[]);
        let our_std = our_f32.std(false).double_value(&[]);
        let hf_std = hf.std(false).double_value(&[]);
        println!("{label:<20} max_diff={max_diff:.6} mean_diff={mean_diff:.6} our_std={our_std:.6} hf_std={hf_std:.6}");
    }
}
