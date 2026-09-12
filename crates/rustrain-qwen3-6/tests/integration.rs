//! Integration tests for Qwen3.6 config parsing and forward pass

use std::collections::BTreeMap;

use rustrain_qwen3_6::config::{read_qwen36_runtime_config, resolve_qwen36_model_path, LayerType};
use rustrain_qwen3_6::model::{qwen36_forward_from_ids, Qwen36LayerWeights};

const MODEL_PATH: &str = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0";

#[test]
fn test_config_parsing() {
    let model_path = std::path::Path::new(MODEL_PATH);
    let config = read_qwen36_runtime_config(model_path).expect("config parse");

    // Core
    assert_eq!(config.num_hidden_layers, 40);
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.vocab_size, 248320);
    assert!((config.rms_norm_eps - 1e-6).abs() < 1e-10);
    assert!(!config.tie_word_embeddings);
    assert_eq!(config.hidden_act, "silu");

    // Layer types (3 linear + 1 full, repeated 10×)
    assert_eq!(config.layer_types.len(), 40);
    assert_eq!(config.layer_types[0], LayerType::LinearAttention);
    assert_eq!(config.layer_types[1], LayerType::LinearAttention);
    assert_eq!(config.layer_types[2], LayerType::LinearAttention);
    assert_eq!(config.layer_types[3], LayerType::FullAttention);
    assert_eq!(config.layer_types[4], LayerType::LinearAttention);
    assert_eq!(config.layer_types[7], LayerType::FullAttention);
    assert_eq!(config.layer_types[39], LayerType::FullAttention);
    assert_eq!(config.full_attention_interval, 4);

    // Full attention
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 2);
    assert_eq!(config.head_dim, 256);
    assert!(config.attn_output_gate);
    assert!((config.partial_rotary_factor - 0.25).abs() < 1e-10);
    assert_eq!(config.rope_theta, 10_000_000.0);
    assert!(config.mrope_interleaved);

    // Linear attention
    assert_eq!(config.linear_num_key_heads, 16);
    assert_eq!(config.linear_key_head_dim, 128);
    assert_eq!(config.linear_num_value_heads, 32);
    assert_eq!(config.linear_value_head_dim, 128);
    assert_eq!(config.linear_conv_kernel_dim, 4);

    // MoE
    assert_eq!(config.num_experts, 256);
    assert_eq!(config.num_experts_per_tok, 8);
    assert_eq!(config.moe_intermediate_size, 512);
    assert_eq!(config.shared_expert_intermediate_size, 512);

    // MTP
    assert_eq!(config.mtp_num_hidden_layers, 1);

    // Vision
    assert!(config.has_vision);
    assert_eq!(config.vision_depth, 27);
    assert_eq!(config.vision_hidden_size, 1152);

    // Weight prefix
    assert_eq!(config.weight_prefix, "model.language_model.");
}

#[test]
fn test_forward_single_layer() {
    let model_path = std::path::Path::new(MODEL_PATH);
    let config = read_qwen36_runtime_config(model_path).expect("config parse");

    // Load only layer 0 weights using filtered safetensors loading
    use rustrain_checkpoint::safetensors::read_safetensors_dir_filtered;
    use std::collections::HashSet;
    let p = &config.weight_prefix;
    let layer_prefix = format!("{p}layers.0");
    let needed: HashSet<String> = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "linear_attn.A_log",
        "linear_attn.conv1d.weight",
        "linear_attn.dt_bias",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
        "mlp.gate.weight",
        "mlp.shared_expert_gate.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]
    .iter()
    .map(|suffix| format!("{layer_prefix}.{suffix}"))
    .collect();

    let weights = read_safetensors_dir_filtered(model_path, &needed)
        .expect("load layer 0 weights");

    // Load layer 0 (linear attention)
    let layer = Qwen36LayerWeights::load(&weights, &config, 0, tch::Kind::BFloat16)
        .expect("load layer 0");

    // Run forward on CPU (no CUDA in this env)
    let device = tch::Device::Cpu;
    let input = tch::Tensor::randn(
        [1, 5, config.hidden_size],
        (tch::Kind::BFloat16, device),
    );

    let output = rustrain_qwen3_6::model::qwen36_layer(&input, &layer, &config, tch::Kind::BFloat16);

    assert_eq!(output.size()[0], 1);
    assert_eq!(output.size()[1], 5);
    assert_eq!(output.size()[2], config.hidden_size);
}
