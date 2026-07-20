//! Integration tests for Qwen3.6 config parsing and forward pass

use rustrain_qwen3_6::config::{LayerType, read_qwen36_runtime_config};
use rustrain_qwen3_6::lora::{
    Qwen36AdapterArtifact, Qwen36LoraConfig, Qwen36LoraTargetModule, native_lora_slots,
    validate_lora_targets,
};
use rustrain_qwen3_6::model::Qwen36LayerWeights;

const MODEL_PATH: &str = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0";
const QWEN35_MODEL_PATH: &str = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

#[test]
fn test_native_lora_target_module_contract() {
    let names = [
        ("q_proj", "self_attn.q_proj"),
        ("k_proj", "self_attn.k_proj"),
        ("v_proj", "self_attn.v_proj"),
        ("o_proj", "self_attn.o_proj"),
        ("in_proj_qkv", "linear_attn.in_proj_qkv"),
        ("in_proj_z", "linear_attn.in_proj_z"),
        ("out_proj", "linear_attn.out_proj"),
    ];
    for (cpp_name, suffix) in names {
        let module = Qwen36LoraTargetModule::parse(cpp_name).expect("supported target");
        assert_eq!(module.cpp_name(), cpp_name);
        assert_eq!(module.suffix(), suffix);
    }
    for name in [
        "in_proj_a",
        "in_proj_b",
        "gate_proj",
        "up_proj",
        "down_proj",
        "shared_gate_proj",
        "shared_up_proj",
        "shared_down_proj",
        "experts_gate_up_proj",
        "experts_down_proj",
    ] {
        assert!(Qwen36LoraTargetModule::parse(name).is_ok(), "{name}");
    }
}

#[test]
fn test_routed_expert_adapter_roundtrip_uses_rank3_local_shards() {
    let runtime =
        read_qwen36_runtime_config(std::path::Path::new(MODEL_PATH)).expect("Qwen3.6 config parse");
    let lora = Qwen36LoraConfig {
        rank: 2,
        alpha: 8.0,
        target_layers: vec![0],
        target_modules: vec![
            Qwen36LoraTargetModule::ExpertsGateUpProj,
            Qwen36LoraTargetModule::ExpertsDownProj,
        ],
    };
    validate_lora_targets(&runtime, &lora).expect("MoE routed expert targets");
    let slots = native_lora_slots(&runtime, &lora);
    let exported = slots
        .iter()
        .map(|slot| match slot.module {
            Qwen36LoraTargetModule::ExpertsGateUpProj if slot.active => (
                tch::Tensor::ones([2, 2, 3], (tch::Kind::Float, tch::Device::Cpu)),
                tch::Tensor::zeros([2, 4, 2], (tch::Kind::Float, tch::Device::Cpu)),
            ),
            Qwen36LoraTargetModule::ExpertsDownProj if slot.active => (
                tch::Tensor::ones([2, 2, 4], (tch::Kind::Float, tch::Device::Cpu)),
                tch::Tensor::zeros([2, 3, 2], (tch::Kind::Float, tch::Device::Cpu)),
            ),
            _ => (
                tch::Tensor::zeros([], (tch::Kind::Float, tch::Device::Cpu)),
                tch::Tensor::zeros([], (tch::Kind::Float, tch::Device::Cpu)),
            ),
        })
        .collect();
    let artifact = Qwen36AdapterArtifact::from_native_exports(
        "Qwen3.6-35B-A3B",
        "qwen3_6_lora_sft",
        Some(std::path::Path::new(MODEL_PATH)),
        &runtime,
        &lora,
        exported,
    )
    .expect("build expert adapter artifact");

    assert_eq!(artifact.tensors.len(), 4);
    let key = format!(
        "base_model.model.{}layers.0.mlp.experts.gate_up_proj.lora_A.weight",
        runtime.weight_prefix
    );
    assert_eq!(artifact.tensors[&key].size(), [2, 2, 3]);

    let temp = tempfile::tempdir().expect("temporary expert adapter directory");
    artifact.save(temp.path()).expect("save expert adapter");
    let loaded = Qwen36AdapterArtifact::load(temp.path()).expect("reload expert adapter");
    assert_eq!(loaded.tensors[&key].size(), [2, 2, 3]);

    let dense_runtime = read_qwen36_runtime_config(std::path::Path::new(QWEN35_MODEL_PATH))
        .expect("Qwen3.5 dense config parse");
    assert!(validate_lora_targets(&dense_runtime, &lora).is_err());
}

#[test]
fn test_adapter_artifact_roundtrip_uses_projection_names() {
    let runtime = read_qwen36_runtime_config(std::path::Path::new(QWEN35_MODEL_PATH))
        .expect("Qwen3.5 config parse");
    let lora = Qwen36LoraConfig {
        rank: 2,
        alpha: 4.0,
        target_layers: vec![0, 3],
        target_modules: vec![
            Qwen36LoraTargetModule::InProjQkv,
            Qwen36LoraTargetModule::QProj,
        ],
    };
    let slots = native_lora_slots(&runtime, &lora);
    let exported = slots
        .iter()
        .map(|slot| {
            let value = slot.index as f64 + 1.0;
            (
                tch::Tensor::full([2, 3], value, (tch::Kind::Float, tch::Device::Cpu)),
                tch::Tensor::full([5, 2], -value, (tch::Kind::Float, tch::Device::Cpu)),
            )
        })
        .collect();
    let artifact = Qwen36AdapterArtifact::from_native_exports(
        "Qwen3.5-0.8B",
        "qwen3_5_lora_sft",
        Some(std::path::Path::new(QWEN35_MODEL_PATH)),
        &runtime,
        &lora,
        exported,
    )
    .expect("build adapter artifact");

    assert_eq!(artifact.tensors.len(), 4);
    let linear_key = format!(
        "base_model.model.{}layers.0.linear_attn.in_proj_qkv.lora_A.weight",
        runtime.weight_prefix
    );
    let full_key = format!(
        "base_model.model.{}layers.3.self_attn.q_proj.lora_B.weight",
        runtime.weight_prefix
    );
    assert!(artifact.tensors.contains_key(&linear_key));
    assert!(artifact.tensors.contains_key(&full_key));

    let temp = tempfile::tempdir().expect("temporary adapter directory");
    let tensor_path = artifact.save(temp.path()).expect("save adapter artifact");
    assert_eq!(
        tensor_path.file_name().and_then(|name| name.to_str()),
        Some("adapter_model.safetensors")
    );
    assert!(temp.path().join("rustrain_adapter.json").is_file());
    let peft_config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(temp.path().join("adapter_config.json")).expect("read PEFT config"),
    )
    .expect("parse PEFT config");
    assert_eq!(peft_config["peft_type"], "LORA");
    assert_eq!(peft_config["r"], 2);
    assert_eq!(
        peft_config["layers_to_transform"],
        serde_json::json!([0, 3])
    );
    assert_eq!(
        peft_config["target_modules"],
        serde_json::json!(["in_proj_qkv", "q_proj"])
    );
    let loaded = Qwen36AdapterArtifact::load(temp.path()).expect("reload adapter artifact");
    assert_eq!(loaded.config, artifact.config);
    assert_eq!(loaded.tensors.len(), artifact.tensors.len());
    for (name, expected) in &artifact.tensors {
        let actual = loaded.tensors.get(name).expect("roundtrip tensor name");
        assert_eq!(actual.size(), expected.size());
        assert_eq!(
            Vec::<f32>::try_from(&actual.reshape([-1])).expect("actual values"),
            Vec::<f32>::try_from(&expected.reshape([-1])).expect("expected values")
        );
    }
    // A PEFT-only export is accepted when rustrain_adapter.json is absent.
    std::fs::remove_file(temp.path().join("rustrain_adapter.json"))
        .expect("remove private rustrain metadata");
    let peft_only = Qwen36AdapterArtifact::load(temp.path()).expect("load PEFT-only artifact");
    assert_eq!(peft_only.config.r, 2);
    assert_eq!(peft_only.config.target_layers, vec![0, 3]);
}

#[test]
fn test_qwen35_text_config_parsing() {
    let config = read_qwen36_runtime_config(std::path::Path::new(QWEN35_MODEL_PATH))
        .expect("Qwen3.5 config parse");
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.layer_types[3], LayerType::FullAttention);
    assert_eq!(config.full_attention_interval, 4);
    assert_eq!(config.linear_num_key_heads, 16);
    assert_eq!(config.linear_num_value_heads, 16);
    assert!(config.attn_output_gate);
    assert!(config.tie_word_embeddings);
}

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

    let weights = read_safetensors_dir_filtered(model_path, &needed).expect("load layer 0 weights");

    // Load layer 0 (linear attention)
    let layer =
        Qwen36LayerWeights::load(&weights, &config, 0, tch::Kind::BFloat16).expect("load layer 0");

    // Run forward on CPU (no CUDA in this env)
    let device = tch::Device::Cpu;
    let input = tch::Tensor::randn([1, 5, config.hidden_size], (tch::Kind::BFloat16, device));

    let output =
        rustrain_qwen3_6::model::qwen36_layer(&input, &layer, &config, tch::Kind::BFloat16);

    assert_eq!(output.size()[0], 1);
    assert_eq!(output.size()[1], 5);
    assert_eq!(output.size()[2], config.hidden_size);
}
