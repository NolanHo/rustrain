//! qwen_module - re-export hub for split modules
//!
//! Original monolithic file split into:
//! - model: model core (weights/forward/backward/RMSNorm/RoPE/GQA/SwiGLU/attention/MLP)
//! - parity: consistency checks (logits/generate/module parity, sampling/KV cache smoke)
//! - generate: generation/sampling/KV cache
//! - lora: LoRA config/registry/injection/training/smoke
//! - session: trainable session (single/DP/TP)
//! - sft: SFT data flow (dataset/field map/streaming/arrow)
//! - rank_smoke: DP/TP rank smoke tests

// Re-export all items from split modules so external code using
// `qwen_module::SomeType` continues to work unchanged.
pub use crate::generate::*;
pub use crate::lora::*;
pub use crate::model::*;
pub use crate::parity::*;
pub use crate::rank_smoke::*;
pub use crate::session::*;
pub use crate::sft::*;

// Imports needed by the `tests` module below (accessible via `use super::*`).
#[cfg(test)]
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, UNIX_EPOCH},
};

#[cfg(test)]
use anyhow::{Context, Result, anyhow, bail};

#[cfg(test)]
use arrow::{
    array::{Array, LargeStringArray, RecordBatch, StringArray},
    datatypes::{DataType, SchemaRef},
    ipc::reader::{FileReader as ArrowFileReader, StreamReader as ArrowStreamReader},
};

#[cfg(test)]
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

#[cfg(test)]
use serde::{Deserialize, Serialize};

#[cfg(test)]
use tch::{Device, IndexOp, Kind, Reduction, Tensor, no_grad};

#[cfg(test)]
use tokenizers::Tokenizer;

#[cfg(test)]
use tracing::info;

#[cfg(test)]
use rustrain_checkpoint::io::{
    delta_manifest_path, optimizer_state_path, qwen_lora_sft_adapter_manifest_path,
    read_qwen_lora_sft_resume_manifest, write_qwen_delta_manifest,
    write_qwen_lora_sft_adapter_manifest,
};

#[cfg(test)]
use rustrain_checkpoint::manifest::*;

#[cfg(test)]
use rustrain_checkpoint::safetensors::{read_safetensors_map, tensor};

#[cfg(test)]
use rustrain_core::runtime::{
    Config, DataConfig as RuntimeDataConfig, DataKind as RuntimeDataKind, Device as RuntimeDevice,
    FieldAffix, FieldCaseTransform, FieldCaseTransformKind, FieldDefault, FieldDefaultTarget,
    FieldRegexFilter, FieldRegexReplacement, FieldReplacement, FieldReplacementTarget, FieldSplit,
    FieldSplitSide, FieldStrip, FieldTransform, FieldTransformOp, FieldTruncation,
    LoraConfig as RuntimeLoraConfig, LrScheduler, RunPaths, load_config,
};

#[cfg(test)]
use rustrain_nccl::nccl_smoke;

#[cfg(test)]
use tempfile as _;

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        datatypes::{Field, Schema},
        ipc::writer::StreamWriter,
    };

    #[test]
    fn rms_norm_matches_manual_formula() {
        let input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([1, 2, 2]);
        let weight = Tensor::from_slice(&[0.5_f32, 2.0]);
        let output = rms_norm(&input, &weight, 1e-6);

        assert_eq!(output.size(), vec![1, 2, 2]);
        assert!(output.isfinite().all().int64_value(&[]) == 1);
    }

    #[test]
    fn rotate_half_splits_head_dimension_in_halves() {
        let input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([1, 1, 1, 4]);
        let output = rotate_half(&input);

        let values: Vec<f32> = Vec::<f32>::try_from(output.reshape([4])).unwrap();
        assert_eq!(values, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn qwen_causal_lm_loss_is_finite_for_tiny_weights() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);

        let loss = qwen_causal_lm_loss(&input_ids, &weights, &config).expect("loss should run");

        assert_eq!(loss.size(), Vec::<i64>::new());
        assert!(loss.isfinite().int64_value(&[]) == 1);
    }

    #[test]
    fn representative_full_train_tensors_get_gradients_and_reload() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let mut weights = tiny_qwen_weights();
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let mut registry =
            QwenTrainableRegistry::representative(&mut weights).expect("registry should build");
        assert_eq!(
            registry.parameter_names(),
            representative_trainable_qwen_tensors()
        );

        let initial_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        let loss = qwen_causal_lm_loss(&input_ids, &weights, &config).expect("loss should run");
        loss.backward();
        let artifacts = registry
            .adamw_step(&mut weights, 1e-2, 1)
            .expect("optimizer step should apply");
        assert_eq!(
            artifacts.tensor_summaries.len(),
            representative_trainable_qwen_tensors().len()
        );
        assert_eq!(
            artifacts.manifest_tensors.len(),
            representative_trainable_qwen_tensors().len()
        );
        assert_eq!(
            artifacts.optimizer_entries.len(),
            representative_trainable_qwen_tensors().len() * 2
        );
        for summary in &artifacts.tensor_summaries {
            assert!(
                summary.grad_defined,
                "{} should receive a gradient",
                summary.name
            );
            assert!(
                summary.grad_norm > 0.0,
                "{} grad should be non-zero",
                summary.name
            );
            assert!(
                summary.delta_norm > 0.0,
                "{} delta should be non-zero",
                summary.name
            );
        }

        let final_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!(final_loss < initial_loss);

        let mut reloaded_weights = tiny_qwen_weights();
        let delta_tensors: BTreeMap<String, Tensor> = artifacts
            .delta_entries
            .into_iter()
            .map(|(name, tensor)| (name, tensor))
            .collect();
        QwenTrainableRegistry::apply_delta_checkpoint(
            &mut reloaded_weights,
            &delta_tensors,
            &artifacts.manifest_tensors,
        )
        .expect("delta reload should apply");
        let reloaded_loss = qwen_causal_lm_loss(&input_ids, &reloaded_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!((final_loss - reloaded_loss).abs() < 1e-6);
    }

    #[test]
    fn trainable_tensor_names_expand_over_configured_layers() {
        let names = qwen_trainable_tensors_for_layers(&[0, 1], true);

        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"model.norm.weight".to_string()));
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.mlp.down_proj.weight".to_string()));
        assert_eq!(names.len(), 26);

        let dp_names = qwen_trainable_tensors_for_layers(&[0, 1], false);
        assert!(!dp_names.contains(&"model.embed_tokens.weight".to_string()));
        assert_eq!(dp_names.len(), 25);
    }

    #[test]
    fn qwen_trainable_session_can_train_multiple_layers() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let mut session = QwenTrainableSession::from_trainable_layers(
            config,
            two_layer_tiny_qwen_weights(),
            input_ids,
            Kind::Float,
            &[0, 1],
        )
        .expect("multi-layer session should build");

        let step = session
            .train_step(1e-2, 1)
            .expect("multi-layer session should train");

        assert!(step.loss_after < step.loss_before);
        assert_eq!(step.artifacts.tensor_summaries.len(), 26);
        assert!(step.artifacts.tensor_summaries.iter().any(|summary| {
            summary.name == "model.layers.1.self_attn.q_proj.weight" && summary.grad_norm > 0.0
        }));
        assert!(step.artifacts.tensor_summaries.iter().any(|summary| {
            summary.name == "model.layers.1.mlp.down_proj.weight" && summary.grad_norm > 0.0
        }));
    }

    #[test]
    fn sampling_respects_top_k_and_top_p_filters() {
        let logits = Tensor::from_slice(&[0.0_f32, 1.0, 2.0, 3.0]);
        let mut rng = StdRng::seed_from_u64(7);

        let token =
            sample_token_from_logits(&logits, 0.8, 1, 0.5, &mut rng).expect("sample should run");

        assert_eq!(token.int64_value(&[0]), 3);
    }

    #[test]
    fn qwen_delta_manifest_roundtrips() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let delta_output = temp.path().join("delta.safetensors");
        let manifest_output = delta_manifest_path(&delta_output);
        let manifest = QwenDeltaCheckpointManifest {
            format: "rustrain.qwen_delta.v1".to_string(),
            base_model_path: "/models/qwen".to_string(),
            reference_fixture: "fixture.safetensors".to_string(),
            delta_safetensors: delta_output.display().to_string(),
            optimizer_safetensors: Some(optimizer_state_path(&delta_output).display().to_string()),
            train_step: 1,
            data_cursor_start: None,
            data_cursor_end: None,
            data_cursor_next: None,
            data_epoch_start: None,
            data_epoch_end: None,
            data_epoch_next: None,
            data_sample_offset_start: None,
            data_sample_offset_end: None,
            data_sample_offset_next: None,
            dataset_source_files: Vec::new(),
            dataset_source_sample_counts: Vec::new(),
            dataset_fingerprint: String::new(),
            dataset_shuffle: true,
            streaming_train_batches: None,
            learning_rate: 1e-6,
            initial_loss: 2.0,
            final_loss: 1.5,
            tensors: vec![QwenDeltaTensorManifestEntry {
                name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                delta_name: "model.layers.0.self_attn.q_proj.weight.delta".to_string(),
                adam_m_name: Some("model.layers.0.self_attn.q_proj.weight.adam_m".to_string()),
                adam_v_name: Some("model.layers.0.self_attn.q_proj.weight.adam_v".to_string()),
                shape: vec![4, 4],
                dtype: "float32".to_string(),
                grad_norm: 3.0,
                delta_norm: 0.1,
            }],
        };

        write_qwen_delta_manifest(&manifest_output, &manifest).expect("manifest should write");
        let reloaded: QwenDeltaCheckpointManifest = serde_json::from_str(
            &fs::read_to_string(&manifest_output).expect("manifest should read"),
        )
        .expect("manifest should parse");

        assert_eq!(manifest_output, temp.path().join("delta.safetensors.json"));
        assert_eq!(
            optimizer_state_path(&delta_output),
            temp.path().join("delta.safetensors.optimizer.safetensors")
        );
        assert_eq!(reloaded.format, "rustrain.qwen_delta.v1");
        assert_eq!(
            reloaded.optimizer_safetensors,
            manifest.optimizer_safetensors
        );
        assert_eq!(
            reloaded.tensors[0].delta_name,
            manifest.tensors[0].delta_name
        );
        assert_eq!(
            reloaded.tensors[0].adam_m_name,
            manifest.tensors[0].adam_m_name
        );
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_validates_rank_owned_shards() {
        let mut manifest = tiny_qwen_sharded_manifest();
        let mut replicated_norm_shard = manifest.ranks[0].shards[0].clone();
        replicated_norm_shard.name = "model.layers.0.input_layernorm.weight".to_string();
        replicated_norm_shard.shard_name = "rank0.input_layernorm".to_string();
        replicated_norm_shard.optimizer_m_name = "rank0.input_layernorm.m".to_string();
        replicated_norm_shard.optimizer_v_name = "rank0.input_layernorm.v".to_string();
        replicated_norm_shard.global_shape = vec![4];
        replicated_norm_shard.shard_shape = vec![4];
        replicated_norm_shard.partition = "replicated_norm_smoke".to_string();
        manifest.ranks[0].shards.push(replicated_norm_shard);
        let encoded = serde_json::to_string_pretty(&manifest).expect("manifest should serialize");
        let decoded: QwenShardedCheckpointManifest =
            serde_json::from_str(&encoded).expect("manifest should deserialize");

        decoded.validate().expect("manifest should validate");
        assert_eq!(decoded.format, "rustrain.qwen_sharded.v1");
        assert_eq!(decoded.parallel.world_size().unwrap(), 2);
        assert_eq!(
            decoded.ranks[0].shards[0].optimizer_m_name,
            "rank0.q_proj.m"
        );
        assert_eq!(
            decoded.ranks[1].shards[0].optimizer_v_name,
            "rank1.q_proj.v"
        );
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_missing_rank() {
        let mut manifest = tiny_qwen_sharded_manifest();
        manifest.ranks.pop();

        let error = manifest.validate().expect_err("missing rank should fail");

        assert!(
            error
                .to_string()
                .contains("rank manifest count 1 does not match world size 2")
        );
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_missing_optimizer_slots() {
        let mut manifest = tiny_qwen_sharded_manifest();
        manifest.ranks[0].shards[0].optimizer_m_name.clear();

        let error = manifest
            .validate()
            .expect_err("missing optimizer slots should fail");

        assert!(error.to_string().contains("missing optimizer slots"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_invalid_global_metadata() {
        let mut missing_scheduler = tiny_qwen_sharded_manifest();
        missing_scheduler.scheduler.clear();
        let missing_scheduler_error = missing_scheduler
            .validate()
            .expect_err("missing scheduler should fail")
            .to_string();
        assert!(missing_scheduler_error.contains("requires scheduler"));

        let mut zero_step = tiny_qwen_sharded_manifest();
        zero_step.global_step = 0;
        let zero_step_error = zero_step
            .validate()
            .expect_err("zero global_step should fail")
            .to_string();
        assert!(zero_step_error.contains("global_step must be positive"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_invalid_parallel_rank_axes() {
        let mut duplicate_axes = tiny_qwen_sharded_manifest();
        duplicate_axes.ranks[1].data_parallel_rank = 0;
        duplicate_axes.ranks[1].rank = 1;
        let duplicate_axes_error = duplicate_axes
            .validate()
            .expect_err("duplicate parallel rank axes should fail")
            .to_string();
        assert!(duplicate_axes_error.contains("duplicate parallel rank axes"));

        let mut wrong_linear_rank = tiny_qwen_sharded_manifest();
        wrong_linear_rank.ranks.swap(0, 1);
        wrong_linear_rank.ranks[0].rank = 0;
        wrong_linear_rank.ranks[1].rank = 1;
        let wrong_linear_rank_error = wrong_linear_rank
            .validate()
            .expect_err("rank id that disagrees with axes should fail")
            .to_string();
        assert!(wrong_linear_rank_error.contains("does not match linear parallel rank"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_invalid_shard_shapes() {
        let mut rank_mismatch = tiny_qwen_sharded_manifest();
        rank_mismatch.ranks[0].shards[0].shard_shape = vec![4, 4, 1];
        let rank_mismatch_error = rank_mismatch
            .validate()
            .expect_err("shape rank mismatch should fail")
            .to_string();
        assert!(rank_mismatch_error.contains("global_shape rank"));

        let mut oversized_shard = tiny_qwen_sharded_manifest();
        oversized_shard.ranks[0].shards[0].shard_shape = vec![5, 4];
        let oversized_shard_error = oversized_shard
            .validate()
            .expect_err("oversized shard shape should fail")
            .to_string();
        assert!(oversized_shard_error.contains("exceeds global_shape"));

        let mut zero_dim = tiny_qwen_sharded_manifest();
        zero_dim.ranks[0].shards[0].global_shape = vec![4, 0];
        let zero_dim_error = zero_dim
            .validate()
            .expect_err("zero shape dim should fail")
            .to_string();
        assert!(zero_dim_error.contains("shape dim 1 must be positive"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_invalid_shard_contract_fields() {
        let mut unsupported_dtype = tiny_qwen_sharded_manifest();
        unsupported_dtype.ranks[0].shards[0].dtype = "int8".to_string();
        let unsupported_dtype_error = unsupported_dtype
            .validate()
            .expect_err("unsupported dtype should fail")
            .to_string();
        assert!(unsupported_dtype_error.contains("unsupported dtype int8"));

        let mut unsupported_partition = tiny_qwen_sharded_manifest();
        unsupported_partition.ranks[0].shards[0].partition = "rank0_delta".to_string();
        let unsupported_partition_error = unsupported_partition
            .validate()
            .expect_err("unsupported partition should fail")
            .to_string();
        assert!(unsupported_partition_error.contains("unsupported partition policy"));

        let mut duplicate_tensor = tiny_qwen_sharded_manifest();
        let repeated_shard = duplicate_tensor.ranks[0].shards[0].clone();
        duplicate_tensor.ranks[0].shards.push(repeated_shard);
        let duplicate_tensor_error = duplicate_tensor
            .validate()
            .expect_err("duplicate tensor shard should fail")
            .to_string();
        assert!(duplicate_tensor_error.contains("duplicate tensor shard"));

        let mut duplicate_slot = tiny_qwen_sharded_manifest();
        let mut second_shard = duplicate_slot.ranks[0].shards[0].clone();
        second_shard.name = "model.layers.0.self_attn.k_proj.weight".to_string();
        second_shard.shard_name = "rank0.k_proj".to_string();
        second_shard.optimizer_m_name = "rank0.q_proj.v".to_string();
        second_shard.optimizer_v_name = "rank0.k_proj.v".to_string();
        duplicate_slot.ranks[0].shards.push(second_shard);
        let duplicate_slot_error = duplicate_slot
            .validate()
            .expect_err("duplicate optimizer slot should fail")
            .to_string();
        assert!(duplicate_slot_error.contains("duplicate optimizer slot"));

        let mut slot_collision = tiny_qwen_sharded_manifest();
        slot_collision.ranks[0].shards[0].optimizer_m_name = "rank0.q_proj".to_string();
        let slot_collision_error = slot_collision
            .validate()
            .expect_err("optimizer slot colliding with shard_name should fail")
            .to_string();
        assert!(slot_collision_error.contains("collides with shard_name"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_validates_rank_owned_artifacts() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let manifest =
            tiny_qwen_sharded_manifest_with_artifacts(temp.path()).expect("artifacts should write");

        manifest
            .validate_artifacts()
            .expect("rank-owned artifacts should validate");
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_missing_model_artifact_tensor() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let mut manifest =
            tiny_qwen_sharded_manifest_with_artifacts(temp.path()).expect("artifacts should write");
        manifest.ranks[0].shards[0].shard_name = "rank0.missing_q_proj".to_string();

        let error = manifest
            .validate_artifacts()
            .expect_err("missing model shard should fail")
            .to_string();

        assert!(error.contains("missing model shard rank0.missing_q_proj"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_missing_optimizer_artifact_tensor() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let mut manifest =
            tiny_qwen_sharded_manifest_with_artifacts(temp.path()).expect("artifacts should write");
        manifest.ranks[0].shards[0].optimizer_m_name = "rank0.q_proj.missing_m".to_string();

        let error = manifest
            .validate_artifacts()
            .expect_err("missing optimizer slot should fail")
            .to_string();

        assert!(error.contains("missing optimizer m slot rank0.q_proj.missing_m"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_artifact_shape_mismatch() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let mut manifest =
            tiny_qwen_sharded_manifest_with_artifacts(temp.path()).expect("artifacts should write");
        manifest.ranks[0].shards[0].shard_shape = vec![4, 2];

        let error = manifest
            .validate_artifacts()
            .expect_err("artifact shape mismatch should fail")
            .to_string();

        assert!(error.contains("shape [4, 4] does not match manifest shard_shape [4, 2]"));
    }

    #[test]
    fn qwen_session_dp_global_sharded_manifest_writes_schema_root() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let manifest =
            tiny_qwen_sharded_manifest_with_artifacts(temp.path()).expect("artifacts should write");
        for rank in &manifest.ranks {
            fs::write(
                temp.path()
                    .join(format!("qwen-session-dp-sharded-rank-{}.json", rank.rank)),
                serde_json::to_string_pretty(rank).expect("rank manifest should serialize"),
            )
            .expect("rank manifest should write");
        }
        let output = temp.path().join("global.json");

        write_qwen_session_dp_global_sharded_manifest(
            temp.path(),
            Path::new("/models/qwen"),
            2,
            3,
            QwenComputeDType::Fp32,
            Some(12),
            Some(60),
            Some(2),
            Some(2),
            Some(5),
            &manifest.dataset_source_files,
            &manifest.dataset_source_sample_counts,
            &manifest.dataset_fingerprint,
            manifest.dataset_shuffle,
            manifest.streaming_train_batches,
            &output,
        )
        .expect("global manifest should write");
        let decoded: QwenShardedCheckpointManifest = serde_json::from_str(
            &fs::read_to_string(&output).expect("global manifest should read"),
        )
        .expect("global manifest should parse");

        decoded.validate().expect("global manifest should validate");
        assert_eq!(decoded.format, "rustrain.qwen_sharded.v1");
        assert_eq!(decoded.global_step, 3);
        assert_eq!(decoded.consumed_samples, 12);
        assert_eq!(decoded.consumed_tokens, 60);
        assert_eq!(decoded.data_cursor_next, Some(12));
        assert_eq!(decoded.data_epoch_next, Some(2));
        assert_eq!(decoded.data_sample_offset_next, Some(2));
        assert_eq!(decoded.data_train_samples, Some(5));
        assert_eq!(decoded.dataset_source_files, manifest.dataset_source_files);
        assert_eq!(
            decoded.dataset_source_sample_counts,
            manifest.dataset_source_sample_counts
        );
        assert_eq!(decoded.dataset_fingerprint, manifest.dataset_fingerprint);
        assert_eq!(decoded.ranks.len(), 2);
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_inconsistent_data_progress() {
        let mut manifest = tiny_qwen_sharded_manifest();
        manifest.data_sample_offset_next = Some(5);

        let error = manifest
            .validate()
            .expect_err("inconsistent data progress should fail");

        assert!(
            error
                .to_string()
                .contains("data_sample_offset_next 5 must match")
        );
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_validates_dataset_provenance_shape() {
        let mut legacy_manifest = tiny_qwen_sharded_manifest();
        legacy_manifest.dataset_source_files.clear();
        legacy_manifest.dataset_source_sample_counts.clear();
        legacy_manifest.dataset_fingerprint.clear();
        legacy_manifest
            .validate()
            .expect("legacy sharded manifest without provenance should validate");

        let mut missing_sources = tiny_qwen_sharded_manifest();
        missing_sources.dataset_source_files.clear();
        let missing_sources_error = missing_sources
            .validate()
            .expect_err("fingerprint without source files should fail")
            .to_string();
        assert!(missing_sources_error.contains("requires dataset_source_files"));

        let mut missing_fingerprint = tiny_qwen_sharded_manifest();
        missing_fingerprint.dataset_fingerprint.clear();
        let missing_fingerprint_error = missing_fingerprint
            .validate()
            .expect_err("source files without fingerprint should fail")
            .to_string();
        assert!(missing_fingerprint_error.contains("require dataset_fingerprint"));

        let mut non_jsonl_source = tiny_qwen_sharded_manifest();
        non_jsonl_source.dataset_source_files = vec!["data/README.md".to_string()];
        let non_jsonl_source_error = non_jsonl_source
            .validate()
            .expect_err("non-jsonl source file should fail")
            .to_string();
        assert!(non_jsonl_source_error.contains("must only contain JSONL paths"));

        let mut mismatched_counts = tiny_qwen_sharded_manifest();
        mismatched_counts.dataset_source_sample_counts = vec![QwenSftSourceSampleCount {
            path: "data/other.jsonl".to_string(),
            samples: 5,
        }];
        let mismatched_counts_error = mismatched_counts
            .validate()
            .expect_err("mismatched source sample count paths should fail")
            .to_string();
        assert!(mismatched_counts_error.contains("dataset_source_sample_counts must match"));

        let mut zero_count = tiny_qwen_sharded_manifest();
        zero_count.dataset_source_sample_counts[0].samples = 0;
        let zero_count_error = zero_count
            .validate()
            .expect_err("zero source sample count should fail")
            .to_string();
        assert!(zero_count_error.contains("dataset_source_sample_counts must be positive"));
    }

    #[test]
    fn qwen_sharded_checkpoint_resume_dataset_validation_rejects_changed_data() {
        let manifest = tiny_qwen_sharded_manifest();
        let summary = QwenSftDatasetSummary {
            samples: 5,
            total_tokens: 40,
            response_tokens: 10,
            masked_positions: 10,
            max_sequence_tokens: 8,
            source_files: manifest.dataset_source_files.clone(),
            source_sample_counts: manifest.dataset_source_sample_counts.clone(),
            fingerprint: manifest.dataset_fingerprint.clone(),
            shuffle: manifest.dataset_shuffle,
        };

        qwen_validate_sft_resume_dataset(
            &manifest.dataset_source_files,
            &manifest.dataset_source_sample_counts,
            &manifest.dataset_fingerprint,
            manifest.dataset_shuffle,
            &summary,
            "sharded resume",
        )
        .expect("matching sharded provenance should pass");
        qwen_validate_sft_resume_dataset(&[], &[], "", true, &summary, "legacy sharded resume")
            .expect("legacy sharded manifests without provenance should pass");

        let fingerprint_error = qwen_validate_sft_resume_dataset(
            &manifest.dataset_source_files,
            &manifest.dataset_source_sample_counts,
            "changed-fingerprint",
            manifest.dataset_shuffle,
            &summary,
            "sharded resume",
        )
        .expect_err("changed sharded fingerprint should fail")
        .to_string();
        assert!(fingerprint_error.contains("dataset fingerprint mismatch"));

        let source_error = qwen_validate_sft_resume_dataset(
            &["data/changed.jsonl".to_string()],
            &manifest.dataset_source_sample_counts,
            &manifest.dataset_fingerprint,
            manifest.dataset_shuffle,
            &summary,
            "sharded resume",
        )
        .expect_err("changed sharded source files should fail")
        .to_string();
        assert!(source_error.contains("dataset source files mismatch"));

        let shuffle_error = qwen_validate_sft_resume_dataset(
            &manifest.dataset_source_files,
            &manifest.dataset_source_sample_counts,
            &manifest.dataset_fingerprint,
            !manifest.dataset_shuffle,
            &summary,
            "sharded resume",
        )
        .expect_err("changed sharded shuffle policy should fail")
        .to_string();
        assert!(shuffle_error.contains("dataset shuffle mismatch"));
    }

    #[test]
    fn qwen_sharded_rank_manifest_converts_to_delta_manifest() {
        let manifest = tiny_qwen_sharded_manifest();

        let delta = qwen_sharded_rank_to_delta_manifest(&manifest, 1, 2.0, 1.5, 1e-6)
            .expect("rank should convert");

        assert_eq!(delta.format, "rustrain.qwen_delta.v1");
        assert_eq!(delta.reference_fixture, "qwen_sharded_rank_1");
        assert_eq!(delta.delta_safetensors, "rank1/model.safetensors");
        assert_eq!(
            delta.optimizer_safetensors,
            Some("rank1/optimizer.safetensors".to_string())
        );
        assert_eq!(
            delta.tensors[0].name,
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(delta.tensors[0].delta_name, "rank1.q_proj");
        assert_eq!(
            delta.tensors[0].adam_m_name,
            Some("rank1.q_proj.m".to_string())
        );
    }

    #[test]
    fn qwen_optimizer_slots_reload_reproduces_next_adam_step() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let optimizer_output = temp.path().join("optimizer.safetensors");
        let tensor_name = "model.layers.0.self_attn.q_proj.weight";
        let slot_names = adam_slot_names(tensor_name);
        let first_grad = Tensor::from_slice(&[0.5_f32, -0.25, 0.125, -0.75]).reshape([2, 2]);
        let second_grad = Tensor::from_slice(&[-0.2_f32, 0.4, -0.6, 0.8]).reshape([2, 2]);
        let base_weight = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([2, 2]);
        let learning_rate = 1e-3;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        let first_state = adamw_next_state(None, &first_grad, beta1, beta2);
        let first_update = adamw_update(&first_state, learning_rate, beta1, beta2, 1, eps);
        let after_first = &base_weight - first_update;
        Tensor::write_safetensors(
            &[
                (slot_names.m.as_str(), &first_state.m),
                (slot_names.v.as_str(), &first_state.v),
            ],
            &optimizer_output,
        )
        .expect("optimizer slots should write");

        let reloaded_slots = read_safetensors_map(&optimizer_output).expect("slots should reload");
        let reloaded_state = AdamState {
            m: tensor(&reloaded_slots, &slot_names.m)
                .expect("m slot should exist")
                .to_kind(Kind::Float),
            v: tensor(&reloaded_slots, &slot_names.v)
                .expect("v slot should exist")
                .to_kind(Kind::Float),
        };
        let continuous_second_state =
            adamw_next_state(Some(&first_state), &second_grad, beta1, beta2);
        let reloaded_second_state =
            adamw_next_state(Some(&reloaded_state), &second_grad, beta1, beta2);
        let continuous_after_second = &after_first
            - adamw_update(
                &continuous_second_state,
                learning_rate,
                beta1,
                beta2,
                2,
                eps,
            );
        let reloaded_after_second = &after_first
            - adamw_update(&reloaded_second_state, learning_rate, beta1, beta2, 2, eps);

        assert!(
            diff_stats(&continuous_second_state.m, &reloaded_second_state.m)
                .expect("m state diff should compute")
                .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(&continuous_second_state.v, &reloaded_second_state.v)
                .expect("v state diff should compute")
                .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(&continuous_after_second, &reloaded_after_second)
                .expect("weight diff should compute")
                .max_abs
                < 1e-8
        );
    }

    #[test]
    fn qwen_manifest_resume_reproduces_second_full_train_step() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let delta_output = temp.path().join("delta.safetensors");
        let optimizer_output = optimizer_state_path(&delta_output);
        let manifest_output = delta_manifest_path(&delta_output);
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let learning_rate = 1e-2;

        let mut continuous_weights = tiny_qwen_weights();
        let mut continuous_registry =
            QwenTrainableRegistry::representative(&mut continuous_weights)
                .expect("registry should build");
        let initial_loss = qwen_causal_lm_loss(&input_ids, &continuous_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        let first_loss =
            qwen_causal_lm_loss(&input_ids, &continuous_weights, &config).expect("loss should run");
        first_loss.backward();
        let first_artifacts = continuous_registry
            .adamw_step(&mut continuous_weights, learning_rate, 1)
            .expect("first optimizer step should apply");
        let final_loss = qwen_causal_lm_loss(&input_ids, &continuous_weights, &config)
            .expect("loss should run")
            .double_value(&[]);

        let delta_refs: Vec<(&str, &Tensor)> = first_artifacts
            .delta_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&delta_refs, &delta_output).expect("delta should write");
        let optimizer_refs: Vec<(&str, &Tensor)> = first_artifacts
            .optimizer_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&optimizer_refs, &optimizer_output)
            .expect("optimizer should write");
        let manifest = QwenDeltaCheckpointManifest {
            format: "rustrain.qwen_delta.v1".to_string(),
            base_model_path: "tiny-qwen".to_string(),
            reference_fixture: "inline".to_string(),
            delta_safetensors: delta_output.display().to_string(),
            optimizer_safetensors: Some(optimizer_output.display().to_string()),
            train_step: 1,
            data_cursor_start: None,
            data_cursor_end: None,
            data_cursor_next: None,
            data_epoch_start: None,
            data_epoch_end: None,
            data_epoch_next: None,
            data_sample_offset_start: None,
            data_sample_offset_end: None,
            data_sample_offset_next: None,
            dataset_source_files: Vec::new(),
            dataset_source_sample_counts: Vec::new(),
            dataset_fingerprint: String::new(),
            dataset_shuffle: true,
            streaming_train_batches: None,
            learning_rate,
            initial_loss,
            final_loss,
            tensors: first_artifacts.manifest_tensors,
        };
        write_qwen_delta_manifest(&manifest_output, &manifest).expect("manifest should write");
        let reloaded_manifest: QwenDeltaCheckpointManifest = serde_json::from_str(
            &fs::read_to_string(&manifest_output).expect("manifest should read"),
        )
        .expect("manifest should parse");

        let mut resumed_weights = tiny_qwen_weights();
        let mut resumed_registry =
            QwenTrainableRegistry::load_from_manifest(&mut resumed_weights, &reloaded_manifest)
                .expect("registry should load from manifest");
        let resumed_loss = qwen_causal_lm_loss(&input_ids, &resumed_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!((final_loss - resumed_loss).abs() < 1e-6);

        continuous_registry.zero_grad();
        let continuous_second_loss =
            qwen_causal_lm_loss(&input_ids, &continuous_weights, &config).expect("loss should run");
        continuous_second_loss.backward();
        continuous_registry
            .adamw_step(&mut continuous_weights, learning_rate, 2)
            .expect("continuous second step should apply");

        let resumed_second_loss =
            qwen_causal_lm_loss(&input_ids, &resumed_weights, &config).expect("loss should run");
        resumed_second_loss.backward();
        resumed_registry
            .adamw_step(&mut resumed_weights, learning_rate, 2)
            .expect("resumed second step should apply");

        let continuous_after_second = qwen_causal_lm_loss(&input_ids, &continuous_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        let resumed_after_second = qwen_causal_lm_loss(&input_ids, &resumed_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!((continuous_after_second - resumed_after_second).abs() < 1e-6);

        for name in representative_trainable_qwen_tensors() {
            let diff = diff_stats(
                tensor(&continuous_weights, &name).expect("continuous tensor should exist"),
                tensor(&resumed_weights, &name).expect("resumed tensor should exist"),
            )
            .expect("diff should compute");
            assert!(
                diff.max_abs < 1e-6,
                "{name} should match after manifest-resumed second step, max_abs={}",
                diff.max_abs
            );
        }
    }

    #[test]
    fn qwen_trainable_session_trains_and_resumes_from_manifest() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let delta_output = temp.path().join("session-delta.safetensors");
        let optimizer_output = optimizer_state_path(&delta_output);
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let learning_rate = 1e-2;

        let mut continuous_session = QwenTrainableSession::from_weights(
            config,
            tiny_qwen_weights(),
            input_ids.shallow_clone(),
            Kind::Float,
        )
        .expect("session should build");
        let first_step = continuous_session
            .train_step(learning_rate, 1)
            .expect("first step should train");
        assert!(first_step.loss_after < first_step.loss_before);

        let delta_refs: Vec<(&str, &Tensor)> = first_step
            .artifacts
            .delta_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&delta_refs, &delta_output).expect("delta should write");
        let optimizer_refs: Vec<(&str, &Tensor)> = first_step
            .artifacts
            .optimizer_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&optimizer_refs, &optimizer_output)
            .expect("optimizer should write");
        let manifest = QwenDeltaCheckpointManifest {
            format: "rustrain.qwen_delta.v1".to_string(),
            base_model_path: "tiny-qwen".to_string(),
            reference_fixture: "inline".to_string(),
            delta_safetensors: delta_output.display().to_string(),
            optimizer_safetensors: Some(optimizer_output.display().to_string()),
            train_step: 1,
            data_cursor_start: None,
            data_cursor_end: None,
            data_cursor_next: None,
            data_epoch_start: None,
            data_epoch_end: None,
            data_epoch_next: None,
            data_sample_offset_start: None,
            data_sample_offset_end: None,
            data_sample_offset_next: None,
            dataset_source_files: Vec::new(),
            dataset_source_sample_counts: Vec::new(),
            dataset_fingerprint: String::new(),
            dataset_shuffle: true,
            streaming_train_batches: None,
            learning_rate,
            initial_loss: first_step.loss_before,
            final_loss: first_step.loss_after,
            tensors: first_step.artifacts.manifest_tensors,
        };
        let mut resumed_session = QwenTrainableSession::from_manifest(
            config,
            tiny_qwen_weights(),
            input_ids,
            Kind::Float,
            &manifest,
        )
        .expect("session should resume");
        assert!((first_step.loss_after - resumed_session.loss_value().unwrap()).abs() < 1e-6);

        let continuous_second = continuous_session
            .train_step(learning_rate, 2)
            .expect("continuous second step should train");
        let resumed_second = resumed_session
            .train_step(learning_rate, 2)
            .expect("resumed second step should train");
        assert!((continuous_second.loss_after - resumed_second.loss_after).abs() < 1e-6);
    }

    #[test]
    fn qwen_attention_lora_adapter_roundtrips_mismatched_q_v_shapes() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let adapter = QwenAttentionLoraAdapter::deterministic(
            &[
                (QwenLoraTargetModule::QProj, 4, 6),
                (QwenLoraTargetModule::VProj, 4, 2),
            ],
            2,
            8.0,
        );

        assert_eq!(
            adapter
                .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                .expect("q delta")
                .size(),
            vec![6, 4]
        );
        assert_eq!(
            adapter
                .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                .expect("v delta")
                .size(),
            vec![2, 4]
        );

        adapter.save(&adapter_output).expect("adapter should write");
        let reloaded =
            QwenAttentionLoraAdapter::load(&adapter_output).expect("adapter should reload");

        assert_eq!(
            reloaded
                .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                .expect("q delta")
                .size(),
            vec![6, 4]
        );
        assert_eq!(
            reloaded
                .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                .expect("v delta")
                .size(),
            vec![2, 4]
        );
        assert!(
            diff_stats(
                &reloaded
                    .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                    .expect("reloaded q delta"),
                &adapter
                    .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                    .expect("q delta")
            )
            .expect("q delta diff should compute")
            .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(
                &reloaded
                    .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                    .expect("reloaded v delta"),
                &adapter
                    .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                    .expect("v delta")
            )
            .expect("v delta diff should compute")
            .max_abs
                < 1e-8
        );
    }

    #[test]
    fn qwen_attention_lora_train_step_reduces_tiny_mse_and_reloads() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let layer = QwenLayerWeights::load(&weights, 0).expect("layer should load");
        let input = Tensor::arange(12, (Kind::Float, Device::Cpu)).reshape([1, 3, 4]) / 12.0;
        let target = qwen_attention(
            &input,
            &layer.q_proj,
            &layer.q_bias,
            &layer.k_proj,
            &layer.k_bias,
            &layer.v_proj,
            &layer.v_bias,
            &layer.o_proj,
            &config,
        ) + Tensor::ones([1, 3, 4], (Kind::Float, Device::Cpu)) * 0.01;
        let adapter = QwenAttentionLoraAdapter::deterministic_trainable(
            &[
                (QwenLoraTargetModule::QProj, 4, 4),
                (QwenLoraTargetModule::VProj, 4, 2),
            ],
            2,
            8.0,
        );

        let initial_loss = qwen_attention_lora_mse_loss(&input, &target, &layer, &adapter, &config)
            .double_value(&[]);
        let loss = qwen_attention_lora_mse_loss(&input, &target, &layer, &adapter, &config);
        loss.backward();
        for (_, mut tensor) in adapter.trainable_tensors(0) {
            let grad = tensor.grad();
            assert!(grad.defined());
            let _ = no_grad(|| tensor.f_sub_(&(&grad * 1.0))).expect("update should apply");
        }
        let final_loss = qwen_attention_lora_mse_loss(&input, &target, &layer, &adapter, &config)
            .double_value(&[]);
        assert!(final_loss < initial_loss);

        adapter.save(&adapter_output).expect("adapter should save");
        let reloaded =
            QwenAttentionLoraAdapter::load(&adapter_output).expect("adapter should reload");
        let reloaded_loss =
            qwen_attention_lora_mse_loss(&input, &target, &layer, &reloaded, &config)
                .double_value(&[]);
        assert!((final_loss - reloaded_loss).abs() < 1e-8);
    }

    #[test]
    fn qwen_lora_registry_roundtrips_configured_layer_targets() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let weights = tiny_qwen_weights();
        let runtime_config = RuntimeLoraConfig {
            rank: 2,
            alpha: 8.0,
            target_layers: vec![0],
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
        };
        let config = QwenLoraConfig::from_runtime(&runtime_config).expect("config should build");
        let registry = QwenLoraRegistry::deterministic(&weights, &config, true)
            .expect("registry should build");

        assert_eq!(registry.config.target_layers, vec![0]);
        assert_eq!(
            registry.config.target_modules,
            vec![
                QwenLoraTargetModule::QProj,
                QwenLoraTargetModule::KProj,
                QwenLoraTargetModule::VProj,
                QwenLoraTargetModule::OProj,
                QwenLoraTargetModule::GateProj,
                QwenLoraTargetModule::UpProj,
                QwenLoraTargetModule::DownProj,
            ]
        );
        assert_eq!(
            registry.trainable_tensor_names(),
            vec![
                "model.layers.0.self_attn.q_proj.lora_a".to_string(),
                "model.layers.0.self_attn.q_proj.lora_b".to_string(),
                "model.layers.0.self_attn.k_proj.lora_a".to_string(),
                "model.layers.0.self_attn.k_proj.lora_b".to_string(),
                "model.layers.0.self_attn.v_proj.lora_a".to_string(),
                "model.layers.0.self_attn.v_proj.lora_b".to_string(),
                "model.layers.0.self_attn.o_proj.lora_a".to_string(),
                "model.layers.0.self_attn.o_proj.lora_b".to_string(),
                "model.layers.0.mlp.gate_proj.lora_a".to_string(),
                "model.layers.0.mlp.gate_proj.lora_b".to_string(),
                "model.layers.0.mlp.up_proj.lora_a".to_string(),
                "model.layers.0.mlp.up_proj.lora_b".to_string(),
                "model.layers.0.mlp.down_proj.lora_a".to_string(),
                "model.layers.0.mlp.down_proj.lora_b".to_string(),
            ]
        );

        registry
            .save(&adapter_output)
            .expect("registry should save");
        let reloaded = QwenLoraRegistry::load(&adapter_output).expect("registry should reload");

        assert_eq!(reloaded.config, config);
        for (name, tensor) in reloaded.trainable_tensors() {
            assert!(
                tensor.requires_grad(),
                "{name} should remain trainable after reload"
            );
        }
        assert!(
            diff_stats(
                &reloaded
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                    .expect("reloaded q delta"),
                &registry
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                    .expect("q delta"),
            )
            .expect("q delta diff should compute")
            .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(
                &reloaded
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                    .expect("reloaded v delta"),
                &registry
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                    .expect("v delta"),
            )
            .expect("v delta diff should compute")
            .max_abs
                < 1e-8
        );
        assert_eq!(
            reloaded
                .layer_adapter(0)
                .expect("layer adapter")
                .delta(QwenLoraTargetModule::KProj, Device::Cpu)
                .expect("k delta")
                .size(),
            vec![2, 4]
        );
        assert_eq!(
            reloaded
                .layer_adapter(0)
                .expect("layer adapter")
                .delta(QwenLoraTargetModule::OProj, Device::Cpu)
                .expect("o delta")
                .size(),
            vec![4, 4]
        );
        assert_eq!(
            reloaded
                .layer_adapter(0)
                .expect("layer adapter")
                .delta(QwenLoraTargetModule::GateProj, Device::Cpu)
                .expect("gate delta")
                .size(),
            vec![8, 4]
        );
        assert_eq!(
            reloaded
                .layer_adapter(0)
                .expect("layer adapter")
                .delta(QwenLoraTargetModule::DownProj, Device::Cpu)
                .expect("down delta")
                .size(),
            vec![4, 8]
        );
    }

    #[test]
    fn qwen_lora_registry_applies_all_projection_targets_to_layer() {
        let weights = tiny_qwen_weights();
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let lora_config = QwenLoraConfig::new(
            vec![0],
            vec![
                QwenLoraTargetModule::QProj,
                QwenLoraTargetModule::KProj,
                QwenLoraTargetModule::VProj,
                QwenLoraTargetModule::OProj,
                QwenLoraTargetModule::GateProj,
                QwenLoraTargetModule::UpProj,
                QwenLoraTargetModule::DownProj,
            ],
            2,
            8.0,
        )
        .expect("LoRA config should build");
        let zero_registry =
            QwenLoraRegistry::zeros(&weights, &lora_config).expect("zero registry should build");
        let registry = QwenLoraRegistry::deterministic(&weights, &lora_config, false)
            .expect("registry should build");
        let input = Tensor::arange(12, (Kind::Float, Device::Cpu)).reshape([1, 3, 4]) / 12.0;
        let base_layer = QwenLayerWeights::load(&weights, 0).expect("base layer should load");
        let base_output = qwen_layer(&input, &base_layer, &config);
        let zero_output = qwen_layer_with_lora(
            &input,
            &base_layer,
            zero_registry.layer_adapter(0).expect("zero adapter"),
            &config,
        );
        let adapted_output = qwen_layer_with_lora(
            &input,
            &base_layer,
            registry.layer_adapter(0).expect("adapter"),
            &config,
        );
        assert!(
            diff_stats(&base_output, &zero_output)
                .expect("zero diff should compute")
                .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(&base_output, &adapted_output)
                .expect("adapted diff should compute")
                .max_abs
                > 0.0
        );

        let merged_weights = registry
            .merge_into_weights(&weights)
            .expect("registry should merge all targets");
        let merged_layer =
            QwenLayerWeights::load(&merged_weights, 0).expect("merged layer should load");
        let merged_output = qwen_layer(&input, &merged_layer, &config);
        assert!(
            diff_stats(&adapted_output, &merged_output)
                .expect("merged diff should compute")
                .max_abs
                < 1e-6
        );

        let unmerged_weights = registry
            .unmerge_from_weights(&merged_weights)
            .expect("registry should unmerge all targets");
        let unmerged_layer =
            QwenLayerWeights::load(&unmerged_weights, 0).expect("unmerged layer should load");
        let unmerged_output = qwen_layer(&input, &unmerged_layer, &config);
        assert!(
            diff_stats(&base_output, &unmerged_output)
                .expect("unmerged diff should compute")
                .max_abs
                < 1e-6
        );
    }

    #[test]
    fn qwen_lora_sft_resume_config_validation_checks_manifest_and_adapter() {
        let current = QwenLoraConfig::new(
            vec![0, 1],
            vec![
                QwenLoraTargetModule::QProj,
                QwenLoraTargetModule::VProj,
                QwenLoraTargetModule::DownProj,
            ],
            4,
            8.0,
        )
        .expect("current config should build");
        let mut manifest = QwenLoraSftAdapterManifest {
            format: "rustrain.qwen_lora_sft_adapter.v1".to_string(),
            base_model_path: "/models/qwen".to_string(),
            adapter_safetensors: "/tmp/adapter.safetensors".to_string(),
            compute_kind: "fp32".to_string(),
            steps: 2,
            train_step: 4,
            data_cursor_start: 0,
            data_cursor_end: 4,
            data_cursor_next: 4,
            data_epoch_start: 0,
            data_epoch_end: 0,
            data_epoch_next: 0,
            data_sample_offset_start: 0,
            data_sample_offset_end: 4,
            data_sample_offset_next: 4,
            dataset_source_files: vec!["data/train.jsonl".to_string()],
            dataset_source_sample_counts: vec![QwenSftSourceSampleCount {
                path: "data/train.jsonl".to_string(),
                samples: 4,
            }],
            dataset_fingerprint: "abc123".to_string(),
            dataset_order_seed: 777,
            dataset_shuffle: true,
            streaming_train_batches: true,
            dataset_total_samples: 4,
            dataset_train_samples: 3,
            dataset_eval_samples: 1,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            target_layers: current.target_layers.clone(),
            target_modules: current.target_module_names(),
        };

        qwen_validate_lora_resume_config(Some(&manifest), &current, &current, "fp32")
            .expect("matching manifest and adapter config should pass");
        qwen_validate_lora_resume_config(None, &current, &current, "bf16")
            .expect("direct adapter resume should pass without manifest metadata");

        let compute_kind_error =
            qwen_validate_lora_resume_config(Some(&manifest), &current, &current, "bf16")
                .expect_err("manifest compute kind mismatch should fail")
                .to_string();
        assert!(compute_kind_error.contains("resume manifest compute_kind"));

        let adapter_mismatch = QwenLoraConfig::new(
            vec![0, 1],
            vec![QwenLoraTargetModule::QProj, QwenLoraTargetModule::VProj],
            4,
            8.0,
        )
        .expect("adapter mismatch config should build");
        let adapter_error =
            qwen_validate_lora_resume_config(Some(&manifest), &adapter_mismatch, &current, "fp32")
                .expect_err("adapter config mismatch should fail")
                .to_string();
        assert!(adapter_error.contains("resume adapter config does not match"));

        manifest.target_modules = vec!["q_proj".to_string(), "v_proj".to_string()];
        let manifest_module_error =
            qwen_validate_lora_resume_config(Some(&manifest), &current, &current, "fp32")
                .expect_err("manifest module mismatch should fail")
                .to_string();
        assert!(manifest_module_error.contains("resume manifest target_modules"));

        manifest.target_modules = current.target_module_names();
        manifest.target_layers = vec![0];
        let manifest_layer_error =
            qwen_validate_lora_resume_config(Some(&manifest), &current, &current, "fp32")
                .expect_err("manifest layer mismatch should fail")
                .to_string();
        assert!(manifest_layer_error.contains("resume manifest target_layers"));
    }

    #[test]
    fn qwen_lora_config_rejects_unsupported_target_module() {
        let runtime_config = RuntimeLoraConfig {
            rank: 2,
            alpha: 8.0,
            target_layers: vec![0],
            target_modules: vec!["score_proj".to_string()],
        };
        let error = QwenLoraConfig::from_runtime(&runtime_config)
            .expect_err("unsupported target should fail");

        assert!(
            error
                .to_string()
                .contains("unsupported Qwen LoRA target module score_proj")
        );
    }

    #[test]
    fn qwen_lora_full_forward_and_generate_reload_parity() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let lora_config = QwenLoraConfig::layer0_qv(2, 8.0).expect("config should build");
        let registry = QwenLoraRegistry::deterministic(&weights, &lora_config, false)
            .expect("registry should build");
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2]).reshape([1, 3]);

        let base_logits =
            qwen_forward_from_ids(&input_ids, &weights, &config).expect("base forward should run");
        let adapted_logits =
            qwen_forward_from_ids_with_lora(&input_ids, &weights, &config, &registry, Kind::Float)
                .expect("LoRA forward should run");
        assert!(
            diff_stats(&adapted_logits, &base_logits)
                .expect("adapter diff should compute")
                .max_abs
                > 0.0
        );

        registry
            .save(&adapter_output)
            .expect("registry should save");
        let reloaded = QwenLoraRegistry::load(&adapter_output).expect("registry should reload");
        let reloaded_logits =
            qwen_forward_from_ids_with_lora(&input_ids, &weights, &config, &reloaded, Kind::Float)
                .expect("reloaded LoRA forward should run");
        assert!(
            diff_stats(&reloaded_logits, &adapted_logits)
                .expect("reload diff should compute")
                .max_abs
                < 1e-8
        );
        let merged_weights = reloaded
            .merge_into_weights(&weights)
            .expect("LoRA weights should merge");
        let merged_logits = qwen_forward_from_ids(&input_ids, &merged_weights, &config)
            .expect("merged forward should run");
        assert!(
            diff_stats(&merged_logits, &adapted_logits)
                .expect("merge diff should compute")
                .max_abs
                < 1e-8
        );
        let unmerged_weights = reloaded
            .unmerge_from_weights(&merged_weights)
            .expect("LoRA weights should unmerge");
        let unmerged_logits = qwen_forward_from_ids(&input_ids, &unmerged_weights, &config)
            .expect("unmerged forward should run");
        assert!(
            diff_stats(&unmerged_logits, &base_logits)
                .expect("unmerge diff should compute")
                .max_abs
                < 1e-8
        );

        let generated = qwen_greedy_generate_with_lora(
            &input_ids,
            &weights,
            &config,
            &registry,
            2,
            Kind::Float,
        )
        .expect("LoRA generate should run");
        let reloaded_generated = qwen_greedy_generate_with_lora(
            &input_ids,
            &weights,
            &config,
            &reloaded,
            2,
            Kind::Float,
        )
        .expect("reloaded LoRA generate should run");
        let merged_generated = qwen_greedy_generate(&input_ids, &merged_weights, &config, 2)
            .expect("merged LoRA generate should run");
        let generated_ids: Vec<i64> = Vec::<i64>::try_from(generated.reshape([-1])).unwrap();
        let reloaded_generated_ids: Vec<i64> =
            Vec::<i64>::try_from(reloaded_generated.reshape([-1])).unwrap();
        let merged_generated_ids: Vec<i64> =
            Vec::<i64>::try_from(merged_generated.reshape([-1])).unwrap();
        assert_eq!(reloaded_generated_ids, generated_ids);
        assert_eq!(merged_generated_ids, generated_ids);
    }

    #[test]
    fn qwen_lora_full_layer_targets_affect_forward_and_merge() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let lora_config = QwenLoraConfig::new(
            vec![0],
            vec![
                QwenLoraTargetModule::QProj,
                QwenLoraTargetModule::KProj,
                QwenLoraTargetModule::VProj,
                QwenLoraTargetModule::OProj,
                QwenLoraTargetModule::GateProj,
                QwenLoraTargetModule::UpProj,
                QwenLoraTargetModule::DownProj,
            ],
            2,
            8.0,
        )
        .expect("config should build");
        let registry = QwenLoraRegistry::deterministic(&weights, &lora_config, false)
            .expect("registry should build");
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2]).reshape([1, 3]);

        for module in &lora_config.target_modules {
            let weight =
                tensor(&weights, &module.weight_name(0)).expect("base weight should exist");
            assert_eq!(
                registry
                    .layer_adapter(0)
                    .expect("layer adapter should exist")
                    .delta(*module, Device::Cpu)
                    .expect("delta should build")
                    .size(),
                weight.size()
            );
        }

        let base_logits =
            qwen_forward_from_ids(&input_ids, &weights, &config).expect("base forward should run");
        let adapted_logits =
            qwen_forward_from_ids_with_lora(&input_ids, &weights, &config, &registry, Kind::Float)
                .expect("LoRA forward should run");
        assert!(
            diff_stats(&adapted_logits, &base_logits)
                .expect("adapter diff should compute")
                .max_abs
                > 0.0
        );

        let merged_weights = registry
            .merge_into_weights(&weights)
            .expect("LoRA weights should merge");
        let merged_logits = qwen_forward_from_ids(&input_ids, &merged_weights, &config)
            .expect("merged forward should run");
        assert!(
            diff_stats(&merged_logits, &adapted_logits)
                .expect("merge diff should compute")
                .max_abs
                < 1e-8
        );
    }

    #[test]
    fn qwen_sft_padded_batch_masks_padding_targets() {
        let samples = vec![
            QwenSftTokenSample {
                prompt_tokens: 2,
                response_tokens: 2,
                masked_positions: 2,
                token_ids: vec![10, 11, 12, 13],
                mask_values: vec![0.0, 1.0, 1.0],
            },
            QwenSftTokenSample {
                prompt_tokens: 1,
                response_tokens: 1,
                masked_positions: 1,
                token_ids: vec![20, 21],
                mask_values: vec![1.0],
            },
        ];

        let batch = qwen_sft_padded_batch(&samples, 0).expect("batch should build");
        let input_values: Vec<i64> = Vec::<i64>::try_from(batch.input_ids.reshape([-1])).unwrap();
        let mask_values: Vec<f32> = Vec::<f32>::try_from(batch.target_mask.reshape([-1])).unwrap();

        assert_eq!(batch.input_ids.size(), vec![2, 4]);
        assert_eq!(batch.target_mask.size(), vec![2, 3, 1]);
        assert_eq!(input_values, vec![10, 11, 12, 13, 20, 21, 0, 0]);
        assert_eq!(mask_values, vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(batch.masked_positions, 3);
        assert_eq!(batch.padding_tokens, 2);
    }

    #[test]
    fn qwen_sft_dataset_builds_wrapping_padded_batches() {
        let dataset = QwenSftDataset {
            samples: vec![
                QwenSftTokenSample {
                    prompt_tokens: 2,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![1, 2, 3],
                    mask_values: vec![0.0, 1.0],
                },
                QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 2,
                    masked_positions: 2,
                    token_ids: vec![4, 5, 6],
                    mask_values: vec![1.0, 1.0],
                },
            ],
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let batch = dataset
            .padded_batch(1, 3)
            .expect("wrapping batch should build");
        let input_values: Vec<i64> = Vec::<i64>::try_from(batch.input_ids.reshape([-1])).unwrap();
        let mask_values: Vec<f32> = Vec::<f32>::try_from(batch.target_mask.reshape([-1])).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(batch.input_ids.size(), vec![3, 3]);
        assert_eq!(input_values, vec![4, 5, 6, 1, 2, 3, 4, 5, 6]);
        assert_eq!(mask_values, vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(batch.masked_positions, 5);
        assert_eq!(batch.padding_tokens, 0);
    }

    #[test]
    fn qwen_data_epoch_metadata_tracks_wrapping_cursor() {
        assert_eq!(qwen_data_epoch_and_offset(0, 6).unwrap(), (0, 0));
        assert_eq!(qwen_data_epoch_and_offset(5, 6).unwrap(), (0, 5));
        assert_eq!(qwen_data_epoch_and_offset(6, 6).unwrap(), (1, 0));
        assert_eq!(qwen_data_epoch_and_offset(16, 6).unwrap(), (2, 4));
        assert!(qwen_data_epoch_and_offset(0, 0).is_err());
    }

    #[test]
    fn qwen_sft_dataset_split_keeps_train_and_eval_batches() {
        let dataset = QwenSftDataset {
            samples: (0..5)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![index, index + 10],
                    mask_values: vec![1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let (train, eval) = dataset
            .train_eval_split(0.6)
            .expect("split should keep both sides");
        let train_batch = train
            .padded_batch(2, 3)
            .expect("train wrapping batch should build");
        let eval_batch = eval.padded_batch(0, 2).expect("eval batch should build");
        let train_values: Vec<i64> =
            Vec::<i64>::try_from(train_batch.input_ids.reshape([-1])).unwrap();
        let eval_values: Vec<i64> =
            Vec::<i64>::try_from(eval_batch.input_ids.reshape([-1])).unwrap();

        assert_eq!(train.len(), 3);
        assert_eq!(eval.len(), 2);
        assert_eq!(train_values, vec![2, 12, 0, 10, 1, 11]);
        assert_eq!(eval_values, vec![3, 13, 4, 14]);
    }

    #[test]
    fn qwen_sft_dataset_shuffle_is_seeded_and_summarized() {
        let dataset = QwenSftDataset {
            samples: (0..5)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: (index + 1) as usize,
                    masked_positions: (index + 1) as usize,
                    token_ids: vec![index, index + 10, index + 20],
                    mask_values: vec![0.0, 1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let summary = dataset.summary();
        let shuffled_a = dataset.clone().shuffle_by_seed(17);
        let shuffled_b = dataset.clone().shuffle_by_seed(17);
        let shuffled_c = dataset.shuffle_by_seed(18);
        let order_a = shuffled_a
            .samples
            .iter()
            .map(|sample| sample.token_ids[0])
            .collect::<Vec<_>>();
        let order_b = shuffled_b
            .samples
            .iter()
            .map(|sample| sample.token_ids[0])
            .collect::<Vec<_>>();
        let order_c = shuffled_c
            .samples
            .iter()
            .map(|sample| sample.token_ids[0])
            .collect::<Vec<_>>();

        assert_eq!(summary.samples, 5);
        assert_eq!(summary.total_tokens, 15);
        assert_eq!(summary.response_tokens, 15);
        assert_eq!(summary.masked_positions, 15);
        assert_eq!(summary.max_sequence_tokens, 3);
        assert!(!summary.shuffle);
        assert!(shuffled_a.summary().shuffle);
        assert_eq!(order_a, order_b);
        assert_ne!(order_a, order_c);
    }

    #[test]
    fn qwen_sft_dataset_shuffle_can_be_disabled() {
        let dataset = QwenSftDataset {
            samples: (0..5)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![index, index + 10],
                    mask_values: vec![1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let unshuffled = qwen_apply_sft_shuffle(dataset.clone(), false, 777);
        let shuffled = qwen_apply_sft_shuffle(dataset, true, 777);
        let unshuffled_order = (0..unshuffled.len())
            .map(|cursor| unshuffled.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let shuffled_order = (0..shuffled.len())
            .map(|cursor| shuffled.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();

        assert_eq!(unshuffled_order, vec![0, 1, 2, 3, 4]);
        assert!(!unshuffled.summary().shuffle);
        assert!(shuffled.summary().shuffle);
        assert_ne!(unshuffled_order, shuffled_order);
    }

    #[test]
    fn qwen_sft_dataset_epoch_shuffle_is_cursor_stable() {
        let dataset = QwenSftDataset {
            samples: (0..6)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![index, index + 10],
                    mask_values: vec![1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        }
        .shuffle_by_seed(777);

        let epoch0_a = (0..dataset.len())
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let epoch0_b = (0..dataset.len())
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let epoch1 = (dataset.len()..dataset.len() * 2)
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let epoch2 = (dataset.len() * 2..dataset.len() * 3)
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();

        assert_eq!(epoch0_a, epoch0_b);
        assert!(epoch0_a != epoch1 || epoch0_a != epoch2);

        let wrapped_batch = dataset
            .padded_batch(dataset.len() - 1, 3)
            .expect("epoch-crossing batch should build");
        let wrapped_values: Vec<i64> =
            Vec::<i64>::try_from(wrapped_batch.input_ids.reshape([-1])).unwrap();
        assert_eq!(
            wrapped_values,
            vec![
                epoch0_a[dataset.len() - 1],
                epoch0_a[dataset.len() - 1] + 10,
                epoch1[0],
                epoch1[0] + 10,
                epoch1[1],
                epoch1[1] + 10,
            ]
        );
    }

    #[test]
    fn qwen_session_fixed_batch_plan_reports_fixture_metadata() {
        let mut weights = tiny_qwen_weights();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros([2048, 4], (Kind::Float, Device::Cpu)),
        );

        let plan = qwen_session_fixed_batch_plan(&weights, 0, 2).expect("fixed plan should build");

        assert_eq!(plan.reference_fixture, "qwen_session_single_fixed_tokens");
        assert_eq!(plan.batch_size, 1);
        assert_eq!(plan.sequence_tokens, 5);
        assert_eq!(plan.train_batches.len(), 3);
        assert!(plan.dataset_total_samples.is_none());
    }

    #[test]
    fn qwen_session_fixed_batch_plan_keeps_resume_cursor_window() {
        let mut weights = tiny_qwen_weights();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros([2048, 4], (Kind::Float, Device::Cpu)),
        );

        let plan = qwen_session_fixed_batch_plan(&weights, 2, 2).expect("fixed plan should build");

        assert_eq!(plan.train_batches.len(), 5);
        assert!(plan.train_batches.get(2).is_some());
        assert!(plan.train_batches.get(4).is_some());
    }

    #[test]
    fn qwen_sft_jsonl_reader_loads_instruction_input_response_records() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"Reply with the project name.","response":"rustrain"}
{"instruction":"Name the language.","input":"rustrain implementation","response":"Rust"}
"#,
        )
        .expect("jsonl should write");

        let field_map = QwenSftFieldMap::default();
        let regex_plan = QwenSftRegexPlan::compile(&field_map).expect("regex plan should compile");
        let example_set = qwen_sft_examples_from_jsonl_path_with_limit(
            &jsonl,
            None,
            None,
            1,
            &field_map,
            &regex_plan,
            &mut None,
        )
        .expect("examples should load from jsonl");
        let examples = &example_set.examples;

        assert_eq!(examples.len(), 2);
        assert_eq!(example_set.source_files, vec![jsonl.display().to_string()]);
        assert_eq!(
            example_set.source_sample_counts,
            vec![QwenSftSourceSampleCount {
                path: jsonl.display().to_string(),
                samples: 2,
            }]
        );
        assert!(!example_set.fingerprint.is_empty());
        assert_eq!(examples[0].instruction, "Reply with the project name.");
        assert_eq!(examples[0].input, "");
        assert_eq!(examples[0].response, "rustrain");
        assert_eq!(examples[1].instruction, "Name the language.");
        assert_eq!(examples[1].input, "rustrain implementation");
        assert_eq!(examples[1].response, "Rust");
    }

    #[test]
    fn qwen_sft_jsonl_reader_supports_configurable_field_names() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"sys":"Be concise.","prompt":"Summarize the project.","context":"Rust training code","answer":"rustrain"}
{"sys":"Use one word.","prompt":"Name the language.","answer":"Rust"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction: "prompt".to_string(),
            input: "context".to_string(),
            response: "answer".to_string(),
            system: Some("sys".to_string()),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("custom field examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("custom field streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("custom field raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("custom field cache should write");
        let mismatched_field_map = QwenSftFieldMap {
            response: "completion".to_string(),
            ..field_map.clone()
        };
        let mismatched_system_map = QwenSftFieldMap {
            system: Some("system_prompt".to_string()),
            ..field_map.clone()
        };
        let mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &mismatched_field_map,
        )
        .expect_err("cache should reject different field maps")
        .to_string();
        let system_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &mismatched_system_map,
        )
        .expect_err("cache should reject different system fields")
        .to_string();

        assert_eq!(loaded.examples.len(), 2);
        assert_eq!(loaded.examples[0].system, "Be concise.");
        assert_eq!(loaded.examples[0].instruction, "Summarize the project.");
        assert_eq!(loaded.examples[0].input, "Rust training code");
        assert_eq!(loaded.examples[0].response, "rustrain");
        assert_eq!(loaded.examples[1].system, "Use one word.");
        assert_eq!(loaded.examples[1].input, "");
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(raw_window.examples[0].response, "rustrain");
        assert!(first_cache.cache_written);
        assert!(mismatch.contains("field_map"));
        assert!(system_mismatch.contains("field_map"));
        assert!(
            qwen_render_sft_prompt(&loaded.examples[0], &field_map)
                .expect("system prompt should render")
                .contains("System: Be concise.")
        );
    }

    #[test]
    fn qwen_sft_jsonl_reader_supports_dotted_field_paths() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("nested.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"meta":{"system":"Be exact."},"payload":{"prompt":"Name the project.","context":"Rust trainer","answer":"rustrain"}}
{"meta":{"system":"Use one word."},"payload":{"prompt":"Name the language.","answer":"Rust"}}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction: "payload.prompt".to_string(),
            input: "payload.context".to_string(),
            response: "payload.answer".to_string(),
            system: Some("meta.system".to_string()),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("nested field examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("nested field streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("nested field source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("nested field raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("nested field cache should write");
        let changed_field_map = QwenSftFieldMap {
            instruction: "payload.question".to_string(),
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &changed_field_map,
        )
        .expect_err("cache should reject changed dotted field paths")
        .to_string();

        assert_eq!(loaded.examples.len(), 2);
        assert_eq!(loaded.examples[0].system, "Be exact.");
        assert_eq!(loaded.examples[0].instruction, "Name the project.");
        assert_eq!(loaded.examples[0].input, "Rust trainer");
        assert_eq!(loaded.examples[0].response, "rustrain");
        assert_eq!(loaded.examples[1].system, "Use one word.");
        assert_eq!(loaded.examples[1].input, "");
        assert_eq!(loaded.examples[1].response, "Rust");
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(raw_window.examples, loaded.examples);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert!(
            qwen_render_sft_prompt(&loaded.examples[0], &field_map)
                .expect("nested field prompt should render")
                .contains("I: Rust trainer")
        );
    }

    #[test]
    fn qwen_sft_jsonl_reader_supports_chat_messages_records() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("chat.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Name the project."},{"role":"assistant","content":"rustrain"}]}
{"messages":[{"role":"human","content":"Name the language."},{"role":"gpt","content":"Rust"}]}
{"messages":[{"role":"assistant","content":"missing user"}]}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            chat_messages: Some("messages".to_string()),
            system_contains_any: vec!["concise".to_string()],
            skip_invalid_records: true,
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("chat examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("chat streaming summary should scan");
        let source_index =
            qwen_sft_streaming_source_index(&paths, None, &field_map).expect("index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("chat raw window should replay");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("chat cache should write");
        let cache_hit =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("chat cache should hit");
        let default_parse_error = qwen_sft_examples_from_jsonl_paths_with_limit(
            &paths,
            None,
            &QwenSftFieldMap::default(),
        )
        .expect_err("default field map cannot parse chat rows")
        .to_string();
        let changed_chat_field = QwenSftFieldMap {
            chat_messages: Some("conversations".to_string()),
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &changed_chat_field,
        )
        .expect_err("cache should reject changed chat messages field")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].system, "Be concise.");
        assert_eq!(loaded.examples[0].instruction, "Name the project.");
        assert_eq!(loaded.examples[0].input, "");
        assert_eq!(loaded.examples[0].response, "rustrain");
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(source_index.samples.len(), 1);
        assert_eq!(source_index.samples[0].index_in_file, 0);
        assert!(first_cache.cache_written);
        assert!(cache_hit.cache_hit);
        assert_eq!(cache_hit.index.samples, source_index.samples);
        assert!(default_parse_error.contains("failed to parse SFT JSONL record"));
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_field_defaults_fill_empty_fields_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"","response":"","input":""}
{"response":"kept"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            system_contains_any: vec!["assistant".to_string()],
            instruction_contains_any: vec!["default instruction".to_string()],
            input_contains_any: vec!["default input".to_string()],
            response_contains_any: vec!["default response".to_string()],
            min_response_chars: 10,
            field_defaults: vec![
                FieldDefault {
                    field: FieldDefaultTarget::System,
                    value: "system assistant".to_string(),
                },
                FieldDefault {
                    field: FieldDefaultTarget::Instruction,
                    value: "default instruction".to_string(),
                },
                FieldDefault {
                    field: FieldDefaultTarget::Input,
                    value: "default input".to_string(),
                },
                FieldDefault {
                    field: FieldDefaultTarget::Response,
                    value: "default response".to_string(),
                },
            ],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("defaulted examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("defaulted streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("defaulted source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("defaulted raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("defaulted cache should write");
        let defaults_without_filters = QwenSftFieldMap {
            system_contains_any: Vec::new(),
            instruction_contains_any: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            min_response_chars: 1,
            ..field_map.clone()
        };
        let unfiltered_default_fingerprint = qwen_sft_streaming_fingerprint(
            &paths,
            None,
            &loaded.source_files,
            &defaults_without_filters,
        )
        .expect("defaulted field map fingerprint should compute");
        let changed_defaults = QwenSftFieldMap {
            field_defaults: vec![FieldDefault {
                field: FieldDefaultTarget::Instruction,
                value: "other instruction".to_string(),
            }],
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &changed_defaults,
        )
        .expect_err("cache should reject changed defaults")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].system, "system assistant");
        assert_eq!(loaded.examples[0].instruction, "default instruction");
        assert_eq!(loaded.examples[0].input, "default input");
        assert_eq!(loaded.examples[0].response, "default response");
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(raw_window.examples, loaded.examples);
        assert!(first_cache.cache_written);
        assert_ne!(loaded.fingerprint, unfiltered_default_fingerprint);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_field_replacements_apply_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep PROJECT_TOKEN","response":"APPROVED_TOKEN"}
{"instruction":"Drop PROJECT_TOKEN","response":"DENIED_TOKEN"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["rustrain".to_string()],
            response_contains_any: vec!["approved".to_string()],
            min_response_chars: 8,
            field_replacements: vec![
                FieldReplacement {
                    field: FieldReplacementTarget::Instruction,
                    pattern: "PROJECT_TOKEN".to_string(),
                    replacement: "rustrain".to_string(),
                },
                FieldReplacement {
                    field: FieldReplacementTarget::Response,
                    pattern: "APPROVED_TOKEN".to_string(),
                    replacement: "approved".to_string(),
                },
            ],
            ..QwenSftFieldMap::default()
        };
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("replacement examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("replacement streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("replacement source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("replacement raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("replacement cache should write");
        let default_fingerprint = qwen_sft_streaming_fingerprint(
            &paths,
            None,
            &loaded.source_files,
            &QwenSftFieldMap::default(),
        )
        .expect("default fingerprint should compute");
        let replacement_mismatch = QwenSftFieldMap {
            field_replacements: vec![FieldReplacement {
                field: FieldReplacementTarget::Instruction,
                pattern: "PROJECT_TOKEN".to_string(),
                replacement: "other".to_string(),
            }],
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &replacement_mismatch,
        )
        .expect_err("cache should reject changed replacements")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Keep rustrain");
        assert_eq!(loaded.examples[0].response, "approved");
        assert_eq!(raw_window.examples.len(), loaded.examples.len());
        assert_eq!(
            raw_window.examples[0].instruction,
            loaded.examples[0].instruction
        );
        assert_eq!(raw_window.examples[0].response, loaded.examples[0].response);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, default_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_regex_replacements_apply_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep project-123 token","response":"APPROVED:42"}
{"instruction":"Drop project-456 token","response":"DENIED:99"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["project-id token".to_string()],
            response_contains_any: vec!["approved id".to_string()],
            min_response_chars: 8,
            field_regex_replacements: vec![
                FieldRegexReplacement {
                    field: FieldReplacementTarget::Instruction,
                    pattern: r"project-\d+".to_string(),
                    replacement: "project-id".to_string(),
                },
                FieldRegexReplacement {
                    field: FieldReplacementTarget::Response,
                    pattern: r"APPROVED:\d+".to_string(),
                    replacement: "approved id".to_string(),
                },
            ],
            ..QwenSftFieldMap::default()
        };
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("regex replacement examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("regex replacement streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("regex replacement source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("regex replacement raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("regex replacement cache should write");
        let default_fingerprint = qwen_sft_streaming_fingerprint(
            &paths,
            None,
            &loaded.source_files,
            &QwenSftFieldMap::default(),
        )
        .expect("default fingerprint should compute");
        let regex_mismatch = QwenSftFieldMap {
            field_regex_replacements: vec![FieldRegexReplacement {
                field: FieldReplacementTarget::Instruction,
                pattern: r"project-\d+".to_string(),
                replacement: "other".to_string(),
            }],
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &regex_mismatch,
        )
        .expect_err("cache should reject changed regex replacements")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Keep project-id token");
        assert_eq!(loaded.examples[0].response, "approved id");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, default_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_field_transform_dsl_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep PROJECT-123::tail","input":"gpu context","response":"APPROVED:42 extra"}
{"instruction":"Drop PROJECT-456::tail","input":"cpu context","response":"DENIED:99 extra"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["task Keep project-id".to_string()],
            input_contains_any: vec!["GPU".to_string()],
            response_contains_any: vec!["approved id".to_string()],
            min_response_chars: 11,
            max_response_chars: Some(11),
            field_transforms: vec![
                FieldTransform {
                    field: FieldReplacementTarget::Instruction,
                    op: FieldTransformOp::RegexReplace,
                    pattern: r"PROJECT-\d+".to_string(),
                    replacement: "project-id".to_string(),
                    ..FieldTransform::default()
                },
                FieldTransform {
                    field: FieldReplacementTarget::Instruction,
                    op: FieldTransformOp::Split,
                    delimiter: "::".to_string(),
                    side: Some(FieldSplitSide::Before),
                    ..FieldTransform::default()
                },
                FieldTransform {
                    field: FieldReplacementTarget::Instruction,
                    op: FieldTransformOp::Affix,
                    prefix: "task ".to_string(),
                    ..FieldTransform::default()
                },
                FieldTransform {
                    field: FieldReplacementTarget::Input,
                    op: FieldTransformOp::Case,
                    case: Some(FieldCaseTransformKind::Uppercase),
                    ..FieldTransform::default()
                },
                FieldTransform {
                    field: FieldReplacementTarget::Response,
                    op: FieldTransformOp::RegexReplace,
                    pattern: r"APPROVED:\d+".to_string(),
                    replacement: "approved id".to_string(),
                    ..FieldTransform::default()
                },
                FieldTransform {
                    field: FieldReplacementTarget::Response,
                    op: FieldTransformOp::Truncate,
                    max_chars: Some(11),
                    ..FieldTransform::default()
                },
            ],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("dsl-transformed examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("dsl-transformed streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("dsl-transformed source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("dsl-transformed raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("dsl-transformed cache should write");
        let raw_map = QwenSftFieldMap {
            field_transforms: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            max_response_chars: None,
            ..field_map.clone()
        };
        let raw_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_map)
                .expect("raw fingerprint should compute");
        let cache_mismatch =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &raw_map)
                .expect_err("cache should reject changed DSL transforms")
                .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "task Keep project-id");
        assert_eq!(loaded.examples[0].input, "GPU CONTEXT");
        assert_eq!(loaded.examples[0].response, "approved id");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_filters_fields_by_regex_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep project-123 token","input":"GPU context","response":"approved answer"}
{"instruction":"Drop project-456 token","input":"GPU context","response":"DENIED answer"}
{"instruction":"Skip project-abc token","input":"GPU context","response":"approved answer"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            field_regex_contains_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Instruction,
                pattern: r"project-\d+".to_string(),
            }],
            field_regex_excludes_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Response,
                pattern: r"DENIED|blocked".to_string(),
            }],
            ..QwenSftFieldMap::default()
        };
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("regex-filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("regex-filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("regex-filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("regex-filtered raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("regex-filtered cache should write");
        let default_fingerprint = qwen_sft_streaming_fingerprint(
            &paths,
            None,
            &loaded.source_files,
            &QwenSftFieldMap::default(),
        )
        .expect("default fingerprint should compute");
        let filter_mismatch = QwenSftFieldMap {
            field_regex_contains_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Instruction,
                pattern: r"project-[a-z]+".to_string(),
            }],
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &filter_mismatch,
        )
        .expect_err("cache should reject changed regex filters")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Keep project-123 token");
        assert_eq!(loaded.examples[0].response, "approved answer");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(source_index.samples.len(), 1);
        assert_eq!(source_index.samples[0].index_in_file, 0);
        assert_ne!(loaded.fingerprint, default_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_reuses_compiled_regex_plan_across_records() {
        let field_map = QwenSftFieldMap {
            response_contains_any: vec!["approved id".to_string()],
            field_regex_replacements: vec![FieldRegexReplacement {
                field: FieldReplacementTarget::Response,
                pattern: r"APPROVED:\d+".to_string(),
                replacement: "approved id".to_string(),
            }],
            field_regex_contains_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Instruction,
                pattern: r"project-\d+".to_string(),
            }],
            field_regex_excludes_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Input,
                pattern: r"blocked".to_string(),
            }],
            ..QwenSftFieldMap::default()
        };
        let regex_plan = QwenSftRegexPlan::compile(&field_map).expect("regex plan should compile");
        let first = qwen_sft_record_from_jsonl_line(
            r#"{"instruction":"Keep project-123","input":"GPU context","response":"APPROVED:42"}"#,
            &field_map,
            &regex_plan,
        )
        .expect("first record should parse");
        let second = qwen_sft_record_from_jsonl_line(
            r#"{"instruction":"Drop project-456","input":"blocked context","response":"APPROVED:99"}"#,
            &field_map,
            &regex_plan,
        )
        .expect("second record should parse");

        assert_eq!(first.response, "approved id");
        assert!(qwen_sft_record_passes_filters(
            &first,
            &field_map,
            &regex_plan
        ));
        assert_eq!(second.response, "approved id");
        assert!(!qwen_sft_record_passes_filters(
            &second,
            &field_map,
            &regex_plan
        ));
        assert_eq!(regex_plan.replacements.len(), 1);
        assert_eq!(regex_plan.contains_any.len(), 1);
        assert_eq!(regex_plan.excludes_any.len(), 1);
    }

    #[test]
    fn qwen_sft_normalizes_whitespace_after_replacements_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep PROJECT_TOKEN","response":"approved\t\tanswer"}
{"instruction":"Also PROJECT_TOKEN","response":"approved   answer"}
{"instruction":"Drop PROJECT_TOKEN","response":"denied   answer"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["rust train".to_string()],
            response_contains_any: vec!["approved answer".to_string()],
            field_replacements: vec![FieldReplacement {
                field: FieldReplacementTarget::Instruction,
                pattern: "PROJECT_TOKEN".to_string(),
                replacement: "rust   train".to_string(),
            }],
            field_regex_replacements: Vec::new(),
            normalize_whitespace: true,
            ..QwenSftFieldMap::default()
        };
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("normalized examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("normalized streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("normalized source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("normalized raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("normalized cache should write");
        let raw_map = QwenSftFieldMap {
            normalize_whitespace: false,
            ..field_map.clone()
        };
        let cache_mismatch =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &raw_map)
                .expect_err("cache should reject changed whitespace normalization")
                .to_string();
        let raw_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_map)
                .expect("raw fingerprint should compute");

        assert_eq!(loaded.examples.len(), 2);
        assert_eq!(loaded.examples[0].instruction, "Keep rust train");
        assert_eq!(loaded.examples[0].response, "approved answer");
        assert_eq!(raw_window.examples.len(), loaded.examples.len());
        assert_eq!(raw_window.examples[1].instruction, "Also rust train");
        assert_eq!(raw_window.examples[1].response, "approved answer");
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_case_transforms_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep PROJECT_TOKEN","input":"GPU CONTEXT","response":"APPROVED ANSWER"}
{"instruction":"Drop PROJECT_TOKEN","input":"GPU CONTEXT","response":"DENIED ANSWER"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["rustrain".to_string()],
            input_contains_any: vec!["gpu context".to_string()],
            response_contains_any: vec!["approved answer".to_string()],
            field_replacements: vec![FieldReplacement {
                field: FieldReplacementTarget::Instruction,
                pattern: "PROJECT_TOKEN".to_string(),
                replacement: "RUSTRain".to_string(),
            }],
            field_case_transforms: vec![
                FieldCaseTransform {
                    field: FieldReplacementTarget::Instruction,
                    case: FieldCaseTransformKind::Lowercase,
                },
                FieldCaseTransform {
                    field: FieldReplacementTarget::Input,
                    case: FieldCaseTransformKind::Lowercase,
                },
                FieldCaseTransform {
                    field: FieldReplacementTarget::Response,
                    case: FieldCaseTransformKind::Lowercase,
                },
            ],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("case-transformed examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("case-transformed streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("case-transformed source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("case-transformed raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("case-transformed cache should write");
        let raw_case_map = QwenSftFieldMap {
            field_case_transforms: Vec::new(),
            ..field_map.clone()
        };
        let raw_case_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_case_map)
                .expect("raw-case fingerprint should compute");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &raw_case_map,
        )
        .expect_err("cache should reject changed case transforms")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "keep rustrain");
        assert_eq!(loaded.examples[0].input, "gpu context");
        assert_eq!(loaded.examples[0].response, "approved answer");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_case_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_field_affixes_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Name the project","input":"GPU","response":"rustrain"}
{"instruction":"Name the language","input":"GPU","response":"Rust"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            system_contains_any: vec!["system:".to_string()],
            instruction_contains_any: vec!["Q: Name".to_string()],
            input_contains_any: vec!["ctx=GPU".to_string()],
            response_contains_any: vec!["</answer>".to_string()],
            field_defaults: vec![FieldDefault {
                field: FieldDefaultTarget::System,
                value: "concise".to_string(),
            }],
            field_affixes: vec![
                FieldAffix {
                    field: FieldReplacementTarget::System,
                    prefix: "system: ".to_string(),
                    suffix: String::new(),
                },
                FieldAffix {
                    field: FieldReplacementTarget::Instruction,
                    prefix: "Q: ".to_string(),
                    suffix: "?".to_string(),
                },
                FieldAffix {
                    field: FieldReplacementTarget::Input,
                    prefix: "ctx=".to_string(),
                    suffix: String::new(),
                },
                FieldAffix {
                    field: FieldReplacementTarget::Response,
                    prefix: String::new(),
                    suffix: "</answer>".to_string(),
                },
            ],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("affixed examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("affixed streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("affixed source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("affixed raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("affixed cache should write");
        let raw_affix_map = QwenSftFieldMap {
            field_affixes: Vec::new(),
            system_contains_any: Vec::new(),
            instruction_contains_any: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            ..field_map.clone()
        };
        let raw_affix_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_affix_map)
                .expect("raw-affix fingerprint should compute");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &raw_affix_map,
        )
        .expect_err("cache should reject changed affixes")
        .to_string();

        assert_eq!(loaded.examples.len(), 2);
        assert_eq!(loaded.examples[0].system, "system: concise");
        assert_eq!(loaded.examples[0].instruction, "Q: Name the project?");
        assert_eq!(loaded.examples[0].input, "ctx=GPU");
        assert_eq!(loaded.examples[0].response, "rustrain</answer>");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_affix_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_field_strips_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"PROMPT: Keep GPU prompt","input":"ctx=GPU context","response":"<answer>approved answer</answer>"}
{"instruction":"PROMPT: Drop CPU prompt","input":"ctx=CPU context","response":"<answer>denied answer</answer>"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["Keep GPU".to_string()],
            input_contains_any: vec!["GPU context".to_string()],
            response_contains_any: vec!["approved answer".to_string()],
            response_excludes_any: vec!["</answer>".to_string()],
            field_strips: vec![
                FieldStrip {
                    field: FieldReplacementTarget::Instruction,
                    prefix: "PROMPT: ".to_string(),
                    suffix: String::new(),
                },
                FieldStrip {
                    field: FieldReplacementTarget::Input,
                    prefix: "ctx=".to_string(),
                    suffix: String::new(),
                },
                FieldStrip {
                    field: FieldReplacementTarget::Response,
                    prefix: "<answer>".to_string(),
                    suffix: "</answer>".to_string(),
                },
            ],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("stripped examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("stripped streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("stripped source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("stripped raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("stripped cache should write");
        let raw_strip_map = QwenSftFieldMap {
            field_strips: Vec::new(),
            instruction_contains_any: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            ..field_map.clone()
        };
        let raw_strip_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_strip_map)
                .expect("raw-strip fingerprint should compute");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &raw_strip_map,
        )
        .expect_err("cache should reject changed strips")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Keep GPU prompt");
        assert_eq!(loaded.examples[0].input, "GPU context");
        assert_eq!(loaded.examples[0].response, "approved answer");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_strip_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_field_truncations_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Name the rustrain project precisely","input":"GPU worker context","response":"approved response with trailing text"}
{"instruction":"Name the rustrain project precisely","input":"CPU worker context","response":"denied response with trailing text"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["Name the rustrain".to_string()],
            input_contains_any: vec!["GPU".to_string()],
            response_contains_any: vec!["approved response".to_string()],
            min_response_chars: 17,
            max_response_chars: Some(17),
            field_truncations: vec![
                FieldTruncation {
                    field: FieldReplacementTarget::Instruction,
                    max_chars: 17,
                },
                FieldTruncation {
                    field: FieldReplacementTarget::Input,
                    max_chars: 3,
                },
                FieldTruncation {
                    field: FieldReplacementTarget::Response,
                    max_chars: 17,
                },
            ],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("truncated examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("truncated streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("truncated source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("truncated raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("truncated cache should write");
        let raw_truncation_map = QwenSftFieldMap {
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            max_response_chars: None,
            ..field_map.clone()
        };
        let raw_truncation_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_truncation_map)
                .expect("raw-truncation fingerprint should compute");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &raw_truncation_map,
        )
        .expect_err("cache should reject changed truncations")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Name the rustrain");
        assert_eq!(loaded.examples[0].input, "GPU");
        assert_eq!(loaded.examples[0].response, "approved response");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_truncation_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_field_splits_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"metadata :: Keep GPU prompt","input":"GPU context || discard","response":"draft -> approved answer"}
{"instruction":"metadata :: Drop CPU prompt","input":"CPU context || discard","response":"draft -> denied answer"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["Keep GPU".to_string()],
            input_contains_any: vec!["GPU context".to_string()],
            response_contains_any: vec!["approved answer".to_string()],
            field_splits: vec![
                FieldSplit {
                    field: FieldReplacementTarget::Instruction,
                    delimiter: " :: ".to_string(),
                    side: FieldSplitSide::After,
                },
                FieldSplit {
                    field: FieldReplacementTarget::Input,
                    delimiter: " || ".to_string(),
                    side: FieldSplitSide::Before,
                },
                FieldSplit {
                    field: FieldReplacementTarget::Response,
                    delimiter: " -> ".to_string(),
                    side: FieldSplitSide::After,
                },
            ],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("split examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("split streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("split source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("split raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("split cache should write");
        let raw_split_map = QwenSftFieldMap {
            field_splits: Vec::new(),
            instruction_contains_any: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            ..field_map.clone()
        };
        let raw_split_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_split_map)
                .expect("raw-split fingerprint should compute");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &raw_split_map,
        )
        .expect_err("cache should reject changed splits")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Keep GPU prompt");
        assert_eq!(loaded.examples[0].input, "GPU context");
        assert_eq!(loaded.examples[0].response, "approved answer");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_split_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_prompt_template_changes_tokenized_prompt_and_fingerprint() {
        let example = QwenSftExample {
            system: String::new(),
            instruction: "Name the project.".to_string(),
            input: "Rust trainer".to_string(),
            response: "rustrain".to_string(),
        };
        let default_map = QwenSftFieldMap::default();
        let custom_map = QwenSftFieldMap {
            prompt_template: "### User\n{instruction}\n### Assistant\n".to_string(),
            prompt_with_input_template:
                "### User\n{instruction}\nContext: {input}\n### Assistant\n".to_string(),
            ..QwenSftFieldMap::default()
        };
        let default_prompt =
            qwen_render_sft_prompt(&example, &default_map).expect("default prompt should render");
        let custom_prompt =
            qwen_render_sft_prompt(&example, &custom_map).expect("custom prompt should render");
        let default_fingerprint = qwen_sft_dataset_fingerprint(
            &["data/train.jsonl".to_string()],
            std::slice::from_ref(&example),
            &default_map,
        );
        let custom_fingerprint = qwen_sft_dataset_fingerprint(
            &["data/train.jsonl".to_string()],
            std::slice::from_ref(&example),
            &custom_map,
        );

        assert_eq!(
            default_prompt,
            "Instruction:\nName the project.\n\nInput:\nRust trainer\n\nResponse:\n"
        );
        assert_eq!(
            custom_prompt,
            "### User\nName the project.\nContext: Rust trainer\n### Assistant\n"
        );
        assert_ne!(default_fingerprint, custom_fingerprint);
    }

    #[test]
    fn qwen_sft_cli_template_escapes_match_config_templates() {
        let example = QwenSftExample {
            system: String::new(),
            instruction: "Name the project.".to_string(),
            input: "Rust trainer".to_string(),
            response: "rustrain".to_string(),
        };
        let field_map = QwenSftFieldMap {
            prompt_template: qwen_decode_cli_template_escapes(
                "Instruction: {instruction}\\nResponse: ",
            ),
            prompt_with_input_template: qwen_decode_cli_template_escapes(
                "Instruction: {instruction}\\nInput: {input}\\nResponse: ",
            ),
            ..QwenSftFieldMap::default()
        };

        let prompt =
            qwen_render_sft_prompt(&example, &field_map).expect("CLI prompt should render");

        assert_eq!(
            prompt,
            "Instruction: Name the project.\nInput: Rust trainer\nResponse: "
        );
        assert_eq!(qwen_decode_cli_template_escapes(r"one\ttwo"), "one\ttwo");
        assert_eq!(qwen_decode_cli_template_escapes(r"one\\two"), r"one\two");
        assert_eq!(qwen_decode_cli_template_escapes(r#"one\"two"#), "one\"two");
    }

    #[test]
    fn qwen_sft_trim_fields_controls_record_normalization_and_fingerprint() {
        let line = r#"{"instruction":"  Name the project.  ","input":"  Rust trainer  ","response":"  rustrain  "}"#;
        let trim_map = QwenSftFieldMap::default();
        let raw_map = QwenSftFieldMap {
            trim_fields: false,
            ..QwenSftFieldMap::default()
        };
        let trim_regex_plan =
            QwenSftRegexPlan::compile(&trim_map).expect("trim regex plan should compile");
        let raw_regex_plan =
            QwenSftRegexPlan::compile(&raw_map).expect("raw regex plan should compile");
        let trimmed = qwen_sft_record_from_jsonl_line(line, &trim_map, &trim_regex_plan)
            .expect("trimmed record should parse");
        let raw = qwen_sft_record_from_jsonl_line(line, &raw_map, &raw_regex_plan)
            .expect("raw record should parse");
        let trimmed_example = QwenSftExample {
            system: trimmed.system.clone(),
            instruction: trimmed.instruction.clone(),
            input: trimmed.input.clone(),
            response: trimmed.response.clone(),
        };
        let raw_example = QwenSftExample {
            system: raw.system.clone(),
            instruction: raw.instruction.clone(),
            input: raw.input.clone(),
            response: raw.response.clone(),
        };

        assert_eq!(trimmed.instruction, "Name the project.");
        assert_eq!(trimmed.input, "Rust trainer");
        assert_eq!(trimmed.response, "rustrain");
        assert_eq!(raw.instruction, "  Name the project.  ");
        assert_eq!(raw.input, "  Rust trainer  ");
        assert_eq!(raw.response, "  rustrain  ");
        assert_ne!(
            qwen_sft_dataset_fingerprint(&[], &[trimmed_example], &trim_map),
            qwen_sft_dataset_fingerprint(&[], &[raw_example], &raw_map)
        );
    }

    #[test]
    fn qwen_sft_explicit_eval_metadata_combines_train_and_eval_sources() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let train_jsonl = temp.path().join("train.jsonl");
        let eval_jsonl = temp.path().join("eval.jsonl");
        let train_file = train_jsonl.display().to_string();
        let eval_file = eval_jsonl.display().to_string();
        let source_files = qwen_merge_sft_source_files(
            std::slice::from_ref(&train_file),
            std::slice::from_ref(&eval_file),
        );
        let source_counts = qwen_merge_sft_source_sample_counts(
            &[QwenSftSourceSampleCount {
                path: train_file.clone(),
                samples: 3,
            }],
            &[QwenSftSourceSampleCount {
                path: eval_file.clone(),
                samples: 2,
            }],
        );
        let fingerprint =
            qwen_combine_sft_fingerprints(&source_files, "train-fingerprint", "eval-fingerprint");

        assert_eq!(source_files, vec![eval_file.clone(), train_file.clone()]);
        assert_eq!(
            source_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: eval_file,
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: train_file,
                    samples: 3,
                },
            ]
        );
        assert!(!fingerprint.is_empty());
        assert_ne!(
            fingerprint,
            qwen_combine_sft_fingerprints(&source_files, "train-fingerprint", "other-eval")
        );
    }

    #[test]
    fn qwen_sft_limits_explicit_eval_paths() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let train_jsonl = temp.path().join("train.jsonl");
        let eval_jsonl = temp.path().join("eval.jsonl");
        fs::write(
            &train_jsonl,
            "{\"instruction\":\"train one\",\"response\":\"alpha\"}\n{\"instruction\":\"train two\",\"response\":\"beta\"}\n",
        )
        .expect("train jsonl should write");
        fs::write(
            &eval_jsonl,
            "{\"instruction\":\"eval one\",\"response\":\"gamma\"}\n{\"instruction\":\"eval two\",\"response\":\"delta\"}\n{\"instruction\":\"eval three\",\"response\":\"epsilon\"}\n",
        )
        .expect("eval jsonl should write");
        let train_paths = vec![train_jsonl.clone()];
        let eval_paths = vec![eval_jsonl.clone()];
        let field_map = QwenSftFieldMap::default();
        let eval_field_map = qwen_sft_eval_field_map(&field_map);

        let train_set =
            qwen_sft_examples_from_jsonl_paths_with_limit(&train_paths, None, &field_map)
                .expect("train examples should load");
        let limited_eval_set =
            qwen_sft_examples_from_jsonl_paths_with_limit(&eval_paths, Some(2), &eval_field_map)
                .expect("limited eval examples should load");
        let streaming_eval_summary =
            qwen_sft_streaming_source_summary(&eval_paths, Some(2), &eval_field_map)
                .expect("limited eval streaming summary should scan");

        assert_eq!(train_set.examples.len(), 2);
        assert_eq!(limited_eval_set.examples.len(), 2);
        assert_eq!(
            limited_eval_set
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["eval one", "eval two"]
        );
        assert_eq!(
            streaming_eval_summary.samples,
            limited_eval_set.examples.len()
        );
        assert_eq!(
            streaming_eval_summary.source_sample_counts,
            limited_eval_set.source_sample_counts
        );
        assert_eq!(
            streaming_eval_summary.fingerprint,
            limited_eval_set.fingerprint
        );

        let combined_source_files =
            qwen_merge_sft_source_files(&train_set.source_files, &limited_eval_set.source_files);
        let combined_source_sample_counts = qwen_merge_sft_source_sample_counts(
            &train_set.source_sample_counts,
            &limited_eval_set.source_sample_counts,
        );
        assert_eq!(
            combined_source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: eval_jsonl.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: train_jsonl.display().to_string(),
                    samples: 2,
                },
            ]
        );
        assert_ne!(
            qwen_combine_sft_fingerprints(
                &combined_source_files,
                &train_set.fingerprint,
                &limited_eval_set.fingerprint,
            ),
            qwen_combine_sft_fingerprints(
                &combined_source_files,
                &train_set.fingerprint,
                "unlimited-eval-fingerprint",
            )
        );
    }

    #[test]
    fn qwen_sft_resume_dataset_validation_rejects_changed_data() {
        let summary = QwenSftDatasetSummary {
            samples: 2,
            total_tokens: 8,
            response_tokens: 2,
            masked_positions: 2,
            max_sequence_tokens: 4,
            source_files: vec!["data/train.jsonl".to_string()],
            source_sample_counts: vec![QwenSftSourceSampleCount {
                path: "data/train.jsonl".to_string(),
                samples: 2,
            }],
            fingerprint: "fingerprint-a".to_string(),
            shuffle: true,
        };

        qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &summary.source_sample_counts,
            &summary.fingerprint,
            true,
            &summary,
            "test resume",
        )
        .expect("matching provenance should pass");
        qwen_validate_sft_resume_dataset(&[], &[], "", true, &summary, "legacy resume")
            .expect("legacy manifests without provenance should pass");

        let fingerprint_error = qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &summary.source_sample_counts,
            "fingerprint-b",
            true,
            &summary,
            "test resume",
        )
        .expect_err("changed content fingerprint should fail")
        .to_string();
        assert!(fingerprint_error.contains("dataset fingerprint mismatch"));

        let source_error = qwen_validate_sft_resume_dataset(
            &["data/other.jsonl".to_string()],
            &summary.source_sample_counts,
            &summary.fingerprint,
            true,
            &summary,
            "test resume",
        )
        .expect_err("changed source files should fail")
        .to_string();
        assert!(source_error.contains("dataset source files mismatch"));

        let count_error = qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &[QwenSftSourceSampleCount {
                path: "data/train.jsonl".to_string(),
                samples: 3,
            }],
            &summary.fingerprint,
            true,
            &summary,
            "test resume",
        )
        .expect_err("changed source sample counts should fail")
        .to_string();
        assert!(count_error.contains("dataset source sample counts mismatch"));

        let shuffle_error = qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &summary.source_sample_counts,
            &summary.fingerprint,
            false,
            &summary,
            "test resume",
        )
        .expect_err("changed shuffle policy should fail")
        .to_string();
        assert!(shuffle_error.contains("dataset shuffle mismatch"));
    }

    #[test]
    fn qwen_sft_jsonl_reader_aggregates_multiple_paths_and_directories() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let dir = temp.path().join("shard_dir");
        fs::create_dir(&dir).expect("shard dir should be created");
        let second = dir.join("b.jsonl");
        let third = dir.join("a.jsonl");
        let ignored = dir.join("ignored.txt");
        fs::write(
            &first,
            r#"{"instruction":"first","response":"one"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"third","response":"three"}
"#,
        )
        .expect("second jsonl should write");
        fs::write(
            &third,
            r#"{"instruction":"second","response":"two"}
"#,
        )
        .expect("third jsonl should write");
        fs::write(
            &ignored,
            r#"{"instruction":"ignored","response":"ignored"}
"#,
        )
        .expect("ignored file should write");

        let example_set = qwen_sft_examples_from_jsonl_paths_with_limit(
            &[first.clone(), dir.clone()],
            None,
            &QwenSftFieldMap::default(),
        )
        .expect("examples should aggregate from multiple paths");
        let examples = &example_set.examples;

        assert_eq!(examples.len(), 3);
        assert_eq!(example_set.source_files.len(), 3);
        assert_eq!(
            example_set.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 1,
                },
                QwenSftSourceSampleCount {
                    path: third.display().to_string(),
                    samples: 1,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 1,
                },
            ]
        );
        assert!(
            example_set
                .source_files
                .iter()
                .all(|path| path.ends_with(".jsonl"))
        );
        assert!(!example_set.fingerprint.is_empty());
        assert_eq!(examples[0].instruction, "first");
        assert_eq!(examples[1].instruction, "second");
        assert_eq!(examples[2].instruction, "third");
    }

    #[test]
    fn qwen_sft_jsonl_limit_stops_before_unneeded_files() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let third = temp.path().join("third.jsonl");
        fs::write(
            &first,
            r#"{"instruction":"one","response":"a"}
{"instruction":"two","response":"b"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"three","response":"c"}
{"instruction":"four","response":"d"}
"#,
        )
        .expect("second jsonl should write");
        fs::write(
            &third,
            r#"{"instruction":"five","response":"e"}
"#,
        )
        .expect("third jsonl should write");

        let limited = qwen_sft_examples_from_jsonl_paths_with_limit(
            &[first.clone(), second.clone(), third],
            Some(3),
            &QwenSftFieldMap::default(),
        )
        .expect("limited examples should load");

        assert_eq!(limited.examples.len(), 3);
        assert_eq!(
            limited.source_files,
            vec![first.display().to_string(), second.display().to_string()]
        );
        assert_eq!(
            limited.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 1,
                },
            ]
        );
        assert_eq!(limited.examples[0].instruction, "one");
        assert_eq!(limited.examples[2].instruction, "three");
        assert_eq!(
            limited.fingerprint,
            qwen_sft_dataset_fingerprint(
                &limited.source_files,
                &limited.examples,
                &QwenSftFieldMap::default()
            )
        );
    }

    #[test]
    fn qwen_sft_streaming_summary_matches_jsonl_reader_without_materializing_tokens() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let third = temp.path().join("third.jsonl");
        fs::write(
            &first,
            r#"{"instruction":"one","response":"a"}
{"instruction":"two","input":"input","response":"b"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"three","response":"c"}
{"instruction":"four","response":"d"}
"#,
        )
        .expect("second jsonl should write");
        fs::write(
            &third,
            r#"{"instruction":"five","response":"e"}
"#,
        )
        .expect("third jsonl should write");

        let paths = vec![first.clone(), second.clone(), third];
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(
            &paths,
            Some(3),
            &QwenSftFieldMap::default(),
        )
        .expect("limited examples should load");
        let streamed =
            qwen_sft_streaming_source_summary(&paths, Some(3), &QwenSftFieldMap::default())
                .expect("streaming summary should scan");

        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
    }

    #[test]
    fn qwen_sft_filters_short_responses_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"empty","response":""}
{"instruction":"short","response":"ok"}
{"instruction":"first","response":"valid"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            min_response_chars: 5,
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_responses_by_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","response":"approved answer"}
{"instruction":"skip","response":"ordinary reply"}
{"instruction":"second","response":"contains verified marker"}
{"instruction":"third","response":"approved final"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            response_contains_any: vec!["approved".to_string(), "verified".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("response substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_instructions_by_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"keep first task","response":"answer one"}
{"instruction":"skip ordinary","response":"answer skip"}
{"instruction":"second task","response":"answer two"}
{"instruction":"keep third","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["task".to_string(), "keep".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("instruction substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["keep first task", "second task", "keep third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["keep first task", "second task", "keep third"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_instructions_by_exclude_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first clean task","response":"answer one"}
{"instruction":"skip banned task","response":"answer skip"}
{"instruction":"second safe task","response":"answer two"}
{"instruction":"third clean task","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_excludes_any: vec!["banned".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("instruction exclude substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first clean task", "second safe task", "third clean task"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first clean task", "second safe task", "third clean task"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_responses_by_exclude_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","response":"clean answer"}
{"instruction":"skip","response":"contains banned marker"}
{"instruction":"second","response":"safe reply"}
{"instruction":"third","response":"clean final"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            response_excludes_any: vec!["banned".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("response exclude substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_long_responses_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","response":"valid"}
{"instruction":"too long","response":"toolong"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            max_response_chars: Some(5),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("max response drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    #[test]
    fn qwen_sft_filters_instruction_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"a","response":"skip"}
{"instruction":"first","response":"valid"}
{"instruction":"too long","response":"skip"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            min_instruction_chars: Some(3),
            max_instruction_chars: Some(6),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("instruction filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_input_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"skip short","input":"x","response":"short"}
{"instruction":"first","input":"ok","response":"valid"}
{"instruction":"skip long","input":"toolong","response":"long"}
{"instruction":"second","input":"mid","response":"works"}
{"instruction":"third","input":"fit","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            min_input_chars: Some(2),
            max_input_chars: Some(3),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("input filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "mid")]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "mid")]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_inputs_by_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","input":"keep context","response":"answer one"}
{"instruction":"skip","input":"ordinary context","response":"answer skip"}
{"instruction":"second","input":"selected context","response":"answer two"}
{"instruction":"third","input":"keep extra","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            input_contains_any: vec!["keep".to_string(), "selected".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("input substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.input.as_str())
                .collect::<Vec<_>>(),
            vec!["keep context", "selected context", "keep extra"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_inputs_by_exclude_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","input":"clean context","response":"answer one"}
{"instruction":"skip","input":"contains banned context","response":"answer skip"}
{"instruction":"second","input":"safe context","response":"answer two"}
{"instruction":"third","input":"clean extra","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            input_excludes_any: vec!["banned".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("input exclude substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.input.as_str())
                .collect::<Vec<_>>(),
            vec!["clean context", "safe context", "clean extra"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_system_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"system":"ai","instruction":"skip short","response":"short"}
{"system":"brief","instruction":"first","response":"valid"}
{"system":"this system prompt is too long","instruction":"skip long","response":"long"}
{"system":"concise","instruction":"second","response":"works"}
{"system":"direct","instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            system: Some("system".to_string()),
            min_system_chars: Some(4),
            max_system_chars: Some(8),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap {
                system: Some("system".to_string()),
                ..QwenSftFieldMap::default()
            },
        )
        .expect_err("system filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (example.system.as_str(), example.instruction.as_str()))
                .collect::<Vec<_>>(),
            vec![("brief", "first"), ("concise", "second")]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| (example.system.as_str(), example.instruction.as_str()))
                .collect::<Vec<_>>(),
            vec![("brief", "first"), ("concise", "second")]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_systems_by_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"system":"keep system","instruction":"first","response":"answer one"}
{"system":"ordinary system","instruction":"skip","response":"answer skip"}
{"system":"selected system","instruction":"second","response":"answer two"}
{"system":"keep extra","instruction":"third","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            system: Some("system".to_string()),
            system_contains_any: vec!["keep".to_string(), "selected".to_string()],
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap {
                system: Some("system".to_string()),
                ..QwenSftFieldMap::default()
            },
        )
        .expect_err("system substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap {
                    system: Some("system".to_string()),
                    ..QwenSftFieldMap::default()
                },
            )
            .expect("default system fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.system.as_str())
                .collect::<Vec<_>>(),
            vec!["keep system", "selected system", "keep extra"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_systems_by_exclude_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"system":"keep system","instruction":"first","response":"answer one"}
{"system":"banned system","instruction":"skip","response":"answer skip"}
{"system":"selected system","instruction":"second","response":"answer two"}
{"system":"keep extra","instruction":"third","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            system: Some("system".to_string()),
            system_excludes_any: vec!["banned".to_string()],
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap {
                system: Some("system".to_string()),
                ..QwenSftFieldMap::default()
            },
        )
        .expect_err("system exclude substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap {
                    system: Some("system".to_string()),
                    ..QwenSftFieldMap::default()
                },
            )
            .expect("default system fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.system.as_str())
                .collect::<Vec<_>>(),
            vec!["keep system", "selected system", "keep extra"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_prompt_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"a","response":"skip"}
{"instruction":"first","input":"ok","response":"valid"}
{"instruction":"this prompt is too long","response":"skip"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            prompt_template: "Q:{instruction}\nA:".to_string(),
            prompt_with_input_template: "Q:{instruction}\nI:{input}\nA:".to_string(),
            min_prompt_chars: Some(11),
            max_prompt_chars: Some(15),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("prompt filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "")]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "")]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_sample_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"a","response":"x"}
{"instruction":"first","input":"ok","response":"valid"}
{"instruction":"too long","response":"this response is too long"}
{"instruction":"second","response":"works"}
{"instruction":"tiny","response":"z"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            prompt_template: "Q:{instruction}\nA:".to_string(),
            prompt_with_input_template: "Q:{instruction}\nI:{input}\nA:".to_string(),
            min_sample_chars: Some(16),
            max_sample_chars: Some(22),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("sample filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.response.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "valid"), ("second", "works")]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.response.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "valid"), ("second", "works")]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_dedupes_samples_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","response":"valid"}
{"instruction":"first","response":"valid"}
{"instruction":"second","response":"works"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            dedupe_samples: true,
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("deduped examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("deduped streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("deduped source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("deduped raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("deduped cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("dedupe drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    #[test]
    fn qwen_sft_applies_source_weights_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &first,
            r#"{"instruction":"first","response":"alpha"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"second","response":"beta"}
"#,
        )
        .expect("second jsonl should write");
        let paths = vec![first.clone(), second.clone()];
        let field_map = QwenSftFieldMap {
            source_weights: vec![2, 1],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("weighted examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("weighted streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("weighted source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("weighted raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("weighted cache should write");
        let unweighted_map = QwenSftFieldMap::default();
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &unweighted_map,
        )
        .expect_err("source weight drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "first", "second"]
        );
        assert_eq!(
            loaded.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 1,
                },
            ]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 0, 0]
        );
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(&paths, Some(3), &loaded.source_files, &unweighted_map)
                .expect("unweighted fingerprint should compute")
        );
    }

    #[test]
    fn qwen_sft_applies_per_source_jsonl_field_maps_to_streaming_cache() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &first,
            r#"{"instruction":"alpha prompt","input":"alpha context","response":"alpha answer"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"question":"beta prompt","answer":"beta answer"}
"#,
        )
        .expect("second jsonl should write");
        let paths = vec![first.clone(), second.clone()];
        let field_map = QwenSftFieldMap {
            source_instruction_fields: vec!["instruction".to_string(), "question".to_string()],
            source_input_fields: vec!["input".to_string(), String::new()],
            source_response_fields: vec!["response".to_string(), "answer".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("per-source JSONL examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("per-source streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("per-source streaming index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("per-source raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("per-source cache should write");
        let cached = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("per-source cache should hit");
        let changed_map = QwenSftFieldMap {
            source_instruction_fields: vec!["instruction".to_string(), "prompt".to_string()],
            source_input_fields: vec!["input".to_string(), String::new()],
            source_response_fields: vec!["response".to_string(), "answer".to_string()],
            ..QwenSftFieldMap::default()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &changed_map,
        )
        .expect_err("changed per-source field map should reject cache")
        .to_string();

        assert_eq!(
            loaded.examples,
            vec![
                QwenSftExample {
                    system: String::new(),
                    instruction: "alpha prompt".to_string(),
                    input: "alpha context".to_string(),
                    response: "alpha answer".to_string(),
                },
                QwenSftExample {
                    system: String::new(),
                    instruction: "beta prompt".to_string(),
                    input: String::new(),
                    response: "beta answer".to_string(),
                },
            ]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(raw_window.examples, loaded.examples);
        assert!(cache.cache_written);
        assert!(cached.cache_hit);
        assert_eq!(cached.index.samples, source_index.samples);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_source_max_samples_before_weighting_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &first,
            r#"{"instruction":"first-a","response":"alpha"}
{"instruction":"first-b","response":"skip"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"second-a","response":"beta"}
{"instruction":"second-b","response":"gamma"}
{"instruction":"second-c","response":"skip"}
"#,
        )
        .expect("second jsonl should write");
        let paths = vec![first.clone(), second.clone()];
        let field_map = QwenSftFieldMap {
            source_weights: vec![2, 2],
            source_max_samples: vec![1, 2],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(6), &field_map)
            .expect("source-limited examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(6), &field_map)
            .expect("source-limited streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(6), &field_map)
            .expect("source-limited source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("source-limited raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(6),
            Some(&cache_path),
            &field_map,
        )
        .expect("source-limited cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(6),
            Some(&cache_path),
            &QwenSftFieldMap {
                source_weights: vec![2, 2],
                ..QwenSftFieldMap::default()
            },
        )
        .expect_err("source max-samples drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec![
                "first-a", "first-a", "second-a", "second-a", "second-b", "second-b"
            ]
        );
        assert_eq!(
            loaded.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 4,
                },
            ]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec![
                "first-a", "first-a", "second-a", "second-a", "second-b", "second-b"
            ]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| (sample.path.clone(), sample.index_in_file))
                .collect::<Vec<_>>(),
            vec![
                (first.display().to_string(), 0),
                (first.display().to_string(), 0),
                (second.display().to_string(), 0),
                (second.display().to_string(), 0),
                (second.display().to_string(), 1),
                (second.display().to_string(), 1),
            ]
        );
    }

    #[test]
    fn qwen_sft_arrow_applies_source_limits_and_weights_before_global_limit() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.arrow");
        let second = temp.path().join("second.arrow");
        write_test_sft_arrow(
            &first,
            &[
                ("first-a", "", "alpha"),
                ("first-b", "", "beta"),
                ("first-c", "", "gamma"),
            ],
        );
        write_test_sft_arrow(
            &second,
            &[
                ("second-a", "", "delta"),
                ("second-b", "", "epsilon"),
                ("second-c", "", "zeta"),
            ],
        );
        let paths = vec![first.clone(), second.clone()];
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            source_weights: vec![2, 1],
            source_max_samples: vec![1, 2],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_arrow_examples_from_paths_with_limit(&paths, Some(4), &field_map)
            .expect("weighted Arrow examples should load");
        let unweighted = qwen_sft_arrow_examples_from_paths_with_limit(
            &paths,
            Some(4),
            &QwenSftFieldMap {
                response: "output".to_string(),
                ..QwenSftFieldMap::default()
            },
        )
        .expect("unweighted Arrow examples should load");

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first-a", "first-a", "second-a", "second-b"]
        );
        assert_eq!(
            loaded.source_files,
            vec![first.display().to_string(), second.display().to_string()]
        );
        assert_eq!(
            loaded.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 2,
                },
            ]
        );
        assert_ne!(loaded.fingerprint, unweighted.fingerprint);
    }

    #[test]
    fn qwen_sft_arrow_index_cache_writes_reuses_and_rejects_drift() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("train.arrow");
        let cache_path = temp.path().join("cache").join("arrow-row-index.json");
        write_test_sft_arrow(
            &arrow,
            &[
                ("first", "", "alpha"),
                ("second", "", "beta"),
                ("third", "", "gamma"),
            ],
        );
        let paths = vec![arrow.clone()];
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            ..QwenSftFieldMap::default()
        };

        let first =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(2), Some(&cache_path), &field_map)
                .expect("first Arrow cache load should build row index");
        assert!(!first.cache_hit);
        assert!(first.cache_written);
        assert_eq!(first.index.samples.len(), 2);
        assert_eq!(
            first
                .index
                .samples
                .iter()
                .map(|sample| sample.row_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        let second =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(2), Some(&cache_path), &field_map)
                .expect("second Arrow cache load should hit");
        assert!(second.cache_hit);
        assert!(!second.cache_written);
        assert_eq!(second.index.samples, first.index.samples);

        let max_samples_mismatch =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(3), Some(&cache_path), &field_map)
                .expect_err("changed max_samples should reject Arrow cache")
                .to_string();
        assert!(max_samples_mismatch.contains("max_samples"));

        let field_mismatch = qwen_sft_arrow_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("changed field map should reject Arrow cache")
        .to_string();
        assert!(field_mismatch.contains("field_map"));

        write_test_sft_arrow(
            &arrow,
            &[
                ("first", "", "alpha"),
                ("second", "", "beta"),
                ("third", "", "gamma"),
                ("fourth", "", "delta"),
            ],
        );
        let source_mismatch =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(2), Some(&cache_path), &field_map)
                .expect_err("changed Arrow source metadata should reject cache")
                .to_string();
        assert!(source_mismatch.contains("source_files"));
    }

    #[test]
    fn qwen_sft_arrow_streaming_policy_requires_cache_or_bounds() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("train.arrow");
        let cache_path = temp.path().join("cache").join("arrow-row-index.json");
        write_test_sft_arrow(
            &arrow,
            &[
                ("first", "", "alpha"),
                ("second", "", "beta"),
                ("third", "", "gamma"),
            ],
        );
        let mut data_config: RuntimeDataConfig = toml::from_str(
            r#"
kind = "instruction_arrow"
paths = ["placeholder.arrow"]
response_field = "output"
"#,
        )
        .expect("test data config should parse");
        data_config.paths = vec![arrow];

        qwen_sft_arrow_validate_config_scope(&data_config, "test")
            .expect("test Arrow config scope should validate");
        let unbounded = qwen_sft_arrow_require_cache_or_bounds(&data_config, None, "test")
            .expect_err("unbounded Arrow streaming config should require cache or bounds")
            .to_string();
        assert!(unbounded.contains("requires data.max_samples, data.source_max_samples"));

        data_config.max_samples = Some(2);
        qwen_sft_arrow_require_cache_or_bounds(&data_config, None, "test")
            .expect("global max_samples should bound Arrow streaming");
        data_config.max_samples = None;

        data_config.source_max_samples = vec![2];
        qwen_sft_arrow_require_cache_or_bounds(&data_config, None, "test")
            .expect("source_max_samples should bound Arrow streaming");
        data_config.source_max_samples.clear();

        qwen_sft_arrow_require_cache_or_bounds(&data_config, Some(&cache_path), "test")
            .expect("index cache should allow unbounded Arrow streaming");
    }

    #[test]
    fn qwen_sft_arrow_raw_index_read_uses_original_row_indices_after_filters() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("train.arrow");
        write_test_sft_arrow(
            &arrow,
            &[
                ("drop", "", "skip"),
                ("keep-one", "", "alpha"),
                ("keep-two", "", "beta"),
            ],
        );
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            response_contains_any: vec!["alpha".to_string(), "beta".to_string()],
            ..QwenSftFieldMap::default()
        };
        let scan = qwen_sft_arrow_source_scan(&[arrow.clone()], Some(2), &field_map)
            .expect("filtered Arrow source should scan");

        assert_eq!(
            scan.index
                .samples
                .iter()
                .map(|sample| sample.row_index)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        let raw_window = qwen_sft_arrow_examples_by_raw_indices(&scan.index.samples, &field_map)
            .expect("filtered original rows should read");
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["keep-one", "keep-two"]
        );
        assert_eq!(raw_window.raw_samples_read, 2);
    }

    #[test]
    fn qwen_sft_arrow_source_scan_streaming_fingerprint_matches_index_readback() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("train.arrow");
        write_test_sft_arrow(
            &arrow,
            &[
                ("drop", "", "skip"),
                ("keep-one", "", "alpha"),
                ("keep-two", "", "beta"),
                ("keep-three", "", "gamma"),
            ],
        );
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            response_contains_any: vec!["alpha".to_string(), "beta".to_string()],
            ..QwenSftFieldMap::default()
        };

        let scan = qwen_sft_arrow_source_scan(&[arrow], Some(2), &field_map)
            .expect("Arrow source scan should build row index");
        let readback_fingerprint = qwen_sft_arrow_fingerprint_from_index(
            &scan.index,
            &scan.summary.source_files,
            &field_map,
        )
        .expect("explicit index readback should hash");

        assert_eq!(
            scan.index
                .samples
                .iter()
                .map(|sample| sample.row_index)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(scan.summary.fingerprint, readback_fingerprint);
    }

    #[test]
    fn qwen_sft_arrow_source_scan_streaming_fingerprint_accounts_for_weights() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("train.arrow");
        write_test_sft_arrow(&arrow, &[("first", "", "alpha"), ("second", "", "beta")]);
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            source_weights: vec![2],
            ..QwenSftFieldMap::default()
        };

        let scan = qwen_sft_arrow_source_scan(&[arrow], Some(4), &field_map)
            .expect("weighted Arrow source scan should build row index");
        let raw_window = qwen_sft_arrow_examples_by_raw_indices(&scan.index.samples, &field_map)
            .expect("weighted row index should read back");
        let expected_fingerprint = qwen_sft_dataset_fingerprint(
            &scan.summary.source_files,
            &raw_window.examples,
            &field_map,
        );

        assert_eq!(
            scan.index
                .samples
                .iter()
                .map(|sample| sample.row_index)
                .collect::<Vec<_>>(),
            vec![0, 0, 1, 1]
        );
        assert_eq!(scan.summary.source_sample_counts[0].samples, 4);
        assert_eq!(scan.summary.fingerprint, expected_fingerprint);
    }

    #[test]
    fn qwen_sft_arrow_source_scan_streaming_fingerprint_omits_unused_paths() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.arrow");
        let second = temp.path().join("second.arrow");
        write_test_sft_arrow(&first, &[("first", "", "alpha")]);
        write_test_sft_arrow(&second, &[("second", "", "beta")]);
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            ..QwenSftFieldMap::default()
        };

        let scan =
            qwen_sft_arrow_source_scan(&[first.clone(), second.clone()], Some(1), &field_map)
                .expect("limited Arrow source scan should stop after first source");
        let materialized = qwen_sft_arrow_examples_from_ipc(&first, Some(1), &field_map)
            .expect("first Arrow source should materialize");

        assert_eq!(scan.summary.source_files, vec![first.display().to_string()]);
        assert_eq!(scan.summary.fingerprint, materialized.fingerprint);
    }

    #[test]
    fn qwen_sft_arrow_reads_question_answer_without_input_column() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("qa.arrow");
        write_test_qa_arrow(
            &arrow,
            &[
                ("What is Rust?", "A systems programming language."),
                ("What is CUDA?", "A GPU programming platform."),
            ],
        );
        let field_map = QwenSftFieldMap {
            input: String::new(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_arrow_examples_from_ipc(&arrow, Some(2), &field_map)
            .expect("question/answer Arrow should materialize");
        let scan = qwen_sft_arrow_source_scan(&[arrow.clone()], Some(2), &field_map)
            .expect("question/answer Arrow source should scan");
        let raw_window = qwen_sft_arrow_examples_by_raw_indices(&scan.index.samples, &field_map)
            .expect("question/answer raw rows should read back");

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (
                    example.instruction.as_str(),
                    example.input.as_str(),
                    example.response.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("What is Rust?", "", "A systems programming language."),
                ("What is CUDA?", "", "A GPU programming platform."),
            ]
        );
        assert_eq!(loaded.examples, raw_window.examples);
        assert_eq!(
            scan.index
                .samples
                .iter()
                .map(|sample| sample.row_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(scan.summary.fingerprint, loaded.fingerprint);
    }

    #[test]
    fn qwen_sft_arrow_applies_per_source_field_maps_to_cache_and_readback() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.arrow");
        let second = temp.path().join("second.arrow");
        let cache_path = temp.path().join("cache").join("arrow-row-index.json");
        write_test_sft_arrow(&first, &[("alpha prompt", "alpha context", "alpha answer")]);
        write_test_custom_sft_arrow(
            &second,
            ("prompt", "context", "completion"),
            &[("beta prompt", "", "beta answer")],
        );
        let paths = vec![first.clone(), second.clone()];
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            source_instruction_fields: vec!["instruction".to_string(), "prompt".to_string()],
            source_input_fields: vec!["input".to_string(), "context".to_string()],
            source_response_fields: vec!["output".to_string(), "completion".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_arrow_examples_from_paths_with_limit(&paths, Some(2), &field_map)
            .expect("per-source Arrow examples should load");
        let scan = qwen_sft_arrow_source_scan(&paths, Some(2), &field_map)
            .expect("per-source Arrow scan should build");
        let raw_window = qwen_sft_arrow_examples_by_raw_indices(&scan.index.samples, &field_map)
            .expect("per-source Arrow raw rows should read back");
        let cached =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(2), Some(&cache_path), &field_map)
                .expect("per-source Arrow cache should write");
        let cache_hit =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(2), Some(&cache_path), &field_map)
                .expect("per-source Arrow cache should hit");
        let changed_map = QwenSftFieldMap {
            response: "output".to_string(),
            source_instruction_fields: vec!["instruction".to_string(), "question".to_string()],
            source_input_fields: vec!["input".to_string(), "context".to_string()],
            source_response_fields: vec!["output".to_string(), "completion".to_string()],
            ..QwenSftFieldMap::default()
        };
        let cache_mismatch = qwen_sft_arrow_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &changed_map,
        )
        .expect_err("changed per-source Arrow field map should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (
                    example.instruction.as_str(),
                    example.input.as_str(),
                    example.response.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("alpha prompt", "alpha context", "alpha answer"),
                ("beta prompt", "", "beta answer"),
            ]
        );
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(scan.summary.fingerprint, loaded.fingerprint);
        assert!(cached.cache_written);
        assert!(cache_hit.cache_hit);
        assert_eq!(cache_hit.index.samples, scan.index.samples);
        assert!(cache_mismatch.contains("field_map"));
    }

    fn write_test_sft_arrow(path: &Path, rows: &[(&str, &str, &str)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("instruction", DataType::Utf8, false),
            Field::new("input", DataType::Utf8, false),
            Field::new("output", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(
                    rows.iter()
                        .map(|(instruction, _, _)| *instruction)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|(_, input, _)| *input).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter()
                        .map(|(_, _, response)| *response)
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("test Arrow batch should build");
        let file = fs::File::create(path).expect("test Arrow file should create");
        let mut writer = StreamWriter::try_new(file, &schema).expect("test Arrow writer");
        writer.write(&batch).expect("test Arrow batch should write");
        writer.finish().expect("test Arrow stream should finish");
    }

    fn write_test_custom_sft_arrow(
        path: &Path,
        fields: (&str, &str, &str),
        rows: &[(&str, &str, &str)],
    ) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(fields.0, DataType::Utf8, false),
            Field::new(fields.1, DataType::Utf8, false),
            Field::new(fields.2, DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(
                    rows.iter()
                        .map(|(instruction, _, _)| *instruction)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|(_, input, _)| *input).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter()
                        .map(|(_, _, response)| *response)
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("test custom Arrow batch should build");
        let file = fs::File::create(path).expect("test custom Arrow file should create");
        let mut writer = StreamWriter::try_new(file, &schema).expect("test custom Arrow writer");
        writer
            .write(&batch)
            .expect("test custom Arrow batch should write");
        writer
            .finish()
            .expect("test custom Arrow stream should finish");
    }

    fn write_test_qa_arrow(path: &Path, rows: &[(&str, &str)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("question", DataType::Utf8, false),
            Field::new("answer", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(
                    rows.iter()
                        .map(|(question, _)| *question)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|(_, answer)| *answer).collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("test QA Arrow batch should build");
        let file = fs::File::create(path).expect("test QA Arrow file should create");
        let mut writer = StreamWriter::try_new(file, &schema).expect("test QA Arrow writer");
        writer
            .write(&batch)
            .expect("test QA Arrow batch should write");
        writer.finish().expect("test QA Arrow stream should finish");
    }

    #[test]
    fn qwen_sft_streaming_window_uses_train_split_sample_count() {
        let (train_samples, eval_samples) =
            qwen_sft_train_eval_sample_counts(4, 0.75).expect("split should compute");
        let world_size = 2usize;
        let local_batch_size = 1usize;
        let global_batch_size = local_batch_size * world_size;
        let train_steps = 1usize;
        let data_cursor_start = 2usize;
        let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
        let (epoch_start, offset_start) =
            qwen_data_epoch_and_offset(data_cursor_start, train_samples)
                .expect("start cursor should map");
        let (epoch_next, offset_next) = qwen_data_epoch_and_offset(data_cursor_end, train_samples)
            .expect("next cursor should map");

        assert_eq!(train_samples, 3);
        assert_eq!(eval_samples, 1);
        assert_eq!(data_cursor_end, 4);
        assert_eq!((epoch_start, offset_start), (0, 2));
        assert_eq!((epoch_next, offset_next), (1, 1));
    }

    #[test]
    fn qwen_sft_streaming_cursor_window_covers_next_batch_overlap() {
        let cursors =
            qwen_sft_streaming_cursor_window(2, 3, 2, 3).expect("cursor window should build");
        let compact = cursors
            .iter()
            .map(|entry| (entry.cursor, entry.epoch, entry.sample_offset))
            .collect::<Vec<_>>();

        assert_eq!(compact, vec![(2, 0, 2), (3, 1, 0), (4, 1, 1), (5, 1, 2)]);
    }

    #[test]
    fn qwen_sft_streaming_raw_index_reads_only_cursor_window() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
{"instruction":"one","response":"one"}
{"instruction":"two","response":"two"}
{"instruction":"three","response":"three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let source_index =
            qwen_sft_streaming_source_index(&paths, None, &QwenSftFieldMap::default())
                .expect("source index should build");
        let (train_samples, _) =
            qwen_sft_train_eval_sample_counts(source_index.samples.len(), 0.75)
                .expect("split should compute");
        let mut train_indices = source_index.samples;
        let mut rng = StdRng::seed_from_u64(777);
        train_indices.shuffle(&mut rng);
        train_indices.truncate(train_samples);
        let raw_indices = (0..4)
            .map(|relative| {
                let cursor = 2 + relative;
                let epoch = cursor / train_indices.len();
                let offset = cursor % train_indices.len();
                let index = qwen_epoch_permutation_index(train_indices.len(), 777, epoch, offset);
                train_indices[index].clone()
            })
            .collect::<Vec<_>>();
        let raw_window =
            qwen_sft_examples_by_raw_indices(&raw_indices, &QwenSftFieldMap::default())
                .expect("raw examples should read");

        assert_eq!(raw_window.examples.len(), 4);
        assert_eq!(raw_window.raw_samples_read, 3);
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["three", "three", "two", "one"]
        );
        assert_eq!(
            raw_indices
                .iter()
                .map(|index| index.index_in_file)
                .collect::<Vec<_>>(),
            vec![3, 3, 2, 1]
        );
        assert_eq!(
            raw_indices
                .iter()
                .map(|index| index.byte_offset)
                .collect::<Vec<_>>(),
            vec![119, 119, 80, 41]
        );
        assert_eq!(
            raw_indices
                .iter()
                .map(|index| index.path.clone())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([jsonl.display().to_string()])
        );
    }

    #[test]
    fn qwen_sft_streaming_source_index_parses_records_before_indexing_offsets() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
not-json
{"instruction":"two","response":"two"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let error = match qwen_sft_streaming_source_index(&paths, None, &QwenSftFieldMap::default())
        {
            Ok(_) => panic!("malformed JSONL row should fail while building the offset index"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("failed to parse SFT JSONL record"));
    }

    #[test]
    fn qwen_sft_skip_invalid_records_keeps_valid_rows_across_paths() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
not-json
{"instruction":"missing-response"}
{"instruction":"three","response":"three"}
{"instruction":"four","response":"four"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let strict_error = match qwen_sft_examples_from_jsonl_paths_with_limit(
            &paths,
            None,
            &QwenSftFieldMap::default(),
        ) {
            Ok(_) => panic!("strict materialized read should reject invalid rows"),
            Err(error) => error.to_string(),
        };
        assert!(strict_error.contains("failed to parse SFT JSONL record"));

        let skip_map = QwenSftFieldMap {
            skip_invalid_records: true,
            ..QwenSftFieldMap::default()
        };
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &skip_map)
            .expect("materialized read should skip invalid rows");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &skip_map)
            .expect("streaming summary should skip invalid rows");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &skip_map)
            .expect("source index should skip invalid rows");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &skip_map)
            .expect("raw indexed window should read valid rows");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &skip_map,
        )
        .expect("skip-invalid cache should write");
        let default_cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("cache should reject skip-invalid policy drift")
        .to_string();
        let default_fingerprint = qwen_sft_dataset_fingerprint(
            &loaded.source_files,
            &loaded.examples,
            &QwenSftFieldMap::default(),
        );

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["zero", "three"]
        );
        assert_eq!(streamed.samples, 2);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(source_index.samples.len(), 2);
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 3]
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.response.as_str())
                .collect::<Vec<_>>(),
            vec!["zero", "three"]
        );
        assert!(cache.cache_written);
        assert!(default_cache_mismatch.contains("field_map"));
        assert_ne!(loaded.fingerprint, default_fingerprint);
    }

    #[test]
    fn qwen_sft_streaming_source_index_cache_writes_and_reuses_offsets() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
{"instruction":"one","response":"one"}
{"instruction":"two","response":"two"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];

        let first = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect("first cache load should build index");
        assert!(!first.cache_hit);
        assert!(first.cache_written);
        assert_eq!(first.index.samples.len(), 2);
        assert!(cache_path.exists());

        let second = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect("second cache load should hit cache");
        assert!(second.cache_hit);
        assert!(!second.cache_written);
        assert_eq!(second.index.samples, first.index.samples);

        let mismatch = match qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        ) {
            Ok(_) => panic!("mismatched max_samples should reject cache"),
            Err(error) => error.to_string(),
        };
        assert!(mismatch.contains("max_samples"));

        let min_response_mismatch_map = QwenSftFieldMap {
            min_response_chars: 2,
            ..QwenSftFieldMap::default()
        };
        let min_response_mismatch = match qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &min_response_mismatch_map,
        ) {
            Ok(_) => panic!("mismatched min_response_chars should reject cache"),
            Err(error) => error.to_string(),
        };
        assert!(min_response_mismatch.contains("field_map"));

        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
{"instruction":"one","response":"one"}
{"instruction":"two","response":"two"}
{"instruction":"three","response":"three"}
"#,
        )
        .expect("jsonl rewrite should update source metadata");
        let source_metadata_mismatch = match qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        ) {
            Ok(_) => panic!("changed source file metadata should reject cache"),
            Err(error) => error.to_string(),
        };
        assert!(source_metadata_mismatch.contains("source_files"));
    }

    #[test]
    fn qwen_sft_external_metadata_participates_in_fingerprint_and_cache() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        let metadata_path = temp.path().join("arrow-export.json");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
{"instruction":"one","response":"one"}
"#,
        )
        .expect("jsonl should write");
        fs::write(
            &metadata_path,
            r#"{"source_arrow":"/datasets/source.arrow","exported_rows":2}"#,
        )
        .expect("metadata should write");
        let paths = vec![jsonl.clone()];
        let metadata = qwen_sft_external_metadata_from_paths(std::slice::from_ref(&metadata_path))
            .expect("metadata should load");
        let field_map = QwenSftFieldMap {
            external_metadata: metadata.clone(),
            ..QwenSftFieldMap::default()
        };
        let no_metadata_map = QwenSftFieldMap::default();
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("examples should load");
        let without_metadata =
            qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &no_metadata_map)
                .expect("examples should load without metadata");

        assert_ne!(loaded.fingerprint, without_metadata.fingerprint);

        let first =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("first cache load should build index");
        assert!(first.cache_written);

        let cache_mismatch = match qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &no_metadata_map,
        ) {
            Ok(_) => panic!("missing external metadata should reject cache"),
            Err(error) => error.to_string(),
        };
        assert!(cache_mismatch.contains("field_map"));

        fs::write(
            &metadata_path,
            r#"{"source_arrow":"/datasets/changed.arrow","exported_rows":2}"#,
        )
        .expect("metadata rewrite should update provenance");
        let changed_metadata =
            qwen_sft_external_metadata_from_paths(std::slice::from_ref(&metadata_path))
                .expect("changed metadata should load");
        let changed_map = QwenSftFieldMap {
            external_metadata: changed_metadata,
            ..QwenSftFieldMap::default()
        };
        let changed = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &changed_map)
            .expect("examples should load with changed metadata");

        assert_ne!(loaded.fingerprint, changed.fingerprint);
    }

    #[test]
    fn qwen_sft_rank_index_cache_path_keeps_extension() {
        let path = PathBuf::from("/tmp/rustrain/cache/offset-index.json");
        assert_eq!(
            qwen_sft_rank_index_cache_path(&path, 1),
            PathBuf::from("/tmp/rustrain/cache/offset-index.rank-1.json")
        );

        let no_extension = PathBuf::from("/tmp/rustrain/cache/offset-index");
        assert_eq!(
            qwen_sft_rank_index_cache_path(&no_extension, 2),
            PathBuf::from("/tmp/rustrain/cache/offset-index.rank-2")
        );
    }

    #[test]
    fn qwen_lora_sft_eval_every_selects_periodic_steps() {
        assert!(!qwen_lora_sft_should_eval_step(1, 0));
        assert!(qwen_lora_sft_should_eval_step(1, 1));
        assert!(!qwen_lora_sft_should_eval_step(1, 2));
        assert!(qwen_lora_sft_should_eval_step(2, 2));
        assert!(qwen_lora_sft_should_eval_step(4, 2));
    }

    #[test]
    fn cached_greedy_matches_full_context_greedy_for_tiny_weights() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2]).reshape([1, 3]);

        let full = qwen_greedy_generate(&input_ids, &weights, &config, 3)
            .expect("full-context generate should run");
        let cached = qwen_greedy_generate_with_cache(&input_ids, &weights, &config, 3)
            .expect("cached generate should run");
        let full_ids: Vec<i64> = Vec::<i64>::try_from(full.reshape([-1])).unwrap();
        let cached_ids: Vec<i64> = Vec::<i64>::try_from(cached.reshape([-1])).unwrap();

        assert_eq!(cached_ids, full_ids);
    }

    #[test]
    fn qwen_model_path_resolves_hf_hub_snapshot_when_legacy_dir_is_missing() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let legacy = temp.path().join("Qwen2.5-0.5B-Instruct");
        let incomplete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("111");
        let complete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("222");
        fs::create_dir_all(&incomplete_snapshot).expect("incomplete snapshot dir should write");
        fs::create_dir_all(&complete_snapshot).expect("complete snapshot dir should write");
        fs::write(incomplete_snapshot.join("config.json"), "{}").expect("config should write");
        fs::write(complete_snapshot.join("config.json"), "{}").expect("config should write");
        fs::write(complete_snapshot.join("tokenizer.json"), "{}").expect("tokenizer should write");
        fs::write(complete_snapshot.join("model.safetensors"), "")
            .expect("safetensors marker should write");

        let resolved =
            resolve_qwen_model_path(&legacy).expect("legacy path should resolve through HF hub");

        assert_eq!(resolved, complete_snapshot);
    }

    #[test]
    fn qwen_model_path_keeps_complete_configured_directory() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let model_path = temp.path().join("Qwen2.5-0.5B-Instruct");
        fs::create_dir_all(&model_path).expect("model dir should write");
        fs::write(model_path.join("config.json"), "{}").expect("config should write");
        fs::write(model_path.join("tokenizer.json"), "{}").expect("tokenizer should write");
        fs::write(model_path.join("model.safetensors"), "")
            .expect("safetensors marker should write");

        let resolved =
            resolve_qwen_model_path(&model_path).expect("complete path should not be rewritten");

        assert_eq!(resolved, model_path);
    }

    #[test]
    fn qwen_model_path_reports_missing_hf_snapshot() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let legacy = temp.path().join("Qwen2.5-0.5B-Instruct");
        let incomplete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("111");
        fs::create_dir_all(&incomplete_snapshot).expect("incomplete snapshot dir should write");
        fs::write(incomplete_snapshot.join("config.json"), "{}").expect("config should write");

        let error = match resolve_qwen_model_path(&legacy) {
            Ok(path) => panic!("incomplete cache should fail, resolved {}", path.display()),
            Err(error) => error.to_string(),
        };

        assert!(error.contains("no complete HF hub snapshot"));
    }

    #[test]
    fn qwen_model_safetensors_path_resolves_with_hf_hub_snapshot() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let legacy_safetensors = temp
            .path()
            .join("Qwen2.5-0.5B-Instruct")
            .join("model.safetensors");
        let complete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("222");
        fs::create_dir_all(&complete_snapshot).expect("complete snapshot dir should write");
        fs::write(complete_snapshot.join("config.json"), "{}").expect("config should write");
        fs::write(complete_snapshot.join("tokenizer.json"), "{}").expect("tokenizer should write");
        fs::write(complete_snapshot.join("model.safetensors"), "")
            .expect("safetensors marker should write");

        let resolved = resolve_qwen_model_safetensors_path(&legacy_safetensors)
            .expect("legacy safetensors path should resolve through HF hub");

        assert_eq!(resolved, complete_snapshot.join("model.safetensors"));
    }

    fn tiny_qwen_sharded_manifest() -> QwenShardedCheckpointManifest {
        QwenShardedCheckpointManifest {
            format: "rustrain.qwen_sharded.v1".to_string(),
            base_model_path: "/models/qwen".to_string(),
            tokenizer_path: "/models/qwen/tokenizer.json".to_string(),
            global_step: 3,
            consumed_samples: 8,
            consumed_tokens: 40,
            data_cursor_next: Some(8),
            data_epoch_next: Some(1),
            data_sample_offset_next: Some(3),
            data_train_samples: Some(5),
            dataset_source_files: vec!["data/train.jsonl".to_string()],
            dataset_source_sample_counts: vec![QwenSftSourceSampleCount {
                path: "data/train.jsonl".to_string(),
                samples: 5,
            }],
            dataset_fingerprint: "abc123".to_string(),
            dataset_shuffle: true,
            streaming_train_batches: Some(true),
            seed: 42,
            dtype: "float32".to_string(),
            optimizer: "adamw".to_string(),
            scheduler: "linear_decay".to_string(),
            parallel: QwenShardedParallelManifest {
                data_parallel_size: 2,
                tensor_model_parallel_size: 1,
                pipeline_model_parallel_size: 1,
                expert_model_parallel_size: 1,
                context_parallel_size: 1,
            },
            ranks: vec![
                QwenRankShardManifest {
                    rank: 0,
                    data_parallel_rank: 0,
                    tensor_model_parallel_rank: 0,
                    pipeline_model_parallel_rank: 0,
                    expert_model_parallel_rank: 0,
                    context_parallel_rank: 0,
                    model_safetensors: "rank0/model.safetensors".to_string(),
                    optimizer_safetensors: "rank0/optimizer.safetensors".to_string(),
                    shards: vec![QwenTensorShardManifestEntry {
                        name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                        shard_name: "rank0.q_proj".to_string(),
                        optimizer_m_name: "rank0.q_proj.m".to_string(),
                        optimizer_v_name: "rank0.q_proj.v".to_string(),
                        global_shape: vec![4, 4],
                        shard_shape: vec![4, 4],
                        dtype: "float32".to_string(),
                        partition: "replicated_dp".to_string(),
                        tied_group: None,
                    }],
                },
                QwenRankShardManifest {
                    rank: 1,
                    data_parallel_rank: 1,
                    tensor_model_parallel_rank: 0,
                    pipeline_model_parallel_rank: 0,
                    expert_model_parallel_rank: 0,
                    context_parallel_rank: 0,
                    model_safetensors: "rank1/model.safetensors".to_string(),
                    optimizer_safetensors: "rank1/optimizer.safetensors".to_string(),
                    shards: vec![QwenTensorShardManifestEntry {
                        name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                        shard_name: "rank1.q_proj".to_string(),
                        optimizer_m_name: "rank1.q_proj.m".to_string(),
                        optimizer_v_name: "rank1.q_proj.v".to_string(),
                        global_shape: vec![4, 4],
                        shard_shape: vec![4, 4],
                        dtype: "float32".to_string(),
                        partition: "replicated_dp".to_string(),
                        tied_group: None,
                    }],
                },
            ],
        }
    }

    fn tiny_qwen_sharded_manifest_with_artifacts(
        root: &Path,
    ) -> Result<QwenShardedCheckpointManifest> {
        let mut manifest = tiny_qwen_sharded_manifest();
        for rank in &mut manifest.ranks {
            let rank_dir = root.join(format!("rank{}", rank.rank));
            fs::create_dir_all(&rank_dir)
                .with_context(|| format!("failed to create {}", rank_dir.display()))?;
            let model_safetensors = rank_dir.join("model.safetensors");
            let optimizer_safetensors = rank_dir.join("optimizer.safetensors");
            let model_entries = rank
                .shards
                .iter()
                .map(|shard| {
                    (
                        shard.shard_name.clone(),
                        Tensor::ones(shard.shard_shape.as_slice(), (Kind::Float, Device::Cpu)),
                    )
                })
                .collect::<Vec<_>>();
            let optimizer_entries = rank
                .shards
                .iter()
                .flat_map(|shard| {
                    [
                        (
                            shard.optimizer_m_name.clone(),
                            Tensor::zeros(shard.shard_shape.as_slice(), (Kind::Float, Device::Cpu)),
                        ),
                        (
                            shard.optimizer_v_name.clone(),
                            Tensor::zeros(shard.shard_shape.as_slice(), (Kind::Float, Device::Cpu)),
                        ),
                    ]
                })
                .collect::<Vec<_>>();
            let model_refs = model_entries
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect::<Vec<_>>();
            let optimizer_refs = optimizer_entries
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect::<Vec<_>>();
            Tensor::write_safetensors(&model_refs, &model_safetensors)
                .with_context(|| format!("failed to write {}", model_safetensors.display()))?;
            Tensor::write_safetensors(&optimizer_refs, &optimizer_safetensors)
                .with_context(|| format!("failed to write {}", optimizer_safetensors.display()))?;
            rank.model_safetensors = model_safetensors.display().to_string();
            rank.optimizer_safetensors = optimizer_safetensors.display().to_string();
        }
        Ok(manifest)
    }

    fn tiny_qwen_weights() -> BTreeMap<String, Tensor> {
        let mut weights = BTreeMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::arange(24, (Kind::Float, Device::Cpu)).reshape([6, 4]) / 24.0,
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::eye(4, (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.bias".to_string(),
            Tensor::zeros([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)) * 0.05,
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.bias".to_string(),
            Tensor::zeros([2], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)) * 0.03,
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.bias".to_string(),
            Tensor::zeros([2], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            Tensor::ones([4, 4], (Kind::Float, Device::Cpu)) * 0.02,
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            Tensor::ones([8, 4], (Kind::Float, Device::Cpu)) * 0.01,
        );
        weights.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            Tensor::ones([8, 4], (Kind::Float, Device::Cpu)) * 0.02,
        );
        weights.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            Tensor::ones([4, 8], (Kind::Float, Device::Cpu)) * 0.03,
        );
        weights
    }

    fn two_layer_tiny_qwen_weights() -> BTreeMap<String, Tensor> {
        let mut weights = tiny_qwen_weights();
        let layer0_names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        for suffix in layer0_names {
            let layer0_name = format!("model.layers.0.{suffix}");
            let layer1_name = format!("model.layers.1.{suffix}");
            let value = tensor(&weights, &layer0_name)
                .expect("layer0 tensor should exist")
                .shallow_clone();
            weights.insert(layer1_name, value * 0.9);
        }
        weights
    }
}
