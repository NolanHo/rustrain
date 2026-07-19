//! CPU-only regression tests for the GLM-5 native MTP/Megatron contract.
//!
//! These tests intentionally operate on token IDs and configuration values only.
//! They must not initialize tch CUDA state, NCCL, or a model checkpoint.

use std::fs;

use rustrain_glm5::model::{
    GLM5_MTP_LOSS_SCALING_FACTOR_DEFAULT, glm5_megatron_raw_seq_len, glm5_mtp_prediction_len,
    glm5_mtp_prediction_len_for_layers, validate_glm5_mtp_contract,
    validate_glm5_mtp_distributed_contract,
};
use serde_json::json;
use tempfile::NamedTempFile;

fn roll_left_with_zero<T: Copy + Default>(values: &[T]) -> Vec<T> {
    let mut rolled = values.to_vec();
    if !rolled.is_empty() {
        rolled.rotate_left(1);
        *rolled.last_mut().expect("non-empty after rotate") = T::default();
    }
    rolled
}

#[test]
fn megatron_extra_token_contract_preserves_one_input_and_label_per_position() {
    // Megatron's add_extra_token_to_sequence path receives S+1 raw tokens and
    // constructs S input tokens plus S labels (tokens[:-1], tokens[1:]).
    let raw = [10_i64, 11, 12, 13, 14, 15, 16];
    let tokens = &raw[..raw.len() - 1];
    let labels = &raw[1..];

    assert_eq!(tokens, &[10, 11, 12, 13, 14, 15]);
    assert_eq!(labels, &[11, 12, 13, 14, 15, 16]);
    assert_eq!(tokens.len(), labels.len());

    // The base LM predicts the next token; the first MTP layer predicts the
    // next-next token by rolling labels once more.
    let mtp_labels = roll_left_with_zero(labels);
    assert_eq!(mtp_labels, vec![12, 13, 14, 15, 16, 0]);
    assert_eq!(mtp_labels[..5], raw[2..]);
}

#[test]
fn native_mtp_length_matches_megatron_raw_sequence_contract() {
    // With raw length R, Megatron has R-1 base positions and R-2 valid
    // next-next targets for one MTP layer. For N layers, the deepest layer has
    // R-N-1 valid positions.
    let raw_len = 17_i64;
    assert_eq!(glm5_mtp_prediction_len(raw_len).unwrap(), raw_len - 2);
    assert_eq!(
        glm5_mtp_prediction_len_for_layers(raw_len, 1).unwrap(),
        raw_len - 2
    );
    assert_eq!(
        glm5_mtp_prediction_len_for_layers(raw_len, 2).unwrap(),
        raw_len - 3
    );
    assert!(glm5_mtp_prediction_len(raw_len - 1).is_ok());
    assert!(glm5_mtp_prediction_len(2).is_err());
    assert!(glm5_mtp_prediction_len_for_layers(3, 2).is_err());
    assert_eq!(glm5_megatron_raw_seq_len(16).unwrap(), raw_len);
}

#[test]
fn mtp_roll_mask_normalizes_only_valid_positions() {
    let base_mask = vec![1_i64; 6];
    let first_depth_mask = roll_left_with_zero(&base_mask);
    let second_depth_mask = roll_left_with_zero(&first_depth_mask);

    assert_eq!(first_depth_mask, vec![1, 1, 1, 1, 1, 0]);
    assert_eq!(second_depth_mask, vec![1, 1, 1, 1, 0, 0]);
    assert_eq!(first_depth_mask.iter().sum::<i64>(), 5);
    assert_eq!(second_depth_mask.iter().sum::<i64>(), 4);
}

#[test]
fn megatron_default_mtp_loss_scaling_factor_is_point_one() {
    assert_eq!(GLM5_MTP_LOSS_SCALING_FACTOR_DEFAULT, 0.1);
}

#[test]
fn native_mtp_rejects_more_than_one_layer() {
    let error = validate_glm5_mtp_contract(2, 1).unwrap_err();
    let message = error.to_string().to_ascii_lowercase();
    assert!(message.contains("mtp"));
    assert!(message.contains("one") || message.contains("exactly"));
}

#[test]
fn native_mtp_rejects_context_parallelism() {
    let error = validate_glm5_mtp_contract(1, 2).unwrap_err();
    let message = error.to_string().to_ascii_lowercase();
    assert!(message.contains("mtp"));
    assert!(message.contains("context") || message.contains("cp"));
    assert!(validate_glm5_mtp_contract(1, 1).is_ok());
    assert!(validate_glm5_mtp_contract(0, 8).is_ok());
}

#[test]
fn native_mtp_accepts_tp_and_ep_only_but_rejects_combined_tp_ep() {
    assert!(validate_glm5_mtp_distributed_contract(1, 2, 1, 1).is_ok());
    assert!(validate_glm5_mtp_distributed_contract(1, 1, 1, 2).is_ok());
    let combined = validate_glm5_mtp_distributed_contract(1, 4, 1, 8).unwrap_err();
    assert!(combined.to_string().contains("sequence-parallel"));
    assert!(validate_glm5_mtp_distributed_contract(1, 1, 1, 1).is_ok());
    assert!(validate_glm5_mtp_distributed_contract(1, 0, 1, 1).is_err());
    assert!(validate_glm5_mtp_distributed_contract(1, 1, 1, 0).is_err());
}

#[test]
fn native_mtp_accumulation_weights_microbatch_means_by_base_tokens() {
    // Megatron's per-token contract is equivalent to accumulating each
    // microbatch numerator after converting its MTP mean back through the
    // base-token count, then dividing once by the aggregate base count.
    let base_counts = [2.0_f64, 5.0];
    let base_means = [0.25_f64, 1.25];
    let mtp_means = [0.5_f64, 2.0];
    let scaling = 0.1_f64;
    let total_base: f64 = base_counts.iter().sum();
    let expected = base_counts
        .iter()
        .zip(base_means.iter().zip(mtp_means.iter()))
        .map(|(count, (base, mtp))| count * (base + scaling * mtp))
        .sum::<f64>()
        / total_base;
    let incorrectly_averaged = base_counts
        .iter()
        .zip(base_means.iter().zip(mtp_means.iter()))
        .map(|(_, (base, mtp))| base + scaling * mtp)
        .sum::<f64>()
        / base_counts.len() as f64;
    assert!((expected - (1.1214285714285714)).abs() < 1.0e-12);
    assert!((expected - incorrectly_averaged).abs() > 0.1);
}

#[test]
fn checkpoint_config_keeps_native_layer_count_explicit() {
    let config = json!({
        "model_type": "glm_moe_dsa",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 128,
        "q_lora_rank": 16,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "rope_parameters": {"rope_theta": 8000000.0, "rope_type": "default"},
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "index_head_dim": 16,
        "index_n_heads": 2,
        "index_topk": 4,
        "indexer_types": ["full"],
        "mlp_layer_types": ["dense"],
        "num_nextn_predict_layers": 1
    });
    let file = NamedTempFile::new().unwrap();
    fs::write(file.path(), serde_json::to_vec(&config).unwrap()).unwrap();
    let parsed = rustrain_glm5::model::read_glm5_config(file.path()).unwrap();
    assert_eq!(parsed.num_nextn_predict_layers, 1);
}
