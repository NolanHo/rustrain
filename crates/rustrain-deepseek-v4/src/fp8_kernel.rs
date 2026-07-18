//! FP8 block-wise GEMM + safetensors loading via C++ FFI (no Python).
//!
//! Two C++ functions:
//! 1. `v4_fp8_scaled_mm` — block-wise FP8 GEMM via at::_scaled_mm (CUTLASS)
//! 2. `v4_create_tensor` — create at::Tensor from raw bytes (FP8 support)
//!
//! Weight loading: Rust parses safetensors header, C++ creates tensors from raw data.

use std::collections::{BTreeMap, HashSet};

use anyhow::{Context, Result, bail};
use tch::{Kind, Tensor};
use tracing::info;

unsafe extern "C" {
    fn v4_glm5_last_error() -> *const std::ffi::c_char;

    fn v4_fp8_scaled_mm(
        a_ptr: *mut std::ffi::c_void,
        b_ptr: *mut std::ffi::c_void,
        scale_a_ptr: *mut std::ffi::c_void,
        scale_b_ptr: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;

    fn v4_fp8_free_tensor(tensor_ptr: *mut std::ffi::c_void);

    fn v4_create_tensor(
        data: *const std::ffi::c_void,
        shape: *const i64,
        shape_len: i32,
        dtype_code: i32,
        device_id: i32,
    ) -> *mut std::ffi::c_void;

    fn v4_dequant_fp8_raw(tensor_ptr: *mut std::ffi::c_void) -> *mut std::ffi::c_void;

    fn v4_fp8_tiled_matmul(
        input_ptr: *mut std::ffi::c_void,
        weight_ptr: *mut std::ffi::c_void,
        scale_ptr: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;

    /// Gradient checkpointing via torch::autograd::Function.
    /// Takes a callback function pointer + input tensor + user context.
    /// Forward: runs callback in no_grad, saves only input.
    /// Backward: recomputes forward with grad, backpropagates.
    fn v4_checkpoint(
        fn_ptr: *mut std::ffi::c_void,
        input_ptr: *mut std::ffi::c_void,
        user_ctx: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;

    /// Helper: create at::Tensor* from raw TensorImpl* (increments refcount)
    fn v4_make_at_tensor(impl_ptr: *mut std::ffi::c_void) -> *mut std::ffi::c_void;

    /// Set the fraction of GPU memory the caching allocator may use.
    fn v4_set_memory_fraction(fraction: f64, device_id: i32);

    /// Empty the caching allocator's free pool (release cached blocks to CUDA).
    fn v4_empty_cache();
}

// ── GLM5 attention kernel (libglm5_attention.so) ──

unsafe extern "C" {
    /// Full DSA attention in one C++ call. See glm5_attention.cpp.
    /// Returns at::Tensor* (owned by caller, free with v4_glm5_free_at_tensor).
    fn v4_glm5_dsa_attention(
        input_ptr: *mut std::ffi::c_void,
        q_a_proj: *mut std::ffi::c_void,
        q_a_layernorm: *mut std::ffi::c_void,
        q_b_proj: *mut std::ffi::c_void,
        kv_a_proj: *mut std::ffi::c_void,
        kv_a_layernorm: *mut std::ffi::c_void,
        kv_b_proj: *mut std::ffi::c_void,
        o_proj: *mut std::ffi::c_void,
        // FP8 scales (nullable)
        q_a_scale: *mut std::ffi::c_void,
        q_b_scale: *mut std::ffi::c_void,
        kv_a_scale: *mut std::ffi::c_void,
        kv_b_scale: *mut std::ffi::c_void,
        o_scale: *mut std::ffi::c_void,
        // Indexer weights (nullable)
        idx_wq_b: *mut std::ffi::c_void,
        idx_wk: *mut std::ffi::c_void,
        idx_k_norm_w: *mut std::ffi::c_void,
        idx_k_norm_b: *mut std::ffi::c_void,
        idx_weights_proj: *mut std::ffi::c_void,
        idx_weights_proj_scale: *mut std::ffi::c_void,
        idx_wq_b_scale: *mut std::ffi::c_void,
        idx_wk_scale: *mut std::ffi::c_void,
        // Config
        batch_i: i32,
        seq_i: i32,
        num_heads_i: i32,
        qk_nope_i: i32,
        qk_rope_i: i32,
        v_head_i: i32,
        kv_lora_i: i32,
        idx_head_dim_i: i32,
        idx_n_heads_i: i32,
        idx_topk_i: i32,
        index_topk_freq_i: i32,
        layer_i: i32,
        is_full_layer: i32,
        rms_eps: f64,
        rope_theta: f64,
        rope_interleave: i32,
        device_id: i32,
        // IndexShare state (in/out)
        topk_indices_ptr: *mut *mut std::ffi::c_void,
        idx_bias_keys_ptr: *mut *mut std::ffi::c_void,
        source_layer: *mut i32,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_free_at_tensor(tensor_ptr: *mut std::ffi::c_void);

    fn v4_glm5_nccl_ring_autograd(
        input_ptr: *mut std::ffi::c_void,
        comm_ptr: *mut std::ffi::c_void,
        send_peer: i64,
        recv_peer: i64,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_nccl_kv_ring_autograd(
        key_ptr: *mut std::ffi::c_void,
        value_ptr: *mut std::ffi::c_void,
        comm_ptr: *mut std::ffi::c_void,
        send_peer: i64,
        recv_peer: i64,
        key_out_ptr: *mut *mut std::ffi::c_void,
        value_out_ptr: *mut *mut std::ffi::c_void,
    );

    fn v4_glm5_mlp_fp8(
        input_ptr: *mut std::ffi::c_void,
        gate_ptr: *mut std::ffi::c_void,
        up_ptr: *mut std::ffi::c_void,
        down_ptr: *mut std::ffi::c_void,
        gate_scale_ptr: *mut std::ffi::c_void,
        up_scale_ptr: *mut std::ffi::c_void,
        down_scale_ptr: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_rms_norm(
        input_ptr: *mut std::ffi::c_void,
        weight_ptr: *mut std::ffi::c_void,
        eps: f64,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_cross_entropy_loss(
        hidden_ptr: *mut std::ffi::c_void,
        lm_head_ptr: *mut std::ffi::c_void,
        targets_ptr: *mut std::ffi::c_void,
        mask_ptr: *mut std::ffi::c_void,
        seq_len: i32,
        vocab: i32,
        chunk_size: i32,
        device_id: i32,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_mtp_prepare(
        hidden_ptr: *mut std::ffi::c_void,
        input_ids_ptr: *mut std::ffi::c_void,
        embed_ptr: *mut std::ffi::c_void,
        enorm_ptr: *mut std::ffi::c_void,
        hnorm_ptr: *mut std::ffi::c_void,
        eh_proj_ptr: *mut std::ffi::c_void,
        eh_proj_scale_ptr: *mut std::ffi::c_void,
        eps: f64,
        token_offset: i32,
        vocab_start: i64,
        global_vocab_size: i64,
        tp_comm: *mut std::ffi::c_void,
        tp_rank: i32,
        tp_size: i32,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_mtp_cross_entropy_loss(
        block_raw_ptr: *mut std::ffi::c_void,
        shared_head_norm_ptr: *mut std::ffi::c_void,
        lm_head_ptr: *mut std::ffi::c_void,
        lm_head_scale_ptr: *mut std::ffi::c_void,
        input_ids_ptr: *mut std::ffi::c_void,
        target_mask_ptr: *mut std::ffi::c_void,
        eps: f64,
        start_offset: i32,
        chunk_size: i32,
        vocab_start: i64,
        global_vocab_size: i64,
        tp_comm: *mut std::ffi::c_void,
        tp_size: i32,
        normalized_out_ptr: *mut *mut std::ffi::c_void,
        loss_sum_out_ptr: *mut *mut std::ffi::c_void,
        token_count_out_ptr: *mut *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_combine_losses(
        lm_loss_ptr: *mut std::ffi::c_void,
        mtp_loss_ptrs: *mut *mut std::ffi::c_void,
        n_mtp_losses: i32,
        mtp_weight: f64,
        mtp_mean_out_ptr: *mut *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;

    fn v4_adam_step(
        params: *mut *mut std::ffi::c_void,
        grads: *mut *mut std::ffi::c_void,
        m_state: *mut *mut std::ffi::c_void,
        v_state: *mut *mut std::ffi::c_void,
        n_params: std::ffi::c_int,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        step: std::ffi::c_int,
    );

    fn v4_glm5_moe_layer(
        mlp_input_ptr: *mut std::ffi::c_void,
        shared_gate: *mut std::ffi::c_void,
        shared_up: *mut std::ffi::c_void,
        shared_down: *mut std::ffi::c_void,
        shared_gate_scale: *mut std::ffi::c_void,
        shared_up_scale: *mut std::ffi::c_void,
        shared_down_scale: *mut std::ffi::c_void,
        gate_weight: *mut std::ffi::c_void,
        correction_bias: *mut std::ffi::c_void,
        expert_gate_weights: *mut *mut std::ffi::c_void,
        expert_up_weights: *mut *mut std::ffi::c_void,
        expert_down_weights: *mut *mut std::ffi::c_void,
        expert_gate_scales: *mut *mut std::ffi::c_void,
        expert_up_scales: *mut *mut std::ffi::c_void,
        expert_down_scales: *mut *mut std::ffi::c_void,
        n_local_experts: i32,
        local_expert_indices: *const i32,
        n_routed_experts: i32,
        topk: i32,
        n_group: i32,
        topk_group: i32,
        scoring_func: i32,
        topk_method: i32,
        norm_topk_prob: i32,
        routed_scaling_factor: f64,
        ep_comm: *mut std::ffi::c_void,
        ep_rank: i32,
        ep_size: i32,
        device_id: i32,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_embedding(
        embed_weight_ptr: *mut std::ffi::c_void,
        input_ids_ptr: *mut std::ffi::c_void,
        device_id: i32,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_layer_forward(
        hidden_ptr: *mut std::ffi::c_void,
        input_norm_weight: *mut std::ffi::c_void,
        post_norm_weight: *mut std::ffi::c_void,
        // Attention weights
        q_a_proj: *mut std::ffi::c_void,
        q_a_layernorm: *mut std::ffi::c_void,
        q_b_proj: *mut std::ffi::c_void,
        kv_a_proj: *mut std::ffi::c_void,
        kv_a_layernorm: *mut std::ffi::c_void,
        kv_b_proj: *mut std::ffi::c_void,
        o_proj: *mut std::ffi::c_void,
        q_a_scale: *mut std::ffi::c_void,
        q_b_scale: *mut std::ffi::c_void,
        kv_a_scale: *mut std::ffi::c_void,
        kv_b_scale: *mut std::ffi::c_void,
        o_scale: *mut std::ffi::c_void,
        idx_wq_b: *mut std::ffi::c_void,
        idx_wk: *mut std::ffi::c_void,
        idx_k_norm_w: *mut std::ffi::c_void,
        idx_k_norm_b: *mut std::ffi::c_void,
        idx_weights_proj: *mut std::ffi::c_void,
        idx_weights_proj_scale: *mut std::ffi::c_void,
        idx_wq_b_scale: *mut std::ffi::c_void,
        idx_wk_scale: *mut std::ffi::c_void,
        // MLP/MoE
        gate_weight: *mut std::ffi::c_void,
        shared_gate: *mut std::ffi::c_void,
        shared_up: *mut std::ffi::c_void,
        shared_down: *mut std::ffi::c_void,
        shared_gate_scale: *mut std::ffi::c_void,
        shared_up_scale: *mut std::ffi::c_void,
        shared_down_scale: *mut std::ffi::c_void,
        dense_gate: *mut std::ffi::c_void,
        dense_up: *mut std::ffi::c_void,
        dense_down: *mut std::ffi::c_void,
        dense_gate_scale: *mut std::ffi::c_void,
        dense_up_scale: *mut std::ffi::c_void,
        dense_down_scale: *mut std::ffi::c_void,
        expert_gate_weights: *mut *mut std::ffi::c_void,
        expert_up_weights: *mut *mut std::ffi::c_void,
        expert_down_weights: *mut *mut std::ffi::c_void,
        expert_gate_scales: *mut *mut std::ffi::c_void,
        expert_up_scales: *mut *mut std::ffi::c_void,
        expert_down_scales: *mut *mut std::ffi::c_void,
        n_local_experts: i32,
        local_expert_indices: *const i32,
        // Config
        batch: i32,
        seq: i32,
        num_heads: i32,
        qk_nope: i32,
        qk_rope: i32,
        v_head: i32,
        kv_lora: i32,
        idx_head_dim: i32,
        idx_n_heads: i32,
        idx_topk: i32,
        index_topk_freq: i32,
        layer: i32,
        is_full_layer: i32,
        is_moe_layer: i32,
        n_routed_experts: i32,
        topk: i32,
        rms_eps: f64,
        rope_theta: f64,
        rope_interleave: i32,
        routed_scaling_factor: f64,
        device_id: i32,
        topk_indices_ptr: *mut *mut std::ffi::c_void,
        idx_bias_keys_ptr: *mut *mut std::ffi::c_void,
        source_layer: *mut i32,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_mtp_decoder_layer(
        descriptor: *const Glm5MtpDecoderDescriptor,
    ) -> *mut std::ffi::c_void;

    fn v4_stream_wait_event(device_id: i32, event_ptr: *mut std::ffi::c_void);
}

/// Stable descriptor for one complete native GLM5 MTP decoder layer. Tensor
/// fields are borrowed `at::Tensor*` pointers and remain owned by Rust for the
/// duration of the call. Null pointers represent optional scales/weights.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Glm5MtpDecoderDescriptor {
    pub hidden: *mut std::ffi::c_void,
    pub input_norm_weight: *mut std::ffi::c_void,
    pub post_norm_weight: *mut std::ffi::c_void,
    pub q_a_proj: *mut std::ffi::c_void,
    pub q_a_layernorm: *mut std::ffi::c_void,
    pub q_b_proj: *mut std::ffi::c_void,
    pub kv_a_proj: *mut std::ffi::c_void,
    pub kv_a_layernorm: *mut std::ffi::c_void,
    pub kv_b_proj: *mut std::ffi::c_void,
    pub o_proj: *mut std::ffi::c_void,
    pub q_a_scale: *mut std::ffi::c_void,
    pub q_b_scale: *mut std::ffi::c_void,
    pub kv_a_scale: *mut std::ffi::c_void,
    pub kv_b_scale: *mut std::ffi::c_void,
    pub o_scale: *mut std::ffi::c_void,
    pub idx_wq_b: *mut std::ffi::c_void,
    pub idx_wk: *mut std::ffi::c_void,
    pub idx_k_norm_w: *mut std::ffi::c_void,
    pub idx_k_norm_b: *mut std::ffi::c_void,
    pub idx_weights_proj: *mut std::ffi::c_void,
    pub idx_weights_proj_scale: *mut std::ffi::c_void,
    pub idx_wq_b_scale: *mut std::ffi::c_void,
    pub idx_wk_scale: *mut std::ffi::c_void,
    pub gate_weight: *mut std::ffi::c_void,
    pub correction_bias: *mut std::ffi::c_void,
    pub shared_gate: *mut std::ffi::c_void,
    pub shared_up: *mut std::ffi::c_void,
    pub shared_down: *mut std::ffi::c_void,
    pub shared_gate_scale: *mut std::ffi::c_void,
    pub shared_up_scale: *mut std::ffi::c_void,
    pub shared_down_scale: *mut std::ffi::c_void,
    pub dense_gate: *mut std::ffi::c_void,
    pub dense_up: *mut std::ffi::c_void,
    pub dense_down: *mut std::ffi::c_void,
    pub dense_gate_scale: *mut std::ffi::c_void,
    pub dense_up_scale: *mut std::ffi::c_void,
    pub dense_down_scale: *mut std::ffi::c_void,
    pub expert_gate_weights: *mut *mut std::ffi::c_void,
    pub expert_up_weights: *mut *mut std::ffi::c_void,
    pub expert_down_weights: *mut *mut std::ffi::c_void,
    pub expert_gate_scales: *mut *mut std::ffi::c_void,
    pub expert_up_scales: *mut *mut std::ffi::c_void,
    pub expert_down_scales: *mut *mut std::ffi::c_void,
    pub local_expert_indices: *const i32,
    pub tp_comm: *mut std::ffi::c_void,
    pub cp_comm: *mut std::ffi::c_void,
    pub ep_comm: *mut std::ffi::c_void,
    pub tp_size: i32,
    pub cp_rank: i32,
    pub cp_size: i32,
    pub ep_rank: i32,
    pub ep_size: i32,
    pub n_local_experts: i32,
    pub n_routed_experts: i32,
    pub topk: i32,
    pub n_group: i32,
    pub topk_group: i32,
    pub scoring_func: i32,
    pub topk_method: i32,
    pub norm_topk_prob: i32,
    pub is_moe_layer: i32,
    pub num_heads: i32,
    pub qk_nope: i32,
    pub qk_rope: i32,
    pub v_head: i32,
    pub kv_lora: i32,
    pub idx_head_dim: i32,
    pub idx_n_heads: i32,
    pub idx_n_heads_global: i32,
    pub idx_topk: i32,
    pub rope_interleave: i32,
    pub indexer_rope_interleave: i32,
    pub rms_eps: f64,
    pub rope_theta: f64,
    pub routed_scaling_factor: f64,
    pub rope_scaling_factor: f64,
    pub rope_beta_fast: f64,
    pub rope_beta_slow: f64,
    pub rope_attention_factor: f64,
    pub rope_original_max_pos: i64,
    pub rope_is_yarn: i32,
}

impl Default for Glm5MtpDecoderDescriptor {
    fn default() -> Self {
        // Pointer fields are intentionally null. Keep neutral RoPE scalar
        // defaults so small local descriptor tests do not zero the rotation.
        let mut descriptor: Self = unsafe { std::mem::zeroed() };
        descriptor.rope_scaling_factor = 1.0;
        descriptor.rope_attention_factor = 1.0;
        descriptor
    }
}

pub fn is_fp8_kernel_available() -> bool {
    let ptr = v4_fp8_scaled_mm as *const ();
    !ptr.is_null()
}

/// Byte-level FP8 → F32 dequant via C++ (bypasses PyTorch's view bug).
/// Takes a GPU FP8 tensor, returns GPU F32 tensor. Scale NOT applied.
pub fn dequant_fp8_raw(fp8_tensor: &Tensor) -> Result<Tensor> {
    let tensor_ptr = unsafe { v4_dequant_fp8_raw(fp8_tensor.as_ptr() as *mut _) };

    if tensor_ptr.is_null() {
        bail!("C++ v4_dequant_fp8_raw returned null");
    }

    let tensor = unsafe { Tensor::clone_from_ptr(tensor_ptr as *mut _) };
    unsafe { v4_fp8_free_tensor(tensor_ptr) };
    Ok(tensor)
}

/// Dequant FP8 weight to BF16 with block-wise scale applied.
/// Uses byte-level C++ dequant (no PyTorch FP8 ops) then applies scale in Rust.
///
/// - `fp8_weight`: [N, K] FP8 tensor on GPU
/// - `scale`: [ceil(N/128), ceil(K/128)] F32 tensor on GPU
/// Returns: [N, K] BF16 tensor on GPU
pub fn dequant_fp8_weight(fp8_weight: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let n = fp8_weight.size()[0];
    let k = fp8_weight.size()[1];

    // Step 1: byte-level FP8 → F32 (no PyTorch view bug)
    let f32_weight = dequant_fp8_raw(fp8_weight)?;

    // Step 2: expand scale from [n_blocks, k_blocks] to [N, K] and multiply
    let n_blocks = (n + 127) / 128;
    let k_blocks = (k + 127) / 128;
    let scale_expanded = if scale.size() == [n_blocks, k_blocks] {
        // Expand scale to [n_blocks*128, k_blocks*128] then crop to [N, K]
        let expanded = scale
            .repeat_interleave_self_int(128, 0, None)
            .repeat_interleave_self_int(128, 1, None);
        // Crop to actual [N, K]
        expanded.narrow(0, 0, n).narrow(1, 0, k)
    } else {
        // Scale already has matching shape
        scale.shallow_clone()
    };

    // Step 3: apply scale and convert to BF16
    let result = (&f32_weight * &scale_expanded).to_kind(Kind::BFloat16);
    Ok(result)
}

pub fn ue8m0_to_float_scale(scale_u8: &Tensor) -> Tensor {
    let f = scale_u8.to_kind(Kind::Float);
    let ln2 = std::f64::consts::LN_2;
    (f * ln2).exp()
}

// ── Safetensors parsing (Rust side, no Python) ──

struct TensorMeta {
    dtype: String,
    shape: Vec<i64>,
    data_offsets: (usize, usize),
}

fn parse_safetensors_header(
    path: &std::path::Path,
) -> Result<std::collections::HashMap<String, TensorMeta>> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;

    let mut header_size_buf = [0u8; 8];
    file.read_exact(&mut header_size_buf)?;
    let header_size = u64::from_le_bytes(header_size_buf) as usize;

    let mut header_json = vec![0u8; header_size];
    file.read_exact(&mut header_json)?;
    let header_str = String::from_utf8(header_json)?;
    let header: serde_json::Value = serde_json::from_str(&header_str)?;

    let mut tensors = std::collections::HashMap::new();
    if let Some(obj) = header.as_object() {
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = info["dtype"].as_str().unwrap_or("").to_string();
            let shape: Vec<i64> = info["shape"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_i64().unwrap_or(0)).collect())
                .unwrap_or_default();
            let offsets = info["data_offsets"].as_array();
            let (start, end) = if let Some(arr) = offsets {
                (
                    arr[0].as_u64().unwrap_or(0) as usize,
                    arr[1].as_u64().unwrap_or(0) as usize,
                )
            } else {
                (0, 0)
            };
            tensors.insert(
                name.clone(),
                TensorMeta {
                    dtype,
                    shape,
                    data_offsets: (start, end),
                },
            );
        }
    }
    Ok(tensors)
}

fn dtype_str_to_code(dtype: &str) -> i32 {
    match dtype {
        "F8_E4M3" => 0,
        "F32" => 1,
        "BF16" | "BF16_" => 2,
        "U8" => 3,
        "I64" => 4,
        _ => 1,
    }
}

/// Load tensors from a safetensors file using C++ (no Python).
///
/// Uses mmap instead of reading the entire file into RAM.
/// Only the needed tensor byte ranges are paged in by the OS.
pub fn load_safetensors_native(
    path: &std::path::Path,
    needed: &HashSet<String>,
    device_id: i32,
) -> Result<BTreeMap<String, Tensor>> {
    use std::io::Read;

    let metadata = parse_safetensors_header(path)?;
    info!(
        tensors_total = metadata.len(),
        needed = needed.len(),
        "parsing safetensors header"
    );

    let header_size = {
        let mut file = std::fs::File::open(path)?;
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)?;
        8 + u64::from_le_bytes(buf) as usize
    };

    // mmap the file — only needed pages are loaded by the OS
    let file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len() as usize;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let file_data = &mmap[..];
    let data_start = header_size;

    let mut result = BTreeMap::new();

    for (name, meta) in &metadata {
        let is_needed = needed.contains(name) || {
            let base = name.replace(".scale", "");
            needed.contains(&base) || needed.contains(&format!("{base}.scale"))
        };

        if !is_needed {
            continue;
        }

        let (offset_start, offset_end) = meta.data_offsets;
        let abs_start = data_start + offset_start;
        let abs_end = data_start + offset_end;
        let tensor_bytes = &file_data[abs_start..abs_end];
        let dtype_code = dtype_str_to_code(&meta.dtype);

        let tensor_ptr = unsafe {
            v4_create_tensor(
                tensor_bytes.as_ptr() as *const _,
                meta.shape.as_ptr(),
                meta.shape.len() as i32,
                dtype_code,
                device_id,
            )
        };

        if tensor_ptr.is_null() {
            bail!("C++ v4_create_tensor returned null for tensor '{name}'");
        }

        let tensor = unsafe { Tensor::clone_from_ptr(tensor_ptr as *mut _) };
        unsafe { v4_fp8_free_tensor(tensor_ptr) };

        if name.ends_with(".scale") {
            let float_scale = if tensor.kind() == Kind::Uint8 {
                ue8m0_to_float_scale(&tensor)
            } else {
                tensor.to_kind(Kind::Float)
            };
            let scale_name = name.replace(".scale", ".scale_f");
            result.insert(scale_name, float_scale);
        } else {
            result.insert(name.clone(), tensor);
        }
    }

    info!(loaded = result.len(), "safetensors loaded (no Python)");
    Ok(result)
}

// ── FP8 GEMM ──

pub fn quantize_to_fp8(input: &Tensor) -> (Tensor, Tensor) {
    let shape = input.size();
    let m = shape[0];
    let k = shape[1];

    // Pad M to 128 multiple (required by _scaled_mm block-wise path)
    let m_padded = ((m + 127) / 128) * 128;
    let m_blocks = m_padded / 128;
    let k_blocks = k / 128;

    let input_padded = if m_padded != m {
        let pad = Tensor::zeros([m_padded - m, k], (input.kind(), input.device()));
        let input_clone = input.shallow_clone();
        let tensors: Vec<&Tensor> = [&input_clone, &pad].to_vec();
        Tensor::cat(&tensors, 0)
    } else {
        input.shallow_clone()
    };

    let reshaped = input_padded
        .to_kind(Kind::Float)
        .reshape([m_blocks, 128, k_blocks, 128]);

    let block_abs_max = reshaped.abs().amax([1, 3].as_slice(), true);
    let fp8_max = 448.0f64;
    let scale = (block_abs_max / fp8_max).clamp_min(1e-12);
    let scale_2d = scale.squeeze_dim(1).squeeze_dim(2).to_kind(Kind::Float);

    let scale_expanded = scale
        .reshape([m_blocks, 1, k_blocks, 1])
        .expand([m_blocks, 128, k_blocks, 128], false);
    let quantized = (reshaped / &scale_expanded).reshape([m_padded, k]);

    // Crop back to original M
    let quantized = if m_padded != m {
        quantized.narrow(0, 0, m)
    } else {
        quantized
    };

    (quantized.to_kind(Kind::Float), scale_2d)
}

pub fn expand_weight_scale(scale_128x128: &Tensor, n: i64, k: i64) -> Tensor {
    let n_blocks = n / 128;
    let k_blocks = k / 128;
    scale_128x128
        .transpose(0, 1)
        .contiguous()
        .reshape([k_blocks, n_blocks, 1])
        .expand([k_blocks, n_blocks, 128], false)
        .reshape([k_blocks, n])
        .contiguous()
}

pub fn fp8_linear(input: &Tensor, weight_fp8: &Tensor, weight_scale: &Tensor) -> Result<Tensor> {
    if !matches!(input.device(), tch::Device::Cuda(_)) {
        bail!("FP8 GEMM requires CUDA tensors");
    }

    // Flatten 3D+ input to 2D [M, K]
    let original_shape = input.size();
    let input_2d = if input.dim() > 2 {
        let batch: i64 = original_shape[..original_shape.len() - 1].iter().product();
        input.reshape([batch, original_shape[original_shape.len() - 1]])
    } else {
        input.shallow_clone()
    };

    let m = input_2d.size()[0];
    let k = input_2d.size()[1];
    let n = weight_fp8.size()[0];

    // Use tiled matmul: dequant 128 rows at a time + BF16 cublas GEMM.
    // Peak temp: ~4.5MB (vs ~75MB for full dequant).
    let result_ptr = unsafe {
        v4_fp8_tiled_matmul(
            input_2d.as_ptr() as *mut _,
            weight_fp8.as_ptr() as *mut _,
            weight_scale.as_ptr() as *mut _,
        )
    };

    if result_ptr.is_null() {
        bail!("FP8 tiled matmul returned null (M={m}, N={n}, K={k})");
    }

    let result = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_fp8_free_tensor(result_ptr) };

    // Reshape back to original dimensions if input was 3D+
    if input.dim() > 2 {
        let mut out_shape: Vec<i64> = original_shape[..original_shape.len() - 1].to_vec();
        out_shape.push(n);
        Ok(result.reshape(out_shape))
    } else {
        Ok(result)
    }
}

pub fn fp8_linear_bias(
    input: &Tensor,
    weight_fp8: &Tensor,
    weight_scale: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let out = fp8_linear(input, weight_fp8, weight_scale)?;
    match bias {
        Some(b) => Ok(out + b),
        None => Ok(out),
    }
}

// ── Gradient Checkpointing ──

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

type ForwardFn = Box<dyn Fn(&Tensor) -> Result<Tensor> + Send>;
type SharedForwardFn = Arc<Mutex<ForwardFn>>;

static CHECKPOINT_REGISTRY: OnceLock<Mutex<HashMap<usize, SharedForwardFn>>> = OnceLock::new();
static CHECKPOINT_ERRORS: OnceLock<Mutex<HashMap<usize, String>>> = OnceLock::new();
static CHECKPOINT_COUNTER: AtomicUsize = AtomicUsize::new(1);

fn registry() -> &'static Mutex<HashMap<usize, SharedForwardFn>> {
    CHECKPOINT_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn registry_lock() -> MutexGuard<'static, HashMap<usize, SharedForwardFn>> {
    registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn checkpoint_errors() -> &'static Mutex<HashMap<usize, String>> {
    CHECKPOINT_ERRORS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn checkpoint_errors_lock() -> MutexGuard<'static, HashMap<usize, String>> {
    checkpoint_errors()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn record_checkpoint_error(key: usize, message: String) {
    eprintln!("[checkpoint_callback] {message}");
    checkpoint_errors_lock().insert(key, message);
}

/// C callback invoked by C++ CheckpointFunction (both forward and backward).
/// Receives a raw TensorImpl* and returns an at::Tensor* (via v4_make_at_tensor).
extern "C" fn checkpoint_callback(
    impl_ptr: *mut std::ffi::c_void,
    user_ctx: *mut std::ffi::c_void,
) -> *mut std::ffi::c_void {
    let key = user_ctx as usize;

    // Clone the tensor from C++ (clone_from_ptr calls at_clone, increments refcount)
    let input = unsafe { Tensor::clone_from_ptr(impl_ptr as *mut _) };

    // Run the forward function from registry
    let output = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let forward = registry_lock()
            .get(&key)
            .cloned()
            .with_context(|| format!("checkpoint forward function not found for key {key}"))?;
        let forward = forward
            .lock()
            .map_err(|_| anyhow::anyhow!("checkpoint forward lock is poisoned for key {key}"))?;
        forward(&input)
    })) {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            record_checkpoint_error(key, format!("forward failed for key {key}: {error:#}"));
            return std::ptr::null_mut();
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            record_checkpoint_error(key, format!("forward panicked for key {key}: {message}"));
            return std::ptr::null_mut();
        }
    };

    // Return a new at::Tensor* to C++ (v4_make_at_tensor copy constructor increments refcount)
    unsafe { v4_make_at_tensor(output.as_ptr() as *mut std::ffi::c_void) }
}

/// Wrap a forward function with gradient checkpointing.
/// Forward: runs `forward(input)` in no_grad, saves only the input.
/// Backward: recomputes `forward(input)` with grad, backpropagates.
///
/// The forward closure must be deterministic (same input → same output).
/// It will be called TWICE: once during forward, once during backward.
pub fn checkpoint<F>(input: &Tensor, forward: F) -> Tensor
where
    F: Fn(&Tensor) -> Tensor + Send + 'static,
{
    checkpoint_result(input, move |input| Ok(forward(input)))
        .unwrap_or_else(|error| panic!("gradient checkpoint failed: {error:#}"))
}

/// Fallible gradient checkpoint wrapper for C++ kernels whose forward can
/// report validation, allocation, or device errors.
pub fn checkpoint_result<F>(input: &Tensor, forward: F) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + 'static,
{
    let key = CHECKPOINT_COUNTER.fetch_add(1, Ordering::SeqCst);
    registry_lock().insert(key, Arc::new(Mutex::new(Box::new(forward))));

    let result_ptr = unsafe {
        v4_checkpoint(
            checkpoint_callback as *mut std::ffi::c_void,
            input.as_ptr() as *mut std::ffi::c_void,
            key as *mut std::ffi::c_void,
        )
    };

    if result_ptr.is_null() {
        registry_lock().remove(&key);
        let detail = checkpoint_errors_lock()
            .remove(&key)
            .unwrap_or_else(|| "C++ v4_checkpoint returned null".to_string());
        bail!("{detail}");
    }

    let result = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_fp8_free_tensor(result_ptr) };

    // Registry entry is kept for backward pass.
    // It will be cleaned up via clear_checkpoint_registry() after each training step.

    Ok(result)
}

/// Clear all checkpoint registry entries. Call after each training step
/// to prevent GPU memory leak (closures hold tensor references).
pub fn clear_checkpoint_registry() {
    registry_lock().clear();
    checkpoint_errors_lock().clear();
    CHECKPOINT_COUNTER.store(1, Ordering::SeqCst);
}

/// Set the fraction of GPU memory the caching allocator is allowed to use.
/// Call early (before any training allocations) to pre-expand the pool.
pub fn set_memory_fraction(fraction: f64, device_id: i32) {
    unsafe { v4_set_memory_fraction(fraction, device_id) };
}

/// Empty the caching allocator's free pool (release cached blocks back to CUDA).
/// Call after warmup pass so that step 0 starts with a clean but pre-warmed pool.
pub fn empty_cache() {
    unsafe { v4_empty_cache() };
}

// ── GLM5 attention C++ kernel wrapper ──

/// Check if the C++ GLM5 attention kernel is available.
pub fn is_glm5_attention_available() -> bool {
    // If libglm5_attention.so was linked, v4_glm5_dsa_attention is non-null.
    let ptr = v4_glm5_dsa_attention as *const ();
    !ptr.is_null()
}

/// C++ IndexShare state — raw pointers to at::Tensor*, managed by C++.
/// Must be freed via `drop_glm5_index_state()` before going out of scope.
pub struct Glm5IndexState {
    pub topk_indices: *mut std::ffi::c_void, // at::Tensor* or null
    pub idx_bias_keys: *mut std::ffi::c_void, // at::Tensor* or null
    pub source_layer: i32,
}

impl Default for Glm5IndexState {
    fn default() -> Self {
        Self {
            topk_indices: std::ptr::null_mut(),
            idx_bias_keys: std::ptr::null_mut(),
            source_layer: -1,
        }
    }
}

impl Glm5IndexState {
    pub fn is_none(&self) -> bool {
        self.topk_indices.is_null()
    }
}

impl Drop for Glm5IndexState {
    fn drop(&mut self) {
        // Free at::Tensor* created by C++ to prevent memory leak
        if !self.topk_indices.is_null() {
            unsafe { v4_glm5_free_at_tensor(self.topk_indices) };
            self.topk_indices = std::ptr::null_mut();
        }
        if !self.idx_bias_keys.is_null() {
            unsafe { v4_glm5_free_at_tensor(self.idx_bias_keys) };
            self.idx_bias_keys = std::ptr::null_mut();
        }
    }
}

fn glm5_ffi_error(operation: &str) -> anyhow::Error {
    let detail = unsafe {
        let ptr = v4_glm5_last_error();
        if ptr.is_null() {
            None
        } else {
            let message = std::ffi::CStr::from_ptr(ptr).to_string_lossy();
            (!message.is_empty()).then(|| message.into_owned())
        }
    };
    anyhow::anyhow!(
        "C++ {operation} failed{}",
        detail
            .map(|message| format!(": {message}"))
            .unwrap_or_default()
    )
}

/// Call the C++ GLM5 DSA attention kernel.
///
/// Returns the attention output tensor. The `index_state` is updated in-place
/// (C++ writes new topk_indices/idx_bias_keys into it when `is_full_layer` is true).
///
/// All weight tensors are passed as `&Tensor` — their raw `at::Tensor*` pointers
/// are extracted via `as_ptr()` and passed directly to C++.
pub fn glm5_dsa_attention_cpp(
    input: &Tensor,
    // Attention weights
    q_a_proj: &Tensor,
    q_a_layernorm: &Tensor,
    q_b_proj: &Tensor,
    kv_a_proj: &Tensor,
    kv_a_layernorm: &Tensor,
    kv_b_proj: &Tensor,
    o_proj: &Tensor,
    // FP8 scales (optional — pass empty tensor if not used)
    q_a_scale: Option<&Tensor>,
    q_b_scale: Option<&Tensor>,
    kv_a_scale: Option<&Tensor>,
    kv_b_scale: Option<&Tensor>,
    o_scale: Option<&Tensor>,
    // Indexer weights (optional)
    idx_wq_b: Option<&Tensor>,
    idx_wk: Option<&Tensor>,
    idx_k_norm_w: Option<&Tensor>,
    idx_k_norm_b: Option<&Tensor>,
    idx_weights_proj: Option<&Tensor>,
    idx_weights_proj_scale: Option<&Tensor>,
    idx_wq_b_scale: Option<&Tensor>,
    idx_wk_scale: Option<&Tensor>,
    // Config
    batch: i32,
    seq: i32,
    num_heads: i32,
    qk_nope: i32,
    qk_rope: i32,
    v_head: i32,
    kv_lora: i32,
    idx_head_dim: i32,
    idx_n_heads: i32,
    idx_topk: i32,
    index_topk_freq: i32,
    layer: i32,
    is_full_layer: bool,
    rms_eps: f64,
    rope_theta: f64,
    rope_interleave: bool,
    device_id: i32,
    // IndexShare state (in/out)
    index_state: &mut Glm5IndexState,
) -> Result<Tensor> {
    fn opt_ptr(t: Option<&Tensor>) -> *mut std::ffi::c_void {
        match t {
            Some(t) if t.numel() > 0 => t.as_ptr() as *mut _,
            _ => std::ptr::null_mut(),
        }
    }

    let result_ptr = unsafe {
        v4_glm5_dsa_attention(
            input.as_ptr() as *mut _,
            q_a_proj.as_ptr() as *mut _,
            q_a_layernorm.as_ptr() as *mut _,
            q_b_proj.as_ptr() as *mut _,
            kv_a_proj.as_ptr() as *mut _,
            kv_a_layernorm.as_ptr() as *mut _,
            kv_b_proj.as_ptr() as *mut _,
            o_proj.as_ptr() as *mut _,
            opt_ptr(q_a_scale),
            opt_ptr(q_b_scale),
            opt_ptr(kv_a_scale),
            opt_ptr(kv_b_scale),
            opt_ptr(o_scale),
            opt_ptr(idx_wq_b),
            opt_ptr(idx_wk),
            opt_ptr(idx_k_norm_w),
            opt_ptr(idx_k_norm_b),
            opt_ptr(idx_weights_proj),
            opt_ptr(idx_weights_proj_scale),
            opt_ptr(idx_wq_b_scale),
            opt_ptr(idx_wk_scale),
            batch,
            seq,
            num_heads,
            qk_nope,
            qk_rope,
            v_head,
            kv_lora,
            idx_head_dim,
            idx_n_heads,
            idx_topk,
            index_topk_freq,
            layer,
            if is_full_layer { 1 } else { 0 },
            rms_eps,
            rope_theta,
            if rope_interleave { 1 } else { 0 },
            device_id,
            &mut index_state.topk_indices,
            &mut index_state.idx_bias_keys,
            &mut index_state.source_layer,
        )
    };

    if result_ptr.is_null() {
        return Err(glm5_ffi_error("v4_glm5_dsa_attention"));
    }

    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Call the C++ GLM5 MLP kernel (SwiGLU: silu(gate(x)) * up(x) → down).
pub fn glm5_mlp_fp8_cpp(
    input: &Tensor,
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
    gate_scale: Option<&Tensor>,
    up_scale: Option<&Tensor>,
    down_scale: Option<&Tensor>,
) -> Result<Tensor> {
    fn opt_ptr(t: Option<&Tensor>) -> *mut std::ffi::c_void {
        match t {
            Some(t) if t.numel() > 0 => t.as_ptr() as *mut _,
            _ => std::ptr::null_mut(),
        }
    }
    let result_ptr = unsafe {
        v4_glm5_mlp_fp8(
            input.as_ptr() as *mut _,
            gate.as_ptr() as *mut _,
            up.as_ptr() as *mut _,
            down.as_ptr() as *mut _,
            opt_ptr(gate_scale),
            opt_ptr(up_scale),
            opt_ptr(down_scale),
        )
    };
    if result_ptr.is_null() {
        return Err(glm5_ffi_error("v4_glm5_mlp_fp8"));
    }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Call the C++ RMSNorm kernel.
pub fn glm5_rms_norm_cpp(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let result_ptr =
        unsafe { v4_glm5_rms_norm(input.as_ptr() as *mut _, weight.as_ptr() as *mut _, eps) };
    if result_ptr.is_null() {
        return Err(glm5_ffi_error("v4_glm5_rms_norm"));
    }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Call the C++ chunked cross-entropy loss kernel.
pub fn glm5_cross_entropy_loss_cpp(
    hidden: &Tensor,
    lm_head: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
    seq_len: i32,
    vocab: i32,
    chunk_size: i32,
    device_id: i32,
) -> Result<Tensor> {
    let result_ptr = unsafe {
        v4_glm5_cross_entropy_loss(
            hidden.as_ptr() as *mut _,
            lm_head.as_ptr() as *mut _,
            targets.as_ptr() as *mut _,
            mask.as_ptr() as *mut _,
            seq_len,
            vocab,
            chunk_size,
            device_id,
        )
    };
    if result_ptr.is_null() {
        return Err(glm5_ffi_error("v4_glm5_cross_entropy_loss"));
    }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Exchange one CP ring block while preserving autograd across ranks.
///
/// Forward sends `input` to `send_peer` and receives from `recv_peer`.
/// Backward reverses those edges so the received activation gradient reaches
/// the rank that produced the corresponding forward block.
pub fn glm5_nccl_ring_autograd_cpp(
    input: &Tensor,
    comm_ptr: *mut std::ffi::c_void,
    send_peer: usize,
    recv_peer: usize,
) -> Result<Tensor> {
    let result_ptr = unsafe {
        v4_glm5_nccl_ring_autograd(
            input.as_ptr() as *mut _,
            comm_ptr,
            send_peer as i64,
            recv_peer as i64,
        )
    };
    if result_ptr.is_null() {
        return Err(glm5_ffi_error("v4_glm5_nccl_ring_autograd"));
    }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

pub fn glm5_nccl_kv_ring_autograd_cpp(
    key: &Tensor,
    value: &Tensor,
    comm_ptr: *mut std::ffi::c_void,
    send_peer: usize,
    recv_peer: usize,
) -> Result<(Tensor, Tensor)> {
    let mut key_ptr = std::ptr::null_mut();
    let mut value_ptr = std::ptr::null_mut();
    unsafe {
        v4_glm5_nccl_kv_ring_autograd(
            key.as_ptr() as *mut _,
            value.as_ptr() as *mut _,
            comm_ptr,
            send_peer as i64,
            recv_peer as i64,
            &mut key_ptr,
            &mut value_ptr,
        );
    }
    if key_ptr.is_null() || value_ptr.is_null() {
        for ptr in [key_ptr, value_ptr] {
            if !ptr.is_null() {
                unsafe { v4_glm5_free_at_tensor(ptr) };
            }
        }
        return Err(glm5_ffi_error("v4_glm5_nccl_kv_ring_autograd"));
    }
    let key_out = unsafe { Tensor::clone_from_ptr(key_ptr as *mut _) };
    let value_out = unsafe { Tensor::clone_from_ptr(value_ptr as *mut _) };
    unsafe {
        v4_glm5_free_at_tensor(key_ptr);
        v4_glm5_free_at_tensor(value_ptr);
    }
    Ok((key_out, value_out))
}

/// Prepare a GLM5 MTP teacher-forcing block.
///
/// At position `t`, this computes
/// `eh_proj(cat(enorm(embed(input_ids[t + token_offset])), hnorm(hidden[t])))`.
/// Its length is `min(hidden_len, input_len - token_offset - 1)` and position
/// `t` is aligned with target `input_ids[t + token_offset + 1]`.
pub fn glm5_mtp_prepare_cpp(
    hidden_post_norm: &Tensor,
    input_ids: &Tensor,
    embed: &Tensor,
    enorm: &Tensor,
    hnorm: &Tensor,
    eh_proj: &Tensor,
    eh_proj_scale: Option<&Tensor>,
    eps: f64,
    token_offset: i32,
) -> Result<Tensor> {
    glm5_mtp_prepare_tp_cpp(
        hidden_post_norm,
        input_ids,
        embed,
        enorm,
        hnorm,
        eh_proj,
        eh_proj_scale,
        eps,
        token_offset,
        0,
        embed.size()[0],
        std::ptr::null_mut(),
        0,
        1,
    )
}

/// TP-aware MTP teacher-forcing projection.
///
/// `eh_proj` is `[H / tp_size, 2H]`. Multi-rank execution gathers the local
/// projection along the hidden dimension in forward, splits it in backward,
/// and sums the replicated projection-input gradient across TP ranks.
#[allow(clippy::too_many_arguments)]
pub fn glm5_mtp_prepare_tp_cpp(
    hidden_post_norm: &Tensor,
    input_ids: &Tensor,
    embed: &Tensor,
    enorm: &Tensor,
    hnorm: &Tensor,
    eh_proj: &Tensor,
    eh_proj_scale: Option<&Tensor>,
    eps: f64,
    token_offset: i32,
    vocab_start: i64,
    global_vocab_size: i64,
    tp_comm: *mut std::ffi::c_void,
    tp_rank: i32,
    tp_size: i32,
) -> Result<Tensor> {
    let scale_ptr = eh_proj_scale
        .filter(|scale| scale.numel() > 0)
        .map_or(std::ptr::null_mut(), |scale| scale.as_ptr() as *mut _);
    let result_ptr = unsafe {
        v4_glm5_mtp_prepare(
            hidden_post_norm.as_ptr() as *mut _,
            input_ids.as_ptr() as *mut _,
            embed.as_ptr() as *mut _,
            enorm.as_ptr() as *mut _,
            hnorm.as_ptr() as *mut _,
            eh_proj.as_ptr() as *mut _,
            scale_ptr,
            eps,
            token_offset,
            vocab_start,
            global_vocab_size,
            tp_comm,
            tp_rank,
            tp_size,
        )
    };
    if result_ptr.is_null() {
        return Err(glm5_ffi_error("v4_glm5_mtp_prepare"));
    }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Normalize one MTP block and compute masked chunked next-token CE in one FFI.
///
/// `block_raw[:, j]` predicts `input_ids[:, start_offset + j + 2]`.
/// The result retains the normalized chain state, numerator, token count, and
/// mask-normalized loss; no MTP loss weight is applied here.
pub struct Glm5MtpPostprocessOutput {
    pub normalized: Tensor,
    pub loss: Tensor,
    pub loss_sum: Tensor,
    pub token_count: Tensor,
}

pub fn glm5_mtp_postprocess_loss_cpp(
    block_raw: &Tensor,
    shared_head_norm: &Tensor,
    lm_head: &Tensor,
    lm_head_scale: Option<&Tensor>,
    input_ids: &Tensor,
    target_mask: &Tensor,
    eps: f64,
    start_offset: i32,
    chunk_size: i32,
) -> Result<Glm5MtpPostprocessOutput> {
    glm5_mtp_postprocess_loss_vocab_parallel_cpp(
        block_raw,
        shared_head_norm,
        lm_head,
        lm_head_scale,
        input_ids,
        target_mask,
        eps,
        start_offset,
        chunk_size,
        0,
        lm_head.size()[0],
        std::ptr::null_mut(),
        1,
    )
}

/// Megatron-compatible vocabulary-parallel MTP CE. `lm_head` contains only
/// the local vocab shard `[vocab_start, vocab_start + local_vocab)`. The NCCL
/// communicator is required only when `tp_size > 1`; the kernel performs the
/// global MAX/SUM reductions and keeps the input gradient all-reduced.
pub fn glm5_mtp_postprocess_loss_vocab_parallel_cpp(
    block_raw: &Tensor,
    shared_head_norm: &Tensor,
    lm_head: &Tensor,
    lm_head_scale: Option<&Tensor>,
    input_ids: &Tensor,
    target_mask: &Tensor,
    eps: f64,
    start_offset: i32,
    chunk_size: i32,
    vocab_start: i64,
    global_vocab_size: i64,
    tp_comm: *mut std::ffi::c_void,
    tp_size: i32,
) -> Result<Glm5MtpPostprocessOutput> {
    let scale_ptr = lm_head_scale
        .filter(|scale| scale.numel() > 0)
        .map_or(std::ptr::null_mut(), |scale| scale.as_ptr() as *mut _);
    let mut normalized_ptr = std::ptr::null_mut();
    let mut loss_sum_ptr = std::ptr::null_mut();
    let mut token_count_ptr = std::ptr::null_mut();
    let result_ptr = unsafe {
        v4_glm5_mtp_cross_entropy_loss(
            block_raw.as_ptr() as *mut _,
            shared_head_norm.as_ptr() as *mut _,
            lm_head.as_ptr() as *mut _,
            scale_ptr,
            input_ids.as_ptr() as *mut _,
            target_mask.as_ptr() as *mut _,
            eps,
            start_offset,
            chunk_size,
            vocab_start,
            global_vocab_size,
            tp_comm,
            tp_size,
            &mut normalized_ptr,
            &mut loss_sum_ptr,
            &mut token_count_ptr,
        )
    };
    if result_ptr.is_null()
        || normalized_ptr.is_null()
        || loss_sum_ptr.is_null()
        || token_count_ptr.is_null()
    {
        for ptr in [result_ptr, normalized_ptr, loss_sum_ptr, token_count_ptr] {
            if !ptr.is_null() {
                unsafe { v4_glm5_free_at_tensor(ptr) };
            }
        }
        return Err(glm5_ffi_error("v4_glm5_mtp_cross_entropy_loss"));
    }
    let normalized = unsafe { Tensor::clone_from_ptr(normalized_ptr as *mut _) };
    let loss = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    let loss_sum = unsafe { Tensor::clone_from_ptr(loss_sum_ptr as *mut _) };
    let token_count = unsafe { Tensor::clone_from_ptr(token_count_ptr as *mut _) };
    unsafe {
        v4_glm5_free_at_tensor(normalized_ptr);
        v4_glm5_free_at_tensor(result_ptr);
        v4_glm5_free_at_tensor(loss_sum_ptr);
        v4_glm5_free_at_tensor(token_count_ptr);
    }
    Ok(Glm5MtpPostprocessOutput {
        normalized,
        loss,
        loss_sum,
        token_count,
    })
}

pub fn glm5_mtp_cross_entropy_loss_cpp(
    block_raw: &Tensor,
    shared_head_norm: &Tensor,
    lm_head: &Tensor,
    lm_head_scale: Option<&Tensor>,
    input_ids: &Tensor,
    target_mask: &Tensor,
    eps: f64,
    start_offset: i32,
    chunk_size: i32,
) -> Result<Tensor> {
    glm5_mtp_postprocess_loss_cpp(
        block_raw,
        shared_head_norm,
        lm_head,
        lm_head_scale,
        input_ids,
        target_mask,
        eps,
        start_offset,
        chunk_size,
    )
    .map(|output| output.loss)
}

pub struct Glm5CombinedLossOutput {
    pub total: Tensor,
    pub mtp_mean: Tensor,
}

pub fn glm5_combine_losses_cpp(
    lm_loss: &Tensor,
    mtp_losses: &[Tensor],
    mtp_weight: f64,
) -> Result<Glm5CombinedLossOutput> {
    if mtp_losses.is_empty() {
        anyhow::bail!("GLM5 loss combine requires at least one MTP loss");
    }
    let mut loss_ptrs: Vec<*mut std::ffi::c_void> = mtp_losses
        .iter()
        .map(|loss| loss.as_ptr() as *mut _)
        .collect();
    let mut mean_ptr = std::ptr::null_mut();
    let total_ptr = unsafe {
        v4_glm5_combine_losses(
            lm_loss.as_ptr() as *mut _,
            loss_ptrs.as_mut_ptr(),
            loss_ptrs.len() as i32,
            mtp_weight,
            &mut mean_ptr,
        )
    };
    if total_ptr.is_null() || mean_ptr.is_null() {
        for ptr in [total_ptr, mean_ptr] {
            if !ptr.is_null() {
                unsafe { v4_glm5_free_at_tensor(ptr) };
            }
        }
        return Err(glm5_ffi_error("v4_glm5_combine_losses"));
    }
    let total = unsafe { Tensor::clone_from_ptr(total_ptr as *mut _) };
    let mtp_mean = unsafe { Tensor::clone_from_ptr(mean_ptr as *mut _) };
    unsafe {
        v4_glm5_free_at_tensor(total_ptr);
        v4_glm5_free_at_tensor(mean_ptr);
    }
    Ok(Glm5CombinedLossOutput { total, mtp_mean })
}

/// Call the C++ Adam optimizer step. Updates params, m, v in-place.
pub fn adam_step_cpp(
    params: &mut [Tensor],
    grads: &[Tensor],
    m: &mut [Tensor],
    v: &mut [Tensor],
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    step: i32,
) {
    let n = params.len() as std::ffi::c_int;
    // Build raw pointer arrays
    let mut param_ptrs: Vec<*mut std::ffi::c_void> =
        params.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut grad_ptrs: Vec<*mut std::ffi::c_void> =
        grads.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut m_ptrs: Vec<*mut std::ffi::c_void> = m.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut v_ptrs: Vec<*mut std::ffi::c_void> = v.iter().map(|t| t.as_ptr() as *mut _).collect();
    unsafe {
        v4_adam_step(
            param_ptrs.as_mut_ptr(),
            grad_ptrs.as_mut_ptr(),
            m_ptrs.as_mut_ptr(),
            v_ptrs.as_mut_ptr(),
            n,
            lr,
            beta1,
            beta2,
            eps,
            step,
        );
    }
}

/// Call the C++ MoE layer kernel (routing + expert dispatch + shared expert + combine).
/// Expert weights are CPU tensors — C++ does to_device internally.
pub fn glm5_moe_layer_ep_cpp(
    mlp_input: &Tensor,
    shared_gate: &Tensor,
    shared_up: &Tensor,
    shared_down: &Tensor,
    shared_gate_scale: Option<&Tensor>,
    shared_up_scale: Option<&Tensor>,
    shared_down_scale: Option<&Tensor>,
    gate_weight: &Tensor,
    correction_bias: Option<&Tensor>,
    // Expert weights (CPU)
    expert_gate_weights: &[&Tensor],
    expert_up_weights: &[&Tensor],
    expert_down_weights: &[&Tensor],
    expert_gate_scales: &[Option<&Tensor>],
    expert_up_scales: &[Option<&Tensor>],
    expert_down_scales: &[Option<&Tensor>],
    local_expert_indices: &[usize],
    n_routed_experts: i32,
    topk: i32,
    n_group: i32,
    topk_group: i32,
    scoring_func: i32,
    topk_method: i32,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
    ep_comm: *mut std::ffi::c_void,
    ep_rank: i32,
    ep_size: i32,
    device_id: i32,
) -> Result<Tensor> {
    fn opt_ptr(t: Option<&Tensor>) -> *mut std::ffi::c_void {
        match t {
            Some(t) if t.numel() > 0 => t.as_ptr() as *mut _,
            _ => std::ptr::null_mut(),
        }
    }
    let n_usize = expert_gate_weights.len();
    if expert_up_weights.len() != n_usize
        || expert_down_weights.len() != n_usize
        || expert_gate_scales.len() != n_usize
        || expert_up_scales.len() != n_usize
        || expert_down_scales.len() != n_usize
        || local_expert_indices.len() != n_usize
    {
        bail!("GLM5 MoE expert weight/scale/index slices must have equal length");
    }
    let n = i32::try_from(n_usize).context("too many local experts for GLM5 ABI")?;
    let mut gate_ptrs: Vec<*mut std::ffi::c_void> = expert_gate_weights
        .iter()
        .map(|t| t.as_ptr() as *mut _)
        .collect();
    let mut up_ptrs: Vec<*mut std::ffi::c_void> = expert_up_weights
        .iter()
        .map(|t| t.as_ptr() as *mut _)
        .collect();
    let mut down_ptrs: Vec<*mut std::ffi::c_void> = expert_down_weights
        .iter()
        .map(|t| t.as_ptr() as *mut _)
        .collect();
    let mut gs_ptrs: Vec<*mut std::ffi::c_void> =
        expert_gate_scales.iter().map(|t| opt_ptr(*t)).collect();
    let mut us_ptrs: Vec<*mut std::ffi::c_void> =
        expert_up_scales.iter().map(|t| opt_ptr(*t)).collect();
    let mut ds_ptrs: Vec<*mut std::ffi::c_void> =
        expert_down_scales.iter().map(|t| opt_ptr(*t)).collect();
    let indices: Vec<i32> = local_expert_indices
        .iter()
        .map(|&i| i32::try_from(i).context("local expert index exceeds GLM5 ABI"))
        .collect::<Result<_>>()?;

    let result_ptr = unsafe {
        v4_glm5_moe_layer(
            mlp_input.as_ptr() as *mut _,
            shared_gate.as_ptr() as *mut _,
            shared_up.as_ptr() as *mut _,
            shared_down.as_ptr() as *mut _,
            opt_ptr(shared_gate_scale),
            opt_ptr(shared_up_scale),
            opt_ptr(shared_down_scale),
            gate_weight.as_ptr() as *mut _,
            opt_ptr(correction_bias),
            gate_ptrs.as_mut_ptr(),
            up_ptrs.as_mut_ptr(),
            down_ptrs.as_mut_ptr(),
            gs_ptrs.as_mut_ptr(),
            us_ptrs.as_mut_ptr(),
            ds_ptrs.as_mut_ptr(),
            n,
            indices.as_ptr(),
            n_routed_experts,
            topk,
            n_group,
            topk_group,
            scoring_func,
            topk_method,
            i32::from(norm_topk_prob),
            routed_scaling_factor,
            ep_comm,
            ep_rank,
            ep_size,
            device_id,
        )
    };
    if result_ptr.is_null() {
        return Err(glm5_ffi_error("v4_glm5_moe_layer"));
    }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Single-rank compatibility wrapper for callers that do not need expert
/// dispatch. The distributed session uses [`glm5_moe_layer_ep_cpp`] directly
/// with the checkpoint's complete router configuration.
pub fn glm5_moe_layer_cpp(
    mlp_input: &Tensor,
    shared_gate: &Tensor,
    shared_up: &Tensor,
    shared_down: &Tensor,
    shared_gate_scale: Option<&Tensor>,
    shared_up_scale: Option<&Tensor>,
    shared_down_scale: Option<&Tensor>,
    gate_weight: &Tensor,
    expert_gate_weights: &[&Tensor],
    expert_up_weights: &[&Tensor],
    expert_down_weights: &[&Tensor],
    expert_gate_scales: &[Option<&Tensor>],
    expert_up_scales: &[Option<&Tensor>],
    expert_down_scales: &[Option<&Tensor>],
    local_expert_indices: &[usize],
    n_routed_experts: i32,
    topk: i32,
    routed_scaling_factor: f64,
    device_id: i32,
) -> Result<Tensor> {
    glm5_moe_layer_ep_cpp(
        mlp_input,
        shared_gate,
        shared_up,
        shared_down,
        shared_gate_scale,
        shared_up_scale,
        shared_down_scale,
        gate_weight,
        None,
        expert_gate_weights,
        expert_up_weights,
        expert_down_weights,
        expert_gate_scales,
        expert_up_scales,
        expert_down_scales,
        local_expert_indices,
        n_routed_experts,
        topk,
        1,
        1,
        0,
        0,
        true,
        routed_scaling_factor,
        std::ptr::null_mut(),
        0,
        1,
        device_id,
    )
}

/// Call the C++ embedding lookup kernel.
pub fn glm5_embedding_cpp(
    embed_weight: &Tensor,
    input_ids: &Tensor,
    device_id: i32,
) -> Result<Tensor> {
    let result_ptr = unsafe {
        v4_glm5_embedding(
            embed_weight.as_ptr() as *mut _,
            input_ids.as_ptr() as *mut _,
            device_id,
        )
    };
    if result_ptr.is_null() {
        return Err(glm5_ffi_error("v4_glm5_embedding"));
    }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Call the C++ full layer forward kernel (RMSNorm + attention + residual + RMSNorm + MoE + residual).
/// One FFI call per layer — all intermediates stay on C++ stack.
pub fn glm5_layer_forward_cpp(
    hidden: &Tensor,
    input_norm_weight: &Tensor,
    post_norm_weight: &Tensor,
    // Attention weights
    q_a_proj: &Tensor,
    q_a_layernorm: &Tensor,
    q_b_proj: &Tensor,
    kv_a_proj: &Tensor,
    kv_a_layernorm: &Tensor,
    kv_b_proj: &Tensor,
    o_proj: &Tensor,
    // FP8 scales
    q_a_scale: Option<&Tensor>,
    q_b_scale: Option<&Tensor>,
    kv_a_scale: Option<&Tensor>,
    kv_b_scale: Option<&Tensor>,
    o_scale: Option<&Tensor>,
    // Indexer
    idx_wq_b: Option<&Tensor>,
    idx_wk: Option<&Tensor>,
    idx_k_norm_w: Option<&Tensor>,
    idx_k_norm_b: Option<&Tensor>,
    idx_weights_proj: Option<&Tensor>,
    idx_weights_proj_scale: Option<&Tensor>,
    idx_wq_b_scale: Option<&Tensor>,
    idx_wk_scale: Option<&Tensor>,
    // MLP/MoE
    gate_weight: Option<&Tensor>,
    shared_gate: Option<&Tensor>,
    shared_up: Option<&Tensor>,
    shared_down: Option<&Tensor>,
    shared_gate_scale: Option<&Tensor>,
    shared_up_scale: Option<&Tensor>,
    shared_down_scale: Option<&Tensor>,
    dense_gate: Option<&Tensor>,
    dense_up: Option<&Tensor>,
    dense_down: Option<&Tensor>,
    dense_gate_scale: Option<&Tensor>,
    dense_up_scale: Option<&Tensor>,
    dense_down_scale: Option<&Tensor>,
    // Expert weights (CPU)
    expert_gate_weights: &[&Tensor],
    expert_up_weights: &[&Tensor],
    expert_down_weights: &[&Tensor],
    expert_gate_scales: &[Option<&Tensor>],
    expert_up_scales: &[Option<&Tensor>],
    expert_down_scales: &[Option<&Tensor>],
    local_expert_indices: &[usize],
    // Config
    batch: i32,
    seq: i32,
    num_heads: i32,
    qk_nope: i32,
    qk_rope: i32,
    v_head: i32,
    kv_lora: i32,
    idx_head_dim: i32,
    idx_n_heads: i32,
    idx_topk: i32,
    index_topk_freq: i32,
    layer: i32,
    is_full_layer: bool,
    is_moe_layer: bool,
    n_routed_experts: i32,
    topk: i32,
    rms_eps: f64,
    rope_theta: f64,
    rope_interleave: bool,
    routed_scaling_factor: f64,
    device_id: i32,
    // IndexShare state
    index_state: &mut Glm5IndexState,
) -> Result<Tensor> {
    fn opt_ptr(t: Option<&Tensor>) -> *mut std::ffi::c_void {
        match t {
            Some(t) if t.numel() > 0 => t.as_ptr() as *mut _,
            _ => std::ptr::null_mut(),
        }
    }
    let n_usize = expert_gate_weights.len();
    if expert_up_weights.len() != n_usize
        || expert_down_weights.len() != n_usize
        || expert_gate_scales.len() != n_usize
        || expert_up_scales.len() != n_usize
        || expert_down_scales.len() != n_usize
        || local_expert_indices.len() != n_usize
    {
        bail!("GLM5 layer expert weight/scale/index slices must have equal length");
    }
    let n = i32::try_from(n_usize).context("too many local experts for GLM5 ABI")?;
    let mut gate_ptrs: Vec<*mut std::ffi::c_void> = expert_gate_weights
        .iter()
        .map(|t| t.as_ptr() as *mut _)
        .collect();
    let mut up_ptrs: Vec<*mut std::ffi::c_void> = expert_up_weights
        .iter()
        .map(|t| t.as_ptr() as *mut _)
        .collect();
    let mut down_ptrs: Vec<*mut std::ffi::c_void> = expert_down_weights
        .iter()
        .map(|t| t.as_ptr() as *mut _)
        .collect();
    let mut gs_ptrs: Vec<*mut std::ffi::c_void> =
        expert_gate_scales.iter().map(|t| opt_ptr(*t)).collect();
    let mut us_ptrs: Vec<*mut std::ffi::c_void> =
        expert_up_scales.iter().map(|t| opt_ptr(*t)).collect();
    let mut ds_ptrs: Vec<*mut std::ffi::c_void> =
        expert_down_scales.iter().map(|t| opt_ptr(*t)).collect();
    let indices: Vec<i32> = local_expert_indices
        .iter()
        .map(|&i| i32::try_from(i).context("local expert index exceeds GLM5 ABI"))
        .collect::<Result<_>>()?;

    let result_ptr = unsafe {
        v4_glm5_layer_forward(
            hidden.as_ptr() as *mut _,
            input_norm_weight.as_ptr() as *mut _,
            post_norm_weight.as_ptr() as *mut _,
            q_a_proj.as_ptr() as *mut _,
            q_a_layernorm.as_ptr() as *mut _,
            q_b_proj.as_ptr() as *mut _,
            kv_a_proj.as_ptr() as *mut _,
            kv_a_layernorm.as_ptr() as *mut _,
            kv_b_proj.as_ptr() as *mut _,
            o_proj.as_ptr() as *mut _,
            opt_ptr(q_a_scale),
            opt_ptr(q_b_scale),
            opt_ptr(kv_a_scale),
            opt_ptr(kv_b_scale),
            opt_ptr(o_scale),
            opt_ptr(idx_wq_b),
            opt_ptr(idx_wk),
            opt_ptr(idx_k_norm_w),
            opt_ptr(idx_k_norm_b),
            opt_ptr(idx_weights_proj),
            opt_ptr(idx_weights_proj_scale),
            opt_ptr(idx_wq_b_scale),
            opt_ptr(idx_wk_scale),
            opt_ptr(gate_weight),
            opt_ptr(shared_gate),
            opt_ptr(shared_up),
            opt_ptr(shared_down),
            opt_ptr(shared_gate_scale),
            opt_ptr(shared_up_scale),
            opt_ptr(shared_down_scale),
            opt_ptr(dense_gate),
            opt_ptr(dense_up),
            opt_ptr(dense_down),
            opt_ptr(dense_gate_scale),
            opt_ptr(dense_up_scale),
            opt_ptr(dense_down_scale),
            gate_ptrs.as_mut_ptr(),
            up_ptrs.as_mut_ptr(),
            down_ptrs.as_mut_ptr(),
            gs_ptrs.as_mut_ptr(),
            us_ptrs.as_mut_ptr(),
            ds_ptrs.as_mut_ptr(),
            n,
            indices.as_ptr(),
            batch,
            seq,
            num_heads,
            qk_nope,
            qk_rope,
            v_head,
            kv_lora,
            idx_head_dim,
            idx_n_heads,
            idx_topk,
            index_topk_freq,
            layer,
            if is_full_layer { 1 } else { 0 },
            if is_moe_layer { 1 } else { 0 },
            n_routed_experts,
            topk,
            rms_eps,
            rope_theta,
            if rope_interleave { 1 } else { 0 },
            routed_scaling_factor,
            device_id,
            &mut index_state.topk_indices,
            &mut index_state.idx_bias_keys,
            &mut index_state.source_layer,
        )
    };
    if result_ptr.is_null() {
        return Err(glm5_ffi_error("v4_glm5_layer_forward"));
    }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Execute one complete native GLM5 MTP decoder layer in one Rust -> C++ FFI
/// call. The descriptor owns no tensors; all pointers are borrowed for this
/// call. TP/EP use SUM-forward/identity-backward collectives and CP uses the
/// autograd-aware K/V ring implemented in the C++ kernel.
pub fn glm5_mtp_decoder_layer_cpp(descriptor: &Glm5MtpDecoderDescriptor) -> Result<Tensor> {
    let result_ptr = unsafe { v4_glm5_mtp_decoder_layer(descriptor) };
    if result_ptr.is_null() {
        return Err(glm5_ffi_error("v4_glm5_mtp_decoder_layer"));
    }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Make PyTorch's current CUDA stream wait for a CUDA event.
/// This is GPU-side dependency — CPU is NOT blocked.
/// Used for async pipeline: after all_reduce_async returns (output, event),
/// call this before using the output tensor in the next compute.
pub fn stream_wait_event(device_id: i32, event: &rustrain_nccl::nccl::CudaEventHandle) {
    unsafe { v4_stream_wait_event(device_id, event.0) };
}

#[cfg(test)]
mod mtp_tests {
    use super::*;
    use tch::{Cuda, Device, Kind};

    #[test]
    fn checkpoint_result_preserves_gradients_and_reports_forward_errors() {
        let input = Tensor::from_slice(&[1.0_f32, -2.0, 3.0]).set_requires_grad(true);
        let output = checkpoint_result(&input, |value| Ok(value * value))
            .expect("fallible checkpoint forward should succeed");
        output.sum(Kind::Float).backward();
        let expected = Tensor::from_slice(&[2.0_f32, -4.0, 6.0]);
        assert!(input.grad().allclose(&expected, 1e-6, 1e-6, false));
        clear_checkpoint_registry();

        let error_input = Tensor::ones([1], (Kind::Float, Device::Cpu));
        let error = checkpoint_result(&error_input, |_| {
            Err(anyhow::anyhow!("synthetic checkpoint failure"))
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("synthetic checkpoint failure"));
        clear_checkpoint_registry();
    }

    #[test]
    fn mtp_descriptor_places_weights_proj_scale_next_to_weight() {
        let pointer_size = std::mem::size_of::<*mut std::ffi::c_void>();
        assert_eq!(
            std::mem::offset_of!(Glm5MtpDecoderDescriptor, idx_weights_proj_scale),
            std::mem::offset_of!(Glm5MtpDecoderDescriptor, idx_weights_proj) + pointer_size
        );
        assert_eq!(
            std::mem::offset_of!(Glm5MtpDecoderDescriptor, idx_wq_b_scale),
            std::mem::offset_of!(Glm5MtpDecoderDescriptor, idx_weights_proj_scale) + pointer_size
        );

        let descriptor = Glm5MtpDecoderDescriptor::default();
        assert!(descriptor.idx_weights_proj_scale.is_null());
    }

    fn assert_finite_nonzero_grad(name: &str, tensor: &Tensor) {
        let grad = tensor.grad();
        assert!(grad.defined(), "{name} gradient is undefined");
        assert_eq!(
            grad.isfinite().all().int64_value(&[]),
            1,
            "{name} gradient contains non-finite values"
        );
        assert!(
            grad.abs().sum(Kind::Float).double_value(&[]) > 0.0,
            "{name} gradient is identically zero"
        );
    }

    #[test]
    fn mtp_prepare_aligns_next_token_embedding() {
        let hidden = Tensor::zeros([1, 4, 4], (Kind::Float, Device::Cpu));
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let embed = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let norm = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let eh_proj = Tensor::cat(
            &[
                &Tensor::eye(4, (Kind::Float, Device::Cpu)),
                &Tensor::zeros([4, 4], (Kind::Float, Device::Cpu)),
            ],
            1,
        );

        let projected = glm5_mtp_prepare_cpp(
            &hidden, &input_ids, &embed, &norm, &norm, &eh_proj, None, 1e-6, 1,
        )
        .unwrap();

        assert_eq!(projected.size(), vec![1, 2, 4]);
        let selected: Vec<i64> =
            Vec::<i64>::try_from(&projected.argmax(-1, false).reshape([-1])).unwrap();
        assert_eq!(selected, vec![1, 2]);
    }

    #[test]
    fn mtp_prepare_preserves_autograd_inputs() {
        let hidden = Tensor::zeros([1, 4, 4], (Kind::Float, Device::Cpu));
        let _ = hidden.set_requires_grad(true);
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let embed = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let norm = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let mut eh_proj = Tensor::eye(4, (Kind::Float, Device::Cpu));
        eh_proj = Tensor::cat(
            &[&eh_proj, &Tensor::zeros([4, 4], (Kind::Float, Device::Cpu))],
            1,
        );
        let _ = eh_proj.set_requires_grad(true);

        let projected = glm5_mtp_prepare_cpp(
            &hidden, &input_ids, &embed, &norm, &norm, &eh_proj, None, 1e-6, 1,
        )
        .unwrap();
        projected.sum(Kind::Float).backward();

        assert!(hidden.grad().defined(), "hidden gradient was not retained");
        assert!(
            eh_proj.grad().defined(),
            "eh_proj gradient was not retained"
        );
    }

    #[test]
    fn mtp_prepare_supports_deeper_token_offsets() {
        let hidden = Tensor::zeros([1, 3, 4], (Kind::Float, Device::Cpu));
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3, 0]).reshape([1, 5]);
        let embed = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let norm = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let eh_proj = Tensor::cat(
            &[
                &Tensor::eye(4, (Kind::Float, Device::Cpu)),
                &Tensor::zeros([4, 4], (Kind::Float, Device::Cpu)),
            ],
            1,
        );

        let projected = glm5_mtp_prepare_cpp(
            &hidden, &input_ids, &embed, &norm, &norm, &eh_proj, None, 1e-6, 2,
        )
        .unwrap();

        assert_eq!(projected.size(), vec![1, 2, 4]);
        let selected: Vec<i64> =
            Vec::<i64>::try_from(&projected.argmax(-1, false).reshape([-1])).unwrap();
        assert_eq!(selected, vec![2, 3]);
    }

    #[test]
    fn mtp_loss_uses_next_next_token_targets() {
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let mask = Tensor::ones([1, 4], (Kind::Float, Device::Cpu));
        let norm = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let lm_head = Tensor::eye(4, (Kind::Float, Device::Cpu)) * 10.0;
        let aligned = Tensor::from_slice(&[
            0.0_f32, 0.0, 1.0, 0.0, // predicts token 2
            0.0, 0.0, 0.0, 1.0, // predicts token 3
        ])
        .reshape([1, 2, 4]);
        let misaligned =
            Tensor::from_slice(&[0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape([1, 2, 4]);

        let aligned_loss = glm5_mtp_cross_entropy_loss_cpp(
            &aligned, &norm, &lm_head, None, &input_ids, &mask, 1e-6, 0, 16,
        )
        .unwrap()
        .double_value(&[]);
        let misaligned_loss = glm5_mtp_cross_entropy_loss_cpp(
            &misaligned,
            &norm,
            &lm_head,
            None,
            &input_ids,
            &mask,
            1e-6,
            0,
            16,
        )
        .unwrap()
        .double_value(&[]);

        assert!(aligned_loss < 0.01, "aligned loss was {aligned_loss}");
        assert!(
            misaligned_loss > 5.0,
            "misaligned loss was {misaligned_loss}"
        );
    }

    #[test]
    fn two_layer_mtp_chain_preserves_offsets_and_autograd() {
        let hidden = Tensor::randn([1, 5, 4], (Kind::Float, Device::Cpu));
        let _ = hidden.set_requires_grad(true);
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3, 0]).reshape([1, 5]);
        let mask = Tensor::ones([1, 5], (Kind::Float, Device::Cpu));
        let embed = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let norm = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let lm_head = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let projection = Tensor::cat(
            &[
                &Tensor::zeros([4, 4], (Kind::Float, Device::Cpu)),
                &Tensor::eye(4, (Kind::Float, Device::Cpu)),
            ],
            1,
        );
        let _ = projection.set_requires_grad(true);

        let layer0 = glm5_mtp_prepare_cpp(
            &hidden,
            &input_ids,
            &embed,
            &norm,
            &norm,
            &projection,
            None,
            1e-6,
            1,
        )
        .unwrap();
        let output0 = glm5_mtp_postprocess_loss_cpp(
            &layer0, &norm, &lm_head, None, &input_ids, &mask, 1e-6, 0, 16,
        )
        .unwrap();
        assert_eq!(output0.normalized.size(), vec![1, 3, 4]);

        let layer1 = glm5_mtp_prepare_cpp(
            &output0.normalized,
            &input_ids,
            &embed,
            &norm,
            &norm,
            &projection,
            None,
            1e-6,
            2,
        )
        .unwrap();
        let output1 = glm5_mtp_postprocess_loss_cpp(
            &layer1, &norm, &lm_head, None, &input_ids, &mask, 1e-6, 1, 16,
        )
        .unwrap();
        assert_eq!(output1.normalized.size(), vec![1, 2, 4]);
        assert_eq!(output0.token_count.double_value(&[]), 3.0);
        assert_eq!(output1.token_count.double_value(&[]), 2.0);

        let lm_loss = Tensor::from(1.0_f32);
        let combined =
            glm5_combine_losses_cpp(&lm_loss, &[output0.loss, output1.loss], 0.5).unwrap();
        assert!(
            (combined.total.double_value(&[]) - (1.0 + 0.5 * combined.mtp_mean.double_value(&[])))
                .abs()
                < 1e-6
        );
        combined.total.backward();
        assert!(hidden.grad().defined(), "trunk hidden gradient is missing");
        assert!(
            projection.grad().defined(),
            "MTP projection gradient is missing"
        );
    }

    #[test]
    fn kv_ring_rejects_non_cuda_inputs_without_returning_partial_outputs() {
        let key = Tensor::zeros([1, 2, 3, 5], (Kind::Float, Device::Cpu));
        let value = Tensor::zeros([1, 2, 3, 7], (Kind::Float, Device::Cpu));
        let result =
            glm5_nccl_kv_ring_autograd_cpp(&key, &value, 1_usize as *mut std::ffi::c_void, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn mtp_decoder_local_dense_forward_and_autograd() {
        let hidden = Tensor::randn([1, 2, 4], (Kind::Float, Device::Cpu));
        let _ = hidden.set_requires_grad(true);
        let norm = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let q_a = Tensor::randn([4, 4], (Kind::Float, Device::Cpu));
        let q_b = Tensor::randn([4, 4], (Kind::Float, Device::Cpu));
        let kv_a = Tensor::randn([4, 4], (Kind::Float, Device::Cpu));
        let kv_b = Tensor::randn([4, 2], (Kind::Float, Device::Cpu));
        let q_ln = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let kv_ln = Tensor::ones([2], (Kind::Float, Device::Cpu));
        let o_proj = Tensor::randn([4, 2], (Kind::Float, Device::Cpu));
        let idx_wq_b = Tensor::randn([4, 4], (Kind::Float, Device::Cpu));
        let idx_wk = Tensor::randn([4, 4], (Kind::Float, Device::Cpu));
        let idx_k_norm_w = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let idx_k_norm_b = Tensor::zeros([4], (Kind::Float, Device::Cpu));
        let idx_weights_proj = Tensor::randn([1, 4], (Kind::Float, Device::Cpu));
        let dense_gate = Tensor::randn([8, 4], (Kind::Float, Device::Cpu));
        let dense_up = Tensor::randn([8, 4], (Kind::Float, Device::Cpu));
        let dense_down = Tensor::randn([4, 8], (Kind::Float, Device::Cpu));
        let mut d = Glm5MtpDecoderDescriptor::default();
        let ptr = |t: &Tensor| t.as_ptr() as *mut std::ffi::c_void;
        d.hidden = ptr(&hidden);
        d.input_norm_weight = ptr(&norm);
        d.post_norm_weight = ptr(&norm);
        d.q_a_proj = ptr(&q_a);
        d.q_a_layernorm = ptr(&q_ln);
        d.q_b_proj = ptr(&q_b);
        d.kv_a_proj = ptr(&kv_a);
        d.kv_a_layernorm = ptr(&kv_ln);
        d.kv_b_proj = ptr(&kv_b);
        d.o_proj = ptr(&o_proj);
        d.idx_wq_b = ptr(&idx_wq_b);
        d.idx_wk = ptr(&idx_wk);
        d.idx_k_norm_w = ptr(&idx_k_norm_w);
        d.idx_k_norm_b = ptr(&idx_k_norm_b);
        d.idx_weights_proj = ptr(&idx_weights_proj);
        d.dense_gate = ptr(&dense_gate);
        d.dense_up = ptr(&dense_up);
        d.dense_down = ptr(&dense_down);
        d.tp_size = 1;
        d.cp_size = 1;
        d.ep_size = 1;
        d.num_heads = 1;
        d.qk_nope = 2;
        d.qk_rope = 2;
        d.v_head = 2;
        d.kv_lora = 2;
        d.idx_head_dim = 4;
        d.idx_n_heads = 1;
        d.idx_n_heads_global = 1;
        d.idx_topk = 2;
        d.rms_eps = 1e-6;
        d.rope_theta = 10_000.0;
        let output = glm5_mtp_decoder_layer_cpp(&d).unwrap();
        assert_eq!(output.size(), vec![1, 2, 4]);
        output.sum(Kind::Float).backward();
        assert!(
            hidden.grad().defined(),
            "MTP decoder detached hidden autograd"
        );
    }

    #[test]
    fn mtp_cuda_single_gpu_ffi_chain_forward_backward() {
        if !Cuda::is_available() {
            eprintln!("skipping CUDA MTP FFI chain test: CUDA is unavailable");
            return;
        }

        let device = Device::Cuda(0);
        let hidden_size = 4;
        let intermediate_size = 8;
        let vocab_size = 8;
        let sequence_len = 4;
        let raw_sequence_len = 6;

        let hidden = Tensor::randn([1, sequence_len, hidden_size], (Kind::Float, device)) * 0.1;
        let _ = hidden.set_requires_grad(true);
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3, 4, 5])
            .reshape([1, raw_sequence_len])
            .to_device(device);
        let target_mask = Tensor::ones([1, raw_sequence_len], (Kind::Float, device));
        let embed = Tensor::randn([vocab_size, hidden_size], (Kind::Float, device)) * 0.1;
        let enorm = Tensor::ones([hidden_size], (Kind::Float, device));
        let hnorm = Tensor::ones([hidden_size], (Kind::Float, device));
        let eh_proj = Tensor::randn([hidden_size, 2 * hidden_size], (Kind::Float, device)) * 0.1;
        let _ = eh_proj.set_requires_grad(true);

        let prepared = glm5_mtp_prepare_cpp(
            &hidden, &input_ids, &embed, &enorm, &hnorm, &eh_proj, None, 1e-6, 1,
        )
        .expect("CUDA MTP prepare FFI failed");
        assert_eq!(prepared.size(), vec![1, sequence_len, hidden_size]);
        assert_eq!(prepared.device(), device);
        assert_eq!(prepared.isfinite().all().int64_value(&[]), 1);

        let input_norm = Tensor::ones([hidden_size], (Kind::Float, device));
        let post_norm = Tensor::ones([hidden_size], (Kind::Float, device));
        let q_a = Tensor::randn([hidden_size, hidden_size], (Kind::Float, device)) * 0.1;
        let _ = q_a.set_requires_grad(true);
        let q_b = Tensor::randn([hidden_size, hidden_size], (Kind::Float, device)) * 0.1;
        let kv_a = Tensor::randn([hidden_size, hidden_size], (Kind::Float, device)) * 0.1;
        let kv_b = Tensor::randn([hidden_size, 2], (Kind::Float, device)) * 0.1;
        let q_layernorm = Tensor::ones([hidden_size], (Kind::Float, device));
        let kv_layernorm = Tensor::ones([2], (Kind::Float, device));
        let o_proj = Tensor::randn([hidden_size, 2], (Kind::Float, device)) * 0.1;
        let idx_wq_b = Tensor::randn([hidden_size, hidden_size], (Kind::Float, device)) * 0.1;
        let idx_wk = Tensor::randn([hidden_size, hidden_size], (Kind::Float, device)) * 0.1;
        let idx_k_norm_w = Tensor::ones([hidden_size], (Kind::Float, device));
        let idx_k_norm_b = Tensor::zeros([hidden_size], (Kind::Float, device));
        let idx_weights_proj = Tensor::randn([1, hidden_size], (Kind::Float, device)) * 0.1;
        let dense_gate =
            Tensor::randn([intermediate_size, hidden_size], (Kind::Float, device)) * 0.1;
        let dense_up = Tensor::randn([intermediate_size, hidden_size], (Kind::Float, device)) * 0.1;
        let dense_down =
            Tensor::randn([hidden_size, intermediate_size], (Kind::Float, device)) * 0.1;
        let _ = dense_down.set_requires_grad(true);

        let ptr = |tensor: &Tensor| tensor.as_ptr() as *mut std::ffi::c_void;
        let mut descriptor = Glm5MtpDecoderDescriptor::default();
        descriptor.hidden = ptr(&prepared);
        descriptor.input_norm_weight = ptr(&input_norm);
        descriptor.post_norm_weight = ptr(&post_norm);
        descriptor.q_a_proj = ptr(&q_a);
        descriptor.q_a_layernorm = ptr(&q_layernorm);
        descriptor.q_b_proj = ptr(&q_b);
        descriptor.kv_a_proj = ptr(&kv_a);
        descriptor.kv_a_layernorm = ptr(&kv_layernorm);
        descriptor.kv_b_proj = ptr(&kv_b);
        descriptor.o_proj = ptr(&o_proj);
        descriptor.idx_wq_b = ptr(&idx_wq_b);
        descriptor.idx_wk = ptr(&idx_wk);
        descriptor.idx_k_norm_w = ptr(&idx_k_norm_w);
        descriptor.idx_k_norm_b = ptr(&idx_k_norm_b);
        descriptor.idx_weights_proj = ptr(&idx_weights_proj);
        descriptor.dense_gate = ptr(&dense_gate);
        descriptor.dense_up = ptr(&dense_up);
        descriptor.dense_down = ptr(&dense_down);
        descriptor.tp_size = 1;
        descriptor.cp_size = 1;
        descriptor.ep_size = 1;
        descriptor.num_heads = 1;
        descriptor.qk_nope = 2;
        descriptor.qk_rope = 2;
        descriptor.v_head = 2;
        descriptor.kv_lora = 2;
        descriptor.idx_head_dim = hidden_size as i32;
        descriptor.idx_n_heads = 1;
        descriptor.idx_n_heads_global = 1;
        descriptor.idx_topk = 2;
        descriptor.rms_eps = 1e-6;
        descriptor.rope_theta = 10_000.0;

        let decoded = glm5_mtp_decoder_layer_cpp(&descriptor)
            .expect("CUDA local dense MTP decoder FFI failed");
        assert_eq!(decoded.size(), vec![1, sequence_len, hidden_size]);
        assert_eq!(decoded.device(), device);
        assert_eq!(decoded.isfinite().all().int64_value(&[]), 1);

        let shared_head_norm = Tensor::ones([hidden_size], (Kind::Float, device));
        let lm_head = Tensor::randn([vocab_size, hidden_size], (Kind::Float, device)) * 0.1;
        let _ = lm_head.set_requires_grad(true);
        let output = glm5_mtp_postprocess_loss_cpp(
            &decoded,
            &shared_head_norm,
            &lm_head,
            None,
            &input_ids,
            &target_mask,
            1e-6,
            0,
            2,
        )
        .expect("CUDA MTP postprocess/CE FFI failed");
        assert_eq!(output.normalized.size(), vec![1, sequence_len, hidden_size]);
        assert_eq!(output.loss.size(), Vec::<i64>::new());
        assert_eq!(output.loss_sum.size(), Vec::<i64>::new());
        assert_eq!(output.token_count.size(), Vec::<i64>::new());
        assert_eq!(output.token_count.double_value(&[]), sequence_len as f64);
        for (name, tensor) in [
            ("normalized", &output.normalized),
            ("loss", &output.loss),
            ("loss_sum", &output.loss_sum),
        ] {
            assert_eq!(
                tensor.isfinite().all().int64_value(&[]),
                1,
                "{name} contains non-finite values"
            );
        }

        output.loss.backward();
        assert_finite_nonzero_grad("trunk hidden", &hidden);
        assert_finite_nonzero_grad("MTP eh_proj", &eh_proj);
        assert_finite_nonzero_grad("decoder q_a_proj", &q_a);
        assert_finite_nonzero_grad("decoder dense_down", &dense_down);
        assert_finite_nonzero_grad("shared lm_head", &lm_head);
    }

    #[test]
    fn mtp_decoder_distributed_requires_cuda_and_comm() {
        let hidden = Tensor::zeros([1, 1, 4], (Kind::Float, Device::Cpu));
        let norm = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let q_a = Tensor::randn([4, 4], (Kind::Float, Device::Cpu));
        let q_b = Tensor::randn([4, 4], (Kind::Float, Device::Cpu));
        let kv_a = Tensor::randn([4, 4], (Kind::Float, Device::Cpu));
        let kv_b = Tensor::randn([4, 2], (Kind::Float, Device::Cpu));
        let q_ln = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let kv_ln = Tensor::ones([2], (Kind::Float, Device::Cpu));
        let o_proj = Tensor::randn([4, 2], (Kind::Float, Device::Cpu));
        let idx_bias = Tensor::zeros([4], (Kind::Float, Device::Cpu));
        let idx_weights_proj = Tensor::randn([1, 4], (Kind::Float, Device::Cpu));
        let dense_gate = Tensor::randn([8, 4], (Kind::Float, Device::Cpu));
        let dense_up = Tensor::randn([8, 4], (Kind::Float, Device::Cpu));
        let dense_down = Tensor::randn([4, 8], (Kind::Float, Device::Cpu));
        let mut d = Glm5MtpDecoderDescriptor::default();
        let ptr = |t: &Tensor| t.as_ptr() as *mut std::ffi::c_void;
        d.hidden = ptr(&hidden);
        d.input_norm_weight = ptr(&norm);
        d.post_norm_weight = ptr(&norm);
        d.q_a_proj = ptr(&q_a);
        d.q_a_layernorm = ptr(&q_ln);
        d.q_b_proj = ptr(&q_b);
        d.kv_a_proj = ptr(&kv_a);
        d.kv_a_layernorm = ptr(&kv_ln);
        d.kv_b_proj = ptr(&kv_b);
        d.o_proj = ptr(&o_proj);
        d.idx_wq_b = ptr(&q_a);
        d.idx_wk = ptr(&q_a);
        d.idx_k_norm_w = ptr(&norm);
        d.idx_k_norm_b = ptr(&idx_bias);
        d.idx_weights_proj = ptr(&idx_weights_proj);
        d.dense_gate = ptr(&dense_gate);
        d.dense_up = ptr(&dense_up);
        d.dense_down = ptr(&dense_down);
        d.tp_comm = 1_usize as *mut std::ffi::c_void;
        d.tp_size = 2;
        d.cp_size = 1;
        d.ep_size = 1;
        d.num_heads = 1;
        d.qk_nope = 2;
        d.qk_rope = 2;
        d.v_head = 2;
        d.kv_lora = 2;
        d.idx_head_dim = 4;
        d.idx_n_heads = 1;
        d.idx_n_heads_global = 1;
        d.idx_topk = 1;
        d.rms_eps = 1e-6;
        d.rope_theta = 10_000.0;
        let error = glm5_mtp_decoder_layer_cpp(&d).unwrap_err().to_string();
        assert!(error.contains("TP collective requires CUDA input"));
    }

    #[test]
    fn cross_entropy_cpu_supports_batch_and_chunking() {
        let hidden = Tensor::arange(24, (Kind::Float, Device::Cpu))
            .reshape([2, 4, 3])
            .set_requires_grad(true);
        let lm_head = Tensor::arange(15, (Kind::Float, Device::Cpu)).reshape([5, 3]);
        let targets = Tensor::from_slice(&[0_i64, 1, 2, 3, 4, 3, 2, 1]).reshape([2, 4]);
        let mask =
            Tensor::from_slice(&[1.0_f32, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).reshape([2, 4]);
        let loss = glm5_cross_entropy_loss_cpp(&hidden, &lm_head, &targets, &mask, 4, 5, 1, 0)
            .expect("CPU CE should support batch > 1");
        let shifted_hidden = hidden.narrow(1, 0, 3).reshape([-1, 3]);
        let shifted_targets = targets.narrow(1, 1, 3).reshape([-1]);
        let shifted_mask = mask.narrow(1, 1, 3).reshape([-1]);
        let reference = shifted_hidden
            .linear::<&Tensor>(&lm_head, None)
            .log_softmax(-1, Kind::Float)
            .g_nll_loss::<&Tensor>(&shifted_targets, None, tch::Reduction::None, -100);
        let reference = (reference * &shifted_mask).sum(Kind::Float)
            / shifted_mask.sum(Kind::Float).clamp_min(1.0);
        assert!((loss.double_value(&[]) - reference.double_value(&[])).abs() < 1e-5);
        loss.backward();
        assert!(hidden.grad().defined());
    }
}
