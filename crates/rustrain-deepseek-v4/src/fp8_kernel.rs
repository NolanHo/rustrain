//! FP8 block-wise GEMM + safetensors loading via C++ FFI (no Python).
//!
//! Two C++ functions:
//! 1. `v4_fp8_scaled_mm` — block-wise FP8 GEMM via at::_scaled_mm (CUTLASS)
//! 2. `v4_create_tensor` — create at::Tensor from raw bytes (FP8 support)
//!
//! Weight loading: Rust parses safetensors header, C++ creates tensors from raw data.

use std::collections::{BTreeMap, HashSet};

use anyhow::{bail, Context, Result};
use tch::{Kind, Tensor};
use tracing::info;

unsafe extern "C" {
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
        idx_wq_b_scale: *mut std::ffi::c_void,
        idx_wk_scale: *mut std::ffi::c_void,
        // Config
        batch_i: std::ffi::c_int,
        seq_i: std::ffi::c_int,
        num_heads_i: std::ffi::c_int,
        qk_nope_i: std::ffi::c_int,
        qk_rope_i: std::ffi::c_int,
        v_head_i: std::ffi::c_int,
        kv_lora_i: std::ffi::c_int,
        idx_head_dim_i: std::ffi::c_int,
        idx_n_heads_i: std::ffi::c_int,
        idx_topk_i: std::ffi::c_int,
        layer_i: std::ffi::c_int,
        is_full_layer: std::ffi::c_int,
        rms_eps: f64,
        rope_theta: f64,
        rope_interleave: std::ffi::c_int,
        device_id: std::ffi::c_int,
        // IndexShare state (in/out)
        topk_indices_ptr: *mut *mut std::ffi::c_void,
        idx_bias_keys_ptr: *mut *mut std::ffi::c_void,
        source_layer: *mut std::ffi::c_int,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_free_at_tensor(tensor_ptr: *mut std::ffi::c_void);

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
        seq_len: std::ffi::c_int,
        vocab: std::ffi::c_int,
        chunk_size: std::ffi::c_int,
        device_id: std::ffi::c_int,
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
        expert_gate_weights: *mut *mut std::ffi::c_void,
        expert_up_weights: *mut *mut std::ffi::c_void,
        expert_down_weights: *mut *mut std::ffi::c_void,
        expert_gate_scales: *mut *mut std::ffi::c_void,
        expert_up_scales: *mut *mut std::ffi::c_void,
        expert_down_scales: *mut *mut std::ffi::c_void,
        n_local_experts: std::ffi::c_int,
        local_expert_indices: *const std::ffi::c_int,
        n_routed_experts: std::ffi::c_int,
        topk: std::ffi::c_int,
        routed_scaling_factor: f64,
        device_id: std::ffi::c_int,
    ) -> *mut std::ffi::c_void;

    fn v4_glm5_embedding(
        embed_weight_ptr: *mut std::ffi::c_void,
        input_ids_ptr: *mut std::ffi::c_void,
        device_id: std::ffi::c_int,
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
        n_local_experts: std::ffi::c_int,
        local_expert_indices: *const std::ffi::c_int,
        // Config
        batch: std::ffi::c_int,
        seq: std::ffi::c_int,
        num_heads: std::ffi::c_int,
        qk_nope: std::ffi::c_int,
        qk_rope: std::ffi::c_int,
        v_head: std::ffi::c_int,
        kv_lora: std::ffi::c_int,
        idx_head_dim: std::ffi::c_int,
        idx_n_heads: std::ffi::c_int,
        idx_topk: std::ffi::c_int,
        layer: std::ffi::c_int,
        is_full_layer: std::ffi::c_int,
        is_moe_layer: std::ffi::c_int,
        n_routed_experts: std::ffi::c_int,
        topk: std::ffi::c_int,
        rms_eps: f64,
        rope_theta: f64,
        rope_interleave: std::ffi::c_int,
        routed_scaling_factor: f64,
        device_id: std::ffi::c_int,
        topk_indices_ptr: *mut *mut std::ffi::c_void,
        idx_bias_keys_ptr: *mut *mut std::ffi::c_void,
        source_layer: *mut std::ffi::c_int,
    ) -> *mut std::ffi::c_void;

    fn v4_stream_wait_event(device_id: std::ffi::c_int, event_ptr: *mut std::ffi::c_void);
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
        let n_padded = n_blocks * 128;
        let k_padded = k_blocks * 128;
        let expanded = scale
            .unsqueeze(-1)                         // [n_blocks, k_blocks, 1]
            .unsqueeze(-1)                         // [n_blocks, k_blocks, 1, 1]
            .expand([n_blocks, k_blocks, 128, 128], false)
            .reshape([n_padded, k_padded])
            .contiguous();
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
use std::sync::{Mutex, OnceLock};

type ForwardFn = Box<dyn Fn(&Tensor) -> Tensor + Send>;

static CHECKPOINT_REGISTRY: OnceLock<Mutex<HashMap<usize, ForwardFn>>> = OnceLock::new();
static CHECKPOINT_COUNTER: AtomicUsize = AtomicUsize::new(1);

fn registry() -> &'static Mutex<HashMap<usize, ForwardFn>> {
    CHECKPOINT_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
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
    let output = {
        let reg = registry().lock().unwrap();
        match reg.get(&key) {
            Some(forward) => forward(&input),
            None => {
                eprintln!("[checkpoint_callback] forward function not found for key {key}");
                return std::ptr::null_mut();
            }
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
    let key = CHECKPOINT_COUNTER.fetch_add(1, Ordering::SeqCst);
    registry().lock().unwrap().insert(key, Box::new(forward));

    let result_ptr = unsafe {
        v4_checkpoint(
            checkpoint_callback as *mut std::ffi::c_void,
            input.as_ptr() as *mut std::ffi::c_void,
            key as *mut std::ffi::c_void,
        )
    };

    if result_ptr.is_null() {
        registry().lock().unwrap().remove(&key);
        panic!("C++ v4_checkpoint returned null");
    }

    let result = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_fp8_free_tensor(result_ptr) };

    // Registry entry is kept for backward pass.
    // It will be cleaned up via clear_checkpoint_registry() after each training step.

    result
}

/// Clear all checkpoint registry entries. Call after each training step
/// to prevent GPU memory leak (closures hold tensor references).
pub fn clear_checkpoint_registry() {
    registry().lock().unwrap().clear();
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
#[derive(Default)]
pub struct Glm5IndexState {
    pub topk_indices: *mut std::ffi::c_void,  // at::Tensor* or null
    pub idx_bias_keys: *mut std::ffi::c_void, // at::Tensor* or null
    pub source_layer: i32,
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
    q_a_proj: &Tensor, q_a_layernorm: &Tensor, q_b_proj: &Tensor,
    kv_a_proj: &Tensor, kv_a_layernorm: &Tensor, kv_b_proj: &Tensor,
    o_proj: &Tensor,
    // FP8 scales (optional — pass empty tensor if not used)
    q_a_scale: Option<&Tensor>, q_b_scale: Option<&Tensor>,
    kv_a_scale: Option<&Tensor>, kv_b_scale: Option<&Tensor>, o_scale: Option<&Tensor>,
    // Indexer weights (optional)
    idx_wq_b: Option<&Tensor>, idx_wk: Option<&Tensor>,
    idx_k_norm_w: Option<&Tensor>, idx_k_norm_b: Option<&Tensor>,
    idx_weights_proj: Option<&Tensor>,
    idx_wq_b_scale: Option<&Tensor>, idx_wk_scale: Option<&Tensor>,
    // Config
    batch: i32, seq: i32, num_heads: i32, qk_nope: i32, qk_rope: i32,
    v_head: i32, kv_lora: i32, idx_head_dim: i32, idx_n_heads: i32,
    idx_topk: i32, layer: i32, is_full_layer: bool,
    rms_eps: f64, rope_theta: f64, rope_interleave: bool,
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
            opt_ptr(q_a_scale), opt_ptr(q_b_scale),
            opt_ptr(kv_a_scale), opt_ptr(kv_b_scale), opt_ptr(o_scale),
            opt_ptr(idx_wq_b), opt_ptr(idx_wk),
            opt_ptr(idx_k_norm_w), opt_ptr(idx_k_norm_b),
            opt_ptr(idx_weights_proj),
            opt_ptr(idx_wq_b_scale), opt_ptr(idx_wk_scale),
            batch, seq, num_heads, qk_nope, qk_rope,
            v_head, kv_lora, idx_head_dim, idx_n_heads,
            idx_topk, layer, if is_full_layer { 1 } else { 0 },
            rms_eps, rope_theta, if rope_interleave { 1 } else { 0 },
            device_id,
            &mut index_state.topk_indices,
            &mut index_state.idx_bias_keys,
            &mut index_state.source_layer,
        )
    };

    if result_ptr.is_null() {
        bail!("C++ v4_glm5_dsa_attention returned null");
    }

    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Call the C++ GLM5 MLP kernel (SwiGLU: silu(gate(x)) * up(x) → down).
pub fn glm5_mlp_fp8_cpp(
    input: &Tensor,
    gate: &Tensor, up: &Tensor, down: &Tensor,
    gate_scale: Option<&Tensor>, up_scale: Option<&Tensor>, down_scale: Option<&Tensor>,
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
            opt_ptr(gate_scale), opt_ptr(up_scale), opt_ptr(down_scale),
        )
    };
    if result_ptr.is_null() { bail!("C++ v4_glm5_mlp_fp8 returned null"); }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Call the C++ RMSNorm kernel.
pub fn glm5_rms_norm_cpp(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let result_ptr = unsafe {
        v4_glm5_rms_norm(input.as_ptr() as *mut _, weight.as_ptr() as *mut _, eps)
    };
    if result_ptr.is_null() { bail!("C++ v4_glm5_rms_norm returned null"); }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Call the C++ chunked cross-entropy loss kernel.
pub fn glm5_cross_entropy_loss_cpp(
    hidden: &Tensor, lm_head: &Tensor, targets: &Tensor, mask: &Tensor,
    seq_len: i32, vocab: i32, chunk_size: i32, device_id: i32,
) -> Result<Tensor> {
    let result_ptr = unsafe {
        v4_glm5_cross_entropy_loss(
            hidden.as_ptr() as *mut _, lm_head.as_ptr() as *mut _,
            targets.as_ptr() as *mut _, mask.as_ptr() as *mut _,
            seq_len, vocab, chunk_size, device_id,
        )
    };
    if result_ptr.is_null() { bail!("C++ v4_glm5_cross_entropy_loss returned null"); }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Call the C++ Adam optimizer step. Updates params, m, v in-place.
pub fn adam_step_cpp(
    params: &mut [Tensor], grads: &[Tensor], m: &mut [Tensor], v: &mut [Tensor],
    lr: f64, beta1: f64, beta2: f64, eps: f64, step: i32,
) {
    let n = params.len() as std::ffi::c_int;
    // Build raw pointer arrays
    let mut param_ptrs: Vec<*mut std::ffi::c_void> = params.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut grad_ptrs: Vec<*mut std::ffi::c_void> = grads.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut m_ptrs: Vec<*mut std::ffi::c_void> = m.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut v_ptrs: Vec<*mut std::ffi::c_void> = v.iter().map(|t| t.as_ptr() as *mut _).collect();
    unsafe {
        v4_adam_step(
            param_ptrs.as_mut_ptr(), grad_ptrs.as_mut_ptr(),
            m_ptrs.as_mut_ptr(), v_ptrs.as_mut_ptr(),
            n, lr, beta1, beta2, eps, step,
        );
    }
}

/// Call the C++ MoE layer kernel (routing + expert dispatch + shared expert + combine).
/// Expert weights are CPU tensors — C++ does to_device internally.
pub fn glm5_moe_layer_cpp(
    mlp_input: &Tensor,
    shared_gate: &Tensor, shared_up: &Tensor, shared_down: &Tensor,
    shared_gate_scale: Option<&Tensor>, shared_up_scale: Option<&Tensor>, shared_down_scale: Option<&Tensor>,
    gate_weight: &Tensor,
    // Expert weights (CPU)
    expert_gate_weights: &[&Tensor], expert_up_weights: &[&Tensor], expert_down_weights: &[&Tensor],
    expert_gate_scales: &[Option<&Tensor>], expert_up_scales: &[Option<&Tensor>], expert_down_scales: &[Option<&Tensor>],
    local_expert_indices: &[usize],
    n_routed_experts: i32, topk: i32, routed_scaling_factor: f64,
    device_id: i32,
) -> Result<Tensor> {
    fn opt_ptr(t: Option<&Tensor>) -> *mut std::ffi::c_void {
        match t { Some(t) if t.numel() > 0 => t.as_ptr() as *mut _, _ => std::ptr::null_mut() }
    }
    let n = expert_gate_weights.len() as std::ffi::c_int;
    let mut gate_ptrs: Vec<*mut std::ffi::c_void> = expert_gate_weights.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut up_ptrs: Vec<*mut std::ffi::c_void> = expert_up_weights.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut down_ptrs: Vec<*mut std::ffi::c_void> = expert_down_weights.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut gs_ptrs: Vec<*mut std::ffi::c_void> = expert_gate_scales.iter().map(|t| opt_ptr(*t)).collect();
    let mut us_ptrs: Vec<*mut std::ffi::c_void> = expert_up_scales.iter().map(|t| opt_ptr(*t)).collect();
    let mut ds_ptrs: Vec<*mut std::ffi::c_void> = expert_down_scales.iter().map(|t| opt_ptr(*t)).collect();
    let indices: Vec<std::ffi::c_int> = local_expert_indices.iter().map(|&i| i as std::ffi::c_int).collect();

    let result_ptr = unsafe {
        v4_glm5_moe_layer(
            mlp_input.as_ptr() as *mut _,
            shared_gate.as_ptr() as *mut _, shared_up.as_ptr() as *mut _, shared_down.as_ptr() as *mut _,
            opt_ptr(shared_gate_scale), opt_ptr(shared_up_scale), opt_ptr(shared_down_scale),
            gate_weight.as_ptr() as *mut _,
            gate_ptrs.as_mut_ptr(), up_ptrs.as_mut_ptr(), down_ptrs.as_mut_ptr(),
            gs_ptrs.as_mut_ptr(), us_ptrs.as_mut_ptr(), ds_ptrs.as_mut_ptr(),
            n, indices.as_ptr(),
            n_routed_experts, topk, routed_scaling_factor, device_id,
        )
    };
    if result_ptr.is_null() { bail!("C++ v4_glm5_moe_layer returned null"); }
    let tensor = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_glm5_free_at_tensor(result_ptr) };
    Ok(tensor)
}

/// Call the C++ embedding lookup kernel.
pub fn glm5_embedding_cpp(embed_weight: &Tensor, input_ids: &Tensor, device_id: i32) -> Result<Tensor> {
    let result_ptr = unsafe {
        v4_glm5_embedding(embed_weight.as_ptr() as *mut _, input_ids.as_ptr() as *mut _, device_id)
    };
    if result_ptr.is_null() { bail!("C++ v4_glm5_embedding returned null"); }
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
    q_a_proj: &Tensor, q_a_layernorm: &Tensor, q_b_proj: &Tensor,
    kv_a_proj: &Tensor, kv_a_layernorm: &Tensor, kv_b_proj: &Tensor, o_proj: &Tensor,
    // FP8 scales
    q_a_scale: Option<&Tensor>, q_b_scale: Option<&Tensor>,
    kv_a_scale: Option<&Tensor>, kv_b_scale: Option<&Tensor>, o_scale: Option<&Tensor>,
    // Indexer
    idx_wq_b: Option<&Tensor>, idx_wk: Option<&Tensor>,
    idx_k_norm_w: Option<&Tensor>, idx_k_norm_b: Option<&Tensor>,
    idx_weights_proj: Option<&Tensor>,
    idx_wq_b_scale: Option<&Tensor>, idx_wk_scale: Option<&Tensor>,
    // MLP/MoE
    gate_weight: Option<&Tensor>,
    shared_gate: Option<&Tensor>, shared_up: Option<&Tensor>, shared_down: Option<&Tensor>,
    shared_gate_scale: Option<&Tensor>, shared_up_scale: Option<&Tensor>, shared_down_scale: Option<&Tensor>,
    dense_gate: Option<&Tensor>, dense_up: Option<&Tensor>, dense_down: Option<&Tensor>,
    dense_gate_scale: Option<&Tensor>, dense_up_scale: Option<&Tensor>, dense_down_scale: Option<&Tensor>,
    // Expert weights (CPU)
    expert_gate_weights: &[&Tensor], expert_up_weights: &[&Tensor], expert_down_weights: &[&Tensor],
    expert_gate_scales: &[Option<&Tensor>], expert_up_scales: &[Option<&Tensor>], expert_down_scales: &[Option<&Tensor>],
    local_expert_indices: &[usize],
    // Config
    batch: i32, seq: i32, num_heads: i32, qk_nope: i32, qk_rope: i32,
    v_head: i32, kv_lora: i32, idx_head_dim: i32, idx_n_heads: i32, idx_topk: i32,
    layer: i32, is_full_layer: bool, is_moe_layer: bool, n_routed_experts: i32, topk: i32,
    rms_eps: f64, rope_theta: f64, rope_interleave: bool, routed_scaling_factor: f64,
    device_id: i32,
    // IndexShare state
    index_state: &mut Glm5IndexState,
) -> Result<Tensor> {
    fn opt_ptr(t: Option<&Tensor>) -> *mut std::ffi::c_void {
        match t { Some(t) if t.numel() > 0 => t.as_ptr() as *mut _, _ => std::ptr::null_mut() }
    }
    let n = expert_gate_weights.len() as std::ffi::c_int;
    let mut gate_ptrs: Vec<*mut std::ffi::c_void> = expert_gate_weights.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut up_ptrs: Vec<*mut std::ffi::c_void> = expert_up_weights.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut down_ptrs: Vec<*mut std::ffi::c_void> = expert_down_weights.iter().map(|t| t.as_ptr() as *mut _).collect();
    let mut gs_ptrs: Vec<*mut std::ffi::c_void> = expert_gate_scales.iter().map(|t| opt_ptr(*t)).collect();
    let mut us_ptrs: Vec<*mut std::ffi::c_void> = expert_up_scales.iter().map(|t| opt_ptr(*t)).collect();
    let mut ds_ptrs: Vec<*mut std::ffi::c_void> = expert_down_scales.iter().map(|t| opt_ptr(*t)).collect();
    let indices: Vec<std::ffi::c_int> = local_expert_indices.iter().map(|&i| i as std::ffi::c_int).collect();

    let result_ptr = unsafe {
        v4_glm5_layer_forward(
            hidden.as_ptr() as *mut _,
            input_norm_weight.as_ptr() as *mut _,
            post_norm_weight.as_ptr() as *mut _,
            q_a_proj.as_ptr() as *mut _, q_a_layernorm.as_ptr() as *mut _, q_b_proj.as_ptr() as *mut _,
            kv_a_proj.as_ptr() as *mut _, kv_a_layernorm.as_ptr() as *mut _, kv_b_proj.as_ptr() as *mut _,
            o_proj.as_ptr() as *mut _,
            opt_ptr(q_a_scale), opt_ptr(q_b_scale), opt_ptr(kv_a_scale), opt_ptr(kv_b_scale), opt_ptr(o_scale),
            opt_ptr(idx_wq_b), opt_ptr(idx_wk), opt_ptr(idx_k_norm_w), opt_ptr(idx_k_norm_b),
            opt_ptr(idx_weights_proj), opt_ptr(idx_wq_b_scale), opt_ptr(idx_wk_scale),
            opt_ptr(gate_weight), opt_ptr(shared_gate), opt_ptr(shared_up), opt_ptr(shared_down),
            opt_ptr(shared_gate_scale), opt_ptr(shared_up_scale), opt_ptr(shared_down_scale),
            opt_ptr(dense_gate), opt_ptr(dense_up), opt_ptr(dense_down),
            opt_ptr(dense_gate_scale), opt_ptr(dense_up_scale), opt_ptr(dense_down_scale),
            gate_ptrs.as_mut_ptr(), up_ptrs.as_mut_ptr(), down_ptrs.as_mut_ptr(),
            gs_ptrs.as_mut_ptr(), us_ptrs.as_mut_ptr(), ds_ptrs.as_mut_ptr(),
            n, indices.as_ptr(),
            batch, seq, num_heads, qk_nope, qk_rope, v_head, kv_lora, idx_head_dim, idx_n_heads, idx_topk,
            layer, if is_full_layer { 1 } else { 0 }, if is_moe_layer { 1 } else { 0 }, n_routed_experts, topk,
            rms_eps, rope_theta, if rope_interleave { 1 } else { 0 }, routed_scaling_factor, device_id,
            &mut index_state.topk_indices, &mut index_state.idx_bias_keys, &mut index_state.source_layer,
        )
    };
    if result_ptr.is_null() { bail!("C++ v4_glm5_layer_forward returned null"); }
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
