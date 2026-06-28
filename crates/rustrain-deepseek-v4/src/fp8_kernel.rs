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

type ForwardFn = Box<dyn Fn(&Tensor) -> Tensor + Send + Sync>;

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
    F: Fn(&Tensor) -> Tensor + Send + Sync + 'static,
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

    // Note: registry entry is kept for backward pass.
    // It should be cleaned up after backward completes.
    // For simplicity, entries persist until the process exits.

    result
}
