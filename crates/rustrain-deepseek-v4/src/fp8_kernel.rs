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
}

pub fn is_fp8_kernel_available() -> bool {
    let ptr = v4_fp8_scaled_mm as *const ();
    !ptr.is_null()
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
    let m_blocks = m / 128;
    let k_blocks = k / 128;

    let reshaped = input
        .to_kind(Kind::Float)
        .reshape([m_blocks, 128, k_blocks, 128]);

    let block_abs_max = reshaped.abs().amax([1, 3].as_slice(), true);
    let fp8_max = 448.0f64;
    let scale = (block_abs_max / fp8_max).clamp_min(1e-12);
    let scale_2d = scale.squeeze_dim(1).squeeze_dim(2).to_kind(Kind::Float);

    let scale_expanded = scale
        .reshape([m_blocks, 1, k_blocks, 1])
        .expand([m_blocks, 128, k_blocks, 128], false);
    let quantized = (reshaped / &scale_expanded).reshape([m, k]);

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

    let m = input.size()[0];
    let k = input.size()[1];
    let n = weight_fp8.size()[0];

    let (input_fp8, scale_a) = quantize_to_fp8(input);
    let scale_b = expand_weight_scale(weight_scale, n, k);

    let result_ptr = unsafe {
        v4_fp8_scaled_mm(
            input_fp8.as_ptr() as *mut _,
            weight_fp8.as_ptr() as *mut _,
            scale_a.as_ptr() as *mut _,
            scale_b.as_ptr() as *mut _,
        )
    };

    if result_ptr.is_null() {
        bail!("FP8 GEMM returned null (M={m}, N={n}, K={k})");
    }

    let result = unsafe { Tensor::clone_from_ptr(result_ptr as *mut _) };
    unsafe { v4_fp8_free_tensor(result_ptr) };

    Ok(result)
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
