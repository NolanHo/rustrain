//! C++ FFI bindings for Qwen3.6 native kernels.
//!
//! Uses dlopen to dynamically load libqwen36_kernels.so.

use std::ffi::c_void;
use std::sync::OnceLock;
use tch::Tensor;
use anyhow::{Result, bail};

// ── Function pointer types ──

type FnGemm = unsafe extern "C" fn(*mut c_void, *mut c_void, i32) -> *mut c_void;
type FnSwigluGemm = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
type FnChunkedDelta = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, i64) -> *mut c_void;
type FnSdpa = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, i32, f64) -> *mut c_void;
type FnFree = unsafe extern "C" fn(*mut c_void);
type FnSetGrad = unsafe extern "C" fn(i32);
type FnGetGrad = unsafe extern "C" fn() -> i32;

#[repr(C)]
pub struct CppLayerConfig {
    pub layer_type: i64,      // 0=full, 1=linear
    pub num_heads: i64,
    pub num_kv_heads: i64,
    pub head_dim: i64,
    pub num_k_heads: i64,
    pub key_dim: i64,
    pub num_v_heads: i64,
    pub val_dim: i64,
    pub conv_kernel: i64,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub rms_eps: f64,
    pub num_experts: i64,
    pub top_k: i64,
    pub moe_intermediate: i64,
    pub norm_topk_prob: i32,
    pub expert_start: i64,
    pub expert_count: i64,
}

type FnCheckpointForward = unsafe extern "C" fn(
    input: *mut c_void,
    num_layers: i64,
    weight_ptrs: *const *mut c_void,
    layer_configs: *mut c_void,
    compute_type: i32,
    lora_scaling: f64,
    lora_a_ptrs: *const *mut c_void,
    lora_b_ptrs: *const *mut c_void,
) -> *mut c_void;

struct KernelHandles {
    gemm: FnGemm,
    swiglu_gemm: FnSwigluGemm,
    chunked_delta: FnChunkedDelta,
    sdpa: FnSdpa,
    free_tensor: FnFree,
    set_grad: FnSetGrad,
    get_grad: FnGetGrad,
    checkpoint_forward: FnCheckpointForward,
}

static KERNELS: OnceLock<Option<KernelHandles>> = OnceLock::new();

// ── dlopen loading ──

mod libc {
    use std::ffi::c_void;
    pub const RTLD_LAZY: i32 = 1;
    pub const RTLD_NOLOAD: i32 = 4;

    unsafe extern "C" {
        pub fn dlopen(filename: *const i8, flag: i32) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
        pub fn dlerror() -> *mut i8;
    }
}

unsafe fn load_kernels() -> Option<KernelHandles> {
    use std::ffi::CString;
    let lib_name = CString::new("libqwen36_kernels.so").unwrap();
    let handle = libc::dlopen(lib_name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_NOLOAD);
    let handle = if handle.is_null() {
        let h = libc::dlopen(lib_name.as_ptr(), libc::RTLD_LAZY);
        if h.is_null() {
            let err = libc::dlerror();
            if !err.is_null() {
                let err_str = std::ffi::CStr::from_ptr(err).to_string_lossy();
                eprintln!("[qwen36 kernels] dlopen failed: {err_str}");
            }
            return None;
        }
        h
    } else {
        handle
    };

    macro_rules! sym {
        ($name:expr) => {{
            let sym_name = CString::new($name).unwrap();
            let ptr = libc::dlsym(handle, sym_name.as_ptr());
            if ptr.is_null() { return None; }
            std::mem::transmute::<*mut c_void, _>(ptr)
        }};
    }

    Some(KernelHandles {
        gemm: sym!("qwen36_gemm"),
        swiglu_gemm: sym!("qwen36_swiglu_gemm"),
        chunked_delta: sym!("qwen36_chunked_delta_rule"),
        sdpa: sym!("qwen36_sdpa"),
        free_tensor: sym!("qwen36_free_tensor"),
        set_grad: sym!("qwen36_set_grad_enabled"),
        get_grad: sym!("qwen36_get_grad_enabled"),
        checkpoint_forward: sym!("qwen36_checkpoint_forward"),
    })
}

pub fn kernels_available() -> bool {
    KERNELS.get_or_init(|| unsafe { load_kernels() }).is_some()
}

fn get_kernels() -> Option<&'static KernelHandles> {
    KERNELS.get_or_init(|| unsafe { load_kernels() }).as_ref()
}

fn ptr_to_tensor(ptr: *mut c_void) -> Result<Tensor> {
    if ptr.is_null() {
        bail!("C++ kernel returned null");
    }
    let tensor = unsafe { Tensor::clone_from_ptr(ptr as *mut _) };
    Ok(tensor)
}

/// Native GEMM
pub fn native_gemm(a: &Tensor, b: &Tensor, transpose_b: bool) -> Result<Tensor> {
    let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
    let ptr = unsafe { (kh.gemm)(a.as_ptr() as *mut _, b.as_ptr() as *mut _, transpose_b as i32) };
    ptr_to_tensor(ptr)
}

/// Fused SwiGLU GEMM
pub fn native_swiglu_gemm(a: &Tensor, gate_up: &Tensor, down: &Tensor) -> Result<Tensor> {
    let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
    let ptr = unsafe { (kh.swiglu_gemm)(a.as_ptr() as *mut _, gate_up.as_ptr() as *mut _, down.as_ptr() as *mut _) };
    ptr_to_tensor(ptr)
}

/// Flash attention via C++ SDPA
pub fn native_sdpa(q: &Tensor, k: &Tensor, v: &Tensor, is_causal: bool, scale: f64) -> Result<Tensor> {
    let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
    let ptr = unsafe { (kh.sdpa)(q.as_ptr() as *mut _, k.as_ptr() as *mut _, v.as_ptr() as *mut _, is_causal as i32, scale) };
    ptr_to_tensor(ptr)
}

// ── Gradient checkpointing via C++ autograd::Function ──

/// Build weight pointer array for a set of layers (to pass to C++ forward).
/// Returns a flat Vec<*mut c_void> with all layer weights concatenated.
pub fn build_weight_ptrs(
    weights: &std::collections::BTreeMap<String, tch::Tensor>,
    config: &crate::config::Qwen36RuntimeConfig,
    layer_indices: &[usize],
) -> Vec<*mut c_void> {
    let p = &config.weight_prefix;
    let mut ptrs = Vec::new();
    for &layer in layer_indices {
        let lp = format!("{p}layers.{layer}");
        // Always add norms
        ptrs.push(get_tensor_ptr(weights, &format!("{lp}.input_layernorm.weight")));
        ptrs.push(get_tensor_ptr(weights, &format!("{lp}.post_attention_layernorm.weight")));
        match config.layer_types[layer] {
            crate::config::LayerType::FullAttention => {
                for w in &["q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"] {
                    ptrs.push(get_tensor_ptr(weights, &format!("{lp}.self_attn.{w}.weight")));
                }
            }
            crate::config::LayerType::LinearAttention => {
                ptrs.push(get_tensor_ptr(weights, &format!("{lp}.linear_attn.in_proj_qkv.weight")));
                ptrs.push(get_tensor_ptr(weights, &format!("{lp}.linear_attn.in_proj_z.weight")));
                ptrs.push(get_tensor_ptr(weights, &format!("{lp}.linear_attn.in_proj_a.weight")));
                ptrs.push(get_tensor_ptr(weights, &format!("{lp}.linear_attn.in_proj_b.weight")));
                ptrs.push(get_tensor_ptr(weights, &format!("{lp}.linear_attn.A_log")));
                ptrs.push(get_tensor_ptr(weights, &format!("{lp}.linear_attn.dt_bias")));
                ptrs.push(get_tensor_ptr(weights, &format!("{lp}.linear_attn.conv1d.weight")));
                ptrs.push(get_tensor_ptr(weights, &format!("{lp}.linear_attn.norm.weight")));
                ptrs.push(get_tensor_ptr(weights, &format!("{lp}.linear_attn.out_proj.weight")));
            }
        }
        // MoE weights
        ptrs.push(get_tensor_ptr(weights, &format!("{lp}.mlp.gate.weight")));
        ptrs.push(get_tensor_ptr(weights, &format!("{lp}.mlp.shared_expert_gate.weight")));
        ptrs.push(get_tensor_ptr(weights, &format!("{lp}.mlp.shared_expert.gate_proj.weight")));
        ptrs.push(get_tensor_ptr(weights, &format!("{lp}.mlp.shared_expert.up_proj.weight")));
        ptrs.push(get_tensor_ptr(weights, &format!("{lp}.mlp.shared_expert.down_proj.weight")));
        ptrs.push(get_tensor_ptr(weights, &format!("{lp}.mlp.experts.gate_up_proj")));
        ptrs.push(get_tensor_ptr(weights, &format!("{lp}.mlp.experts.down_proj")));
    }
    ptrs
}

fn get_tensor_ptr(weights: &std::collections::BTreeMap<String, tch::Tensor>, name: &str) -> *mut c_void {
    match weights.get(name) {
        Some(t) => t.as_ptr() as *mut c_void,
        None => std::ptr::null_mut(),
    }
}

/// Build C++ LayerConfig array for a set of layers.
pub fn build_layer_configs(
    config: &crate::config::Qwen36RuntimeConfig,
    layer_indices: &[usize],
    expert_start: usize,
    expert_count: usize,
) -> Vec<CppLayerConfig> {
    layer_indices.iter().map(|&layer| {
        let lt = &config.layer_types[layer];
        CppLayerConfig {
            layer_type: match lt {
                crate::config::LayerType::FullAttention => 0,
                crate::config::LayerType::LinearAttention => 1,
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_k_heads: config.linear_num_key_heads,
            key_dim: config.linear_key_head_dim,
            num_v_heads: config.linear_num_value_heads,
            val_dim: config.linear_value_head_dim,
            conv_kernel: config.linear_conv_kernel_dim,
            partial_rotary_factor: config.partial_rotary_factor,
            rope_theta: config.rope_theta,
            rms_eps: config.rms_norm_eps,
            num_experts: config.num_experts as i64,
            top_k: config.num_experts_per_tok as i64,
            moe_intermediate: config.moe_intermediate_size,
            norm_topk_prob: if config.norm_topk_prob { 1 } else { 0 },
            expert_start: expert_start as i64,
            expert_count: expert_count as i64,
        }
    }).collect()
}

/// Build LoRA adapter pointer arrays.
/// Returns (lora_a_ptrs, lora_b_ptrs) — 4 entries per layer for full attn,
/// 3 entries per layer for linear attn, null if no LoRA.
///
/// Layout per layer:
/// - Full attention: [q_proj_a, k_proj_a, v_proj_a, o_proj_a, q_proj_b, k_proj_b, v_proj_b, o_proj_b]
///   (null if that module has no LoRA)
/// - Linear attention: [in_proj_qkv_a, in_proj_z_a, out_proj_a, in_proj_qkv_b, in_proj_z_b, out_proj_b]
///
/// C++ reads indices 0-3 for full attn A and B respectively.
pub fn build_lora_ptrs(
    registry: &crate::lora::Qwen36LoraRegistry,
    layer_indices: &[usize],
    config: &crate::config::Qwen36RuntimeConfig,
) -> (Vec<*mut c_void>, Vec<*mut c_void>) {
    let mut a_ptrs = Vec::new();
    let mut b_ptrs = Vec::new();
    for &layer in layer_indices {
        match config.layer_types[layer] {
            crate::config::LayerType::FullAttention => {
                // 4 A pointers + 4 B pointers for full attn
                for module in &[
                    crate::lora::Qwen36LoraTargetModule::QProj,
                    crate::lora::Qwen36LoraTargetModule::KProj,
                    crate::lora::Qwen36LoraTargetModule::VProj,
                    crate::lora::Qwen36LoraTargetModule::OProj,
                ] {
                    if let Some((a, b)) = registry.adapter_ref(layer, *module) {
                        a_ptrs.push(a.as_ptr() as *mut c_void);
                        b_ptrs.push(b.as_ptr() as *mut c_void);
                    } else {
                        a_ptrs.push(std::ptr::null_mut());
                        b_ptrs.push(std::ptr::null_mut());
                    }
                }
            }
            crate::config::LayerType::LinearAttention => {
                // 3 A pointers + 3 B pointers for linear attn
                for module in &[
                    crate::lora::Qwen36LoraTargetModule::InProjQkv,
                    crate::lora::Qwen36LoraTargetModule::InProjZ,
                    crate::lora::Qwen36LoraTargetModule::OutProj,
                ] {
                    if let Some((a, b)) = registry.adapter_ref(layer, *module) {
                        a_ptrs.push(a.as_ptr() as *mut c_void);
                        b_ptrs.push(b.as_ptr() as *mut c_void);
                    } else {
                        a_ptrs.push(std::ptr::null_mut());
                        b_ptrs.push(std::ptr::null_mut());
                    }
                }
            }
        }
    }
    (a_ptrs, b_ptrs)
}

/// C++ checkpoint forward — runs a group of layers with gradient checkpointing.
///
/// Forward: runs in no_grad, saves only the input.
/// Backward: recomputes with grad enabled, accumulates LoRA gradients.
pub fn checkpoint_forward(
    input: &Tensor,
    layer_indices: &[usize],
    weights: &std::collections::BTreeMap<String, Tensor>,
    config: &crate::config::Qwen36RuntimeConfig,
    registry: &crate::lora::Qwen36LoraRegistry,
    expert_start: usize,
    expert_count: usize,
    compute_type: tch::Kind,
) -> Result<Tensor> {
    let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;

    let mut weight_ptrs = build_weight_ptrs(weights, config, layer_indices);
    let layer_configs = build_layer_configs(config, layer_indices, expert_start, expert_count);
    let (mut lora_a, mut lora_b) = build_lora_ptrs(registry, layer_indices, config);

    // Pass the actual at::ScalarType enum value (tch::Kind maps to the same enum)
    let compute_type_code = compute_type as i32;

    let scaling = registry.scaling();

    // LEAK the Vecs so their backing arrays survive until backward (which may
    // happen much later when loss.backward() is called). The C++ autograd::Function
    // stores raw pointers to these arrays in saved_data.
    let weight_ptrs_ptr = weight_ptrs.as_ptr();
    std::mem::forget(weight_ptrs);
    let layer_configs_ptr = layer_configs.as_ptr() as *mut c_void;
    std::mem::forget(layer_configs);
    let lora_a_ptr = lora_a.as_ptr();
    std::mem::forget(lora_a);
    let lora_b_ptr = lora_b.as_ptr();
    std::mem::forget(lora_b);

    let ptr = unsafe {
        (kh.checkpoint_forward)(
            input.as_ptr() as *mut _,
            layer_indices.len() as i64,
            weight_ptrs_ptr,
            layer_configs_ptr,
            compute_type_code as i32,
            scaling,
            lora_a_ptr,
            lora_b_ptr,
        )
    };

    if ptr.is_null() {
        bail!("C++ checkpoint_forward returned null");
    }
    ptr_to_tensor(ptr)
}
