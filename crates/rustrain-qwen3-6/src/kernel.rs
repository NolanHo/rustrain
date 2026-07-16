//! C++ FFI bindings for Qwen3.6 native training — all-in-C++ path.
//!
//! TrainingContext, train_step, and adapter export all happen in C++.
//! Rust only handles: weight loading, data loading, training loop orchestration.

use crate::lora::Qwen36LoraTargetModule;
use anyhow::{Result, bail};
use std::ffi::c_void;
use std::sync::OnceLock;
use tch::{Kind, Tensor};

// ── dlopen ──

type FnCreateCtx = unsafe extern "C" fn(
    *mut *mut c_void,
    i64,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i64,
    i32,
    f64,
    f64,
    f64,
    f64,
    f64,
    i64,
    f64,
    i64,
    *const i64,
    i64,
    *const i8,
) -> *mut c_void;
type FnKernelAbiVersion = unsafe extern "C" fn() -> i64;
type FnTrainStep = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> f64;
type FnTrainMicroStep =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, f64, i32) -> f64;
type FnTrainMultiLora =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, i32, i32) -> f64;
type FnEvalStep = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> f64;
type FnGetLoraCount = unsafe extern "C" fn(*mut c_void) -> i64;
type FnGetLoraA = unsafe extern "C" fn(*mut c_void, i64) -> *mut c_void;
type FnGetLoraB = unsafe extern "C" fn(*mut c_void, i64) -> *mut c_void;
type FnSetLoraTensor = unsafe extern "C" fn(*mut c_void, i64, i32, *mut c_void) -> i32;
type FnGetStepCount = unsafe extern "C" fn(*mut c_void) -> i64;
type FnExportOptimizer =
    unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut *mut c_void, i64) -> i64;
type FnImportOptimizer =
    unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut *mut c_void, i64) -> i64;
type FnFreeCtx = unsafe extern "C" fn(*mut c_void);
type FnGemm = unsafe extern "C" fn(*mut c_void, *mut c_void, i32) -> *mut c_void;
type FnFreeTensor = unsafe extern "C" fn(*mut c_void);
type FnSetMtpWeights = unsafe extern "C" fn(
    *mut c_void,      // ctx_ptr
    *mut c_void,      // mtp_fc_ptr
    *mut c_void,      // mtp_pre_fc_norm_emb_ptr
    *mut c_void,      // mtp_pre_fc_norm_hidden_ptr
    *mut c_void,      // mtp_norm_ptr
    *mut *mut c_void, // mtp_layer_weight_ptrs
    i64,              // num_mtp_layer_weights
    *mut c_void,      // mtp_layer_configs_ptr
    i64,              // num_mtp_layers
);
type FnSetCheckpoint = unsafe extern "C" fn(*mut c_void, i32, i64);
type FnSetNcclComm = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, i32, i32);
type FnInitNccl = unsafe extern "C" fn(*mut c_void) -> i32;
type FnSetCudaDevice = unsafe extern "C" fn(i32);
type FnAddLora = unsafe extern "C" fn(*mut c_void, i64, f64, *const i64, i64, *const i8) -> i64;
type FnRemoveLora = unsafe extern "C" fn(*mut c_void, i64) -> i32;
type FnListLora = unsafe extern "C" fn(*mut c_void, *mut i64, i64) -> i64;
type FnGetAdapterLoraTensor =
    unsafe extern "C" fn(*mut c_void, i64, i64, *const i8, i32) -> *mut c_void;
type FnSetAdapterLoraTensor =
    unsafe extern "C" fn(*mut c_void, i64, i64, *const i8, i32, *mut c_void) -> i32;
type FnSetAdapterId = unsafe extern "C" fn(*mut c_void, i64, i64) -> i32;
type FnGetAdapterOptimizerTensor =
    unsafe extern "C" fn(*mut c_void, i64, i64, *const i8, i32, i32) -> *mut c_void;
type FnSetAdapterOptimizerTensor =
    unsafe extern "C" fn(*mut c_void, i64, i64, *const i8, i32, i32, *mut c_void) -> i32;

#[repr(C)]
pub struct CppLayerConfig {
    pub layer_type: i64,
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
    pub expert_start: i64,
    pub expert_count: i64,
    pub intermediate_size: i64,
    pub norm_topk_prob: i32,
    // NCCL handles for EP all-reduce (must match C++ LayerConfig layout)
    pub nccl_comm: *mut c_void,
    pub nccl_stream: *mut c_void,
}

struct KernelHandles {
    create_ctx: FnCreateCtx,
    train_step: FnTrainStep,
    train_micro_step: FnTrainMicroStep,
    train_multi_lora: FnTrainMultiLora,
    eval_step: FnEvalStep,
    get_lora_count: FnGetLoraCount,
    get_lora_a: FnGetLoraA,
    get_lora_b: FnGetLoraB,
    set_lora_tensor: FnSetLoraTensor,
    get_step_count: FnGetStepCount,
    export_optimizer: FnExportOptimizer,
    import_optimizer: FnImportOptimizer,
    free_ctx: FnFreeCtx,
    gemm: FnGemm,
    free_tensor: FnFreeTensor,
    set_mtp_weights: FnSetMtpWeights,
    set_checkpoint: FnSetCheckpoint,
    set_nccl_comm: FnSetNcclComm,
    init_nccl: FnInitNccl,
    set_cuda_device: FnSetCudaDevice,
    add_lora: FnAddLora,
    remove_lora: FnRemoveLora,
    list_lora: FnListLora,
    get_adapter_lora_tensor: FnGetAdapterLoraTensor,
    set_adapter_lora_tensor: FnSetAdapterLoraTensor,
    set_adapter_id: FnSetAdapterId,
    get_adapter_optimizer_tensor: FnGetAdapterOptimizerTensor,
    set_adapter_optimizer_tensor: FnSetAdapterOptimizerTensor,
}

static KERNELS: OnceLock<Option<KernelHandles>> = OnceLock::new();

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
            return None;
        }
        h
    } else {
        handle
    };

    macro_rules! sym {
        ($name:expr) => {{
            let s = CString::new($name).unwrap();
            let p = libc::dlsym(handle, s.as_ptr());
            if p.is_null() {
                return None;
            }
            std::mem::transmute::<*mut c_void, _>(p)
        }};
    }
    let abi_version: FnKernelAbiVersion = sym!("qwen36_kernel_abi_version");
    if abi_version() != 7 {
        return None;
    }
    Some(KernelHandles {
        create_ctx: sym!("qwen36_create_training_context"),
        train_step: sym!("qwen36_train_step"),
        train_micro_step: sym!("qwen36_train_micro_step"),
        train_multi_lora: sym!("qwen36_train_multi_lora"),
        eval_step: sym!("qwen36_eval_step"),
        get_lora_count: sym!("qwen36_get_lora_count"),
        get_lora_a: sym!("qwen36_get_lora_a"),
        get_lora_b: sym!("qwen36_get_lora_b"),
        set_lora_tensor: sym!("qwen36_set_lora_tensor"),
        get_step_count: sym!("qwen36_get_step_count"),
        export_optimizer: sym!("qwen36_export_optimizer_state"),
        import_optimizer: sym!("qwen36_import_optimizer_state"),
        free_ctx: sym!("qwen36_free_training_context"),
        gemm: sym!("qwen36_gemm"),
        free_tensor: sym!("qwen36_free_tensor"),
        set_mtp_weights: sym!("qwen36_set_mtp_weights"),
        set_checkpoint: sym!("qwen36_set_checkpoint"),
        set_nccl_comm: sym!("qwen36_set_nccl_comm"),
        init_nccl: sym!("qwen36_init_nccl"),
        set_cuda_device: sym!("qwen36_set_cuda_device"),
        add_lora: sym!("qwen36_add_lora"),
        remove_lora: sym!("qwen36_remove_lora"),
        list_lora: sym!("qwen36_list_lora"),
        get_adapter_lora_tensor: sym!("qwen36_get_adapter_lora_tensor"),
        set_adapter_lora_tensor: sym!("qwen36_set_adapter_lora_tensor"),
        set_adapter_id: sym!("qwen36_set_adapter_id"),
        get_adapter_optimizer_tensor: sym!("qwen36_get_adapter_optimizer_tensor"),
        set_adapter_optimizer_tensor: sym!("qwen36_set_adapter_optimizer_tensor"),
    })
}

pub fn kernels_available() -> bool {
    KERNELS.get_or_init(|| unsafe { load_kernels() }).is_some()
}

fn get_kernels() -> Option<&'static KernelHandles> {
    KERNELS.get_or_init(|| unsafe { load_kernels() }).as_ref()
}

// ── Weight pointer helpers ──

fn get_ptr(weights: &std::collections::BTreeMap<String, Tensor>, name: &str) -> *mut c_void {
    match weights.get(name) {
        Some(t) => t.as_ptr() as *mut c_void,
        None => std::ptr::null_mut(),
    }
}

pub fn build_weight_ptrs(
    weights: &std::collections::BTreeMap<String, Tensor>,
    config: &crate::config::Qwen36RuntimeConfig,
) -> Vec<*mut c_void> {
    let p = &config.weight_prefix;
    let mut ptrs = Vec::new();
    for layer in 0..config.num_hidden_layers {
        let lp = format!("{p}layers.{layer}");
        ptrs.push(get_ptr(weights, &format!("{lp}.input_layernorm.weight")));
        ptrs.push(get_ptr(
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
        ));
        match config.layer_types[layer] {
            crate::config::LayerType::FullAttention => {
                for w in &["q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"] {
                    ptrs.push(get_ptr(weights, &format!("{lp}.self_attn.{w}.weight")));
                }
            }
            crate::config::LayerType::LinearAttention => {
                ptrs.push(get_ptr(
                    weights,
                    &format!("{lp}.linear_attn.in_proj_qkv.weight"),
                ));
                ptrs.push(get_ptr(
                    weights,
                    &format!("{lp}.linear_attn.in_proj_z.weight"),
                ));
                ptrs.push(get_ptr(
                    weights,
                    &format!("{lp}.linear_attn.in_proj_a.weight"),
                ));
                ptrs.push(get_ptr(
                    weights,
                    &format!("{lp}.linear_attn.in_proj_b.weight"),
                ));
                ptrs.push(get_ptr(weights, &format!("{lp}.linear_attn.A_log")));
                ptrs.push(get_ptr(weights, &format!("{lp}.linear_attn.dt_bias")));
                ptrs.push(get_ptr(weights, &format!("{lp}.linear_attn.conv1d.weight")));
                ptrs.push(get_ptr(weights, &format!("{lp}.linear_attn.norm.weight")));
                ptrs.push(get_ptr(
                    weights,
                    &format!("{lp}.linear_attn.out_proj.weight"),
                ));
            }
        }
        if config.is_moe {
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.gate.weight")));
            ptrs.push(get_ptr(
                weights,
                &format!("{lp}.mlp.shared_expert_gate.weight"),
            ));
            ptrs.push(get_ptr(
                weights,
                &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
            ));
            ptrs.push(get_ptr(
                weights,
                &format!("{lp}.mlp.shared_expert.up_proj.weight"),
            ));
            ptrs.push(get_ptr(
                weights,
                &format!("{lp}.mlp.shared_expert.down_proj.weight"),
            ));
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.experts.gate_up_proj")));
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.experts.down_proj")));
        } else {
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.gate_proj.weight")));
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.up_proj.weight")));
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.down_proj.weight")));
        }
    }
    ptrs
}

pub fn build_layer_configs(
    config: &crate::config::Qwen36RuntimeConfig,
    expert_start: usize,
    expert_count: usize,
) -> Vec<CppLayerConfig> {
    (0..config.num_hidden_layers)
        .map(|layer| {
            let lt = &config.layer_types[layer];
            CppLayerConfig {
                layer_type: match lt {
                    crate::config::LayerType::FullAttention => 0,
                    _ => 1,
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
                intermediate_size: config.intermediate_size,
                nccl_comm: std::ptr::null_mut(),
                nccl_stream: std::ptr::null_mut(),
            }
        })
        .collect()
}

/// Build weight pointers for MTP layers (full attention layers).
/// MoE: 15 weights per layer (2 norm + 6 attn + 7 MoE)
/// Dense: 11 weights per layer (2 norm + 6 attn + 3 dense MLP)
pub fn build_mtp_weight_ptrs(
    weights: &std::collections::BTreeMap<String, Tensor>,
    config: &crate::config::Qwen36RuntimeConfig,
) -> Vec<*mut c_void> {
    let mut ptrs = Vec::new();
    for layer in 0..config.mtp_num_hidden_layers {
        let lp = format!("mtp.layers.{layer}");
        ptrs.push(get_ptr(weights, &format!("{lp}.input_layernorm.weight")));
        ptrs.push(get_ptr(
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
        ));
        // Full attention: q, q_norm, k, k_norm, v, o
        for w in &["q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"] {
            ptrs.push(get_ptr(weights, &format!("{lp}.self_attn.{w}.weight")));
        }
        if config.is_moe {
            // MoE: gate, shared_expert_gate, shared_gate_proj, shared_up_proj, shared_down_proj, experts_gate_up, experts_down
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.gate.weight")));
            ptrs.push(get_ptr(
                weights,
                &format!("{lp}.mlp.shared_expert_gate.weight"),
            ));
            ptrs.push(get_ptr(
                weights,
                &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
            ));
            ptrs.push(get_ptr(
                weights,
                &format!("{lp}.mlp.shared_expert.up_proj.weight"),
            ));
            ptrs.push(get_ptr(
                weights,
                &format!("{lp}.mlp.shared_expert.down_proj.weight"),
            ));
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.experts.gate_up_proj")));
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.experts.down_proj")));
        } else {
            // Dense MLP: gate_proj, up_proj, down_proj
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.gate_proj.weight")));
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.up_proj.weight")));
            ptrs.push(get_ptr(weights, &format!("{lp}.mlp.down_proj.weight")));
        }
    }
    ptrs
}

/// Build layer configs for MTP layers (all full attention type).
pub fn build_mtp_layer_configs(
    config: &crate::config::Qwen36RuntimeConfig,
    expert_start: usize,
    expert_count: usize,
) -> Vec<CppLayerConfig> {
    (0..config.mtp_num_hidden_layers)
        .map(|_| {
            CppLayerConfig {
                layer_type: 0, // MTP layers are always full attention
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
                intermediate_size: config.intermediate_size,
                nccl_comm: std::ptr::null_mut(),
                nccl_stream: std::ptr::null_mut(),
            }
        })
        .collect()
}

/// Opaque training context handle.
pub struct CppTrainingContext {
    ptr: *mut c_void,
    lora_count: i64,
}

impl CppTrainingContext {
    /// Create training context — LoRA A/B created in C++ as at::Tensor (requires_grad=true).
    pub fn new(
        weights: &std::collections::BTreeMap<String, Tensor>,
        config: &crate::config::Qwen36RuntimeConfig,
        compute_kind: Kind,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        lora_scaling: f64,
        lora_rank: i64,
        target_layers: &[usize],
        target_modules: &[Qwen36LoraTargetModule],
        expert_start: usize,
        expert_count: usize,
    ) -> Result<Self> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let mut weight_ptrs = build_weight_ptrs(weights, config);
        let layer_configs = build_layer_configs(config, expert_start, expert_count);

        let embed_ptr = get_ptr(
            weights,
            &format!("{}embed_tokens.weight", config.weight_prefix),
        );
        let final_norm_ptr = get_ptr(weights, &format!("{}norm.weight", config.weight_prefix));
        let lm_head_ptr = if config.tie_word_embeddings {
            // Tied embeddings: use embed_tokens as lm_head
            embed_ptr
        } else {
            get_ptr(weights, "lm_head.weight")
        };

        let compute_type = compute_kind as i32;

        // LEAK the Vecs so their backing arrays survive (C++ holds raw pointers)
        let wp_ptr = weight_ptrs.as_ptr() as *mut *mut c_void;
        let wp_len = weight_ptrs.len();
        std::mem::forget(weight_ptrs);
        let lc_ptr = layer_configs.as_ptr() as *mut c_void;
        std::mem::forget(layer_configs);

        // Convert target_layers to i64 array (leaked to keep alive)
        let target_i64: Vec<i64> = target_layers.iter().map(|&x| x as i64).collect();
        let tl_ptr = if target_i64.is_empty() {
            std::ptr::null()
        } else {
            let p = target_i64.as_ptr();
            std::mem::forget(target_i64);
            p
        };
        let tl_len = target_layers.len() as i64;

        let module_names = target_modules
            .iter()
            .map(Qwen36LoraTargetModule::cpp_name)
            .collect::<Vec<_>>()
            .join(",");
        let module_names_c = std::ffi::CString::new(module_names)
            .map_err(|_| anyhow::anyhow!("LoRA target module contains NUL"))?;
        let modules_ptr = if target_modules.is_empty() {
            std::ptr::null()
        } else {
            module_names_c.as_ptr()
        };

        let ptr = unsafe {
            (kh.create_ctx)(
                wp_ptr,
                wp_len as i64,
                embed_ptr,
                final_norm_ptr,
                lm_head_ptr,
                lc_ptr,
                config.num_hidden_layers as i64,
                compute_type,
                lora_scaling,
                lr,
                beta1,
                beta2,
                eps,
                config.vocab_size,
                config.rms_norm_eps,
                lora_rank,
                tl_ptr,
                tl_len,
                modules_ptr,
            )
        };
        if ptr.is_null() {
            bail!("C++ create_training_context returned null");
        }
        let lora_count = unsafe { (kh.get_lora_count)(ptr) };
        Ok(Self { ptr, lora_count })
    }

    /// Run one training step: forward + loss + backward + Adam update.
    /// Returns loss value.
    pub fn train_step(
        &self,
        input_ids: &Tensor,
        target_mask: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<f64> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let loss = unsafe {
            (kh.train_step)(
                self.ptr,
                input_ids.as_ptr() as *mut _,
                target_mask.as_ptr() as *mut _,
                attention_mask.as_ptr() as *mut _,
            )
        };
        if loss < 0.0 {
            bail!("C++ train_step failed");
        }
        Ok(loss)
    }

    /// Run one micro-batch and optionally apply the synchronized Adam update.
    pub fn train_micro_step(
        &self,
        input_ids: &Tensor,
        target_mask: &Tensor,
        attention_mask: &Tensor,
        gradient_scale: f64,
        apply_optimizer: bool,
    ) -> Result<f64> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let loss = unsafe {
            (kh.train_micro_step)(
                self.ptr,
                input_ids.as_ptr() as *mut _,
                target_mask.as_ptr() as *mut _,
                attention_mask.as_ptr() as *mut _,
                gradient_scale,
                i32::from(apply_optimizer),
            )
        };
        if loss < 0.0 {
            bail!("C++ train_micro_step failed");
        }
        Ok(loss)
    }

    /// Train all adapters in batched activation chunks. Chunks accumulate
    /// gradients and share one logical Adam update. Input is expanded to [N, seq].
    /// n_total: total number of adapters. lora_rank: LoRA rank for N_max calc.
    /// Returns average loss across chunks.
    pub fn train_multi_lora(
        &self,
        input_ids: &Tensor,
        target_mask: &Tensor,
        attention_mask: &Tensor,
        n_total: i32,
        lora_rank: i32,
    ) -> Result<f64> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let loss = unsafe {
            (kh.train_multi_lora)(
                self.ptr,
                input_ids.as_ptr() as *mut _,
                target_mask.as_ptr() as *mut _,
                attention_mask.as_ptr() as *mut _,
                n_total,
                lora_rank,
            )
        };
        if loss < 0.0 {
            bail!("C++ train_multi_lora failed");
        }
        Ok(loss)
    }

    /// Get LoRA A tensor by index (for saving).
    pub fn get_lora_a(&self, index: i64) -> Option<Tensor> {
        let kh = get_kernels()?;
        let ptr = unsafe { (kh.get_lora_a)(self.ptr, index) };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { Tensor::clone_from_ptr(ptr as *mut _) })
    }

    /// Get LoRA B tensor by index (for saving).
    pub fn get_lora_b(&self, index: i64) -> Option<Tensor> {
        let kh = get_kernels()?;
        let ptr = unsafe { (kh.get_lora_b)(self.ptr, index) };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { Tensor::clone_from_ptr(ptr as *mut _) })
    }

    pub fn set_lora_tensor(&self, index: i64, is_b: bool, tensor: &Tensor) -> Result<()> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let status = unsafe {
            (kh.set_lora_tensor)(
                self.ptr,
                index,
                if is_b { 1 } else { 0 },
                tensor.as_ptr() as *mut c_void,
            )
        };
        if status != 0 {
            bail!("C++ set_lora_tensor failed for slot {index}");
        }
        Ok(())
    }

    pub fn lora_count(&self) -> i64 {
        self.lora_count
    }

    /// Set MTP weights on the C++ training context.
    /// Must be called after `new()` if MTP is enabled.
    pub fn set_mtp_weights(
        &self,
        weights: &std::collections::BTreeMap<String, Tensor>,
        config: &crate::config::Qwen36RuntimeConfig,
        expert_start: usize,
        expert_count: usize,
    ) -> Result<()> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;

        let mtp_fc_ptr = get_ptr(weights, "mtp.fc.weight");
        let mtp_pre_fc_norm_emb_ptr = get_ptr(weights, "mtp.pre_fc_norm_embedding.weight");
        let mtp_pre_fc_norm_hidden_ptr = get_ptr(weights, "mtp.pre_fc_norm_hidden.weight");
        let mtp_norm_ptr = get_ptr(weights, "mtp.norm.weight");

        let mut mtp_weight_ptrs = build_mtp_weight_ptrs(weights, config);
        let mtp_layer_configs = build_mtp_layer_configs(config, expert_start, expert_count);

        let wp_ptr = mtp_weight_ptrs.as_ptr() as *mut *mut c_void;
        let wp_len = mtp_weight_ptrs.len();
        std::mem::forget(mtp_weight_ptrs);
        let lc_ptr = mtp_layer_configs.as_ptr() as *mut c_void;
        std::mem::forget(mtp_layer_configs);

        unsafe {
            (kh.set_mtp_weights)(
                self.ptr,
                mtp_fc_ptr,
                mtp_pre_fc_norm_emb_ptr,
                mtp_pre_fc_norm_hidden_ptr,
                mtp_norm_ptr,
                wp_ptr,
                wp_len as i64,
                lc_ptr,
                config.mtp_num_hidden_layers as i64,
            );
        }
        Ok(())
    }

    /// Enable/disable gradient checkpointing.
    /// When enabled, layers are grouped by `group_size` and intermediate
    /// activations are recomputed during backward instead of stored.
    pub fn set_checkpoint(&self, enable: bool, group_size: i64) {
        let kh = get_kernels().expect("kernels not loaded");
        unsafe {
            (kh.set_checkpoint)(self.ptr, if enable { 1 } else { 0 }, group_size);
        }
    }

    /// Set NCCL communicator for Expert Parallel all-reduce.
    /// Must be called after `new()` if EP is enabled.
    /// comm_ptr / stream_ptr from `NcclPersistentComm::raw_comm_ptr()` / `raw_stream_ptr()`.
    pub fn set_nccl_comm(
        &self,
        comm_ptr: *mut c_void,
        stream_ptr: *mut c_void,
        ep_rank: i32,
        ep_world_size: i32,
    ) {
        let kh = get_kernels().expect("kernels not loaded");
        unsafe {
            (kh.set_nccl_comm)(self.ptr, comm_ptr, stream_ptr, ep_rank, ep_world_size);
        }
    }

    /// Initialize NCCL communicator directly in C++ (preferred over set_nccl_comm).
    /// Reads RANK/WORLD_SIZE/LOCAL_RANK from env vars.
    /// Returns 0 on success, -1 on failure.
    pub fn init_nccl(&self) -> i32 {
        let kh = get_kernels().expect("kernels not loaded");
        unsafe { (kh.init_nccl)(self.ptr) }
    }

    /// Set CUDA device and force PyTorch CUDA context initialization.
    /// Must be called before any GPU operation in worker processes.
    pub fn set_cuda_device(device: i32) {
        let kh = get_kernels().expect("kernels not loaded");
        unsafe { (kh.set_cuda_device)(device) }
    }

    /// Add a new LoRA adapter. Returns adapter ID (>0) on success.
    /// target_layers: empty = all layers
    /// target_modules: comma-separated, e.g. "q_proj,k_proj,v_proj,o_proj". Empty = all.
    pub fn add_lora(
        &self,
        rank: i64,
        alpha: f64,
        target_layers: &[i64],
        target_modules: &str,
    ) -> Result<i64> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let tl_ptr = if target_layers.is_empty() {
            std::ptr::null()
        } else {
            target_layers.as_ptr()
        };
        let tl_len = target_layers.len() as i64;
        let modules_c = std::ffi::CString::new(target_modules).unwrap();
        let modules_ptr = if target_modules.is_empty() {
            std::ptr::null()
        } else {
            modules_c.as_ptr()
        };
        let id = unsafe { (kh.add_lora)(self.ptr, rank, alpha, tl_ptr, tl_len, modules_ptr) };
        if id < 0 {
            bail!("C++ add_lora failed");
        }
        Ok(id)
    }

    /// Remove a LoRA adapter by ID.
    pub fn remove_lora(&self, adapter_id: i64) -> Result<bool> {
        if adapter_id == 0 {
            return Ok(false);
        }
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let found = unsafe { (kh.remove_lora)(self.ptr, adapter_id) };
        Ok(found != 0)
    }

    /// Restore a dynamic adapter's stable external ID from a checkpoint.
    pub fn set_adapter_id(&self, current_id: i64, requested_id: i64) -> Result<()> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let status = unsafe { (kh.set_adapter_id)(self.ptr, current_id, requested_id) };
        if status != 0 {
            bail!("C++ set_adapter_id failed: {current_id} -> {requested_id}");
        }
        Ok(())
    }

    /// List all active adapter IDs.
    pub fn list_lora(&self) -> Vec<i64> {
        let kh = match get_kernels() {
            Some(k) => k,
            None => return Vec::new(),
        };
        let mut ids = Vec::with_capacity(65);
        if self.lora_count > 0 {
            // ID 0 is the fixed adapter created with the training context.
            ids.push(0);
        }
        let mut dynamic_ids = vec![0i64; 64];
        let count = unsafe { (kh.list_lora)(self.ptr, dynamic_ids.as_mut_ptr(), 64) };
        ids.extend_from_slice(&dynamic_ids[..count as usize]);
        ids
    }

    pub fn get_adapter_lora_tensor(
        &self,
        adapter_id: i64,
        layer: i64,
        module: &str,
        is_b: bool,
    ) -> Option<Tensor> {
        let kh = get_kernels()?;
        let module = std::ffi::CString::new(module).ok()?;
        let ptr = unsafe {
            (kh.get_adapter_lora_tensor)(
                self.ptr,
                adapter_id,
                layer,
                module.as_ptr(),
                if is_b { 1 } else { 0 },
            )
        };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { Tensor::clone_from_ptr(ptr as *mut _) })
    }

    pub fn set_adapter_lora_tensor(
        &self,
        adapter_id: i64,
        layer: i64,
        module: &str,
        is_b: bool,
        tensor: &Tensor,
    ) -> Result<()> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let module = std::ffi::CString::new(module)?;
        let status = unsafe {
            (kh.set_adapter_lora_tensor)(
                self.ptr,
                adapter_id,
                layer,
                module.as_ptr(),
                if is_b { 1 } else { 0 },
                tensor.as_ptr() as *mut c_void,
            )
        };
        if status != 0 {
            bail!(
                "C++ set_adapter_lora_tensor failed for adapter {adapter_id}, layer {layer}, module {module:?}"
            );
        }
        Ok(())
    }

    pub fn get_adapter_optimizer_tensor(
        &self,
        adapter_id: i64,
        layer: i64,
        module: &str,
        is_b: bool,
        is_v: bool,
    ) -> Option<Tensor> {
        let kh = get_kernels()?;
        let module = std::ffi::CString::new(module).ok()?;
        let ptr = unsafe {
            (kh.get_adapter_optimizer_tensor)(
                self.ptr,
                adapter_id,
                layer,
                module.as_ptr(),
                if is_b { 1 } else { 0 },
                if is_v { 1 } else { 0 },
            )
        };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { Tensor::clone_from_ptr(ptr as *mut _) })
    }

    pub fn set_adapter_optimizer_tensor(
        &self,
        adapter_id: i64,
        layer: i64,
        module: &str,
        is_b: bool,
        is_v: bool,
        tensor: &Tensor,
    ) -> Result<()> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let module = std::ffi::CString::new(module)?;
        let status = unsafe {
            (kh.set_adapter_optimizer_tensor)(
                self.ptr,
                adapter_id,
                layer,
                module.as_ptr(),
                if is_b { 1 } else { 0 },
                if is_v { 1 } else { 0 },
                tensor.as_ptr() as *mut c_void,
            )
        };
        if status != 0 {
            bail!(
                "C++ set_adapter_optimizer_tensor failed for adapter {adapter_id}, layer {layer}, module {module:?}"
            );
        }
        Ok(())
    }

    /// Eval step: forward + loss, no backward, no Adam update.
    pub fn eval_step(
        &self,
        input_ids: &Tensor,
        target_mask: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<f64> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let loss = unsafe {
            (kh.eval_step)(
                self.ptr,
                input_ids.as_ptr() as *mut _,
                target_mask.as_ptr() as *mut _,
                attention_mask.as_ptr() as *mut _,
            )
        };
        if loss < 0.0 {
            bail!("C++ eval_step failed");
        }
        Ok(loss)
    }

    /// Get current training step count.
    pub fn get_step_count(&self) -> i64 {
        let kh = match get_kernels() {
            Some(k) => k,
            None => return 0,
        };
        unsafe { (kh.get_step_count)(self.ptr) }
    }

    /// Export Adam optimizer state (m and v vectors).
    /// Returns (m_tensors, v_tensors) — owned copies on CPU.
    pub fn export_optimizer_state(&self) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let count = self.lora_count * 2; // m and v per LoRA param (a+b)
        let mut m_ptrs: Vec<*mut c_void> = vec![std::ptr::null_mut(); count as usize];
        let mut v_ptrs: Vec<*mut c_void> = vec![std::ptr::null_mut(); count as usize];
        let actual = unsafe {
            (kh.export_optimizer)(self.ptr, m_ptrs.as_mut_ptr(), v_ptrs.as_mut_ptr(), count)
        };
        let mut m_tensors = Vec::new();
        let mut v_tensors = Vec::new();
        for i in 0..actual as usize {
            if !m_ptrs[i].is_null() {
                m_tensors.push(unsafe { Tensor::clone_from_ptr(m_ptrs[i] as *mut _) });
            }
            if !v_ptrs[i].is_null() {
                v_tensors.push(unsafe { Tensor::clone_from_ptr(v_ptrs[i] as *mut _) });
            }
        }
        Ok((m_tensors, v_tensors))
    }

    /// Import Adam optimizer state (m and v vectors).
    pub fn import_optimizer_state(
        &self,
        m_tensors: &[Tensor],
        v_tensors: &[Tensor],
    ) -> Result<i64> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let count = m_tensors.len().min(v_tensors.len());
        let m_ptrs: Vec<*mut c_void> = m_tensors
            .iter()
            .map(|t| t.as_ptr() as *mut c_void)
            .collect();
        let v_ptrs: Vec<*mut c_void> = v_tensors
            .iter()
            .map(|t| t.as_ptr() as *mut c_void)
            .collect();
        let imported = unsafe {
            (kh.import_optimizer)(
                self.ptr,
                m_ptrs.as_ptr() as *mut *mut c_void,
                v_ptrs.as_ptr() as *mut *mut c_void,
                count as i64,
            )
        };
        Ok(imported)
    }
}

impl Drop for CppTrainingContext {
    fn drop(&mut self) {
        if let Some(kh) = get_kernels() {
            if !self.ptr.is_null() {
                unsafe { (kh.free_ctx)(self.ptr) };
            }
        }
    }
}
