//! C++ FFI bindings for DeepSeek V4 Flash native training — all-in-C++ path.
//!
//! TrainingContext, train_step, and adapter export all happen in C++.
//! Rust only handles: weight loading, data loading, training loop orchestration.

use std::ffi::c_void;
use std::sync::OnceLock;
use tch::{Kind, Tensor};
use anyhow::{Result, bail};

// ── dlopen ──

type FnCreateCtx = unsafe extern "C" fn(
    *mut *mut c_void, i64, *mut c_void, *mut c_void, *mut c_void,
    *mut c_void, i64, i32,
    f64, f64, f64, f64, f64, i64, f64,
    *mut *mut c_void, i64, *mut c_void,
) -> *mut c_void;
type FnTrainStep = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> f64;
type FnGetLoraCount = unsafe extern "C" fn(*mut c_void) -> i64;
type FnGetLoraA = unsafe extern "C" fn(*mut c_void, i64) -> *mut c_void;
type FnGetLoraB = unsafe extern "C" fn(*mut c_void, i64) -> *mut c_void;
type FnFreeCtx = unsafe extern "C" fn(*mut c_void);
type FnSetCheckpoint = unsafe extern "C" fn(*mut c_void, i32, i64);

#[repr(C)]
pub struct V4LayerConfig {
    pub head_dim: i64,
    pub num_heads: i64,
    pub qk_rope_dim: i64,
    pub o_groups: i64,
    pub kv_lora_rank: i64,
    pub rope_theta: f64,
    pub use_yarn: i32,
    pub yarn_factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub orig_max_pos: i64,
    pub rms_eps: f64,
    pub swiglu_limit: f64,
    pub sliding_window: i64,
    pub n_experts: i64,
    pub top_k: i64,
    pub moe_intermediate: i64,
    pub n_shared_experts: i64,
    pub routed_scaling_factor: f64,
    pub norm_topk_prob: i32,
    pub scoring_func: [u8; 32],
    pub topk_method: [u8; 32],
    pub hc_sinkhorn_iters: i64,
    pub hc_mult: i64,
    pub hc_eps: f64,
    pub index_n_heads: i64,
    pub index_head_dim: i64,
    pub index_topk: i64,
    pub num_hash_layers: i64,
    pub expert_start: i64,
    pub expert_count: i64,
    pub compress_ratio: i64,
    pub has_lora: i32,
}

struct KernelHandles {
    create_ctx: FnCreateCtx,
    train_step: FnTrainStep,
    get_lora_count: FnGetLoraCount,
    get_lora_a: FnGetLoraA,
    get_lora_b: FnGetLoraB,
    free_ctx: FnFreeCtx,
    set_checkpoint: FnSetCheckpoint,
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
    let lib_name = CString::new("libv4_flash_kernels.so").unwrap();
    let handle = libc::dlopen(lib_name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_NOLOAD);
    let handle = if handle.is_null() {
        let h = libc::dlopen(lib_name.as_ptr(), libc::RTLD_LAZY);
        if h.is_null() { return None; }
        h
    } else { handle };

    macro_rules! sym {
        ($name:expr) => {{
            let s = CString::new($name).unwrap();
            let p = libc::dlsym(handle, s.as_ptr());
            if p.is_null() { return None; }
            std::mem::transmute::<*mut c_void, _>(p)
        }};
    }
    Some(KernelHandles {
        create_ctx: sym!("v4_create_training_context"),
        train_step: sym!("v4_train_step"),
        get_lora_count: sym!("v4_get_lora_count"),
        get_lora_a: sym!("v4_get_lora_a"),
        get_lora_b: sym!("v4_get_lora_b"),
        free_ctx: sym!("v4_free_training_context"),
        set_checkpoint: sym!("v4_set_checkpoint"),
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

/// Build weight pointers for all layers.
/// Layout per layer (no HC): [attn_norm, ffn_norm, wq_a, wq_b, wkv, wo_a, wo_b, q_norm, kv_norm, attn_sink,
///   gate, shared_w1, shared_w2, shared_w3, expert_w1[0..N-1], expert_w2[0..N-1], expert_w3[0..N-1]]
/// With HC: + [hc_attn_base, hc_attn_fn, hc_attn_scale, hc_ffn_base, hc_ffn_fn, hc_ffn_scale]
pub fn build_weight_ptrs(
    weights: &std::collections::BTreeMap<String, Tensor>,
    config: &crate::model::V4RuntimeConfig,
    ep_shard: Option<&crate::ep::V4EpShard>,
) -> Vec<*mut c_void> {
    let mut ptrs = Vec::new();
    let local_indices: Vec<usize> = ep_shard
        .map(|s| s.local_expert_indices.clone())
        .unwrap_or_else(|| (0..config.n_routed_experts).collect());

    for layer in 0..config.num_hidden_layers {
        let p = format!("layers.{layer}");
        ptrs.push(get_ptr(weights, &format!("{p}.attn_norm.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.ffn_norm.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.attn.wq_a.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.attn.wq_b.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.attn.wkv.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.attn.wo_a.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.attn.wo_b.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.attn.q_norm.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.attn.kv_norm.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.attn.attn_sink")));
        ptrs.push(get_ptr(weights, &format!("{p}.ffn.gate.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.ffn.shared_experts.w1.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.ffn.shared_experts.w2.weight")));
        ptrs.push(get_ptr(weights, &format!("{p}.ffn.shared_experts.w3.weight")));
        // Local experts
        for &e in &local_indices {
            ptrs.push(get_ptr(weights, &format!("{p}.ffn.experts.{e}.w1.weight")));
        }
        for &e in &local_indices {
            ptrs.push(get_ptr(weights, &format!("{p}.ffn.experts.{e}.w2.weight")));
        }
        for &e in &local_indices {
            ptrs.push(get_ptr(weights, &format!("{p}.ffn.experts.{e}.w3.weight")));
        }
        // HC weights (if this layer has them)
        if crate::hc::HcWeights::exists(weights, layer) {
            for hc_name in &[
                "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
                "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
            ] {
                ptrs.push(get_ptr(weights, &format!("{p}.{hc_name}")));
            }
        }
    }
    ptrs
}

fn fill_str(buf: &mut [u8; 32], s: &str) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(31);
    buf[..n].copy_from_slice(&bytes[..n]);
    buf[n] = 0;
}

pub fn build_layer_configs(
    config: &crate::model::V4RuntimeConfig,
    ep_shard: Option<&crate::ep::V4EpShard>,
    has_lora: bool,
) -> Vec<V4LayerConfig> {
    let expert_start = ep_shard.map(|s| s.expert_start).unwrap_or(0) as i64;
    let expert_count = ep_shard.map(|s| s.experts_per_rank).unwrap_or(config.n_routed_experts) as i64;

    (0..config.num_hidden_layers).map(|layer| {
        let ratio = if layer < config.compress_ratios.len() {
            config.compress_ratios[layer] as i64
        } else {
            0
        };
        let has_hc = config.num_hash_layers > 0 && ratio > 1;

        let mut scoring_buf = [0u8; 32];
        fill_str(&mut scoring_buf, &config.scoring_func);
        let mut topk_buf = [0u8; 32];
        fill_str(&mut topk_buf, &config.topk_method);

        V4LayerConfig {
            head_dim: 512,  // V4 always uses 512
            num_heads: config.num_attention_heads,
            qk_rope_dim: config.qk_rope_head_dim,
            o_groups: config.o_groups,
            kv_lora_rank: config.kv_lora_rank,
            rope_theta: config.rope_theta,
            use_yarn: if config.rope_scaling_type.as_deref() == Some("yarn") { 1 } else { 0 },
            yarn_factor: config.rope_scaling_factor,
            beta_fast: config.rope_beta_fast,
            beta_slow: config.rope_beta_slow,
            orig_max_pos: config.rope_original_max_pos,
            rms_eps: config.rms_norm_eps,
            swiglu_limit: config.swiglu_limit,
            sliding_window: config.sliding_window as i64,
            n_experts: config.n_routed_experts as i64,
            top_k: config.num_experts_per_tok as i64,
            moe_intermediate: config.moe_intermediate_size as i64,
            n_shared_experts: config.n_shared_experts as i64,
            routed_scaling_factor: config.routed_scaling_factor,
            norm_topk_prob: 0,  // V4 uses noaux_tc, not norm_topk_prob
            scoring_func: scoring_buf,
            topk_method: topk_buf,
            hc_sinkhorn_iters: config.hc_sinkhorn_iters as i64,
            hc_mult: config.hc_mult as i64,
            hc_eps: config.hc_eps,
            index_n_heads: config.index_n_heads,
            index_head_dim: config.index_head_dim,
            index_topk: config.index_topk,
            num_hash_layers: config.num_hash_layers as i64,
            expert_start,
            expert_count,
            compress_ratio: ratio,
            has_lora: if has_lora { 1 } else { 0 },
        }
    }).collect()
}

/// Build MTP weight pointers.
pub fn build_mtp_weight_ptrs(
    weights: &std::collections::BTreeMap<String, Tensor>,
    config: &crate::model::V4RuntimeConfig,
) -> (Vec<*mut c_void>, *mut c_void) {
    if config.num_nextn_predict_layers == 0 {
        return (Vec::new(), std::ptr::null_mut());
    }
    // MTP layer 0 weights: norm, hnorm, head, ffn_norm, ffn_w1, ffn_w2, ffn_w3
    let ptrs = vec![
        get_ptr(weights, "mtp.0.norm.weight"),
        get_ptr(weights, "mtp.0.hnorm.weight"),
        get_ptr(weights, "mtp.0.head.weight"),
        get_ptr(weights, "mtp.0.ffn_norm.weight"),
        get_ptr(weights, "mtp.0.ffn.shared_experts.w1.weight"),
        get_ptr(weights, "mtp.0.ffn.shared_experts.w2.weight"),
        get_ptr(weights, "mtp.0.ffn.shared_experts.w3.weight"),
    ];
    let embed_ptr = get_ptr(weights, "embed.weight");
    (ptrs, embed_ptr)
}

/// Opaque training context handle.
pub struct V4CppTrainingContext {
    ptr: *mut c_void,
    lora_count: i64,
}

impl V4CppTrainingContext {
    pub fn new(
        weights: &std::collections::BTreeMap<String, Tensor>,
        config: &crate::model::V4RuntimeConfig,
        compute_kind: Kind,
        lr: f64, beta1: f64, beta2: f64, eps: f64,
        lora_scaling: f64,
        ep_shard: Option<&crate::ep::V4EpShard>,
        has_lora: bool,
    ) -> Result<Self> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("v4 kernels not loaded"))?;
        let mut weight_ptrs = build_weight_ptrs(weights, config, ep_shard);
        let layer_configs = build_layer_configs(config, ep_shard, has_lora);

        let embed_ptr = get_ptr(weights, "embed.weight");
        let final_norm_ptr = get_ptr(weights, "norm.weight");
        let lm_head_ptr = if config.tie_word_embeddings {
            embed_ptr
        } else {
            get_ptr(weights, "head.weight")
        };

        let compute_type = compute_kind as i32;

        let mut mtp_ptrs: Vec<*mut c_void> = if config.num_nextn_predict_layers == 0 {
            Vec::new()
        } else {
            let (ptrs, _) = build_mtp_weight_ptrs(weights, config);
            ptrs
        };
        let mtp_embed_ptr = get_ptr(weights, "embed.weight");

        let wp_ptr = weight_ptrs.as_mut_ptr();
        let wp_len = weight_ptrs.len();
        std::mem::forget(weight_ptrs);
        let lc_ptr = layer_configs.as_ptr() as *mut c_void;
        std::mem::forget(layer_configs);

        let (mtp_ptr, mtp_len) = if mtp_ptrs.is_empty() {
            (std::ptr::null_mut::<*mut c_void>(), 0i64)
        } else {
            (mtp_ptrs.as_mut_ptr(), mtp_ptrs.len() as i64)
        };
        std::mem::forget(mtp_ptrs);

        let ptr = unsafe {
            (kh.create_ctx)(
                wp_ptr, wp_len as i64,
                embed_ptr, final_norm_ptr, lm_head_ptr,
                lc_ptr, config.num_hidden_layers as i64,
                compute_type,
                lora_scaling, lr, beta1, beta2, eps,
                config.vocab_size, config.rms_norm_eps,
                mtp_ptr, mtp_len,
                mtp_embed_ptr,
            )
        };
        if ptr.is_null() {
            bail!("C++ v4_create_training_context returned null");
        }
        let lora_count = unsafe { (kh.get_lora_count)(ptr) };
        Ok(Self { ptr, lora_count })
    }

    pub fn train_step(&self, input_ids: &Tensor, target_mask: &Tensor) -> Result<f64> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let loss = unsafe {
            (kh.train_step)(
                self.ptr,
                input_ids.as_ptr() as *mut _,
                target_mask.as_ptr() as *mut _,
            )
        };
        if loss < 0.0 {
            bail!("C++ v4_train_step failed");
        }
        Ok(loss)
    }

    pub fn get_lora_a(&self, index: i64) -> Option<Tensor> {
        let kh = get_kernels()?;
        let ptr = unsafe { (kh.get_lora_a)(self.ptr, index) };
        if ptr.is_null() { return None; }
        Some(unsafe { Tensor::clone_from_ptr(ptr as *mut _) })
    }

    pub fn get_lora_b(&self, index: i64) -> Option<Tensor> {
        let kh = get_kernels()?;
        let ptr = unsafe { (kh.get_lora_b)(self.ptr, index) };
        if ptr.is_null() { return None; }
        Some(unsafe { Tensor::clone_from_ptr(ptr as *mut _) })
    }

    pub fn lora_count(&self) -> i64 {
        self.lora_count
    }

    pub fn set_checkpoint(&self, enable: bool, group_size: i64) {
        if let Some(kh) = get_kernels() {
            unsafe { (kh.set_checkpoint)(self.ptr, if enable { 1 } else { 0 }, group_size) };
        }
    }
}

impl Drop for V4CppTrainingContext {
    fn drop(&mut self) {
        if let Some(kh) = get_kernels() {
            unsafe { (kh.free_ctx)(self.ptr) };
        }
    }
}
