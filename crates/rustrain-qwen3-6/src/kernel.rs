//! C++ FFI bindings for Qwen3.6 native training — all-in-C++ path.
//!
//! TrainingContext, train_step, and adapter export all happen in C++.
//! Rust only handles: weight loading, data loading, training loop orchestration.

use crate::lora::Qwen36LoraTargetModule;
use crate::pipeline::PipelineStageLayout;
use anyhow::{Context, Result, bail};
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
    i64,
    i64,
    i32,
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
    i32,
) -> *mut c_void;
type FnKernelAbiVersion = unsafe extern "C" fn() -> i64;
type FnTrainStep = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> f64;
type FnTrainMicroStep =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, f64, i32) -> f64;
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PipelineWindowV1 {
    pub struct_size: u32,
    pub version: u32,
    pub window_id: i64,
    pub num_microbatches: i64,
    pub schedule: i32,
    pub num_chunks: i32,
    pub flags: i32,
}
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PipelineTickV1 {
    pub struct_size: u32,
    pub version: u32,
    pub window_id: i64,
    pub forward_mb: i64,
    pub backward_mb: i64,
    pub chunk_id: i32,
    pub phase: i32,
    pub input_ids: *mut c_void,
    pub target_mask: *mut c_void,
    pub attention_mask: *mut c_void,
    pub gradient_scale: f64,
}
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PipelineResultV1 {
    pub struct_size: u32,
    pub version: u32,
    pub status: i32,
    pub completed_fwd: i64,
    pub completed_bwd: i64,
    pub in_flight: i64,
    pub optimizer_step: i64,
    pub loss: f64,
}
type FnPipelineBegin = unsafe extern "C" fn(*mut c_void, *const PipelineWindowV1) -> i32;
type FnPipelineTick =
    unsafe extern "C" fn(*mut c_void, *const PipelineTickV1, *mut PipelineResultV1) -> i32;
type FnPipelineFinish = unsafe extern "C" fn(*mut c_void, i32, *mut PipelineResultV1) -> i32;
type FnPipelineAbort = unsafe extern "C" fn(*mut c_void) -> i32;
type FnTrainMultiLoraSelectedV2 = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *const i64,
    i32,
) -> f64;
type FnTrainMultiLoraSelectedV3 = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *const i64,
    i32,
    *mut f64,
    *mut f64,
    i32,
) -> i32;
type FnEvalStep = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> f64;
type FnHostBatchStep =
    unsafe extern "C" fn(*mut c_void, *const i64, *const i64, *const i64, i64, i64) -> f64;
type FnHostMultiLoraStep = unsafe extern "C" fn(
    *mut c_void,
    *const i64,
    *const i64,
    *const i64,
    i64,
    i64,
    i32,
    i32,
    *const i64,
    i32,
) -> f64;
type FnHostMultiLoraReport = unsafe extern "C" fn(
    *mut c_void,
    *const i64,
    *const i64,
    *const i64,
    i64,
    i64,
    i32,
    i32,
    *const i64,
    i32,
    *mut f64,
    *mut f64,
    i32,
) -> i32;

#[derive(Debug, Clone, PartialEq)]
pub struct MultiLoraLossReport {
    pub aggregate_loss: f64,
    pub adapter_losses: Vec<f64>,
}
type FnGetLoraCount = unsafe extern "C" fn(*mut c_void) -> i64;
type FnGetLoraA = unsafe extern "C" fn(*mut c_void, i64) -> *mut c_void;
type FnGetLoraB = unsafe extern "C" fn(*mut c_void, i64) -> *mut c_void;
type FnSetLoraTensor = unsafe extern "C" fn(*mut c_void, i64, i32, *mut c_void) -> i32;
type FnGetLoraGradAccumulator = unsafe extern "C" fn(*mut c_void, i64, i32) -> *mut c_void;
type FnAbortGradientAccumulation = unsafe extern "C" fn(*mut c_void) -> i32;
type FnGetStepCount = unsafe extern "C" fn(*mut c_void) -> i64;
type FnSetStepCount = unsafe extern "C" fn(*mut c_void, i64) -> i32;
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
) -> i32;
type FnSetCheckpoint = unsafe extern "C" fn(*mut c_void, i32, i64);
type FnSetNcclComm = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, i32, i32);
type FnInitNccl = unsafe extern "C" fn(*mut c_void) -> i32;
type FnInitParallelNccl = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
) -> i32;
type FnSetCudaDevice = unsafe extern "C" fn(i32);
type FnSetBaseTpMlp = unsafe extern "C" fn(*mut c_void, i32) -> i32;
type FnAddLora = unsafe extern "C" fn(*mut c_void, i64, f64, *const i64, i64, *const i8) -> i64;
type FnAddLoraWithOptimizer =
    unsafe extern "C" fn(*mut c_void, i64, f64, *const i64, i64, *const i8, f64) -> i64;
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
type FnGetAdapterStepCount = unsafe extern "C" fn(*mut c_void, i64) -> i64;
type FnSetAdapterStepCount = unsafe extern "C" fn(*mut c_void, i64, i64) -> i32;

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
    pipeline_begin: FnPipelineBegin,
    pipeline_tick: FnPipelineTick,
    pipeline_finish: FnPipelineFinish,
    pipeline_abort: FnPipelineAbort,
    train_multi_lora_selected_v2: FnTrainMultiLoraSelectedV2,
    train_multi_lora_selected_v3: FnTrainMultiLoraSelectedV3,
    eval_step: FnEvalStep,
    train_step_host_i64: FnHostBatchStep,
    train_multi_lora_host_i64: FnHostMultiLoraStep,
    train_multi_lora_host_i64_v2: FnHostMultiLoraReport,
    eval_step_host_i64: FnHostBatchStep,
    get_lora_count: FnGetLoraCount,
    get_lora_a: FnGetLoraA,
    get_lora_b: FnGetLoraB,
    set_lora_tensor: FnSetLoraTensor,
    get_lora_grad_accumulator: FnGetLoraGradAccumulator,
    abort_gradient_accumulation: FnAbortGradientAccumulation,
    get_step_count: FnGetStepCount,
    set_step_count: FnSetStepCount,
    export_optimizer: FnExportOptimizer,
    import_optimizer: FnImportOptimizer,
    free_ctx: FnFreeCtx,
    gemm: FnGemm,
    free_tensor: FnFreeTensor,
    set_mtp_weights: FnSetMtpWeights,
    set_checkpoint: FnSetCheckpoint,
    set_nccl_comm: FnSetNcclComm,
    init_nccl: FnInitNccl,
    init_parallel_nccl: FnInitParallelNccl,
    attach_parallel_nccl_no_sync: FnInitParallelNccl,
    set_cuda_device: FnSetCudaDevice,
    set_base_tp_mlp: FnSetBaseTpMlp,
    add_lora: FnAddLora,
    add_lora_v2: FnAddLora,
    add_lora_with_optimizer: Option<FnAddLoraWithOptimizer>,
    add_lora_for_restore: FnAddLora,
    add_lora_for_restore_with_optimizer: FnAddLoraWithOptimizer,
    remove_lora: FnRemoveLora,
    list_lora: FnListLora,
    get_adapter_lora_tensor: FnGetAdapterLoraTensor,
    set_adapter_lora_tensor: FnSetAdapterLoraTensor,
    set_adapter_id: FnSetAdapterId,
    get_adapter_optimizer_tensor: FnGetAdapterOptimizerTensor,
    set_adapter_optimizer_tensor: FnSetAdapterOptimizerTensor,
    get_adapter_step_count: FnGetAdapterStepCount,
    set_adapter_step_count: FnSetAdapterStepCount,
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
    if abi_version() != 28 {
        return None;
    }
    let add_lora_with_optimizer = {
        let name = CString::new("qwen36_add_lora_with_optimizer").unwrap();
        let symbol = libc::dlsym(handle, name.as_ptr());
        if symbol.is_null() {
            None
        } else {
            Some(std::mem::transmute::<*mut c_void, FnAddLoraWithOptimizer>(
                symbol,
            ))
        }
    };
    Some(KernelHandles {
        create_ctx: sym!("qwen36_create_training_context_v2"),
        train_step: sym!("qwen36_train_step"),
        train_micro_step: sym!("qwen36_train_micro_step"),
        pipeline_begin: sym!("qwen36_pipeline_begin_v1"),
        pipeline_tick: sym!("qwen36_pipeline_tick_v1"),
        pipeline_finish: sym!("qwen36_pipeline_finish_v1"),
        pipeline_abort: sym!("qwen36_pipeline_abort_v1"),
        train_multi_lora_selected_v2: sym!("qwen36_train_multi_lora_selected_v2"),
        train_multi_lora_selected_v3: sym!("qwen36_train_multi_lora_selected_v3"),
        eval_step: sym!("qwen36_eval_step"),
        train_step_host_i64: sym!("qwen36_train_step_host_i64"),
        train_multi_lora_host_i64: sym!("qwen36_train_multi_lora_host_i64"),
        train_multi_lora_host_i64_v2: sym!("qwen36_train_multi_lora_host_i64_v2"),
        eval_step_host_i64: sym!("qwen36_eval_step_host_i64"),
        get_lora_count: sym!("qwen36_get_lora_count"),
        get_lora_a: sym!("qwen36_get_lora_a"),
        get_lora_b: sym!("qwen36_get_lora_b"),
        set_lora_tensor: sym!("qwen36_set_lora_tensor"),
        get_lora_grad_accumulator: sym!("qwen36_get_lora_grad_accumulator"),
        abort_gradient_accumulation: sym!("qwen36_abort_gradient_accumulation"),
        get_step_count: sym!("qwen36_get_step_count"),
        set_step_count: sym!("qwen36_set_step_count"),
        export_optimizer: sym!("qwen36_export_optimizer_state"),
        import_optimizer: sym!("qwen36_import_optimizer_state"),
        free_ctx: sym!("qwen36_free_training_context"),
        gemm: sym!("qwen36_gemm"),
        free_tensor: sym!("qwen36_free_tensor"),
        set_mtp_weights: sym!("qwen36_set_mtp_weights"),
        set_checkpoint: sym!("qwen36_set_checkpoint"),
        set_nccl_comm: sym!("qwen36_set_nccl_comm"),
        init_nccl: sym!("qwen36_init_nccl"),
        init_parallel_nccl: sym!("qwen36_init_parallel_nccl_v2"),
        attach_parallel_nccl_no_sync: sym!("qwen36_attach_parallel_nccl_no_sync_v2"),
        set_cuda_device: sym!("qwen36_set_cuda_device"),
        set_base_tp_mlp: sym!("qwen36_set_base_tp_mlp"),
        add_lora: sym!("qwen36_add_lora"),
        add_lora_v2: sym!("qwen36_add_lora_v2"),
        add_lora_with_optimizer,
        add_lora_for_restore: sym!("qwen36_add_lora_for_restore"),
        add_lora_for_restore_with_optimizer: sym!("qwen36_add_lora_for_restore_with_optimizer"),
        remove_lora: sym!("qwen36_remove_lora"),
        list_lora: sym!("qwen36_list_lora"),
        get_adapter_lora_tensor: sym!("qwen36_get_adapter_lora_tensor"),
        set_adapter_lora_tensor: sym!("qwen36_set_adapter_lora_tensor"),
        set_adapter_id: sym!("qwen36_set_adapter_id"),
        get_adapter_optimizer_tensor: sym!("qwen36_get_adapter_optimizer_tensor"),
        set_adapter_optimizer_tensor: sym!("qwen36_set_adapter_optimizer_tensor"),
        get_adapter_step_count: sym!("qwen36_get_adapter_step_count"),
        set_adapter_step_count: sym!("qwen36_set_adapter_step_count"),
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

/// Return the rank-local frozen vocabulary shard for TP, or `None` when the
/// tensor is not an embedding or LM-head weight. This runs during CPU loading.
pub fn shard_vocab_weight_for_tp(
    name: &str,
    tensor: &Tensor,
    vocab_size: i64,
    tp_size: usize,
    tp_rank: usize,
) -> Result<Option<Tensor>> {
    if !name.ends_with("embed_tokens.weight") && !name.ends_with("lm_head.weight") {
        return Ok(None);
    }
    if tp_size <= 1 || tp_rank >= tp_size {
        bail!("invalid vocabulary TP shard: tp_rank={tp_rank}, tp_size={tp_size}");
    }
    if vocab_size <= 0 || vocab_size % tp_size as i64 != 0 {
        bail!("vocab_size={vocab_size} is not divisible by TP_SIZE={tp_size}");
    }
    let shape = tensor.size();
    if shape.len() != 2 || shape[0] != vocab_size {
        bail!("vocabulary TP weight {name} must have shape [{vocab_size}, hidden], got {shape:?}");
    }
    let local_vocab_size = vocab_size / tp_size as i64;
    Ok(Some(
        tensor
            .narrow(0, tp_rank as i64 * local_vocab_size, local_vocab_size)
            .contiguous(),
    ))
}

/// Return the rank-local frozen dense MLP shard for TP, or `None` when the
/// tensor is not one of gate/up/down. This runs during CPU weight loading.
pub fn shard_dense_mlp_weight_for_tp(
    name: &str,
    tensor: &Tensor,
    tp_size: usize,
    tp_rank: usize,
) -> Result<Option<Tensor>> {
    let dim = if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight") {
        0
    } else if name.ends_with(".mlp.down_proj.weight") {
        1
    } else {
        return Ok(None);
    };
    if tp_size <= 1 || tp_rank >= tp_size {
        bail!("invalid dense MLP TP shard: tp_rank={tp_rank}, tp_size={tp_size}");
    }
    let full = *tensor
        .size()
        .get(dim as usize)
        .ok_or_else(|| anyhow::anyhow!("base TP MLP weight {name} has no dimension {dim}"))?;
    if full <= 0 || full % tp_size as i64 != 0 {
        bail!(
            "base TP MLP weight {name} dimension {dim}={full} is not divisible by TP_SIZE={tp_size}"
        );
    }
    let shard = full / tp_size as i64;
    let start = tp_rank as i64 * shard;
    Ok(Some(tensor.narrow(dim, start, shard).contiguous()))
}

/// Return the rank-local frozen MoE MLP shard for expert tensor parallelism.
/// Routed expert tensors may already be narrowed on their leading EP axis.
/// The packed gate/up layout is `[gate_all | up_all]`, so each half must be
/// sliced independently before the local halves are concatenated.
pub fn shard_moe_mlp_weight_for_tp(
    name: &str,
    tensor: &Tensor,
    tp_size: usize,
    tp_rank: usize,
) -> Result<Option<Tensor>> {
    let is_shared_gate_up = name.ends_with(".mlp.shared_expert.gate_proj.weight")
        || name.ends_with(".mlp.shared_expert.up_proj.weight");
    let is_shared_down = name.ends_with(".mlp.shared_expert.down_proj.weight");
    let is_expert_gate_up = name.ends_with(".mlp.experts.gate_up_proj");
    let is_expert_down = name.ends_with(".mlp.experts.down_proj");
    if !(is_shared_gate_up || is_shared_down || is_expert_gate_up || is_expert_down) {
        return Ok(None);
    }
    if tp_size <= 1 || tp_rank >= tp_size {
        bail!("invalid MoE MLP TP shard: tp_rank={tp_rank}, tp_size={tp_size}");
    }
    let tp_size_i64 = tp_size as i64;
    let rank = tp_rank as i64;
    let shape = tensor.size();

    if is_expert_gate_up {
        if shape.len() != 3 || shape[1] <= 0 || shape[1] % 2 != 0 {
            bail!(
                "packed expert gate/up TP weight {name} must have shape [experts, 2*intermediate, hidden], got {shape:?}"
            );
        }
        let intermediate = shape[1] / 2;
        if intermediate % tp_size_i64 != 0 {
            bail!(
                "packed expert gate/up intermediate={intermediate} is not divisible by TP_SIZE={tp_size}"
            );
        }
        let local = intermediate / tp_size_i64;
        let gate = tensor.narrow(1, rank * local, local);
        let up = tensor.narrow(1, intermediate + rank * local, local);
        return Ok(Some(Tensor::cat(&[&gate, &up], 1).contiguous()));
    }

    let dim = if is_shared_gate_up {
        if shape.len() != 2 {
            bail!("shared expert gate/up TP weight {name} must be a matrix, got {shape:?}");
        }
        0
    } else if is_shared_down {
        if shape.len() != 2 {
            bail!("shared expert down TP weight {name} must be a matrix, got {shape:?}");
        }
        1
    } else {
        if shape.len() != 3 {
            bail!("routed expert down TP weight {name} must be rank 3, got {shape:?}");
        }
        2
    };
    let full = shape[dim];
    if full <= 0 || full % tp_size_i64 != 0 {
        bail!(
            "MoE MLP TP weight {name} dimension {dim}={full} is not divisible by TP_SIZE={tp_size}"
        );
    }
    let local = full / tp_size_i64;
    Ok(Some(
        tensor.narrow(dim as i64, rank * local, local).contiguous(),
    ))
}

/// Return the rank-local frozen full-attention shard for TP. Q/K/V own
/// contiguous output-head bundles; O owns the matching input columns.
pub fn shard_full_attention_weight_for_tp(
    name: &str,
    tensor: &Tensor,
    tp_size: usize,
    tp_rank: usize,
) -> Result<Option<Tensor>> {
    let dim = if name.ends_with(".self_attn.q_proj.weight")
        || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight")
    {
        0
    } else if name.ends_with(".self_attn.o_proj.weight") {
        1
    } else {
        return Ok(None);
    };
    if tp_size <= 1 || tp_rank >= tp_size {
        bail!("invalid full-attention TP shard: tp_rank={tp_rank}, tp_size={tp_size}");
    }
    let full = *tensor
        .size()
        .get(dim as usize)
        .ok_or_else(|| anyhow::anyhow!("base TP attention weight {name} has no dimension {dim}"))?;
    if full <= 0 || full % tp_size as i64 != 0 {
        bail!(
            "base TP attention weight {name} dimension {dim}={full} is not divisible by TP_SIZE={tp_size}"
        );
    }
    let shard = full / tp_size as i64;
    Ok(Some(
        tensor
            .narrow(dim, tp_rank as i64 * shard, shard)
            .contiguous(),
    ))
}

/// Return the rank-local frozen GDN shard for TP. Q/K/V use the model's flat
/// `[Q_all | K_all | V_all]` layout, so QKV and depthwise-conv tensors must be
/// sliced segment by segment before being packed into the local flat layout.
#[allow(clippy::too_many_arguments)]
pub fn shard_linear_attention_weight_for_tp(
    name: &str,
    tensor: &Tensor,
    tp_size: usize,
    tp_rank: usize,
    num_k_heads: i64,
    key_dim: i64,
    num_v_heads: i64,
    val_dim: i64,
) -> Result<Option<Tensor>> {
    let is_qkv = name.ends_with(".linear_attn.in_proj_qkv.weight");
    let is_conv = name.ends_with(".linear_attn.conv1d.weight");
    let is_z = name.ends_with(".linear_attn.in_proj_z.weight");
    let is_ab = name.ends_with(".linear_attn.in_proj_a.weight")
        || name.ends_with(".linear_attn.in_proj_b.weight");
    let is_head = name.ends_with(".linear_attn.A_log") || name.ends_with(".linear_attn.dt_bias");
    let is_out = name.ends_with(".linear_attn.out_proj.weight");
    if !(is_qkv || is_conv || is_z || is_ab || is_head || is_out) {
        return Ok(None);
    }
    if tp_size <= 1 || tp_rank >= tp_size {
        bail!("invalid linear-attention TP shard: tp_rank={tp_rank}, tp_size={tp_size}");
    }
    let tp_size_i64 = tp_size as i64;
    if num_k_heads <= 0
        || num_v_heads <= 0
        || key_dim <= 0
        || val_dim <= 0
        || num_v_heads % num_k_heads != 0
        || num_k_heads % tp_size_i64 != 0
        || num_v_heads % tp_size_i64 != 0
    {
        bail!(
            "linear-attention heads/dimensions must preserve value-head groups and be divisible by TP_SIZE={tp_size}: k_heads={num_k_heads}, v_heads={num_v_heads}, key_dim={key_dim}, val_dim={val_dim}"
        );
    }

    let q_total = num_k_heads * key_dim;
    let v_total = num_v_heads * val_dim;
    let qkv_total = q_total * 2 + v_total;
    let local_q = q_total / tp_size_i64;
    let local_v = v_total / tp_size_i64;
    let local_heads = num_v_heads / tp_size_i64;
    let rank = tp_rank as i64;
    let shape = tensor.size();

    let require_axis = |axis: usize, expected: i64| -> Result<()> {
        let actual = shape.get(axis).copied().ok_or_else(|| {
            anyhow::anyhow!("base TP linear-attention weight {name} has no dimension {axis}")
        })?;
        if actual != expected {
            bail!(
                "base TP linear-attention weight {name} dimension {axis}={actual}, expected {expected}"
            );
        }
        Ok(())
    };

    if is_qkv || is_conv {
        require_axis(0, qkv_total)?;
        if is_qkv && shape.len() != 2 {
            bail!("base TP linear-attention QKV weight {name} must be rank 2");
        }
        if is_conv && (shape.len() != 3 || shape[1] != 1) {
            bail!(
                "base TP linear-attention depthwise conv weight {name} must have shape [channels, 1, kernel]"
            );
        }
        let q = tensor.narrow(0, rank * local_q, local_q);
        let k = tensor.narrow(0, q_total + rank * local_q, local_q);
        let v = tensor.narrow(0, q_total * 2 + rank * local_v, local_v);
        return Ok(Some(Tensor::cat(&[&q, &k, &v], 0).contiguous()));
    }
    if is_z {
        require_axis(0, v_total)?;
        return Ok(Some(tensor.narrow(0, rank * local_v, local_v).contiguous()));
    }
    if is_ab || is_head {
        require_axis(0, num_v_heads)?;
        return Ok(Some(
            tensor
                .narrow(0, rank * local_heads, local_heads)
                .contiguous(),
        ));
    }
    require_axis(1, v_total)?;
    Ok(Some(tensor.narrow(1, rank * local_v, local_v).contiguous()))
}

pub fn build_weight_ptrs(
    weights: &std::collections::BTreeMap<String, Tensor>,
    config: &crate::config::Qwen36RuntimeConfig,
) -> Vec<*mut c_void> {
    let stage = PipelineStageLayout::full(config.num_hidden_layers)
        .expect("a full-model pipeline layout is always valid");
    build_weight_ptrs_for_stage(weights, config, &stage)
}

pub fn build_weight_ptrs_for_stage(
    weights: &std::collections::BTreeMap<String, Tensor>,
    config: &crate::config::Qwen36RuntimeConfig,
    stage: &PipelineStageLayout,
) -> Vec<*mut c_void> {
    let p = &config.weight_prefix;
    let mut ptrs = Vec::new();
    for layer in stage.layer_range.clone() {
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
    let stage = PipelineStageLayout::full(config.num_hidden_layers)
        .expect("a full-model pipeline layout is always valid");
    build_layer_configs_for_stage(config, expert_start, expert_count, &stage)
}

pub fn build_layer_configs_for_stage(
    config: &crate::config::Qwen36RuntimeConfig,
    expert_start: usize,
    expert_count: usize,
    stage: &PipelineStageLayout,
) -> Vec<CppLayerConfig> {
    stage
        .layer_range
        .clone()
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
        base_tp_attention: bool,
        base_tp_mlp: bool,
        vocab_parallel: bool,
        data_parallel: bool,
        expert_parallel: bool,
        target_layers: &[usize],
        target_modules: &[Qwen36LoraTargetModule],
        expert_start: usize,
        expert_count: usize,
    ) -> Result<Self> {
        let stage = PipelineStageLayout::full(config.num_hidden_layers)?;
        Self::new_for_stage(
            weights,
            config,
            &stage,
            compute_kind,
            lr,
            beta1,
            beta2,
            eps,
            lora_scaling,
            lora_rank,
            base_tp_attention,
            base_tp_mlp,
            vocab_parallel,
            data_parallel,
            expert_parallel,
            target_layers,
            target_modules,
            expert_start,
            expert_count,
        )
    }

    /// Create a context that owns exactly one contiguous pipeline stage.
    /// Target layers remain global IDs and are mapped to local slots by C++.
    #[allow(clippy::too_many_arguments)]
    pub fn new_for_stage(
        weights: &std::collections::BTreeMap<String, Tensor>,
        config: &crate::config::Qwen36RuntimeConfig,
        stage: &PipelineStageLayout,
        compute_kind: Kind,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        lora_scaling: f64,
        lora_rank: i64,
        base_tp_attention: bool,
        base_tp_mlp: bool,
        vocab_parallel: bool,
        data_parallel: bool,
        expert_parallel: bool,
        target_layers: &[usize],
        target_modules: &[Qwen36LoraTargetModule],
        expert_start: usize,
        expert_count: usize,
    ) -> Result<Self> {
        if stage.global_num_layers != config.num_hidden_layers {
            bail!(
                "pipeline layout has {} global layers, but model config has {}",
                stage.global_num_layers,
                config.num_hidden_layers
            );
        }
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let weight_ptrs = build_weight_ptrs_for_stage(weights, config, stage);
        let layer_configs =
            build_layer_configs_for_stage(config, expert_start, expert_count, stage);

        let embedding_weight_ptr = get_ptr(
            weights,
            &format!("{}embed_tokens.weight", config.weight_prefix),
        );
        let embed_ptr = if stage.is_first() {
            embedding_weight_ptr
        } else {
            std::ptr::null_mut()
        };
        let final_norm_ptr = if stage.is_last() {
            get_ptr(weights, &format!("{}norm.weight", config.weight_prefix))
        } else {
            std::ptr::null_mut()
        };
        let lm_head_ptr = if !stage.is_last() {
            std::ptr::null_mut()
        } else if config.tie_word_embeddings {
            // Tied models duplicate the frozen vocabulary shard on the two
            // boundary stages; the last stage receives it only as LM head.
            embedding_weight_ptr
        } else {
            get_ptr(weights, "lm_head.weight")
        };

        let compute_type = compute_kind as i32;

        // The C++ constructor copies these pointer/config arrays into its own
        // vectors before returning. The tensors they point to remain owned by
        // the Rust session for the full context lifetime.
        let wp_ptr = weight_ptrs.as_ptr() as *mut *mut c_void;
        let wp_len = weight_ptrs.len();
        let lc_ptr = layer_configs.as_ptr() as *mut c_void;

        // The constructor also consumes the target layer array synchronously.
        let target_i64: Vec<i64> = target_layers.iter().map(|&x| x as i64).collect();
        let tl_ptr = if target_i64.is_empty() {
            std::ptr::null()
        } else {
            target_i64.as_ptr()
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
                stage.local_num_layers() as i64,
                stage.layer_range.start as i64,
                stage.global_num_layers as i64,
                stage.native_flags(),
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
                i32::from(base_tp_attention)
                    | (i32::from(data_parallel) << 1)
                    | (i32::from(vocab_parallel) << 2)
                    | (i32::from(expert_parallel) << 3)
                    | (i32::from(base_tp_mlp) << 4),
            )
        };
        if ptr.is_null() {
            bail!("C++ create_training_context returned null");
        }
        let base_tp_status = unsafe { (kh.set_base_tp_mlp)(ptr, i32::from(base_tp_mlp)) };
        if base_tp_status != 0 {
            unsafe { (kh.free_ctx)(ptr) };
            bail!("C++ base dense MLP TP configuration failed");
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

    /// Open a fixed-shape PP2/1F1B pipeline window. The native side owns all
    /// activation/gradient slots until `pipeline_finish_v1` or abort.
    pub fn pipeline_begin_v1(&self, window_id: i64, num_microbatches: i64) -> Result<()> {
        if window_id < 0 {
            bail!("pipeline window id must be non-negative");
        }
        if num_microbatches <= 0 {
            bail!("pipeline window must contain at least one microbatch");
        }
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let spec = PipelineWindowV1 {
            struct_size: std::mem::size_of::<PipelineWindowV1>() as u32,
            version: 1,
            window_id,
            num_microbatches,
            schedule: 0,
            num_chunks: 1,
            flags: 0,
        };
        let status = unsafe { (kh.pipeline_begin)(self.ptr, &spec) };
        if status != 0 {
            bail!("C++ pipeline window begin failed with status {status}");
        }
        Ok(())
    }

    /// Advance a pipeline window by one canonical non-interleaved 1F1B tick.
    pub fn pipeline_tick_v1(
        &self,
        window_id: i64,
        forward_mb: Option<i64>,
        backward_mb: Option<i64>,
        input_ids: Option<&Tensor>,
        target_mask: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        gradient_scale: f64,
    ) -> Result<PipelineResultV1> {
        if !gradient_scale.is_finite() || gradient_scale <= 0.0 {
            bail!("pipeline gradient scale must be finite and positive");
        }
        if forward_mb.is_some() != input_ids.is_some()
            || forward_mb.is_some() != target_mask.is_some()
            || forward_mb.is_some() != attention_mask.is_some()
        {
            bail!("pipeline forward microbatch requires all input tensors");
        }
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let tick = PipelineTickV1 {
            struct_size: std::mem::size_of::<PipelineTickV1>() as u32,
            version: 1,
            window_id,
            forward_mb: forward_mb.unwrap_or(-1),
            backward_mb: backward_mb.unwrap_or(-1),
            chunk_id: 0,
            phase: match (forward_mb, backward_mb) {
                (Some(0), _) => 0,
                (Some(_), _) => 1,
                (None, Some(_)) => 2,
                (None, None) => 2,
            },
            input_ids: input_ids
                .map(|tensor| tensor.as_ptr() as *mut c_void)
                .unwrap_or(std::ptr::null_mut()),
            target_mask: target_mask
                .map(|tensor| tensor.as_ptr() as *mut c_void)
                .unwrap_or(std::ptr::null_mut()),
            attention_mask: attention_mask
                .map(|tensor| tensor.as_ptr() as *mut c_void)
                .unwrap_or(std::ptr::null_mut()),
            gradient_scale,
        };
        let mut result = PipelineResultV1 {
            struct_size: std::mem::size_of::<PipelineResultV1>() as u32,
            version: 1,
            status: 0,
            completed_fwd: 0,
            completed_bwd: 0,
            in_flight: 0,
            optimizer_step: 0,
            loss: 0.0,
        };
        let status = unsafe { (kh.pipeline_tick)(self.ptr, &tick, &mut result) };
        if status != 0 || result.status != 0 {
            bail!("C++ pipeline window tick failed with status {status}");
        }
        Ok(result)
    }

    /// Finish a pipeline window and apply its single accumulated optimizer step.
    pub fn pipeline_finish_v1(&self) -> Result<PipelineResultV1> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let mut result = PipelineResultV1 {
            struct_size: std::mem::size_of::<PipelineResultV1>() as u32,
            version: 1,
            status: 0,
            completed_fwd: 0,
            completed_bwd: 0,
            in_flight: 0,
            optimizer_step: 0,
            loss: 0.0,
        };
        let status = unsafe { (kh.pipeline_finish)(self.ptr, 1, &mut result) };
        if status != 0 || result.status != 0 {
            bail!("C++ pipeline window finish failed with status {status}");
        }
        Ok(result)
    }

    /// Abort a pipeline window and discard any accumulated gradients.
    pub fn pipeline_abort_v1(&self) -> Result<()> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let status = unsafe { (kh.pipeline_abort)(self.ptr) };
        if status != 0 {
            bail!("C++ pipeline window abort failed with status {status}");
        }
        Ok(())
    }

    /// Train every live dynamic adapter, including heterogeneous signatures.
    /// The rank argument is retained for source compatibility with ABI22 callers.
    pub fn train_multi_lora(
        &self,
        input_ids: &Tensor,
        target_mask: &Tensor,
        attention_mask: &Tensor,
        n_total: i32,
        lora_rank: i32,
    ) -> Result<f64> {
        if n_total <= 0 {
            bail!("n_total must be positive, got {n_total}");
        }
        let adapter_ids = self.list_dynamic_lora();
        if adapter_ids.len() != n_total as usize {
            bail!(
                "live adapter count {} does not match n_total={n_total}",
                adapter_ids.len()
            );
        }
        self.train_multi_lora_selected(
            input_ids,
            target_mask,
            attention_mask,
            &adapter_ids,
            lora_rank,
        )
    }

    /// Train only the requested dynamic adapters. The native wrapper scopes
    /// its registry to these IDs so unselected tenants keep their parameters,
    /// gradients, and optimizer clocks untouched.
    pub fn train_multi_lora_selected(
        &self,
        input_ids: &Tensor,
        target_mask: &Tensor,
        attention_mask: &Tensor,
        adapter_ids: &[i64],
        _lora_rank: i32,
    ) -> Result<f64> {
        if adapter_ids.is_empty() {
            bail!("selected multi-LoRA requires at least one adapter ID");
        }
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let loss = unsafe {
            (kh.train_multi_lora_selected_v2)(
                self.ptr,
                input_ids.as_ptr() as *mut _,
                target_mask.as_ptr() as *mut _,
                attention_mask.as_ptr() as *mut _,
                adapter_ids.as_ptr(),
                i32::try_from(adapter_ids.len()).context("selected adapter count exceeds i32")?,
            )
        };
        if loss < 0.0 {
            bail!("C++ train_multi_lora_selected_v2 failed");
        }
        Ok(loss)
    }

    /// Train selected adapters once and return globally normalized losses in
    /// the same order as `adapter_ids`.
    pub fn train_multi_lora_selected_report(
        &self,
        input_ids: &Tensor,
        target_mask: &Tensor,
        attention_mask: &Tensor,
        adapter_ids: &[i64],
    ) -> Result<MultiLoraLossReport> {
        if adapter_ids.is_empty() {
            bail!("selected multi-LoRA loss report requires adapter IDs");
        }
        let adapter_count =
            i32::try_from(adapter_ids.len()).context("selected adapter count exceeds i32")?;
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let report = kh.train_multi_lora_selected_v3;
        let mut aggregate_loss = f64::NAN;
        let mut adapter_losses = vec![f64::NAN; adapter_ids.len()];
        let status = unsafe {
            report(
                self.ptr,
                input_ids.as_ptr() as *mut _,
                target_mask.as_ptr() as *mut _,
                attention_mask.as_ptr() as *mut _,
                adapter_ids.as_ptr(),
                adapter_count,
                &mut aggregate_loss,
                adapter_losses.as_mut_ptr(),
                adapter_count,
            )
        };
        if status != 0
            || !aggregate_loss.is_finite()
            || aggregate_loss < 0.0
            || adapter_losses
                .iter()
                .any(|loss| !loss.is_finite() || *loss < 0.0)
        {
            bail!("C++ train_multi_lora_selected_v3 failed");
        }
        Ok(MultiLoraLossReport {
            aggregate_loss,
            adapter_losses,
        })
    }

    /// Run one complete fixed-LoRA step from a borrowed host int64 batch.
    /// The native entry owns validation, H2D copies, forward/backward, and Adam.
    pub fn train_step_host_i64(
        &self,
        input_ids: &[i64],
        target_mask: &[i64],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<f64> {
        validate_host_batch(input_ids, target_mask, attention_mask, batch_size, seq_len)?;
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let loss = unsafe {
            (kh.train_step_host_i64)(
                self.ptr,
                input_ids.as_ptr(),
                target_mask.as_ptr(),
                attention_mask.as_ptr(),
                i64::try_from(batch_size).context("host batch_size exceeds i64")?,
                i64::try_from(seq_len).context("host seq_len exceeds i64")?,
            )
        };
        if loss < 0.0 {
            bail!("C++ train_step_host_i64 failed");
        }
        Ok(loss)
    }

    /// Run one complete dynamic multi-LoRA step from a borrowed host batch.
    pub fn train_multi_lora_host_i64(
        &self,
        input_ids: &[i64],
        target_mask: &[i64],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
        n_total: i32,
        lora_rank: i32,
        adapter_ids: &[i64],
    ) -> Result<f64> {
        validate_host_batch(input_ids, target_mask, attention_mask, batch_size, seq_len)?;
        if n_total <= 0 {
            bail!("n_total must be positive, got {n_total}");
        }
        if !adapter_ids.is_empty() && adapter_ids.len() != n_total as usize {
            bail!(
                "selected adapter count {} does not match n_total={n_total}",
                adapter_ids.len()
            );
        }
        let adapter_count =
            i32::try_from(adapter_ids.len()).context("selected adapter count exceeds i32")?;
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let loss = unsafe {
            (kh.train_multi_lora_host_i64)(
                self.ptr,
                input_ids.as_ptr(),
                target_mask.as_ptr(),
                attention_mask.as_ptr(),
                i64::try_from(batch_size).context("host batch_size exceeds i64")?,
                i64::try_from(seq_len).context("host seq_len exceeds i64")?,
                n_total,
                lora_rank,
                if adapter_ids.is_empty() {
                    std::ptr::null()
                } else {
                    adapter_ids.as_ptr()
                },
                adapter_count,
            )
        };
        if loss < 0.0 {
            bail!("C++ train_multi_lora_host_i64 failed");
        }
        Ok(loss)
    }

    /// Borrow a host batch for selected adapters and return one globally
    /// normalized loss per adapter alongside the scalar entry point.
    pub fn train_multi_lora_host_i64_report(
        &self,
        input_ids: &[i64],
        target_mask: &[i64],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
        n_total: i32,
        lora_rank: i32,
        adapter_ids: &[i64],
    ) -> Result<MultiLoraLossReport> {
        validate_host_batch(input_ids, target_mask, attention_mask, batch_size, seq_len)?;
        if n_total <= 0 || adapter_ids.len() != n_total as usize {
            bail!("multi-LoRA loss report requires n_total positive selected adapter IDs");
        }
        let adapter_count =
            i32::try_from(adapter_ids.len()).context("selected adapter count exceeds i32")?;
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let report = kh.train_multi_lora_host_i64_v2;
        let mut aggregate_loss = f64::NAN;
        let mut adapter_losses = vec![f64::NAN; adapter_ids.len()];
        let status = unsafe {
            report(
                self.ptr,
                input_ids.as_ptr(),
                target_mask.as_ptr(),
                attention_mask.as_ptr(),
                i64::try_from(batch_size).context("host batch_size exceeds i64")?,
                i64::try_from(seq_len).context("host seq_len exceeds i64")?,
                n_total,
                lora_rank,
                adapter_ids.as_ptr(),
                adapter_count,
                &mut aggregate_loss,
                adapter_losses.as_mut_ptr(),
                adapter_count,
            )
        };
        if status != 0
            || !aggregate_loss.is_finite()
            || aggregate_loss < 0.0
            || adapter_losses
                .iter()
                .any(|loss| !loss.is_finite() || *loss < 0.0)
        {
            bail!("C++ train_multi_lora_host_i64_v2 failed");
        }
        Ok(MultiLoraLossReport {
            aggregate_loss,
            adapter_losses,
        })
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

    /// Inspect the native FP32 gradient accumulator for one fixed LoRA slot.
    /// The returned tensor is a shallow handle owned by the C++ context.
    pub fn get_lora_gradient_accumulator(&self, index: i64, is_b: bool) -> Option<Tensor> {
        let kh = get_kernels()?;
        let ptr =
            unsafe { (kh.get_lora_grad_accumulator)(self.ptr, index, if is_b { 1 } else { 0 }) };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { Tensor::clone_from_ptr(ptr as *mut _) })
    }

    /// Abort an incomplete micro-batch window without changing parameters,
    /// Adam state, or optimizer clocks. Safe to call when no window is active.
    pub fn abort_gradient_accumulation(&self) -> Result<()> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let status = unsafe { (kh.abort_gradient_accumulation)(self.ptr) };
        if status != 0 {
            bail!("C++ abort_gradient_accumulation failed");
        }
        Ok(())
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

        let status = unsafe {
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
            )
        };
        if status != 0 {
            bail!("C++ set_mtp_weights rejected the requested context state");
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

    /// Initialize the orthogonal TP, CP, EP, DP, and PP process grid.
    pub fn init_parallel_nccl(
        &self,
        rank: usize,
        world_size: usize,
        tp_rank: usize,
        tp_size: usize,
        tp_color: usize,
        cp_rank: usize,
        cp_size: usize,
        cp_color: usize,
        ep_rank: usize,
        ep_size: usize,
        ep_color: usize,
        dp_rank: usize,
        dp_size: usize,
        dp_color: usize,
        pp_rank: usize,
        pp_size: usize,
        pp_color: usize,
    ) -> Result<()> {
        let values = [
            rank, world_size, tp_rank, tp_size, tp_color, cp_rank, cp_size, cp_color, ep_rank,
            ep_size, ep_color, dp_rank, dp_size, dp_color, pp_rank, pp_size, pp_color,
        ]
        .map(|value| i32::try_from(value).context("parallel topology exceeds i32"));
        let [
            rank,
            world_size,
            tp_rank,
            tp_size,
            tp_color,
            cp_rank,
            cp_size,
            cp_color,
            ep_rank,
            ep_size,
            ep_color,
            dp_rank,
            dp_size,
            dp_color,
            pp_rank,
            pp_size,
            pp_color,
        ] = values;
        let kh = get_kernels().expect("kernels not loaded");
        let status = unsafe {
            (kh.init_parallel_nccl)(
                self.ptr,
                rank?,
                world_size?,
                tp_rank?,
                tp_size?,
                tp_color?,
                cp_rank?,
                cp_size?,
                cp_color?,
                ep_rank?,
                ep_size?,
                ep_color?,
                dp_rank?,
                dp_size?,
                dp_color?,
                pp_rank?,
                pp_size?,
                pp_color?,
            )
        };
        if status != 0 {
            bail!("C++ parallel NCCL init failed (code {status})");
        }
        Ok(())
    }

    /// Attach a checkpoint shadow context to process-cached communicators.
    /// This deliberately skips parameter broadcasts because restore replaces
    /// every active LoRA/Adam tensor before the context can become live.
    pub fn attach_parallel_nccl_no_sync(
        &self,
        rank: usize,
        world_size: usize,
        tp_rank: usize,
        tp_size: usize,
        tp_color: usize,
        cp_rank: usize,
        cp_size: usize,
        cp_color: usize,
        ep_rank: usize,
        ep_size: usize,
        ep_color: usize,
        dp_rank: usize,
        dp_size: usize,
        dp_color: usize,
        pp_rank: usize,
        pp_size: usize,
        pp_color: usize,
    ) -> Result<()> {
        let values = [
            rank, world_size, tp_rank, tp_size, tp_color, cp_rank, cp_size, cp_color, ep_rank,
            ep_size, ep_color, dp_rank, dp_size, dp_color, pp_rank, pp_size, pp_color,
        ]
        .map(|value| i32::try_from(value).context("parallel topology exceeds i32"));
        let [
            rank,
            world_size,
            tp_rank,
            tp_size,
            tp_color,
            cp_rank,
            cp_size,
            cp_color,
            ep_rank,
            ep_size,
            ep_color,
            dp_rank,
            dp_size,
            dp_color,
            pp_rank,
            pp_size,
            pp_color,
        ] = values;
        let kh = get_kernels().expect("kernels not loaded");
        let status = unsafe {
            (kh.attach_parallel_nccl_no_sync)(
                self.ptr,
                rank?,
                world_size?,
                tp_rank?,
                tp_size?,
                tp_color?,
                cp_rank?,
                cp_size?,
                cp_color?,
                ep_rank?,
                ep_size?,
                ep_color?,
                dp_rank?,
                dp_size?,
                dp_color?,
                pp_rank?,
                pp_size?,
                pp_color?,
            )
        };
        if status != 0 {
            bail!("C++ parallel NCCL restore attach failed (code {status})");
        }
        Ok(())
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
        let id = unsafe { (kh.add_lora_v2)(self.ptr, rank, alpha, tl_ptr, tl_len, modules_ptr) };
        if id < 0 {
            bail!("C++ add_lora failed");
        }
        Ok(id)
    }

    /// Add a dynamic adapter with a tenant-specific Adam learning rate.
    ///
    /// Beta1, beta2, and epsilon still inherit the training context defaults.
    /// The legacy `add_lora` entry point continues to inherit every optimizer
    /// hyperparameter from the context.
    pub fn add_lora_with_optimizer_lr(
        &self,
        rank: i64,
        alpha: f64,
        target_layers: &[i64],
        target_modules: &str,
        optimizer_lr: f64,
    ) -> Result<i64> {
        if !optimizer_lr.is_finite() || optimizer_lr < 0.0 {
            bail!("dynamic LoRA optimizer learning rate must be finite and non-negative");
        }
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let add_lora = kh.add_lora_with_optimizer.ok_or_else(|| {
            anyhow::anyhow!("loaded Qwen kernel does not support tenant optimizer overrides")
        })?;
        let tl_ptr = if target_layers.is_empty() {
            std::ptr::null()
        } else {
            target_layers.as_ptr()
        };
        let modules = std::ffi::CString::new(target_modules)?;
        let modules_ptr = if target_modules.is_empty() {
            std::ptr::null()
        } else {
            modules.as_ptr()
        };
        let id = unsafe {
            add_lora(
                self.ptr,
                rank,
                alpha,
                tl_ptr,
                i64::try_from(target_layers.len()).context("target layer count exceeds i64")?,
                modules_ptr,
                optimizer_lr,
            )
        };
        if id < 0 {
            bail!("C++ add_lora_with_optimizer failed");
        }
        Ok(id)
    }

    /// Allocate a dynamic adapter for checkpoint hydration without collective
    /// synchronization of its temporary random initialization.
    pub fn add_lora_for_restore(
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
        let modules_c = std::ffi::CString::new(target_modules)?;
        let modules_ptr = if target_modules.is_empty() {
            std::ptr::null()
        } else {
            modules_c.as_ptr()
        };
        let id = unsafe {
            (kh.add_lora_for_restore)(self.ptr, rank, alpha, tl_ptr, tl_len, modules_ptr)
        };
        if id < 0 {
            bail!("C++ restore adapter allocation failed");
        }
        Ok(id)
    }

    /// Allocate a dynamic adapter with a checkpointed learning rate while
    /// suppressing synchronization of its temporary random initialization.
    pub fn add_lora_for_restore_with_optimizer_lr(
        &self,
        rank: i64,
        alpha: f64,
        target_layers: &[i64],
        target_modules: &str,
        optimizer_lr: f64,
    ) -> Result<i64> {
        if !optimizer_lr.is_finite() || optimizer_lr < 0.0 {
            bail!("dynamic LoRA optimizer learning rate must be finite and non-negative");
        }
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let tl_ptr = if target_layers.is_empty() {
            std::ptr::null()
        } else {
            target_layers.as_ptr()
        };
        let modules_c = std::ffi::CString::new(target_modules)?;
        let modules_ptr = if target_modules.is_empty() {
            std::ptr::null()
        } else {
            modules_c.as_ptr()
        };
        let id = unsafe {
            (kh.add_lora_for_restore_with_optimizer)(
                self.ptr,
                rank,
                alpha,
                tl_ptr,
                target_layers.len() as i64,
                modules_ptr,
                optimizer_lr,
            )
        };
        if id < 0 {
            bail!("C++ restore adapter allocation with optimizer state failed");
        }
        Ok(id)
    }

    /// Remove a LoRA adapter by ID.
    pub fn remove_lora(&self, adapter_id: i64) -> Result<bool> {
        if adapter_id == 0 {
            return Ok(false);
        }
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let status = unsafe { (kh.remove_lora)(self.ptr, adapter_id) };
        if status < 0 {
            bail!("C++ remove_lora failed for adapter {adapter_id}");
        }
        Ok(status != 0)
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
        let mut ids = self.list_dynamic_lora();
        if self.lora_count > 0 {
            // ID 0 is the fixed adapter created with the training context.
            ids.insert(0, 0);
        }
        ids
    }

    /// List only dynamic adapter IDs. Fixed adapter ID 0 is intentionally
    /// excluded because selected multi-LoRA training accepts dynamic tenants.
    fn list_dynamic_lora(&self) -> Vec<i64> {
        let kh = match get_kernels() {
            Some(k) => k,
            None => return Vec::new(),
        };
        // Query the native registry size first; the old fixed 64-entry buffer
        // silently truncated large multi-tenant registries.
        let total = unsafe { (kh.list_lora)(self.ptr, std::ptr::null_mut(), 0) };
        if total <= 0 {
            return Vec::new();
        }
        let mut dynamic_ids = vec![0i64; total as usize];
        let count =
            unsafe { (kh.list_lora)(self.ptr, dynamic_ids.as_mut_ptr(), dynamic_ids.len() as i64) };
        if count <= 0 {
            return Vec::new();
        }
        dynamic_ids.truncate((count as usize).min(dynamic_ids.len()));
        dynamic_ids
    }

    /// Returns a shallow snapshot of the adapter tensor at call time.
    ///
    /// Dynamic Adam may atomically replace the native registry handle. Do not
    /// cache this tensor across a training call; fetch it again after training.
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

    /// Returns a shallow snapshot of the optimizer tensor at call time.
    /// Fetch it again after training because transactional Adam swaps native
    /// registry handles instead of mutating the previously returned handle.
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

    /// Return the independent Adam bias-correction clock for a dynamic tenant.
    pub fn get_adapter_step_count(&self, adapter_id: i64) -> Result<i64> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let step = unsafe { (kh.get_adapter_step_count)(self.ptr, adapter_id) };
        if step < 0 {
            bail!("C++ get_adapter_step_count failed for adapter {adapter_id}");
        }
        Ok(step)
    }

    /// Restore a dynamic tenant's Adam bias-correction clock.
    pub fn set_adapter_step_count(&self, adapter_id: i64, step_count: i64) -> Result<()> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let status = unsafe { (kh.set_adapter_step_count)(self.ptr, adapter_id, step_count) };
        if status != 0 {
            bail!("C++ set_adapter_step_count failed for adapter {adapter_id}");
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

    /// Evaluate a borrowed host int64 batch without constructing tensors in Rust.
    pub fn eval_step_host_i64(
        &self,
        input_ids: &[i64],
        target_mask: &[i64],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<f64> {
        validate_host_batch(input_ids, target_mask, attention_mask, batch_size, seq_len)?;
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let loss = unsafe {
            (kh.eval_step_host_i64)(
                self.ptr,
                input_ids.as_ptr(),
                target_mask.as_ptr(),
                attention_mask.as_ptr(),
                i64::try_from(batch_size).context("host batch_size exceeds i64")?,
                i64::try_from(seq_len).context("host seq_len exceeds i64")?,
            )
        };
        if loss < 0.0 {
            bail!("C++ eval_step_host_i64 failed");
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

    /// Restore the native Adam bias-correction step from a checkpoint.
    pub fn set_step_count(&self, step_count: i64) -> Result<()> {
        let kh = get_kernels().ok_or_else(|| anyhow::anyhow!("kernels not loaded"))?;
        let status = unsafe { (kh.set_step_count)(self.ptr, step_count) };
        if status != 0 {
            bail!("C++ set_step_count failed for step {step_count}");
        }
        Ok(())
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
        if imported < 0 {
            anyhow::bail!("native Adam optimizer state import failed");
        }
        Ok(imported)
    }
}

fn validate_host_batch(
    input_ids: &[i64],
    target_mask: &[i64],
    attention_mask: &[i64],
    batch_size: usize,
    seq_len: usize,
) -> Result<()> {
    if batch_size == 0 || seq_len == 0 {
        bail!("host batch dimensions must be positive");
    }
    let expected = batch_size
        .checked_mul(seq_len)
        .context("host batch shape overflows usize")?;
    for (name, actual) in [
        ("input_ids", input_ids.len()),
        ("target_mask", target_mask.len()),
        ("attention_mask", attention_mask.len()),
    ] {
        if actual != expected {
            bail!(
                "host {name} length {actual} does not match batch_size={batch_size} * seq_len={seq_len}"
            );
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::{
        shard_dense_mlp_weight_for_tp, shard_full_attention_weight_for_tp,
        shard_linear_attention_weight_for_tp, shard_moe_mlp_weight_for_tp,
        shard_vocab_weight_for_tp,
    };
    use tch::{Kind, Tensor};

    #[test]
    fn dense_mlp_tp_shards_matching_intermediate_axes() {
        let gate = Tensor::arange(48, (Kind::Float, tch::Device::Cpu)).reshape([12, 4]);
        let down = Tensor::arange(48, (Kind::Float, tch::Device::Cpu)).reshape([4, 12]);
        let gate_rank_one =
            shard_dense_mlp_weight_for_tp("model.layers.0.mlp.gate_proj.weight", &gate, 2, 1)
                .unwrap()
                .unwrap();
        let down_rank_one =
            shard_dense_mlp_weight_for_tp("model.layers.0.mlp.down_proj.weight", &down, 2, 1)
                .unwrap()
                .unwrap();
        assert_eq!(gate_rank_one.size(), [6, 4]);
        assert_eq!(down_rank_one.size(), [4, 6]);
        assert_eq!(gate_rank_one.double_value(&[0, 0]), 24.0);
        assert_eq!(down_rank_one.double_value(&[0, 0]), 6.0);
    }

    #[test]
    fn vocabulary_tp_shards_embedding_and_head_rows() {
        let tensor = Tensor::arange(48, (Kind::Float, tch::Device::Cpu)).reshape([12, 4]);
        let embed = shard_vocab_weight_for_tp(
            "model.language_model.embed_tokens.weight",
            &tensor,
            12,
            2,
            1,
        )
        .unwrap()
        .unwrap();
        let head = shard_vocab_weight_for_tp("lm_head.weight", &tensor, 12, 2, 1)
            .unwrap()
            .unwrap();
        assert_eq!(embed.size(), [6, 4]);
        assert_eq!(head.size(), [6, 4]);
        assert_eq!(embed.double_value(&[0, 0]), 24.0);
        assert_eq!(head.double_value(&[0, 0]), 24.0);
    }

    #[test]
    fn vocabulary_tp_ignores_other_weights_and_rejects_invalid_layouts() {
        let tensor = Tensor::zeros([12, 4], (Kind::Float, tch::Device::Cpu));
        assert!(
            shard_vocab_weight_for_tp("model.norm.weight", &tensor, 12, 2, 0)
                .unwrap()
                .is_none()
        );
        assert!(shard_vocab_weight_for_tp("lm_head.weight", &tensor, 11, 2, 0).is_err());
        assert!(
            shard_vocab_weight_for_tp(
                "model.embed_tokens.weight",
                &Tensor::zeros([11, 4], (Kind::Float, tch::Device::Cpu)),
                12,
                2,
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn dense_mlp_tp_ignores_non_mlp_weights_and_rejects_bad_shapes() {
        let tensor = Tensor::zeros([5, 4], (Kind::Float, tch::Device::Cpu));
        assert!(
            shard_dense_mlp_weight_for_tp("model.layers.0.self_attn.q_proj.weight", &tensor, 2, 0)
                .unwrap()
                .is_none()
        );
        assert!(
            shard_dense_mlp_weight_for_tp("model.layers.0.mlp.up_proj.weight", &tensor, 2, 0)
                .is_err()
        );
    }

    #[test]
    fn moe_tp_repacks_each_gate_up_half_before_concatenation() {
        let packed = Tensor::arange(96, (Kind::Float, tch::Device::Cpu)).reshape([2, 12, 4]);
        let rank_zero =
            shard_moe_mlp_weight_for_tp("model.layers.0.mlp.experts.gate_up_proj", &packed, 2, 0)
                .unwrap()
                .unwrap();
        let rank_one =
            shard_moe_mlp_weight_for_tp("model.layers.0.mlp.experts.gate_up_proj", &packed, 2, 1)
                .unwrap()
                .unwrap();
        assert_eq!(rank_zero.size(), [2, 6, 4]);
        assert_eq!(rank_one.size(), [2, 6, 4]);
        assert_eq!(rank_zero.double_value(&[0, 3, 0]), 24.0);
        assert_eq!(rank_one.double_value(&[0, 0, 0]), 12.0);
        assert_eq!(rank_one.double_value(&[0, 3, 0]), 36.0);

        let rebuilt_gate = Tensor::cat(&[&rank_zero.narrow(1, 0, 3), &rank_one.narrow(1, 0, 3)], 1);
        let rebuilt_up = Tensor::cat(&[&rank_zero.narrow(1, 3, 3), &rank_one.narrow(1, 3, 3)], 1);
        let rebuilt = Tensor::cat(&[&rebuilt_gate, &rebuilt_up], 1);
        assert_eq!(
            rebuilt
                .f_sub(&packed)
                .unwrap()
                .abs()
                .max()
                .double_value(&[]),
            0.0
        );
    }

    #[test]
    fn moe_tp_shards_routed_and_shared_down_input_axes() {
        let routed = Tensor::arange(96, (Kind::Float, tch::Device::Cpu)).reshape([2, 4, 12]);
        let shared = Tensor::arange(48, (Kind::Float, tch::Device::Cpu)).reshape([4, 12]);
        let routed_rank_one =
            shard_moe_mlp_weight_for_tp("model.layers.0.mlp.experts.down_proj", &routed, 2, 1)
                .unwrap()
                .unwrap();
        let shared_rank_one = shard_moe_mlp_weight_for_tp(
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            &shared,
            2,
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(routed_rank_one.size(), [2, 4, 6]);
        assert_eq!(shared_rank_one.size(), [4, 6]);
        assert_eq!(routed_rank_one.double_value(&[0, 0, 0]), 6.0);
        assert_eq!(shared_rank_one.double_value(&[0, 0]), 6.0);
    }

    #[test]
    fn moe_tp_rejects_invalid_packed_and_intermediate_shapes() {
        let odd_packed = Tensor::zeros([2, 10, 4], (Kind::Float, tch::Device::Cpu));
        assert!(
            shard_moe_mlp_weight_for_tp(
                "model.layers.0.mlp.experts.gate_up_proj",
                &odd_packed,
                2,
                0,
            )
            .is_err()
        );
        let down = Tensor::zeros([2, 4, 5], (Kind::Float, tch::Device::Cpu));
        assert!(
            shard_moe_mlp_weight_for_tp("model.layers.0.mlp.experts.down_proj", &down, 2, 0,)
                .is_err()
        );
    }

    #[test]
    fn full_attention_tp_shards_head_and_output_axes() {
        let q = Tensor::arange(96, (Kind::Float, tch::Device::Cpu)).reshape([24, 4]);
        let o = Tensor::arange(48, (Kind::Float, tch::Device::Cpu)).reshape([4, 12]);
        let q_rank_one =
            shard_full_attention_weight_for_tp("model.layers.0.self_attn.q_proj.weight", &q, 2, 1)
                .unwrap()
                .unwrap();
        let o_rank_one =
            shard_full_attention_weight_for_tp("model.layers.0.self_attn.o_proj.weight", &o, 2, 1)
                .unwrap()
                .unwrap();
        assert_eq!(q_rank_one.size(), [12, 4]);
        assert_eq!(o_rank_one.size(), [4, 6]);
        assert_eq!(q_rank_one.double_value(&[0, 0]), 48.0);
        assert_eq!(o_rank_one.double_value(&[0, 0]), 6.0);
    }

    #[test]
    fn full_attention_tp_ignores_other_weights_and_rejects_bad_shapes() {
        let tensor = Tensor::zeros([5, 4], (Kind::Float, tch::Device::Cpu));
        assert!(
            shard_full_attention_weight_for_tp(
                "model.layers.0.mlp.gate_proj.weight",
                &tensor,
                2,
                0,
            )
            .unwrap()
            .is_none()
        );
        assert!(
            shard_full_attention_weight_for_tp(
                "model.layers.0.self_attn.q_proj.weight",
                &tensor,
                2,
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn linear_attention_tp_preserves_flat_qkv_and_conv_segments() {
        let qkv = Tensor::arange(48, (Kind::Float, tch::Device::Cpu)).reshape([12, 4]);
        let conv = Tensor::arange(24, (Kind::Float, tch::Device::Cpu)).reshape([12, 1, 2]);
        let qkv_rank_one = shard_linear_attention_weight_for_tp(
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            &qkv,
            2,
            1,
            2,
            2,
            4,
            1,
        )
        .unwrap()
        .unwrap();
        let conv_rank_one = shard_linear_attention_weight_for_tp(
            "model.layers.0.linear_attn.conv1d.weight",
            &conv,
            2,
            1,
            2,
            2,
            4,
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(qkv_rank_one.size(), [6, 4]);
        assert_eq!(conv_rank_one.size(), [6, 1, 2]);
        // Global rows are Q=[0..4], K=[4..8], V=[8..12]. Rank 1 owns the
        // second half of each segment and repacks them as local [Q|K|V].
        assert_eq!(qkv_rank_one.double_value(&[0, 0]), 8.0);
        assert_eq!(qkv_rank_one.double_value(&[2, 0]), 24.0);
        assert_eq!(qkv_rank_one.double_value(&[4, 0]), 40.0);
        assert_eq!(conv_rank_one.double_value(&[0, 0, 0]), 4.0);
        assert_eq!(conv_rank_one.double_value(&[2, 0, 0]), 12.0);
        assert_eq!(conv_rank_one.double_value(&[4, 0, 0]), 20.0);
    }

    #[test]
    fn linear_attention_tp_shards_value_head_tensors_and_output_columns() {
        let z = Tensor::arange(32, (Kind::Float, tch::Device::Cpu)).reshape([8, 4]);
        let a = Tensor::arange(16, (Kind::Float, tch::Device::Cpu)).reshape([4, 4]);
        let a_log = Tensor::arange(4, (Kind::Float, tch::Device::Cpu));
        let out = Tensor::arange(32, (Kind::Float, tch::Device::Cpu)).reshape([4, 8]);
        let shard = |name: &str, tensor: &Tensor| {
            shard_linear_attention_weight_for_tp(name, tensor, 2, 1, 2, 2, 4, 2)
                .unwrap()
                .unwrap()
        };
        let z = shard("model.layers.0.linear_attn.in_proj_z.weight", &z);
        let a = shard("model.layers.0.linear_attn.in_proj_a.weight", &a);
        let a_log = shard("model.layers.0.linear_attn.A_log", &a_log);
        let out = shard("model.layers.0.linear_attn.out_proj.weight", &out);
        assert_eq!(z.size(), [4, 4]);
        assert_eq!(a.size(), [2, 4]);
        assert_eq!(a_log.size(), [2]);
        assert_eq!(out.size(), [4, 4]);
        assert_eq!(z.double_value(&[0, 0]), 16.0);
        assert_eq!(a.double_value(&[0, 0]), 8.0);
        assert_eq!(a_log.double_value(&[0]), 2.0);
        assert_eq!(out.double_value(&[0, 0]), 4.0);
    }

    #[test]
    fn linear_attention_tp_rejects_invalid_groups_and_shapes() {
        let tensor = Tensor::zeros([12, 4], (Kind::Float, tch::Device::Cpu));
        assert!(
            shard_linear_attention_weight_for_tp(
                "model.layers.0.linear_attn.norm.weight",
                &tensor,
                2,
                0,
                2,
                2,
                4,
                1,
            )
            .unwrap()
            .is_none()
        );
        assert!(
            shard_linear_attention_weight_for_tp(
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                &tensor,
                2,
                0,
                3,
                2,
                4,
                1,
            )
            .is_err()
        );
        assert!(
            shard_linear_attention_weight_for_tp(
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                &Tensor::zeros([11, 4], (Kind::Float, tch::Device::Cpu)),
                2,
                0,
                2,
                2,
                4,
                1,
            )
            .is_err()
        );
    }
}
