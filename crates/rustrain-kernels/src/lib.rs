//! `rustrain-kernels` — the **reference** operator provider.
//!
//! Pure-Rust, CPU, f32-only implementations of the spec §2.4 vocabulary,
//! published as an ABI plugin under the variant name `reference.f32`. Its
//! job is to be the numerical ground truth for conformance checking and the
//! backend that makes the framework testable without a GPU. It is explicitly
//! not a performance path: clarity and bit-reproducibility matter more than
//! speed, so every reduction runs in a fixed ascending index order.
//!
//! House rules (see the operator docs for per-op detail):
//!
//! * **infer** is implemented for every op and is pure — it computes
//!   shape/dtype into stack arrays, never allocates, never reads element
//!   data. Plan validation depends on it.
//! * **memory** is implemented for every op. Outputs are always
//!   caller-provided (contract C-3); the reference provider never allocates
//!   on the output side. A null `memory` slot means "cannot plan" to the
//!   compiler, so every descriptor registers a reporter: the shared zero
//!   reporter for ops that use no scratch, and a per-op reporter for `sdpa`,
//!   whose fused body needs two S*T f32 scratch buffers per call.
//! * Outputs are caller-provided: the executor allocates from `infer()`'s
//!   declared shape/dtype, and `execute` validates that before writing. A
//!   mismatch is a non-zero status with a message on `last_error`.
//! * Non-contiguous *inputs* are rejected with a clear message; contiguity
//!   handling is a documented later improvement.
//! * Variant selection is by attribute, not by extra ops: the op table stays
//!   exactly the fixed vocabulary, and `kind`/`scheme`/`format` strings
//!   select behaviour. Unknown values are hard errors naming the accepted
//!   values.
//! * Quantization is data-driven (contract R-3): the scheme is declared in
//!   attributes, never inferred from scale-tensor shapes.
//! * Composites declare an `expansion` (contract R-4) over the local-tensor
//!   id convention in `rustrain_op.h`.

pub mod attrs;
pub mod dispatch;
pub mod error;
pub mod op;
pub mod tensor;

use std::sync::OnceLock;

use rustrain_abi::author::{OpSpec, PluginBuilder};
use rustrain_abi::ffi::{
    RsBackwardKind, RsCollective, RsCollectiveKind, RsDtype, RsExecuteFn, RsGroupKind, RsInferFn,
    RsNumerics, RsPlugin, RsShardRule,
};

use crate::op::composite::{
    adamw_exec, adamw_expansion, adamw_infer, ce_exec, ce_infer, cross_entropy_expansion,
    sdpa_exec, sdpa_expansion, sdpa_infer, sdpa_memory, topk_exec, topk_expansion, topk_infer,
};
use crate::op::compute::{
    binary_exec, binary_infer, bmm_exec, bmm_infer, causal_conv1d_exec, causal_conv1d_infer,
    compare_exec, compare_infer, l2norm_exec, l2norm_infer, layernorm_exec, layernorm_infer,
    linear_exec, linear_infer, matmul_exec, matmul_infer, reduce_exec, reduce_infer, rmsnorm_exec,
    rmsnorm_gated_exec, rmsnorm_gated_infer, rmsnorm_infer, rope_exec, rope_infer, softmax_exec,
    softmax_infer, unary_exec, unary_infer,
};
use crate::op::meta::{
    broadcast_exec, broadcast_infer, cat_exec, cat_infer, narrow_exec, narrow_infer, reshape_exec,
    reshape_infer, transpose_exec, transpose_infer, view_exec, view_infer,
};
use crate::op::moe::{moe_exec, moe_infer, moe_memory};
use crate::op::movement::{
    embedding_exec, embedding_infer, gather_exec, gather_infer, scatter_exec, scatter_infer,
};
use crate::op::quant::{
    amax_exec, amax_infer, dequantize_exec, dequantize_infer, quantize_exec, quantize_infer,
};
use crate::op::recurrent::{
    gated_delta_rule_exec, gated_delta_rule_infer, gated_delta_rule_memory,
};

pub const VARIANT: &str = "reference.f32";

const F32: &[RsDtype] = &[RsDtype::F32];
const F32_IDX: &[RsDtype] = &[RsDtype::F32, RsDtype::I32, RsDtype::I64];
const FP8: &[RsDtype] = &[RsDtype::F32, RsDtype::F8E4M3, RsDtype::F8E5M2];

/// Shared descriptor template: f32 in / f32 out / f32 accumulate, zero
/// workspace, `last_error` wired, and the ABI-required `execute`/`infer`/
/// `memory` slots always present.
fn spec(
    name: &'static str,
    shard: RsShardRule,
    doc: &'static str,
    dtypes: &'static [RsDtype],
    backward: RsBackwardKind,
    infer: RsInferFn,
    exec: RsExecuteFn,
) -> OpSpec {
    OpSpec::new(name, VARIANT)
        .shard(shard)
        .doc(doc)
        .dtypes(dtypes)
        .numerics(RsNumerics {
            in_dtype: RsDtype::F32,
            out_dtype: RsDtype::F32,
            accum_dtype: RsDtype::F32,
            grad_dtype: RsDtype::F32,
            ..Default::default()
        })
        .backward(backward)
        .infer(infer)
        .memory(crate::dispatch::memory_zero)
        .execute(exec)
        .last_error(crate::error::plugin_last_error)
}

// Operator documentation is part of the deliverable: it records the semantic
// choices where the spec leaves them open (conventions, tie-breaking, NaN
// policy), so conformance has something to check against.

const VIEW_DOC: &str = "Zero-copy alias: the output descriptor (shape/stride/data) is an exact copy of the input and aliases its buffer (out.data = in.data). No allocation, no copy; the executor must not pre-allocate this output.";

const RESHAPE_DOC: &str = "Reinterprets the input's ROW-MAJOR LOGICAL order as the 'shape' attribute (list of i64, required; one -1 allowed, filled so numel is preserved). A contiguous input is a zero-copy alias (out.data = in.data); a strided input (a narrow of a flat projection, say) is materialised into the output slot in logical order, which is what makes a per-head regroup of a flat QKV slice correct rather than a read of the neighbouring segments.";

const TRANSPOSE_DOC: &str = "Zero-copy view: swaps axes dim0/dim1 (i64 attrs, defaults -2/-1; negative axes count from the end), swapping shape and strides while aliasing the input buffer. The result is usually non-contiguous; downstream compute ops reject it — by design.";

const NARROW_DOC: &str = "Zero-copy view: selects [start, start+length) along 'dim' (i64, default -1) by offsetting out.data = in.data + start*stride[dim]*4 bytes with identical strides. Attributes: dim, start (default 0), length (default shape[dim]-start). No copy.";

const CAT_DOC: &str = "Concatenates inputs along 'dim' (default -1). The exception among the meta ops: cat COPIES, because the concatenated buffer is new. Inputs share rank, dtype and all non-concat dims, and must be contiguous; the copy runs input order then index order (deterministic).";

const BROADCAST_DOC: &str = "Zero-copy view: right-aligns the input against the target 'shape' attribute (list of i64, required). Size-1 input dims get stride 0 (the same element is read repeatedly); new leading dims also get stride 0. out.data aliases the input.";

const MATMUL_DOC: &str = "2D matrix multiply C = A@B: [M,K] x [K,N] -> [M,N], f32 accumulate. Optional 'transpose_b' (bool, default false) computes A @ B^T with B as [N, K] — the sdpa expansion needs this form because its primitives require contiguous inputs and a transpose view would be strided. The attribute is declared, never inferred from shapes. Naive triple loop with k accumulated in ascending order — the fixed summation order is what makes results bitwise reproducible. Inputs must be contiguous.";

const LINEAR_DOC: &str = "y = x @ w (+ b). Conventions: w is [K, N] (output features last), x is [..., K], optional bias is [N]; y is [..., N]. Literally a batched matmul plus optional bias broadcast over the last dim. 2 or 3 inputs.";

const BMM_DOC: &str = "Batched matmul: [L..., M, K] x [L..., K, N] -> [L..., M, N]. Optional 'transpose_b' (bool, default false) computes A @ B^T with B as [L..., N, K]. Batch dims must be identical (no batch broadcasting). Per-batch naive triple loop with k ascending (deterministic).";

const UNARY_DOC: &str = "Applies the 'kind' attribute (required string) elementwise: silu, gelu, sigmoid, tanh, relu, exp, log, neg, sqrt, rsqrt, softplus, negative_exp, silu_grad, gelu_grad, sigmoid_grad, tanh_grad, relu_grad. Choices: gelu uses the tanh approximation (the vocabulary has no erf); log is natural; rsqrt = 1/sqrt(x) (rsqrt(0) = +inf, rsqrt of a negative is NaN per IEEE); softplus = ln(1+e^x) with the exact identity for x > 20 (torch's beta=1/threshold=20 form, Qwen3.6's dt gate); negative_exp = -exp(x) (Qwen3.6's GDN decay). The *_grad kinds compute f'(x) at x, same shape, so a VJP is written as elementwise_binary(mul, f_grad(x), dy) — composable, no new ops. gelu_grad is the derivative of the tanh-approximation gelu above, not of the erf form. relu_grad is the subgradient convention 0 at x = 0 (the strict '>' follows IEEE, so relu_grad(NaN) = 0). 'neg' and 'sqrt' extend the core list because the declared expansions of cross_entropy and adamw must be expressible in the fixed vocabulary. An unknown kind is a hard error naming the accepted values.";

const BINARY_DOC: &str = "Applies the 'kind' attribute (required: add, sub, mul, div, maximum, pow) with right-aligned broadcasting. One input is allowed when the scalar attribute 'rhs' (f64) is given (y = x op rhs) — the adamw expansion uses that form. div follows IEEE (x/0 -> +/-inf); pow follows IEEE powf (0^0 = 1, a negative base with a fractional exponent is NaN); maximum uses f32::max, which ignores NaN (NaN inputs are outside the contract).";

const COMPARE_DOC: &str = "Elementwise comparison — the vocabulary's mask primitive: two equal-shape f32 inputs to one f32 output with exactly 1.0 where the comparison holds and 0.0 elsewhere. 'kind' (required: eq, ne, lt, le, gt, ge). NaN policy: every comparison involving NaN yields 0.0 — IEEE ordered comparisons are false with NaN, and 'ne' deliberately follows suit (C's NaN != x would be true) so a NaN never smuggles a 1 into a mask. The mask is f32 by design: the ABI has no boolean dtype, so masking composes as elementwise_binary(mul, x, compare(...)). Both inputs must have the same shape and dtype; a mismatch is a hard error naming both.";

const REDUCE_DOC: &str = "Reduces along 'axis' with 'kind' (required: sum, mean, max, amax; amax = max|x|). With no 'axis', reduces everything to a rank-0 scalar. The reduced axis is removed by default; 'keepdim' (bool, default false) keeps it as a size-1 dim — the softmax/layernorm VJPs need the rank preserved. With no 'axis', keepdim = true makes every dim size 1 (torch convention). Negative axes count from the end. All accumulations run in ascending index order (deterministic).";

const SOFTMAX_DOC: &str = "Stable softmax over 'axis' (i64, default -1) of x*scale, 'scale' (f64) defaulting to 1.0; shape is preserved. Per-lane max subtraction keeps large-magnitude inputs finite; sums accumulate in ascending order.";

const RMSNORM_DOC: &str = "y = x / sqrt(mean(x^2) + eps) (* (w + weight_offset)), normalized over the last dim. eps (f64, default 1e-5) sits inside the sqrt (near-zero inputs are safe); optional weight w is [D]. The weight convention is declared by 'weight_offset' (f64, default 0.0): y multiplies the RAW weight plus the offset — the trunk uses offset 1.0 (1 + weight), GDN's gated normalisation uses the raw weight (offset 0.0, the pre-D5 behaviour). 1 or 2 inputs.";

const LAYERNORM_DOC: &str = "y = (x - mean) / sqrt(var + eps) (* w) (+ b) over the last dim; biased variance; eps (f64, default 1e-5) sits outside the sqrt. Optional w, b are [D]. 1, 2 or 3 inputs.";

const ROPE_DOC: &str = "Rotary embedding for two operands (q, k) at once: y0, y1 = rope(x, y [, pos]). The rotation is half-split over the first 'rotary_dim' dims of the last axis — pairs (i, i+h) with h = rotary_dim/2: out[i] = x[i]*cos - x[i+h]*sin, out[i+h] = x[i+h]*cos + x[i]*sin (HF rotate_half) — and the remaining dims pass through unchanged (the last axis is the per-head axis: [seq, heads, head_dim] rotates each head identically). cos/sin are computed internally from 'theta' (f64, default 1e7) and the positions: inv_freq[j] = theta^(-2j/rotary_dim). Positions vary along the FIRST axis (the plan's sequence-first layout), so row `t * R + b` of the flattened buffer — `R` = the product of the dims between the position axis and the last one — rotates with position `t`'s angle: for a `[seq, heads, head_dim]` input every head of a position turns by that position's angle, and the rank-2 `[seq, head_dim]` case (`R = 1`) is the same rule; the optional 3rd input 'pos' (f32/i32/i64, one entry per position) selects them, and when absent positions default to arange(S) (causal 0..S-1). Attributes: rotary_dim (i64, default D — the whole last dim), partial_rotary (bool, default false; true declares the prefix rotation and allows rotary_dim < D; false requires rotary_dim == D), theta. x and y must have identical shape of rank >= 2.";

const QUANTIZE_DOC: &str = "Data-driven quantization (contract R-3): scheme = 'per_tensor' | 'per_token' | 'per_block' and format = 'f8e4m3' | 'f8e5m2' are attributes, never inferred from tensor shapes. Outputs: (q, scale) — q carries the target fp8 dtype, one byte per element; scale is an explicit f32 tensor whose shape the declared scheme dictates (per_tensor: scalar; per_token: shape minus the last dim; per_block: [..., rows/m, cols/n] with block = [m, n] required and dividing the last two dims exactly). scale = max|x| / max_finite(format); a zero amax gets scale 1.0 (0/0 would be NaN). The fp8 payloads are a faithful round-to-nearest-even EMULATION of the target grid (subnormals included), not the hardware instruction. NaN inputs are rejected.";

const DEQUANTIZE_DOC: &str = "Inverse of quantize: (q, scale) -> x = fp8_decode(q) * scale. scheme/format/block attributes mirror quantize; the scale tensor must have the shape the declared scheme dictates (validated, never inferred). The fp8 decode reproduces the exact value of the emulated grid.";

const AMAX_DOC: &str = "amax' = max(amax, |x|) aggregated under the declared scheme ('per_tensor' | 'per_token' | 'per_block' attribute; block = [m, n] for per_block). Maintains the running max used for dynamic scaling; output keeps the amax shape. NaN inputs are rejected — a NaN would silently poison the running max for every future step.";

const EMBEDDING_DOC: &str = "out = w[indices]: weight w [V, D] (f32), indices i32/i64 of any rank -> out = indices.shape + [D]. Negative indices are rejected (no wrap-around — a reference backend must not guess).";

const GATHER_DOC: &str = "Torch-gather convention: indices (i32/i64) have the same rank as x, dims equal to x's except along 'axis' (i64, default -1), and the output has the indices' shape: out[i] = x[i with axis value indices[i]]. Negative indices are rejected (no wrap-around).";

const SCATTER_DOC: &str = "Copy of x, then out[..., indices[k], ...] = values[..., k, ...] along 'axis' (default -1), k ascending. values has x's shape with the axis dim equal to K. 'reduce' ('assign' | 'add', default 'assign'): assign is the existing semantics — duplicate indices: the last writer (largest k) wins; add accumulates duplicates instead, and because accumulation runs in the fixed ascending-k order, two runs are bitwise identical. Deterministic and documented either way.";

const SDPA_DOC: &str = "Scaled-dot-product attention with GQA: o = softmax(q @ k^T * scale + mask) @ v. Two declared input forms: with 'per_head' (bool, default false) the inputs are per-head — q [.., S, H, D], k/v [.., T, Hkv, D/Dv] (S at the third-to-last axis, the plan's sequence-first layout); without it, the legacy flat form q [.., S, D], k [.., T, D], v [.., T, Dv] (one head). Identical batch dims; S and T may differ. The head counts are READ FROM THE TENSORS' OWN AXES, never from an attribute: the plan hands this node the rank's local slice, so a declared global count would contradict the tensor it describes (a sharded q has fewer heads than the model). Each kv head serves H / Hkv query heads — the KV repeat the design refuses to add as a primitive — and H must be a positive multiple of Hkv. 'causal' (bool, default false) masks j > i to -inf. 'scale' (f64) is explicit when given; when absent the declared convention applies: 1/sqrt(head_dim) in the per-head form, and the legacy default 1.0 otherwise — the old flat form keeps its old default, and the declared expansion's softmax node uses the same default, so fused == expansion stays checkable. The output is [.., S, H, Dv] (per-head) or [.., S, Dv] (flat). Optional 4th input: an additive mask (f32) broadcastable to [.., S, T], added to the scores BEFORE the softmax (0.0 = attend, -inf masks a position — the HF attention_mask convention). Expansion: bmm(q, k, transpose_b=true) -> softmax -> bmm(p, v) — valid for the legacy flat form (one head); GQA/causal cases cannot be replayed from the primitive vocabulary, which is exactly why the fused body exists.";

const CE_DOC: &str = "Mean cross-entropy loss of logits [N, C] against targets (i32/i64, [N]): mean_n(logsumexp_n - logits[n, t_n]). The fused body uses the stable log-sum-exp form, finite even for extreme logits (e.g. [1000, -1000, 0] -> 1000); the declared expansion is the naive -mean(log(softmax)) composition (softmax -> log -> reshape(targets,[-1,1]) -> gather(axis=-1) -> neg -> reduce mean), which agrees within tolerance wherever the naive form is well-conditioned.";

const ADAMW_DOC: &str = "AdamW step with decoupled weight decay: m' = b1*m + (1-b1)g; v' = b2*v + (1-b2)g^2; p' = p - lr * ((m'/(1-b1^t)) / (sqrt(v'/(1-b2^t)) + eps) + wd*p). Inputs (param, grad, exp_avg, exp_avg_sq), outputs (param', exp_avg', exp_avg_sq') so the caller persists the moments. Attributes: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, step=1 (i64, >= 1). Bias corrections use integer powers — exact and deterministic.";

const TOPK_DOC: &str = "MoE router: (weights, indices) = top-k of softmax(logits [N, E]) per row. 'top_k' (i64, default 2, 1..=E). Weights are the softmax probabilities of the selected experts; 'norm_topk_prob' (bool, default false) renormalises each row's k weights to sum to 1 — HF's Qwen3_5MoeTopKRouter does this unconditionally (router_top_value /= router_top_value.sum(-1, keepdim=True)); the description declares the flag so the numerics stay configuration, not convention. The row sum runs in ascending k order (deterministic). Ties break toward the lower expert index (strict '>' in an insertion selection — deterministic). indices are i32. The top-k selection itself has no primitive in spec §2.4, so the expansion covers the gating-probability path (softmax) and the selection is documented here.";

const L2NORM_DOC: &str = "y = x / sqrt(sum(x^2, dim) + eps): the L2 normalisation GDN applies to q and k (HF l2norm in modeling_qwen3_5_moe.py, aligned with FLA). The SUM of squares — not the mean — with eps added inside the sqrt, all in ascending order. Attributes: dim (i64, default -1), eps (f64, default 1e-6). No learnable parameter; the delta rule's 1/sqrt(D) query scaling is gated_delta_rule's own, always-applied convention.";

const RMSNORM_GATED_DOC: &str = "GDN's output normalisation (HF Qwen3_5MoeRMSNormGated): the reduce and the gating run in ONE pass. x is treated as [rows, D] with D = w.len() (any layout whose numel is a multiple of D — HF reshapes the v-heads flat before the norm) and y = (x / sqrt(mean(x^2) + eps)) * (w + weight_offset) * silu(gate). Weight convention: RAW weight — weight_offset (f64, default 0.0) must be 0.0 for GDN, unlike the trunk's 1 + weight. Attributes: eps (f64, default 1e-6, inside the sqrt), gate_act (string, default 'silu'; only 'silu' exists today), weight_offset. Inputs: (x, w [D], gate) with gate having x's element count (applied elementwise after the row normalisation).";

const CAUSAL_CONV1D_DOC: &str = "Qwen3.6's depthwise causal sequence convolution (HF causal_conv1d_fn): out[t, c] = sum_k w[c, 0, k] * x[t + k - pad, c] with x[j < 0] = 0, optionally fused with silu. Input x is [.., L, C] (sequence first, channels last); the weight is [C, 1, K] with one input channel per output channel (depthwise). Attributes: kernel (i64, default K — the weight's own tap count, which a declared kernel must equal: the checkpoint's kernel size is the contract, never a silent truncation), groups (string, default 'channels'; only 'channels' = depthwise exists), activation (string, default '' = none; 'silu' fuses the activation into the same sweep), pad (i64, default kernel - 1 — strictly causal; left-only padding of pad zeros, matching F.conv1d(padding=pad) cropped to the original length).";

const GATED_DELTA_RULE_DOC: &str = "The GDN recurrence itself (HF torch_chunk_gated_delta_rule, chunked form): inputs (q, k, v, g, beta) with q/k [.., S, kh*D], v [.., S, vh*Dv], g/beta [.., S, vh] (identical batch dims), output [.., S, vh*Dv]. Conventions: the state is fp32 ('state_dtype' attr, only 'f32' accepted — the declaration is enforced, not assumed); 'chunk_size' (i64, default 64) sizes the chunked scan (padding to a multiple is internal, the output keeps S); k_head_dim == v_head_dim (declared — both 128 in Qwen3.6; a mismatch is a hard error); q and k heads are repeat_interleaved by vh/kh (GQA for the delta rule); the query is scaled by D^-0.5 inside (the L2 normalisation itself is the caller's l2norm nodes); g is log-space decay (<= 0) and beta gates the state update: S_t = S_{t-1} * exp(g_t) + k_t * ((v_t - S_{t-1} k_t) * beta_t), read AFTER the update o_t = q_t^T S_t. The CP>1 dependency is declared, not derived: this descriptor carries collectives = [all_gather {cp}] for the affine-map merge (op-vocabulary §4); the reference provider itself runs single-rank.";

const MOE_LAYER_DOC: &str = "Qwen3.6's sparse MoE layer as ONE explicit operator (op-vocabulary §5, option A): the router (topk_router) runs upstream, and dispatch/combine are all_to_all({tp, ep}) exchanges whose per-rank token counts are only known at run time — so the whole layer is a single node with static [.., H] in/out and the two collectives declared on this descriptor (layout arithmetic cannot derive them: the routing is data, op-vocabulary §4). Ten inputs, in order: (0) h f32 [.., H] (rank >= 2, rows = product of leading dims); (1) routing_weights f32 [rows, K] — topk_router output 0, the softmax probabilities of the selected experts, used as-is and never renormalised here (any renormalisation is the router's semantics); (2) routing_indices i32/i64 [rows, K] — topk_router output 1; (3) experts_gate_proj f32 [E, H, I] ([out, in] per expert: gate[j] = x @ gate[e][:, j]) and (4) experts_up_proj f32 [E, H, I] — the DE-FUSED halves the description's binding splits off the checkpoint's gate_up_proj (which is [E, 2*I, H] there); a fused [E, 2*I, H] input is deliberately not accepted: the TP shard cuts the 2*I axis across the gate/up boundary, so the fused form cannot be sharded correctly (qwen36-5d-example.md §3) — the description's de-fused slots are the contract; (5) experts_down_proj f32 [E, H, I] ([out, in] per expert); (6/7) shared_gate_proj/shared_up_proj f32 [I, H]; (8) shared_down_proj f32 [H, I]; (9) shared_expert_gate f32 [1, H]. Output: h's exact shape. Numerics are HF's Qwen3_5MoeSparseMoeBlock: per token, EVERY selected expert runs (dropless — no capacity truncation), out = sum_k w_k * down_k(silu(gate_k(x)) * up_k(x)) with k ascending, plus sigmoid(shared_expert_gate @ x) * shared_expert(x). silu/sigmoid use the x/(1+e^-x) and 1/(1+e^-x) forms of elementwise_unary, and every accumulation runs in fixed ascending order (k, j, hidden), so two runs are bitwise identical. No attributes: K is the routing tensors' last dim (validated 1..=E), and the topk/norm_topk_prob attributes the description may attach belong to the topk_router node upstream, not to this op. Out-of-range expert indices are a hard error (no wrap-around). Collectives: two ALL_TO_ALL {tp, ep} — dispatch (tensor_index 0, the h input) and combine (tensor_index 10, the output at offset n_inputs in the io list); the reference provider runs single-rank and the declaration is the planner's data. Memory: one H-element f32 accumulation buffer per call, reported by this descriptor's memory hook (4*H bytes).";

fn build_plugin() -> &'static RsPlugin {
    PluginBuilder::new("reference", env!("CARGO_PKG_VERSION"))
        .op(spec(
            "view",
            RsShardRule::ELEMENTWISE,
            VIEW_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            view_infer,
            view_exec,
        ))
        .op(spec(
            "reshape",
            RsShardRule::ELEMENTWISE,
            RESHAPE_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            reshape_infer,
            reshape_exec,
        ))
        .op(spec(
            "transpose",
            RsShardRule::ELEMENTWISE,
            TRANSPOSE_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            transpose_infer,
            transpose_exec,
        ))
        .op(spec(
            "narrow",
            RsShardRule::ELEMENTWISE,
            NARROW_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            narrow_infer,
            narrow_exec,
        ))
        .op(spec(
            "cat",
            RsShardRule::ELEMENTWISE,
            CAT_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            cat_infer,
            cat_exec,
        ))
        .op(spec(
            "broadcast",
            RsShardRule::ELEMENTWISE,
            BROADCAST_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            broadcast_infer,
            broadcast_exec,
        ))
        .op(spec(
            "matmul",
            RsShardRule::MATMUL,
            MATMUL_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            matmul_infer,
            matmul_exec,
        ))
        .op(spec(
            "linear",
            RsShardRule::LINEAR,
            LINEAR_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            linear_infer,
            linear_exec,
        ))
        .op(spec(
            "bmm",
            RsShardRule::MATMUL,
            BMM_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            bmm_infer,
            bmm_exec,
        ))
        .op(spec(
            "elementwise_unary",
            RsShardRule::ELEMENTWISE,
            UNARY_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            unary_infer,
            unary_exec,
        ))
        .op(spec(
            "elementwise_binary",
            RsShardRule::ELEMENTWISE,
            BINARY_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            binary_infer,
            binary_exec,
        ))
        .op(spec(
            "compare",
            RsShardRule::ELEMENTWISE,
            COMPARE_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            compare_infer,
            compare_exec,
        ))
        .op(spec(
            "reduce",
            RsShardRule::DECLARED,
            REDUCE_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            reduce_infer,
            reduce_exec,
        ))
        .op(spec(
            "softmax",
            RsShardRule::ELEMENTWISE,
            SOFTMAX_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            softmax_infer,
            softmax_exec,
        ))
        .op(spec(
            "rmsnorm",
            RsShardRule::ELEMENTWISE,
            RMSNORM_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            rmsnorm_infer,
            rmsnorm_exec,
        ))
        .op(spec(
            "layernorm",
            RsShardRule::ELEMENTWISE,
            LAYERNORM_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            layernorm_infer,
            layernorm_exec,
        ))
        .op(spec(
            "rope",
            RsShardRule::ELEMENTWISE,
            ROPE_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            rope_infer,
            rope_exec,
        ))
        .op(spec(
            "quantize",
            RsShardRule::ELEMENTWISE,
            QUANTIZE_DOC,
            F32,
            RsBackwardKind::NONDIFF,
            quantize_infer,
            quantize_exec,
        ))
        .op(spec(
            "dequantize",
            RsShardRule::ELEMENTWISE,
            DEQUANTIZE_DOC,
            FP8,
            RsBackwardKind::NONDIFF,
            dequantize_infer,
            dequantize_exec,
        ))
        .op(spec(
            "amax_update",
            RsShardRule::ELEMENTWISE,
            AMAX_DOC,
            F32,
            RsBackwardKind::NONDIFF,
            amax_infer,
            amax_exec,
        ))
        .op(spec(
            "embedding",
            RsShardRule::EMBEDDING,
            EMBEDDING_DOC,
            F32_IDX,
            RsBackwardKind::AUTODIFF,
            embedding_infer,
            embedding_exec,
        ))
        .op(spec(
            "gather",
            RsShardRule::ELEMENTWISE,
            GATHER_DOC,
            F32_IDX,
            RsBackwardKind::AUTODIFF,
            gather_infer,
            gather_exec,
        ))
        .op(spec(
            "scatter",
            RsShardRule::ELEMENTWISE,
            SCATTER_DOC,
            F32_IDX,
            RsBackwardKind::AUTODIFF,
            scatter_infer,
            scatter_exec,
        ))
        .op(spec(
            "sdpa",
            RsShardRule::PASS_THROUGH,
            SDPA_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            sdpa_infer,
            sdpa_exec,
        )
        // sdpa's fused body needs scratch (scores + probs); report it so
        // planning sees the real footprint instead of the shared zeros.
        .memory(sdpa_memory)
        .expansion(sdpa_expansion()))
        .op(spec(
            "cross_entropy",
            RsShardRule::ELEMENTWISE,
            CE_DOC,
            F32_IDX,
            RsBackwardKind::AUTODIFF,
            ce_infer,
            ce_exec,
        )
        .expansion(cross_entropy_expansion()))
        .op(spec(
            "adamw",
            RsShardRule::DECLARED,
            ADAMW_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            adamw_infer,
            adamw_exec,
        )
        .expansion(adamw_expansion()))
        .op(spec(
            "topk_router",
            RsShardRule::PASS_THROUGH,
            TOPK_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            topk_infer,
            topk_exec,
        )
        .expansion(topk_expansion()))
        .op(spec(
            "l2norm",
            RsShardRule::PASS_THROUGH,
            L2NORM_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            l2norm_infer,
            l2norm_exec,
        ))
        .op(spec(
            "rmsnorm_gated",
            RsShardRule::PASS_THROUGH,
            RMSNORM_GATED_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            rmsnorm_gated_infer,
            rmsnorm_gated_exec,
        ))
        .op(spec(
            "causal_conv1d",
            RsShardRule::PASS_THROUGH,
            CAUSAL_CONV1D_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            causal_conv1d_infer,
            causal_conv1d_exec,
        ))
        .op(spec(
            "gated_delta_rule",
            RsShardRule::PASS_THROUGH,
            GATED_DELTA_RULE_DOC,
            F32,
            RsBackwardKind::AUTODIFF,
            gated_delta_rule_infer,
            gated_delta_rule_exec,
        )
        // The chunked body allocates one flat f32 scratch buffer; report its
        // exact size so planning sees the real footprint (like sdpa's).
        .memory(gated_delta_rule_memory)
        // CP>1 needs the affine-map all_gather + fp32 merge inside this op —
        // declared (op-vocabulary §4), not derivable from layout arithmetic.
        // tensor_index is nominal: the gathered state is internal, not an io
        // tensor; the declaration records the dependency for the planner.
        .collective(RsCollective {
            kind: RsCollectiveKind::ALL_GATHER,
            group: RsGroupKind::CP,
            tensor_index: 0,
            on_side_stream: 0,
        }))
        .op(spec(
            "moe_layer",
            RsShardRule::PASS_THROUGH,
            MOE_LAYER_DOC,
            F32_IDX,
            RsBackwardKind::AUTODIFF,
            moe_infer,
            moe_exec,
        )
        // The fused body allocates one H-element f32 accumulation buffer per
        // call; report it so planning sees the real footprint.
        .memory(moe_memory)
        // Dispatch and combine are all_to_all({tp, ep}) — declared, not
        // derivable: the routing is data (op-vocabulary §4), which is exactly
        // why the layer is one EXPLICIT operator. tensor_index follows the
        // expansion convention (inputs first, then outputs): 0 = the h input
        // the dispatch sends, 10 = the output (offset n_inputs) the combine
        // assembles.
        .collective(RsCollective {
            kind: RsCollectiveKind::ALL_TO_ALL,
            group: RsGroupKind::from_raw(RsGroupKind::TP.raw() | RsGroupKind::EP.raw()),
            tensor_index: 0,
            on_side_stream: 0,
        })
        .collective(RsCollective {
            kind: RsCollectiveKind::ALL_TO_ALL,
            group: RsGroupKind::from_raw(RsGroupKind::TP.raw() | RsGroupKind::EP.raw()),
            tensor_index: 10,
            on_side_stream: 0,
        }))
        .build()
}

/// The process-lifetime plugin handle. Exposed for the in-process tests,
/// which drive the operator bodies directly instead of going through dlopen.
///
/// `RsPlugin` is `Send + Sync` (declared by `rustrain-abi`, whose argument is
/// the same one that makes `&'static` shared data sound: the descriptor and
/// everything it points to is `*const`, leaked, and never mutated after
/// `build()`).
pub fn plugin() -> &'static RsPlugin {
    static PLUGIN: OnceLock<&'static RsPlugin> = OnceLock::new();
    PLUGIN.get_or_init(build_plugin)
}

/// The single exported symbol of this plugin (contract C-1). Built once over
/// [`PluginBuilder`]; everything the descriptors point to is leaked on
/// purpose, so the returned pointer stays valid for the process lifetime.
///
/// # Safety
/// Safe to call from any thread at any time: the return value is a pointer
/// to a process-lifetime immutable descriptor. (A panic in `build()` would
/// abort, not unwind across the `extern "C"` boundary — the builder asserts
/// on malformed specs, and this static table satisfies them.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustrain_plugin_v1() -> *const RsPlugin {
    plugin() as *const RsPlugin
}
