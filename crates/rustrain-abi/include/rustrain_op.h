/*
 * rustrain_op.h — rustrain operator plugin ABI, version 1.
 *
 * This header is the ONLY contract between the framework and a kernel plugin.
 * It must not depend on libtorch, CUDA headers, or any Rust type.
 *
 * Contract summary (see docs/design/kernel-first/spec.md):
 *   C-1  A plugin exports exactly one symbol: rustrain_plugin_v1().
 *   C-2  abi_version must equal RUSTRAIN_ABI_VERSION; mismatch is a hard error.
 *   C-3  Plugins obtain memory, streams and collectives through rs_services,
 *        never by calling cudaMalloc / ncclAllReduce themselves.
 *   C-4  Everything crossing this boundary is POD. Backend-private pointers
 *        (e.g. at::Tensor*) may only live in rs_tensor.reserved[].
 */
#ifndef RUSTRAIN_OP_H
#define RUSTRAIN_OP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RUSTRAIN_ABI_VERSION 2u
#define RS_MAX_RANK 8u
#define RS_MAX_RESERVED 4u

/* ── data model ─────────────────────────────────────────────────────────── */

typedef enum {
    RS_F32 = 0, RS_F16 = 1, RS_BF16 = 2,
    RS_F8E4M3 = 3, RS_F8E5M2 = 4, RS_FP4E2M1 = 5,
    RS_I32 = 6, RS_I64 = 7, RS_U8 = 8,
    RS_DTYPE_COUNT = 9
} rs_dtype;

typedef enum { RS_CPU = 0, RS_CUDA = 1 } rs_device_kind;

typedef enum {
    RS_AUTODIFF = 0,  /* differentiable by composing the declared expansion */
    RS_EXPLICIT = 1,  /* has a registered backward op (see backward_op) */
    RS_NONDIFF  = 2   /* not differentiable (optimizer, quantize, ...) */
} rs_backward_kind;

/*
 * How a sharded distribution propagates through the operator. The framework
 * owns the derivation algebra; the operator declares which kind it uses, so a
 * new operator never needs a framework-side name table (invariant I-5).
 * Appended in v2, with the `shard` field at the end of rs_op_desc.
 */
typedef enum {
    RS_SHARD_DECLARED = 0,      /* no derivation; slots keep their layouts */
    RS_SHARD_ELEMENTWISE = 1,   /* one shared distribution on every operand */
    RS_SHARD_LINEAR = 2,        /* out = x @ w^T */
    RS_SHARD_EMBEDDING = 3,     /* out = w[ids] */
    RS_SHARD_MATMUL = 4,        /* batched contraction */
    RS_SHARD_PASS_THROUGH = 5   /* outputs inherit input 0, others keep theirs */
} rs_shard_rule;

typedef enum {
    RS_Q_NONE = 0, RS_Q_PER_TENSOR = 1, RS_Q_PER_TOKEN = 2, RS_Q_PER_BLOCK = 3
} rs_quant_kind;

typedef enum {
    RS_SCALE_STATIC = 0, RS_SCALE_DYNAMIC_AMAX = 1, RS_SCALE_DELAYED = 2
} rs_scale_mode;

/*
 * Backend-agnostic tensor descriptor.
 *   data   device (or host) pointer to the element buffer
 *   scale  optional RS_TENSOR* carrying quantization scales
 *   amax   optional RS_TENSOR* carrying dynamic-scaling history
 *   reserved backend-private slots; the aten backend stores at::Tensor* in [0]
 */
typedef struct {
    rs_dtype dtype;
    uint32_t rank;
    int64_t  shape[RS_MAX_RANK];
    int64_t  stride[RS_MAX_RANK];
    void*    data;
    void*    scale;
    void*    amax;
    void*    reserved[RS_MAX_RESERVED];
} rs_tensor;

typedef struct {
    const char* name;
    const char* variant;
    uint32_t    version;
} rs_op_id;

/* Numerics contract. Quantization scheme is DATA, never inferred from shapes. */
typedef struct {
    rs_dtype      in_dtype, out_dtype, accum_dtype, grad_dtype;
    rs_quant_kind quant;
    uint32_t      block_m, block_n;   /* per-block granularity; 0 otherwise */
    rs_dtype      scale_dtype;
    rs_scale_mode scale_mode;
    uint32_t      amax_history;
    uint32_t      _pad;
} rs_numerics;

/* ── attributes ──────────────────────────────────────────────────────────── */

typedef enum {
    RS_ATTR_I64 = 0, RS_ATTR_F64 = 1, RS_ATTR_BOOL = 2,
    RS_ATTR_STR = 3, RS_ATTR_I64S = 4
} rs_attr_kind;

typedef struct {
    const char*    key;
    rs_attr_kind   kind;
    uint32_t       _pad0;
    int64_t        i64;
    double         f64;
    int32_t        boolean;
    uint32_t       _pad1;
    const char*    str;
    const int64_t* i64s;
    uint32_t       n_i64s;
    uint32_t       _pad2;
} rs_attr;

typedef struct {
    const rs_attr* items;
    uint32_t       len;
    uint32_t       _pad;
} rs_attrs;

/* ── declared capabilities ───────────────────────────────────────────────── */

/* dtype_mask: bit i set => RS_dtype i is accepted. */
typedef struct {
    uint32_t dtype_mask;
    uint32_t min_sm;         /* 0 = no constraint */
    int64_t  min_world_size; /* 0 = no constraint */
    uint32_t needs_groups;   /* bitmask of rs_group_kind; 0 = none */
    uint32_t _pad;
} rs_requires;

typedef enum { RS_G_TP = 1u, RS_G_EP = 2u, RS_G_CP = 4u, RS_G_DP = 8u } rs_group_kind;
typedef enum {
    RS_C_ALL_REDUCE = 0, RS_C_ALL_GATHER = 1,
    RS_C_REDUCE_SCATTER = 2, RS_C_SEND_RECV = 3,
    /* Appended (D5); existing discriminants are unchanged. */
    RS_C_ALL_TO_ALL = 4
} rs_collective_kind;

typedef struct {
    rs_collective_kind kind;
    rs_group_kind      group;
    uint32_t           tensor_index; /* index into the op's io list */
    uint32_t           on_side_stream;
} rs_collective;

typedef struct {
    uint64_t workspace_bytes;
    uint64_t save_for_backward_bytes;
    uint32_t save_tensor_count;
    uint32_t _pad;
} rs_mem_req;

/*
 * Declarative expansion of a composite/fused operator into primitives.
 *
 * Local tensor id space:
 *   [0, n_inputs)                          -> the parent op's inputs, in order
 *   [n_inputs, n_inputs + n_outputs)       -> the parent op's outputs, in order
 *   [n_inputs + n_outputs, n_tensors)      -> temporaries
 */
typedef struct {
    const char*      op;
    const rs_attrs*  attrs;
    const int32_t*   inputs;
    uint32_t         n_inputs;
    const int32_t*   outputs;
    uint32_t         n_outputs;
} rs_expansion_node;

typedef struct {
    uint32_t                 n_nodes;
    uint32_t                 _pad;
    const rs_expansion_node* nodes;
    uint32_t                 n_tensors;
    uint32_t                 n_inputs;
    uint32_t                 n_outputs;
    uint32_t                 _pad2;
} rs_expansion;

/* ── services injected by the framework ──────────────────────────────────── */

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    void*    user;

    void* (*alloc)(void* user, uint64_t bytes, int32_t device);
    void  (*free)(void* user, void* ptr);
    void* (*current_stream)(void* user, int32_t device);
    int32_t (*collective)(void* user, rs_collective_kind kind, rs_group_kind group,
                          void* tensor, void* comm);
    void  (*log)(void* user, int32_t level, const char* msg);
} rs_services;

typedef struct {
    void*                user;
    const rs_services*   svc;
} rs_ctx;

/* ── the operator descriptor ─────────────────────────────────────────────── */

typedef struct {
    uint32_t            abi_version;
    uint32_t            struct_size;
    rs_op_id            id;
    const char*         doc;

    const rs_requires*  requires;
    rs_numerics         numerics;

    /* Shape/type inference. Outputs are pre-allocated descriptors to fill in. */
    int32_t (*infer)(const rs_tensor* const* in, uint32_t n_in,
                     rs_tensor* const* out, uint32_t n_out,
                     const rs_attrs* attrs);

    /* Declared memory footprint. */
    int32_t (*memory)(const rs_tensor* const* io, uint32_t n_io,
                      const rs_attrs* attrs, rs_mem_req* out);

    const rs_expansion* expansion;

    rs_backward_kind    backward;
    rs_op_id            backward_op;

    const rs_collective* collectives;
    uint32_t             n_collectives;

    /* The call itself. Returns 0 on success. */
    int32_t (*execute)(rs_ctx* ctx,
                       const rs_tensor* const* in, uint32_t n_in,
                       rs_tensor* const* out, uint32_t n_out,
                       const rs_attrs* attrs);

    const char* (*last_error)(rs_ctx* ctx);

    /* v2: how this operator's sharded distribution propagates. Appended, so a
     * v1 descriptor is shorter and is rejected by struct_size before this
     * field is read. Declare it — the framework has no name table (I-5). */
    rs_shard_rule       shard;
} rs_op_desc;

typedef struct {
    uint32_t                abi_version;
    uint32_t                struct_size;
    const char*             plugin_name;
    const char*             plugin_version;
    uint32_t                n_ops;
    uint32_t                _pad;
    const rs_op_desc* const* ops;
    /* Optional. Called once at load time with the service table. */
    int32_t (*init)(rs_services* svc);
} rs_plugin;

/* ── the single exported entry point ─────────────────────────────────────── */

typedef const rs_plugin* (*rustrain_plugin_v1_fn)(void);

#ifdef __cplusplus
}
#endif
#endif /* RUSTRAIN_OP_H */
