// glm5_attention.cpp — C++ implementation of GLM-5.2 DSA attention
//
// Migrated from Rust (model.rs glm5_dsa_attention) to C++ for:
// - Coarse-grained kernel fusion (one FFI call per layer instead of ~30)
// - Direct CUDA stream control (for async overlap with NCCL)
// - No tch-rs dependency in compute path
//
// All intermediate tensors live on C++ stack — zero FFI crossings per operation.

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/transpose.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/scatter.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/log_softmax.h>
#include <ATen/ops/nll_loss.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/where.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/stack.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/csrc/autograd/custom_function.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <optional>

// Keep this kernel independent of the NCCL headers.  The Rust NCCL crate owns
// the communicator; we only need the stable send/recv ABI here.  The symbols
// are resolved from libnccl already linked by rustrain-nccl.
using ncclComm_t = void*;
using ncclResult_t = int;
using ncclDataType_t = int;
using ncclRedOp_t = int;
constexpr ncclDataType_t kNcclFloat32 = 7;
constexpr ncclDataType_t kNcclBfloat16 = 9;
constexpr ncclDataType_t kNcclInt64 = 4;
constexpr ncclRedOp_t kNcclSum = 0;
constexpr ncclRedOp_t kNcclMax = 2;
constexpr ncclResult_t kNcclSuccess = 0;
extern "C" {
ncclResult_t ncclGroupStart();
ncclResult_t ncclGroupEnd();
ncclResult_t ncclSend(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
ncclResult_t ncclRecv(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
ncclResult_t ncclAllReduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t,
                          ncclComm_t, cudaStream_t);
ncclResult_t ncclAllGather(const void*, void*, size_t, ncclDataType_t,
                          ncclComm_t, cudaStream_t);
}

static ncclDataType_t glm5_nccl_dtype(const at::Tensor& t) {
    TORCH_CHECK(t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat ||
                t.scalar_type() == at::kLong,
                "NCCL tensor must use BF16, FP32, or int64");
    if (t.scalar_type() == at::kLong) return kNcclInt64;
    return t.scalar_type() == at::kBFloat16 ? kNcclBfloat16 : kNcclFloat32;
}

// NCCL is not represented as a PyTorch op, so a raw ring receive otherwise
// severs the graph.  This custom Function performs the reverse ring in
// backward: grad(y_rank) is sent to the owner of the received block, while
// grad(x_rank) is received from the rank that consumed this block.
struct Glm5NcclRingFunction : public torch::autograd::Function<Glm5NcclRingFunction> {
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                              at::Tensor input, int64_t comm_ptr,
                              int64_t send_peer, int64_t recv_peer) {
        TORCH_CHECK(input.is_cuda(), "autograd ring requires CUDA input");
        auto input_c = input.contiguous();
        auto comm = reinterpret_cast<ncclComm_t>(comm_ptr);
        auto stream = c10::cuda::getCurrentCUDAStream(input_c.device().index()).stream();
        auto dtype = glm5_nccl_dtype(input_c);
        auto output = at::empty_like(input_c);
        TORCH_CHECK(ncclGroupStart() == kNcclSuccess, "ncclGroupStart failed");
        auto send_rc = ncclSend(input_c.data_ptr(), input_c.numel(), dtype,
                                static_cast<int>(send_peer), comm, stream);
        auto recv_rc = ncclRecv(output.data_ptr(), output.numel(), dtype,
                                static_cast<int>(recv_peer), comm, stream);
        auto end_rc = ncclGroupEnd();
        TORCH_CHECK(send_rc == kNcclSuccess && recv_rc == kNcclSuccess && end_rc == kNcclSuccess,
                    "NCCL autograd ring forward failed");
        ctx->saved_data["comm"] = comm_ptr;
        ctx->saved_data["send_peer"] = send_peer;
        ctx->saved_data["recv_peer"] = recv_peer;
        return output;
    }

    static std::vector<at::Tensor> backward(torch::autograd::AutogradContext* ctx,
                                            std::vector<at::Tensor> grad_output) {
        auto grad = grad_output[0].contiguous();
        auto comm = reinterpret_cast<ncclComm_t>(ctx->saved_data["comm"].toInt());
        auto stream = c10::cuda::getCurrentCUDAStream(grad.device().index()).stream();
        auto send_peer = static_cast<int>(ctx->saved_data["send_peer"].toInt());
        auto recv_peer = static_cast<int>(ctx->saved_data["recv_peer"].toInt());
        auto dtype = glm5_nccl_dtype(grad);
        auto grad_input = at::empty_like(grad);
        // Reverse the forward edges: send this rank's output gradient back to
        // the rank that supplied its input, then receive the gradient for the
        // input block sent to the forward receiver.
        TORCH_CHECK(ncclGroupStart() == kNcclSuccess, "ncclGroupStart failed");
        auto send_rc = ncclSend(grad.data_ptr(), grad.numel(), dtype,
                                recv_peer, comm, stream);
        auto recv_rc = ncclRecv(grad_input.data_ptr(), grad_input.numel(), dtype,
                                send_peer, comm, stream);
        auto end_rc = ncclGroupEnd();
        TORCH_CHECK(send_rc == kNcclSuccess && recv_rc == kNcclSuccess && end_rc == kNcclSuccess,
                    "NCCL autograd ring backward failed");
        return {grad_input, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

struct Glm5NcclKvRingFunction : public torch::autograd::Function<Glm5NcclKvRingFunction> {
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor key,
        at::Tensor value,
        int64_t comm_ptr,
        int64_t send_peer,
        int64_t recv_peer) {
        TORCH_CHECK(key.is_cuda() && value.is_cuda(), "autograd KV ring requires CUDA inputs");
        TORCH_CHECK(key.device() == value.device(),
                    "autograd KV ring key/value devices must match");
        TORCH_CHECK(key.dim() == value.dim() && key.dim() >= 1,
                    "autograd KV ring key/value ranks must match");
        for (int64_t dim = 0; dim + 1 < key.dim(); ++dim) {
            TORCH_CHECK(key.size(dim) == value.size(dim),
                        "autograd KV ring key/value non-feature dimensions must match");
        }
        auto key_c = key.contiguous();
        auto value_c = value.contiguous();
        TORCH_CHECK(key_c.scalar_type() == value_c.scalar_type(),
                    "autograd KV ring key/value dtypes must match");
        auto comm = reinterpret_cast<ncclComm_t>(comm_ptr);
        auto stream = c10::cuda::getCurrentCUDAStream(key_c.device().index()).stream();
        auto dtype = glm5_nccl_dtype(key_c);
        auto recv_key = at::empty_like(key_c);
        auto recv_value = at::empty_like(value_c);
        TORCH_CHECK(ncclGroupStart() == kNcclSuccess, "ncclGroupStart failed");
        auto send_key_rc = ncclSend(key_c.data_ptr(), key_c.numel(), dtype,
                                    static_cast<int>(send_peer), comm, stream);
        auto send_value_rc = ncclSend(value_c.data_ptr(), value_c.numel(), dtype,
                                      static_cast<int>(send_peer), comm, stream);
        auto recv_key_rc = ncclRecv(recv_key.data_ptr(), recv_key.numel(), dtype,
                                    static_cast<int>(recv_peer), comm, stream);
        auto recv_value_rc = ncclRecv(recv_value.data_ptr(), recv_value.numel(), dtype,
                                      static_cast<int>(recv_peer), comm, stream);
        auto end_rc = ncclGroupEnd();
        TORCH_CHECK(send_key_rc == kNcclSuccess && send_value_rc == kNcclSuccess &&
                    recv_key_rc == kNcclSuccess && recv_value_rc == kNcclSuccess &&
                    end_rc == kNcclSuccess, "NCCL autograd KV ring forward failed");
        ctx->saved_data["comm"] = comm_ptr;
        ctx->saved_data["send_peer"] = send_peer;
        ctx->saved_data["recv_peer"] = recv_peer;
        return {std::move(recv_key), std::move(recv_value)};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output) {
        auto grad_key = grad_output[0].contiguous();
        auto grad_value = grad_output[1].contiguous();
        auto comm = reinterpret_cast<ncclComm_t>(ctx->saved_data["comm"].toInt());
        auto stream = c10::cuda::getCurrentCUDAStream(grad_key.device().index()).stream();
        auto send_peer = static_cast<int>(ctx->saved_data["send_peer"].toInt());
        auto recv_peer = static_cast<int>(ctx->saved_data["recv_peer"].toInt());
        auto dtype = glm5_nccl_dtype(grad_key);
        auto grad_key_input = at::empty_like(grad_key);
        auto grad_value_input = at::empty_like(grad_value);
        TORCH_CHECK(ncclGroupStart() == kNcclSuccess, "ncclGroupStart failed");
        auto send_key_rc = ncclSend(grad_key.data_ptr(), grad_key.numel(), dtype,
                                    recv_peer, comm, stream);
        auto send_value_rc = ncclSend(grad_value.data_ptr(), grad_value.numel(), dtype,
                                      recv_peer, comm, stream);
        auto recv_key_rc = ncclRecv(grad_key_input.data_ptr(), grad_key_input.numel(), dtype,
                                    send_peer, comm, stream);
        auto recv_value_rc = ncclRecv(grad_value_input.data_ptr(), grad_value_input.numel(), dtype,
                                      send_peer, comm, stream);
        auto end_rc = ncclGroupEnd();
        TORCH_CHECK(send_key_rc == kNcclSuccess && send_value_rc == kNcclSuccess &&
                    recv_key_rc == kNcclSuccess && recv_value_rc == kNcclSuccess &&
                    end_rc == kNcclSuccess, "NCCL autograd KV ring backward failed");
        return {
            std::move(grad_key_input),
            std::move(grad_value_input),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
        };
    }
};

// TP/EP output tensors are already replicated after the forward SUM.  Their
// backward must remain local: the outer training loop synchronizes parameter
// gradients, while another all-reduce here would multiply activation gradients
// by the group size.  This Function exposes exactly that SUM-forward / identity-
// backward contract to autograd.
struct Glm5NcclAllReduceIdentityBackward
    : public torch::autograd::Function<Glm5NcclAllReduceIdentityBackward> {
    static at::Tensor forward(torch::autograd::AutogradContext*, at::Tensor input,
                              int64_t comm_ptr) {
        TORCH_CHECK(input.is_cuda(), "autograd all-reduce requires CUDA input");
        auto input_c = input.contiguous();
        auto output = at::empty_like(input_c);
        auto comm = reinterpret_cast<ncclComm_t>(comm_ptr);
        auto stream = c10::cuda::getCurrentCUDAStream(input_c.device().index()).stream();
        auto rc = ncclAllReduce(input_c.data_ptr(), output.data_ptr(), input_c.numel(),
                                glm5_nccl_dtype(input_c), kNcclSum, comm, stream);
        TORCH_CHECK(rc == kNcclSuccess, "NCCL autograd all-reduce forward failed");
        return output;
    }

    static std::vector<at::Tensor> backward(torch::autograd::AutogradContext*,
                                            std::vector<at::Tensor> grad_output) {
        return {grad_output[0], at::Tensor()};
    }
};

// Tensor-parallel linear layers keep a vocabulary shard on each rank.  The
// forward input is replicated, while its gradient must be summed across TP
// ranks (the ColumnParallelLinear copy-to-TP contract in Megatron).
struct Glm5NcclIdentityAllReduceBackward
    : public torch::autograd::Function<Glm5NcclIdentityAllReduceBackward> {
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor input,
                              int64_t comm_ptr) {
        TORCH_CHECK(input.is_cuda(), "autograd TP identity requires CUDA input");
        ctx->saved_data["comm"] = comm_ptr;
        return input;
    }

    static std::vector<at::Tensor> backward(torch::autograd::AutogradContext* ctx,
                                            std::vector<at::Tensor> grad_output) {
        auto grad = grad_output[0].contiguous();
        auto comm = reinterpret_cast<ncclComm_t>(ctx->saved_data["comm"].toInt());
        auto stream = c10::cuda::getCurrentCUDAStream(grad.device().index()).stream();
        auto reduced = at::empty_like(grad);
        auto rc = ncclAllReduce(grad.data_ptr(), reduced.data_ptr(), grad.numel(),
                                glm5_nccl_dtype(grad), kNcclSum, comm, stream);
        TORCH_CHECK(rc == kNcclSuccess, "NCCL TP identity backward failed");
        return {reduced, at::Tensor()};
    }
};

// ColumnParallelLinear(gather_output=true): NCCL lays out all-gather output as
// [rank, B, S, H_local]. Reorder it to [B, S, H] in forward and select this
// rank's feature slice in backward. The replicated linear input is wrapped in
// Glm5NcclIdentityAllReduceBackward separately so its partial input gradients
// are summed across TP ranks.
struct Glm5NcclAllGatherSplitBackward
    : public torch::autograd::Function<Glm5NcclAllGatherSplitBackward> {
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor input,
                              int64_t comm_ptr, int64_t tp_rank, int64_t tp_size) {
        TORCH_CHECK(input.dim() == 3, "TP all-gather input must have shape [B,S,H_local]");
        TORCH_CHECK(input.is_cuda(), "TP all-gather requires CUDA input");
        TORCH_CHECK(tp_size > 1 && tp_rank >= 0 && tp_rank < tp_size,
                    "invalid TP all-gather rank or size");
        auto input_c = input.contiguous();
        auto gathered = at::empty(
            {tp_size, input_c.size(0), input_c.size(1), input_c.size(2)},
            input_c.options());
        auto comm = reinterpret_cast<ncclComm_t>(comm_ptr);
        auto stream = c10::cuda::getCurrentCUDAStream(input_c.device().index()).stream();
        auto rc = ncclAllGather(input_c.data_ptr(), gathered.data_ptr(), input_c.numel(),
                                glm5_nccl_dtype(input_c), comm, stream);
        TORCH_CHECK(rc == kNcclSuccess, "NCCL TP all-gather forward failed");
        ctx->saved_data["tp_rank"] = tp_rank;
        ctx->saved_data["tp_size"] = tp_size;
        ctx->saved_data["batch"] = input_c.size(0);
        ctx->saved_data["seq"] = input_c.size(1);
        ctx->saved_data["local_hidden"] = input_c.size(2);
        return gathered.permute({1, 2, 0, 3}).reshape(
            {input_c.size(0), input_c.size(1), tp_size * input_c.size(2)});
    }

    static std::vector<at::Tensor> backward(torch::autograd::AutogradContext* ctx,
                                            std::vector<at::Tensor> grad_output) {
        const auto tp_rank = ctx->saved_data["tp_rank"].toInt();
        const auto tp_size = ctx->saved_data["tp_size"].toInt();
        const auto batch = ctx->saved_data["batch"].toInt();
        const auto seq = ctx->saved_data["seq"].toInt();
        const auto local_hidden = ctx->saved_data["local_hidden"].toInt();
        auto grad = grad_output[0].reshape({batch, seq, tp_size, local_hidden});
        auto local_grad = grad.select(2, tp_rank).contiguous();
        return {local_grad, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

static at::Tensor glm5_ep_variable_exchange(
    const at::Tensor& input,
    const at::Tensor& send_counts,
    const at::Tensor& recv_counts,
    int64_t comm_ptr,
    int64_t ep_rank,
    int64_t ep_size) {
    TORCH_CHECK(input.is_cuda() && send_counts.is_cuda() && recv_counts.is_cuda(),
                "EP exchange tensors must be CUDA tensors");
    TORCH_CHECK(input.dim() >= 1 && send_counts.dim() == 1 && recv_counts.dim() == 1,
                "invalid EP exchange tensor ranks");
    TORCH_CHECK(send_counts.scalar_type() == at::kLong &&
                recv_counts.scalar_type() == at::kLong,
                "EP exchange counts must be int64");
    TORCH_CHECK(send_counts.numel() == ep_size && recv_counts.numel() == ep_size,
                "EP exchange count vector size mismatch");
    TORCH_CHECK(ep_size > 1 && ep_rank >= 0 && ep_rank < ep_size && comm_ptr != 0,
                "invalid EP exchange communicator coordinates");

    auto input_c = input.contiguous();
    auto send_cpu = send_counts.to(at::kCPU).contiguous();
    auto recv_cpu = recv_counts.to(at::kCPU).contiguous();
    auto send_a = send_cpu.accessor<int64_t, 1>();
    auto recv_a = recv_cpu.accessor<int64_t, 1>();
    int64_t total_send = 0;
    int64_t total_recv = 0;
    for (int64_t peer = 0; peer < ep_size; ++peer) {
        TORCH_CHECK(send_a[peer] >= 0 && recv_a[peer] >= 0,
                    "EP exchange counts must be non-negative");
        total_send += send_a[peer];
        total_recv += recv_a[peer];
    }
    TORCH_CHECK(total_send == input_c.size(0),
                "EP exchange send counts do not cover the input");
    auto output_shape = input_c.sizes().vec();
    output_shape[0] = total_recv;
    auto output = at::empty(output_shape, input_c.options());
    int64_t row_width = 1;
    for (int64_t dim = 1; dim < input_c.dim(); ++dim) {
        row_width *= input_c.size(dim);
    }
    auto comm = reinterpret_cast<ncclComm_t>(comm_ptr);
    auto stream = c10::cuda::getCurrentCUDAStream(input_c.device().index()).stream();
    auto dtype = glm5_nccl_dtype(input_c);
    auto* send_base = static_cast<char*>(input_c.data_ptr());
    auto* recv_base = static_cast<char*>(output.data_ptr());
    const int64_t element_size = input_c.element_size();
    int64_t send_offset = 0;
    int64_t recv_offset = 0;

    TORCH_CHECK(ncclGroupStart() == kNcclSuccess, "ncclGroupStart failed");
    ncclResult_t exchange_rc = kNcclSuccess;
    for (int64_t peer = 0; peer < ep_size; ++peer) {
        const int64_t send_rows = send_a[peer];
        const int64_t recv_rows = recv_a[peer];
        if (peer == ep_rank) {
            TORCH_CHECK(send_rows == recv_rows,
                        "EP self-exchange send/receive counts differ");
            if (send_rows > 0) {
                output.narrow(0, recv_offset, recv_rows)
                    .copy_(input_c.narrow(0, send_offset, send_rows));
            }
        } else {
            if (send_rows > 0) {
                auto rc = ncclSend(
                    send_base + send_offset * row_width * element_size,
                    send_rows * row_width, dtype, static_cast<int>(peer), comm, stream);
                if (rc != kNcclSuccess) exchange_rc = rc;
            }
            if (recv_rows > 0) {
                auto rc = ncclRecv(
                    recv_base + recv_offset * row_width * element_size,
                    recv_rows * row_width, dtype, static_cast<int>(peer), comm, stream);
                if (rc != kNcclSuccess) exchange_rc = rc;
            }
        }
        send_offset += send_rows;
        recv_offset += recv_rows;
    }
    auto end_rc = ncclGroupEnd();
    TORCH_CHECK(exchange_rc == kNcclSuccess && end_rc == kNcclSuccess,
                "NCCL EP variable exchange failed");
    return output;
}

// Dispatch sorted token assignments from their origin ranks to expert owners.
// The reverse exchange in backward returns expert-input gradients to origins.
struct Glm5NcclEpDispatchFunction
    : public torch::autograd::Function<Glm5NcclEpDispatchFunction> {
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor sorted_input,
        at::Tensor sorted_experts,
        at::Tensor send_counts,
        int64_t comm_ptr,
        int64_t ep_rank,
        int64_t ep_size) {
        TORCH_CHECK(sorted_input.size(0) == sorted_experts.numel(),
                    "EP dispatch input/expert assignment count mismatch");
        TORCH_CHECK(sorted_experts.scalar_type() == at::kLong,
                    "EP dispatch expert IDs must be int64");
        auto counts_matrix = at::empty({ep_size, ep_size}, send_counts.options());
        auto stream = c10::cuda::getCurrentCUDAStream(sorted_input.device().index()).stream();
        auto rc = ncclAllGather(send_counts.data_ptr(), counts_matrix.data_ptr(), ep_size,
                                kNcclInt64, reinterpret_cast<ncclComm_t>(comm_ptr), stream);
        TORCH_CHECK(rc == kNcclSuccess, "NCCL EP count all-gather failed");
        auto recv_counts = counts_matrix.select(1, ep_rank).contiguous();
        auto received_input = glm5_ep_variable_exchange(
            sorted_input, send_counts, recv_counts, comm_ptr, ep_rank, ep_size);
        auto received_experts = glm5_ep_variable_exchange(
            sorted_experts, send_counts, recv_counts, comm_ptr, ep_rank, ep_size);
        ctx->save_for_backward({send_counts, recv_counts});
        ctx->saved_data["comm"] = comm_ptr;
        ctx->saved_data["ep_rank"] = ep_rank;
        ctx->saved_data["ep_size"] = ep_size;
        ctx->mark_non_differentiable({received_experts, recv_counts});
        return {received_input, received_experts, recv_counts};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output) {
        auto saved = ctx->get_saved_variables();
        auto send_counts = saved[0];
        auto recv_counts = saved[1];
        auto comm_ptr = ctx->saved_data["comm"].toInt();
        auto ep_rank = ctx->saved_data["ep_rank"].toInt();
        auto ep_size = ctx->saved_data["ep_size"].toInt();
        auto grad_input = glm5_ep_variable_exchange(
            grad_output[0], recv_counts, send_counts, comm_ptr, ep_rank, ep_size);
        return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

// Return owner-computed expert outputs to their origin ranks. Backward repeats
// the forward dispatch direction for the expert-output gradients.
struct Glm5NcclEpReturnFunction
    : public torch::autograd::Function<Glm5NcclEpReturnFunction> {
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor received_output,
        at::Tensor send_counts,
        at::Tensor recv_counts,
        int64_t comm_ptr,
        int64_t ep_rank,
        int64_t ep_size) {
        ctx->save_for_backward({send_counts, recv_counts});
        ctx->saved_data["comm"] = comm_ptr;
        ctx->saved_data["ep_rank"] = ep_rank;
        ctx->saved_data["ep_size"] = ep_size;
        return glm5_ep_variable_exchange(
            received_output, recv_counts, send_counts, comm_ptr, ep_rank, ep_size);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output) {
        auto saved = ctx->get_saved_variables();
        auto send_counts = saved[0];
        auto recv_counts = saved[1];
        auto comm_ptr = ctx->saved_data["comm"].toInt();
        auto ep_rank = ctx->saved_data["ep_rank"].toInt();
        auto ep_size = ctx->saved_data["ep_size"].toInt();
        auto grad_received = glm5_ep_variable_exchange(
            grad_output[0], send_counts, recv_counts, comm_ptr, ep_rank, ep_size);
        return {grad_received, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

// Megatron's VocabParallelCrossEntropy, expressed as an autograd Function so
// the cross-rank statistics are not detached from the local logits.  Forward
// all-reduces MAX(logits), SUM(exp(logits-max)), and the target logit.  Backward
// is local (softmax minus the local target one-hot), exactly as in Megatron.
struct Glm5VocabParallelCrossEntropy
    : public torch::autograd::Function<Glm5VocabParallelCrossEntropy> {
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor logits,
                              at::Tensor targets, int64_t vocab_start,
                              int64_t global_vocab_size, int64_t comm_ptr,
                              int64_t tp_size) {
        TORCH_CHECK(logits.dim() >= 1 && targets.dim() == logits.dim() - 1,
                    "vocab-parallel CE shape mismatch");
        TORCH_CHECK(targets.scalar_type() == at::kLong,
                    "vocab-parallel CE targets must be int64");
        TORCH_CHECK(global_vocab_size > 0 && tp_size > 0,
                    "invalid vocab-parallel CE configuration");
        TORCH_CHECK(logits.size(-1) > 0 && vocab_start >= 0 &&
                    vocab_start + logits.size(-1) <= global_vocab_size,
                    "invalid local vocabulary range");
        TORCH_CHECK(tp_size == 1 || (logits.is_cuda() && comm_ptr != 0),
                    "multi-rank vocab-parallel CE requires CUDA NCCL communicator");

        auto logits_f = logits.to(at::kFloat);
        auto local_max = std::get<0>(logits_f.max(-1, false)).contiguous();
        at::Tensor global_max;
        if (tp_size == 1) {
            global_max = local_max;
        } else {
            global_max = at::empty_like(local_max);
            auto stream = c10::cuda::getCurrentCUDAStream(logits.device().index()).stream();
            auto rc = ncclAllReduce(local_max.data_ptr(), global_max.data_ptr(), local_max.numel(),
                                    kNcclFloat32, kNcclMax,
                                    reinterpret_cast<ncclComm_t>(comm_ptr), stream);
            TORCH_CHECK(rc == kNcclSuccess, "NCCL vocab-parallel MAX failed");
        }

        auto shifted = logits_f - global_max.unsqueeze(-1);
        auto exp_local = shifted.exp();
        auto sum_local = exp_local.sum(-1).contiguous();
        at::Tensor sum_global;
        if (tp_size == 1) {
            sum_global = sum_local;
        } else {
            sum_global = at::empty_like(sum_local);
            auto stream = c10::cuda::getCurrentCUDAStream(logits.device().index()).stream();
            auto rc = ncclAllReduce(sum_local.data_ptr(), sum_global.data_ptr(), sum_local.numel(),
                                    kNcclFloat32, kNcclSum,
                                    reinterpret_cast<ncclComm_t>(comm_ptr), stream);
            TORCH_CHECK(rc == kNcclSuccess, "NCCL vocab-parallel SUM(exp) failed");
        }

        auto local_end = vocab_start + logits.size(-1);
        auto target_mask = (targets < vocab_start) | (targets >= local_end);
        auto local_target = (targets - vocab_start).masked_fill(target_mask, 0);
        auto target_logits = logits_f.gather(-1, local_target.unsqueeze(-1)).squeeze(-1);
        target_logits = target_logits.masked_fill(target_mask, 0.0).contiguous();
        at::Tensor target_global;
        if (tp_size == 1) {
            target_global = target_logits;
        } else {
            target_global = at::empty_like(target_logits);
            auto stream = c10::cuda::getCurrentCUDAStream(logits.device().index()).stream();
            auto rc = ncclAllReduce(target_logits.data_ptr(), target_global.data_ptr(),
                                    target_logits.numel(), kNcclFloat32, kNcclSum,
                                    reinterpret_cast<ncclComm_t>(comm_ptr), stream);
            TORCH_CHECK(rc == kNcclSuccess, "NCCL vocab-parallel target SUM failed");
        }

        auto softmax = exp_local / sum_global.unsqueeze(-1);
        ctx->save_for_backward({softmax, local_target, target_mask});
        ctx->saved_data["logits_dtype"] = static_cast<int64_t>(logits.scalar_type());
        return sum_global.log() + global_max - target_global;
    }

    static std::vector<at::Tensor> backward(torch::autograd::AutogradContext* ctx,
                                            std::vector<at::Tensor> grad_output) {
        auto saved = ctx->get_saved_variables();
        auto softmax = saved[0];
        auto local_target = saved[1];
        auto target_mask = saved[2];
        auto grad = grad_output[0].unsqueeze(-1).to(softmax.scalar_type());
        auto grad_logits = softmax;
        auto target_one_hot = at::zeros_like(grad_logits);
        target_one_hot.scatter_(-1, local_target.unsqueeze(-1),
                                target_mask.logical_not().to(grad_logits.scalar_type()).unsqueeze(-1));
        grad_logits = (grad_logits - target_one_hot) * grad;
        auto original_dtype = static_cast<at::ScalarType>(ctx->saved_data["logits_dtype"].toInt());
        return {grad_logits.to(original_dtype), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor()};
    }
};

extern "C" {

static thread_local std::string g_glm5_last_error;

static void set_glm5_error(const char* operation, const std::exception& error) {
    g_glm5_last_error = std::string(operation) + ": " + error.what();
    fprintf(stderr, "[%s] FAILED: %s\n", operation, error.what());
}

const char* v4_glm5_last_error() {
    return g_glm5_last_error.c_str();
}

// ── Helper: RMSNorm ──
static at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double eps) {
    auto dtype = input.scalar_type();
    auto w = weight.to(dtype);
    // variance = mean(input^2, -1, keepdim)
    auto sq = input.pow(2.0);
    auto variance = sq.mean(-1, /*keepdim=*/true);
    auto result = input * (variance + eps).rsqrt().to(dtype) * w;
    return result.to(dtype);
}

// ── Helper: RMSNorm with bias (indexer k_norm) ──
static at::Tensor rms_norm_with_bias(const at::Tensor& input, const at::Tensor& weight,
                                       const at::Tensor& bias, double eps) {
    auto dtype = input.scalar_type();
    auto w = weight.to(dtype);
    auto b = bias.to(dtype);
    auto sq = input.pow(2.0);
    auto variance = sq.mean(-1, /*keepdim=*/true);
    auto result = input * (variance + eps).rsqrt().to(dtype) * w + b;
    return result.to(dtype);
}

// ── Helper: RoPE cos/sin ──
// ── RoPE cache: avoid recomputing cos/sin every layer (78 layers × 2 calls/layer = 156×/step) ──
#include <map>
#include <tuple>
using RopeCacheKey = std::tuple<int64_t, int64_t, double, int, bool, double, double,
                                double, int64_t, double>;
static std::map<RopeCacheKey, std::pair<at::Tensor, at::Tensor>> g_rope_cache;

static std::pair<at::Tensor, at::Tensor> rope_cos_sin(int64_t seq_len, int64_t head_dim,
                                                       double theta, int32_t device_id,
                                                       bool is_yarn = false,
                                                       double scaling_factor = 1.0,
                                                       double beta_fast = 32.0,
                                                       double beta_slow = 1.0,
                                                       int64_t original_max_pos = 0,
                                                       double attention_factor = 1.0) {
    static std::mutex rope_cache_mutex;
    std::lock_guard<std::mutex> lock(rope_cache_mutex);
    auto key = std::make_tuple(seq_len, head_dim, theta, device_id, is_yarn,
                               scaling_factor, beta_fast, beta_slow,
                               original_max_pos, attention_factor);
    auto it = g_rope_cache.find(key);
    if (it != g_rope_cache.end()) {
        return it->second;
    }
    auto device = device_id >= 0
        ? at::Device(at::Device::Type::CUDA, device_id)
        : at::Device(at::Device::Type::CPU);
    auto positions = at::arange(seq_len, at::TensorOptions().dtype(at::kFloat).device(device));
    auto dim_indices = at::arange(head_dim / 2, at::TensorOptions().dtype(at::kFloat).device(device));
    TORCH_CHECK(theta > 0.0, "rope_theta must be positive");
    TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0, "RoPE head_dim must be positive and even");
    // theta^(-(2*i/head_dim)) == exp(-log(theta) * 2*i/head_dim)
    auto inv_freq = at::exp(dim_indices * (-2.0 * std::log(theta) / (double)head_dim));
    if (is_yarn) {
        TORCH_CHECK(scaling_factor > 1.0 && original_max_pos > 0 &&
                    beta_fast > beta_slow && beta_slow > 0.0,
                    "invalid YaRN scaling parameters");
        constexpr double kPi = 3.14159265358979323846;
        const auto correction_dim = [&](double rotations) {
            return static_cast<double>(head_dim) *
                std::log(static_cast<double>(original_max_pos) /
                         (rotations * 2.0 * kPi)) /
                (2.0 * std::log(theta));
        };
        const double low = std::max(std::floor(correction_dim(beta_fast)), 0.0);
        double high = std::min(std::ceil(correction_dim(beta_slow)),
                               static_cast<double>(head_dim - 1));
        if (std::abs(high - low) < std::numeric_limits<double>::epsilon()) {
            high += 0.001;
        }
        auto ramp = ((dim_indices - low) / (high - low)).clamp(0.0, 1.0);
        auto interpolation = inv_freq / scaling_factor;
        auto extrapolation_weight = 1.0 - ramp;
        inv_freq = interpolation * (1.0 - extrapolation_weight) +
                   inv_freq * extrapolation_weight;
    }
    auto freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0); // [S, D/2]
    auto cos = at::cos(freqs) * attention_factor;
    auto sin = at::sin(freqs) * attention_factor;
    cos = at::cat({cos, cos}, -1);
    sin = at::cat({sin, sin}, -1);
    auto result = std::make_pair(cos, sin);
    g_rope_cache[key] = result;
    return result;
}

// ── Helper: apply rotary interleave ──
static at::Tensor apply_rotary_interleave(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin) {
    int64_t seq = x.size(2);
    int64_t half = x.size(-1) / 2;
    auto x_even = x.slice(-1, 0, at::nullopt, 2);
    auto x_odd = x.slice(-1, 1, at::nullopt, 2);
    auto rotated = at::stack({x_odd.neg(), x_even}, -1).flatten(-2);
    auto cos_half = cos.narrow(0, 0, seq).narrow(-1, 0, half);
    auto sin_half = sin.narrow(0, 0, seq).narrow(-1, 0, half);
    auto c = at::stack({cos_half, cos_half}, -1).flatten(-2).unsqueeze(0).unsqueeze(0);
    auto s = at::stack({sin_half, sin_half}, -1).flatten(-2).unsqueeze(0).unsqueeze(0);
    return x * c + rotated * s;
}

// ── Helper: apply rotary (non-interleave) ──
static at::Tensor apply_rotary(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin) {
    int64_t seq = x.size(2);
    auto c = cos.narrow(0, 0, seq).unsqueeze(0).unsqueeze(0);
    auto s = sin.narrow(0, 0, seq).unsqueeze(0).unsqueeze(0);
    int64_t half = x.size(-1) / 2;
    auto x1 = x.narrow(-1, 0, half);
    auto x2 = x.narrow(-1, half, half);
    auto rotated = at::cat({x2.neg(), x1}, -1);
    return x * c + rotated * s;
}

static at::Tensor apply_indexer_rope(const at::Tensor& value, const at::Tensor& cos,
                                     const at::Tensor& sin, int64_t rope_dim,
                                     bool interleave) {
    int64_t head_dim = value.size(-1);
    TORCH_CHECK(rope_dim > 0 && rope_dim <= head_dim && rope_dim % 2 == 0,
                "indexer RoPE dimension must be positive, even, and <= index_head_dim");
    int64_t nope_dim = head_dim - rope_dim;
    if (interleave) {
        auto nope = value.narrow(-1, 0, nope_dim);
        auto rope = value.narrow(-1, nope_dim, rope_dim);
        return at::cat({nope, apply_rotary_interleave(rope, cos, sin)}, -1);
    }
    auto rope = value.narrow(-1, 0, rope_dim);
    auto nope = value.narrow(-1, rope_dim, nope_dim);
    return at::cat({apply_rotary(rope, cos, sin), nope}, -1);
}

// ── Helper: FP8 dequant (byte-level FP8→F32, then × scale, →BF16) ──
// Mirrors Rust's dequant_fp8_weight: expand block-wise scale [n_blocks, k_blocks]
// to [N, K], multiply, convert to target dtype.
static at::Tensor dequant_fp8(const at::Tensor& fp8_weight, const at::Tensor& scale,
                                at::ScalarType target_dtype) {
    TORCH_CHECK(fp8_weight.dim() == 2, "FP8 weight must be rank 2");
    TORCH_CHECK(fp8_weight.scalar_type() == at::kFloat8_e4m3fn,
                "FP8 dequant requires an e4m3fn weight");
    TORCH_CHECK(scale.dim() == 2, "FP8 scale must be rank 2");
    TORCH_CHECK(fp8_weight.device() == scale.device(),
                "FP8 weight and scale must be on the same device before dequant");
    int64_t n = fp8_weight.size(0);
    int64_t k = fp8_weight.size(1);

    // Step 1: FP8 → F32. Avoid shape-dependent FP8 view conversions by
    // converting a contiguous tensor, then restoring the original shape.
    auto f32_weight = fp8_weight.contiguous().reshape({-1}).to(at::kFloat).reshape({n, k});

    // Step 2: expand scale from [n_blocks, k_blocks] to [N, K]
    auto scale_f32 = scale.to(at::kFloat);
    int64_t n_blocks = (n + 127) / 128;
    int64_t k_blocks = (k + 127) / 128;
    at::Tensor scale_expanded;
    if (scale_f32.size(0) == n_blocks && scale_f32.size(1) == k_blocks) {
        // Repeat each block scale over its contiguous 128x128 weight tile.
        auto expanded = at::repeat_interleave(scale_f32, 128, 0)
                            .repeat_interleave(128, 1);
        // Crop to actual [N, K]
        scale_expanded = expanded.narrow(0, 0, n).narrow(1, 0, k);
    } else if (scale_f32.sizes() == fp8_weight.sizes()) {
        scale_expanded = scale_f32;
    } else {
        TORCH_CHECK(false, "FP8 scale must be [ceil(N/128), ceil(K/128)] or [N, K]");
    }

    // Step 3: apply scale and convert to target dtype
    auto result = (f32_weight * scale_expanded).to(target_dtype);
    return result;
}

// ── Helper: FP8-safe linear ──
// If weight is FP8 and scale is provided: dequant (FP8→BF16 with scale) then at::linear.
// If weight is already BF16: ensure on same device as input, then at::linear.
static at::Tensor safe_linear(const at::Tensor& input, const at::Tensor& weight,
                               std::optional<at::Tensor> weight_scale) {
    auto dtype = input.scalar_type();
    auto device = input.device();
    if (weight_scale.has_value() && weight.scalar_type() == at::kFloat8_e4m3fn) {
        const auto& scale = weight_scale.value();
        TORCH_CHECK(weight.device() == device || weight.device().is_cpu(),
                    "FP8 weight must reside on CPU or the input device");
        TORCH_CHECK(scale.device() == device || scale.device().is_cpu(),
                    "FP8 scale must reside on CPU or the input device");

        // CPU checkpoint tensors must reach the compute device before FP8
        // conversion.  Tensor::to uses a blocking copy by default.  Already
        // colocated GPU tensors reuse their existing storage without a copy.
        auto weight_on_device = weight.device() == device ? weight : weight.to(device);
        auto scale_on_device = scale.device() == device ? scale : scale.to(device);
        auto w_bf16 = dequant_fp8(weight_on_device, scale_on_device, dtype);
        return at::linear(input, w_bf16);
    }
    TORCH_CHECK(weight.scalar_type() != at::kFloat8_e4m3fn,
                "FP8 weight requires an explicit block scale");
    // Standard path: ensure weight is on the same device as input
    auto w = weight.to(device).to(dtype);
    return at::linear(input, w);
}

// ── Helper: chunked topk (for large seq) ──
static at::Tensor finalize_causal_topk(const at::Tensor& scores, const at::Tensor& indices) {
    return at::where(scores.isfinite(), indices, at::full_like(indices, -1));
}

static at::Tensor indexer_scores(const at::Tensor& idx_q, const at::Tensor& idx_k,
                                 const at::Tensor& head_weights) {
    // [B,H,Q,D] @ [B,H,D,K] -> [B,H,Q,K], then weighted head reduction.
    auto per_head = at::relu(at::matmul(idx_q, idx_k.transpose(-2, -1)));
    auto weights = head_weights.transpose(1, 2).unsqueeze(-1);
    return (per_head * weights).sum(1, /*keepdim=*/false);
}

static at::Tensor chunked_topk(const at::Tensor& idx_q, const at::Tensor& idx_k,
                                const at::Tensor& head_weights, int64_t actual_topk,
                                int64_t batch, int64_t seq, int32_t device_id) {
    // For seq <= 4096, compute full scores in one matmul (fast, no chunking overhead)
    // The selected positions are shared by all MLA attention heads: [B,S,K].
    if (seq <= 4096) {
        auto scores = indexer_scores(idx_q, idx_k, head_weights);
        auto q_pos = at::arange(seq, at::TensorOptions().dtype(at::kLong).device(scores.device()));
        auto k_pos = at::arange(seq, at::TensorOptions().dtype(at::kLong).device(scores.device()));
        auto future = k_pos.unsqueeze(0).gt(q_pos.unsqueeze(1)).unsqueeze(0);
        scores = scores.masked_fill(future, -std::numeric_limits<double>::infinity());
        auto [selected_scores, indices] = scores.topk(actual_topk, -1, true, true);
        return finalize_causal_topk(selected_scores, indices);
    }

    // For seq > 4096, use chunked approach but with larger chunks (2048)
    // to reduce kernel launch overhead
    int64_t score_chunk = 2048;
    at::Tensor best_scores, best_indices;
    bool has_best = false;
    for (int64_t k_start = 0; k_start < seq; k_start += score_chunk) {
        int64_t k_end = std::min(k_start + score_chunk, seq);
        int64_t k_len = k_end - k_start;
        auto idx_k_chunk = idx_k.narrow(-2, k_start, k_len);
        auto scores_chunk = indexer_scores(idx_q, idx_k_chunk, head_weights);
        auto q_pos = at::arange(seq, at::TensorOptions().dtype(at::kLong).device(scores_chunk.device()));
        auto k_pos = at::arange(k_start, k_end, at::TensorOptions().dtype(at::kLong).device(scores_chunk.device()));
        auto future = k_pos.unsqueeze(0).gt(q_pos.unsqueeze(1)).unsqueeze(0);
        scores_chunk = scores_chunk.masked_fill(future, -std::numeric_limits<double>::infinity());
        int64_t local_topk = std::min(actual_topk, k_len);
        auto [ls, li] = scores_chunk.topk(local_topk, -1, true, true);
        auto offset = at::full(li.sizes(), (double)k_start,
                              at::TensorOptions().dtype(at::kFloat).device(at::Device(at::Device::Type::CUDA, device_id)));
        auto li_offset = li.to(at::kFloat) + offset;
        if (has_best) {
            auto merged = at::cat({best_scores, ls}, -1);
            auto merged_idx = at::cat({best_indices, li_offset.to(at::kLong)}, -1);
            auto [s, pos] = merged.topk(actual_topk, -1, true, true);
            best_scores = s;
            best_indices = merged_idx.gather(-1, pos, false);
        } else {
            best_scores = ls;
            best_indices = li_offset.to(at::kLong);
            has_best = true;
        }
    }
    return finalize_causal_topk(best_scores, best_indices);
}

static void replace_index_state(void** topk_indices_ptr, void** idx_bias_keys_ptr,
                                at::Tensor topk_indices, at::Tensor idx_bias_keys,
                                int32_t* source_layer, int32_t layer) {
    TORCH_CHECK(topk_indices_ptr && idx_bias_keys_ptr && source_layer,
                "IndexShare state pointers must not be null");
    auto new_topk = std::make_unique<at::Tensor>(std::move(topk_indices));
    auto new_bias = std::make_unique<at::Tensor>(std::move(idx_bias_keys));
    delete reinterpret_cast<at::Tensor*>(*topk_indices_ptr);
    delete reinterpret_cast<at::Tensor*>(*idx_bias_keys_ptr);
    *topk_indices_ptr = new_topk.release();
    *idx_bias_keys_ptr = new_bias.release();
    *source_layer = layer;
}

static void clear_index_state(void** topk_indices_ptr, void** idx_bias_keys_ptr,
                              int32_t* source_layer) {
    if (topk_indices_ptr) {
        delete reinterpret_cast<at::Tensor*>(*topk_indices_ptr);
        *topk_indices_ptr = nullptr;
    }
    if (idx_bias_keys_ptr) {
        delete reinterpret_cast<at::Tensor*>(*idx_bias_keys_ptr);
        *idx_bias_keys_ptr = nullptr;
    }
    if (source_layer) {
        *source_layer = -1;
    }
}

static void validate_index_state(void** topk_indices_ptr, void** idx_bias_keys_ptr,
                                 int32_t* source_layer) {
    TORCH_CHECK(topk_indices_ptr && idx_bias_keys_ptr && source_layer,
                "IndexShare state pointers must not be null");
    TORCH_CHECK((*topk_indices_ptr == nullptr) == (*idx_bias_keys_ptr == nullptr),
                "IndexShare top-k and bias state must either both be set or both be null");
}

static at::Tensor sparse_causal_attention(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& topk_indices, double attn_scale) {
    TORCH_CHECK(topk_indices.dim() == 3, "IndexShare top-k must be [batch, seq, topk]");
    int64_t batch = q.size(0), heads = q.size(1), seq = q.size(2);
    TORCH_CHECK(topk_indices.size(0) == batch && topk_indices.size(1) == seq,
                "IndexShare top-k shape does not match attention input");
    auto dtype = q.scalar_type();
    auto device = q.device();
    int64_t actual_topk = topk_indices.size(-1);
    int64_t query_chunk = seq > 2048 ? 512 : seq;
    std::vector<at::Tensor> outputs;

    for (int64_t q_start = 0; q_start < seq; q_start += query_chunk) {
        int64_t q_len = std::min(query_chunk, seq - q_start);
        auto q_chunk = q.narrow(2, q_start, q_len);
        auto selected = topk_indices.narrow(1, q_start, q_len)
                            .unsqueeze(1).expand({batch, heads, q_len, actual_topk}, false);
        auto valid = selected.ge(0);
        auto sparse_count = at::zeros({batch, heads, q_len, seq},
            at::TensorOptions().dtype(dtype).device(device));
        sparse_count.scatter_add_(-1, selected.clamp_min(0), valid.to(dtype));

        auto q_pos = at::arange(q_start, q_start + q_len,
            at::TensorOptions().dtype(at::kLong).device(device));
        auto k_pos = at::arange(seq, at::TensorOptions().dtype(at::kLong).device(device));
        auto causal = k_pos.unsqueeze(0).le(q_pos.unsqueeze(1))
                         .unsqueeze(0).unsqueeze(0)
                         .expand({batch, heads, q_len, seq}, false);
        auto allowed = sparse_count.gt(0).logical_and(causal);
        auto bias = at::zeros({batch, heads, q_len, seq},
            at::TensorOptions().dtype(dtype).device(device));
        bias.masked_fill_(allowed.logical_not(), -std::numeric_limits<double>::infinity());
        outputs.push_back(at::scaled_dot_product_attention(
            q_chunk, k, v, bias, 0.0, false, attn_scale));
    }
    return outputs.size() == 1 ? outputs.front() : at::cat(outputs, 2);
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_dsa_attention — full DSA attention in one C++ call
//
// Input: hidden [B, S, H] BF16 on GPU
// Output: attention output [B, S, num_heads * v_head] BF16 on GPU
//
// IndexShare state is passed in/out via raw pointers:
//   topk_indices_ptr: void** — points to at::Tensor* (or nullptr to recompute)
//   idx_bias_keys_ptr: void** — points to at::Tensor*
// ══════════════════════════════════════════════════════════════════════

void* v4_glm5_dsa_attention(
    // Input
    void* input_ptr,            // at::Tensor* [B, S, H]
    // Attention weights (all at::Tensor*)
    void* q_a_proj, void* q_a_layernorm, void* q_b_proj,
    void* kv_a_proj, void* kv_a_layernorm, void* kv_b_proj,
    void* o_proj,
    // FP8 scales (nullable: pass nullptr if not FP8)
    void* q_a_scale, void* q_b_scale, void* kv_a_scale, void* kv_b_scale, void* o_scale,
    // Indexer weights (nullable for non-full layers)
    void* idx_wq_b, void* idx_wk, void* idx_k_norm_w, void* idx_k_norm_b,
    void* idx_weights_proj, void* idx_weights_proj_scale,
    void* idx_wq_b_scale, void* idx_wk_scale,
    // Config
    int32_t batch_i, int32_t seq_i, int32_t num_heads_i, int32_t qk_nope_i, int32_t qk_rope_i,
    int32_t v_head_i, int32_t kv_lora_i, int32_t idx_head_dim_i, int32_t idx_n_heads_i,
    int32_t idx_topk_i, int32_t index_topk_freq_i, int32_t layer_i, int32_t is_full_layer_i,
    double rms_eps, double rope_theta, int32_t rope_interleave_i,
    int32_t device_id,
    // IndexShare state (in/out)
    void** topk_indices_ptr,    // &at::Tensor* or &nullptr
    void** idx_bias_keys_ptr,   // &at::Tensor* or &nullptr
    int32_t* source_layer
) {
    try {
        g_glm5_last_error.clear();
        validate_index_state(topk_indices_ptr, idx_bias_keys_ptr, source_layer);
        TORCH_CHECK(input_ptr && q_a_proj && q_a_layernorm && q_b_proj && kv_a_proj &&
                    kv_a_layernorm && kv_b_proj && o_proj, "required attention tensor is null");
        TORCH_CHECK(batch_i > 0 && seq_i > 0 && num_heads_i > 0, "invalid attention shape");
        TORCH_CHECK(idx_topk_i > 0 && index_topk_freq_i > 0,
                    "idx_topk and index_topk_freq must be positive");
        const bool is_full_layer = is_full_layer_i != 0;
        const bool rope_interleave = rope_interleave_i != 0;
        auto& input = *reinterpret_cast<at::Tensor*>(input_ptr);
        TORCH_CHECK(input.dim() == 3 && input.size(0) == batch_i && input.size(1) == seq_i,
                    "input shape does not match batch/seq ABI values");
        auto compute_dtype = input.scalar_type();
        int64_t batch = batch_i, seq = seq_i;
        int64_t nh = num_heads_i, qn = qk_nope_i, qr = qk_rope_i, vh = v_head_i;
        int64_t kvl = kv_lora_i, ihd = idx_head_dim_i, inh = idx_n_heads_i;
        int64_t itk = idx_topk_i;
        auto device = at::Device(at::Device::Type::CUDA, device_id);

        // ── Q/K/V projections ──
        auto& q_a_w = *reinterpret_cast<at::Tensor*>(q_a_proj);
        auto& q_a_ln = *reinterpret_cast<at::Tensor*>(q_a_layernorm);
        auto& q_b_w = *reinterpret_cast<at::Tensor*>(q_b_proj);
        auto& kv_a_w = *reinterpret_cast<at::Tensor*>(kv_a_proj);
        auto& kv_a_ln = *reinterpret_cast<at::Tensor*>(kv_a_layernorm);
        auto& kv_b_w = *reinterpret_cast<at::Tensor*>(kv_b_proj);
        auto& o_w = *reinterpret_cast<at::Tensor*>(o_proj);

        // FP8 scales
        auto qa_s = q_a_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(q_a_scale)) : std::nullopt;
        auto qb_s = q_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(q_b_scale)) : std::nullopt;
        auto kva_s = kv_a_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(kv_a_scale)) : std::nullopt;
        auto kvb_s = kv_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(kv_b_scale)) : std::nullopt;
        auto o_s = o_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(o_scale)) : std::nullopt;
        auto iwq_s = idx_wq_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(idx_wq_b_scale)) : std::nullopt;
        auto iwk_s = idx_wk_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(idx_wk_scale)) : std::nullopt;
        auto iwp_s = idx_weights_proj_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(idx_weights_proj_scale)) : std::nullopt;

        auto q_a = safe_linear(input, q_a_w, qa_s);
        auto q_a_normed = rms_norm(q_a, q_a_ln.to(compute_dtype), rms_eps);
        auto q_b = safe_linear(q_a_normed, q_b_w, qb_s);
        auto q = q_b.reshape({batch, seq, nh, qn + qr}).transpose(1, 2);
        auto q_nope = q.narrow(-1, 0, qn);
        auto q_rope = q.narrow(-1, qn, qr);

        auto kv_a = safe_linear(input, kv_a_w, kva_s);
        auto kv_lora_raw = kv_a.narrow(-1, 0, kvl);
        auto k_rope = kv_a.narrow(-1, kvl, qr);
        auto kv_lora_part = rms_norm(kv_lora_raw, kv_a_ln.to(compute_dtype), rms_eps);
        auto kv_b = safe_linear(kv_lora_part, kv_b_w, kvb_s);
        kv_b = kv_b.reshape({batch, seq, nh, qn + vh});
        auto k_nope = kv_b.narrow(-1, 0, qn).transpose(1, 2);
        auto v = kv_b.narrow(-1, qn, vh).transpose(1, 2);

        // ── RoPE ──
        auto k_rope_expanded = k_rope.unsqueeze(2).transpose(1, 2)
                                   .expand({batch, nh, seq, qr}, /*implicit=*/false);
        auto [cos, sin] = rope_cos_sin(seq, qr, rope_theta, device_id);
        cos = cos.to(compute_dtype);
        sin = sin.to(compute_dtype);
        at::Tensor q_rope_rot, k_rope_rot;
        if (rope_interleave) {
            q_rope_rot = apply_rotary_interleave(q_rope, cos, sin);
            k_rope_rot = apply_rotary_interleave(k_rope_expanded, cos, sin);
        } else {
            q_rope_rot = apply_rotary(q_rope, cos, sin);
            k_rope_rot = apply_rotary(k_rope_expanded, cos, sin);
        }

        auto q_full = at::cat({q_nope, q_rope_rot}, -1);
        auto k_full = at::cat({k_nope, k_rope_rot}, -1);
        double attn_scale = 1.0 / std::sqrt((double)(qn + qr));

        // ── DSA Indexer ──
        bool state_shape_mismatch = *topk_indices_ptr &&
            (reinterpret_cast<at::Tensor*>(*topk_indices_ptr)->dim() != 3 ||
             reinterpret_cast<at::Tensor*>(*topk_indices_ptr)->size(0) != batch ||
             reinterpret_cast<at::Tensor*>(*topk_indices_ptr)->size(1) != seq);
        bool should_compute_topk = is_full_layer &&
            (*topk_indices_ptr == nullptr || state_shape_mismatch ||
             layer_i % index_topk_freq_i == 0);

        if (idx_wq_b && idx_wk && idx_k_norm_w && idx_k_norm_b && idx_weights_proj) {
            if (should_compute_topk) {
                auto& wq_b = *reinterpret_cast<at::Tensor*>(idx_wq_b);
                auto& wk = *reinterpret_cast<at::Tensor*>(idx_wk);
                auto& kn_w = *reinterpret_cast<at::Tensor*>(idx_k_norm_w);
                auto& kn_b = *reinterpret_cast<at::Tensor*>(idx_k_norm_b);
                auto& wproj = *reinterpret_cast<at::Tensor*>(idx_weights_proj);

                // Indexer Q — with FP8 scale
                auto idx_q = safe_linear(q_a, wq_b, iwq_s);
                idx_q = idx_q.reshape({batch, seq, inh, ihd}).transpose(1, 2);

                // Indexer K — with FP8 scale
                auto idx_k_raw = safe_linear(input, wk, iwk_s);
                auto idx_k = rms_norm_with_bias(idx_k_raw, kn_w.to(compute_dtype),
                                                kn_b.to(compute_dtype), rms_eps);
                auto idx_k_expanded = idx_k.unsqueeze(1).expand({batch, inh, seq, ihd}, /*implicit=*/false);

                // Indexer RoPE
                auto [ci, si] = rope_cos_sin(seq, qr, rope_theta, device_id);
                ci = ci.to(compute_dtype);
                si = si.to(compute_dtype);
                auto idx_q_rot = apply_indexer_rope(idx_q, ci, si, qr, rope_interleave);
                auto idx_k_rot = apply_indexer_rope(
                    idx_k_expanded, ci, si, qr, rope_interleave);

                int64_t actual_topk = std::min(itk, seq);
                auto head_weights = safe_linear(input, wproj, iwp_s)
                    .reshape({batch, seq, inh}) *
                    (1.0 / std::sqrt((double)(inh * ihd)));
                auto topk_indices = chunked_topk(
                    idx_q_rot, idx_k_rot, head_weights, actual_topk,
                    batch, seq, device_id);

                replace_index_state(topk_indices_ptr, idx_bias_keys_ptr,
                                    std::move(topk_indices), std::move(head_weights),
                                    source_layer, layer_i);
            }
        } else {
            // A shared/indexer layer may be absent for this call. Never reuse
            // a top-k map from a previous source layer in that case.
            clear_index_state(topk_indices_ptr, idx_bias_keys_ptr, source_layer);
        }

        // ── Attention via SDPA ──
        at::Tensor context;
        if (*topk_indices_ptr) {
            auto& topk_indices = *reinterpret_cast<at::Tensor*>(*topk_indices_ptr);
            context = sparse_causal_attention(
                q_full, k_full, v, topk_indices, attn_scale);
        } else {
            // Full causal SDPA
            context = at::scaled_dot_product_attention(
                q_full, k_full, v, std::nullopt, 0.0, true, attn_scale);
        }

        // ── Output projection ──
        auto out = context.transpose(1, 2).reshape({batch, seq, nh * vh});
        auto result = safe_linear(out, o_w, o_s);
        return new at::Tensor(std::move(result));
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_dsa_attention", e);
        return nullptr;
    }
}

// ── Helper: free at::Tensor* created by v4_glm5_dsa_attention ──
void v4_glm5_free_at_tensor(void* tensor_ptr) {
    if (tensor_ptr) {
        delete reinterpret_cast<at::Tensor*>(tensor_ptr);
    }
}

// Autograd-preserving NCCL ring exchange.  Unlike the low-level Rust helper,
// this op records a reverse ring for backward and therefore propagates K/V and
// indexer activation gradients to their owning CP rank.
void* v4_glm5_nccl_ring_autograd(
    void* input_ptr, void* comm_ptr,
    int64_t send_peer, int64_t recv_peer
) {
    try {
        TORCH_CHECK(input_ptr && comm_ptr, "autograd ring received null pointer");
        auto& input = *reinterpret_cast<at::Tensor*>(input_ptr);
        auto output = Glm5NcclRingFunction::apply(
            input, reinterpret_cast<int64_t>(comm_ptr),
            send_peer, recv_peer);
        return new at::Tensor(std::move(output));
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_nccl_ring_autograd", e);
        return nullptr;
    }
}

void v4_glm5_nccl_kv_ring_autograd(
    void* key_ptr,
    void* value_ptr,
    void* comm_ptr,
    int64_t send_peer,
    int64_t recv_peer,
    void** key_out_ptr,
    void** value_out_ptr
) {
    try {
        if (key_out_ptr) {
            *key_out_ptr = nullptr;
        }
        if (value_out_ptr) {
            *value_out_ptr = nullptr;
        }
        TORCH_CHECK(key_ptr && value_ptr && comm_ptr && key_out_ptr && value_out_ptr,
                    "autograd KV ring received a null pointer");
        auto& key = *reinterpret_cast<at::Tensor*>(key_ptr);
        auto& value = *reinterpret_cast<at::Tensor*>(value_ptr);
        auto outputs = Glm5NcclKvRingFunction::apply(
            key, value, reinterpret_cast<int64_t>(comm_ptr), send_peer, recv_peer);
        auto key_out = std::make_unique<at::Tensor>(std::move(outputs[0]));
        auto value_out = std::make_unique<at::Tensor>(std::move(outputs[1]));
        *key_out_ptr = key_out.release();
        *value_out_ptr = value_out.release();
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_nccl_kv_ring_autograd", e);
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_mlp_fp8 — SwiGLU MLP with optional FP8 weights
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_mlp_fp8(
    void* input_ptr,
    void* gate_ptr, void* up_ptr, void* down_ptr,
    void* gate_scale_ptr, void* up_scale_ptr, void* down_scale_ptr
) {
    try {
        g_glm5_last_error.clear();
        TORCH_CHECK(input_ptr && gate_ptr && up_ptr && down_ptr, "MLP tensor is null");
        auto& input = *reinterpret_cast<at::Tensor*>(input_ptr);
        auto& gate = *reinterpret_cast<at::Tensor*>(gate_ptr);
        auto& up = *reinterpret_cast<at::Tensor*>(up_ptr);
        auto& down = *reinterpret_cast<at::Tensor*>(down_ptr);
        auto dtype = input.scalar_type();

        auto gate_scale = gate_scale_ptr ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(gate_scale_ptr)) : std::nullopt;
        auto up_scale = up_scale_ptr ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(up_scale_ptr)) : std::nullopt;
        auto down_scale = down_scale_ptr ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(down_scale_ptr)) : std::nullopt;

        auto gate_out = safe_linear(input, gate, gate_scale);
        auto up_out = safe_linear(input, up, up_scale);
        auto activated = at::silu(gate_out) * up_out;
        auto result = safe_linear(activated, down, down_scale);
        return new at::Tensor(std::move(result));
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_mlp_fp8", e);
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_rms_norm — RMSNorm
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_rms_norm(void* input_ptr, void* weight_ptr, double eps) {
    try {
        g_glm5_last_error.clear();
        TORCH_CHECK(input_ptr && weight_ptr, "RMSNorm tensor is null");
        auto& input = *reinterpret_cast<at::Tensor*>(input_ptr);
        auto& weight = *reinterpret_cast<at::Tensor*>(weight_ptr);
        return new at::Tensor(rms_norm(input, weight, eps));
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_rms_norm", e);
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_cross_entropy_loss — chunked cross-entropy loss
// Returns scalar loss tensor (F32)
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_cross_entropy_loss(
    void* hidden_ptr,       // [B, S, H] BF16
    void* lm_head_ptr,      // [vocab, H] BF16
    void* targets_ptr,      // [B, S] int64
    void* mask_ptr,          // [B, S] float
    int32_t seq_len_i, int32_t vocab_i, int32_t chunk_size_i, int32_t device_id
) {
    try {
        g_glm5_last_error.clear();
        TORCH_CHECK(hidden_ptr && lm_head_ptr && targets_ptr && mask_ptr,
                    "cross entropy tensor is null");
        TORCH_CHECK(seq_len_i > 1 && vocab_i > 0 && chunk_size_i > 0,
                    "invalid cross entropy shape/config");
        auto& hidden = *reinterpret_cast<at::Tensor*>(hidden_ptr);
        auto& lm_head = *reinterpret_cast<at::Tensor*>(lm_head_ptr);
        auto& targets = *reinterpret_cast<at::Tensor*>(targets_ptr);
        auto& mask = *reinterpret_cast<at::Tensor*>(mask_ptr);
        // Keep the accumulator on the hidden tensor's device. The device_id
        // argument remains part of the CUDA ABI, but CPU parity fixtures must
        // not manufacture a CUDA scalar tensor.
        auto device = hidden.device();
        int64_t seq_len = seq_len_i, vocab = vocab_i, chunk = chunk_size_i;

        // Shifted: targets[1:], mask[1:]
        auto shifted_targets = targets.narrow(1, 1, seq_len - 1);
        auto shifted_mask = mask.narrow(1, 1, seq_len - 1).to(at::kFloat);
        auto total_mask = shifted_mask.sum(at::kFloat);

        auto loss_acc = at::zeros({}, at::TensorOptions().dtype(at::kFloat).device(device));
        for (int64_t start = 0; start < seq_len - 1; start += chunk) {
            int64_t end = std::min(start + chunk, seq_len - 1);
            int64_t len = end - start;
            auto normed_chunk = hidden.narrow(1, start, len);
            auto logits = at::linear(normed_chunk, lm_head);
            auto log_probs = logits.reshape({-1, vocab}).log_softmax(-1, at::kFloat);
            auto t = shifted_targets.narrow(1, start, len).reshape({-1});
            auto m = shifted_mask.narrow(1, start, len);
            auto per_token = at::nll_loss(log_probs, t, std::nullopt, at::Reduction::None, -100)
                                 .reshape({hidden.size(0), len});
            loss_acc = loss_acc + (per_token * m).sum(at::kFloat);
        }
        return new at::Tensor(loss_acc / total_mask.clamp_min(1.0));
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_cross_entropy_loss", e);
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_mtp_prepare — prepare one MTP teacher-forcing block
//
// For each position t, construct [enorm(embed(ids[t+offset])), hnorm(hidden[t])]
// and project it with eh_proj.  The returned sequence is clipped to the
// available hidden/input span, and position t predicts
// input_ids[t+offset+1].  The operation is composed
// entirely of ATen operations, preserving the autograd graph for trainable
// MTP parameters.
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_mtp_prepare(
    void* hidden_ptr,       // [B, S, H]
    void* input_ids_ptr,    // [B, S], int64
    void* embed_ptr,        // [vocab, H]
    void* enorm_ptr,        // [H]
    void* hnorm_ptr,        // [H]
    void* eh_proj_ptr,      // [H, 2H]
    void* eh_proj_scale_ptr,// optional FP8 block scale
    double eps,
    int32_t token_offset_i,
    int64_t vocab_start_i,
    int64_t global_vocab_size_i,
    void* tp_comm_ptr,
    int32_t tp_rank_i,
    int32_t tp_size_i
) {
    try {
        g_glm5_last_error.clear();
        TORCH_CHECK(hidden_ptr && input_ids_ptr && embed_ptr && enorm_ptr && hnorm_ptr && eh_proj_ptr,
                    "MTP prepare tensor is null");
        TORCH_CHECK(token_offset_i >= 1, "MTP prepare token offset must be positive");
        TORCH_CHECK(tp_size_i > 0 && tp_rank_i >= 0 && tp_rank_i < tp_size_i,
                    "invalid MTP prepare TP rank or size");
        TORCH_CHECK(vocab_start_i >= 0 && global_vocab_size_i > 0,
                    "invalid MTP prepare vocabulary range");
        TORCH_CHECK(tp_size_i == 1 || tp_comm_ptr,
                    "multi-rank MTP prepare requires a TP communicator");
        auto& hidden = *reinterpret_cast<at::Tensor*>(hidden_ptr);
        auto& input_ids = *reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto& embed = *reinterpret_cast<at::Tensor*>(embed_ptr);
        auto& enorm = *reinterpret_cast<at::Tensor*>(enorm_ptr);
        auto& hnorm = *reinterpret_cast<at::Tensor*>(hnorm_ptr);
        auto& eh_proj = *reinterpret_cast<at::Tensor*>(eh_proj_ptr);
        TORCH_CHECK(hidden.dim() == 3, "MTP hidden must have shape [B,S,H]");
        TORCH_CHECK(input_ids.dim() == 2, "MTP input_ids must have shape [B,S]");
        TORCH_CHECK(input_ids.scalar_type() == at::kLong,
                    "MTP input_ids must be int64 (torch.long)");
        TORCH_CHECK(hidden.size(0) == input_ids.size(0),
                    "MTP hidden and input_ids batch dimensions must match");
        TORCH_CHECK(input_ids.size(1) >= static_cast<int64_t>(token_offset_i) + 2,
                    "MTP prepare has no prediction positions after token offset");
        TORCH_CHECK(embed.dim() == 2 && embed.size(0) > 0 &&
                    embed.size(1) == hidden.size(2) && vocab_start_i < global_vocab_size_i,
                    "MTP embedding shard has an invalid shape or vocabulary start");
        TORCH_CHECK(enorm.numel() == hidden.size(2) && hnorm.numel() == hidden.size(2),
                    "MTP norm weights must have H elements");
        TORCH_CHECK(hidden.size(2) % tp_size_i == 0,
                    "MTP hidden size must be divisible by TP size");
        TORCH_CHECK(eh_proj.dim() == 2 && eh_proj.size(1) == 2 * hidden.size(2) &&
                    eh_proj.size(0) == hidden.size(2) / tp_size_i,
                    "MTP eh_proj must have shape [H/TP,2H]");
        TORCH_CHECK(embed.scalar_type() != at::kFloat8_e4m3fn,
                    "MTP embedding FP8 is unsupported; provide a dequantized embedding");
        TORCH_CHECK(hidden.device() == input_ids.device() && hidden.device() == embed.device(),
                    "MTP prepare tensors must be on the same device");

        const int64_t available = input_ids.size(1) - static_cast<int64_t>(token_offset_i) - 1;
        const int64_t out_len = std::min<int64_t>(hidden.size(1), available);
        TORCH_CHECK(out_len > 0, "MTP prepare has no aligned hidden positions");
        auto h = rms_norm(hidden.narrow(1, 0, out_len), hnorm, eps);
        auto next_ids = input_ids.narrow(1, token_offset_i, out_len);
        at::Tensor e;
        if (tp_size_i == 1) {
            e = at::embedding(embed, next_ids, -1, false, false);
        } else {
            auto local_end = vocab_start_i + embed.size(0);
            auto outside = next_ids.lt(vocab_start_i).logical_or(next_ids.ge(local_end));
            auto local_ids = (next_ids - vocab_start_i).masked_fill(outside, 0);
            auto local_embedding = at::embedding(embed, local_ids, -1, false, false)
                .masked_fill(outside.unsqueeze(-1), 0);
            e = Glm5NcclAllReduceIdentityBackward::apply(
                local_embedding, reinterpret_cast<int64_t>(tp_comm_ptr));
        }
        e = rms_norm(e, enorm, eps);
        auto combined = at::cat({e, h}, -1);
        std::optional<at::Tensor> eh_scale = eh_proj_scale_ptr
            ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(eh_proj_scale_ptr))
            : std::nullopt;
        auto linear_input = tp_size_i == 1
            ? combined
            : Glm5NcclIdentityAllReduceBackward::apply(
                  combined, reinterpret_cast<int64_t>(tp_comm_ptr));
        auto local_projected = safe_linear(linear_input, eh_proj, eh_scale);
        auto projected = tp_size_i == 1
            ? local_projected
            : Glm5NcclAllGatherSplitBackward::apply(
                  local_projected, reinterpret_cast<int64_t>(tp_comm_ptr),
                  tp_rank_i, tp_size_i);
        return new at::Tensor(std::move(projected));
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_mtp_prepare", e);
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_mtp_cross_entropy_loss — chunked next-next-token CE
//
// block_raw[j] predicts input_ids[start_offset+j+2].  target_mask is indexed
// in the same full-sequence coordinate system.  No loss weighting is applied
// here; callers combine this scalar with the configured MTP weight.
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_mtp_cross_entropy_loss(
    void* block_raw_ptr,       // [B, L, H]
    void* shared_head_norm_ptr,// [H]
    void* lm_head_ptr,         // [vocab, H]
    void* lm_head_scale_ptr,   // optional FP8 block scale
    void* input_ids_ptr,       // [B, S], int64
    void* target_mask_ptr,     // [B, S], numeric mask
    double eps,
    int32_t start_offset_i,
    int32_t chunk_size_i,
    int64_t vocab_start_i,
    int64_t global_vocab_size_i,
    void* tp_comm_ptr,
    int32_t tp_size_i,
    void** normalized_out_ptr,
    void** loss_sum_out_ptr,
    void** token_count_out_ptr
) {
    try {
        g_glm5_last_error.clear();
        if (normalized_out_ptr) {
            *normalized_out_ptr = nullptr;
        }
        if (loss_sum_out_ptr) {
            *loss_sum_out_ptr = nullptr;
        }
        if (token_count_out_ptr) {
            *token_count_out_ptr = nullptr;
        }
        TORCH_CHECK(block_raw_ptr && shared_head_norm_ptr && lm_head_ptr && input_ids_ptr &&
                    target_mask_ptr, "MTP CE tensor is null");
        TORCH_CHECK(start_offset_i >= 0 && chunk_size_i > 0,
                    "MTP CE start_offset must be non-negative and chunk_size positive");
        TORCH_CHECK(tp_size_i > 0, "MTP CE TP size must be positive");
        auto& block = *reinterpret_cast<at::Tensor*>(block_raw_ptr);
        auto& norm = *reinterpret_cast<at::Tensor*>(shared_head_norm_ptr);
        auto& lm_head = *reinterpret_cast<at::Tensor*>(lm_head_ptr);
        auto& input_ids = *reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto& mask = *reinterpret_cast<at::Tensor*>(target_mask_ptr);
        TORCH_CHECK(block.dim() == 3, "MTP block_raw must have shape [B,L,H]");
        TORCH_CHECK(input_ids.dim() == 2 && mask.dim() == 2,
                    "MTP input_ids and target_mask must have shape [B,S]");
        TORCH_CHECK(input_ids.scalar_type() == at::kLong,
                    "MTP input_ids must be int64 (torch.long)");
        TORCH_CHECK(input_ids.sizes() == mask.sizes(),
                    "MTP input_ids and target_mask shapes must match");
        TORCH_CHECK(block.size(0) == input_ids.size(0),
                    "MTP block_raw and input_ids batch dimensions must match");
        TORCH_CHECK(norm.numel() == block.size(2),
                    "MTP shared_head_norm must have H elements");
        TORCH_CHECK(lm_head.dim() == 2 && lm_head.size(1) == block.size(2),
                    "MTP lm_head must have shape [vocab,H]");
        TORCH_CHECK(vocab_start_i >= 0 &&
                    vocab_start_i + lm_head.size(0) <= global_vocab_size_i,
                    "MTP CE local vocabulary range exceeds global vocabulary");
        TORCH_CHECK(tp_size_i == 1 && vocab_start_i == 0 &&
                    global_vocab_size_i == lm_head.size(0)
                        ? tp_comm_ptr == nullptr
                        : (tp_size_i > 1 && tp_comm_ptr != nullptr),
                    "MTP CE TP communicator/range contract is invalid");
        TORCH_CHECK(block.device() == input_ids.device() && block.device() == mask.device(),
                    "MTP CE tensors must be on the same device");
        TORCH_CHECK(start_offset_i < input_ids.size(1), "MTP CE start_offset exceeds sequence length");

        const int64_t available = input_ids.size(1) - static_cast<int64_t>(start_offset_i) - 2;
        const int64_t usable = std::min<int64_t>(block.size(1), std::max<int64_t>(available, 0));
        TORCH_CHECK(usable > 0, "MTP CE has no target positions after start_offset");
        auto normalized = rms_norm(block.narrow(1, 0, usable), norm, eps);
        std::optional<at::Tensor> lm_scale = lm_head_scale_ptr
            ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(lm_head_scale_ptr))
            : std::nullopt;
        auto total_mask = mask.narrow(1, start_offset_i + 2, usable).to(at::kFloat);
        auto mask_sum = total_mask.sum(at::kFloat);
        auto finish = [&](at::Tensor loss_sum) -> void* {
            auto normalized_out = normalized_out_ptr
                ? std::make_unique<at::Tensor>(normalized)
                : nullptr;
            auto loss_sum_out = loss_sum_out_ptr
                ? std::make_unique<at::Tensor>(loss_sum)
                : nullptr;
            auto token_count_out = token_count_out_ptr
                ? std::make_unique<at::Tensor>(mask_sum)
                : nullptr;
            auto loss = loss_sum / mask_sum.clamp_min(1.0);
            auto result = std::make_unique<at::Tensor>(std::move(loss));
            if (normalized_out_ptr) {
                *normalized_out_ptr = normalized_out.release();
            }
            if (loss_sum_out_ptr) {
                *loss_sum_out_ptr = loss_sum_out.release();
            }
            if (token_count_out_ptr) {
                *token_count_out_ptr = token_count_out.release();
            }
            return result.release();
        };

        const bool vocab_parallel = tp_size_i > 1;
        auto ce_input = [&](const at::Tensor& input) {
            if (!vocab_parallel) {
                return input;
            }
            return Glm5NcclIdentityAllReduceBackward::apply(
                input, reinterpret_cast<int64_t>(tp_comm_ptr));
        };

        // MTP sequences normally fit in one CE chunk (the caller uses a
        // 4K chunk for 512-2 token blocks).  Avoid the scalar accumulator and
        // repeated narrow/dispatch bookkeeping in that case.  All operations
        // remain ATen operations, so gradients still flow through block,
        // shared_head_norm, and lm_head exactly as in the chunked path.
        if (usable <= chunk_size_i) {
            auto logits = safe_linear(ce_input(normalized), lm_head, lm_scale);
            auto targets = input_ids.narrow(1, start_offset_i + 2, usable);
            auto per_token = vocab_parallel
                ? Glm5VocabParallelCrossEntropy::apply(
                      logits, targets, vocab_start_i, global_vocab_size_i,
                      reinterpret_cast<int64_t>(tp_comm_ptr), tp_size_i)
                : at::nll_loss(
                      logits.reshape({-1, lm_head.size(0)}).log_softmax(-1, at::kFloat),
                      targets.reshape({-1}), std::nullopt, at::Reduction::None, -100);
            per_token = per_token.reshape({block.size(0), usable});
            return finish((per_token * total_mask).sum(at::kFloat));
        }

        auto loss_acc = at::zeros({}, at::TensorOptions().dtype(at::kFloat).device(block.device()));
        for (int64_t start = 0; start < usable; start += chunk_size_i) {
            const int64_t len = std::min<int64_t>(chunk_size_i, usable - start);
            auto logits = safe_linear(ce_input(normalized.narrow(1, start, len)), lm_head, lm_scale);
            auto targets = input_ids.narrow(1, start_offset_i + 2 + start, len);
            auto chunk_mask = total_mask.narrow(1, start, len);
            auto per_token = vocab_parallel
                ? Glm5VocabParallelCrossEntropy::apply(
                      logits, targets, vocab_start_i, global_vocab_size_i,
                      reinterpret_cast<int64_t>(tp_comm_ptr), tp_size_i)
                : at::nll_loss(
                      logits.reshape({-1, lm_head.size(0)}).log_softmax(-1, at::kFloat),
                      targets.reshape({-1}), std::nullopt, at::Reduction::None, -100);
            per_token = per_token.reshape({block.size(0), len});
            loss_acc = loss_acc + (per_token * chunk_mask).sum(at::kFloat);
        }
        return finish(std::move(loss_acc));
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_mtp_cross_entropy_loss", e);
        return nullptr;
    }
}

// Combine all independently normalized MTP layer losses in one C++ dispatch.
// The configured weight is the total auxiliary weight, not a per-layer weight.
void* v4_glm5_combine_losses(
    void* lm_loss_ptr,
    void** mtp_loss_ptrs,
    int32_t n_mtp_losses,
    double mtp_weight,
    void** mtp_mean_out_ptr
) {
    try {
        g_glm5_last_error.clear();
        if (mtp_mean_out_ptr) {
            *mtp_mean_out_ptr = nullptr;
        }
        TORCH_CHECK(lm_loss_ptr && mtp_loss_ptrs && n_mtp_losses > 0,
                    "MTP loss combine requires LM loss and at least one MTP loss");
        auto& lm_loss = *reinterpret_cast<at::Tensor*>(lm_loss_ptr);
        std::vector<at::Tensor> mtp_losses;
        mtp_losses.reserve(n_mtp_losses);
        for (int32_t i = 0; i < n_mtp_losses; ++i) {
            TORCH_CHECK(mtp_loss_ptrs[i], "MTP loss combine received a null layer loss");
            auto& layer_loss = *reinterpret_cast<at::Tensor*>(mtp_loss_ptrs[i]);
            TORCH_CHECK(layer_loss.numel() == 1 && layer_loss.device() == lm_loss.device(),
                        "MTP layer losses must be scalar tensors on the LM loss device");
            mtp_losses.push_back(layer_loss);
        }
        auto mtp_mean = at::stack(mtp_losses).mean();
        auto total = lm_loss + mtp_mean * mtp_weight;
        auto mean_out = mtp_mean_out_ptr
            ? std::make_unique<at::Tensor>(mtp_mean)
            : nullptr;
        auto result = std::make_unique<at::Tensor>(std::move(total));
        if (mtp_mean_out_ptr) {
            *mtp_mean_out_ptr = mean_out.release();
        }
        return result.release();
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_combine_losses", e);
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_adam_step — Adam optimizer in C++
// Updates params, m, v in-place
// ══════════════════════════════════════════════════════════════════════
void v4_adam_step(
    void** params,     // array of at::Tensor* (trainable params)
    void** grads,       // array of at::Tensor* (gradients)
    void** m_state,     // array of at::Tensor* (Adam m)
    void** v_state,     // array of at::Tensor* (Adam v)
    int n_params,
    double lr, double beta1, double beta2, double eps, int step_i
) {
    try {
        double sn = (double)(step_i + 1);
        double bias1 = 1.0 - std::pow(beta1, sn);
        double bias2 = 1.0 - std::pow(beta2, sn);

        for (int i = 0; i < n_params; i++) {
            auto& param = *reinterpret_cast<at::Tensor*>(params[i]);
            auto& grad = *reinterpret_cast<at::Tensor*>(grads[i]);
            auto& m = *reinterpret_cast<at::Tensor*>(m_state[i]);
            auto& v = *reinterpret_cast<at::Tensor*>(v_state[i]);

            if (!grad.defined() || grad.numel() == 0) continue;

            auto g = grad.to(at::kFloat);
            m = m * beta1 + g * (1.0 - beta1);
            v = v * beta2 + (g * g) * (1.0 - beta2);
            auto mh = m / bias1;
            auto vh = v / bias2;
            auto update = mh / (vh.sqrt() + eps);
            param.add_(update * (-lr));
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_adam_step] FAILED: %s\n", e.what());
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_moe_layer — MoE routing + expert dispatch + shared expert + combine
//
// Replaces ~20 tch-rs calls per layer with 1 FFI call.
// Expert weights are passed as CPU at::Tensor* arrays — C++ does to_device internally.
// ══════════════════════════════════════════════════════════════════════
static std::optional<at::Tensor> optional_tensor_at(void** tensors, int32_t index) {
    return tensors && tensors[index]
        ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(tensors[index]))
        : std::nullopt;
}

static at::Tensor moe_forward(
    const at::Tensor& mlp_input,
    void* shared_gate, void* shared_up, void* shared_down,
    void* shared_gate_scale, void* shared_up_scale, void* shared_down_scale,
    void* gate_weight, void* correction_bias,
    void** expert_gate_weights, void** expert_up_weights, void** expert_down_weights,
    void** expert_gate_scales, void** expert_up_scales, void** expert_down_scales,
    int32_t n_local_experts, const int32_t* local_expert_indices,
    int32_t n_routed_experts, int32_t topk,
    int32_t n_group, int32_t topk_group, int32_t scoring_func,
    int32_t topk_method, int32_t norm_topk_prob,
    double routed_scaling_factor,
    void* ep_comm, int32_t ep_rank, int32_t ep_size) {
    TORCH_CHECK(mlp_input.dim() == 3, "MoE input must be [batch, seq, hidden]");
    TORCH_CHECK(shared_gate && shared_up && shared_down && gate_weight,
                "required MoE tensor is null");
    TORCH_CHECK(n_local_experts >= 0 && n_routed_experts > 0 && ep_size > 0,
                "invalid expert count");
    TORCH_CHECK(topk > 0 && topk <= n_routed_experts, "invalid routed top-k");
    TORCH_CHECK(n_group > 0 && n_routed_experts % n_group == 0 &&
                topk_group > 0 && topk_group <= n_group,
                "invalid router group configuration");
    TORCH_CHECK(ep_rank >= 0 && ep_rank < ep_size && n_routed_experts % ep_size == 0,
                "invalid EP topology");
    TORCH_CHECK(ep_size == 1 || (ep_comm && mlp_input.is_cuda()),
                "multi-rank EP MoE requires a CUDA NCCL communicator");
    TORCH_CHECK(n_local_experts == 0 ||
                (local_expert_indices && expert_gate_weights && expert_up_weights && expert_down_weights),
                "local expert arrays are null");

    auto dtype = mlp_input.scalar_type();
    auto device = mlp_input.device();
    int64_t batch = mlp_input.size(0);
    int64_t seq = mlp_input.size(1);
    int64_t hidden = mlp_input.size(2);

    auto& sg = *reinterpret_cast<at::Tensor*>(shared_gate);
    auto& su = *reinterpret_cast<at::Tensor*>(shared_up);
    auto& sd = *reinterpret_cast<at::Tensor*>(shared_down);
    auto sg_scale = shared_gate_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(shared_gate_scale)) : std::nullopt;
    auto su_scale = shared_up_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(shared_up_scale)) : std::nullopt;
    auto sd_scale = shared_down_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(shared_down_scale)) : std::nullopt;
    auto shared_output = safe_linear(
        at::silu(safe_linear(mlp_input, sg, sg_scale)) * safe_linear(mlp_input, su, su_scale),
        sd, sd_scale);

    auto& gate_w = *reinterpret_cast<at::Tensor*>(gate_weight);
    auto logits = safe_linear(mlp_input, gate_w, std::nullopt).to(at::kFloat);
    at::Tensor scores;
    if (scoring_func == 0) scores = at::sigmoid(logits);
    else if (scoring_func == 1) scores = at::softmax(logits, -1, at::kFloat);
    else TORCH_CHECK(false, "unsupported GLM5 router scoring_func");

    at::Tensor selection_scores;
    if (topk_method == 1) {
        TORCH_CHECK(scoring_func == 0 && correction_bias,
                    "noaux_tc requires sigmoid scoring and correction bias");
        selection_scores = scores +
            reinterpret_cast<at::Tensor*>(correction_bias)->to(scores.device()).to(at::kFloat);
    } else if (topk_method == 0) {
        selection_scores = scores;
    } else {
        TORCH_CHECK(false, "unsupported GLM5 router topk_method");
    }

    const int64_t experts_per_group = n_routed_experts / n_group;
    auto grouped = selection_scores.reshape({-1, n_group, experts_per_group});
    at::Tensor group_scores;
    if (topk_method == 1) {
        group_scores = std::get<0>(grouped.topk(
            std::min<int64_t>(2, experts_per_group), -1, true, true)).sum(-1);
    } else {
        group_scores = std::get<0>(grouped.max(-1, false));
    }
    auto group_indices = std::get<1>(group_scores.topk(topk_group, -1, true, true));
    auto group_mask = at::zeros_like(group_scores).scatter(
        -1, group_indices, at::ones_like(group_indices, group_scores.options()));
    auto expert_mask = group_mask.unsqueeze(-1)
        .expand({-1, -1, experts_per_group}, false)
        .reshape({-1, n_routed_experts}).eq(0);
    auto masked = selection_scores.reshape({-1, n_routed_experts})
        .masked_fill(expert_mask, -std::numeric_limits<double>::infinity());
    auto topk_indices = std::get<1>(masked.topk(topk, -1, true, true));
    auto topk_weights = scores.reshape({-1, n_routed_experts})
        .gather(-1, topk_indices, false);
    if (norm_topk_prob) {
        topk_weights = topk_weights /
            topk_weights.sum(-1, true).clamp_min(1e-20);
    }
    topk_weights = topk_weights * routed_scaling_factor;

    auto flat_input = mlp_input.reshape({batch * seq, hidden});
    auto tk_indices = topk_indices.reshape({batch * seq, topk});
    auto tk_weights = topk_weights.reshape({batch * seq, topk});
    at::Tensor expert_input;
    at::Tensor expert_ids;
    at::Tensor send_counts;
    at::Tensor recv_counts;
    at::Tensor sort_order;
    if (ep_size > 1) {
        const int64_t assignments = tk_indices.numel();
        auto assignment_ids = at::arange(assignments, tk_indices.options());
        auto token_ids = at::floor_divide(assignment_ids, topk);
        auto flat_experts = tk_indices.reshape({-1});
        auto owners = at::floor_divide(flat_experts, n_routed_experts / ep_size);
        auto sort_key = owners * assignments + assignment_ids;
        sort_order = std::get<1>(sort_key.sort(0, false));
        auto sorted_input = flat_input.index_select(0, token_ids).index_select(0, sort_order);
        auto sorted_experts = flat_experts.index_select(0, sort_order);
        std::vector<at::Tensor> count_parts;
        count_parts.reserve(ep_size);
        for (int32_t peer = 0; peer < ep_size; ++peer) {
            count_parts.push_back(owners.eq(peer).sum(at::kLong));
        }
        send_counts = at::stack(count_parts).contiguous();
        auto dispatched = Glm5NcclEpDispatchFunction::apply(
            sorted_input, sorted_experts, send_counts,
            static_cast<int64_t>(reinterpret_cast<intptr_t>(ep_comm)), ep_rank, ep_size);
        expert_input = dispatched[0];
        expert_ids = dispatched[1];
        recv_counts = dispatched[2];
    } else {
        auto assignment_ids = at::arange(tk_indices.numel(), tk_indices.options());
        expert_input = flat_input.index_select(
            0, at::floor_divide(assignment_ids, topk));
        expert_ids = tk_indices.reshape({-1});
    }
    auto expert_output = at::zeros_like(expert_input);

    // Expert weights remain local to their owner. Router weights are deliberately
    // applied after the inverse return so they participate exactly once on the
    // originating rank, matching Megatron's token-dispatch contract.
    for (int32_t local = 0; local < n_local_experts; ++local) {
        TORCH_CHECK(expert_gate_weights[local] && expert_up_weights[local] &&
                    expert_down_weights[local], "local expert weight is null");
        auto positions = expert_ids.eq(local_expert_indices[local]).nonzero();
        if (positions.size(0) == 0) continue;
        auto token_ids = positions.select(1, 0);
        auto selected_input = expert_input.index_select(0, token_ids);

        auto& gate = *reinterpret_cast<at::Tensor*>(expert_gate_weights[local]);
        auto& up = *reinterpret_cast<at::Tensor*>(expert_up_weights[local]);
        auto& down = *reinterpret_cast<at::Tensor*>(expert_down_weights[local]);
        auto gate_out = safe_linear(selected_input, gate, optional_tensor_at(expert_gate_scales, local));
        auto up_out = safe_linear(selected_input, up, optional_tensor_at(expert_up_scales, local));
        auto expert_out = safe_linear(at::silu(gate_out) * up_out, down,
                                      optional_tensor_at(expert_down_scales, local));
        expert_output.index_add_(0, token_ids, expert_out);
    }

    at::Tensor returned;
    if (ep_size > 1) {
        auto returned_sorted = Glm5NcclEpReturnFunction::apply(
            expert_output, send_counts, recv_counts,
            static_cast<int64_t>(reinterpret_cast<intptr_t>(ep_comm)), ep_rank, ep_size);
        auto assignment_ids = at::arange(sort_order.numel(), sort_order.options());
        auto inverse_order = at::empty_like(sort_order);
        inverse_order.scatter_(0, sort_order, assignment_ids);
        returned = returned_sorted.index_select(0, inverse_order)
            .reshape({batch * seq, topk, hidden});
    } else {
        returned = expert_output.reshape({batch * seq, topk, hidden});
    }
    auto routed = (returned * tk_weights.to(dtype).unsqueeze(-1)).sum(1)
        .reshape({batch, seq, hidden});
    return routed + shared_output;
}

void* v4_glm5_moe_layer(
    void* mlp_input_ptr,
    void* shared_gate, void* shared_up, void* shared_down,
    void* shared_gate_scale, void* shared_up_scale, void* shared_down_scale,
    void* gate_weight, void* correction_bias,
    void** expert_gate_weights, void** expert_up_weights, void** expert_down_weights,
    void** expert_gate_scales, void** expert_up_scales, void** expert_down_scales,
    int32_t n_local_experts, const int32_t* local_expert_indices,
    int32_t n_routed_experts, int32_t topk,
    int32_t n_group, int32_t topk_group, int32_t scoring_func,
    int32_t topk_method, int32_t norm_topk_prob,
    double routed_scaling_factor,
    void* ep_comm, int32_t ep_rank, int32_t ep_size,
    int32_t device_id) {
    try {
        g_glm5_last_error.clear();
        TORCH_CHECK(mlp_input_ptr, "MoE input is null");
        auto& mlp_input = *reinterpret_cast<at::Tensor*>(mlp_input_ptr);
        TORCH_CHECK(!mlp_input.is_cuda() || mlp_input.device().index() == device_id,
                    "MoE input device does not match device_id");
        auto output = moe_forward(
            mlp_input, shared_gate, shared_up, shared_down,
            shared_gate_scale, shared_up_scale, shared_down_scale, gate_weight,
            correction_bias,
            expert_gate_weights, expert_up_weights, expert_down_weights,
            expert_gate_scales, expert_up_scales, expert_down_scales,
            n_local_experts, local_expert_indices, n_routed_experts, topk,
            n_group, topk_group, scoring_func, topk_method, norm_topk_prob,
            routed_scaling_factor, ep_comm, ep_rank, ep_size);
        return new at::Tensor(std::move(output));
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_moe_layer", e);
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_embedding — embedding lookup
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_embedding(void* embed_weight_ptr, void* input_ids_ptr, int32_t device_id) {
    try {
        g_glm5_last_error.clear();
        TORCH_CHECK(embed_weight_ptr && input_ids_ptr, "embedding input is null");
        auto& embed_weight = *reinterpret_cast<at::Tensor*>(embed_weight_ptr);
        auto& input_ids = *reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto result = at::embedding(embed_weight, input_ids, -1, false, false);
        return new at::Tensor(std::move(result));
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_embedding", e);
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_layer_forward — full transformer layer in one C++ call
//
// Combines: RMSNorm → attention → residual → RMSNorm → MoE/dense → residual
// All intermediate tensors stay on C++ stack — zero FFI crossings.
// Returns new hidden state [B, S, H].
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_layer_forward(
    void* hidden_ptr,              // [B, S, H] BF16 on GPU
    // Norm weights
    void* input_norm_weight,       // RMSNorm weight for attention input
    void* post_norm_weight,        // RMSNorm weight for MLP input
    // Attention weights (all at::Tensor* — see v4_glm5_dsa_attention)
    void* q_a_proj, void* q_a_layernorm, void* q_b_proj,
    void* kv_a_proj, void* kv_a_layernorm, void* kv_b_proj, void* o_proj,
    void* q_a_scale, void* q_b_scale, void* kv_a_scale, void* kv_b_scale, void* o_scale,
    void* idx_wq_b, void* idx_wk, void* idx_k_norm_w, void* idx_k_norm_b,
    void* idx_weights_proj, void* idx_weights_proj_scale,
    void* idx_wq_b_scale, void* idx_wk_scale,
    // MLP/MoE weights
    void* gate_weight,             // MoE router (or nullptr for dense)
    void* shared_gate, void* shared_up, void* shared_down,  // shared expert (or nullptr for dense)
    void* shared_gate_scale, void* shared_up_scale, void* shared_down_scale,
    // Dense MLP weights (if not MoE)
    void* dense_gate, void* dense_up, void* dense_down,
    void* dense_gate_scale, void* dense_up_scale, void* dense_down_scale,
    // Expert weights (CPU, for MoE only)
    void** expert_gate_weights, void** expert_up_weights, void** expert_down_weights,
    void** expert_gate_scales, void** expert_up_scales, void** expert_down_scales,
    int32_t n_local_experts,
    const int32_t* local_expert_indices,
    // Config
    int32_t batch_i, int32_t seq_i, int32_t num_heads_i, int32_t qk_nope_i, int32_t qk_rope_i,
    int32_t v_head_i, int32_t kv_lora_i, int32_t idx_head_dim_i, int32_t idx_n_heads_i,
    int32_t idx_topk_i, int32_t index_topk_freq_i, int32_t layer_i, int32_t is_full_layer_i,
    int32_t is_moe_layer_i, int32_t n_routed_experts, int32_t topk,
    double rms_eps, double rope_theta, int32_t rope_interleave_i,
    double routed_scaling_factor,
    int32_t device_id,
    // IndexShare state (in/out)
    void** topk_indices_ptr, void** idx_bias_keys_ptr, int32_t* source_layer
) {
    try {
        g_glm5_last_error.clear();
        validate_index_state(topk_indices_ptr, idx_bias_keys_ptr, source_layer);
        TORCH_CHECK(hidden_ptr && input_norm_weight && post_norm_weight && q_a_proj &&
                    q_a_layernorm && q_b_proj && kv_a_proj && kv_a_layernorm &&
                    kv_b_proj && o_proj, "required layer tensor is null");
        TORCH_CHECK(batch_i > 0 && seq_i > 0 && num_heads_i > 0, "invalid layer shape");
        TORCH_CHECK(idx_topk_i > 0 && index_topk_freq_i > 0,
                    "idx_topk and index_topk_freq must be positive");
        const bool is_full_layer = is_full_layer_i != 0;
        const bool is_moe_layer = is_moe_layer_i != 0;
        const bool rope_interleave = rope_interleave_i != 0;
        auto& hidden = *reinterpret_cast<at::Tensor*>(hidden_ptr);
        TORCH_CHECK(hidden.dim() == 3 && hidden.size(0) == batch_i && hidden.size(1) == seq_i,
                    "hidden shape does not match batch/seq ABI values");
        auto dtype = hidden.scalar_type();
        int64_t batch = batch_i, seq = seq_i;
        int64_t nh = num_heads_i;
        auto device = at::Device(at::Device::Type::CUDA, device_id);



        // ── 1. Attention RMSNorm ──
        auto& attn_norm_w = *reinterpret_cast<at::Tensor*>(input_norm_weight);
        auto hidden_norm = rms_norm(hidden, attn_norm_w.to(dtype), rms_eps);

        // ── 2. Attention ──
        // Build indexer weights check
        bool has_indexer = (idx_wq_b != nullptr && idx_wk != nullptr &&
                           idx_k_norm_w != nullptr && idx_k_norm_b != nullptr &&
                           idx_weights_proj != nullptr);

        // Delegate to v4_glm5_dsa_attention logic (inline for zero FFI overhead)
        // Q/K/V projections
        auto& q_a_w = *reinterpret_cast<at::Tensor*>(q_a_proj);
        auto& q_a_ln = *reinterpret_cast<at::Tensor*>(q_a_layernorm);
        auto& q_b_w = *reinterpret_cast<at::Tensor*>(q_b_proj);
        auto& kv_a_w = *reinterpret_cast<at::Tensor*>(kv_a_proj);
        auto& kv_a_ln = *reinterpret_cast<at::Tensor*>(kv_a_layernorm);
        auto& kv_b_w = *reinterpret_cast<at::Tensor*>(kv_b_proj);
        auto& o_w = *reinterpret_cast<at::Tensor*>(o_proj);
        int64_t qn = qk_nope_i, qr = qk_rope_i, vh = v_head_i, kvl = kv_lora_i;
        int64_t ihd = idx_head_dim_i, inh = idx_n_heads_i, itk = idx_topk_i;

        // Q/K/V projections — use safe_linear for FP8 scale support
        auto q_a_scale_t = q_a_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(q_a_scale)) : std::nullopt;
        auto q_b_scale_t = q_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(q_b_scale)) : std::nullopt;
        auto kv_a_scale_t = kv_a_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(kv_a_scale)) : std::nullopt;
        auto kv_b_scale_t = kv_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(kv_b_scale)) : std::nullopt;
        auto o_scale_t = o_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(o_scale)) : std::nullopt;

        auto q_a = safe_linear(hidden_norm, q_a_w, q_a_scale_t);
        auto q_a_normed = rms_norm(q_a, q_a_ln.to(dtype), rms_eps);
        auto q_b = safe_linear(q_a_normed, q_b_w, q_b_scale_t);
        auto q = q_b.reshape({batch, seq, nh, qn + qr}).transpose(1, 2);
        auto q_nope = q.narrow(-1, 0, qn);
        auto q_rope = q.narrow(-1, qn, qr);

        auto kv_a = safe_linear(hidden_norm, kv_a_w, kv_a_scale_t);
        auto kv_lora_raw = kv_a.narrow(-1, 0, kvl);
        auto k_rope = kv_a.narrow(-1, kvl, qr);
        auto kv_lora_part = rms_norm(kv_lora_raw, kv_a_ln.to(dtype), rms_eps);
        auto kv_b = safe_linear(kv_lora_part, kv_b_w, kv_b_scale_t);
        kv_b = kv_b.reshape({batch, seq, nh, qn + vh});
        auto k_nope = kv_b.narrow(-1, 0, qn).transpose(1, 2);
        auto v = kv_b.narrow(-1, qn, vh).transpose(1, 2);

        auto k_rope_expanded = k_rope.unsqueeze(2).transpose(1, 2)
                                   .expand({batch, nh, seq, qr}, /*implicit=*/false);
        auto [cos, sin] = rope_cos_sin(seq, qr, rope_theta, device_id);
        cos = cos.to(dtype); sin = sin.to(dtype);
        at::Tensor q_rope_rot, k_rope_rot;
        if (rope_interleave) {
            q_rope_rot = apply_rotary_interleave(q_rope, cos, sin);
            k_rope_rot = apply_rotary_interleave(k_rope_expanded, cos, sin);
        } else {
            q_rope_rot = apply_rotary(q_rope, cos, sin);
            k_rope_rot = apply_rotary(k_rope_expanded, cos, sin);
        }

        auto q_full = at::cat({q_nope, q_rope_rot}, -1);
        auto k_full = at::cat({k_nope, k_rope_rot}, -1);
        double attn_scale = 1.0 / std::sqrt((double)(qn + qr));

        // DSA indexer
        bool state_shape_mismatch = *topk_indices_ptr &&
            (reinterpret_cast<at::Tensor*>(*topk_indices_ptr)->dim() != 3 ||
             reinterpret_cast<at::Tensor*>(*topk_indices_ptr)->size(0) != batch ||
             reinterpret_cast<at::Tensor*>(*topk_indices_ptr)->size(1) != seq);
        bool should_compute = is_full_layer &&
            (*topk_indices_ptr == nullptr || state_shape_mismatch ||
             layer_i % index_topk_freq_i == 0);

        if (has_indexer && should_compute) {
            auto& wq_b = *reinterpret_cast<at::Tensor*>(idx_wq_b);
            auto& wk = *reinterpret_cast<at::Tensor*>(idx_wk);
            auto& kn_w = *reinterpret_cast<at::Tensor*>(idx_k_norm_w);
            auto& kn_b = *reinterpret_cast<at::Tensor*>(idx_k_norm_b);
            auto& wproj = *reinterpret_cast<at::Tensor*>(idx_weights_proj);
            auto idx_weights_proj_scale_t = idx_weights_proj_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(idx_weights_proj_scale)) : std::nullopt;
            auto idx_wq_b_scale_t = idx_wq_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(idx_wq_b_scale)) : std::nullopt;
            auto idx_wk_scale_t = idx_wk_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(idx_wk_scale)) : std::nullopt;
            auto idx_q = safe_linear(q_a, wq_b, idx_wq_b_scale_t);
            idx_q = idx_q.reshape({batch, seq, inh, ihd}).transpose(1, 2);
            auto idx_k_raw = safe_linear(hidden, wk, idx_wk_scale_t);
            auto idx_k = rms_norm_with_bias(idx_k_raw, kn_w.to(dtype), kn_b.to(dtype), rms_eps);
            auto idx_k_exp = idx_k.unsqueeze(1).expand({batch, inh, seq, ihd}, /*implicit=*/false);
            auto [ci, si] = rope_cos_sin(seq, qr, rope_theta, device_id);
            ci = ci.to(dtype);
            si = si.to(dtype);
            auto idx_q_rot = apply_indexer_rope(idx_q, ci, si, qr, rope_interleave);
            auto idx_k_rot = apply_indexer_rope(idx_k_exp, ci, si, qr, rope_interleave);
            int64_t actual_topk = std::min(itk, seq);
            auto head_weights = safe_linear(hidden, wproj, idx_weights_proj_scale_t)
                .reshape({batch, seq, inh}) *
                (1.0 / std::sqrt((double)(inh * ihd)));
            auto topk_indices = chunked_topk(
                idx_q_rot, idx_k_rot, head_weights, actual_topk,
                batch, seq, device_id);
            replace_index_state(topk_indices_ptr, idx_bias_keys_ptr,
                                std::move(topk_indices), std::move(head_weights),
                                source_layer, layer_i);
        } else if (!has_indexer) {
            clear_index_state(topk_indices_ptr, idx_bias_keys_ptr, source_layer);
        }

        // Attention computation (same as v4_glm5_dsa_attention)
        at::Tensor context;
        if (*topk_indices_ptr) {
            auto& topk_indices = *reinterpret_cast<at::Tensor*>(*topk_indices_ptr);
            context = sparse_causal_attention(
                q_full, k_full, v, topk_indices, attn_scale);
        } else {
            context = at::scaled_dot_product_attention(q_full, k_full, v, std::nullopt, 0.0, true, attn_scale);
        }

        auto attn_out = safe_linear(context.transpose(1, 2).reshape({batch, seq, nh * vh}), o_w, o_scale_t);

        // ── 3. Residual ──
        auto residual = hidden + attn_out;

        // ── 4. MLP RMSNorm ──
        auto& post_norm_w = *reinterpret_cast<at::Tensor*>(post_norm_weight);
        auto mlp_input = rms_norm(residual, post_norm_w.to(dtype), rms_eps);

        // ── 5. MoE or Dense MLP ──
        at::Tensor mlp_output;
        if (is_moe_layer) {
            mlp_output = moe_forward(
                mlp_input, shared_gate, shared_up, shared_down,
                shared_gate_scale, shared_up_scale, shared_down_scale, gate_weight,
                nullptr,
                expert_gate_weights, expert_up_weights, expert_down_weights,
                expert_gate_scales, expert_up_scales, expert_down_scales,
                n_local_experts, local_expert_indices, n_routed_experts, topk,
                1, 1, 0, 0, 1, routed_scaling_factor,
                nullptr, 0, 1);
        } else {
            // Dense MLP
            auto& dg = *reinterpret_cast<at::Tensor*>(dense_gate);
            auto& du = *reinterpret_cast<at::Tensor*>(dense_up);
            auto& dd = *reinterpret_cast<at::Tensor*>(dense_down);
            auto dg_scale = dense_gate_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(dense_gate_scale)) : std::nullopt;
            auto du_scale = dense_up_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(dense_up_scale)) : std::nullopt;
            auto dd_scale = dense_down_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(dense_down_scale)) : std::nullopt;
            auto go = safe_linear(mlp_input, dg, dg_scale);
            auto uo = safe_linear(mlp_input, du, du_scale);
            auto act = at::silu(go) * uo;
            mlp_output = safe_linear(act, dd, dd_scale);
        }

        // ── 6. Residual ──
        auto new_hidden = residual + mlp_output;
        return new at::Tensor(std::move(new_hidden));
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_layer_forward", e);
        return nullptr;
    }
}

// A stable descriptor keeps the native MTP decoder ABI extensible while making
// the Rust -> C++ boundary one call per layer.  Pointer fields are at::Tensor*;
// nullable weights/scales are represented by nullptr.
struct Glm5MtpDecoderDescriptor {
    void* hidden;
    void* input_norm_weight;
    void* post_norm_weight;

    void* q_a_proj; void* q_a_layernorm; void* q_b_proj;
    void* kv_a_proj; void* kv_a_layernorm; void* kv_b_proj; void* o_proj;
    void* q_a_scale; void* q_b_scale; void* kv_a_scale; void* kv_b_scale; void* o_scale;
    void* idx_wq_b; void* idx_wk; void* idx_k_norm_w; void* idx_k_norm_b;
    void* idx_weights_proj; void* idx_weights_proj_scale;
    void* idx_wq_b_scale; void* idx_wk_scale;

    void* gate_weight; void* correction_bias;
    void* shared_gate; void* shared_up; void* shared_down;
    void* shared_gate_scale; void* shared_up_scale; void* shared_down_scale;
    void* dense_gate; void* dense_up; void* dense_down;
    void* dense_gate_scale; void* dense_up_scale; void* dense_down_scale;
    void** expert_gate_weights; void** expert_up_weights; void** expert_down_weights;
    void** expert_gate_scales; void** expert_up_scales; void** expert_down_scales;
    const int32_t* local_expert_indices;

    void* tp_comm; void* cp_comm; void* ep_comm;
    int32_t tp_size; int32_t cp_rank; int32_t cp_size; int32_t ep_rank; int32_t ep_size;
    int32_t n_local_experts; int32_t n_routed_experts; int32_t topk;
    int32_t n_group; int32_t topk_group;
    int32_t scoring_func;       // 0=sigmoid, 1=softmax
    int32_t topk_method;        // 0=groupwise, 1=noaux_tc
    int32_t norm_topk_prob;
    int32_t is_moe_layer;

    int32_t num_heads; int32_t qk_nope; int32_t qk_rope; int32_t v_head;
    int32_t kv_lora; int32_t idx_head_dim; int32_t idx_n_heads;
    int32_t idx_n_heads_global; int32_t idx_topk;
    int32_t rope_interleave;
    int32_t indexer_rope_interleave;
    double rms_eps; double rope_theta; double routed_scaling_factor;
    double rope_scaling_factor; double rope_beta_fast; double rope_beta_slow;
    double rope_attention_factor;
    int64_t rope_original_max_pos;
    int32_t rope_is_yarn;
};

static at::Tensor descriptor_tensor(void* ptr, const char* name) {
    TORCH_CHECK(ptr, name, " is null");
    return *reinterpret_cast<at::Tensor*>(ptr);
}

static std::optional<at::Tensor> descriptor_optional(void* ptr) {
    return ptr ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(ptr)) : std::nullopt;
}

static at::Tensor distributed_sum_identity(const at::Tensor& input, void* comm,
                                           int32_t group_size, const char* group_name) {
    if (group_size == 1) return input;
    TORCH_CHECK(group_size > 1 && comm, group_name, " communicator is missing");
    TORCH_CHECK(input.is_cuda(), group_name, " collective requires CUDA input");
    return Glm5NcclAllReduceIdentityBackward::apply(
        input, static_cast<int64_t>(reinterpret_cast<intptr_t>(comm)));
}

static std::pair<at::Tensor, at::Tensor> glm5_router(
    const at::Tensor& router_logits, const Glm5MtpDecoderDescriptor& d) {
    TORCH_CHECK(d.n_routed_experts > 0 && d.n_group > 0 &&
                d.n_routed_experts % d.n_group == 0, "invalid router group configuration");
    TORCH_CHECK(d.topk > 0 && d.topk <= d.n_routed_experts, "invalid router top-k");
    TORCH_CHECK(d.topk_group > 0 && d.topk_group <= d.n_group,
                "invalid router top-k group");
    auto logits_f = router_logits.to(at::kFloat);
    at::Tensor scores;
    if (d.scoring_func == 0) scores = at::sigmoid(logits_f);
    else if (d.scoring_func == 1) scores = at::softmax(logits_f, -1, at::kFloat);
    else TORCH_CHECK(false, "unsupported GLM5 router scoring_func");

    at::Tensor selection_scores;
    if (d.topk_method == 1) {
        TORCH_CHECK(d.scoring_func == 0 && d.correction_bias,
                    "noaux_tc requires sigmoid scoring and correction bias");
        selection_scores = scores + descriptor_tensor(d.correction_bias, "correction_bias")
                                      .to(scores.device()).to(at::kFloat);
    } else if (d.topk_method == 0) {
        selection_scores = scores;
    } else {
        TORCH_CHECK(false, "unsupported GLM5 router topk_method");
    }

    const int64_t experts_per_group = d.n_routed_experts / d.n_group;
    auto grouped = selection_scores.reshape({-1, d.n_group, experts_per_group});
    at::Tensor group_scores;
    if (d.topk_method == 1) {
        group_scores = std::get<0>(grouped.topk(std::min<int64_t>(2, experts_per_group),
                                                -1, true, true)).sum(-1);
    } else {
        group_scores = std::get<0>(grouped.max(-1, false));
    }
    auto group_indices = std::get<1>(group_scores.topk(d.topk_group, -1, true, true));
    auto group_mask = at::zeros_like(group_scores).scatter(
        -1, group_indices, at::ones_like(group_indices, group_scores.options()));
    auto expert_mask = group_mask.unsqueeze(-1)
        .expand({-1, -1, experts_per_group}, false)
        .reshape({-1, d.n_routed_experts}).eq(0);
    auto masked = selection_scores.reshape({-1, d.n_routed_experts})
        .masked_fill(expert_mask, -std::numeric_limits<double>::infinity());
    auto topk_indices = std::get<1>(masked.topk(d.topk, -1, true, true));
    auto topk_weights = scores.reshape({-1, d.n_routed_experts})
        .gather(-1, topk_indices, false);
    if (d.norm_topk_prob) {
        topk_weights = topk_weights /
            topk_weights.sum(-1, true).clamp_min(1e-20);
    }
    return {topk_weights * d.routed_scaling_factor, topk_indices};
}

static at::Tensor glm5_mtp_attention(const at::Tensor& input,
                                     const Glm5MtpDecoderDescriptor& d) {
    TORCH_CHECK(input.dim() == 3, "MTP decoder input must be [batch, seq, hidden]");
    TORCH_CHECK(d.tp_size > 0 && d.cp_size > 0 && d.cp_rank >= 0 && d.cp_rank < d.cp_size,
                "invalid MTP TP/CP topology");
    const auto dtype = input.scalar_type();
    const int64_t batch = input.size(0), seq = input.size(1);
    const int64_t global_seq = seq * d.cp_size;
    const int64_t nh = d.num_heads, qn = d.qk_nope, qr = d.qk_rope;
    const int64_t vh = d.v_head, kvl = d.kv_lora;
    const int32_t device_id = input.is_cuda() ? input.device().index() : -1;

    auto q_a = safe_linear(input, descriptor_tensor(d.q_a_proj, "q_a_proj"),
                           descriptor_optional(d.q_a_scale));
    auto q_an = rms_norm(q_a, descriptor_tensor(d.q_a_layernorm, "q_a_layernorm"), d.rms_eps);
    auto q = safe_linear(q_an, descriptor_tensor(d.q_b_proj, "q_b_proj"),
                         descriptor_optional(d.q_b_scale))
        .reshape({batch, seq, nh, qn + qr}).transpose(1, 2);
    auto q_nope = q.narrow(-1, 0, qn);
    auto q_rope = q.narrow(-1, qn, qr);

    auto kv_a = safe_linear(input, descriptor_tensor(d.kv_a_proj, "kv_a_proj"),
                            descriptor_optional(d.kv_a_scale));
    auto kv_lora_raw = kv_a.narrow(-1, 0, kvl);
    auto k_rope = kv_a.narrow(-1, kvl, qr);
    auto kv_an = rms_norm(kv_lora_raw,
                          descriptor_tensor(d.kv_a_layernorm, "kv_a_layernorm"), d.rms_eps);
    auto kv = safe_linear(kv_an, descriptor_tensor(d.kv_b_proj, "kv_b_proj"),
                          descriptor_optional(d.kv_b_scale))
        .reshape({batch, seq, nh, qn + vh});
    auto k_nope = kv.narrow(-1, 0, qn).transpose(1, 2);
    auto value = kv.narrow(-1, qn, vh).transpose(1, 2);

    auto [cos_global, sin_global] = rope_cos_sin(
        global_seq, qr, d.rope_theta, device_id, d.rope_is_yarn != 0,
        d.rope_scaling_factor, d.rope_beta_fast, d.rope_beta_slow,
        d.rope_original_max_pos, d.rope_attention_factor);
    const int64_t offset = static_cast<int64_t>(d.cp_rank) * seq;
    auto cos = cos_global.narrow(0, offset, seq).to(dtype);
    auto sin = sin_global.narrow(0, offset, seq).to(dtype);
    auto k_rope_expanded = k_rope.unsqueeze(2).transpose(1, 2)
        .expand({batch, nh, seq, qr}, false);
    auto q_rot = d.rope_interleave
        ? apply_rotary_interleave(q_rope, cos, sin) : apply_rotary(q_rope, cos, sin);
    auto k_rot = d.rope_interleave
        ? apply_rotary_interleave(k_rope_expanded, cos, sin)
        : apply_rotary(k_rope_expanded, cos, sin);
    auto query = at::cat({q_nope, q_rot}, -1);
    auto key = at::cat({k_nope, k_rot}, -1);
    const double scale = 1.0 / std::sqrt(static_cast<double>(qn + qr));

    // Native MTP layers are independent full DSA layers: compute their global
    // indexer state in this call instead of inheriting trunk IndexShare state.
    std::optional<at::Tensor> global_topk;
    const bool has_indexer = d.idx_wq_b && d.idx_wk && d.idx_k_norm_w &&
                             d.idx_k_norm_b && d.idx_weights_proj;
    if (has_indexer) {
        TORCH_CHECK(d.idx_n_heads > 0 && d.idx_n_heads_global >= d.idx_n_heads,
                    "invalid local/global indexer head counts");
        auto idx_q = safe_linear(q_a, descriptor_tensor(d.idx_wq_b, "idx_wq_b"),
                                 descriptor_optional(d.idx_wq_b_scale))
            .reshape({batch, seq, d.idx_n_heads, d.idx_head_dim}).transpose(1, 2);
        auto idx_k = safe_linear(input, descriptor_tensor(d.idx_wk, "idx_wk"),
                                 descriptor_optional(d.idx_wk_scale));
        idx_k = rms_norm_with_bias(idx_k,
            descriptor_tensor(d.idx_k_norm_w, "idx_k_norm_w"),
            descriptor_tensor(d.idx_k_norm_b, "idx_k_norm_b"), d.rms_eps)
            .unsqueeze(1).expand({batch, d.idx_n_heads, seq, d.idx_head_dim}, false);
        auto idx_q_rot = apply_indexer_rope(
            idx_q, cos, sin, qr, d.indexer_rope_interleave != 0);
        auto idx_k_rot = apply_indexer_rope(
            idx_k, cos, sin, qr, d.indexer_rope_interleave != 0);
        auto head_weights = safe_linear(input,
            descriptor_tensor(d.idx_weights_proj, "idx_weights_proj"),
            descriptor_optional(d.idx_weights_proj_scale))
            .reshape({batch, seq, d.idx_n_heads}).transpose(1, 2).to(at::kFloat) *
            (1.0 / std::sqrt(static_cast<double>(
                d.idx_n_heads_global * d.idx_head_dim)));
        std::optional<at::Tensor> best_scores, best_indices;
        auto current = idx_k_rot;
        const int64_t actual_topk = std::min<int64_t>(d.idx_topk, global_seq);
        for (int32_t block = 0; block < d.cp_size; ++block) {
            const int32_t peer = (d.cp_rank + d.cp_size - block) % d.cp_size;
            const int64_t key_offset = static_cast<int64_t>(peer) * seq;
            auto local_scores = (idx_q_rot.matmul(current.transpose(-2, -1)).relu()
                .to(at::kFloat) * head_weights.unsqueeze(-1)).sum(1, true);
            auto scores = distributed_sum_identity(local_scores, d.tp_comm, d.tp_size, "TP");
            auto q_pos = at::arange(seq, scores.options().dtype(at::kLong)) + offset;
            auto k_pos = at::arange(seq, scores.options().dtype(at::kLong)) + key_offset;
            auto future = k_pos.unsqueeze(0) > q_pos.unsqueeze(1);
            scores = scores.masked_fill(future.unsqueeze(0).unsqueeze(0),
                                        -std::numeric_limits<double>::infinity());
            auto block_result = scores.topk(std::min<int64_t>(actual_topk, seq), -1, true, true);
            auto block_scores = std::get<0>(block_result);
            auto block_indices = std::get<1>(block_result) + key_offset;
            if (best_scores) {
                auto merged_s = at::cat({*best_scores, block_scores}, -1);
                auto merged_i = at::cat({*best_indices, block_indices}, -1);
                auto keep = merged_s.topk(std::min<int64_t>(actual_topk, merged_s.size(-1)),
                                          -1, true, true);
                best_scores = std::get<0>(keep);
                best_indices = merged_i.gather(-1, std::get<1>(keep), false);
            } else {
                best_scores = block_scores;
                best_indices = block_indices;
            }
            if (block + 1 < d.cp_size) {
                TORCH_CHECK(d.cp_comm && input.is_cuda(),
                            "CP indexer ring requires CUDA input and communicator");
                const int64_t next = (d.cp_rank + 1) % d.cp_size;
                const int64_t prev = (d.cp_rank + d.cp_size - 1) % d.cp_size;
                current = Glm5NcclRingFunction::apply(current,
                    static_cast<int64_t>(reinterpret_cast<intptr_t>(d.cp_comm)), next, prev);
            }
        }
        global_topk = best_indices->expand({batch, nh, seq, actual_topk}, false);
    }

    at::Tensor context;
    if (d.cp_size == 1) {
        if (global_topk) context = sparse_causal_attention(query, key, value, *global_topk, scale);
        else context = at::scaled_dot_product_attention(query, key, value, std::nullopt,
                                                        0.0, true, scale);
    } else {
        TORCH_CHECK(d.cp_comm && input.is_cuda(),
                    "CP attention ring requires CUDA input and communicator");
        std::optional<at::Tensor> run_max, run_num, run_den;
        auto current_k = key;
        auto current_v = value;
        for (int32_t block = 0; block < d.cp_size; ++block) {
            const int32_t peer = (d.cp_rank + d.cp_size - block) % d.cp_size;
            const int64_t key_offset = static_cast<int64_t>(peer) * seq;
            auto q_pos = at::arange(seq, query.options().dtype(at::kLong)) + offset;
            auto k_pos = at::arange(seq, query.options().dtype(at::kLong)) + key_offset;
            auto allowed = (k_pos.unsqueeze(0) <= q_pos.unsqueeze(1))
                .unsqueeze(0).unsqueeze(0).expand({batch, nh, seq, seq}, false);
            if (global_topk) {
                auto in_block = global_topk->ge(key_offset).logical_and(global_topk->lt(key_offset + seq));
                auto local_indices = (*global_topk - key_offset).masked_fill(in_block.logical_not(), 0);
                auto sparse = at::zeros({batch, nh, seq, seq}, query.options());
                sparse.scatter_add_(-1, local_indices,
                                    at::ones_like(local_indices, query.options()) * in_block.to(dtype));
                allowed = allowed.logical_and(sparse.ne(0));
            }
            auto scores = (query.matmul(current_k.transpose(-2, -1)) * scale)
                .masked_fill(allowed.logical_not(), -std::numeric_limits<double>::infinity());
            auto finite = scores.isfinite();
            auto valid = finite.any(-1, true);
            auto block_max = scores.masked_fill(finite.logical_not(), 0).amax(-1, true)
                .masked_fill(valid.logical_not(), -std::numeric_limits<double>::infinity());
            auto exp_scores = (scores - block_max).exp().masked_fill(finite.logical_not(), 0);
            auto block_num = exp_scores.matmul(current_v);
            auto block_den = exp_scores.sum(-1, true);
            if (run_max) {
                auto new_max = at::maximum(*run_max, block_max);
                auto old_scale = (*run_max - new_max).exp().masked_fill(run_max->isfinite().logical_not(), 0);
                auto new_scale = (block_max - new_max).exp().masked_fill(block_max.isfinite().logical_not(), 0);
                run_num = *run_num * old_scale + block_num * new_scale;
                run_den = *run_den * old_scale + block_den * new_scale;
                run_max = new_max;
            } else {
                run_max = block_max; run_num = block_num; run_den = block_den;
            }
            if (block + 1 < d.cp_size) {
                const int64_t next = (d.cp_rank + 1) % d.cp_size;
                const int64_t prev = (d.cp_rank + d.cp_size - 1) % d.cp_size;
                auto received = Glm5NcclKvRingFunction::apply(
                    current_k, current_v,
                    static_cast<int64_t>(reinterpret_cast<intptr_t>(d.cp_comm)), next, prev);
                current_k = received[0]; current_v = received[1];
            }
        }
        context = *run_num / run_den->clamp_min(1e-20);
    }

    auto local_out = safe_linear(context.transpose(1, 2).reshape({batch, seq, nh * vh}),
                                 descriptor_tensor(d.o_proj, "o_proj"),
                                 descriptor_optional(d.o_scale));
    return distributed_sum_identity(local_out, d.tp_comm, d.tp_size, "TP");
}

void* v4_glm5_mtp_decoder_layer(const Glm5MtpDecoderDescriptor* descriptor) {
    try {
        g_glm5_last_error.clear();
        TORCH_CHECK(descriptor, "MTP decoder descriptor is null");
        const auto& d = *descriptor;
        TORCH_CHECK(d.tp_size > 0 && d.cp_size > 0 && d.ep_size > 0,
                    "MTP parallel sizes must be positive");
        auto hidden = descriptor_tensor(d.hidden, "hidden");
        auto normalized = rms_norm(hidden,
            descriptor_tensor(d.input_norm_weight, "input_norm_weight"), d.rms_eps);
        auto attention = glm5_mtp_attention(normalized, d);
        auto residual = hidden + attention;
        auto mlp_input = rms_norm(residual,
            descriptor_tensor(d.post_norm_weight, "post_norm_weight"), d.rms_eps);

        at::Tensor mlp_output;
        if (d.is_moe_layer) {
            TORCH_CHECK(d.gate_weight && d.shared_gate && d.shared_up && d.shared_down,
                        "MTP MoE weights are incomplete");
            auto shared_input = d.tp_size == 1
                ? mlp_input
                : Glm5NcclIdentityAllReduceBackward::apply(
                      mlp_input, static_cast<int64_t>(reinterpret_cast<intptr_t>(d.tp_comm)));
            auto shared_local = safe_linear(
                at::silu(safe_linear(shared_input, descriptor_tensor(d.shared_gate, "shared_gate"),
                                     descriptor_optional(d.shared_gate_scale))) *
                safe_linear(shared_input, descriptor_tensor(d.shared_up, "shared_up"),
                            descriptor_optional(d.shared_up_scale)),
                descriptor_tensor(d.shared_down, "shared_down"),
                descriptor_optional(d.shared_down_scale));
            auto shared = distributed_sum_identity(shared_local, d.tp_comm, d.tp_size, "TP");
            auto router_logits = safe_linear(mlp_input,
                descriptor_tensor(d.gate_weight, "gate_weight"), std::nullopt);
            auto [weights, indices] = glm5_router(router_logits, d);
            const int64_t hidden_size = mlp_input.size(-1);
            auto flat = mlp_input.reshape({-1, hidden_size});
            at::Tensor expert_input;
            at::Tensor expert_ids;
            at::Tensor send_counts;
            at::Tensor recv_counts;
            at::Tensor sort_order;
            if (d.ep_size > 1) {
                TORCH_CHECK(d.ep_comm && d.ep_rank >= 0 && d.ep_rank < d.ep_size,
                            "MTP EP communicator coordinates are invalid");
                TORCH_CHECK(d.n_routed_experts % d.ep_size == 0,
                            "MTP routed expert count must be divisible by EP size");
                const int64_t assignments = indices.numel();
                auto assignment_ids = at::arange(assignments, indices.options());
                auto token_ids = at::floor_divide(assignment_ids, d.topk);
                auto flat_experts = indices.reshape({-1});
                auto owners = at::floor_divide(
                    flat_experts, d.n_routed_experts / d.ep_size);
                auto sort_key = owners * assignments + assignment_ids;
                sort_order = std::get<1>(sort_key.sort(0, false));
                auto sorted_input = flat.index_select(0, token_ids).index_select(0, sort_order);
                auto sorted_experts = flat_experts.index_select(0, sort_order);
                std::vector<at::Tensor> count_parts;
                count_parts.reserve(d.ep_size);
                for (int32_t peer = 0; peer < d.ep_size; ++peer) {
                    count_parts.push_back(owners.eq(peer).sum(at::kLong));
                }
                send_counts = at::stack(count_parts).contiguous();
                auto dispatched = Glm5NcclEpDispatchFunction::apply(
                    sorted_input, sorted_experts, send_counts,
                    static_cast<int64_t>(reinterpret_cast<intptr_t>(d.ep_comm)),
                    d.ep_rank, d.ep_size);
                expert_input = dispatched[0];
                expert_ids = dispatched[1];
                recv_counts = dispatched[2];
            } else {
                auto assignment_ids = at::arange(indices.numel(), indices.options());
                expert_input = flat.index_select(
                    0, at::floor_divide(assignment_ids, d.topk));
                expert_ids = indices.reshape({-1});
            }

            auto expert_output = at::zeros_like(expert_input);
            for (int32_t local = 0; local < d.n_local_experts; ++local) {
                TORCH_CHECK(d.local_expert_indices && d.expert_gate_weights &&
                            d.expert_up_weights && d.expert_down_weights,
                            "MTP local expert arrays are null");
                auto positions = expert_ids.eq(d.local_expert_indices[local]).nonzero();
                if (positions.size(0) == 0) continue;
                auto token_ids = positions.select(1, 0);
                auto selected = expert_input.index_select(0, token_ids);
                auto gate = safe_linear(selected,
                    descriptor_tensor(d.expert_gate_weights[local], "expert_gate"),
                    optional_tensor_at(d.expert_gate_scales, local));
                auto up = safe_linear(selected,
                    descriptor_tensor(d.expert_up_weights[local], "expert_up"),
                    optional_tensor_at(d.expert_up_scales, local));
                auto expert = safe_linear(at::silu(gate) * up,
                    descriptor_tensor(d.expert_down_weights[local], "expert_down"),
                    optional_tensor_at(d.expert_down_scales, local));
                expert_output.index_add_(0, token_ids, expert);
            }
            at::Tensor routed;
            if (d.ep_size > 1) {
                auto returned_sorted = Glm5NcclEpReturnFunction::apply(
                    expert_output, send_counts, recv_counts,
                    static_cast<int64_t>(reinterpret_cast<intptr_t>(d.ep_comm)),
                    d.ep_rank, d.ep_size);
                auto assignment_ids = at::arange(sort_order.numel(), sort_order.options());
                auto inverse_order = at::empty_like(sort_order);
                inverse_order.scatter_(0, sort_order, assignment_ids);
                auto returned = returned_sorted.index_select(0, inverse_order)
                    .reshape({flat.size(0), d.topk, hidden_size});
                routed = (returned * weights.to(hidden.scalar_type()).unsqueeze(-1))
                    .sum(1).reshape_as(mlp_input);
            } else {
                auto returned = expert_output.reshape({flat.size(0), d.topk, hidden_size});
                routed = (returned * weights.to(hidden.scalar_type()).unsqueeze(-1))
                    .sum(1).reshape_as(mlp_input);
            }
            mlp_output = routed + shared;
        } else {
            TORCH_CHECK(d.dense_gate && d.dense_up && d.dense_down,
                        "MTP dense MLP weights are incomplete");
            auto dense_input = d.tp_size == 1
                ? mlp_input
                : Glm5NcclIdentityAllReduceBackward::apply(
                      mlp_input, static_cast<int64_t>(reinterpret_cast<intptr_t>(d.tp_comm)));
            auto gate = safe_linear(dense_input, descriptor_tensor(d.dense_gate, "dense_gate"),
                                    descriptor_optional(d.dense_gate_scale));
            auto up = safe_linear(dense_input, descriptor_tensor(d.dense_up, "dense_up"),
                                  descriptor_optional(d.dense_up_scale));
            auto dense_local = safe_linear(at::silu(gate) * up,
                                           descriptor_tensor(d.dense_down, "dense_down"),
                                           descriptor_optional(d.dense_down_scale));
            mlp_output = distributed_sum_identity(dense_local, d.tp_comm, d.tp_size, "TP");
        }
        return new at::Tensor(residual + mlp_output);
    } catch (const std::exception& e) {
        set_glm5_error("v4_glm5_mtp_decoder_layer", e);
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_stream_wait_event — make PyTorch's current CUDA stream wait for an event
// This is the key for async overlap: CPU doesn't block, GPU handles the dependency.
// ══════════════════════════════════════════════════════════════════════
void v4_stream_wait_event(int32_t device_id, void* event_ptr) {
    try {
        cudaSetDevice(device_id);
        auto stream = c10::cuda::getCurrentCUDAStream(c10::cuda::current_device());
        cudaEvent_t event = reinterpret_cast<cudaEvent_t>(event_ptr);
        cudaStreamWaitEvent(stream.stream(), event, 0);
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_stream_wait_event] FAILED: %s\n", e.what());
    }
}

} // extern "C"
