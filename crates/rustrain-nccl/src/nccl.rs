use std::{
    ffi::{c_char, c_int, c_uint, c_void, CStr},
    fs,
    path::{Path, PathBuf},
    ptr,
    thread::sleep,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use tch::{Kind, Tensor};

const NCCL_UNIQUE_ID_BYTES: usize = 128;
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct NcclUniqueId {
    internal: [c_char; NCCL_UNIQUE_ID_BYTES],
}

type NcclComm = *mut c_void;
type NcclResult = c_int;
type NcclDataType = c_int;
type NcclRedOp = c_int;
type CudaError = c_int;
type CudaStream = *mut c_void;
type CudaEvent = *mut c_void;

const NCCL_FLOAT32: NcclDataType = 7;
const NCCL_BF16: NcclDataType = 9;
const NCCL_SUM: NcclRedOp = 0;
const CUDA_EVENT_DISABLE_TIMING: c_uint = 0x2;

#[link(name = "nccl")]
unsafe extern "C" {
    fn ncclGetUniqueId(unique_id: *mut NcclUniqueId) -> NcclResult;
    fn ncclCommInitRank(
        comm: *mut NcclComm,
        nranks: c_int,
        unique_id: NcclUniqueId,
        rank: c_int,
    ) -> NcclResult;
    fn ncclAllReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: CudaStream,
    ) -> NcclResult;
    fn ncclSend(
        sendbuff: *const c_void,
        count: usize,
        datatype: NcclDataType,
        peer: c_int,
        comm: NcclComm,
        stream: CudaStream,
    ) -> NcclResult;
    fn ncclRecv(
        recvbuff: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        peer: c_int,
        comm: NcclComm,
        stream: CudaStream,
    ) -> NcclResult;
    fn ncclGroupStart() -> NcclResult;
    fn ncclGroupEnd() -> NcclResult;
    fn ncclCommDestroy(comm: NcclComm) -> NcclResult;
    fn ncclGetErrorString(result: NcclResult) -> *const c_char;
}

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaSetDevice(device: c_int) -> CudaError;
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> CudaError;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> CudaError;
    fn cudaDeviceSynchronize() -> CudaError;
    fn cudaFree(dev_ptr: *mut c_void) -> CudaError;
    fn cudaGetErrorString(error: CudaError) -> *const c_char;
    fn cudaStreamCreate(stream: *mut CudaStream) -> CudaError;
    fn cudaStreamSynchronize(stream: CudaStream) -> CudaError;
    fn cudaStreamDestroy(stream: CudaStream) -> CudaError;
    fn cudaEventCreate(event: *mut CudaEvent) -> CudaError;
    fn cudaEventCreateWithFlags(event: *mut CudaEvent, flags: c_uint) -> CudaError;
    fn cudaEventRecord(event: CudaEvent, stream: CudaStream) -> CudaError;
    fn cudaStreamWaitEvent(stream: CudaStream, event: CudaEvent, flags: c_uint) -> CudaError;
    fn cudaEventDestroy(event: CudaEvent) -> CudaError;
}

const DP_WEIGHT: [f32; 2] = [0.2, -0.1];
const DP_DATASET: [([f32; 2], f32); 4] = [
    ([1.0, 0.0], 0.7),
    ([0.0, 1.0], -0.3),
    ([1.0, 1.0], 0.4),
    ([2.0, -1.0], 1.2),
];

#[derive(Debug, Serialize, Deserialize)]
struct NcclRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    input: f32,
    reduced: f32,
    expected: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct NcclDpGradientSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    local_sample_count: usize,
    total_sample_count: f32,
    local_grad_sum: [f32; 2],
    reduced_grad_sum: [f32; 2],
    averaged_grad: [f32; 2],
    expected_grad: [f32; 2],
    grad_max_delta: f32,
    local_loss_sum: f32,
    reduced_loss_sum: f32,
    global_loss: f32,
    expected_loss: f32,
    loss_delta: f32,
}

pub fn run_nccl_all_reduce_rank(output_dir: PathBuf) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }

    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let id_path = output_dir.join("nccl-unique-id.bin");
    let unique_id = if rank == 0 {
        let id = nccl_unique_id()?;
        fs::write(&id_path, unique_id_to_bytes(&id))
            .with_context(|| format!("failed to write {}", id_path.display()))?;
        id
    } else {
        wait_for_unique_id(&id_path, Duration::from_secs(30))?
    };

    let input = (rank + 1) as f32;
    let expected = (world_size * (world_size + 1) / 2) as f32;
    let reduced = nccl_all_reduce_values(unique_id, rank, world_size, local_rank, &[input])?[0];
    if (reduced - expected).abs() > 1e-5 {
        bail!("NCCL all-reduce mismatch: rank={rank}, reduced={reduced}, expected={expected}");
    }

    let summary = NcclRankSummary {
        rank,
        world_size,
        local_rank,
        input,
        reduced,
        expected,
    };
    let summary_path = output_dir.join(format!("nccl-rank-{rank}.json"));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn run_nccl_dp_gradient_rank(output_dir: PathBuf) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }

    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let unique_id = shared_unique_id(&output_dir, rank)?;

    let local = compute_dp_stats(rank, world_size);
    let reduced = nccl_all_reduce_values(
        unique_id,
        rank,
        world_size,
        local_rank,
        &[
            local.grad_sum[0],
            local.grad_sum[1],
            local.loss_sum,
            local.sample_count as f32,
        ],
    )?;
    let total_sample_count = reduced[3];
    let averaged_grad = [
        reduced[0] / total_sample_count,
        reduced[1] / total_sample_count,
    ];
    let global_loss = reduced[2] / total_sample_count;

    let expected = compute_dp_stats(0, 1);
    let expected_sample_count = expected.sample_count as f32;
    let expected_grad = [
        expected.grad_sum[0] / expected_sample_count,
        expected.grad_sum[1] / expected_sample_count,
    ];
    let expected_loss = expected.loss_sum / expected_sample_count;
    let grad_max_delta = averaged_grad
        .into_iter()
        .zip(expected_grad)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let loss_delta = (global_loss - expected_loss).abs();

    if grad_max_delta > 1e-6 || loss_delta > 1e-6 {
        bail!(
            "NCCL DP gradient mismatch: rank={rank}, grad_max_delta={grad_max_delta}, loss_delta={loss_delta}"
        );
    }

    let summary = NcclDpGradientSummary {
        rank,
        world_size,
        local_rank,
        local_sample_count: local.sample_count,
        total_sample_count,
        local_grad_sum: local.grad_sum,
        reduced_grad_sum: [reduced[0], reduced[1]],
        averaged_grad,
        expected_grad,
        grad_max_delta,
        local_loss_sum: local.loss_sum,
        reduced_loss_sum: reduced[2],
        global_loss,
        expected_loss,
        loss_delta,
    };
    let summary_path = output_dir.join(format!("nccl-dp-gradient-rank-{rank}.json"));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn all_reduce_f32_for_launch(output_dir: &Path, values: &[f32]) -> Result<Vec<f32>> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let unique_id = shared_unique_id(output_dir, rank)?;
    nccl_all_reduce_values(unique_id, rank, world_size, local_rank, values)
}

pub fn all_reduce_tensor_f32_for_launch(output_dir: &Path, tensor: &Tensor) -> Result<Tensor> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let tensor = tensor.to_kind(Kind::Float).contiguous();
    if tensor.numel() == 0 {
        bail!("NCCL tensor all-reduce input must not be empty");
    }
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let unique_id = shared_unique_id(output_dir, rank)?;
    unsafe { nccl_all_reduce_tensor_unsafe(unique_id, rank, world_size, local_rank, &tensor) }
}

pub fn send_recv_tensors_f32_for_launch(
    output_dir: &Path,
    sends: &[(usize, Tensor)],
    recvs: &[(usize, Vec<i64>)],
) -> Result<Vec<(usize, Tensor)>> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    if sends.is_empty() && recvs.is_empty() {
        return Ok(Vec::new());
    }
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let unique_id = shared_unique_id(output_dir, rank)?;
    unsafe { nccl_send_recv_tensors_unsafe(unique_id, rank, world_size, local_rank, sends, recvs) }
}

fn shared_unique_id(output_dir: &Path, rank: usize) -> Result<NcclUniqueId> {
    let id_path = output_dir.join("nccl-unique-id.bin");
    if rank == 0 {
        let id = nccl_unique_id()?;
        fs::write(&id_path, unique_id_to_bytes(&id))
            .with_context(|| format!("failed to write {}", id_path.display()))?;
        Ok(id)
    } else {
        // Multi-node: increase timeout to 300s for cross-node vePFS access
        let timeout = if std::env::var("NNODES").unwrap_or("1".to_string()).parse::<usize>().unwrap_or(1) > 1 {
            Duration::from_secs(300)
        } else {
            Duration::from_secs(30)
        };
        wait_for_unique_id(&id_path, timeout)
    }
}

fn wait_for_unique_id(path: &Path, timeout: Duration) -> Result<NcclUniqueId> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if path.exists() {
            let bytes =
                fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
            if bytes.len() == NCCL_UNIQUE_ID_BYTES {
                return unique_id_from_bytes(&bytes);
            }
            // File exists but not fully written yet (0 bytes on network FS) — keep waiting
        }
        sleep(Duration::from_millis(50));
    }
    bail!("timed out waiting for {}", path.display())
}

fn nccl_unique_id() -> Result<NcclUniqueId> {
    let mut id = NcclUniqueId {
        internal: [0; NCCL_UNIQUE_ID_BYTES],
    };
    unsafe {
        check_nccl(ncclGetUniqueId(&mut id), "ncclGetUniqueId")?;
    }
    Ok(id)
}

fn nccl_all_reduce_values(
    unique_id: NcclUniqueId,
    rank: usize,
    world_size: usize,
    local_rank: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    unsafe { nccl_all_reduce_values_unsafe(unique_id, rank, world_size, local_rank, input) }
}

unsafe fn nccl_all_reduce_tensor_unsafe(
    unique_id: NcclUniqueId,
    rank: usize,
    world_size: usize,
    local_rank: usize,
    input: &Tensor,
) -> Result<Tensor> {
    check_cuda(
        unsafe { cudaSetDevice(local_rank as c_int) },
        "cudaSetDevice",
    )?;
    let output = input.zeros_like();
    let mut comm: NcclComm = ptr::null_mut();
    check_nccl(
        unsafe { ncclCommInitRank(&mut comm, world_size as c_int, unique_id, rank as c_int) },
        "ncclCommInitRank",
    )?;
    let reduce_result = check_nccl(
        unsafe {
            ncclAllReduce(
                input.data_ptr().cast_const(),
                output.data_ptr(),
                input.numel(),
                NCCL_FLOAT32,
                NCCL_SUM,
                comm,
                ptr::null_mut(),
            )
        },
        "ncclAllReduce",
    )
    .and_then(|_| check_cuda(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize"));
    let destroy_result = check_nccl(unsafe { ncclCommDestroy(comm) }, "ncclCommDestroy");
    reduce_result?;
    destroy_result?;
    Ok(output)
}

unsafe fn nccl_all_reduce_values_unsafe(
    unique_id: NcclUniqueId,
    rank: usize,
    world_size: usize,
    local_rank: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    if input.is_empty() {
        bail!("NCCL all-reduce input must not be empty");
    }
    check_cuda(
        unsafe { cudaSetDevice(local_rank as c_int) },
        "cudaSetDevice",
    )?;

    let mut send: *mut c_void = ptr::null_mut();
    let mut recv: *mut c_void = ptr::null_mut();
    let bytes = std::mem::size_of_val(input);
    check_cuda(unsafe { cudaMalloc(&mut send, bytes) }, "cudaMalloc(send)")?;
    check_cuda(unsafe { cudaMalloc(&mut recv, bytes) }, "cudaMalloc(recv)")?;

    let result = (|| {
        check_cuda(
            unsafe {
                cudaMemcpy(
                    send,
                    input.as_ptr().cast::<c_void>(),
                    bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "cudaMemcpy host-to-device",
        )?;

        let mut comm: NcclComm = ptr::null_mut();
        check_nccl(
            unsafe { ncclCommInitRank(&mut comm, world_size as c_int, unique_id, rank as c_int) },
            "ncclCommInitRank",
        )?;
        let reduce_result = check_nccl(
            unsafe {
                ncclAllReduce(
                    send.cast_const(),
                    recv,
                    input.len(),
                    NCCL_FLOAT32,
                    NCCL_SUM,
                    comm,
                    ptr::null_mut(),
                )
            },
            "ncclAllReduce",
        )
        .and_then(|_| check_cuda(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize"));
        let destroy_result = check_nccl(unsafe { ncclCommDestroy(comm) }, "ncclCommDestroy");
        reduce_result?;
        destroy_result?;

        let mut output = vec![0.0_f32; input.len()];
        check_cuda(
            unsafe {
                cudaMemcpy(
                    output.as_mut_ptr().cast::<c_void>(),
                    recv.cast_const(),
                    bytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "cudaMemcpy device-to-host",
        )?;
        Ok(output)
    })();

    let output = result;
    check_cuda(unsafe { cudaFree(send) }, "cudaFree(send)")?;
    check_cuda(unsafe { cudaFree(recv) }, "cudaFree(recv)")?;
    output
}

unsafe fn nccl_send_recv_tensors_unsafe(
    unique_id: NcclUniqueId,
    rank: usize,
    world_size: usize,
    local_rank: usize,
    sends: &[(usize, Tensor)],
    recvs: &[(usize, Vec<i64>)],
) -> Result<Vec<(usize, Tensor)>> {
    check_cuda(
        unsafe { cudaSetDevice(local_rank as c_int) },
        "cudaSetDevice",
    )?;

    let send_tensors = sends
        .iter()
        .map(|(peer, tensor)| {
            if *peer >= world_size {
                bail!("NCCL send peer {peer} must be smaller than world_size {world_size}");
            }
            let tensor = tensor.to_kind(Kind::Float).contiguous();
            Ok((*peer, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    let recv_tensors = recvs
        .iter()
        .map(|(peer, shape)| {
            if *peer >= world_size {
                bail!("NCCL recv peer {peer} must be smaller than world_size {world_size}");
            }
            if shape.is_empty() || shape.iter().any(|dim| *dim < 0) {
                bail!("NCCL recv shape for peer {peer} must be non-empty and non-negative");
            }
            let tensor = Tensor::zeros(
                shape.as_slice(),
                (Kind::Float, tch::Device::Cuda(local_rank)),
            );
            Ok((*peer, tensor))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut comm: NcclComm = ptr::null_mut();
    check_nccl(
        unsafe { ncclCommInitRank(&mut comm, world_size as c_int, unique_id, rank as c_int) },
        "ncclCommInitRank",
    )?;
    let transfer_result = (|| {
        check_nccl(unsafe { ncclGroupStart() }, "ncclGroupStart")?;
        for (peer, tensor) in &recv_tensors {
            if tensor.numel() == 0 {
                continue;
            }
            check_nccl(
                unsafe {
                    ncclRecv(
                        tensor.data_ptr(),
                        tensor.numel(),
                        NCCL_FLOAT32,
                        *peer as c_int,
                        comm,
                        ptr::null_mut(),
                    )
                },
                "ncclRecv",
            )?;
        }
        for (peer, tensor) in &send_tensors {
            if tensor.numel() == 0 {
                continue;
            }
            check_nccl(
                unsafe {
                    ncclSend(
                        tensor.data_ptr().cast_const(),
                        tensor.numel(),
                        NCCL_FLOAT32,
                        *peer as c_int,
                        comm,
                        ptr::null_mut(),
                    )
                },
                "ncclSend",
            )?;
        }
        check_nccl(unsafe { ncclGroupEnd() }, "ncclGroupEnd")?;
        check_cuda(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
    })();
    let destroy_result = check_nccl(unsafe { ncclCommDestroy(comm) }, "ncclCommDestroy");
    transfer_result?;
    destroy_result?;
    Ok(recv_tensors)
}

#[derive(Debug)]
struct DpStats {
    sample_count: usize,
    loss_sum: f32,
    grad_sum: [f32; 2],
}

fn compute_dp_stats(rank: usize, world_size: usize) -> DpStats {
    let mut sample_count = 0;
    let mut loss_sum = 0.0;
    let mut grad_sum = [0.0_f32; 2];

    for (sample_index, (features, target)) in DP_DATASET.iter().enumerate() {
        if sample_index % world_size != rank {
            continue;
        }
        let prediction = DP_WEIGHT[0] * features[0] + DP_WEIGHT[1] * features[1];
        let error = prediction - target;
        loss_sum += 0.5 * error * error;
        grad_sum[0] += error * features[0];
        grad_sum[1] += error * features[1];
        sample_count += 1;
    }

    DpStats {
        sample_count,
        loss_sum,
        grad_sum,
    }
}

fn unique_id_to_bytes(id: &NcclUniqueId) -> Vec<u8> {
    id.internal.iter().map(|value| *value as u8).collect()
}

fn unique_id_from_bytes(bytes: &[u8]) -> Result<NcclUniqueId> {
    if bytes.len() != NCCL_UNIQUE_ID_BYTES {
        bail!(
            "NCCL unique ID must be {NCCL_UNIQUE_ID_BYTES} bytes, got {}",
            bytes.len()
        );
    }
    let mut internal = [0 as c_char; NCCL_UNIQUE_ID_BYTES];
    for (dst, src) in internal.iter_mut().zip(bytes.iter().copied()) {
        *dst = src as c_char;
    }
    Ok(NcclUniqueId { internal })
}

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set; run through rustrain launch"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

fn check_nccl(result: NcclResult, context: &str) -> Result<()> {
    if result == 0 {
        return Ok(());
    }
    let message = unsafe { c_string(ncclGetErrorString(result)) };
    Err(anyhow!(
        "{context} failed with NCCL error {result}: {message}"
    ))
}

fn check_cuda(result: CudaError, context: &str) -> Result<()> {
    if result == 0 {
        return Ok(());
    }
    let message = unsafe { c_string(cudaGetErrorString(result)) };
    Err(anyhow!(
        "{context} failed with CUDA error {result}: {message}"
    ))
}

unsafe fn c_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return "<null>".to_string();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nccl_unique_id_roundtrips_bytes() {
        let mut id = NcclUniqueId {
            internal: [0; NCCL_UNIQUE_ID_BYTES],
        };
        for index in 0..NCCL_UNIQUE_ID_BYTES {
            id.internal[index] = index as c_char;
        }
        let bytes = unique_id_to_bytes(&id);
        let restored = unique_id_from_bytes(&bytes).expect("unique ID bytes should roundtrip");
        assert_eq!(unique_id_to_bytes(&restored), bytes);
    }

    #[test]
    fn dp_gradient_partitions_match_global_batch() {
        let rank0 = compute_dp_stats(0, 2);
        let rank1 = compute_dp_stats(1, 2);
        let single = compute_dp_stats(0, 1);

        assert_eq!(rank0.sample_count + rank1.sample_count, single.sample_count);
        assert!((rank0.loss_sum + rank1.loss_sum - single.loss_sum).abs() < 1e-6);
        let reduced_grad = [
            rank0.grad_sum[0] + rank1.grad_sum[0],
            rank0.grad_sum[1] + rank1.grad_sum[1],
        ];
        for (actual, expected) in reduced_grad.into_iter().zip(single.grad_sum) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn subgroup_coordinates_reject_invalid_sizes_and_ranks() {
        let zero = validate_group_coordinates(0, 0, 0).unwrap_err();
        assert!(zero.to_string().contains("greater than zero"));

        let out_of_range = validate_group_coordinates(2, 2, 0).unwrap_err();
        assert!(out_of_range.to_string().contains("rank 2"));
    }

    #[test]
    fn subgroup_coordinates_accept_valid_explicit_ranks() {
        validate_group_coordinates(1, 4, 3).expect("valid subgroup coordinates should pass");
    }

    #[test]
    fn persistent_exchange_rejects_stale_files_across_arrival_orders() {
        let exchange_dir = std::env::temp_dir().join(format!(
            "rustrain-nccl-rendezvous-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::create_dir_all(&exchange_dir).expect("create exchange directory");
        let stale = test_unique_id(11);
        let fresh = test_unique_id(29);
        write_bytes_atomically(
            &exchange_dir.join("nccl-persistent-id.bin"),
            &unique_id_to_bytes(&stale),
        )
        .expect("write stale ID");
        let mut stale_release = unique_id_to_bytes(&stale);
        stale_release.extend(b"stale-participant");
        write_bytes_atomically(
            &exchange_dir.join("nccl-release-rank-1.bin"),
            &stale_release,
        )
        .expect("write stale release");

        let peer_dir = exchange_dir.clone();
        let peer = std::thread::spawn(move || {
            exchange_persistent_unique_id(&peer_dir, 1, 2, None, Duration::from_secs(2))
        });
        std::thread::sleep(Duration::from_millis(100));
        let root_id =
            exchange_persistent_unique_id(&exchange_dir, 0, 2, Some(fresh), Duration::from_secs(2))
                .expect("root rendezvous");
        let peer_id = peer.join().expect("join peer").expect("peer rendezvous");
        assert_eq!(unique_id_to_bytes(&root_id), unique_id_to_bytes(&fresh));
        assert_eq!(unique_id_to_bytes(&peer_id), unique_id_to_bytes(&fresh));

        let second = test_unique_id(47);
        let root_dir = exchange_dir.clone();
        let root = std::thread::spawn(move || {
            exchange_persistent_unique_id(&root_dir, 0, 2, Some(second), Duration::from_secs(2))
        });
        std::thread::sleep(Duration::from_millis(100));
        let peer_id =
            exchange_persistent_unique_id(&exchange_dir, 1, 2, None, Duration::from_secs(2))
                .expect("second peer rendezvous");
        let root_id = root
            .join()
            .expect("join root")
            .expect("second root rendezvous");
        assert_eq!(unique_id_to_bytes(&root_id), unique_id_to_bytes(&second));
        assert_eq!(unique_id_to_bytes(&peer_id), unique_id_to_bytes(&second));

        fs::remove_dir_all(exchange_dir).expect("remove exchange directory");
    }

    fn test_unique_id(seed: u8) -> NcclUniqueId {
        let mut id = NcclUniqueId {
            internal: [0; NCCL_UNIQUE_ID_BYTES],
        };
        for (index, byte) in id.internal.iter_mut().enumerate() {
            *byte = seed.wrapping_add(index as u8) as c_char;
        }
        id
    }
}

// ── Persistent NCCL Communicator ─────────────────────────────────────────────
//
// Creates a single NCCL communicator that is reused across all all-reduce calls
// in a training loop. This avoids the overhead of creating/destroying a
// communicator (which involves file-system unique ID exchange) for every layer.

pub struct NcclPersistentComm {
    comm: NcclComm,
    rank: usize,
    world_size: usize,
    local_rank: usize,
    /// Dedicated NCCL stream — separate from compute stream for overlap.
    comm_stream: CudaStream,
}

/// CUDA event for stream synchronization without global blocking.
pub struct CudaEventHandle(pub CudaEvent);

impl Drop for CudaEventHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { cudaEventDestroy(self.0) };
        }
    }
}

impl NcclPersistentComm {
    /// Create a persistent NCCL communicator.
    /// Rank 0 generates the unique ID and writes it to a file; other ranks read it.
    pub fn new(output_dir: &Path) -> Result<Self> {
        let rank = parse_env_usize("RANK")?;
        let local_rank = parse_env_usize("LOCAL_RANK")?;
        let world_size = parse_env_usize("WORLD_SIZE")?;
        Self::new_group(output_dir, rank, world_size, local_rank)
    }

    /// Create a communicator for an explicit process subgroup.
    ///
    /// `rank` and `world_size` are relative to the subgroup, while `local_rank`
    /// identifies the CUDA device on this host. Concurrent subgroups must use
    /// distinct exchange directories so their NCCL unique IDs cannot collide.
    pub fn new_group(
        exchange_dir: &Path,
        rank: usize,
        world_size: usize,
        local_rank: usize,
    ) -> Result<Self> {
        validate_group_coordinates(rank, world_size, local_rank)?;
        fs::create_dir_all(exchange_dir)
            .with_context(|| format!("failed to create {}", exchange_dir.display()))?;

        let rank0_id = if rank == 0 {
            Some(nccl_unique_id()?)
        } else {
            None
        };
        let unique_id = exchange_persistent_unique_id(
            exchange_dir,
            rank,
            world_size,
            rank0_id,
            Duration::from_secs(300),
        )?;

        // Initialize communicator once
        check_cuda(
            unsafe { cudaSetDevice(local_rank as c_int) },
            "cudaSetDevice",
        )?;
        let mut comm: NcclComm = ptr::null_mut();
        check_nccl(
            unsafe { ncclCommInitRank(&mut comm, world_size as c_int, unique_id, rank as c_int) },
            "ncclCommInitRank",
        )?;

        // Create dedicated NCCL stream (separate from compute stream for overlap)
        let mut comm_stream: CudaStream = ptr::null_mut();
        check_cuda(
            unsafe { cudaStreamCreate(&mut comm_stream) },
            "cudaStreamCreate",
        )?;

        Ok(Self {
            comm,
            rank,
            world_size,
            local_rank,
            comm_stream,
        })
    }

    /// Get raw NCCL communicator pointer (for passing to C++ kernels).
    pub fn raw_comm_ptr(&self) -> *mut c_void {
        self.comm as *mut c_void
    }

    /// Get raw CUDA stream pointer (for passing to C++ kernels).
    pub fn raw_stream_ptr(&self) -> *mut c_void {
        self.comm_stream as *mut c_void
    }

    /// Get rank.
    pub fn rank(&self) -> usize { self.rank }

    /// Get world size.
    pub fn world_size(&self) -> usize { self.world_size }

    /// All-reduce a tensor (sum) using the persistent communicator.
    /// Returns the reduced tensor (sum across all ranks).
    /// Uses comm_stream — does NOT block CPU. Call `comm_sync()` to wait.
    pub fn all_reduce(&self, tensor: &Tensor) -> Result<Tensor> {
        let tensor = tensor.to_kind(Kind::Float).contiguous();
        if tensor.numel() == 0 {
            bail!("NCCL all-reduce input must not be empty");
        }
        check_cuda(
            unsafe { cudaSetDevice(self.local_rank as c_int) },
            "cudaSetDevice",
        )?;
        let output = tensor.zeros_like();
        check_nccl(
            unsafe {
                ncclAllReduce(
                    tensor.data_ptr().cast_const(),
                    output.data_ptr(),
                    tensor.numel(),
                    NCCL_FLOAT32,
                    NCCL_SUM,
                    self.comm,
                    self.comm_stream,
                )
            },
            "ncclAllReduce",
        )?;
        // Sync only comm_stream (not global device sync)
        check_cuda(
            unsafe { cudaStreamSynchronize(self.comm_stream) },
            "cudaStreamSynchronize",
        )?;
        Ok(output)
    }

    /// Async all-reduce: launches on comm_stream, returns (output, event).
    /// Does NOT block CPU. Caller should `cudaStreamWaitEvent(compute_stream, event)`
    /// before using the output tensor.
    pub fn all_reduce_async(&self, tensor: &Tensor) -> Result<(Tensor, CudaEventHandle)> {
        let tensor = tensor.to_kind(Kind::Float).contiguous();
        if tensor.numel() == 0 {
            bail!("NCCL all-reduce input must not be empty");
        }
        check_cuda(
            unsafe { cudaSetDevice(self.local_rank as c_int) },
            "cudaSetDevice",
        )?;
        let output = tensor.zeros_like();
        check_nccl(
            unsafe {
                ncclAllReduce(
                    tensor.data_ptr().cast_const(),
                    output.data_ptr(),
                    tensor.numel(),
                    NCCL_FLOAT32,
                    NCCL_SUM,
                    self.comm,
                    self.comm_stream,
                )
            },
            "ncclAllReduce",
        )?;
        // Record event on comm_stream — caller waits on this before using output
        let mut event: CudaEvent = ptr::null_mut();
        check_cuda(
            unsafe { cudaEventCreateWithFlags(&mut event, CUDA_EVENT_DISABLE_TIMING) },
            "cudaEventCreateWithFlags",
        )?;
        check_cuda(
            unsafe { cudaEventRecord(event, self.comm_stream) },
            "cudaEventRecord",
        )?;
        Ok((output, CudaEventHandle(event)))
    }

    /// Ring send/recv: send our tensor to `send_peer`, receive from `recv_peer`.
    /// Returns the received tensor. Both tensors must have the same shape/dtype.
    /// Uses ncclGroupStart/End for atomic exchange (no deadlock risk in ring).
    pub fn ring_send_recv(
        &self,
        send_tensor: &Tensor,
        send_peer: usize,
        recv_peer: usize,
    ) -> Result<Tensor> {
        let send_tensor = send_tensor.contiguous();
        if send_tensor.numel() == 0 {
            bail!("NCCL ring_send_recv: empty tensor");
        }
        check_cuda(
            unsafe { cudaSetDevice(self.local_rank as c_int) },
            "cudaSetDevice",
        )?;

        let recv_tensor = send_tensor.zeros_like();
        let count = send_tensor.numel();

        // Use BF16 directly for ring exchange — avoids F32 upcast overhead.
        // NCCL supports BF16 natively on H20 (sm90+).
        let dtype = match send_tensor.kind() {
            Kind::BFloat16 => NCCL_BF16,
            Kind::Float => NCCL_FLOAT32,
            _ => NCCL_FLOAT32, // fallback (convert to F32)
        };

        let send_ptr = if dtype == NCCL_FLOAT32 && send_tensor.kind() != Kind::Float {
            let f32_tensor = send_tensor.to_kind(Kind::Float);
            f32_tensor.data_ptr().cast_const()
        } else {
            send_tensor.data_ptr().cast_const()
        };

        // Group Start → Send + Recv → Group End (atomic exchange on comm_stream)
        check_nccl(unsafe { ncclGroupStart() }, "ncclGroupStart")?;
        check_nccl(
            unsafe {
                ncclSend(
                    send_ptr,
                    count,
                    dtype,
                    send_peer as c_int,
                    self.comm,
                    self.comm_stream,
                )
            },
            "ncclSend",
        )?;
        check_nccl(
            unsafe {
                ncclRecv(
                    recv_tensor.data_ptr(),
                    count,
                    dtype,
                    recv_peer as c_int,
                    self.comm,
                    self.comm_stream,
                )
            },
            "ncclRecv",
        )?;
        check_nccl(unsafe { ncclGroupEnd() }, "ncclGroupEnd")?;
        check_cuda(
            unsafe { cudaStreamSynchronize(self.comm_stream) },
            "cudaStreamSynchronize",
        )?;

        Ok(recv_tensor)
    }
}

fn exchange_persistent_unique_id(
    exchange_dir: &Path,
    rank: usize,
    world_size: usize,
    rank0_id: Option<NcclUniqueId>,
    timeout: Duration,
) -> Result<NcclUniqueId> {
    let epoch_path = exchange_dir.join("nccl-persistent-id.bin");
    let ack_path = |peer: usize| exchange_dir.join(format!("nccl-ack-rank-{peer}.bin"));
    let release_path = |peer: usize| exchange_dir.join(format!("nccl-release-rank-{peer}.bin"));
    let deadline = Instant::now() + timeout;

    if rank == 0 {
        let unique_id = rank0_id.context("rank 0 must provide a fresh NCCL unique ID")?;
        let epoch = unique_id_to_bytes(&unique_id);
        write_bytes_atomically(&epoch_path, &epoch)?;

        let mut peer_nonces = vec![None; world_size];
        while peer_nonces.iter().skip(1).any(Option::is_none) {
            for (peer, nonce) in peer_nonces.iter_mut().enumerate().skip(1) {
                if nonce.is_none()
                    && let Ok(bytes) = fs::read(ack_path(peer))
                    && bytes.len() > NCCL_UNIQUE_ID_BYTES
                    && bytes.starts_with(&epoch)
                {
                    *nonce = Some(bytes[NCCL_UNIQUE_ID_BYTES..].to_vec());
                }
            }
            if Instant::now() > deadline {
                let missing = peer_nonces
                    .iter()
                    .enumerate()
                    .skip(1)
                    .filter_map(|(peer, nonce)| nonce.is_none().then_some(peer))
                    .collect::<Vec<_>>();
                bail!(
                    "timed out waiting for NCCL epoch acknowledgements in {}: missing ranks {missing:?}",
                    exchange_dir.display(),
                );
            }
            sleep(Duration::from_millis(50));
        }

        for (peer, nonce) in peer_nonces.into_iter().enumerate().skip(1) {
            let mut release = epoch.clone();
            release.extend(nonce.context("missing NCCL participant nonce")?);
            write_bytes_atomically(&release_path(peer), &release)?;
        }
        return Ok(unique_id);
    }

    let nonce = participant_nonce(rank)?;
    loop {
        if let Ok(epoch) = fs::read(&epoch_path)
            && epoch.len() == NCCL_UNIQUE_ID_BYTES
        {
            let mut ack = epoch.clone();
            ack.extend(&nonce);
            write_bytes_atomically(&ack_path(rank), &ack)?;

            if let Ok(release) = fs::read(release_path(rank))
                && release.len() == NCCL_UNIQUE_ID_BYTES + nonce.len()
                && release.starts_with(&epoch)
                && release[NCCL_UNIQUE_ID_BYTES..] == nonce
            {
                return unique_id_from_bytes(&epoch);
            }
        }
        if Instant::now() > deadline {
            bail!(
                "timed out waiting for NCCL epoch release for rank {rank} in {}",
                exchange_dir.display()
            );
        }
        sleep(Duration::from_millis(50));
    }
}

fn participant_nonce(rank: usize) -> Result<Vec<u8>> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX epoch")?
        .as_nanos();
    Ok(format!("{}:{rank}:{timestamp}", std::process::id()).into_bytes())
}

fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;

    let temp_path = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = std::fs::File::create(&temp_path)
        .with_context(|| format!("failed to create {}", temp_path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("failed to write {}", temp_path.display()))?;
    file.sync_all()
        .with_context(|| format!("failed to sync {}", temp_path.display()))?;
    fs::rename(&temp_path, path).with_context(|| {
        format!(
            "failed to publish NCCL rendezvous file {} -> {}",
            temp_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

impl Drop for NcclPersistentComm {
    fn drop(&mut self) {
        if !self.comm.is_null() {
            unsafe { ncclCommDestroy(self.comm) };
        }
        if !self.comm_stream.is_null() {
            unsafe { cudaStreamDestroy(self.comm_stream) };
        }
    }
}

unsafe impl Send for NcclPersistentComm {}

fn validate_group_coordinates(rank: usize, world_size: usize, local_rank: usize) -> Result<()> {
    if world_size == 0 {
        bail!("NCCL group world_size must be greater than zero");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    if world_size > c_int::MAX as usize {
        bail!("NCCL group world_size {world_size} exceeds c_int::MAX");
    }
    if local_rank > c_int::MAX as usize {
        bail!("NCCL local_rank {local_rank} exceeds c_int::MAX");
    }
    Ok(())
}
