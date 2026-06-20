use std::{
    ffi::{CStr, c_char, c_int, c_void},
    fs,
    path::{Path, PathBuf},
    ptr,
    thread::sleep,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
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

const NCCL_FLOAT32: NcclDataType = 7;
const NCCL_SUM: NcclRedOp = 0;

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

fn shared_unique_id(output_dir: &Path, rank: usize) -> Result<NcclUniqueId> {
    let id_path = output_dir.join("nccl-unique-id.bin");
    if rank == 0 {
        let id = nccl_unique_id()?;
        fs::write(&id_path, unique_id_to_bytes(&id))
            .with_context(|| format!("failed to write {}", id_path.display()))?;
        Ok(id)
    } else {
        wait_for_unique_id(&id_path, Duration::from_secs(30))
    }
}

fn wait_for_unique_id(path: &Path, timeout: Duration) -> Result<NcclUniqueId> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if path.exists() {
            let bytes =
                fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
            return unique_id_from_bytes(&bytes);
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
}
