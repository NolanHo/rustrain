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

#[derive(Debug, Serialize, Deserialize)]
struct NcclRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    input: f32,
    reduced: f32,
    expected: f32,
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
    let reduced =
        unsafe { nccl_all_reduce_scalar(unique_id, rank, world_size, local_rank, input)? };
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

unsafe fn nccl_all_reduce_scalar(
    unique_id: NcclUniqueId,
    rank: usize,
    world_size: usize,
    local_rank: usize,
    input: f32,
) -> Result<f32> {
    check_cuda(
        unsafe { cudaSetDevice(local_rank as c_int) },
        "cudaSetDevice",
    )?;

    let mut send: *mut c_void = ptr::null_mut();
    let mut recv: *mut c_void = ptr::null_mut();
    let bytes = std::mem::size_of::<f32>();
    check_cuda(unsafe { cudaMalloc(&mut send, bytes) }, "cudaMalloc(send)")?;
    check_cuda(unsafe { cudaMalloc(&mut recv, bytes) }, "cudaMalloc(recv)")?;

    let result = (|| {
        check_cuda(
            unsafe {
                cudaMemcpy(
                    send,
                    (&input as *const f32).cast::<c_void>(),
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
                    1,
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

        let mut output = 0.0_f32;
        check_cuda(
            unsafe {
                cudaMemcpy(
                    (&mut output as *mut f32).cast::<c_void>(),
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
}
