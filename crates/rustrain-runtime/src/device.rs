//! CUDA device memory through the driver API, loaded at runtime.
//!
//! The core crates must not link CUDA (invariant I-1): this module loads
//! `libcuda.so.1` (falling back to `libcuda.so`) with `libloading` and
//! resolves the driver API by symbol, so a machine with no CUDA at all still
//! compiles and runs every test — `CudaAllocator::new` is where a missing
//! driver becomes a *reported* error, never a link failure.
//!
//! The allocator retains the device's **primary context**, which is what makes
//! libtorch — which also uses the primary context — share it with us: the
//! buffers this allocator hands out are the same device memory torch kernels
//! already read and write.
//!
//! # Threading limitation
//!
//! A CUDA context can only be **current** on one host thread at a time. This
//! allocator sets the primary context current on the thread that constructs it
//! and uses it from the same thread, so **one allocator (one context) serves
//! one execution thread**. A multi-rank CUDA launch therefore needs one
//! process per rank; the CLI refuses `--device cuda` with a mesh whose world
//! size is > 1 up front (`crates/rustrain-cli/src/run.rs`) — that refusal is
//! the real guard, this comment is only the explanation.

use std::ffi::c_void;

use rustrain_abi::ffi::RsDeviceKind;

use crate::Allocator;

// Driver API types, spelled as the CUDA 12 header spells them.
type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuDevicePtr = u64;

const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_NO_DEVICE: CuResult = 100;
const CUDA_ERROR_NOT_INITIALIZED: CuResult = 201;

/// Names the driver status codes the framework reasons about. Every other code
/// is reported as its raw number — never guessed into a name that might be
/// wrong.
fn cu_result_name(code: CuResult) -> String {
    match code {
        CUDA_SUCCESS => "CUDA_SUCCESS".to_string(),
        CUDA_ERROR_NO_DEVICE => "CUDA_ERROR_NO_DEVICE".to_string(),
        CUDA_ERROR_NOT_INITIALIZED => "CUDA_ERROR_NOT_INITIALIZED".to_string(),
        other => format!("CUresult({other})"),
    }
}

/// The one gate every driver call passes through: a non-success status is an
/// error naming the call and its code.
fn check(code: CuResult, call: &str) -> Result<(), String> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{call} returned {} ({code})", cu_result_name(code)))
    }
}

/// Device memory for one CUDA device, over the runtime-loaded driver API.
pub struct CudaAllocator {
    /// Kept mapped for the lifetime of the allocator so the function pointers
    /// below stay valid. Declared first: fields drop in declaration order, and
    /// nothing in the drop paths below may outlive the library.
    _library: libloading::Library,
    /// The device whose primary context this allocator retained.
    device: CuDevice,
    mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    memcpy_htod: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    ctx_sync: unsafe extern "C" fn() -> CuResult,
    primary_ctx_release: unsafe extern "C" fn(CuDevice) -> CuResult,
    /// Every live allocation this allocator made, so `Drop` can free them
    /// before the context goes away.
    live: Vec<(CuDevicePtr, u64)>,
}

/// Resolves one driver symbol as a plain function pointer. The library stays
/// mapped for the lifetime of the allocator, so the copy is valid for just as
/// long.
///
/// # Safety
/// `name` must be a static, NUL-terminated symbol name.
unsafe fn resolve<T: Copy>(library: &libloading::Library, name: &[u8]) -> Result<T, String> {
    // SAFETY: the caller guarantees a NUL-terminated name; the symbol, if
    // present, is a function pointer of exactly the type `T` the caller asks
    // for.
    let symbol: libloading::Symbol<T> = unsafe { library.get(name) }
        .map_err(|e| format!("resolving {}: {e}", String::from_utf8_lossy(name)))?;
    Ok(*symbol)
}

impl CudaAllocator {
    /// Loads the driver, initialises CUDA, picks `device_index`, retains its
    /// primary context and makes it current on this thread.
    pub fn new(device_index: usize) -> Result<Self, String> {
        let ordinal = i32::try_from(device_index).map_err(|_| {
            format!("device index {device_index} does not fit a CUDA device ordinal")
        })?;

        // `libcuda.so.1` is the CUDA 11+ name; `libcuda.so` is the legacy one.
        // Loading the library runs no device code and creates no context.
        let library = unsafe { libloading::Library::new("libcuda.so.1") }.or_else(|first| {
            unsafe { libloading::Library::new("libcuda.so") }.map_err(|second| {
                format!(
                    "cannot load the CUDA driver: tried libcuda.so.1 ({first}) and libcuda.so ({second})"
                )
            })
        })?;

        // SAFETY: every symbol is a static driver entry point with exactly the
        // signature given; the library is held for the allocator's lifetime.
        let cu_init: unsafe extern "C" fn(u32) -> CuResult =
            unsafe { resolve(&library, b"cuInit\0") }?;
        let cu_device_get: unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult =
            unsafe { resolve(&library, b"cuDeviceGet\0") }?;
        let cu_primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult =
            unsafe { resolve(&library, b"cuDevicePrimaryCtxRetain\0") }?;
        let cu_ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult =
            unsafe { resolve(&library, b"cuCtxSetCurrent\0") }?;
        let mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult =
            unsafe { resolve(&library, b"cuMemAlloc_v2\0") }?;
        let mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult =
            unsafe { resolve(&library, b"cuMemFree_v2\0") }?;
        let memcpy_htod: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult =
            unsafe { resolve(&library, b"cuMemcpyHtoD_v2\0") }?;
        let memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult =
            unsafe { resolve(&library, b"cuMemcpyDtoH_v2\0") }?;
        let ctx_sync: unsafe extern "C" fn() -> CuResult =
            unsafe { resolve(&library, b"cuCtxSynchronize\0") }?;
        // `Drop` releases the retained primary context; the frozen symbol list
        // in the D6-GPU brief does not name this call, but "releases the
        // context" cannot be done without it.
        let primary_ctx_release: unsafe extern "C" fn(CuDevice) -> CuResult =
            unsafe { resolve(&library, b"cuDevicePrimaryCtxRelease\0") }?;

        // SAFETY: the calls below touch only the driver's own state; the
        // parameters point at locals that outlive each call.
        check(unsafe { cu_init(0) }, "cuInit")?;
        let mut device: CuDevice = 0;
        check(
            unsafe { cu_device_get(&mut device, ordinal) },
            "cuDeviceGet",
        )?;
        let mut context: CuContext = std::ptr::null_mut();
        check(
            unsafe { cu_primary_ctx_retain(&mut context, device) },
            "cuDevicePrimaryCtxRetain",
        )?;
        // Retaining is what makes libtorch — a user of the same primary
        // context — share it with us; setting it current is what makes every
        // later driver call on this thread use it.
        check(unsafe { cu_ctx_set_current(context) }, "cuCtxSetCurrent")?;

        Ok(Self {
            _library: library,
            device,
            mem_alloc,
            mem_free,
            memcpy_htod,
            memcpy_dtoh,
            ctx_sync,
            primary_ctx_release,
            live: Vec::new(),
        })
    }
}

impl Drop for CudaAllocator {
    fn drop(&mut self) {
        // Every allocation this allocator made is freed while the context is
        // still live (this thread is the only user of the context — see the
        // module-level threading note).
        for (ptr, _) in self.live.drain(..) {
            // SAFETY: `ptr` came from `cuMemAlloc_v2` on the context this
            // allocator set current, which is still current here.
            unsafe { (self.mem_free)(ptr) };
        }
        // The context was retained in `new`; release the reference.
        // SAFETY: as above — the context is current and `device` is the one it
        // was retained for.
        unsafe { (self.primary_ctx_release)(self.device) };
    }
}

// SAFETY: `alloc` returns a device pointer from `cuMemAlloc_v2`, valid until
// `dealloc` frees it with `cuMemFree_v2`; every pointer handed out is tracked
// in `live` so `Drop` frees it before the context is released.
unsafe impl Allocator for CudaAllocator {
    fn alloc(&mut self, bytes: u64, _device: RsDeviceKind) -> Result<*mut c_void, String> {
        let bytes = bytes.max(1);
        let n = usize::try_from(bytes)
            .map_err(|_| format!("allocation of {bytes} bytes does not fit this host"))?;
        let mut ptr: CuDevicePtr = 0;
        // SAFETY: the context is current on this thread (set in `new`), and
        // `ptr` is written by the driver with the allocation's address.
        let code = unsafe { (self.mem_alloc)(&mut ptr as *mut CuDevicePtr, n) };
        check(code, "cuMemAlloc_v2")?;
        self.live.push((ptr, bytes));
        Ok(ptr as *mut c_void)
    }

    fn dealloc(&mut self, ptr: *mut c_void, _bytes: u64) {
        if ptr.is_null() {
            return;
        }
        if let Some(index) = self.live.iter().position(|(p, _)| *p == ptr as CuDevicePtr) {
            let (device_ptr, _) = self.live.swap_remove(index);
            // SAFETY: `device_ptr` was allocated by this allocator and not yet
            // freed.
            unsafe { (self.mem_free)(device_ptr) };
        }
    }

    fn device(&self) -> RsDeviceKind {
        RsDeviceKind::CUDA
    }

    fn copy_in(&mut self, dst: *mut c_void, bytes: u64, src: &[u8]) -> Result<(), String> {
        if dst.is_null() {
            return Err("cuMemcpyHtoD_v2: null destination".to_string());
        }
        let n = usize::try_from(bytes)
            .map_err(|_| format!("copy of {bytes} bytes does not fit this host"))?;
        if n != src.len() {
            return Err(format!(
                "cuMemcpyHtoD_v2: {bytes} byte(s) requested but the slice holds {}",
                src.len()
            ));
        }
        // SAFETY: the destination is a live device allocation of at least `n`
        // bytes, and the source slice holds exactly `n` bytes.
        let code =
            unsafe { (self.memcpy_htod)(dst as CuDevicePtr, src.as_ptr() as *const c_void, n) };
        check(code, "cuMemcpyHtoD_v2")
    }

    fn copy_out(&self, src: *const c_void, bytes: u64) -> Result<Vec<u8>, String> {
        if src.is_null() {
            return Err("cuMemcpyDtoH_v2: null source".to_string());
        }
        let n = usize::try_from(bytes)
            .map_err(|_| format!("copy of {bytes} bytes does not fit this host"))?;
        let mut host = vec![0u8; n];
        // The bytes were written by kernels launched on torch's stream, and a
        // driver copy alone does not order against them — synchronise the
        // context first so the copy reads the finished data.
        // SAFETY: the context is current on this thread.
        check(unsafe { (self.ctx_sync)() }, "cuCtxSynchronize")?;
        // SAFETY: `src` is a live device allocation of at least `n` bytes, and
        // `host` holds exactly `n` bytes.
        let code =
            unsafe { (self.memcpy_dtoh)(host.as_mut_ptr() as *mut c_void, src as CuDevicePtr, n) };
        check(code, "cuMemcpyDtoH_v2")?;
        Ok(host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three codes the framework reasons about get their names; every
    /// other code keeps its raw number.
    #[test]
    fn driver_status_names_the_codes_the_framework_reads() {
        assert_eq!(cu_result_name(CUDA_SUCCESS), "CUDA_SUCCESS");
        assert_eq!(cu_result_name(CUDA_ERROR_NO_DEVICE), "CUDA_ERROR_NO_DEVICE");
        assert_eq!(
            cu_result_name(CUDA_ERROR_NOT_INITIALIZED),
            "CUDA_ERROR_NOT_INITIALIZED"
        );
        assert!(
            cu_result_name(999).contains("999"),
            "unknown codes keep their raw number, got {}",
            cu_result_name(999)
        );
    }

    /// On a machine with no CUDA (like the edit box) `new` must fail with an
    /// error naming the library it could not load or the failing driver call
    /// and its code — never a panic. On a machine with a real device, the
    /// same call succeeds and the honest device round trip must hold, so the
    /// test never silently passes on a GPU-less machine while pretending to
    /// have exercised allocation.
    #[test]
    fn cuda_allocator_names_its_failure_without_a_device_or_round_trips_with_one() {
        match CudaAllocator::new(0) {
            Ok(mut allocator) => {
                // A real device: exercise the copy path for real. The kernel
                // bytes survive a device round trip and the allocator reports
                // its device.
                assert_eq!(allocator.device(), RsDeviceKind::CUDA);
                let ptr = allocator
                    .alloc(1024, RsDeviceKind::CUDA)
                    .expect("allocating 1024 device bytes");
                let data: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
                allocator.copy_in(ptr, 1024, &data).expect("copy in");
                assert_eq!(allocator.copy_out(ptr, 1024).expect("copy out"), data);
                allocator.dealloc(ptr, 1024);
            }
            Err(error) => {
                assert!(
                    error.contains("tried libcuda.so.1")
                        || error.contains("cuInit")
                        || error.contains("cuDeviceGet")
                        || error.contains("cuDevicePrimaryCtxRetain")
                        || error.contains("cuCtxSetCurrent"),
                    "the failure must name the library it could not load or the driver call and \
                     its code, got: {error}"
                );
            }
        }
    }
}
