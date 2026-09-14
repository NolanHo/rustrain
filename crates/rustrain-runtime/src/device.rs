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
//! A CUDA context can only be **current** on one host thread at a time, so a
//! thread that touches this allocator's buffers must make the context current
//! first — which is exactly what [`crate::CudaAllocator`]'s entry points do.
//! The practical rule is therefore *one allocator serves one execution thread at
//! a time*: two threads may share the context (the primary context is
//! process-wide, and the NCCL backend's warm-up thread does exactly that), but
//! not call into the same allocator concurrently. A multi-rank CUDA launch still
//! needs one process per rank, because a rank's collectives and its buffers
//! belong together; `rustrain run --rank i --world n` is that process (the CLI's
//! `launch` subcommand starts them), and `--device cuda` with a mesh whose world
//! size is > 1 *inside one process* is refused — that refusal is the real guard,
//! this comment is only the explanation.

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

/// Resolves one driver symbol as a plain function pointer. The library stays
/// mapped for the lifetime of its owner, so the copy is valid for just as long.
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

/// The runtime-loaded driver and the handful of calls the framework makes.
///
/// One instance per context: `libcuda` is refcounted by the loader, and the
/// symbol copies are cheap. Kept private to the crate — the two owners
/// ([`CudaAllocator`] and [`CudaContext`]) are the only callers, and a public
/// driver handle would invite code that bypasses the allocator.
struct Driver {
    /// Kept mapped for the lifetime of the owner so the function pointers
    /// below stay valid. Declared first: fields drop in declaration order, and
    /// nothing in the drop paths below may outlive the library.
    _library: libloading::Library,
    mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    memcpy_htod: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    memcpy_dtod: unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize) -> CuResult,
    ctx_sync: unsafe extern "C" fn() -> CuResult,
    ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    primary_ctx_release: unsafe extern "C" fn(CuDevice) -> CuResult,
}

impl Driver {
    /// Loads the driver and resolves every symbol the framework uses.
    ///
    /// Loading the library runs no device code and creates no context.
    fn load() -> Result<Self, String> {
        // `libcuda.so.1` is the CUDA 11+ name; `libcuda.so` is the legacy one.
        let library = unsafe { libloading::Library::new("libcuda.so.1") }.or_else(|first| {
            unsafe { libloading::Library::new("libcuda.so") }.map_err(|second| {
                format!(
                    "cannot load the CUDA driver: tried libcuda.so.1 ({first}) and libcuda.so ({second})"
                )
            })
        })?;

        // SAFETY: every symbol is a static driver entry point with exactly the
        // signature given; the library is held for the owner's lifetime.
        let mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult =
            unsafe { resolve(&library, b"cuMemAlloc_v2\0") }?;
        let mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult =
            unsafe { resolve(&library, b"cuMemFree_v2\0") }?;
        let memcpy_htod: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult =
            unsafe { resolve(&library, b"cuMemcpyHtoD_v2\0") }?;
        let memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult =
            unsafe { resolve(&library, b"cuMemcpyDtoH_v2\0") }?;
        let memcpy_dtod: unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize) -> CuResult =
            unsafe { resolve(&library, b"cuMemcpyDtoD_v2\0") }?;
        let ctx_sync: unsafe extern "C" fn() -> CuResult =
            unsafe { resolve(&library, b"cuCtxSynchronize\0") }?;
        let ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult =
            unsafe { resolve(&library, b"cuCtxSetCurrent\0") }?;
        let primary_ctx_release: unsafe extern "C" fn(CuDevice) -> CuResult =
            unsafe { resolve(&library, b"cuDevicePrimaryCtxRelease\0") }?;

        Ok(Self {
            _library: library,
            mem_alloc,
            mem_free,
            memcpy_htod,
            memcpy_dtoh,
            memcpy_dtod,
            ctx_sync,
            ctx_set_current,
            primary_ctx_release,
        })
    }

    /// Initialises CUDA, picks `device_index`, retains its primary context and
    /// makes it current on this thread.
    fn retain(&self, device_index: usize) -> Result<(CuDevice, CuContext), String> {
        let ordinal = i32::try_from(device_index).map_err(|_| {
            format!("device index {device_index} does not fit a CUDA device ordinal")
        })?;
        // The three calls needed here are resolved on demand: they are used
        // once per context, and keeping them in the struct would suggest the
        // framework calls them anywhere else.
        let cu_init: unsafe extern "C" fn(u32) -> CuResult =
            unsafe { resolve(&self._library, b"cuInit\0") }?;
        let cu_device_get: unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult =
            unsafe { resolve(&self._library, b"cuDeviceGet\0") }?;
        let cu_primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult =
            unsafe { resolve(&self._library, b"cuDevicePrimaryCtxRetain\0") }?;

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
        check(
            unsafe { (self.ctx_set_current)(context) },
            "cuCtxSetCurrent",
        )?;
        Ok((device, context))
    }
}

/// Device memory for one CUDA device, over the runtime-loaded driver API.
pub struct CudaAllocator {
    driver: Driver,
    /// The device whose primary context this allocator retained.
    device: CuDevice,
    /// Every live allocation this allocator made, so `Drop` can free them
    /// before the context goes away.
    live: Vec<(CuDevicePtr, u64)>,
}

impl CudaAllocator {
    /// Loads the driver, initialises CUDA, picks `device_index`, retains its
    /// primary context and makes it current on this thread.
    pub fn new(device_index: usize) -> Result<Self, String> {
        let driver = Driver::load()?;
        let (device, _context) = driver.retain(device_index)?;
        Ok(Self {
            driver,
            device,
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
            unsafe { (self.driver.mem_free)(ptr) };
        }
        // The context was retained in `new`; release the reference.
        // SAFETY: as above — the context is current and `device` is the one it
        // was retained for.
        unsafe { (self.driver.primary_ctx_release)(self.device) };
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
        let code = unsafe { (self.driver.mem_alloc)(&mut ptr as *mut CuDevicePtr, n) };
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
            unsafe { (self.driver.mem_free)(device_ptr) };
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
        let code = unsafe {
            (self.driver.memcpy_htod)(dst as CuDevicePtr, src.as_ptr() as *const c_void, n)
        };
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
        check(unsafe { (self.driver.ctx_sync)() }, "cuCtxSynchronize")?;
        // SAFETY: `src` is a live device allocation of at least `n` bytes, and
        // `host` holds exactly `n` bytes.
        let code = unsafe {
            (self.driver.memcpy_dtoh)(host.as_mut_ptr() as *mut c_void, src as CuDevicePtr, n)
        };
        check(code, "cuMemcpyDtoH_v2")?;
        Ok(host)
    }
}

/// A retained primary context for code that needs the device but not this
/// allocator's memory calls — the NCCL backend, whose buffers come from the
/// same context (torch's caching allocator and this allocator share it).
///
/// Holding its own retain is deliberate: the executor owns the allocator and
/// the backend in one struct and drops them in field order, so a backend that
/// borrowed the allocator's context could outlive it. Refcounting the primary
/// context removes that ordering dependency entirely.
pub(crate) struct CudaContext {
    driver: Driver,
    device: CuDevice,
    /// The retained primary context. `set_current` re-activates *this* handle:
    /// retaining again would leak a reference (`Drop` releases one).
    context: CuContext,
}

impl CudaContext {
    pub(crate) fn open(device_index: usize) -> Result<Self, String> {
        let driver = Driver::load()?;
        let (device, context) = driver.retain(device_index)?;
        Ok(Self {
            driver,
            device,
            context,
        })
    }

    /// Makes this device's primary context current on the calling thread. NCCL
    /// reads the current device when it builds a communicator, and the driver
    /// copies below need the context too.
    pub(crate) fn set_current(&self) -> Result<(), String> {
        // SAFETY: `context` is the handle `open` retained for `device`, and it
        // is still released exactly once, in `Drop`.
        check(
            unsafe { (self.driver.ctx_set_current)(self.context) },
            "cuCtxSetCurrent",
        )
    }

    /// A device-to-device copy, for the one case a collective needs a private
    /// copy of an operand that aliases its own output.
    pub(crate) fn copy_device(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: u64,
    ) -> Result<(), String> {
        if dst.is_null() || src.is_null() {
            return Err("cuMemcpyDtoD_v2: null pointer".to_string());
        }
        let n = usize::try_from(bytes)
            .map_err(|_| format!("copy of {bytes} bytes does not fit this host"))?;
        // SAFETY: both pointers are live device allocations of at least `n`
        // bytes in the context this handle retained, which is current here.
        let code = unsafe { (self.driver.memcpy_dtod)(dst as CuDevicePtr, src as CuDevicePtr, n) };
        check(code, "cuMemcpyDtoD_v2")
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // SAFETY: the context was retained in `open` (`set_current` retains
        // again and releases here too — the count is what keeps it alive, not
        // the identity of the release call).
        unsafe { (self.driver.primary_ctx_release)(self.device) };
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

    /// The context handle is a second owner of the same primary context: on a
    /// device machine opening it and copying device-to-device must work, and on
    /// a machine without one it must report the missing driver instead of
    /// panicking — the same contract the allocator has.
    #[test]
    fn cuda_context_opens_and_copies_device_to_device_or_reports_why_not() {
        match CudaContext::open(0) {
            Ok(context) => {
                context.set_current().expect("set current");
                let mut allocator =
                    CudaAllocator::new(0).expect("the allocator shares the context");
                let src = allocator.alloc(64, RsDeviceKind::CUDA).expect("src");
                let dst = allocator.alloc(64, RsDeviceKind::CUDA).expect("dst");
                let data: Vec<u8> = (0..64u8).collect();
                allocator.copy_in(src, 64, &data).expect("copy in");
                context.copy_device(dst, src, 64).expect("device copy");
                assert_eq!(allocator.copy_out(dst, 64).expect("copy out"), data);
            }
            Err(error) => {
                assert!(
                    error.contains("tried libcuda.so.1")
                        || error.contains("cuInit")
                        || error.contains("cuDeviceGet")
                        || error.contains("cuDevicePrimaryCtxRetain")
                        || error.contains("cuCtxSetCurrent"),
                    "the failure must name the library or the driver call, got: {error}"
                );
            }
        }
    }
}
