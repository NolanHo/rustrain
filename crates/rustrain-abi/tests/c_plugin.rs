//! End-to-end tests for the plugin ABI: real C plugins are compiled by the
//! system C compiler from `tests/cdata/`, then loaded, enumerated and called
//! across the language boundary.
//!
//! The fixtures are built from inside the test rather than from a `build.rs` so
//! that `cargo test -p rustrain-abi` is the entire verification: no GPU, no
//! torch, and nothing to remember to run first.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;

use rustrain_abi::{ABI_VERSION, AbiError, Plugin, RsDtype, RsServices, RsTensor};

/// Compiles the C fixtures from `tests/cdata/` with the system C compiler.
///
/// The plugin is built from inside the test rather than from a `build.rs` so
/// that `cargo test -p rustrain-abi` is the entire verification: no GPU, no
/// torch, and nothing to remember to run first.
fn cdata_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cdata")
}

fn include_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("include")
}

/// `cc` is the portable name; `gcc` is the fallback when only that exists.
fn compiler() -> &'static str {
    for candidate in ["cc", "gcc"] {
        if let Ok(out) = Command::new(candidate).arg("--version").output()
            && out.status.success()
        {
            return candidate;
        }
    }
    panic!("neither `cc` nor `gcc` is on PATH to build the ABI test fixture");
}

/// A compiled fixture. The `.so` lives in the temporary directory, so the
/// directory has to outlive every plugin loaded from it.
struct Fixture {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl Fixture {
    fn compile(source: &str, defines: &[&str]) -> Self {
        let dir = tempfile::tempdir().expect("create temp dir for the C fixture");
        let path = dir.path().join("libfixture.so");
        let source_path = cdata_dir().join(source);

        let mut cmd = Command::new(compiler());
        cmd.args(["-shared", "-fPIC", "-std=c11", "-O1", "-Wall", "-Wextra"])
            .arg("-I")
            .arg(include_dir());
        for define in defines {
            cmd.arg(format!("-D{define}"));
        }
        let output = cmd
            .arg("-o")
            .arg(&path)
            .arg(&source_path)
            .output()
            .unwrap_or_else(|e| panic!("cannot run {}: {e}", compiler()));

        assert!(
            output.status.success(),
            "compiling {source} failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        // The fixtures are the reference for how a plugin is written, so they
        // are held to a warning-free build rather than merely a successful one.
        assert!(
            output.stderr.is_empty(),
            "{source} compiled with warnings:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self { _dir: dir, path }
    }

    fn load(&self, services: Option<&RsServices>) -> Result<Plugin, AbiError> {
        // SAFETY: the service table, when present, outlives the plugin in every
        // caller below.
        unsafe { Plugin::load(&self.path, services) }
    }

    /// Loads and asserts rejection, returning the error for inspection.
    fn load_error(&self) -> AbiError {
        match self.load(None) {
            Ok(plugin) => panic!(
                "{} loaded as plugin `{}` but should have been rejected",
                self.path.display(),
                plugin.name()
            ),
            Err(err) => err,
        }
    }
}

/// Describes an existing f32 buffer. The plugin is handed a descriptor, not a
/// copy: the buffer stays owned by the caller.
fn f32_tensor(shape: &[i64], data: *mut f32) -> RsTensor {
    let mut t = RsTensor::new(RsDtype::F32, shape);
    t.data = data as *mut c_void;
    t
}

/// A service table with no callbacks: the fixtures only inspect the header.
fn noop_services() -> RsServices {
    RsServices {
        abi_version: ABI_VERSION,
        struct_size: std::mem::size_of::<RsServices>() as u32,
        user: std::ptr::null_mut(),
        alloc: None,
        free: None,
        current_stream: None,
        collective: None,
        log: None,
    }
}

// ── (a) the happy path ────────────────────────────────────────────────────

#[test]
fn c_plugin_loads_enumerates_and_executes() {
    let fixture = Fixture::compile("plugin_add.c", &[]);
    let plugin = fixture.load(None).expect("plugin_add.c must load");

    // Strings and counts written by C, read back through the Rust mirror.
    assert_eq!(plugin.name(), "cdata_add");
    assert_eq!(plugin.version(), "0.1.0");
    assert_eq!(plugin.identity(), "cdata_add@0.1.0");
    assert_eq!(plugin.origin(), fixture.path.as_path());

    let ops = plugin.ops();
    assert_eq!(ops.len(), 1, "the plugin declares exactly one op");
    let op = &ops[0];
    assert_eq!(op.spec_name(), "add@c");
    assert_eq!(op.name(), "add");
    assert_eq!(op.variant(), "c");
    assert_eq!(op.doc(), "elementwise sum of two contiguous f32 buffers");
    assert_eq!(op.plugin_identity(), "cdata_add@0.1.0");

    let requires = op.requires().expect("the C plugin declares requires");
    assert!(requires.accepts(RsDtype::F32));
    assert_eq!(requires.dtypes(), vec![RsDtype::F32]);
    assert!(!requires.accepts(RsDtype::BF16));
    assert_eq!(requires.min_world_size, 0);
    assert!(op.collectives().is_empty());
    assert!(op.expansion().is_none());

    // Element buffers owned by Rust, read and written by C.
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let mut out = [0.0f32; 4];
    // The plugin reads the inputs through `const float*`, so handing it a mut
    // pointer to a shared buffer is only a cast, not a second owner.
    let ta = f32_tensor(&[4], a.as_ptr() as *mut f32);
    let tb = f32_tensor(&[4], b.as_ptr() as *mut f32);
    let mut to = f32_tensor(&[4], out.as_mut_ptr());

    // SAFETY: three contiguous f32 [4] buffers; `add` reads two and writes one.
    unsafe {
        op.execute(
            std::ptr::null_mut(),
            &[&ta as *const RsTensor, &tb as *const RsTensor],
            &[&mut to as *mut RsTensor],
            std::ptr::null(),
        )
    }
    .expect("execute add");

    assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
}

/// A handle must keep the library mapped: the descriptor it points at lives in
/// the `.so`, and the plugin handle itself is dropped here before the call.
#[test]
fn loaded_op_outlives_the_plugin_handle() {
    let fixture = Fixture::compile("plugin_add.c", &[]);
    let op = {
        let plugin = fixture.load(None).expect("plugin_add.c must load");
        plugin.ops().remove(0)
    };

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let mut out = [0.0f32; 4];
    let ta = f32_tensor(&[4], a.as_ptr() as *mut f32);
    let tb = f32_tensor(&[4], b.as_ptr() as *mut f32);
    let mut to = f32_tensor(&[4], out.as_mut_ptr());

    // SAFETY: as in the happy path.
    unsafe {
        op.execute(
            std::ptr::null_mut(),
            &[&ta as *const RsTensor, &tb as *const RsTensor],
            &[&mut to as *mut RsTensor],
            std::ptr::null(),
        )
    }
    .expect("the op's own handle must keep the library mapped");

    assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn c_plugin_execute_failure_carries_the_plugins_own_message() {
    let fixture = Fixture::compile("plugin_add.c", &[]);
    let plugin = fixture.load(None).expect("plugin_add.c must load");
    let op = &plugin.ops()[0];

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [1.0f32, 2.0, 3.0];
    let mut out = [0.0f32; 4];
    let ta = f32_tensor(&[4], a.as_ptr() as *mut f32);
    let tb = f32_tensor(&[3], b.as_ptr() as *mut f32);
    let mut to = f32_tensor(&[4], out.as_mut_ptr());

    // SAFETY: as above; the fixture rejects the mismatched shapes itself.
    let err = unsafe {
        op.execute(
            std::ptr::null_mut(),
            &[&ta as *const RsTensor, &tb as *const RsTensor],
            &[&mut to as *mut RsTensor],
            std::ptr::null(),
        )
    }
    .expect_err("shape mismatch must fail");

    match err {
        AbiError::ExecuteFailed {
            status, message, ..
        } => {
            assert_eq!(status, 5);
            assert_eq!(message, "add: input shapes differ");
        }
        other => panic!("expected ExecuteFailed, got {other:?}"),
    }
    assert_eq!(out, [0.0; 4], "a rejected call must not write the output");
}

// ── (c) version negotiation ───────────────────────────────────────────────

#[test]
fn c_plugin_with_wrong_abi_version_is_rejected() {
    let fixture = Fixture::compile("plugin_bad_version.c", &[]);
    match fixture.load_error() {
        AbiError::VersionMismatch {
            found, expected, ..
        } => {
            assert_eq!(found, 999);
            assert_eq!(expected, ABI_VERSION);
        }
        other => panic!("expected VersionMismatch, got {other:?}"),
    }
}

// ── (d) missing entry symbol ──────────────────────────────────────────────

#[test]
fn c_plugin_without_entry_symbol_is_rejected() {
    let fixture = Fixture::compile("plugin_nothing.c", &[]);
    let err = fixture.load_error();
    assert!(
        matches!(&err, AbiError::MissingSymbol { .. }),
        "expected MissingSymbol, got {err:?}"
    );
    assert!(err.to_string().contains("rustrain_plugin_v1"), "{err}");
}

// ── (e) malformed descriptors ─────────────────────────────────────────────

#[test]
fn c_plugin_op_without_execute_is_rejected() {
    let fixture = Fixture::compile("plugin_no_execute.c", &[]);
    match fixture.load_error() {
        AbiError::OpWithoutExecute { op, variant, .. } => {
            assert_eq!(op, "add");
            assert_eq!(variant, "c");
        }
        other => panic!("expected OpWithoutExecute, got {other:?}"),
    }
}

#[test]
fn c_plugin_with_a_null_op_slot_is_rejected() {
    let fixture = Fixture::compile("plugin_malformed.c", &["NULL_OP_SLOT"]);
    match fixture.load_error() {
        // Index 1, not 0: the slot before it is a real op and must not be
        // renumbered by dropping the hole.
        AbiError::NullOp { index, .. } => assert_eq!(index, 1),
        other => panic!("expected NullOp, got {other:?}"),
    }
}

#[test]
fn c_plugin_with_a_null_op_table_is_rejected() {
    let fixture = Fixture::compile("plugin_malformed.c", &["NULL_OP_TABLE"]);
    match fixture.load_error() {
        AbiError::NullOpTable { count, .. } => assert_eq!(count, 2),
        other => panic!("expected NullOpTable, got {other:?}"),
    }
}

// ── services (contract C-3) ───────────────────────────────────────────────

#[test]
fn c_plugin_init_receives_the_service_table() {
    let fixture = Fixture::compile("plugin_init.c", &[]);
    let services = noop_services();
    let plugin = fixture
        .load(Some(&services))
        .expect("plugin_init.c must load");
    let op = &plugin.ops()[0];

    // SAFETY: the op ignores its arguments and only reports what init() saw.
    let result = unsafe { op.execute(std::ptr::null_mut(), &[], &[], std::ptr::null()) };
    result.expect("init() must have received a table with this build's ABI version");
}

#[test]
fn c_plugin_without_services_gets_a_null_table() {
    let fixture = Fixture::compile("plugin_init.c", &[]);
    let plugin = fixture.load(None).expect("plugin_init.c must load");
    let op = &plugin.ops()[0];

    // SAFETY: as above.
    let err = unsafe { op.execute(std::ptr::null_mut(), &[], &[], std::ptr::null()) }
        .expect_err("no service table was supplied");

    match err {
        AbiError::ExecuteFailed { status, .. } => assert_eq!(status, 10),
        other => panic!("expected ExecuteFailed, got {other:?}"),
    }
}

#[test]
fn c_plugin_init_failure_is_a_load_error() {
    let fixture = Fixture::compile("plugin_init.c", &["INIT_STATUS=42"]);
    let services = noop_services();
    match fixture.load(Some(&services)) {
        Ok(plugin) => panic!("init() returned 42 yet `{}` loaded", plugin.name()),
        Err(AbiError::InitFailed { name, status }) => {
            assert_eq!(name, "cdata_init");
            assert_eq!(status, 42);
        }
        Err(other) => panic!("expected InitFailed, got {other:?}"),
    }
}
