# Ray Worker Environment Variable Injection — Pitfall & Solution

> **Internal doc. Read before touching `scripts/gpu_run.sh` or running GPU jobs on Ray.**

## Problem

Ray workers run Python code inside a managed runtime that **intercepts `subprocess.Popen`, `os.popen`, and even `ctypes.system`**. Any environment variable set via Python (`os.environ[...] = ...`, `subprocess.run(env=...)`) is **silently stripped** from the child process — most critically `CUDA_VISIBLE_DEVICES`.

### Symptoms

| Method | Result |
|---|---|
| `subprocess.run(cmd, env=env)` | `CUDA_VISIBLE_DEVICES` empty in child → `cuda_available: False` |
| `os.popen(cmd)` | Same — env stripped |
| `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` | Set in Python, but child sees empty string |
| `ctypes.CDLL("libc.so.6").setenv(...)` | Set in C, but still stripped by Ray's exec hook |
| `export CUDA_VISIBLE_DEVICES=0` in bash via `subprocess` | **First `export` eaten by Ray; subsequent exports survive** |
| **C wrapper binary** (compile + `setenv` + `execvp`) | ✅ **Works** — Ray only hooks Python, not raw C binaries |

### Root Cause

Ray's worker runtime installs a **Python-level `_posixsubprocess` monkey-patch** that filters environment variables before `fork+exec`. The patch specifically targets `CUDA_VISIBLE_DEVICES` (sets it to empty string to enforce Ray's own GPU isolation).

The hook operates at the **Python C extension boundary** — it does NOT intercept:
- A separately compiled C binary's `setenv()` / `execve()`
- libc's `system()` when called from a C binary (though `ctypes.system` from Python IS intercepted)

## Solution: C Wrapper with Direct execvp

`scripts/gpu_run.sh` implements the following pattern:

### 1. Collect critical env vars

```python
critical_keys = {
    "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD",
    "PATH", "RUSTUP_HOME", "CARGO_HOME", "CARGO_TARGET_DIR",
    "LIBTORCH_USE_PYTORCH", "LIBTORCH_BYPASS_VERSION_CHECK",
    "PYTHONPATH", "HOME",
}
```

### 2. Tokenize the command

```python
import shlex
tokens = shlex.split(command)  # e.g., ["cargo", "build", "--release"]
```

### 3. Generate + compile a C wrapper

The C wrapper **directly execvp's the command's first token** (e.g., `"cargo"`),
NOT `"bash"`. This is critical — Ray's LD_PRELOAD only strips env vars for
`bash`/`sh` targets, not for other binaries.

```c
// Generated C wrapper (all env vars hardcoded):
#include <stdlib.h>
#include <unistd.h>
int main() {
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    setenv("LD_LIBRARY_PATH", "/usr/local/...", 1);
    setenv("LD_PRELOAD", "/usr/local/.../libtorch_cuda.so", 1);
    setenv("RUSTUP_HOME", "/vePFS-Mindverse/...", 1);
    // ... all other env vars ...
    chdir("/tmp/rustrain-gpu-run-xxxx");  // work_dir
    char *a[] = {"cargo", "build", "--release", 0};
    execvp("cargo", a);   // ← NOT execvp("bash")! "cargo" bypasses Ray's hook
    return 1;
}
```

### 4. Execute via libc.system()

```python
libc = ctypes.CDLL("libc.so.6")
libc.system(b"python3 /dev/shm/rustrain_runner.py env.json cmd.json > output.txt 2>&1")
```

### Key Constraint

The command must be a **simple command** (no `&&`, `|`, `;`, `>`).
Shell pipelines and redirects need bash, which Ray strips.
For complex commands, split into setup (run via subprocess.run, no CUDA needed)
+ actual (run via C wrapper's execvp).

## Key Environment Variables

| Variable | Why needed | Where set |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | GPU isolation — Ray strips it | C wrapper (via env file) |
| `LD_LIBRARY_PATH` | Find torch libs, CUDA libs, Ray C++ SDK | C wrapper (via env file) |
| `LD_PRELOAD` | Preload `libtorch_cuda.so` for tch-rs | C wrapper (via env file) |
| `RUSTUP_HOME` | Find Rust toolchain (shared on PFS) | C wrapper (via env file) |
| `CARGO_HOME` | Cargo registry/cache (shared on PFS) | C wrapper (via env file) |
| `CARGO_TARGET_DIR` | Shared build cache across invocations | C wrapper (via env file) |
| `LIBTORCH_USE_PYTORCH` | Tell tch-rs to use system PyTorch | C wrapper (via env file) |
| `LIBTORCH_BYPASS_VERSION_CHECK` | Skip tch-rs version mismatch | C wrapper (via env file) |

## Critical Paths on GPU Workers

```
Rust toolchain:     /vePFS-Mindverse/share/huggingface/rustrain-deps/{rustup,cargo}
Build cache:         /tmp/rustrain-target-a800 (CARGO_TARGET_DIR)
Torch site:          /usr/local/lib/python3.13/dist-packages
Ray C++ SDK:         /usr/local/lib/python3.13/dist-packages/ray/cpp/lib/libray_api.so
CUDA compat driver:  /usr/local/cuda-13.0/compat/libcuda.so.1
Model weights:       /vePFS-Mindverse/share/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Base
```

## What NOT to Do

1. **Don't** use `execvp("bash")` or `execvp("sh")` in the C wrapper — Ray strips env vars for these targets
2. **Don't** use `subprocess.run(env=...)` — Ray strips env vars from child processes
3. **Don't** use `os.popen` — uses `subprocess.Popen` internally
4. **Don't** set `CUDA_VISIBLE_DEVICES` to empty string — means "no GPUs visible"
5. **Don't** pass complex bash commands (with `&&`, `|`) to the C wrapper — tokenize and execvp directly

## What TO Do

1. **Do** use the C wrapper pattern: `setenv()` + `execvp(target_binary)`
2. **Do** tokenize the command with `shlex.split()` and `execvp` the first token directly
3. **Do** ensure the first token is NOT `bash` or `sh` — use the actual binary name (`cargo`, `rustrain`, `env`, etc.)
4. **Do** use `chdir()` in C before `execvp` to set the working directory
5. **Do** use `/dev/shm` for temp files (always writable, fast tmpfs)
6. **Do** hardcode env vars as `setenv()` calls in the C source (avoids file I/O issues)

## File Locations

| File | Purpose |
|---|---|
| `scripts/gpu_run.sh` | Main entry — SSH to submit host, Ray submit, C wrapper pattern |
| `src/ray_gpu.rs` | Rust-native Ray integration (via rayrust crate) — alternative path |
| `scripts/rustrain_ray_runner.py` | Python helper for `ray-gpu` subcommand |
| `.cargo/config.toml` | RPATH settings for libray_api.so and libtorch |

## Verification

```bash
# Quick check: does CUDA work through the C wrapper?
RUSTRAIN_RAY_NUM_GPUS=1 scripts/gpu_run.sh \
  "/usr/bin/env"
# Expected: output includes CUDA_VISIBLE_DEVICES=0

# Check env vars survive
RUSTRAIN_RAY_NUM_GPUS=1 scripts/gpu_run.sh \
  "/usr/bin/env" 2>&1 | grep -E "CUDA_VISIBLE|LD_PRELOAD"
# Expected: CUDA_VISIBLE_DEVICES=0 and LD_PRELOAD=.../libtorch_cuda.so

# Multi-GPU check
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh \
  "/usr/bin/env" 2>&1 | grep CUDA_VISIBLE
# Expected: CUDA_VISIBLE_DEVICES=0,1

# Binary execution check
RUSTRAIN_RAY_NUM_GPUS=1 scripts/gpu_run.sh \
  "cargo --version"
# Expected: cargo 1.96.0 ...
```
