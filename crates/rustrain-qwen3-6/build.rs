// build.rs — Compile C++ Qwen3.6 kernels and link against libtorch.
use std::process::Command;

/// Detect PyTorch's _GLIBCXX_USE_CXX11_ABI setting by running Python.
/// Returns "1" or "0". Defaults to "1" if detection fails.
fn detect_cxx11_abi() -> String {
    // Check environment override first
    if let Ok(v) = std::env::var("GLIBCXX_USE_CXX11_ABI") {
        return v;
    }
    // Try python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))'
    for py in &["python3", "python"] {
        if let Ok(out) = std::process::Command::new(py)
            .args([
                "-c",
                "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))",
            ])
            .output()
        {
            if out.status.success() {
                if let Ok(s) = String::from_utf8(out.stdout) {
                    let s = s.trim();
                    if s == "0" || s == "1" {
                        return s.to_string();
                    }
                }
            }
        }
    }
    "1".to_string()
}

fn main() {
    println!("cargo:rerun-if-env-changed=TORCH_INCLUDE_PATH");
    println!("cargo:rerun-if-env-changed=TORCH_LIB_PATH");
    println!("cargo:rerun-if-env-changed=NCCL_INCLUDE_PATH");
    println!("cargo:rerun-if-env-changed=NCCL_LIB_PATH");
    println!("cargo:rerun-if-changed=kernels/qwen3_6_kernels.cpp");
    println!("cargo:rerun-if-changed=kernels/delta_rule.cu");
    println!("cargo:rerun-if-changed=kernels/delta_rule.cuh");
    println!("cargo:rerun-if-changed=kernels/fused_kernels.cu");
    println!("cargo:rerun-if-changed=build.rs");

    let cxx11_abi = detect_cxx11_abi();
    let cxx11_flag = format!("-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}");
    println!("cargo:warning=CXX11 ABI detected: {cxx11_abi}");

    let torch_include = std::env::var("TORCH_INCLUDE_PATH").or_else(|_| {
        let candidates = [
            "/share/code/nolanho/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/include",
            "/vePFS-Mindverse/user/nolanho/rustrain-env/lib/python3.12/site-packages/torch/include",
            "/usr/local/lib/python3.13/dist-packages/torch/include",
        ];
        for c in &candidates {
            if std::path::Path::new(&format!("{c}/ATen/ATen.h")).exists() {
                return Ok(c.to_string());
            }
        }
        Err(std::env::VarError::NotPresent)
    });

    let torch_include = match torch_include {
        Ok(p) => p,
        Err(_) => {
            println!("cargo:warning=Torch headers not found, C++ kernels disabled");
            return;
        }
    };

    let torch_lib = std::env::var("TORCH_LIB_PATH").or_else(|_| {
        let candidates = [
            "/share/code/nolanho/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib",
            "/vePFS-Mindverse/user/nolanho/rustrain-env/lib/python3.12/site-packages/torch/lib",
            "/usr/local/lib/python3.13/dist-packages/torch/lib",
        ];
        for c in &candidates {
            if std::path::Path::new(&format!("{c}/libtorch.so")).exists() {
                return Ok(c.to_string());
            }
        }
        Err(std::env::VarError::NotPresent)
    });

    let torch_lib = match torch_lib {
        Ok(p) => p,
        Err(_) => {
            println!("cargo:warning=Torch libs not found, C++ kernels disabled");
            return;
        }
    };

    let torch_package_dir = std::path::Path::new(&torch_lib)
        .parent()
        .and_then(std::path::Path::parent);
    let sibling_nccl = torch_package_dir.map(|dir| dir.join("nvidia/nccl"));
    let nccl_include = std::env::var("NCCL_INCLUDE_PATH").unwrap_or_else(|_| {
        let mut candidates = Vec::new();
        if let Some(root) = &sibling_nccl {
            candidates.push(root.join("include"));
        }
        candidates.extend([
            std::path::PathBuf::from(
                "/share/code/nolanho/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/nvidia/nccl/include",
            ),
            std::path::PathBuf::from(
                "/usr/local/lib/python3.13/dist-packages/nvidia/nccl/include",
            ),
            std::path::PathBuf::from("/usr/include"),
        ]);
        candidates
            .into_iter()
            .find(|path| path.join("nccl.h").exists())
            .unwrap_or_else(|| std::path::PathBuf::from("/usr/include"))
            .display()
            .to_string()
    });
    let nccl_lib = std::env::var("NCCL_LIB_PATH").unwrap_or_else(|_| {
        let mut candidates = Vec::new();
        if let Some(root) = &sibling_nccl {
            candidates.push(root.join("lib"));
        }
        candidates.extend([
            std::path::PathBuf::from(
                "/share/code/nolanho/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/nvidia/nccl/lib",
            ),
            std::path::PathBuf::from(
                "/usr/local/lib/python3.13/dist-packages/nvidia/nccl/lib",
            ),
            std::path::PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        ]);
        candidates
            .into_iter()
            .find(|path| {
                path.join("libnccl.so").exists() || path.join("libnccl.so.2").exists()
            })
            .unwrap_or_else(|| std::path::PathBuf::from("/usr/lib/x86_64-linux-gnu"))
            .display()
            .to_string()
    });
    let nccl_link = if std::path::Path::new(&nccl_lib).join("libnccl.so").exists() {
        "-lnccl"
    } else {
        "-l:libnccl.so.2"
    };

    let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| "target/debug".to_string());
    let kernel_src = "kernels/qwen3_6_kernels.cpp";
    let output_lib = format!("{out_dir}/libqwen36_kernels.so");
    // Never leave a previously built kernel at the current OUT_DIR after a
    // failed rebuild. Runtime loading must not silently pick up stale code.
    let _ = std::fs::remove_file(&output_lib);

    println!("cargo:warning=Compiling Qwen3.6 kernels: include={torch_include} lib={torch_lib}");

    let cuda_inc = std::env::var("CUDA_INCLUDE_PATH").unwrap_or_else(|_| {
        let candidates = [
            "/share/code/nolanho/pydeps/lora-research/nvidia/cu13/include",
            "/usr/local/cuda-13/include",
            "/usr/local/cuda-13.0/include",
            "/usr/local/cuda/include",
        ];
        for c in &candidates {
            if std::path::Path::new(&format!("{c}/cuda_runtime_api.h")).exists()
                && std::path::Path::new(&format!("{c}/crt/host_defines.h")).exists()
            {
                return c.to_string();
            }
        }
        "/usr/local/cuda/include".to_string()
    });

    let cpp_ok = Command::new("g++")
        .args([
            "-shared".to_string(),
            "-fPIC".to_string(),
            "-std=c++17".to_string(),
            "-O2".to_string(),
            cxx11_flag.clone(),
            "-fvisibility=default".to_string(),
            "-o".to_string(),
            output_lib.clone(),
            kernel_src.to_string(),
            format!("-I{torch_include}"),
            format!("-I{torch_include}/ATen"),
            format!("-I{torch_include}/c10"),
            format!("-I{torch_include}/caffe2"),
            format!("-I{cuda_inc}"),
            format!("-I{nccl_include}"),
            format!("-L{torch_lib}"),
            format!("-Wl,-rpath,{torch_lib}"),
            format!("-L{nccl_lib}"),
            format!("-Wl,-rpath,{nccl_lib}"),
            "-Wl,--no-as-needed".to_string(),
            "-Wl,--export-dynamic".to_string(),
            "-ltorch".to_string(),
            "-ltorch_cuda".to_string(),
            "-ltorch_cpu".to_string(),
            "-lc10".to_string(),
            "-lc10_cuda".to_string(),
            nccl_link.to_string(),
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    // ── Compile CUDA kernels (.cu files) with nvcc ──
    let nvcc_path = {
        let cuda_from_include = std::path::Path::new(&cuda_inc)
            .parent()
            .map(|home| home.join("bin/nvcc").display().to_string());
        let candidates = [
            std::env::var("CUDA_HOME")
                .ok()
                .map(|h| format!("{h}/bin/nvcc")),
            cuda_from_include,
            Some("nvcc".to_string()),
        ];
        let mut found = String::new();
        for c in &candidates {
            if let Some(c) = c {
                if std::path::Path::new(c).exists() {
                    found = c.clone();
                    break;
                }
                if std::process::Command::new(c)
                    .arg("--version")
                    .output()
                    .is_ok()
                {
                    found = c.clone();
                    break;
                }
            }
        }
        found
    };

    let build_ok = if !nvcc_path.is_empty() {
        // CUDA files needing nvcc: delta_rule.cu (has __global__) + fused_kernels.cu (hand-written CUDA)
        let cu_files_nvcc = ["kernels/delta_rule.cu", "kernels/fused_kernels.cu"];

        let mut obj_files = vec![];

        // Compile CUDA kernels with nvcc
        for cu_file in &cu_files_nvcc {
            if !std::path::Path::new(cu_file).exists() {
                continue;
            }
            let obj_file = format!(
                "{out_dir}/{}.o",
                cu_file.replace("kernels/", "").replace(".cu", "")
            );
            let cu_status = Command::new(&nvcc_path)
                .args([
                    "-c",
                    cu_file,
                    "-o",
                    &obj_file,
                    "-O2",
                    "-std=c++17",
                    &cxx11_flag,
                    &format!("-I{torch_include}"),
                    &format!("-I{torch_include}/ATen"),
                    &format!("-I{torch_include}/c10"),
                    &format!("-I{torch_include}/caffe2"),
                    &format!("-I{cuda_inc}"),
                    "-Xcompiler",
                    "-fPIC",
                ])
                .status();
            if cu_status.map(|s| s.success()).unwrap_or(false) {
                obj_files.push(obj_file);
            }
        }

        // Re-link only when every required CUDA translation unit compiled.
        if obj_files.len() == cu_files_nvcc.len() {
            let mut link_args = vec![
                "-shared".to_string(),
                "-fPIC".to_string(),
                "-std=c++17".to_string(),
                "-O2".to_string(),
                cxx11_flag.clone(),
                "-o".to_string(),
                output_lib.clone(),
                kernel_src.to_string(),
                format!("-I{torch_include}"),
                format!("-I{torch_include}/ATen"),
                format!("-I{torch_include}/c10"),
                format!("-I{torch_include}/caffe2"),
                format!("-I{cuda_inc}"),
                format!("-I{nccl_include}"),
                format!("-L{torch_lib}"),
                format!("-Wl,-rpath,{torch_lib}"),
                format!("-L{nccl_lib}"),
                format!("-Wl,-rpath,{nccl_lib}"),
                format!("-L{cuda_inc}/../lib64"),
                format!("-Wl,-rpath,{cuda_inc}/../lib64"),
                "-Wl,--no-as-needed".to_string(),
                "-ltorch".to_string(),
                "-ltorch_cuda".to_string(),
                "-ltorch_cpu".to_string(),
                "-lc10".to_string(),
                "-lc10_cuda".to_string(),
                "-lcudart".to_string(),
                nccl_link.to_string(),
            ];
            for obj in &obj_files {
                link_args.push(obj.clone());
            }
            cpp_ok
                && Command::new("g++")
                    .args(&link_args)
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false)
        } else {
            false
        }
    } else {
        false
    };

    if build_ok {
        println!("cargo:rustc-link-search=native={torch_lib}");
        println!("cargo:rustc-link-search=native={out_dir}");
        println!("cargo:rustc-link-lib=dylib=qwen36_kernels");
        println!("cargo:rustc-link-lib=dylib=c10");
        println!("cargo:rustc-link-lib=dylib=torch");
        println!("cargo:rustc-link-lib=dylib=torch_cpu");
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");
    } else {
        let _ = std::fs::remove_file(&output_lib);
        println!(
            "cargo:warning=Failed to compile complete Qwen3.6 C++/CUDA kernels; native training path disabled"
        );
    }
}
