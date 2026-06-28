// build.rs — Compile C++ FP8 GEMM shim and link against libtorch.
//
// Detects the PyTorch installation (same one tch-rs uses) and compiles
// kernels/fp8_gemm.cpp with g++, linking against libtorch/libtorch_cuda.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Always re-run when env vars change
    println!("cargo:rerun-if-env-changed=TORCH_INCLUDE_PATH");
    println!("cargo:rerun-if-env-changed=TORCH_LIB_PATH");
    println!("cargo:rerun-if-changed=kernels/fp8_gemm.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    // Skip on non-CUDA builds or when torch isn't available
    let torch_include = std::env::var("TORCH_INCLUDE_PATH")
        .or_else(|_| {
            let candidates = [
                "/vePFS-Mindverse/user/nolanho/rustrain-env/lib/python3.12/site-packages/torch/include",
                "/share/code/nolanho/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/include",
                "/vePFS-Mindverse/user/nolanho/hackathon-env/venv/lib/python3.12/site-packages/torch/include",
                "/vePFS-mindverse/user/nolanho/venv/lib/python3.12/site-packages/torch/include",
                "/usr/local/lib/python3.13/dist-packages/torch/include",
                "/usr/local/lib/python3.12/dist-packages/torch/include",
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
            println!("cargo:warning=Torch headers not found, FP8 kernel disabled");
            return;
        }
    };

    let torch_lib = std::env::var("TORCH_LIB_PATH")
        .or_else(|_| {
            let candidates = [
                "/vePFS-Mindverse/user/nolanho/rustrain-env/lib/python3.12/site-packages/torch/lib",
                "/share/code/nolanho/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib",
                "/vePFS-Mindverse/user/nolanho/hackathon-env/venv/lib/python3.12/site-packages/torch/lib",
                "/vePFS-mindverse/user/nolanho/venv/lib/python3.12/site-packages/torch/lib",
                "/usr/local/lib/python3.13/dist-packages/torch/lib",
                "/usr/local/lib/python3.12/dist-packages/torch/lib",
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
            println!("cargo:warning=Torch libs not found, FP8 kernel disabled");
            return;
        }
    };

    let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| "target/debug".to_string());
    let kernel_src = "kernels/fp8_gemm.cpp";
    let output_lib = format!("{out_dir}/libfp8_gemm.so");

    println!("cargo:warning=Compiling FP8 GEMM kernel: include={torch_include} lib={torch_lib}");

    // Detect CXX11 ABI from PyTorch
    let cxx11_abi = "-D_GLIBCXX_USE_CXX11_ABI=1";

    // Find CUDA include path
    let cuda_inc = std::env::var("CUDA_INCLUDE_PATH")
        .unwrap_or_else(|_| {
            let candidates = [
                "/share/code/nolanho/pydeps/lora-research/nvidia/cu13/include",
                "/usr/local/cuda-13.0/include",
                "/usr/local/cuda/include",
            ];
            for c in &candidates {
                if std::path::Path::new(&format!("{c}/cuda_runtime_api.h")).exists() {
                    return c.to_string();
                }
            }
            "/usr/local/cuda/include".to_string()
        });

    // CUDA headers are split across two dirs (cuda_runtime_api.h in one, crt/ in another)
    let cuda_inc2 = std::env::var("CUDA_INCLUDE_PATH2")
        .unwrap_or_else(|_| {
            let candidates = [
                "/share/code/nolanho/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/triton/backends/nvidia/include",
            ];
            for c in &candidates {
                if std::path::Path::new(&format!("{c}/crt/host_defines.h")).exists() {
                    return c.to_string();
                }
            }
            String::new()
        });

    // Compile with g++ (no nvcc needed)
    let status = Command::new("g++")
        .args([
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-O2",
            cxx11_abi,
            "-o",
            &output_lib,
            kernel_src,
            &format!("-I{torch_include}"),
            &format!("-I{torch_include}/ATen"),
            &format!("-I{torch_include}/c10"),
            &format!("-I{torch_include}/caffe2"),
            &format!("-I{cuda_inc}"),
        ])
        .args(if cuda_inc2.is_empty() {
            vec![]
        } else {
            vec![format!("-I{cuda_inc2}")]
        })
        .args([
            &format!("-L{torch_lib}"),
            &format!("-Wl,-rpath,{torch_lib}"),
            "-Wl,--no-as-needed",  // Force all libs into NEEDED list
            "-ltorch",
            "-ltorch_cuda",
            "-ltorch_cpu",
            "-lc10",
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-link-search=native={torch_lib}");
            println!("cargo:rustc-link-search=native={out_dir}");
            println!("cargo:rustc-link-lib=dylib=fp8_gemm");
            println!("cargo:rustc-link-lib=dylib=c10");
            println!("cargo:rustc-link-lib=dylib=torch");
            println!("cargo:rustc-link-lib=dylib=torch_cpu");
            println!("cargo:rustc-link-lib=dylib=torch_cuda");
            // Force all libs into NEEDED list (rust-lld drops unused ones)
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            // Allow unresolved shared lib symbols (libfp8_gemm.so depends
            // on libc10.so/torch libs which are found at runtime via LD_LIBRARY_PATH)
            println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");
        }
        _ => {
            println!("cargo:warning=Failed to compile FP8 GEMM kernel, FP8 path disabled");
        }
    }
}
