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
    println!("cargo:rerun-if-changed=kernels/glm5_attention.cpp");
    println!("cargo:rerun-if-changed=kernels/v4_flash_kernels.cpp");
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

    // ── Compile FP8 GEMM kernel ──
    let fp8_src = "kernels/fp8_gemm.cpp";
    let fp8_lib = format!("{out_dir}/libfp8_gemm.so");

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
            &fp8_lib,
            fp8_src,
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
            "-lc10_cuda",
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-link-search=native={torch_lib}");
            println!("cargo:rustc-link-search=native={out_dir}");
            println!("cargo:rustc-link-lib=dylib=fp8_gemm");
            println!("cargo:rustc-link-lib=dylib=c10");
            println!("cargo:rustc-link-lib=dylib=c10_cuda");
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

    // ── Compile GLM5 attention kernel ──
    let glm5_src = "kernels/glm5_attention.cpp";
    let glm5_lib = format!("{out_dir}/libglm5_attention.so");

    println!("cargo:warning=Compiling GLM5 attention kernel: src={glm5_src}");

    let rpath_arg = format!("-Wl,-rpath,{torch_lib}");
    let l_arg = format!("-L{torch_lib}");
    let inc_aten = format!("-I{torch_include}/ATen");
    let inc_c10 = format!("-I{torch_include}/c10");
    let inc_caffe2 = format!("-I{torch_include}/caffe2");
    let inc_torch = format!("-I{torch_include}");
    let inc_cuda = format!("-I{cuda_inc}");

    let glm5_status = Command::new("g++")
        .args([
            "-shared", "-fPIC", "-std=c++17", "-O2",
            cxx11_abi,
            "-o", &glm5_lib,
            glm5_src,
            &inc_torch, &inc_aten, &inc_c10, &inc_caffe2, &inc_cuda,
            &l_arg, &rpath_arg,
            "-Wl,--no-as-needed",
            "-ltorch", "-ltorch_cuda", "-ltorch_cpu", "-lc10", "-lc10_cuda",
        ])
        .args(if cuda_inc2.is_empty() { vec![] } else { vec![format!("-I{cuda_inc2}")] })
        .status();

    match glm5_status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-link-lib=dylib=glm5_attention");
        }
        _ => {
            println!("cargo:warning=Failed to compile GLM5 attention kernel, C++ attention disabled");
        }
    }

    // ── Compile V4 Flash kernel ──
    let v4_src = "kernels/v4_flash_kernels.cpp";
    let v4_lib = format!("{out_dir}/libv4_flash_kernels.so");

    // Find NCCL include + lib paths
    let nccl_inc = std::env::var("NCCL_INCLUDE_PATH")
        .unwrap_or_else(|_| {
            let candidates = [
                "/share/code/nolanho/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/nvidia/nccl/include",
            ];
            for c in &candidates {
                if std::path::Path::new(&format!("{c}/nccl.h")).exists() {
                    return c.to_string();
                }
            }
            String::new()
        });
    let nccl_lib_dir = std::env::var("NCCL_LIB_PATH")
        .unwrap_or_else(|_| {
            let candidates = [
                "/share/code/nolanho/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/nvidia/nccl/lib",
            ];
            for c in &candidates {
                if std::path::Path::new(&format!("{c}/libnccl.so")).exists() {
                    return c.to_string();
                }
            }
            String::new()
        });

    let nccl_inc_arg = if nccl_inc.is_empty() { String::new() } else { format!("-I{nccl_inc}") };
    let nccl_lib_args: Vec<String> = if nccl_lib_dir.is_empty() {
        vec![]
    } else {
        vec![format!("-L{nccl_lib_dir}"), format!("-Wl,-rpath,{nccl_lib_dir}"), "-lnccl".to_string()]
    };

    println!("cargo:warning=Compiling V4 Flash kernel: src={v4_src} nccl_inc={nccl_inc} nccl_lib={nccl_lib_dir}");

    let mut v4_args: Vec<String> = vec![
        "-shared", "-fPIC", "-std=c++17", "-O2",
        cxx11_abi,
        "-o", &v4_lib,
        v4_src,
        &inc_torch, &inc_aten, &inc_c10, &inc_caffe2, &inc_cuda,
        &l_arg, &rpath_arg,
        "-Wl,--no-as-needed",
        "-ltorch", "-ltorch_cuda", "-ltorch_cpu", "-lc10", "-lc10_cuda",
    ].into_iter().map(String::from).collect();
    if !nccl_inc_arg.is_empty() { v4_args.push(nccl_inc_arg); }
    if !cuda_inc2.is_empty() { v4_args.push(format!("-I{cuda_inc2}")); }
    v4_args.extend(nccl_lib_args.iter().cloned());

    let v4_status = Command::new("g++")
        .args(&v4_args)
        .status();

    match v4_status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-link-lib=dylib=v4_flash_kernels");
            if !nccl_lib_dir.is_empty() {
                println!("cargo:rustc-link-search=native={nccl_lib_dir}");
                println!("cargo:rustc-link-lib=dylib=nccl");
            }
        }
        _ => {
            println!("cargo:warning=Failed to compile V4 Flash kernel, V4 C++ path disabled");
        }
    }

    // ── Compile fused_kernels.cu with nvcc ──
    let nvcc_path = {
        let candidates = [
            std::env::var("CUDA_HOME").ok().map(|h| format!("{h}/bin/nvcc")),
            Some("nvcc".to_string()),
        ];
        let mut found = String::new();
        for c in &candidates {
            if let Some(c) = c {
                if std::path::Path::new(c).exists() {
                    found = c.clone();
                    break;
                }
                if std::process::Command::new(c).arg("--version").output().is_ok() {
                    found = c.clone();
                    break;
                }
            }
        }
        found
    };

    if !nvcc_path.is_empty() {
        let fused_cu = "kernels/fused_kernels.cu";
        if std::path::Path::new(fused_cu).exists() {
            let fused_obj = format!("{out_dir}/fused_kernels.o");
            let fused_status = Command::new(&nvcc_path)
                .args([
                    "-c", fused_cu, "-o", &fused_obj,
                    "-O2", "-std=c++17",
                    "-D_GLIBCXX_USE_CXX11_ABI=1",
                    &format!("-I{inc_torch}"),
                    &format!("-I{inc_aten}"),
                    &format!("-I{inc_c10}"),
                    &format!("-I{inc_caffe2}"),
                    &format!("-I{inc_cuda}"),
                    "-Xcompiler", "-fPIC",
                ])
                .status();
            if fused_status.map(|s| s.success()).unwrap_or(false) {
                // Re-link v4_flash_kernels.so with fused_kernels.o
                let mut relink_args = vec![
                    "-shared".to_string(), "-fPIC".to_string(), "-std=c++17".to_string(), "-O2".to_string(),
                    cxx11_abi.to_string(),
                    "-o".to_string(), v4_lib.clone(), v4_src.to_string(), fused_obj.clone(),
                    inc_torch.clone(), inc_aten.clone(), inc_c10.clone(), inc_caffe2.clone(), inc_cuda.clone(),
                    l_arg.clone(), rpath_arg.clone(),
                    "-Wl,--no-as-needed".to_string(),
                    "-ltorch".to_string(), "-ltorch_cuda".to_string(), "-ltorch_cpu".to_string(),
                    "-lc10".to_string(), "-lc10_cuda".to_string(),
                ];
                relink_args.extend(nccl_lib_args.iter().cloned());
                let _ = Command::new("g++").args(&relink_args).status();
                println!("cargo:warning=Fused CUDA kernels compiled and linked into v4_flash_kernels.so");
            }
        }
    }
}
