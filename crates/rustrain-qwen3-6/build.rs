// build.rs — Compile C++ Qwen3.6 kernels and link against libtorch.
use std::path::PathBuf;
use std::process::Command;

fn which(cmd: &str) -> Option<String> {
    std::process::Command::new("which")
        .arg(cmd)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn main() {
    println!("cargo:rerun-if-env-changed=TORCH_INCLUDE_PATH");
    println!("cargo:rerun-if-env-changed=TORCH_LIB_PATH");
    println!("cargo:rerun-if-changed=kernels/qwen3_6_kernels.cpp");
    println!("cargo:rerun-if-changed=build.rs");

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

    let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| "target/debug".to_string());
    let kernel_src = "kernels/qwen3_6_kernels.cpp";
    let output_lib = format!("{out_dir}/libqwen36_kernels.so");

    println!("cargo:warning=Compiling Qwen3.6 kernels: include={torch_include} lib={torch_lib}");

    let cuda_inc = std::env::var("CUDA_INCLUDE_PATH").unwrap_or_else(|_| {
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

    let status = Command::new("g++")
        .args([
            "-shared", "-fPIC", "-std=c++17", "-O2",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
            "-o", &output_lib, kernel_src,
            &format!("-I{torch_include}"),
            &format!("-I{torch_include}/ATen"),
            &format!("-I{torch_include}/c10"),
            &format!("-I{torch_include}/caffe2"),
            &format!("-I{cuda_inc}"),
        ])
        .args([
            &format!("-L{torch_lib}"),
            &format!("-Wl,-rpath,{torch_lib}"),
            "-Wl,--no-as-needed",
            "-ltorch", "-ltorch_cuda", "-ltorch_cpu", "-lc10",
        ])
        .status();

    // ── Compile CUDA kernels (.cu files) with nvcc ──
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
        // CUDA files needing nvcc: delta_rule.cu (has __global__) + fused_kernels.cu (hand-written CUDA)
        let cu_files_nvcc = ["kernels/delta_rule.cu", "kernels/fused_kernels.cu"];

        let mut obj_files = vec![];

        // Compile CUDA kernels with nvcc
        for cu_file in &cu_files_nvcc {
            if !std::path::Path::new(cu_file).exists() { continue; }
            let obj_file = format!("{out_dir}/{}.o",
                cu_file.replace("kernels/", "").replace(".cu", ""));
            let cu_status = Command::new(&nvcc_path)
                .args([
                    "-c", cu_file, "-o", &obj_file,
                    "-O2", "-std=c++17",
                    "-D_GLIBCXX_USE_CXX11_ABI=1",
                    &format!("-I{torch_include}"),
                    &format!("-I{torch_include}/ATen"),
                    &format!("-I{torch_include}/c10"),
                    &format!("-I{torch_include}/caffe2"),
                    &format!("-I{cuda_inc}"),
                    "-Xcompiler", "-fPIC",
                ])
                .status();
            if cu_status.map(|s| s.success()).unwrap_or(false) {
                obj_files.push(obj_file);
            }
        }

        // Re-link the .so with all .o files
        if !obj_files.is_empty() {
            let mut link_args = vec![
                "-shared".to_string(), "-fPIC".to_string(), "-std=c++17".to_string(), "-O2".to_string(),
                "-D_GLIBCXX_USE_CXX11_ABI=1".to_string(),
                "-o".to_string(), output_lib.clone(), kernel_src.to_string(),
                format!("-I{torch_include}"),
                format!("-I{torch_include}/ATen"),
                format!("-I{torch_include}/c10"),
                format!("-I{torch_include}/caffe2"),
                format!("-I{cuda_inc}"),
                format!("-L{torch_lib}"),
                format!("-Wl,-rpath,{torch_lib}"),
                format!("-L{cuda_inc}/../lib64"),
                format!("-Wl,-rpath,{cuda_inc}/../lib64"),
                "-Wl,--no-as-needed".to_string(),
                "-ltorch".to_string(), "-ltorch_cuda".to_string(), "-ltorch_cpu".to_string(),
                "-lc10".to_string(), "-lcudart".to_string(),
            ];
            for obj in &obj_files {
                link_args.push(obj.clone());
            }
            let _ = Command::new("g++").args(&link_args).status();
        }
    }

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-link-search=native={torch_lib}");
            println!("cargo:rustc-link-search=native={out_dir}");
            println!("cargo:rustc-link-lib=dylib=qwen36_kernels");
            println!("cargo:rustc-link-lib=dylib=c10");
            println!("cargo:rustc-link-lib=dylib=torch");
            println!("cargo:rustc-link-lib=dylib=torch_cpu");
            println!("cargo:rustc-link-lib=dylib=torch_cuda");
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");
        }
        _ => {
            println!("cargo:warning=Failed to compile Qwen3.6 kernels, C++ path disabled");
        }
    }
}
