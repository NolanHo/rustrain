fn main() {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!(
        "cargo:rustc-link-search=native=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib"
    );
    println!(
        "cargo:rustc-link-search=native=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib"
    );
}
