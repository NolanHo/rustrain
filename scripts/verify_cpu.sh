#!/usr/bin/env bash
set -euo pipefail

cargo test
cargo run -- train --config configs/debug.toml
cargo run -- train --config configs/qwen3_mini.toml
cargo run -- train --config configs/tch_smoke.toml
cargo run -- train --config configs/text_debug.toml
cargo run -- train --config configs/gsm8k_toy.toml
cargo run -- train --config configs/sft_debug.toml
cargo run -- qwen-lora-smoke
cargo run -- qwen-kv-cache-parity
cargo run -- parallel-dp-smoke --output-dir runs/parallel-dp-smoke
cargo run -- parallel-tp-smoke
cargo run -- parallel-ep-smoke
