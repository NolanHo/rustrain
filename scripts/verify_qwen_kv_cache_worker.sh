#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

MAX_NEW_TOKENS="${RUSTRAIN_QWEN_KV_CACHE_MAX_NEW_TOKENS:-4}"
OUTPUT="$(mktemp)"

cargo run -- qwen-kv-cache-parity --max-new-tokens "${MAX_NEW_TOKENS}" | tee "${OUTPUT}"

python - "${OUTPUT}" "${MAX_NEW_TOKENS}" <<'PY'
import json
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
expected_new_tokens = int(sys.argv[2])
start = text.find("{")
end = text.rfind("}")
if start < 0 or end < start:
    raise SystemExit("qwen-kv-cache-parity did not print a JSON summary")
summary = json.loads(text[start : end + 1])

required = [
    "prompt_len",
    "max_new_tokens",
    "full_context_ids",
    "cached_ids",
    "new_token_ids",
    "reference_match",
]
missing = [key for key in required if key not in summary]
if missing:
    raise SystemExit(f"summary is missing fields: {missing}")
if summary["max_new_tokens"] != expected_new_tokens:
    raise SystemExit(
        f"expected max_new_tokens {expected_new_tokens}, got {summary['max_new_tokens']}"
    )
if summary["reference_match"] is not True:
    raise SystemExit("cached greedy generation did not match full-context generation")
if summary["full_context_ids"] != summary["cached_ids"]:
    raise SystemExit("full_context_ids and cached_ids differ")
prompt_len = int(summary["prompt_len"])
expected_len = prompt_len + expected_new_tokens
if len(summary["cached_ids"]) != expected_len:
    raise SystemExit(
        f"expected cached sequence length {expected_len}, got {len(summary['cached_ids'])}"
    )
if len(summary["new_token_ids"]) != expected_new_tokens:
    raise SystemExit(
        f"expected {expected_new_tokens} new tokens, got {summary['new_token_ids']}"
    )
if summary["cached_ids"][prompt_len:] != summary["new_token_ids"]:
    raise SystemExit("new_token_ids do not match cached suffix")

print(
    "qwen_kv_cache_verified: "
    f"prompt_len={prompt_len} "
    f"new_token_ids={summary['new_token_ids']}"
)
PY
