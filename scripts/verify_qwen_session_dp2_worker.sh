#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR:?RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR is required}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  train --config configs/qwen_session_dp2.toml

python - <<'PY'
import json
import os
import pathlib

output_dir = pathlib.Path(os.environ["RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR"])
rank_summaries = sorted(output_dir.rglob("qwen-session-dp-rank-*.json"))
if len(rank_summaries) != 2:
    raise SystemExit(
        f"expected 2 qwen session DP rank summaries under {output_dir}, found {len(rank_summaries)}"
    )

evidence = []
for path in rank_summaries:
    data = json.loads(path.read_text())
    sharded_reload_delta = data.get("sharded_reload_delta")
    if sharded_reload_delta is None:
        raise SystemExit(f"{path} is missing sharded_reload_delta")
    if sharded_reload_delta > 1e-5:
        raise SystemExit(
            f"{path} sharded_reload_delta {sharded_reload_delta} exceeds tolerance"
        )
    evidence.append(
        {
            "rank": data["rank"],
            "checkpoint_written": data["checkpoint_written"],
            "reload_delta": data["reload_delta"],
            "sharded_reload_delta": sharded_reload_delta,
            "sharded_global_manifest_output": data["sharded_global_manifest_output"],
        }
    )

print(json.dumps({"qwen_session_dp_sharded_restore": evidence}, indent=2, sort_keys=True))
PY
