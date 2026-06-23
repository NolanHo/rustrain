#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_EP_RANK_LOCAL_OUTPUT_DIR:-/tmp/rustrain-runs/ep-rank-local-verify-$$}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  moe parallel-ep-rank-smoke \
  --output-dir "${OUTPUT_DIR}/ranks"

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
rank_dir = output_dir / "ranks"
summaries = sorted(rank_dir.glob("ep-rank-*.json"))
summaries = [path for path in summaries if not path.name.endswith("-output.json")]
if len(summaries) != 2:
    raise SystemExit(f"expected 2 EP rank summaries under {rank_dir}, found {len(summaries)}")

tokens = [
    [1.0, 0.0, 0.5],
    [0.0, 1.0, -0.5],
    [1.0, 1.0, 0.0],
    [-1.0, 0.5, 1.0],
]
router = [
    [0.9, -0.2, 0.1, 0.0],
    [0.1, 0.8, -0.4, 0.2],
    [0.0, -0.3, 0.7, 0.6],
]
expert_scales = [
    [1.0, 0.5, -0.25],
    [-0.5, 1.5, 0.25],
    [0.25, -1.0, 1.25],
    [1.2, 0.3, 0.8],
]

assignments = []
for token in tokens:
    scores = []
    for expert_index in range(len(expert_scales)):
        scores.append(sum(token[h] * router[h][expert_index] for h in range(len(token))))
    assignments.append(max(range(len(scores)), key=lambda index: scores[index]))

reference = []
for token, expert_index in zip(tokens, assignments):
    reference.append(
        [token[h] * expert_scales[expert_index][h] for h in range(len(token))]
    )

gathered = [[0.0 for _ in tokens[0]] for _ in tokens]
covered_tokens = []
rank_evidence = []
expert_load = [0 for _ in expert_scales]
owned_ranges = []

for path in summaries:
    data = json.loads(path.read_text())
    rank = int(data["rank"])
    if data["world_size"] != 2:
        raise SystemExit(f"{path} world_size {data['world_size']} != 2")
    if data["local_rank"] != rank:
        raise SystemExit(f"{path} local_rank {data['local_rank']} != rank {rank}")

    start = int(data["owned_expert_start"])
    end = int(data["owned_expert_end"])
    owned_ranges.append((start, end))
    for token_index in data["owned_token_indices"]:
        expert_index = assignments[int(token_index)]
        if not start <= expert_index < end:
            raise SystemExit(
                f"{path} token {token_index} routed to expert {expert_index}, "
                f"outside owned range [{start}, {end})"
            )
    output_path = pathlib.Path(data["local_output_path"])
    if not output_path.exists():
        raise SystemExit(f"{path} references missing local output {output_path}")
    local_output = json.loads(output_path.read_text())
    if len(local_output) != len(tokens):
        raise SystemExit(f"{output_path} row count {len(local_output)} != {len(tokens)}")
    for row in local_output:
        if len(row) != len(tokens[0]):
            raise SystemExit(f"{output_path} has wrong hidden dimension in row {row}")
    for token_index in data["owned_token_indices"]:
        token_index = int(token_index)
        covered_tokens.append(token_index)
        for hidden_index in range(len(tokens[0])):
            gathered[token_index][hidden_index] += float(local_output[token_index][hidden_index])
    for expert_index, load in enumerate(data["expert_load"]):
        expert_load[expert_index] += int(load)
    rank_evidence.append(
        {
            "rank": rank,
            "owned_experts": [start, end],
            "owned_token_indices": data["owned_token_indices"],
        }
    )

if sorted(owned_ranges) != [(0, 2), (2, 4)]:
    raise SystemExit(f"unexpected expert ownership ranges: {owned_ranges}")

if sorted(covered_tokens) != list(range(len(tokens))):
    raise SystemExit(f"rank-local outputs did not cover each token exactly once: {covered_tokens}")

expected_load = [assignments.count(expert_index) for expert_index in range(len(expert_scales))]
if expert_load != expected_load:
    raise SystemExit(f"expert load {expert_load} != expected {expected_load}")

max_abs = 0.0
for actual_row, expected_row in zip(gathered, reference):
    for actual, expected in zip(actual_row, expected_row):
        max_abs = max(max_abs, abs(actual - expected))
if max_abs > 1e-12:
    raise SystemExit(f"EP rank-local output mismatch: max_abs={max_abs}")

launch_summary = json.loads((output_dir / "launch-summary.json").read_text())
assigned = [rank["assigned_cuda_visible_device"] for rank in launch_summary["ranks"]]
if len(set(assigned)) != 2:
    raise SystemExit(f"expected distinct launch GPU assignments, got {assigned}")

print(
    json.dumps(
        {
            "ep_rank_local_verified": rank_evidence,
            "assignments": assignments,
            "expert_load": expert_load,
            "output_max_delta": max_abs,
            "assigned_cuda_visible_devices": assigned,
        },
        indent=2,
    )
)
PY
