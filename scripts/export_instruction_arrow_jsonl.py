#!/usr/bin/env python3
"""Export instruction-style JSONL from a local HuggingFace Arrow cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


DEFAULT_ARROW = Path(
    "/vePFS-Mindverse/share/huggingface/datasets/"
    "iamtarun___code_instructions_120k_alpaca/default/0.0.0/"
    "31f725b2d714c1b4f038e80fbaa6b977870a50b7/"
    "code_instructions_120k_alpaca-train.arrow"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export instruction/input/response JSONL from a local Arrow cache."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=positive_int, default=128)
    parser.add_argument("--shards", type=positive_int, default=1)
    parser.add_argument("--instruction-column", default="instruction")
    parser.add_argument("--input-column", default="input")
    parser.add_argument("--response-column", default="output")
    parser.add_argument("--metadata-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Arrow input is missing: {args.input}")

    table, arrow_ipc_format = read_arrow_table(args.input)

    column_map = {
        "instruction": args.instruction_column,
        "input": args.input_column,
        "response": args.response_column,
    }
    missing = sorted(set(column_map.values()).difference(table.schema.names))
    if missing:
        raise SystemExit(f"{args.input} is missing required columns: {missing}")
    if table.num_rows < args.limit:
        raise SystemExit(f"{args.input} has {table.num_rows} rows, below limit {args.limit}")
    if args.shards > args.limit:
        raise SystemExit(f"--shards {args.shards} cannot exceed --limit {args.limit}")

    rows = []
    for index in range(args.limit):
        rows.append(
            {
                "instruction": table.column(args.instruction_column)[index].as_py(),
                "input": table.column(args.input_column)[index].as_py(),
                "response": table.column(args.response_column)[index].as_py(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_files = []
    if args.shards == 1:
        output_files.append({"path": str(args.output), "rows": len(rows)})
        write_jsonl(args.output, rows)
    else:
        for shard_index in range(args.shards):
            start = shard_index * len(rows) // args.shards
            end = (shard_index + 1) * len(rows) // args.shards
            shard_path = args.output.with_name(
                f"{args.output.stem}_{shard_index}{args.output.suffix}"
            )
            shard_rows = rows[start:end]
            output_files.append({"path": str(shard_path), "rows": len(shard_rows)})
            write_jsonl(shard_path, shard_rows)

    metadata = {
        "source_arrow": str(args.input),
        "source_rows": table.num_rows,
        "arrow_ipc_format": arrow_ipc_format,
        "exported_rows": args.limit,
        "columns": table.schema.names,
        "column_map": column_map,
        "output": str(args.output),
        "shards": args.shards,
        "output_files": output_files,
    }
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    print(
        "exported instruction JSONL: "
        f"source_rows={table.num_rows} exported_rows={args.limit} "
        f"shards={args.shards} output={args.output}"
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_arrow_table(path: Path) -> tuple[pa.Table, str]:
    try:
        with ipc.open_stream(path) as reader:
            return reader.read_all(), "stream"
    except (pa.ArrowInvalid, OSError) as stream_error:
        try:
            with ipc.open_file(path) as reader:
                return reader.read_all(), "file"
        except (pa.ArrowInvalid, OSError) as file_error:
            raise SystemExit(
                f"failed to open {path} as Arrow IPC stream or file: "
                f"stream={stream_error}; file={file_error}"
            ) from file_error


if __name__ == "__main__":
    main()
