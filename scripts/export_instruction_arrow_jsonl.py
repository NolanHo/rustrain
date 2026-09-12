#!/usr/bin/env python3
"""Export instruction-style JSONL from a local HuggingFace Arrow cache."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


DEFAULT_ARROW = Path(
    "/path/to/huggingface/datasets/"
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
    parser.add_argument(
        "--no-full-row-count",
        action="store_true",
        help=(
            "stop scanning after --limit exported rows; metadata records "
            "source_rows as a lower bound instead of the exact source total"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Arrow input is missing: {args.input}")

    column_map = {
        "instruction": args.instruction_column,
        "input": args.input_column,
        "response": args.response_column,
    }
    if args.shards > args.limit:
        raise SystemExit(f"--shards {args.shards} cannot exceed --limit {args.limit}")
    rows, source_rows, source_rows_exact, columns, arrow_ipc_format = export_arrow_rows(
        args.input,
        column_map,
        args.limit,
        full_row_count=not args.no_full_row_count,
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
        "source_rows": source_rows,
        "source_rows_exact": source_rows_exact,
        "source_rows_lower_bound": not source_rows_exact,
        "arrow_ipc_format": arrow_ipc_format,
        "exported_rows": args.limit,
        "columns": columns,
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
        f"source_rows={source_rows} exported_rows={args.limit} "
        f"shards={args.shards} output={args.output}"
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def export_arrow_rows(
    path: Path,
    column_map: dict[str, str],
    limit: int,
    *,
    full_row_count: bool,
) -> tuple[list[dict[str, object]], int, bool, list[str], str]:
    try:
        with ipc.open_stream(path) as reader:
            rows, source_rows, source_rows_exact, columns = collect_rows_from_batches(
                reader.schema,
                reader,
                column_map,
                limit,
                full_row_count=full_row_count,
            )
            return rows, source_rows, source_rows_exact, columns, "stream"
    except (pa.ArrowInvalid, OSError) as stream_error:
        try:
            with ipc.open_file(path) as reader:
                batches = (reader.get_batch(index) for index in range(reader.num_record_batches))
                rows, source_rows, source_rows_exact, columns = collect_rows_from_batches(
                    reader.schema,
                    batches,
                    column_map,
                    limit,
                    full_row_count=full_row_count,
                )
                return rows, source_rows, source_rows_exact, columns, "file"
        except (pa.ArrowInvalid, OSError) as file_error:
            raise SystemExit(
                f"failed to open {path} as Arrow IPC stream or file: "
                f"stream={stream_error}; file={file_error}"
            ) from file_error


def collect_rows_from_batches(
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
    column_map: dict[str, str],
    limit: int,
    *,
    full_row_count: bool,
) -> tuple[list[dict[str, object]], int, bool, list[str]]:
    columns = schema.names
    missing = sorted(set(column_map.values()).difference(columns))
    if missing:
        raise SystemExit(f"Arrow input is missing required columns: {missing}")

    column_indices = {
        output_name: schema.get_field_index(input_name)
        for output_name, input_name in column_map.items()
    }
    rows: list[dict[str, object]] = []
    source_rows = 0
    for batch in batches:
        source_rows += batch.num_rows
        remaining = limit - len(rows)
        if remaining <= 0:
            if not full_row_count:
                break
            continue
        take = min(remaining, batch.num_rows)
        arrays = {
            output_name: batch.column(column_index).slice(0, take).to_pylist()
            for output_name, column_index in column_indices.items()
        }
        for index in range(take):
            rows.append(
                {
                    "instruction": arrays["instruction"][index],
                    "input": arrays["input"][index],
                    "response": arrays["response"][index],
                }
            )
        if len(rows) >= limit and not full_row_count:
            break

    if source_rows < limit:
        raise SystemExit(f"Arrow input has {source_rows} rows, below limit {limit}")
    source_rows_exact = full_row_count
    return rows, source_rows, source_rows_exact, columns


if __name__ == "__main__":
    main()
