#!/usr/bin/env python3
"""Export instruction-style JSONL from a local HuggingFace Arrow cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--instruction-column", default="instruction")
    parser.add_argument("--input-column", default="input")
    parser.add_argument("--response-column", default="output")
    parser.add_argument("--metadata-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Arrow input is missing: {args.input}")

    with ipc.open_stream(args.input) as reader:
        table = reader.read_all()

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index in range(args.limit):
            record = {
                "instruction": table.column(args.instruction_column)[index].as_py(),
                "input": table.column(args.input_column)[index].as_py(),
                "response": table.column(args.response_column)[index].as_py(),
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    metadata = {
        "source_arrow": str(args.input),
        "source_rows": table.num_rows,
        "exported_rows": args.limit,
        "columns": table.schema.names,
        "column_map": column_map,
        "output": str(args.output),
    }
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    print(
        "exported instruction JSONL: "
        f"source_rows={table.num_rows} exported_rows={args.limit} output={args.output}"
    )


if __name__ == "__main__":
    main()
