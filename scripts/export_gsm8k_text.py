#!/usr/bin/env python3
"""Export a small GSM8K text corpus from the local HuggingFace Arrow cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


DEFAULT_ARROW = Path(
    "/vePFS-Mindverse/share/huggingface/datasets/gsm8k/main/0.0.0/"
    "e53f048856ff4f594e959d75785d2c2d37b678ee/gsm8k-train.arrow"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--output", type=Path, default=Path("data/gsm8k_toy/gsm8k_toy.txt"))
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    dataset = Dataset.from_file(str(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output:
        for index, row in enumerate(dataset):
            if index >= args.limit:
                break
            question = " ".join(str(row["question"]).split())
            answer = " ".join(str(row["answer"]).split())
            output.write(f"Question: {question}\nAnswer: {answer}\n\n")

    print(f"wrote {min(args.limit, len(dataset))} examples to {args.output}")


if __name__ == "__main__":
    main()
