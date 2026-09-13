#!/usr/bin/env python3
"""Regenerate a `rustrain.ckpt_meta.v1` snapshot from a HuggingFace safeTensors checkpoint.

Only metadata is fetched: the index, and the **header** of every shard, over HTTP Range. A 72 GB
checkpoint costs about a megabyte of traffic, and no weight byte is ever requested.

The output is deterministic — tensors in ASCII order, one compact object per tensor, exactly the
layout of the committed fixture — so the same checkpoint revision always regenerates the same
bytes:

    python3 scripts/fetch_qwen36_meta.py --out /var/tmp/qwen36.meta.json
    sha256sum /var/tmp/qwen36.meta.json
    # b89564e31020da760be9fa671396793a71e4a527bf35ed9e42c1ac2347d5d20b

`--repo` / `--revision` / `--index-url` exist so a *different* checkpoint can be snapshotted the
same way (spec C5: the script and the snapshot are each independently reproducible). Passing an
`--index-url` without a scheme reads a local model directory instead, so the snapshot can also be
rebuilt offline from a checkpoint already on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_REPO = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_REVISION = "main"
INDEX_FILE = "model.safetensors.index.json"
SNAPSHOT_FORMAT = "rustrain.ckpt_meta.v1"
RETRIES = 3
TIMEOUT_SECONDS = 60.0


def read_bytes(source: str, byte_range: tuple[int, int] | None = None) -> bytes:
    """One read, over HTTP Range or from a local file. A URL without a scheme is a path."""
    if "://" not in source:
        with open(source, "rb") as handle:
            if byte_range is not None:
                handle.seek(byte_range[0])
                return handle.read(byte_range[1] - byte_range[0] + 1)
            return handle.read()

    request = urllib.request.Request(source, headers={"User-Agent": "rustrain-fetch-meta"})
    if byte_range is not None:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < RETRIES:
                time.sleep(2**attempt)
    raise SystemExit(f"cannot read {source}: {last_error}")


def read_shard_header(url: str) -> dict:
    """A safeTensors header: 8 little-endian bytes of length, then that many bytes of JSON."""
    prefix = read_bytes(url, (0, 7))
    if len(prefix) != 8:
        raise SystemExit(f"{url}: expected an 8-byte header length, got {len(prefix)} bytes")
    length = int.from_bytes(prefix, "little")
    header = read_bytes(url, (8, 8 + length - 1))
    if len(header) != length:
        raise SystemExit(f"{url}: a {length}-byte header was announced, {len(header)} came back")
    return json.loads(header)


def snapshot(index_url: str) -> dict[str, dict]:
    """`{tensor name: {"dtype": ..., "shape": [...]}}` for every tensor the index maps."""
    index = json.loads(read_bytes(index_url))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    # The shard files are siblings of the index, whatever the index was read from.
    base = index_url[: index_url.rindex("/") + 1]

    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)

    tensors: dict[str, dict] = {}
    for number, shard in enumerate(shards, start=1):
        header = read_shard_header(base + shard)
        for name in by_shard[shard]:
            entry = header.get(name)
            if entry is None:
                raise SystemExit(f"{shard}: the index lists `{name}`, but the shard header does not")
            tensors[name] = {
                # safeTensors spells dtypes in upper case (`BF16`, `F8_E4M3`); the description
                # language's vocabulary (model-description §3.6 #5) is the lower-case spelling.
                "dtype": str(entry["dtype"]).lower().replace("_", ""),
                "shape": [int(dim) for dim in entry["shape"]],
            }
        print(f"  [{number}/{len(shards)}] {shard}: {len(by_shard[shard])} tensor(s)", file=sys.stderr)

    missing = sorted(set(weight_map) - set(tensors))
    if missing:
        raise SystemExit(f"{len(missing)} tensor(s) the index lists are in no shard header: {missing[:8]}")
    return tensors


def write_snapshot(path: str, source: str, tensors: dict[str, dict]) -> None:
    """The committed layout: 2-space indent, one compact object per tensor, ASCII order, `\\n`."""
    names = sorted(tensors)
    lines = [
        "{",
        f'  "format": "{SNAPSHOT_FORMAT}",',
        f'  "source": {json.dumps(source)},',
        '  "tensors": {',
    ]
    for position, name in enumerate(names):
        comma = "" if position + 1 == len(names) else ","
        lines.append(f"    {json.dumps(name)}: {json.dumps(tensors[name])}{comma}")
    lines += ["  }", "}"]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate a rustrain.ckpt_meta.v1 snapshot by reading shard headers only."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HuggingFace repository id")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="HuggingFace revision (branch, tag or commit)")
    parser.add_argument(
        "--index-url",
        default=None,
        help="read the index from here instead of --repo/--revision "
        "(a URL, or a local path to a model directory's index for an offline rebuild)",
    )
    parser.add_argument("--out", required=True, help="where to write the snapshot")
    args = parser.parse_args()

    index_url = args.index_url or (
        f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{INDEX_FILE}"
    )
    print(f"reading {index_url}", file=sys.stderr)
    tensors = snapshot(index_url)
    write_snapshot(args.out, index_url, tensors)
    print(f"wrote {len(tensors)} tensor(s) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
