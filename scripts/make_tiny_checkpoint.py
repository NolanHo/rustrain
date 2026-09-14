#!/usr/bin/env python3
"""Write the tiny safetensors checkpoint the CLI run tests use, on this box.

The CPU test `crates/rustrain-cli/tests/run_multi.rs` writes the same bytes
in-process. This script exists for the **verification host**, where the same
tiny model is run through the real NCCL transport: `launch --sweep tp=2` on the
`run-tiny-logits` fixture is the cheapest end-to-end cover of a vocabulary
gather (the head's axis 1 is sharded by tp), and it must agree with the
world-1 forward.

Usage (host):

    python3 scripts/make_tiny_checkpoint.py --out /root/rustrain-gpu/d6tiny-logits --with-head
    ./target/release/rustrain launch \
        --model crates/rustrain-cli/tests/fixtures/run-tiny-logits \
        --checkpoint /root/rustrain-gpu/d6tiny-logits \
        --tokens 0,1,2,3 --sweep tp=2 --out /var/tmp/tiny-logits-sweep.json \
        --plugin /root/rustrain-gpu/aten-build/librustrain_aten.so \
        --recipe plugins/aten/aten.toml --device cuda --nccl-lib "$NCCL" --keep-rdzv

An extra tensor no binding consumes is a hard error, so `--with-head` must be
passed exactly for the fixtures that bind `model.head.weight`
(`run-tiny-logits`), and omitted for the rest.
"""

import argparse
import json
import os
import struct

BASE = [
    ("model.embed.weight", [8, 4], lambda o: o + 1),
    ("model.t0.up.weight", [12, 4], lambda o: o + 1),
    ("model.t0.down.weight", [4, 12], lambda o: o + 1),
    ("model.t1.up.weight", [12, 4], lambda o: o + 8),
    ("model.t1.down.weight", [4, 12], lambda o: o + 3),
]
HEAD = ("model.head.weight", [8, 4], lambda o: o + 5)


def bf16(value: float) -> bytes:
    """Round an f32 to bf16 (nearest even on the dropped 16 bits)."""
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    lower = bits & 0xFFFF
    upper = bits >> 16
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper = (upper + 1) & 0xFFFF
    return struct.pack("<H", upper)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="checkpoint directory to write")
    parser.add_argument(
        "--with-head",
        action="store_true",
        help="also write model.head.weight (for run-tiny-logits)",
    )
    args = parser.parse_args()

    tensors = list(BASE) + ([HEAD] if args.with_head else [])
    payload = b""
    header = {}
    for name, shape, value in tensors:
        count = shape[0] * shape[1]
        start = len(payload)
        payload += b"".join(bf16(value(o)) for o in range(count))
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header).encode()
    shard = struct.pack("<Q", len(header_bytes)) + header_bytes + payload

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "model.safetensors"), "wb") as handle:
        handle.write(shard)
    index = {
        "metadata": {"total_size": len(payload)},
        "weight_map": {name: "model.safetensors" for name, _, _ in tensors},
    }
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as handle:
        json.dump(index, handle, indent=1)
    print(f"wrote {args.out}: {len(tensors)} tensor(s), {len(shard)} bytes")


if __name__ == "__main__":
    main()
