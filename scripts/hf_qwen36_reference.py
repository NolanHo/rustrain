#!/usr/bin/env python3
"""The HuggingFace side of D5's numeric alignment, and the comparison that judges it.

Two modes, one tolerance table:

    # on the machine that has transformers + the weights (the verification host)
    python3 scripts/hf_qwen36_reference.py dump --model Qwen/Qwen3.6-35B-A3B --out /var/tmp/hf-ref.npz

    # anywhere, once rustrain has dumped its own forward with the same names
    python3 scripts/hf_qwen36_reference.py compare --reference /var/tmp/hf-ref.npz \
        --candidate /var/tmp/rustrain.npz

`dump` runs **one** forward pass with `output_hidden_states=True` and writes:

    input_ids         int64  [seq]      the fixed probe tokens below
    logits            f32    [seq, vocab]
    hidden_summaries  f32    [L+2, 3]   per hidden state: mean, std, max

`compare` reads two such files and applies the acceptance from
`docs/design/qwen36-text/spec.md` D5:

    logits : max_abs_diff / max_abs(reference) < 1%
    hidden : per-layer max over {mean, std, max} of |cand - ref| / max(|ref|, eps) < 1%

It prints the first layer that goes out of tolerance (that is what locates the
first divergence) and exits non-zero when any layer or the logits fail. `--json`
emits the same as one machine-readable object.

Why a fixed token list instead of a tokenizer call: the probe must be the *same
numbers* on both sides, and it must survive a tokenizer revision. The list below
is written out literally, so it is the same input for HF, for rustrain, and for
anyone re-running this a year from now. Change it only by changing it here.

This file has not been executed on the CPU box (no torch there, by design); it
runs where the weights are. Treat its own first run as part of the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The probe: `SEQ` fixed tokens. Kept small — the acceptance is about numerics, not throughput, and
# a short sequence keeps the reference forward cheap on a shared GPU.
PROBE_TOKENS = [
    9707,
    11,
    1879,
    0,
    323,
    358,
    314,
    279,
]
SEQ = len(PROBE_TOKENS)

# D5's tolerance: a one-percent relative gap, on the logits and on every hidden state's summary.
TOLERANCE = 0.01


def dump(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    tokens = np.array(PROBE_TOKENS, dtype=np.int64)
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        revision=args.revision,
    )
    model.eval()

    device = next(model.parameters()).device
    ids = torch.from_numpy(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)

    logits = out.logits[0].to(torch.float32).cpu().numpy()
    # `hidden_states` is (embedding output, layer 1, ..., final norm) — num_hidden_layers + 2
    # entries (42 for the 40-layer Qwen3.6-35B-A3B). The rustrain side must declare the same set:
    # `outputs.hidden = ["embed.y", "layers.*.y", "norm.y"]` in the description's `outputs` section.
    summaries = np.stack(
        [
            np.array(
                [
                    float(h.mean().to(torch.float32)),
                    float(h.std().to(torch.float32)),
                    float(h.abs().max().to(torch.float32)),
                ],
                dtype=np.float32,
            )
            for h in out.hidden_states
        ]
    )

    out_path = Path(args.out)
    np.savez(
        out_path,
        input_ids=tokens,
        logits=logits,
        hidden_summaries=summaries,
    )
    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "model": args.model,
                "revision": args.revision,
                "dtype": args.dtype,
                "seq": SEQ,
                "probe_tokens": PROBE_TOKENS,
                "hidden_states": int(summaries.shape[0]),
                "transformers": __import__("transformers").__version__,
                "torch": torch.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out_path} ({summaries.shape[0]} hidden states, logits {logits.shape})")
    print(f"wrote {sidecar}")
    return 0


def _relative(gap: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """`|gap| / max(|scale|, eps)`, elementwise, with a scale that cannot be zero."""
    return np.abs(gap) / np.maximum(np.abs(scale), np.finfo(np.float32).tiny)


def compare(args: argparse.Namespace) -> int:
    ref = np.load(args.reference)
    cand = np.load(args.candidate)

    report: dict[str, object] = {"tolerance": TOLERANCE, "failures": []}

    if not np.array_equal(ref["input_ids"], cand["input_ids"]):
        report["failures"].append(
            {
                "what": "input_ids",
                "detail": "the two dumps did not run the same probe tokens; nothing else is comparable",
            }
        )
        _emit(report, args)
        return 1

    ref_logits = ref["logits"].astype(np.float32)
    cand_logits = cand["logits"].astype(np.float32)
    if ref_logits.shape != cand_logits.shape:
        report["failures"].append(
            {
                "what": "logits shape",
                "detail": f"reference {ref_logits.shape} vs candidate {cand_logits.shape}",
            }
        )
        _emit(report, args)
        return 1

    logits_scale = float(np.abs(ref_logits).max())
    logits_gap = float(np.abs(ref_logits - cand_logits).max())
    logits_relative = logits_gap / max(logits_scale, np.finfo(np.float32).tiny)
    report["logits"] = {
        "max_abs_diff": logits_gap,
        "max_abs_reference": logits_scale,
        "relative": logits_relative,
        "ok": logits_relative < TOLERANCE,
    }
    if not report["logits"]["ok"]:  # type: ignore[index]
        report["failures"].append({"what": "logits", "detail": f"relative {logits_relative:.3e}"})

    ref_hidden = ref["hidden_summaries"].astype(np.float32)
    cand_hidden = cand["hidden_summaries"].astype(np.float32)
    stats = ["mean", "std", "max"]
    if ref_hidden.shape != cand_hidden.shape:
        report["failures"].append(
            {
                "what": "hidden summary shape",
                "detail": f"reference {ref_hidden.shape} vs candidate {cand_hidden.shape}",
            }
        )
        _emit(report, args)
        return 1

    relative = _relative(ref_hidden - cand_hidden, ref_hidden)  # [L+1, 3]
    per_layer = relative.max(axis=1)
    worst_stat = relative.argmax(axis=1)
    first_bad = next((i for i, value in enumerate(per_layer) if value >= TOLERANCE), None)
    report["hidden"] = {
        "layers": int(per_layer.shape[0]),
        "per_layer_relative_max": [float(v) for v in per_layer],
        "per_layer_worst_stat": [stats[i] for i in worst_stat],
        "first_out_of_tolerance": first_bad,
        "ok": first_bad is None,
    }
    if first_bad is not None:
        report["failures"].append(
            {
                "what": f"hidden layer {first_bad}",
                "detail": f"{stats[int(worst_stat[first_bad])]} relative {per_layer[first_bad]:.3e}",
            }
        )

    _emit(report, args)
    return 0 if not report["failures"] else 1


def _emit(report: dict[str, object], args: argparse.Namespace) -> None:
    if args.json:
        print(json.dumps(report, indent=2))
        return
    logits = report.get("logits")
    if isinstance(logits, dict):
        print(
            "logits: max_abs_diff {:.4e} / max_abs {:.4e} = {:.3e}  [{}]".format(
                logits["max_abs_diff"],
                logits["max_abs_reference"],
                logits["relative"],
                "ok" if logits["ok"] else "FAIL",
            )
        )
    hidden = report.get("hidden")
    if isinstance(hidden, dict):
        worst = max(hidden["per_layer_relative_max"]) if hidden["per_layer_relative_max"] else 0.0
        print(
            "hidden: {} layers, worst relative {:.3e}, first out of tolerance: {}".format(
                hidden["layers"], worst, hidden["first_out_of_tolerance"]
            )
        )
        for index, value in enumerate(hidden["per_layer_relative_max"]):
            flag = "  <-- first divergence" if index == hidden["first_out_of_tolerance"] else ""
            print(
                "  layer {:3d}  {:.3e}  (worst: {}){}".format(
                    index, value, hidden["per_layer_worst_stat"][index], flag
                )
            )
    for failure in report["failures"]:  # type: ignore[union-attr]
        print(f"FAIL {failure['what']}: {failure['detail']}", file=sys.stderr)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    dump_parser = sub.add_parser("dump", help="run the HF reference forward and save it")
    dump_parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    dump_parser.add_argument("--revision", default=None)
    dump_parser.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"])
    dump_parser.add_argument(
        "--device-map",
        default="auto",
        help="passed to from_pretrained; 'auto' spreads a 70 GB bf16 model over the free GPUs",
    )
    dump_parser.add_argument("--out", default="/var/tmp/hf-ref.npz")
    dump_parser.set_defaults(func=dump)

    compare_parser = sub.add_parser("compare", help="check a rustrain dump against a reference")
    compare_parser.add_argument("--reference", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--json", action="store_true")
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
