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
    hidden_values     f32    [L+2, seq, hidden]  the rows themselves

`compare` reads two such files and applies the acceptance from
`docs/design/qwen36-text/spec.md` D5/D6.11, **for a pair of dumps in one dtype**:

    logits  : max_abs_diff / max_abs(reference) < tolerance(dtype pair)
    hidden  : per-layer max over {mean, std, max} of a gap scaled by a layer scale that cannot
              vanish — |Δmean| and |Δmax| by the layer's max|x|, |Δstd| by the reference's std
              (`mean` alone is near zero, and dividing by it is dividing by noise)
    values  : per-row relative L2 when both dumps kept the rows

The tolerance is a property of the dtype pair, not of the implementation: HF's own bf16 forward
sits 1.5e-1 from its own f32 forward on the logits, so a bf16 dump against an f32 reference
reports the dtype. `compare` refuses a mixed pair — and a pair whose sidecar does not say —
unless `--allow-dtype-mismatch` is passed.

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

# D5's tolerance: a one-percent relative gap, on the logits and on every hidden state, **for a
# pair of dumps in the same dtype**. The f32 number is the one the spec states and the one the
# f32 path is held to; the bf16 number is derived from measurement, not from ambition:
#
#   pair            logits   worst hidden state   what the gap is
#   f32  vs f32     2.0e-3   1.7e-3 (per element) the implementation
#   bf16 vs bf16    8.8e-2   1.3e-1 (per element) the implementation *inside* the dtype's noise
#   HF bf16 vs f32  1.5e-1   1.6e-1 (per element) the dtype itself — no implementation involved
#
# The last row is why the pair matters: HF's own bf16 forward against its own f32 forward differs
# by more than the f32 tolerance, so a bf16 candidate judged against an f32 reference measures
# the dtype and can never fail for the right reason. `compare` therefore refuses a mixed pair
# (see `_dtype_of`) and applies the bound of the pair it was given.
TOLERANCE = {
    ("f32", "f32"): 0.01,
    ("bf16", "bf16"): 0.25,
}
# What applies when the pair's dtype is not established (a mixed pair, or a sidecar that does
# not say). `compare` refuses those outright — this bound is only for the runs that pass
# `--allow-dtype-mismatch` to see a number anyway.
TOLERANCE_UNKNOWN = 0.01


def _forward(model, ids, torch) -> tuple[object, list, str]:
    """One forward, returning `(logits, hidden states, where they came from)`.

    `output_hidden_states=True` is not enough on transformers 5.12.1: its `capture_outputs`
    hook collects embed + every layer, then `tie_last_hidden_states` overwrites the last
    entry with the final norm's output — yielding L+1 rows and losing layer L's own hidden
    state. The first dump taken that way had 41 rows against the description's 42, and
    `compare` refuses mismatched shapes rather than silently dropping the layer it cannot
    name (row 0 is the embedding, rows 1..L the layers, the last row the norm).

    So the sequence is captured explicitly with forward hooks on the three module groups
    the declaration names. The fallback keeps the run honest when a model's module layout
    does not expose them: it uses whatever `output_hidden_states` returned and says so in
    the sidecar, rather than pretending the rows mean the same thing.
    """
    captured: list = []
    handles: list = []
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    embed = getattr(inner, "embed_tokens", None) if inner is not None else None
    norm = getattr(inner, "norm", None) if inner is not None else None
    expected = len(layers) + 2 if layers is not None else 0
    if expected and embed is not None and norm is not None:

        def capture(_module, _inputs, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        handles.append(embed.register_forward_hook(capture))
        for layer in layers:
            handles.append(layer.register_forward_hook(capture))
        handles.append(norm.register_forward_hook(capture))

    try:
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if len(captured) == expected:
        return out.logits, list(captured), "forward_hooks"
    return out.logits, list(out.hidden_states), "output_hidden_states"


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
    logits, hidden_states, hidden_source = _forward(model, ids, torch)
    logits = logits[0].to(torch.float32).cpu().numpy()
    # The summaries are taken on an f32 view of each hidden state: the tolerance is meant to
    # absorb bf16's *value* rounding (which is what the rustrain side widens away), not the
    # reduction-order noise of summing 16k bf16 values, which is larger. The population
    # standard deviation (correction=0) is what the rustrain side computes.
    summaries = np.stack(
        [
            np.array(
                [
                    float(h.to(torch.float32).mean()),
                    float(h.to(torch.float32).std(correction=0)),
                    float(h.to(torch.float32).abs().max()),
                ],
                dtype=np.float32,
            )
            for h in hidden_states
        ]
    )

    # The probe rows of every hidden state, so the comparison can be made per
    # element instead of three statistics per layer: the summaries say a layer
    # differs, these say where and by how much.
    hidden_values = np.stack(
        [h[0, :SEQ, :].to(torch.float32).cpu().numpy() for h in hidden_states]
    )

    out_path = Path(args.out)
    np.savez(
        out_path,
        input_ids=tokens,
        logits=logits,
        hidden_summaries=summaries,
        hidden_values=hidden_values,
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
                "hidden_rows": "row 0 = embed.y, rows 1..L = layers.<i>.y, last row = norm.y",
                "hidden_source": hidden_source,
                "summary_convention": "f32 view, population std (correction=0), max = max|x|",
                "hidden_values": "probe rows, [hidden states, seq, hidden] in f32",
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


def _dtype_of(path: Path) -> str | None:
    """The dtype a dump declares in its sidecar, or `None` when it does not say.

    `dump` always writes one; a `rustrain` dump writes the machine-readable `dtype` token and
    a prose `precision` field beside it. An older dump has only the prose — hence `None`
    rather than a guess: a guessed dtype is how a bf16 candidate came to be judged against an
    f32 reference at a one-percent bound.
    """
    sidecar = path.with_name(path.name + ".json")
    if not sidecar.is_file():
        return None
    try:
        value = json.loads(sidecar.read_text()).get("dtype")
    except (OSError, ValueError):
        return None
    return value if isinstance(value, str) else None


def compare(args: argparse.Namespace) -> int:
    ref = np.load(args.reference)
    cand = np.load(args.candidate)
    ref_dtype = _dtype_of(Path(args.reference))
    cand_dtype = _dtype_of(Path(args.candidate))

    report: dict[str, object] = {
        "reference_dtype": ref_dtype,
        "candidate_dtype": cand_dtype,
        "failures": [],
    }

    # The pair decides the tolerance, and a pair whose dtype is not established has no
    # tolerance at all: falling back to the f32 bound on a bf16/bf16 pair would report a
    # failure (our bf16 sits at 8.8e-2) for the crime of an old sidecar. So a mix, or a
    # silence, stops the comparison unless the caller says it wants the number anyway.
    silent = [
        name
        for name, dtype in (("reference", ref_dtype), ("candidate", cand_dtype))
        if dtype is None
    ]
    mismatch = ref_dtype is not None and cand_dtype is not None and ref_dtype != cand_dtype
    if mismatch or silent:
        detail = (
            f"reference is {ref_dtype or 'silent'}, candidate is {cand_dtype or 'silent'}: "
        ) + (
            "a mixed pair measures the dtype, not the implementation (HF's own bf16 forward "
            "is 1.5e-1 from its own f32 forward on the logits)"
            if mismatch
            else f"the {', '.join(silent)} dump's sidecar declares no `dtype`, so the pair's "
            "bound is unknown"
        )
        report["failures"].append(
            {
                "what": "dtype pair",
                "detail": detail
                + ". Dump both sides in one dtype (each sidecar carries a `dtype` token), or "
                "pass --allow-dtype-mismatch to see the number anyway",
            }
        )
        if not args.allow_dtype_mismatch:
            _emit(report, args)
            return 1
    tolerance = None if mismatch or silent else TOLERANCE.get((ref_dtype, cand_dtype))  # type: ignore[arg-type]
    report["tolerance"] = tolerance if tolerance is not None else TOLERANCE_UNKNOWN
    if tolerance is None:
        tolerance = TOLERANCE_UNKNOWN
        report["tolerance_source"] = (
            f"the strictest bound ({TOLERANCE_UNKNOWN}) applies: the pair's dtype is mixed, "
            "silent or outside the table"
        )

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
        "ok": logits_relative < tolerance,
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

    # Every statistic is scaled by a **layer scale that cannot vanish**. The mean of a residual
    # stream is a near-zero quantity, so `|Δmean| / |mean|` divides by noise: it reported 2.6e-2
    # on a layer where the f32 candidate matches HF to 4e-4 — the layer's max|x| is the scale
    # that means something, and `std` is compared against the reference's own `std`.
    layer_scale = np.maximum(np.abs(ref_hidden[:, 2:3]), np.finfo(np.float32).tiny)
    relative = np.abs(ref_hidden - cand_hidden) / layer_scale
    relative[:, 1] = np.abs(ref_hidden[:, 1] - cand_hidden[:, 1]) / np.maximum(
        np.abs(ref_hidden[:, 1]), np.finfo(np.float32).tiny
    )
    per_layer = relative.max(axis=1)
    worst_stat = relative.argmax(axis=1)
    first_bad = next((i for i, value in enumerate(per_layer) if value >= tolerance), None)
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

    # The rows themselves, when both sides dumped them: three statistics per layer can agree
    # while the tensors do not, and a per-element relative L2 is the only first-order measure
    # in the table above.
    if "hidden_values" in ref.files and "hidden_values" in cand.files:  # type: ignore[operator]
        ref_values = ref["hidden_values"].astype(np.float64)
        cand_values = cand["hidden_values"].astype(np.float64)
        if ref_values.shape == cand_values.shape:
            axes = tuple(range(1, ref_values.ndim))
            norm = np.maximum(
                np.linalg.norm(ref_values, axis=axes), np.finfo(np.float64).tiny
            )
            values_relative = np.linalg.norm(cand_values - ref_values, axis=axes) / norm
            values_bad = next(
                (i for i, value in enumerate(values_relative) if value >= tolerance), None
            )
            report["hidden_values"] = {
                "rows": int(values_relative.shape[0]),
                "per_row_relative_l2": [float(v) for v in values_relative],
                "first_out_of_tolerance": values_bad,
                "ok": values_bad is None,
            }
            if values_bad is not None:
                report["failures"].append(
                    {
                        "what": f"hidden values row {values_bad}",
                        "detail": f"relative L2 {values_relative[values_bad]:.3e}",
                    }
                )
        else:
            # A check that silently does not run is worse than no check: `conformance.rs`'s own
            # rule is that a skipped case must name what is missing, and a shape disagreement
            # between two dumps of the same forward is a defect in its own right.
            report["hidden_values"] = {
                "ok": False,
                "skipped": "the two dumps kept different row shapes",
            }
            report["failures"].append(
                {
                    "what": "hidden values shape",
                    "detail": f"reference {ref_values.shape} vs candidate {cand_values.shape}: "
                    "the per-element check did not run",
                }
            )

    _emit(report, args)
    return 0 if not report["failures"] else 1


def _emit(report: dict[str, object], args: argparse.Namespace) -> None:
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(
        "pair: reference {} vs candidate {}  tolerance {}".format(
            report.get("reference_dtype") or "(sidecar silent)",
            report.get("candidate_dtype") or "(sidecar silent)",
            report.get("tolerance", "not applied — the comparison stopped first"),
        )
    )
    if "tolerance_source" in report:
        print(f"  tolerance source: {report['tolerance_source']}")
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
    values = report.get("hidden_values")
    if isinstance(values, dict):
        if "per_row_relative_l2" not in values:
            print(f"hidden values: did not run — {values.get('skipped', 'unknown reason')}")
        else:
            worst = (
                max(values["per_row_relative_l2"]) if values["per_row_relative_l2"] else 0.0
            )
            print(
                "hidden values: {} rows, worst relative L2 {:.3e}, first out of tolerance: "
                "{}".format(
                    values["rows"], worst, values["first_out_of_tolerance"]
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
    compare_parser.add_argument(
        "--allow-dtype-mismatch",
        action="store_true",
        help="report a mixed-dtype pair instead of refusing it; the number is the dtype's "
        "spread, not the implementation's error",
    )
    compare_parser.add_argument("--json", action="store_true")
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
