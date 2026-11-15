"""Original XSA static/dynamic, baseline/fixed comparisons on CUDA or MPS.

Exit 0: all checks pass. Exit 1: compilation/numerical regression. Exit 2:
fusion-count mismatch or a baseline-preserved BF16 eager-gradient discrepancy.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
import traceback
import types
from pathlib import Path
from unittest.mock import patch

from bench import timed
from runtime import (
    BASELINE_COMMIT,
    environment,
    kernel_count,
    load_baseline,
    profile_cuda,
    sync,
)
from workloads import xsa_expand, xsa_unsqueeze

import torch
from torch._dynamo.utils import counters
from torch._inductor.graph import GraphLowering


def compare(values, refs, tolerance):
    result = []
    for actual, reference in zip(values, refs, strict=True):
        a, b = actual.detach().float(), reference.detach().float()
        close = torch.isclose(a, b, rtol=tolerance, atol=tolerance)
        diff = (a - b).abs()
        result.append(
            {
                "passed": bool(close.all().item()),
                "outside_tolerance": int((~close).sum().item()),
                "elements": a.numel(),
                "max_abs_error": float(diff.max().item()),
                "rmse": float(diff.square().mean().sqrt().item()),
            }
        )
    return result


def prepare(fn, name, dynamic, mode, args, baseline):
    torch.manual_seed(1234)
    dtype = getattr(torch, args.dtype)
    training = not args.inference
    x = torch.randn(
        1, 2048, 32, 64, device=args.device, dtype=dtype, requires_grad=training
    )
    v = torch.randn(
        1, 2048, 4, 64, device=args.device, dtype=dtype, requires_grad=training
    )
    weights = (
        torch.randn_like(x),
        torch.randn(1, 2048, 4, 8, device=args.device, dtype=dtype),
        torch.randn(1, 2048, 4, 8, device=args.device, dtype=dtype),
    )
    if dynamic:
        for tensor, dim in [
            (x, 2),
            (x, 3),
            (v, 3),
            (weights[0], 2),
            (weights[0], 3),
            (weights[1], 3),
            (weights[2], 3),
        ]:
            torch._dynamo.mark_dynamic(tensor, dim)

    def fwd_bwd():
        values = fn(x, v)
        if training:
            torch.autograd.backward(list(values), list(weights))
        return values

    # Distinct code identities keep all four compiled cases alive without resets.
    fwd_bwd = types.FunctionType(
        fwd_bwd.__code__.replace(co_filename=f"<xsa-{name}-{mode}-{dynamic}>"),
        fwd_bwd.__globals__,
        fwd_bwd.__name__,
        fwd_bwd.__defaults__,
        fwd_bwd.__closure__,
    )
    compiled = torch.compile(fwd_bwd)

    def step():
        x.grad = v.grad = None
        return compiled()

    label = f"{name}-{'dynamic' if dynamic else 'static'}-{mode}"
    codes = []
    sync(args.device)
    start = time.perf_counter()
    with (
        baseline.use(mode),
        patch.object(GraphLowering, "save_output_code", codes.append),
    ):
        outputs = step()
        sync(args.device)
    setup = time.perf_counter() - start
    values = [out.detach().clone() for out in outputs]
    if training:
        values += [x.grad.detach().clone(), v.grad.detach().clone()]
    refs = [t.detach().clone().requires_grad_(training) for t in (x, v)]
    expected = fn(*refs)
    output_checks = compare(values[:3], expected, args.tolerance)
    grad_checks = []
    if training:
        grad_checks = compare(
            values[3:], torch.autograd.grad(expected, refs, weights), args.tolerance
        )
    record = {
        "variant": name,
        "shapes": "dynamic" if dynamic else "static",
        "compiler": mode,
        "setup_seconds": setup,
        "kernel_definitions_by_graph": [
            kernel_count(code, args.device) for code in codes
        ],
        "against_eager": {"outputs": output_checks, "gradients": grad_checks},
        "code_sha256": [hashlib.sha256(code.encode()).hexdigest() for code in codes],
    }
    for index, code in enumerate(codes):
        (args.output.parent / "code" / f"{label}-{index}.py").write_text(code)
    return record, values, step


def measure_pair(steps, args, baseline):
    for mode, step in steps.items():
        with baseline.use(mode):
            for _ in range(16):
                step()
        sync(args.device)
    records = []
    orderings = [["baseline", "fixed"], ["fixed", "baseline"]] * (args.rounds // 2)
    random.Random(args.seed).shuffle(orderings)
    codes = []
    before = counters["stats"]["unique_graphs"]
    with patch.object(GraphLowering, "save_output_code", codes.append):
        for ordering in orderings:
            pair = {"order": ordering}
            for mode in ordering:
                call = steps[mode]
                pair[mode] = {
                    "queued_us": timed(call, [()], args.repeats, device=args.device),
                    "sync_per_call_us": timed(call, [()], 8, True, device=args.device),
                }
            records.append(pair)
    if codes or counters["stats"]["unique_graphs"] != before:
        raise RuntimeError("Recompilation during XSA timings; discard measurements")
    medians = {
        metric: {
            mode: statistics.median(pair[mode][metric] for pair in records)
            for mode in ("baseline", "fixed")
        }
        for metric in ("queued_us", "sync_per_call_us")
    }
    return {
        "pairs": records,
        "median_us": medians,
        "speedup": {
            metric: values["baseline"] / values["fixed"]
            for metric, values in medians.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    ap.add_argument("--expected-build-commit", required=True)
    ap.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--inference", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.rounds < 2 or args.rounds % 2 or args.repeats < 1:
        raise ValueError("Use positive repeats and positive even rounds")
    args.tolerance = 0.02 if args.dtype == "bfloat16" else 2e-4
    info = environment(args.device, args.expected_build_commit, args.dtype)
    baseline = load_baseline(args.baseline_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "code").mkdir(exist_ok=True)
    report = {
        "environment": info,
        "baseline": baseline.metadata(),
        "dtype": args.dtype,
        "training": not args.inference,
        "shape": {"B": 1, "T": 2048, "HQ": 32, "HKV": 4, "D": 64},
        "repro_commit": "275f764ee84812a8db53ff93e915577f9b97b2bc",
        "tolerance": {"rtol": args.tolerance, "atol": args.tolerance},
        "cases": [],
        "timings": {},
        "status": "running",
    }
    hard_failure = False
    review = False
    for name, fn in [("unsqueeze", xsa_unsqueeze), ("expand", xsa_expand)]:
        torch._dynamo.reset()
        values, records, steps = {}, {}, {}
        variant_bad = False
        for mode in ("baseline", "fixed"):
            for dynamic in (False, True):
                key = (mode, dynamic)
                try:
                    record, value, step = prepare(
                        fn, name, dynamic, mode, args, baseline
                    )
                    values[key], records[key] = value, record
                    if dynamic:
                        steps[mode] = step
                    checks = record["against_eager"]
                    if not all(c["passed"] for c in checks["outputs"]):
                        hard_failure = variant_bad = True
                    if not all(c["passed"] for c in checks["gradients"]):
                        if args.dtype == "float32":
                            hard_failure = variant_bad = True
                        else:
                            review = variant_bad = True
                    if mode == "fixed" and ("baseline", dynamic) in values:
                        comparison = compare(
                            value, values[("baseline", dynamic)], args.tolerance
                        )
                        record["against_baseline"] = comparison
                        if not all(c["passed"] for c in comparison):
                            hard_failure = variant_bad = True
                except Exception as exc:
                    hard_failure = variant_bad = True
                    record = {
                        "variant": name,
                        "shapes": "dynamic" if dynamic else "static",
                        "compiler": mode,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                report["cases"].append(record)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in record.items()
                            if k not in ("traceback", "code_sha256")
                        }
                    ),
                    flush=True,
                )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        if ("fixed", False) in records and ("fixed", True) in records:
            matches = (
                records["fixed", False]["kernel_definitions_by_graph"]
                == records["fixed", True]["kernel_definitions_by_graph"]
            )
            report.setdefault("dynamic_matches_static", {})[name] = matches
            if not matches:
                review = True
        if args.benchmark:
            if variant_bad or len(steps) != 2:
                report["timings"][name] = {
                    "skipped": "Resolve correctness/compilation checks before timing"
                }
            else:
                try:
                    report["timings"][name] = measure_pair(steps, args, baseline)
                except Exception as exc:
                    hard_failure = True
                    report["timings"][name] = {
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
        if args.profile and args.device == "cuda" and not variant_bad:
            for mode, step in steps.items():
                try:
                    report.setdefault("cuda_profiles", {})[f"{name}-{mode}"] = (
                        profile_cuda(
                            step, args.output.parent / f"{name}-{mode}-trace.json"
                        )
                    )
                except RuntimeError as exc:
                    review = True
                    report.setdefault("cuda_profiles", {})[f"{name}-{mode}"] = {
                        "error": str(exc)
                    }
        report["status"] = (
            "failed" if hard_failure else "needs_review" if review else "passed"
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("STATUS", report["status"], flush=True)
    raise SystemExit(1 if hard_failure else 2 if review else 0)


if __name__ == "__main__":
    main()
