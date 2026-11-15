"""Paired baseline/fixed GPU timings on the same native PyTorch build."""

import argparse
import ast
import hashlib
import json
import random
import statistics
import subprocess
import time
import types
from pathlib import Path
from unittest.mock import patch

from runtime import (
    empty_cache,
    environment,
    kernel_count,
    load_baseline,
    profile_cuda,
    sync,
)
from workloads import (
    layernorm_split,
    no_split,
    rmsnorm_split,
    softmax_split,
    stats_from_other,
)

import torch
from torch._inductor import config, sizevars
from torch._inductor.graph import GraphLowering


ROOT = Path(__file__).resolve().parents[3]
BASELINE_COMMIT = "c79633996ebc2114482ae5a4ff3cb409d79a8a9c"
MEASURED_COMMIT = "6345ea65549d76751f09501580074b700b7c56c6"


def original_simplifier(baseline_commit):
    source = subprocess.check_output(
        ["git", "show", f"{baseline_commit}:torch/_inductor/sizevars.py"],
        cwd=ROOT,
        text=True,
    )
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SizeVarAllocator"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "simplify"
    )
    ns = dict(vars(sizevars))
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), "<parent-sizevars>", "exec"), ns
    )
    return ns["simplify"], hashlib.sha256(ast.unparse(fn).encode()).hexdigest()


def clone_fn(fn, label):
    return types.FunctionType(
        fn.__code__.replace(co_filename=f"<bench-{label}>"), fn.__globals__, fn.__name__
    )


def specs():
    result = []
    for scale, t, c in [
        ("small", 32, 128),
        ("original", 512, 512),
        ("large", 2048, 2048),
    ]:
        for name, fn in [
            ("softmax", softmax_split),
            ("rmsnorm", rmsnorm_split),
            ("layernorm", layernorm_split),
        ]:
            result.append(
                {
                    "id": f"{name}-{scale}",
                    "kind": name,
                    "fn": fn,
                    "shapes": [(2, t, c)],
                    "scale": scale,
                }
            )
    for scale, m, d in [
        ("small", 128, 128),
        ("original", 4096, 512),
        ("large", 4096, 2048),
    ]:
        for unequal in (False, True):
            result.append(
                {
                    "id": f"equality-{'unequal' if unequal else 'equal'}-{scale}",
                    "kind": "equality",
                    "fn": stats_from_other,
                    "shapes": [(m, d, d * 3 // 2 if unequal else d)],
                    "scale": scale,
                }
            )
    result.append(
        {
            "id": "no-split-control",
            "kind": "control",
            "fn": no_split,
            "shapes": [(2, 512, 512)],
            "scale": "original",
        }
    )
    result.append(
        {
            "id": "equality-mixed-original",
            "kind": "equality",
            "fn": stats_from_other,
            "shapes": [
                (4096, 512, 512),
                (4096, 512, 768),
                (4096, 256, 256),
                (4096, 512, 768),
            ],
            "scale": "original",
        }
    )
    return result


def make_inputs(spec, device="mps", dtype=torch.float32):
    batches = []
    for shape in spec["shapes"]:
        if spec["kind"] == "equality":
            m, a, b = shape
            inputs = (
                torch.randn(m, a, device=device, dtype=dtype),
                torch.randn(m, b, device=device, dtype=dtype),
            )
            for t in inputs:
                torch._dynamo.mark_dynamic(t, 0)
                torch._dynamo.mark_dynamic(t, 1)
        else:
            inputs = (torch.randn(*shape, device=device, dtype=dtype),)
            torch._dynamo.mark_dynamic(inputs[0], 1)
            torch._dynamo.mark_dynamic(inputs[0], 2)
            if spec["kind"] == "rmsnorm":
                inputs += (torch.randn(shape[-1] // 8, device=device, dtype=dtype),)
        batches.append(inputs)
    return batches


def timed(compiled, batches, repeats, sync_each=False, device="mps"):
    sync(device)
    start = time.perf_counter_ns()
    for _ in range(repeats):
        for inputs in batches:
            output = compiled(*inputs)
            if sync_each:
                sync(device)
    sync(device)
    elapsed = time.perf_counter_ns() - start
    # Keep the last result alive until synchronization, like a real caller.
    del output
    return elapsed / (repeats * len(batches)) / 1000


def run_case(spec, args, original, rng):
    device = args.device
    torch._dynamo.reset()
    empty_cache(device)
    torch.manual_seed(12)
    inputs = make_inputs(spec, device, getattr(torch, args.dtype))
    refs = [spec["fn"](*i) for i in inputs]
    sync(device)
    records, compiled = {}, {}
    compile_order = ["baseline", "fixed"]
    rng.shuffle(compile_order)
    for mode in compile_order:
        record = {
            "graphs": 0,
            "setup_calls_s": [],
            "kernel_definitions_by_compile": [],
            "code_sha256": [],
        }
        records[mode] = record

        def backend(gm, example_inputs, mode=mode, record=record):
            record["graphs"] += 1
            with original.use(mode):
                return torch._inductor.compile(gm, example_inputs)

        compiled[mode] = torch.compile(
            clone_fn(spec["fn"], spec["id"] + "-" + mode),
            backend=backend,
            fullgraph=True,
            dynamic=None,
        )
        for i, ref in zip(inputs, refs):
            codes = []
            sync(device)
            start = time.perf_counter()
            with patch.object(GraphLowering, "save_output_code", codes.append):
                actual = compiled[mode](*i)
                sync(device)
            record["setup_calls_s"].append(time.perf_counter() - start)
            torch.testing.assert_close(
                actual, ref, rtol=args.tolerance, atol=args.tolerance
            )
            for j, code in enumerate(codes):
                record["kernel_definitions_by_compile"].append(
                    kernel_count(code, device)
                )
                record["code_sha256"].append(hashlib.sha256(code.encode()).hexdigest())
                (
                    args.output.parent
                    / "code"
                    / f"{spec['id']}-{mode}-{len(record['code_sha256'])}.py"
                ).write_text(code)
    for _ in range(32):
        for mode in compile_order:
            for i in inputs:
                compiled[mode](*i)
    sync(device)
    pilots = [timed(compiled[m], inputs, 8, device=device) for m in compile_order]
    largest = max(r.numel() * r.element_size() for r in refs)
    repeats = max(
        8,
        min(
            1024,
            int(80000 / (max(pilots) * len(inputs))),
            max(8, 1024**3 // (largest * len(inputs))),
        ),
    )
    orders = [["baseline", "fixed"], ["fixed", "baseline"]] * (args.rounds // 2)
    rng.shuffle(orders)
    pairs = []
    counts_before = {m: r["graphs"] for m, r in records.items()}
    for order in orders:
        pair = {"order": order}
        for mode in order:
            pair[mode] = {
                "queued_us": timed(compiled[mode], inputs, repeats, device=device),
                "sync_per_call_us": timed(
                    compiled[mode], inputs, 8, True, device=device
                ),
            }
        pairs.append(pair)
    for mode, record in records.items():
        if record["graphs"] != counts_before[mode]:
            raise RuntimeError("Recompilation inside timed region")
        record["correctness"] = "passed"
    if args.profile and device == "cuda":
        for mode in compile_order:
            try:
                records[mode]["cuda_device_events"] = profile_cuda(
                    lambda: [compiled[mode](*i) for i in inputs],
                    args.output.parent / f"{spec['id']}-{mode}-trace.json",
                )
            except RuntimeError as exc:
                records[mode]["profile_error"] = str(exc)
            if records[mode]["graphs"] != counts_before[mode]:
                raise RuntimeError(
                    "Profiling triggered a recompilation; discard the trace"
                )
    result = {k: v for k, v in spec.items() if k != "fn"}
    result.update(
        {
            "compile_order": compile_order,
            "repeats_per_queued_block": repeats,
            "records": records,
            "pairs": pairs,
        }
    )
    print(
        json.dumps(
            {
                "case": spec["id"],
                "graphs": counts_before,
                "kernels": {
                    m: r["kernel_definitions_by_compile"] for m, r in records.items()
                },
                "queued_us": {
                    m: statistics.median(p[m]["queued_us"] for p in pairs)
                    for m in records
                },
            }
        ),
        flush=True,
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--only", default="")
    ap.add_argument("--device", choices=("mps", "cuda"), default="mps")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Save separate warm CUDA traces outside timing",
    )
    ap.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    ap.add_argument(
        "--expected-build-commit",
        default=MEASURED_COMMIT,
        help="Expected torch.version.git_version; override for a rebuilt revision",
    )
    args = ap.parse_args()
    if args.rounds < 2 or args.rounds % 2:
        raise ValueError("Use a positive even number of rounds (at least 2)")
    device = args.device
    dtype = getattr(torch, args.dtype)
    args.tolerance = 0.02 if args.dtype == "bfloat16" else 1e-4
    info = environment(device, args.expected_build_commit, args.dtype)
    if args.output.exists():
        raise FileExistsError(f"Choose a new output path: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "code").mkdir(exist_ok=True)
    original = load_baseline(args.baseline_commit)
    original_hash = original.method_sha256["simplify"]
    rng = random.Random(args.seed)
    report = {
        "torch": torch.__version__,
        "commit": torch.version.git_version,
        "parent": subprocess.check_output(
            ["git", "rev-parse", args.baseline_commit], cwd=ROOT, text=True
        ).strip(),
        "platform": info["platform"],
        "cpu": info["cpu"],
        "battery": info["battery"],
        "device": device,
        "environment": info,
        "baseline_metadata": original.metadata(),
        "baseline": "all modified SizeVarAllocator methods restored + equality pass disabled; same native build",
        "original_simplify_sha256": original_hash,
        "dtype": args.dtype,
        "mode": "inference",
        "seed": args.seed,
        "rounds": args.rounds,
        "timer": f"perf_counter_ns, {device} synchronization around every block",
        "cases": [],
    }
    with config.patch(force_disable_caches=True, **{"triton.cudagraphs": False}):
        # Remove one-time backend initialization from per-case setup measurements.
        warm = torch.compile(lambda x: x.sin() + 1, fullgraph=True)
        warm(torch.randn(128, device=device, dtype=dtype))
        sync(device)
        cases = specs()
        if args.only:
            requested = args.only.split(",")
            unknown = [
                name for name in requested if name not in [c["id"] for c in cases]
            ]
            if unknown:
                raise ValueError(f"Unknown cases: {unknown}")
            cases = [c for c in cases if c["id"] in requested]
        rng.shuffle(cases)
        with torch.inference_mode():
            for case in cases:
                report["cases"].append(run_case(case, args, original, rng))
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", len(report["cases"]), flush=True)
    if any(
        "profile_error" in record
        for case in report["cases"]
        for record in case["records"].values()
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
