"""Paired MPS timings for generalized reduction shape guards."""

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
import types
from pathlib import Path
from unittest.mock import patch

from runtime import environment, kernel_count, ROOT

import torch
from torch._inductor import config
from torch._inductor.fx_passes import shape_relation_fusion
from torch._inductor.graph import GraphLowering


BASELINE_COMMIT = "01d544c8af54330e035a2eaf38554b7efef3a274"


def original_pass():
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{BASELINE_COMMIT}:torch/_inductor/fx_passes/shape_relation_fusion.py",
        ],
        cwd=ROOT,
        text=True,
    )
    namespace = {"__package__": "torch._inductor.fx_passes"}
    exec(compile(source, "<baseline-shape-relations>", "exec"), namespace)
    return namespace["guard_reduction_width_equality"], source


def workload(x, y, kind, dims):
    if kind in ("var", "std"):
        reduced = getattr(x, kind)(dim=dims, keepdim=True, correction=0)
    elif kind in ("max", "min"):
        reduced = getattr(x, kind)(dim=dims[-1], keepdim=True).values
    elif kind == "prod":
        reduced = x.prod(dim=dims[-1], keepdim=True)
    elif kind == "var_mean":
        variance, mean = torch.var_mean(x, dim=dims, keepdim=True, correction=0)
        reduced = variance + mean
    elif kind == "norm":
        reduced = torch.linalg.vector_norm(x, dim=dims, keepdim=True)
    else:
        reduced = getattr(x, kind)(dim=dims, keepdim=True)
    return y * (reduced + 0.1)


def specs():
    cases = []
    for kind in ("amax", "amin", "prod", "var", "std", "max", "min", "var_mean"):
        for unequal in (False, True):
            cases.append(
                dict(
                    id=f"{kind}-{'unequal' if unequal else 'equal'}-medium",
                    kind=kind,
                    dims=[-1],
                    shapes=[[(512, 256), (512, 384 if unequal else 256)]],
                )
            )
    for kind in ("amax", "var"):
        for scale, rows, width in (("small", 17, 64), ("large", 4096, 512)):
            for unequal in (False, True):
                cases.append(
                    dict(
                        id=f"{kind}-{'unequal' if unequal else 'equal'}-{scale}",
                        kind=kind,
                        dims=[-1],
                        shapes=[
                            [
                                (rows, width),
                                (rows, width * 3 // 2 if unequal else width),
                            ]
                        ],
                    )
                )
    for kind in ("amax", "var", "var_mean"):
        for unequal in (False, True):
            cases.append(
                dict(
                    id=f"{kind}-{'unequal' if unequal else 'equal'}-two-axis",
                    kind=kind,
                    dims=[-2, -1],
                    shapes=[[(512, 8, 32), (512, 16, 16) if unequal else (512, 8, 32)]],
                )
            )
    cases.append(
        dict(
            id="var-mixed-widths",
            kind="var",
            dims=[-1],
            shapes=[
                [(512, 256), (512, 256)],
                [(512, 256), (512, 384)],
                [(768, 128), (768, 128)],
                [(512, 192), (512, 288)],
                [(512, 256), (512, 256)],
            ],
        )
    )
    cases.append(
        dict(
            id="var-mixed-axes-unequal-first",
            kind="var",
            dims=[-2, -1],
            shapes=[
                [(512, 8, 32), (512, 16, 16)],
                [(512, 8, 32), (512, 8, 32)],
                [(768, 6, 24), (768, 8, 24)],
                [(768, 6, 24), (768, 6, 32)],
                [(768, 6, 24), (768, 6, 24)],
            ],
        )
    )
    for kind in ("mean", "norm"):
        cases.append(
            dict(
                id=f"{kind}-control",
                kind=kind,
                dims=[-1],
                shapes=[[(512, 256), (512, 256)]],
            )
        )
    return cases


def timed(fn, batches, repeats, sync_each):
    torch.mps.synchronize()
    start = time.perf_counter_ns()
    for _ in range(repeats):
        for inputs in batches:
            output = fn(*inputs)
            if sync_each:
                torch.mps.synchronize()
    torch.mps.synchronize()
    elapsed = (time.perf_counter_ns() - start) / (repeats * len(batches)) / 1000
    del output
    return elapsed


def run_case(spec, args, old_pass, rng):
    torch._dynamo.reset()
    torch.mps.empty_cache()
    torch.manual_seed(42)
    batches = []
    for pair in spec["shapes"]:
        inputs = tuple(0.99 + 0.02 * torch.rand(shape, device="mps") for shape in pair)
        for tensor in inputs:
            for dim in range(tensor.ndim):
                torch._dynamo.mark_dynamic(tensor, dim)
        batches.append(inputs)
    kind, dims = spec["kind"], tuple(spec["dims"])

    def fn(x, y):
        return workload(x, y, kind, dims)

    refs = [fn(*inputs) for inputs in batches]
    compiled, records = {}, {}
    modes = ["baseline", "fixed"]
    compile_order = rng.sample(modes, len(modes))
    for mode in compile_order:
        record = {"graphs": 0, "setup_calls_s": [], "code": [], "correctness": "passed"}
        records[mode] = record
        selected_pass = (
            old_pass
            if mode == "baseline"
            else shape_relation_fusion.guard_reduction_shape_equality
        )

        def backend(gm, inputs, record=record, selected_pass=selected_pass):
            record["graphs"] += 1
            with (
                config.patch(force_disable_caches=True, guarded_shape_fusion=True),
                patch.object(
                    shape_relation_fusion,
                    "guard_reduction_shape_equality",
                    selected_pass,
                ),
            ):
                return torch._inductor.compile(gm, inputs)

        cloned = types.FunctionType(
            fn.__code__.replace(co_filename=f"<reduction-{spec['id']}-{mode}>"),
            fn.__globals__,
            fn.__name__,
            fn.__defaults__,
            fn.__closure__,
        )
        compiled[mode] = torch.compile(cloned, backend=backend, fullgraph=True)
        for inputs, ref in zip(batches, refs):
            codes = []
            torch.mps.synchronize()
            start = time.perf_counter()
            with patch.object(GraphLowering, "save_output_code", codes.append):
                actual = compiled[mode](*inputs)
                torch.mps.synchronize()
            record["setup_calls_s"].append(time.perf_counter() - start)
            torch.testing.assert_close(actual, ref, atol=1e-4, rtol=1e-4)
            for code in codes:
                name = f"{spec['id']}-{mode}-{len(record['code'])}.py.txt"
                (args.output.parent / "code" / name).write_text(code)
                record["code"].append(
                    dict(
                        file=f"code/{name}",
                        sha256=hashlib.sha256(code.encode()).hexdigest(),
                        metal_definitions=kernel_count(code, "mps"),
                    )
                )
        expected_graphs = 2 if mode == "fixed" and len(batches) > 1 else 1
        if record["graphs"] != expected_graphs:
            raise RuntimeError(
                f"Unexpected graph count: {spec['id']} / {mode}: {record['graphs']}"
            )
    before = [records[m]["graphs"] for m in modes]
    unexpected = []
    with patch.object(GraphLowering, "save_output_code", unexpected.append):
        for _ in range(16):
            for mode in compile_order:
                for inputs in batches:
                    compiled[mode](*inputs)
        orders = [modes, modes[::-1]] * (args.rounds // 2)
        rng.shuffle(orders)
        pairs = []
        for order in orders:
            pair = {"order": order}
            for mode in order:
                pair[mode] = dict(
                    queued_us=timed(compiled[mode], batches, 32, False),
                    sync_us=timed(compiled[mode], batches, 8, True),
                )
            pairs.append(pair)
    if unexpected or before != [records[m]["graphs"] for m in modes]:
        raise RuntimeError(f"Recompilation during warmup or timing: {spec['id']}")
    medians = {
        metric: {
            mode: statistics.median(p[mode][metric] for p in pairs) for mode in modes
        }
        for metric in ("queued_us", "sync_us")
    }
    return dict(
        **spec,
        records=records,
        pairs=pairs,
        median_us=medians,
        speedup={metric: v["baseline"] / v["fixed"] for metric, v in medians.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-build-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument(
        "--only", help="Comma-separated case IDs for a functional smoke run"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.rounds < 2 or args.rounds % 2:
        raise ValueError("Use a positive even number of rounds, at least two")
    old_pass, source = original_pass()
    cases = specs()
    if args.only:
        selected = set(args.only.split(","))
        if selected - {c["id"] for c in cases}:
            raise ValueError("Unknown case ID")
        cases = [c for c in cases if c["id"] in selected]
    report = dict(
        environment=environment("mps", args.expected_build_commit),
        baseline_commit=BASELINE_COMMIT,
        baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dtype="float32",
        training=False,
        rounds=args.rounds,
        seed=args.seed,
        case_ids=sorted(c["id"] for c in cases),
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "code").mkdir(exist_ok=True)
    (args.output.parent / "baseline_shape_relation_fusion.py.txt").write_text(source)
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    with torch.no_grad():
        for spec in cases:
            case = run_case(spec, args, old_pass, rng)
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "id": case["id"],
                        "speedup": case["speedup"],
                        "graphs": {m: r["graphs"] for m, r in case["records"].items()},
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
