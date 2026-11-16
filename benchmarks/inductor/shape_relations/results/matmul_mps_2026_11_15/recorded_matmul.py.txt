"""MPS profitability study for symbolic/guarded batch-linear fusion.

Compare the original pass, proof-only symbolic support, bounded guards, and a
larger guard budget. Counts describe wrapper calls, not physical GPU launches.
"""

import argparse
import ast
import hashlib
import json
import random
import statistics
import subprocess
import time
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from runtime import environment, ROOT

import torch
import torch.nn.functional as F
from torch._inductor import config
from torch._inductor.fx_passes import group_batch_fusion as gb
from torch._inductor.graph import GraphLowering


ORIGINAL_BATCH_COMMIT = "986da9d7deb1b44db0e6dbd7e938d7636fb26870"


def shared(x, *weights):
    return tuple(F.linear(x, w) for w in weights)


def independent(*args):
    return tuple(F.linear(args[i], args[i + 1]) for i in range(0, len(args), 2))


def original_rule():
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{ORIGINAL_BATCH_COMMIT}:torch/_inductor/fx_passes/group_batch_fusion.py",
        ],
        cwd=ROOT,
        text=True,
    )
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "PostGradBatchLinearFusion"
    )
    node.decorator_list = []
    namespace = dict(vars(gb))
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), "<original-batch-linear>", "exec"
        ),
        namespace,
    )
    return namespace["PostGradBatchLinearFusion"], hashlib.sha256(
        source.encode()
    ).hexdigest()


def time_block(fn, inputs, repeats, sync_each):
    torch.mps.synchronize()
    start = time.perf_counter_ns()
    for _ in range(repeats):
        result = fn(*inputs)
        if sync_each:
            torch.mps.synchronize()
    torch.mps.synchronize()
    elapsed = (time.perf_counter_ns() - start) / repeats / 1000
    del result
    return elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expected-build-commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--rounds", type=int, default=16)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.rounds < 4 or args.rounds % 4:
        raise ValueError("Use rounds divisible by four")
    info = environment("mps", args.expected_build_commit)
    old_cls, old_hash = original_rule()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "code").mkdir(exist_ok=True)
    report = {
        "environment": info,
        "original_batch_commit": ORIGINAL_BATCH_COMMIT,
        "original_batch_source_sha256": old_hash,
        "fixed_batch_source_sha256": hashlib.sha256(
            Path(gb.__file__).read_bytes()
        ).hexdigest(),
        "dtype": "float32",
        "training": False,
        "rounds": args.rounds,
        "seed": args.seed,
        "cases": [],
    }
    cases = [
        ("shared-small", 64, 32, 48, False, True),
        ("independent-small", 64, 32, 48, False, False),
        ("independent-medium", 128, 64, 96, False, False),
        ("independent-large", 256, 256, 256, False, False),
        ("independent-skinny", 16, 1024, 16, False, False),
        ("independent-unequal", 64, 32, 48, True, False),
    ]
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    for label, m, k, n, unequal, is_shared in cases:
        torch._dynamo.reset()
        torch.manual_seed(42)
        if is_shared:
            inputs = (
                torch.randn(m, k, device="mps"),
                *(torch.randn(n, k, device="mps") for _ in range(5)),
            )
            torch._dynamo.mark_dynamic(inputs[0], 0)
            fn = shared
        else:
            inputs = tuple(
                t
                for i in range(5)
                for t in (
                    torch.randn(m, k, device="mps"),
                    torch.randn(n + (8 if unequal and i == 4 else 0), k, device="mps"),
                )
            )
            for t in inputs:
                torch._dynamo.mark_dynamic(t, 0)
                torch._dynamo.mark_dynamic(t, 1)
            fn = independent
        ref = fn(*inputs)
        compiled = {}
        records = {}
        modes = ["original", "known", "guarded", "large_budget"]
        compile_order = modes.copy()
        rng.shuffle(compile_order)
        for mode in compile_order:
            options = {}
            if mode in ("guarded", "large_budget"):
                options["guarded_batching"] = True
            if mode == "large_budget":
                options["max_guarded_working_set_bytes"] = 64 * 1024 * 1024
            cloned = types.FunctionType(
                fn.__code__.replace(co_filename=f"<matmul-{label}-{mode}>"),
                fn.__globals__,
            )
            context = (
                patch.dict(gb.POST_GRAD_FUSIONS, {"batch_linear_post_grad": old_cls})
                if mode == "original"
                else nullcontext()
            )
            codes = []
            torch.mps.synchronize()
            start = time.perf_counter()
            with (
                config.patch(
                    force_disable_caches=True,
                    post_grad_fusion_options={"batch_linear_post_grad": options},
                ),
                context,
                patch.object(GraphLowering, "save_output_code", codes.append),
            ):
                compiled[mode] = torch.compile(cloned, fullgraph=True)
                actual = compiled[mode](*inputs)
                torch.mps.synchronize()
            setup = time.perf_counter() - start
            torch.testing.assert_close(actual, ref, atol=2e-4, rtol=2e-4)
            records[mode] = {
                "setup_seconds": setup,
                "correctness": "passed",
                "mm_calls": sum(c.count("extern_kernels.mm(") for c in codes),
                "bmm_calls": sum(c.count("extern_kernels.bmm(") for c in codes),
                "metal_definitions": sum(
                    c.count("= async_compile.metal(") for c in codes
                ),
            }
            for i, code in enumerate(codes):
                (args.output.parent / "code" / f"{label}-{mode}-{i}.py").write_text(
                    code
                )
        for _ in range(32):
            for mode in compile_order:
                compiled[mode](*inputs)
        torch.mps.synchronize()
        orders = [modes[i:] + modes[:i] for i in range(4)] * (args.rounds // 4)
        rng.shuffle(orders)
        samples = []
        unexpected = []
        with patch.object(GraphLowering, "save_output_code", unexpected.append):
            for order in orders:
                sample = {"order": order}
                for mode in order:
                    sample[mode] = {
                        "queued_us": time_block(compiled[mode], inputs, 32, False),
                        "sync_us": time_block(compiled[mode], inputs, 8, True),
                    }
                samples.append(sample)
        if unexpected:
            raise RuntimeError("Compilation occurred inside timing")
        medians = {
            metric: {
                mode: statistics.median(s[mode][metric] for s in samples)
                for mode in modes
            }
            for metric in ("queued_us", "sync_us")
        }
        case = {
            "case": label,
            "mkn": [m, k, n],
            "unequal": unequal,
            "estimated_working_set_bytes": 5 * 4 * (m * k + k * n + m * n),
            "records": records,
            "samples": samples,
            "median_us": medians,
            "speedup": {
                metric: {mode: values["original"] / values[mode] for mode in modes}
                for metric, values in medians.items()
            },
        }
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in case.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
