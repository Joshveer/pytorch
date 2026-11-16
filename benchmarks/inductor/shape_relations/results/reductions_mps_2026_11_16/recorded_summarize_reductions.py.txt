"""Validate and summarize two paired reduction benchmark processes."""

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def quantile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    lo, hi = int(at), min(int(at) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def summarize(root):
    runs = [json.loads((root / f"run{i}" / "results.json").read_text()) for i in (1, 2)]
    for key in (
        "baseline_commit",
        "baseline_source_sha256",
        "harness_sha256",
        "dtype",
        "training",
        "rounds",
        "case_ids",
    ):
        if runs[0][key] != runs[1][key]:
            raise ValueError(f"Runs differ in {key}")
    for key in (
        "torch",
        "commit",
        "platform",
        "device",
        "cpu",
        "native_extension_sha256",
        "compiler_source_sha256",
        "build_config",
        "deterministic_algorithms",
    ):
        if runs[0]["environment"][key] != runs[1]["environment"][key]:
            raise ValueError(f"Run environments differ in {key}")
    if runs[0]["seed"] == runs[1]["seed"]:
        raise ValueError("Use different timing-order seeds")
    indexed = [{c["id"]: c for c in r["cases"]} for r in runs]
    for i, (run, cases) in enumerate(zip(runs, indexed), 1):
        if sorted(cases) != run["case_ids"] or len(cases) != len(run["cases"]):
            raise ValueError("Incomplete or duplicated cases")
        source = root / f"run{i}" / "baseline_shape_relation_fusion.py.txt"
        if (
            hashlib.sha256(source.read_bytes()).hexdigest()
            != run["baseline_source_sha256"]
        ):
            raise ValueError("Baseline source checksum mismatch")
        for case in cases.values():
            if len(case["pairs"]) != run["rounds"]:
                raise ValueError("Incomplete timing blocks")
            orders = [p["order"] for p in case["pairs"]]
            if (
                orders.count(["baseline", "fixed"]) != run["rounds"] // 2
                or orders.count(["fixed", "baseline"]) != run["rounds"] // 2
            ):
                raise ValueError("Unbalanced timing order")
            for record in case["records"].values():
                if record["correctness"] != "passed":
                    raise ValueError("Failed correctness")
                for code in record["code"]:
                    path = root / f"run{i}" / code["file"]
                    if hashlib.sha256(path.read_bytes()).hexdigest() != code["sha256"]:
                        raise ValueError(f"Generated code checksum mismatch: {path}")
            for pair in case["pairs"]:
                for mode in ("baseline", "fixed"):
                    for metric in ("queued_us", "sync_us"):
                        value = pair[mode][metric]
                        if not math.isfinite(value) or value <= 0:
                            raise ValueError("Invalid timing")
    summary = dict(
        baseline_commit=runs[0]["baseline_commit"],
        dtype=runs[0]["dtype"],
        training=runs[0]["training"],
        device=runs[0]["environment"]["device"],
        runs=2,
        pairs_per_case=2 * runs[0]["rounds"],
        aggregation="Median pooled block means; ratio of baseline median to fixed median. Bootstrap resamples paired blocks within each process.",
        cases=[],
    )
    rng = random.Random(185)
    lines = [
        "Paired MPS reduction benchmarks (FP32 inference)",
        f"Baseline reduction pass: {summary['baseline_commit']}",
        "Speedup = baseline / fixed. Below 1 means slower.",
        "Case | synchronized speedup [95% interval] | queued speedup | baseline/fixed graphs",
    ]
    for name in runs[0]["case_ids"]:
        cases = [index[name] for index in indexed]
        for key in ("shapes", "kind", "dims"):
            if cases[0][key] != cases[1][key]:
                raise ValueError(f"Cases differ in {key}: {name}")
        for mode in ("baseline", "fixed"):
            counts = [
                (
                    c["records"][mode]["graphs"],
                    [x["metal_definitions"] for x in c["records"][mode]["code"]],
                )
                for c in cases
            ]
            if counts[0] != counts[1]:
                raise ValueError(f"Compilation counts differ: {name}/{mode}")
        pairs = [p for c in cases for p in c["pairs"]]
        result = dict(
            id=name,
            kind=cases[0]["kind"],
            shapes=cases[0]["shapes"],
            timings={},
            graphs={m: cases[0]["records"][m]["graphs"] for m in ("baseline", "fixed")},
            metal_definitions={
                m: [x["metal_definitions"] for x in cases[0]["records"][m]["code"]]
                for m in ("baseline", "fixed")
            },
            setup_totals_s={
                m: [sum(c["records"][m]["setup_calls_s"]) for c in cases]
                for m in ("baseline", "fixed")
            },
        )
        for metric in ("queued_us", "sync_us"):
            b, f = [
                statistics.median(p[m][metric] for p in pairs)
                for m in ("baseline", "fixed")
            ]
            boot = []
            for _ in range(2000):
                sample = [
                    p for c in cases for p in rng.choices(c["pairs"], k=len(c["pairs"]))
                ]
                boot.append(
                    statistics.median(p["baseline"][metric] for p in sample)
                    / statistics.median(p["fixed"][metric] for p in sample)
                )
            result["timings"][metric] = dict(
                baseline_median_us=b,
                fixed_median_us=f,
                speedup=b / f,
                paired_bootstrap_95=[quantile(boot, 0.025), quantile(boot, 0.975)],
                per_run_speedup=[
                    statistics.median(p["baseline"][metric] for p in c["pairs"])
                    / statistics.median(p["fixed"][metric] for p in c["pairs"])
                    for c in cases
                ],
            )
        sync, queued = result["timings"]["sync_us"], result["timings"]["queued_us"]
        lo, hi = sync["paired_bootstrap_95"]
        lines.append(
            f"{name} | {sync['speedup']:.3f} [{lo:.3f}, {hi:.3f}] | {queued['speedup']:.3f} | {result['graphs']['baseline']}/{result['graphs']['fixed']}"
        )
        summary["cases"].append(result)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    summarize(parser.parse_args().results_dir)
