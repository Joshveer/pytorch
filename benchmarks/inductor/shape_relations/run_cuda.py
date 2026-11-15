"""Run an existing native CUDA build; never install, build or allocate a GPU."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from runtime import environment, load_baseline, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expected-build-commit", required=True)
    ap.add_argument("--fp32-only", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--tests-only", action="store_true")
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError("Use a fresh output directory")
    out.mkdir(parents=True)
    report = {"stages": [], "status": "running"}

    def save():
        (out / "run.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        report["environment"] = environment("cuda", args.expected_build_commit)
        report["baseline"] = load_baseline().metadata()
    except Exception as exc:
        report.update(status="failed", preflight_error=str(exc))
        save()
        raise SystemExit(str(exc)) from None
    save()
    bench = ROOT / "benchmarks/inductor/shape_relations"
    hard_failure = False
    review = False

    def stage(name, arguments, diagnostic=True):
        directory = out / name
        directory.mkdir()
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        env["TORCHINDUCTOR_CACHE_DIR"] = str(directory / "inductor-cache")
        env["TRITON_CACHE_DIR"] = str(directory / "triton-cache")
        env["PYTORCH_TESTING_DEVICE_ONLY_FOR"] = "cuda"
        if diagnostic:
            env["TORCH_LOGS"] = "fusion,recompiles,guards"
        else:
            env.pop("TORCH_LOGS", None)
        command = [sys.executable, *map(str, arguments)]
        print("Running", name, flush=True)
        with (directory / "log.txt").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        report["stages"].append(
            {
                "name": name,
                "command": command,
                "returncode": result.returncode,
                "log": str(directory / "log.txt"),
            }
        )
        save()
        print(name, "exit", result.returncode, flush=True)
        return result.returncode

    tests = ROOT / "test/inductor/test_shape_relation_fusion.py"
    rc = stage(
        "minimal-tests",
        [
            tests,
            "TestShapeRelationSimplify",
            "TestEqualityGuardPolicy",
            "TestSplitHeadFusionCUDA",
            "TestGuardedShapeFusionCUDA",
            "-v",
        ],
    )
    hard_failure |= rc != 0
    if args.fp32_only:
        for suffix, pattern in [("fp32", "float32"), ("shape-sweep", "shape_sweep")]:
            rc = stage(
                "xsa-tests-" + suffix,
                [tests, "TestXSAShapeFusionCUDA", "-k", pattern, "-v"],
            )
            hard_failure |= rc != 0
    elif report["environment"]["bf16_supported"]:
        rc = stage("xsa-tests", [tests, "TestXSAShapeFusionCUDA", "-v"])
        hard_failure |= rc != 0
    else:
        report["bf16_omitted"] = (
            "GPU lacks native BF16; full validation needs review. Use --fp32-only for intentional FP32-only coverage."
        )
        review = True
        rc = stage(
            "xsa-tests-fp32", [tests, "TestXSAShapeFusionCUDA", "-k", "float32", "-v"]
        )
        hard_failure |= rc != 0
        rc = stage(
            "xsa-tests-shape-sweep",
            [tests, "TestXSAShapeFusionCUDA", "-k", "shape_sweep", "-v"],
        )
        hard_failure |= rc != 0
    dtypes = ["float32"]
    if not args.fp32_only and report["environment"]["bf16_supported"]:
        dtypes.append("bfloat16")
    fp32_ready = not hard_failure
    for dtype in dtypes:
        for inference in (False, True):
            name = f"xsa-{dtype}-" + ("inference" if inference else "training")
            arguments = [
                bench / "xsa.py",
                "--device",
                "cuda",
                "--expected-build-commit",
                args.expected_build_commit,
                "--dtype",
                dtype,
                "--output",
                out / name / "results.json",
            ]
            if inference:
                arguments.append("--inference")
            if not args.tests_only and dtype == "float32" and not hard_failure:
                arguments.append("--benchmark")
            if args.profile:
                arguments.append("--profile")
            rc = stage(name, arguments)
            hard_failure |= rc not in (0, 2)
            review |= rc == 2
            if dtype == "float32":
                fp32_ready &= rc == 0
    if not args.tests_only and fp32_ready:
        ok = True
        for n, seed in [(1, 20261115), (2, 20261116)]:
            name = f"run{n}"
            arguments = [
                bench / "bench.py",
                "--device",
                "cuda",
                "--expected-build-commit",
                args.expected_build_commit,
                "--output",
                out / name / "results.json",
                "--seed",
                seed,
            ]
            if args.profile:
                arguments.append("--profile")
            rc = stage(name, arguments, False)
            ok &= rc in (0, 2)
            hard_failure |= rc not in (0, 2)
            review |= rc == 2
        if ok:
            rc = stage("summary", [bench / "summarize.py", "--results-dir", out], False)
            hard_failure |= rc != 0
    elif not args.tests_only:
        report["minimal_benchmark_skipped"] = (
            "Resolve FP32 regression/XSA checks before benchmarking"
        )
    report["status"] = (
        "failed" if hard_failure else "needs_review" if review else "passed"
    )
    save()
    print("STATUS", report["status"], "report:", out / "run.json", flush=True)
    raise SystemExit(1 if hard_failure else 2 if review else 0)


if __name__ == "__main__":
    main()
