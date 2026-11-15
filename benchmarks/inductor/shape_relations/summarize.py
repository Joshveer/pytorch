import argparse
import json
import random
import statistics as st
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=Path, required=True)
args = parser.parse_args()
ROOT = args.results_dir
runs = [json.loads((ROOT / f"run{i}/results.json").read_text()) for i in (1, 2)]
if any(len(r["cases"]) != 17 for r in runs):
    raise RuntimeError("Incomplete run")
indexed = [{c["id"]: c for c in r["cases"]} for r in runs]
if sorted(indexed[0]) != sorted(indexed[1]):
    raise ValueError("Runs contain different cases")
for key in (
    "torch",
    "commit",
    "parent",
    "original_simplify_sha256",
    "dtype",
    "mode",
    "rounds",
    "cpu",
    "platform",
):
    if runs[0][key] != runs[1][key]:
        raise ValueError(f"Runs differ in {key}")
# Historical MPS records predate the device/environment fields.
if runs[0].get("device", "mps") != runs[1].get("device", "mps"):
    raise ValueError("Cannot combine different devices")
if runs[0].get("baseline_metadata") != runs[1].get("baseline_metadata"):
    raise ValueError("Baseline implementations differ")
for key in (
    "cuda_runtime",
    "triton",
    "driver_version",
    "gpu_name",
    "gpu_uuid",
    "capability",
    "total_memory",
    "compiler_source_sha256",
    "native_extension_sha256",
    "build_config",
    "cuda_arch_list",
    "cudagraphs",
    "deterministic_algorithms",
    "tf32_matmul",
    "tf32_cudnn",
):
    if runs[0].get("environment", {}).get(key) != runs[1].get("environment", {}).get(
        key
    ):
        raise ValueError(f"Environments differ in {key}")
for run in runs:
    for case in run["cases"]:
        if len(case["pairs"]) != run["rounds"]:
            raise ValueError(f"Incomplete timing blocks for {case['id']}")
        for record in case["records"].values():
            if record["correctness"] != "passed":
                raise ValueError(f"Failed correctness for {case['id']}")
rng = random.Random(185)
summary = {
    "environment": {
        k: runs[0][k]
        for k in (
            "torch",
            "commit",
            "parent",
            "platform",
            "cpu",
            "baseline",
            "dtype",
            "mode",
        )
    },
    "runs": 2,
    "pairs_per_case": sum(run["rounds"] for run in runs),
    "cases": [],
}


def median(values):
    return st.median(values)


def quantile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = int(at)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - at) + values[hi] * (at - lo) if hi != lo else values[lo]


for name in sorted(indexed[0]):
    cases = [r[name] for r in indexed]
    if cases[0]["shapes"] != cases[1]["shapes"]:
        raise ValueError(f"Shapes differ for {name}")
    for mode in ("baseline", "fixed"):
        for key in ("graphs", "kernel_definitions_by_compile"):
            if cases[0]["records"][mode][key] != cases[1]["records"][mode][key]:
                raise ValueError(f"Compilation counts differ for {name}/{mode}/{key}")
    pairs = [p for c in cases for p in c["pairs"]]
    result = {
        "id": name,
        "shapes": cases[0]["shapes"],
        "scale": cases[0]["scale"],
        "timings": {},
    }
    for metric in ("queued_us", "sync_per_call_us"):
        b = median([p["baseline"][metric] for p in pairs])
        f = median([p["fixed"][metric] for p in pairs])
        boot = []
        for _ in range(3000):
            sample = [
                p for c in cases for p in rng.choices(c["pairs"], k=len(c["pairs"]))
            ]
            boot.append(
                median([p["baseline"][metric] for p in sample])
                / median([p["fixed"][metric] for p in sample])
            )
        result["timings"][metric] = {
            "baseline_median_us": b,
            "fixed_median_us": f,
            "speedup": b / f,
            "speedup_paired_bootstrap_95": [
                quantile(boot, 0.025),
                quantile(boot, 0.975),
            ],
            "baseline_block_p10_p90": [
                quantile([p["baseline"][metric] for p in pairs], 0.1),
                quantile([p["baseline"][metric] for p in pairs], 0.9),
            ],
            "fixed_block_p10_p90": [
                quantile([p["fixed"][metric] for p in pairs], 0.1),
                quantile([p["fixed"][metric] for p in pairs], 0.9),
            ],
            "speedup_each_run": [
                median([p["baseline"][metric] for p in c["pairs"]])
                / median([p["fixed"][metric] for p in c["pairs"]])
                for c in cases
            ],
        }
    result["kernel_definitions"] = {
        m: cases[0]["records"][m]["kernel_definitions_by_compile"]
        for m in ("baseline", "fixed")
    }
    result["graphs"] = {
        m: cases[0]["records"][m]["graphs"] for m in ("baseline", "fixed")
    }
    result["setup_total_seconds_each_run"] = {
        m: [sum(c["records"][m]["setup_calls_s"]) for c in cases]
        for m in ("baseline", "fixed")
    }
    summary["cases"].append(result)
if "environment" in runs[0]:
    summary["device_environment"] = runs[0]["environment"]
    summary["baseline_metadata"] = runs[0]["baseline_metadata"]
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for c in summary["cases"]:
    a = c["timings"]["queued_us"]
    b = c["timings"]["sync_per_call_us"]
    print(
        f"{c['id']:30} {a['baseline_median_us']:8.1f} -> {a['fixed_median_us']:8.1f} us {a['speedup']:.2f}x CI {a['speedup_paired_bootstrap_95']} latency {b['baseline_median_us']:.1f}->{b['fixed_median_us']:.1f}us {b['speedup']:.2f}x runs {a['speedup_each_run']}"
    )
