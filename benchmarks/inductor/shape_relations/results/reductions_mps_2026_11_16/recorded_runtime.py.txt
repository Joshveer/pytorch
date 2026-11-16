"""Shared device/provenance checks and complete original-compiler ablation."""

import argparse
import ast
import contextlib
import hashlib
import json
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch
from torch._inductor import config, sizevars


ROOT = Path(__file__).resolve().parents[3]
BASELINE_COMMIT = "c79633996ebc2114482ae5a4ff3cb409d79a8a9c"
COMPILER_FILES = (
    "torch/_inductor/sizevars.py",
    "torch/_inductor/config.py",
    "torch/_inductor/fx_passes/post_grad.py",
    "torch/_inductor/fx_passes/shape_relation_fusion.py",
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def class_methods(source):
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SizeVarAllocator"
    )
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


@dataclass
class Baseline:
    revision: str
    source_sha256: str
    methods: dict
    method_sha256: dict

    @contextlib.contextmanager
    def use(self, mode):
        if mode not in ("baseline", "fixed"):
            raise ValueError(mode)
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                config.patch(
                    guarded_shape_fusion=mode == "fixed",
                    force_disable_caches=True,
                    **{"triton.cudagraphs": False},
                )
            )
            if mode == "baseline":
                for name, fn in self.methods.items():
                    stack.enter_context(
                        patch.object(sizevars.SizeVarAllocator, name, fn)
                    )
            yield

    def metadata(self):
        return {
            "revision": self.revision,
            "source_sha256": self.source_sha256,
            "restored_methods": self.method_sha256,
            "guarded_shape_fusion": False,
        }


def load_baseline(revision=BASELINE_COMMIT):
    revision = git("rev-parse", revision)
    source = subprocess.check_output(
        ["git", "show", f"{revision}:torch/_inductor/sizevars.py"], cwd=ROOT, text=True
    )
    old = class_methods(source)
    current = class_methods(Path(sizevars.__file__).read_text())
    if old.keys() != current.keys():
        raise RuntimeError(
            "SizeVarAllocator added/removed methods; review the baseline adapter before comparing"
        )
    changed = [
        name
        for name in old
        if ast.dump(old[name], include_attributes=False)
        != ast.dump(current[name], include_attributes=False)
    ]
    if not changed:
        raise RuntimeError("No shape-relation fixes found in the imported compiler")
    functions, hashes = {}, {}
    for name in changed:
        node = old[name]
        if node.decorator_list:
            raise RuntimeError(f"Review decorated baseline method {name}")
        namespace = dict(vars(sizevars))
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]),
                f"<baseline-{revision}>",
                "exec",
            ),
            namespace,
        )
        functions[name] = namespace[name]
        hashes[name] = hashlib.sha256(ast.unparse(node).encode()).hexdigest()
    return Baseline(
        revision, hashlib.sha256(source.encode()).hexdigest(), functions, hashes
    )


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    else:
        raise ValueError(device)


def empty_cache(device):
    (torch.cuda if device == "cuda" else torch.mps).empty_cache()


def kernel_count(code, device):
    backend = "triton" if device == "cuda" else "metal"
    return len(re.findall(rf"(?m)^\w+\s*=\s*async_compile\.{backend}\(", code))


def command_info(command):
    try:
        p = subprocess.run(command, text=True, capture_output=True, timeout=15)
        return {
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def environment(device, expected_commit, dtype="float32"):
    if torch.version.git_version != expected_commit:
        raise RuntimeError(
            f"Expected native build {expected_commit}, imported {torch.version.git_version}"
        )
    if (
        Path(sizevars.__file__).resolve()
        != (ROOT / "torch/_inductor/sizevars.py").resolve()
    ):
        raise RuntimeError(
            "Use an editable native build from this checkout; imported compiler is elsewhere"
        )
    if not hasattr(config, "guarded_shape_fusion"):
        raise RuntimeError("Imported compiler does not contain the equality fix")
    build_config = torch.__config__.show()
    embedded = re.search(r"COMMIT_SHA=([0-9a-f]+)", build_config)
    if embedded and embedded.group(1) != expected_commit:
        raise RuntimeError("Python version metadata and native library commit disagree")
    result = {
        "torch": torch.__version__,
        "commit": torch.version.git_version,
        "repository_head": git("rev-parse", "HEAD"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "cuda_runtime": torch.version.cuda,
        "native_extension": torch._C.__file__,
        "native_extension_sha256": hashlib.sha256(
            Path(torch._C.__file__).read_bytes()
        ).hexdigest(),
        "torch_path": torch.__file__,
        "compiler_source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in COMPILER_FILES
        },
        "compiler_source_diff": git("diff", "--", *COMPILER_FILES),
        "build_config": build_config,
        "cudagraphs": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    if device == "cuda":
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError(
                "A real NVIDIA CUDA device is required; no CPU/MPS/ROCm fallback"
            )
        import triton

        from torch.utils._triton import has_triton

        if not has_triton():
            raise RuntimeError("Triton is not usable on this CUDA device")
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            raise RuntimeError("Native BF16 is unsupported; run the FP32-only path")
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        result.update(
            triton=triton.__version__,
            gpu_name=prop.name,
            gpu_uuid=str(getattr(prop, "uuid", "unknown")),
            gpu_index=torch.cuda.current_device(),
            capability=list(torch.cuda.get_device_capability()),
            cuda_arch_list=torch.cuda.get_arch_list(),
            total_memory=prop.total_memory,
            bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False),
            nvidia_smi=command_info(["nvidia-smi", "-q"]),
        )
        driver = re.search(
            r"Driver Version\s*:\s*(\S+)", result["nvidia_smi"].get("stdout", "")
        )
        result["driver_version"] = driver.group(1) if driver else None
        result["cpu"] = platform.processor() or platform.machine()
        result["battery"] = None
    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("A real MPS device is required")
        result["triton"] = None
        result["cpu"] = command_info(["sysctl", "-n", "machdep.cpu.brand_string"]).get(
            "stdout", platform.machine()
        )
        result["battery"] = command_info(["pmset", "-g", "batt"]).get("stdout")
    else:
        raise ValueError(device)
    return result


def profile_cuda(call, destination):
    """Keep profiling outside timings. Raw device events include copies/memsets."""
    from torch._dynamo.utils import counters
    from torch._inductor.graph import GraphLowering

    codes = []
    before = counters["stats"]["unique_graphs"]
    torch.cuda.synchronize()
    with (
        patch.object(GraphLowering, "save_output_code", codes.append),
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof,
    ):
        call()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(destination))
    if codes or counters["stats"]["unique_graphs"] != before:
        raise RuntimeError(
            "Profiling triggered compilation; the trace is not a warm execution"
        )
    return [
        {
            "name": event.name,
            "device_index": event.device_index,
            "duration_us": event.time_range.elapsed_us(),
        }
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    parser.add_argument("--expected-build-commit", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = environment(args.device, args.expected_build_commit, args.dtype)
    report["baseline"] = load_baseline().metadata()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
