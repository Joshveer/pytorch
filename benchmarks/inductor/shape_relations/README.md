# Shape-relation fusion benchmarks

Portable GPU runners for the divisibility and guarded-equality fixes. See the
[CUDA/Triton handoff](CUDA.md) for native build, full XSA, baseline and benchmark
commands. CUDA execution is not yet validated; the recorded measurements below
are MPS results. The
recorded runs use the actual native PyTorch 2.16 build, FP32 contiguous inputs,
and inference. Workload functions in `workloads.py` are copied unchanged from
[Cyrus's reproductions](https://github.com/CyrusD123/inductor-dynamic-fusion/tree/275f764ee84812a8db53ff93e915577f9b97b2bc).
No workload equality check or branch is added.

## Matmul batching

The separate `matmul.py` harness compares the original batch-linear pass with
proved-compatible and guarded symbolic batching, including a larger-budget
control to expose packing costs. See the [implementation and usage notes](../../../tools/inductor/symbolic_batch_linear.md).
It uses MPS and records its own baseline revision and compiler-source hashes.
[Recorded matmul results](results/matmul_mps_2026_11_15/README.md) include two
runs against the published compiler source, raw timings and generated code.
The new regression suite can also be run on CUDA as described in the usage
notes; the matmul timing harness remains MPS-only.

## Generalized reduction shape checks

The MPS-only `reductions.py` runner compares the new tagged-reduction shape
policy against the previous pass at `01d544c`, with the earlier project fixes
retained in both variants. It covers eight operations, multiple sizes, unequal
shapes, two-axis reductions and mixed streams. See the
[usage and timing notes](../../../tools/inductor/general_shape_fusion.md) and
[recorded battery-powered MPS results](results/reductions_mps_2026_11_16/README.md).
The `summarize_reductions.py` companion validates and combines two runs.

## Recorded original-fix results

[Full report](results/mps_2026_11_15/report.txt) and
[machine-readable summary](results/mps_2026_11_15/summary.json).

Hardware: Apple M5, 24 GB RAM, AC power; macOS 27.2.1, Python 3.12.12.
Native build: `2.16.0a0+git6345ea6`, commit
`6345ea65549d76751f09501580074b700b7c56c6`.

| Original-size workload | Synchronized latency, original -> fixed | Latency speedup | Queued speedup |
| --- | ---: | ---: | ---: |
| Softmax -> split heads | 499.8 -> 277.2 us | 1.80x | 3.34x |
| RMSNorm -> split heads | 491.1 -> 268.8 us | 1.83x | 3.66x |
| LayerNorm -> split heads | 542.6 -> 309.8 us | 1.75x | 2.74x |
| Equal-width normalization | 600.4 -> 406.2 us | 1.48x | 2.04x |
| Unequal-width fallback | 783.4 -> 792.1 us | 0.99x (within noise) | 1.00x |

Split shapes are `(2, 512, 512)`, with eight heads. Equality shapes are
`x=(4096,512)` and `y=(4096,512)` or `(4096,768)`.
The full report includes all 17 cases: small, original and large sizes, the
no-split control, and a mixed equal/unequal stream. The control stays near 1.00x.
Small-case single-call gains are minor or noisy. The mixed stream improves
1.21x queued and 1.14x synchronized, but compiles two graphs rather than one;
its observed setup totals are about 0.14s versus 0.07s.

## Method and evidence limits

Both sides use the same native build. The recorded MPS baseline restored the
`SizeVarAllocator.simplify` method from
`c79633996ebc2114482ae5a4ff3cb409d79a8a9c` and disables the equality pass.
The current portable runner additionally restores every other modified
`SizeVarAllocator` method, covering the subsequent XSA repairs. Fixed compilation
enables the changes in the current checkout. There were no native-code changes
between these revisions. This isolates compiler behavior; the baseline is not a
second independent installation. The baseline revision is pinned explicitly,
so adding result/documentation commits does not silently change the comparison.

Two separate processes each run 16 balanced, shuffled baseline-first/fixed-first
pairs per case. Both variants use identical inputs and are compiled and warmed
before timing. MPS is synchronized around each block, and graph counts ensure
there is no recompilation during measurement. All 80 eager-output comparisons
passed at `rtol=atol=1e-4`.

- Queued time measures a batch of repeated calls, synchronizing at its boundaries.
  The median block mean is reported as time per call.
- Synchronized latency waits after every call, using eight repetitions of the
  input sequence per block. It includes dispatch and synchronization overhead.
- Neither is a kernel-only GPU measurement. Generated kernel definitions, not
  profiler launch counts, are recorded.
- The 95% bootstrap intervals resample paired blocks within each process. They
  describe observed variation, not uncertainty across machines or models.
- Setup times include first execution after backend initialization. With only two
  samples per variant and no GPU-driver cache reset, they are descriptive, not
  cold-start or isolated compiler-time benchmarks.
- Speedups cover the complete fixes, including changes to indexing and memory
  traffic. Kernel-launch count is not isolated as the sole cause.
- These measurements do not establish training, BF16, CUDA, or whole-model speedups.

## Reproduce

From the repository root, with an installed native build of the fixed compiler:

```sh
BENCH=benchmarks/inductor/shape_relations
OUT=agent_space/mps-benchmark-rerun

env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 .venv/bin/python "$BENCH/bench.py" \
  --output "$OUT/run1/results.json" --seed 20261115
env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 .venv/bin/python "$BENCH/bench.py" \
  --output "$OUT/run2/results.json" --seed 20261116
.venv/bin/python "$BENCH/summarize.py" --results-dir "$OUT"
```

The default expected runtime commit is the measured `6345ea6` revision. If a new
native build has a different commit, explicitly pass
`--expected-build-commit <torch.version.git_version>`; its version is recorded in
new results. Keep `--baseline-commit` at the recorded parent when comparing these
same fixes. A shallow checkout must contain that commit (`git fetch origin
c79633996ebc2114482ae5a4ff3cb409d79a8a9c` if needed).

For a short functional check, use `--rounds 2 --only
softmax-original,equality-equal-original,equality-unequal-original,equality-mixed-original`.
This is not a replacement for the full measurement protocol. Write reruns to a new
output directory; do not replace the recorded results.

## Included artifacts

`results/mps_2026_11_15/` contains:

- `run1/results.json`, `run2/results.json`: unmodified timing samples, setup costs,
  graph counts, kernel counts and code hashes from the two measured runs.
- `run*/code/*.py.txt`: unmodified generated code. The `.txt` suffix prevents
  source formatters from changing the bytes covered by the recorded hashes.
- `summary.json`: the deterministic analysis of those raw samples.
- `report.txt`, `recorded_harness.py.txt`: archived report and exact original
  measurement harness. Its scratch-directory commands are historical; use the
  portable commands above for new runs.
- `native-build-summary.json`, `native-environment.json`, `native-tests.log`:
  evidence of the actual 2.16 source build and its 50 passing tests / one skip.
  Machine-specific checkout prefixes in build evidence are replaced with
  `<checkout>`; raw benchmark timings and generated code are unchanged.
- `sha256.json`: checksums of all other artifacts in this recorded result folder.

The published harness preserves the measurement procedure but vendors the five
workload functions, pins the baseline, and accepts explicit paths/revisions.
The original script depended on a neighboring checkout and used `HEAD^` before
any documentation commit existed. Pilot runs are not included in the analysis.
