# Dynamic-shape fusion project: local implementation

This branch addresses two cases from
[CyrusD123/inductor-dynamic-fusion](https://github.com/CyrusD123/inductor-dynamic-fusion)
at `275f764ee84812a8db53ff93e915577f9b97b2bc`. Workloads do not need an added
shape check or Python branch.

## Divisibility

`SizeVarAllocator.simplify` cancels positive constant floor divisors in products
when ShapeEnv already records the corresponding modulo fact. For example,
`16*T*(C//8)` becomes `2*T*C` if `Mod(C,8) == 0` is established. This keeps
grouping, tiling and indexing consistent without adding guards or using example
values as proof. Unknown divisibility remains unknown.

## Symbolic equality

The post-grad pass `guard_reduction_width_equality` recognizes a last-dimension
mean/sum with `keepdim=True`, followed by a short pointwise chain broadcasting
the result over another width. It considers one pair of independent backed
width symbols per graph, with already-equal outer dimensions.

For `y * rsqrt(mean(x**2, -1, keepdim=True) + eps)`, both equal and unequal
widths are valid. The pass evaluates width equality using `guard_or_false`,
which installs an actual Dynamo guard:

- Equal widths: ShapeEnv unifies the symbols before lowering, enabling the
  existing SIMD fusion checks to succeed.
- Unequal widths: the ordinary unfused graph remains correct, guarded by
  inequality. A later equal-width input can trigger the fused specialization.

The normal Dynamo guard cache dispatches between these graphs. This is lazy
JIT specialization, not AOT generation of both kernels. No unconditional shape
assertion is added to the workload. The scheduler's dependency and tiling checks
are unchanged.

The pass is enabled by `torch._inductor.config.guarded_shape_fusion` on this
project branch; setting it to `False` restores the original equality behavior.
It targets CUDA and MPS, skips CPU, and declines unbacked/compound widths,
frozen ShapeEnvs, calls without an active guard-dispatching compilation context,
and AOTInductor compilation. AOT exports cannot rely on
Dynamo's guard dispatcher. Singleton and zero dimensions retain normal compiler
handling. Profitability beyond this bounded pattern has not been established.

## Native 2.16 validation

The modified fork was built from source at
`6345ea65549d76751f09501580074b700b7c56c6` and installed as
`2.16.0a0+git6345ea6`. This supersedes the initial 2.15 source-overlay checks.
The native extension, embedded build commit and compiler source paths were
verified before testing; the native tests did not use an overlay or monkey patch.

Build: Python 3.12.12, macOS 27.2.1 arm64, Apple clang 21.0.0, Release,
CPU and MPS enabled, CUDA/distributed disabled, six parallel compile jobs.
The command-line-tools-only setup used the supported runtime Metal shader
compilation fallback. Both CPU and MPS native smoke checks passed.

The direct regression suite passed **50 tests, with one CPU kernel-count test
skipped**. It covers outputs and random-weight gradients, equal-first and
unequal-first sequences, strided inputs, zero/singleton dimensions, graph reuse,
Inductor disk-cache hits, missing/unrelated divisibility facts, and refusal to
specialize unbacked sizes or AOT/direct compilation without a guard dispatcher.
The original team split-head and unchecked equality reproductions also passed
on the native MPS build. Generated forward kernel counts were one for the split
and equal-width cases, and two for unequal-width fallback.

An initial CPU test run failed to load `libomp.dylib`. Inductor's generated CPU
kernels used an existing Conda OpenMP library whose runtime search path was not
resolved. A process-local `DYLD_LIBRARY_PATH` pointing to that runtime resolved
all CPU failures. No compiler source changes were needed. This loader setting
is specific to the local environment and was not needed for MPS benchmarking.

Native evidence is in
[the recorded results](../../benchmarks/inductor/shape_relations/results/mps_2026_11_15/native-build-summary.json).
The source baseline before both fixes is
`c79633996ebc2114482ae5a4ff3cb409d79a8a9c`.

### Build and test

After installing the dependencies in `requirements.txt` into `.venv`, the build
used the following settings from the repository root:

```sh
PATH="$PWD/.venv/bin:$PATH" CMAKE_PREFIX_PATH="$PWD/.venv" \
CMAKE_BUILD_TYPE=Release MAX_JOBS=6 USE_MPS=1 USE_CUDA=0 \
USE_DISTRIBUTED=0 BUILD_TEST=0 \
  .venv/bin/python -m pip install -e . -v --no-build-isolation

env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python test/inductor/test_shape_relation_fusion.py -v
```

If CPU generated kernels need an OpenMP loader path, set `DYLD_LIBRARY_PATH`
for that test process to the directory containing the installed `libomp.dylib`.
`tools/inductor/shape_relation_local_test.py` remains an optional overlay helper
for older installed runtimes; it is not how the native validation above ran.

## XSA reproduction repair

The original minimal tests did not cover the full `xsa_unsqueeze` and
`xsa_expand` functions in Cyrus's issue reproduction. Running those functions
exposed a tiling/range-splitting regression in the first divisibility patch and
a remaining fusion rejection in `expand`.

The repair keeps equivalent shapes canonical throughout the analysis:

- Equality queries use the same simplification as grouping and tiling, while
  preserving CPU callers that compare tuples of reduction dimensions.
- Scaled divisibility queries consume the existing modulo fact. For example,
  `Mod(H,4)==0` also proves that `2048*H` is divisible by `8192`.
- `Max(1,H//4)` simplifies only if existing bounds or positive exact-divisibility
  facts prove that the quotient is at least one. Zero remains protected.
- Merged loop extents are simplified before dependency comparison. This lets
  `8192*(H//4)` and `2048*H` describe the same access rather than reporting
  "no shared data". Different offsets still remain distinct dependencies.
- Simplification caches invalidate when divisibility facts or bounds change.

No new guards or workload changes are required for these repairs. Existing
memory-dependency and tiling legality checks remain in place.

On MPS, the unchanged reproductions now give the following generated kernel
counts in both FP32 and BF16 (forward + backward). The reproduction is
[`repro_xsa_issue.py` at `275f764`](https://github.com/CyrusD123/inductor-dynamic-fusion/blob/275f764ee84812a8db53ff93e915577f9b97b2bc/repro_xsa_issue.py),
using `B=1`, `T=2048`, `HQ=32`, `HKV=4`, `D=64` and its original dynamic
annotations. Matching the static reference does not mean combining the entire
forward and backward computation into one kernel.

| Function | Static reference | Dynamic, original compiler behavior | Dynamic, repaired |
| --- | ---: | ---: | ---: |
| `xsa_unsqueeze` | 2 + 4 | 3 + 4 | 2 + 4 |
| `xsa_expand` | 1 + 2 | 2 + 2 | 1 + 2 |

Inference-only counts also match static: two for unsqueeze and one for expand.
The expanded regression suite passes 79 tests with one skip, including all
previous minimal tests and XSA shape/stride/gradient checks. An additional
indexing suite passes 26 tests with three skips. These run directly through
the native 2.16 installation with the updated editable Python compiler source.

FP32 outputs and gradients pass eager comparisons. BF16 outputs pass, and the
repaired outputs and gradients match the corresponding pre-fix dynamic
baseline bit-for-bit on the original seeded reproduction. BF16 gradients
already differ from eager, and `expand` also has a pre-existing static/dynamic
gradient discrepancy due to different intermediate precision. That numerical
issue is not resolved here; the BF16 regression test compares against the
unoptimized dynamic compiler rather than relaxing the tolerance. FP32 eager
checks use `rtol=atol=2e-4`; BF16 output and baseline-gradient comparisons use
`rtol=atol=0.02`. The bit-for-bit statement comes from the separate seeded
reproduction comparison; it is not a claim of BF16 gradient agreement with eager.

Run the targeted reproduction tests on an MPS machine with the native build:

```sh
env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python test/inductor/test_shape_relation_fusion.py TestXSAShapeFusionMPS -v
```

The full command in the build-and-test section also runs the minimal regression
tests and the symbolic proof/cache checks. CUDA/Triton has not been validated.

## CUDA/Triton handoff

The [GPU validation scripts and instructions](../../benchmarks/inductor/shape_relations/CUDA.md)
cover the full XSA reproductions, minimal cases, original compiler ablation,
unequal widths, outputs/gradients, generated code and paired timings. These
scripts require real CUDA/Triton and do not treat MPS checks as CUDA validation.

## MPS performance

The measurements below are from the earlier `6345ea6` implementation and the
minimal workloads. They are not new measurements of the XSA repair.

FP32 inference on an Apple M5 Mac with 24 GB RAM, on AC power. Both variants use
the same native 2.16 build. Baseline restores the parent simplifier during
compilation and disables the equality pass; fixed enables both changes. The
workload functions are unchanged.

| Original-size workload | Synchronized latency, original -> fixed | Latency speedup | Queued speedup |
| --- | ---: | ---: | ---: |
| Softmax -> split heads | 499.8 -> 277.2 us | 1.80x | 3.34x |
| RMSNorm -> split heads | 491.1 -> 268.8 us | 1.83x | 3.66x |
| LayerNorm -> split heads | 542.6 -> 309.8 us | 1.75x | 2.74x |
| Equal-width normalization | 600.4 -> 406.2 us | 1.48x | 2.04x |
| Unequal-width fallback | 783.4 -> 792.1 us | 0.99x (within noise) | 1.00x |

Split shapes are `(2,512,512)` with eight heads. Equality inputs are
`x=(4096,512)` and `y=(4096,512)` or `(4096,768)`.

Two process runs each measured 16 alternating paired blocks per case across 17
cases. Both variants were compiled and warmed first; all 80 eager-output
comparisons passed, and no timed block recompiled. Synchronized latency waits
for every call; queued time synchronizes around batches. Both include Python
and allocation overhead, rather than measuring only GPU kernel duration.

The no-split control and unequal-width cases stayed near baseline. Small-input
single-call gains were minor or noisy. A mixed equal/unequal stream improved
1.21x queued and 1.14x synchronized, but required two graphs and about 0.14s of
observed setup versus one graph and 0.07s for baseline. Those setup measurements
are descriptive first-call costs after backend initialization, not cold-start
or isolated compiler-time measurements.

See the [benchmark harness and reproduction instructions](../../benchmarks/inductor/shape_relations/README.md),
[all timings and intervals](../../benchmarks/inductor/shape_relations/results/mps_2026_11_15/summary.json),
and [full report](../../benchmarks/inductor/shape_relations/results/mps_2026_11_15/report.txt).
The speedups cover the complete compiler fixes, including indexing and memory
traffic changes; they do not isolate kernel-launch count as the sole cause.
CUDA/Triton validation, broader BF16 numerical validation, training/whole-model performance and
broader profitability evaluation remain outstanding.

Prepared with an AI coding assistant for the project; not submitted upstream.
