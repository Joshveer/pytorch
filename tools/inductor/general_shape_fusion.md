# Generalizing guarded shape fusion

This extends the shared shape-guard machinery to more reduction operations.
It removes the mean/sum name list from candidate discovery; it does not implement
arbitrary operation fusion or change the scheduler's fusion legality rules.

## What changed

`guard_reduction_shape_equality` recognizes operators carrying
`torch.Tag.reduction`. Tensor metadata must show a retained prefix and a trailing
block of dimensions reduced to singleton dimensions, with the same rank,
floating dtype and device as the input. Tuple-returning reductions are followed
through their tensor `getitem` results, so both `max`/`min` values and
`var_mean` can participate. Integer index outputs do not become candidates.

The pass follows a bounded chain of pure, tagged pointwise consumers. When the
reduced result is broadcast over another tensor with an already-equal prefix,
it asks the same `guard_shape_compatibility` helper used by matmul batching to
check the trailing dimensions. The implementation has no list of names such as
`amax`, `prod`, or `var`.

For example, `y * (x.amax(-1, keepdim=True) + 0.1)` is valid even when `x` and
`y` have different widths. Matching widths can enable the existing reduction
and elementwise fusion. The guard selects that specialization; unequal widths
retain the ordinary, correct compilation. No workload assertions are added.

Multiple trailing dimensions use one conjunction, for example
`x.size(1) == y.size(1) AND x.size(2) == y.size(2)`. Equal total element counts
alone are not enough. The tested streams reuse two Dynamo graphs across equal
inputs and different mismatch axes, including unequal-first streams.

The existing `guarded_shape_fusion` option controls this pass and remains enabled
by default on this research branch. Set it to false to disable reduction shape
speculation. Matmul batching retains its separate opt-in options.

## Local MPS findings

The reference is commit `01d544c8af54330e035a2eaf38554b7efef3a274`, which already
contains the original divisibility/equality fixes and the matmul extension.
The comparison restores that revision's reduction-guard pass in the same native
installation. All other compiler settings and workload definitions are held
constant. Validation uses native PyTorch `2.16.0a0+git6345ea6` with this checkout's
editable Python compiler code on Apple M5/MPS.

The workloads reduce `x`, then broadcast the result into a pointwise operation
on independently shaped `y`. The single-axis examples use `(17, 64)` inputs;
the two-axis examples use `(7, 4, 16)`.

| Reduction/workload | Static shader definitions | Previous dynamic | Generalized dynamic |
| --- | ---: | ---: | ---: |
| `amax`, `amin` | 1 | 2 | 1 |
| `prod` | 1 | 2 | 1 |
| `var`, `std` | 1 | 2 | 1 |
| `max(...).values`, `min(...).values` | 1 | 2 | 1 |
| Combined outputs of `var_mean` | 1 | 2 | 1 |
| Two-axis `amax`, `var`, `var_mean` | 1 | 2 | 1 |
| Mean, vector norm, logsumexp controls | 1 | 1 | 1 |
| Softmax followed by same-shape pointwise control | 1 | 1 | 1 |

These counts are generated Metal shader definitions, not profiler launch counts
or timing measurements. The separate timing study below measures performance.
The controls already
optimized correctly and are not credited as new fixes. The softmax control
requires matching shapes for broadcasting; it is not an unequal-width example.

## MPS timing results

Two independent processes ran 34 cases with 16 balanced baseline-first/fixed-first
pairs per case. All **168 eager-output comparisons passed**. Each timed block
queues 32 repetitions of the input sequence, or synchronizes after each call for
eight repetitions. Results are time per call, including allocation, dispatch and
synchronization. Compilation and warmup are excluded; recompilation during
warmup or timing is an error.

These measurements use FP32 inference on the M5 **on battery power** (67% and
66% at the start of the two runs), unlike the earlier AC-powered project results.
Both variants use the same tensors and native runtime. They compare the original
reduction pass at `01d544c` with the generalized pass; earlier project fixes
remain enabled in both variants. This is not an unmodified-upstream comparison
or a second native installation.

| Cases | Synchronized latency speedup | Queued throughput speedup |
| --- | ---: | ---: |
| All eight new operations, equal `(512, 256)` | 1.07–1.11x | 1.47–1.81x |
| `amax`, equal `(4096, 512)` | 1.43x | 1.73x |
| `var`, equal `(4096, 512)` | 1.64x | 1.85x |
| `amax`/`var`, equal `(17, 64)` | ~1.00–1.01x | 1.05–1.12x |
| Two-axis equal reductions | 1.04–1.11x | 1.43–1.72x |
| Unequal shapes, all sizes | ~0.99–1.01x | ~1.00–1.02x |
| Mixed equal/unequal widths | 1.04x | 1.29x |
| Mixed axes, unequal first | 1.02x | 1.14x |
| Already-optimized mean/norm controls | ~1.00x | ~1.00x |

Speedups are ratios of medians of the 32 pooled block means. Small-case latency
improvements are within noise; the mixed-axis latency interval also includes
1.0. Larger `amax` varies between processes: individual-run synchronized
speedups were 1.37x and 1.32x, while the pooled-median ratio is 1.43x. Do not
interpret these measurements as a fixed expected gain on other machines.

Mixed streams use two fixed graphs versus one baseline graph. Their measured
setup totals roughly doubled for the single-width stream (about 0.11s versus
0.055s); the two-axis stream also has extra setup cost. Setup numbers are
observations with existing driver caches, not isolated compiler or cold-start
measurements. They are excluded from the warm timing table.

The [recorded results](../../benchmarks/inductor/shape_relations/results/reductions_mps_2026_11_16/README.md)
include all raw pairs, absolute timings, per-run results, paired bootstrap
intervals, generated code, source fingerprints and checksums. Bootstrap intervals
resample paired blocks within each process; they do not quantify variation
across hardware, power modes or models. Fewer shader definitions alone do not
isolate the cause of the gains from changes in memory traffic and execution.

To rerun on MPS from the installed fork's root:

```sh
# Use the native build commit recorded when installing your fork.
# This is the recorded build's value; replace it if you built a newer revision.
EXPECTED_BUILD_COMMIT=6345ea65549d76751f09501580074b700b7c56c6
git fetch origin 01d544c8af54330e035a2eaf38554b7efef3a274
BENCH=benchmarks/inductor/shape_relations
OUT=agent_space/reduction-benchmark-rerun

env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 .venv/bin/python "$BENCH/reductions.py" \
  --expected-build-commit "$EXPECTED_BUILD_COMMIT" --output "$OUT/run1/results.json" --seed 51
env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 .venv/bin/python "$BENCH/reductions.py" \
  --expected-build-commit "$EXPECTED_BUILD_COMMIT" --output "$OUT/run2/results.json" --seed 73
.venv/bin/python "$BENCH/summarize_reductions.py" --results-dir "$OUT"
```

Use a fresh output directory and an otherwise idle GPU. Match and record power
mode when comparing runs. The summarizer validates matching environments and
sources, complete balanced blocks, correctness status and generated-code hashes.
`--only amax-equal-medium,prod-unequal-medium,var-mixed-widths,var-mixed-axes-unequal-first --rounds 2`
provides a functional smoke check, not a performance result.

## Regression coverage

`test/inductor/test_general_shape_fusion.py` checks:

- Static versus dynamic compilation with speculation disabled/enabled.
- Forward outputs and gradients for all eight newly covered operations.
- Dense and stride-2 inputs; equal-first and unequal-first streams.
- Two reduced axes, including equal element counts with unequal axis sizes.
- Zero rows, singleton dimensions, and preservation of max's index output.
- No shape specialization for leading-axis reductions, full scalar reductions,
  index-only reductions, slicing, or an over-budget equality conjunction.

Run from the installed fork's root:

```sh
env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 \
  PYTORCH_TESTING_DEVICE_ONLY_FOR=mps \
  .venv/bin/python test/inductor/test_general_shape_fusion.py -v
```

Local validation passed all **99 new tests**, **79 existing shape/XSA tests**
(one skipped), and **39 matmul tests**: **217 passed, one skipped**. The
static/previous-dynamic/generalized-dynamic survey passed all 45 output
comparisons across 15 cases.

The original `test_shape_relation_fusion.py` and
`test_symbolic_batch_linear.py` suites cover the shared guard refusal policy,
Cyrus's full reproductions, divisibility checks and the matmul fallback.

## Boundaries

This is a generic candidate policy for a reduction/pointwise family, alongside
the existing matmul batching policy. The shared shape helper does not override
backend support, dependencies, layouts or profitability decisions. It does not
provide generic matmul epilogue fusion or support every possible reduction layout.

The reduction pass attempts at most one joint guard per graph, with at most
four unknown equalities and eight visited nodes per candidate. The compared
dimensions must be greater than one; unknown equalities must relate independent
backed symbolic dimensions. Known mismatches and static-to-symbolic equality
specialization are declined. The shared helper refuses unbacked/compound
symbols, frozen shape environments, AOT compilation and contexts without Dynamo
fallback dispatch. Broader rank/dtype/layout changes can still create additional
compiled graphs through normal Dynamo specialization.

CUDA execution remains pending. New numerical coverage is FP32. The previous
XSA BF16 eager-gradient caveat is unchanged; those tests retain their existing
baseline-based checks. Training performance and larger-model benefits remain to be measured.
