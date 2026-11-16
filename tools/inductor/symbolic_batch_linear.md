# Symbolic and guarded matmul batching

This extends the existing **opt-in** `batch_linear_post_grad` pass. It batches
compatible independent `mm`/`addmm` operations into `bmm`; it does not add
matmul-plus-activation kernel generation to the MPS backend.

## Get the branch and enable it

```sh
git fetch origin cse583/shape-relation-fusion
git switch cse583/shape-relation-fusion
git pull --ff-only origin cse583/shape-relation-fusion
```

Use this fork's native editable build, not a released PyTorch wheel. These
changes are Python compiler code, so an existing compatible editable 2.16 build
can use them without rebuilding the native library. For a fresh CUDA build,
follow the [build instructions](../../benchmarks/inductor/shape_relations/CUDA.md#1-use-the-modified-native-build)
on this branch. CUDA execution of this extension remains unvalidated.

For shapes already proved compatible, enable the existing pass:

```python
compiled = torch.compile(fn, options={
    "post_grad_fusion_options": {"batch_linear_post_grad": {}}
})
```

The matcher now accepts rank-two strided tensors with symbolic dimensions and
uses canonical shape keys from `SizeVarAllocator`. The keys retain ShapeEnv
identity for symbolic expressions and include dtype/device, so unrelated
symbols or devices do not accidentally share a group. Existing shape facts,
including divisibility relationships, are used without adding guards.

For independent symbolic shapes that may or may not match, explicitly enable
guarded batching:

```python
compiled = torch.compile(fn, options={
    "post_grad_fusion_options": {
        "batch_linear_post_grad": {
            "guarded_batching": True,
            "max_guarded_group_size": 8,
            "max_guarded_conditions": 32,
            "max_guarded_working_set_bytes": 1024 * 1024,
        }
    }
})
```

The pass itself is still disabled by default, and `guarded_batching` defaults
to false even when the pass is enabled. The default minimum batch group is five,
as before. Existing `min_fuse_set_size`/`max_fuse_set_size` options still apply.
A guard-group cap smaller than the minimum group, or a zero guard/byte budget,
disables speculation; it does not disable the ordinary proved-compatible path.

## Guard and profitability policy

Ordinary candidate grouping/fusion runs first. Remaining candidates use the
same bounded graph search and independent-subset selection. Speculation is
currently restricted to bias-free `mm` on MPS/CUDA, floating dtypes, aligned
symbolic/static dimension positions and operand orientation classes. It refuses
unbacked/compound symbolic dimensions, dimensions not proved greater than one,
frozen shape environments, AOT compilation and contexts without a Dynamo guard
dispatcher. Direct mutable or raw-storage users are excluded from speculation.
The proved-compatible path continues to support `addmm` as well.

A single conjunction checks all necessary equalities and the size bound:

```text
all operand shapes match
AND group_size * element_size * (M*K + K*N + M*N) <= byte_budget
```

If true, the existing batching transform runs. If false, the ordinary unbatched
graph is retained. Unlike sequential equality guards, this joint negation can
cover different mismatch axes and over-budget shapes with the same fallback.
The small/unequal/oversized sequences tested here used two compiled graphs,
including unequal-first and oversized-first cases.

There is at most one speculative group attempt per pass invocation on a graph,
with the node and predicate caps above. This is not a universal two-graph bound
for an entire program: training partitions, nested graphs, dtype/layout changes,
and Dynamo's ordinary zero/one specialization can create other graphs.

The byte budget is a conservative working-set proxy including inputs and output,
not a measured latency model. It is symmetric in M/K/N so forward and gradient
matmuls have comparable bounds. It applies to speculative batching, not to groups
already proved compatible, which retain the existing pass's policy.

## Local MPS validation

Tests use the native 2.16 installation with the updated editable Python source.
They cover canonical shape keys, ShapeEnv isolation, joint guards and refusal
conditions, outputs/gradients, dense and stride-2 inputs, equal/unequal/oversized-first arrival orders,
multiple mismatch axes, runtime size-limit crossings, zero/singleton dimensions,
low-precision gradients, mixed dtypes, dependencies, raw-storage users, and
budgets. No workload shape assertions or branches are required.

```sh
env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python test/inductor/test_symbolic_batch_linear.py -v
```

The local run passed all 39 new tests, plus 79 existing minimal/XSA tests
(one skipped) and four existing batch-fusion tests: **122 passed, one skipped**.
On a CUDA host with this fork installed, run the same new suite with
`PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda`; the existing `run_cuda.py` driver does
not yet include this new suite. CUDA execution is pending. The earlier XSA BF16 gradient discrepancy remains a
separate issue; passing the new linear-operation gradient tests does not resolve
that discrepancy.

## Packing-cost experiment

A dedicated harness compares four modes using the same native runtime and
inputs: the original batch-linear rule at `986da9d`, proved-compatible matching,
guarded batching with the 1 MiB budget, and guarded batching with a 64 MiB budget.
The last mode exposes cases deliberately rejected by the conservative default.
Only the old rule class is restored for the original mode; no workload is rewritten.

```sh
# Use the commit recorded when your native build was installed.
# The archived MPS build used the value below; replace it for your own build.
EXPECTED_BUILD_COMMIT=6345ea65549d76751f09501580074b700b7c56c6
# A shallow clone also needs the pinned original rule:
git fetch origin 986da9d7deb1b44db0e6dbd7e938d7636fb26870
env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python benchmarks/inductor/shape_relations/matmul.py \
  --expected-build-commit "$EXPECTED_BUILD_COMMIT" \
  --output agent_space/matmul-benchmark/run1/results.json --seed 51
# Repeat in a separate process, with a fresh output path and --seed 73.
```

Two local FP32 inference runs on the M5 used 16 balanced timing blocks each.
Each block queues 32 calls or synchronizes after each of eight calls. Compilation
and warmup are excluded, and unexpected compilation inside timings is rejected.
Timings include packing, allocations and dispatch. Generated wrapper calls are
recorded separately from Metal shader definitions; neither is a measured GPU
kernel-launch count.

| Five-projection case, M/K/N | Selected behavior | Approx. synchronized speedup | Approx. queued speedup |
| --- | --- | ---: | ---: |
| Shared input, 64/32/48 | Proved-compatible batching | 1.38x | 2.78x |
| Independent inputs, 64/32/48 | Guarded batching | 1.32x | 2.31x |
| Independent inputs, 128/64/96 | Guarded batching | 1.27x | 2.27x |
| Independent inputs, 16/1024/16 | Guarded batching | 1.22x | 2.20x |
| Independent inputs, 256/256/256 | Default budget keeps separate matmuls | ~1.00x | ~1.00x |
| One unequal output width | Unbatched fallback | ~1.00x | ~1.00x |

Raising the budget to batch the 256-cubed case gave about 1.07x synchronized and
1.33x queued speedup. Thus the conservative default intentionally leaves some
potential throughput benefit on the table. These six shapes do not establish
an optimal threshold or broad model-level profitability; callers can adjust
it after measuring their workload. The table uses the final publication rerun (seeds 51 and 73); small differences
from the earlier exploratory timings reflect run-to-run variation. All 48
output comparisons passed. Raw timings, generated code, exact harness snapshots,
test logs, source fingerprints and checksums are in the
[archived matmul results](../../benchmarks/inductor/shape_relations/results/matmul_mps_2026_11_15/README.md),
separate from the earlier minimal/XSA measurements. The matmul timing harness
currently supports MPS only.
