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

## Local validation

Source base: `c79633996ebc2114482ae5a4ff3cb409d79a8a9c` (`2.16.0a0`).
Executed runtime: PyTorch `2.15.0`, commit
`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`, Python 3.12.12, macOS arm64.
Numerical tests ran on CPU and actual MPS/Metal. CUDA/Triton testing is deferred.

The local runner overlays the edited simplifier, new pass, configuration entry
and exact post-grad call site onto the installed runtime. It does not change
installed files and does not establish that a complete 2.16 source build passes.

| MPS forward case, FP32 | Before | After |
| --- | ---: | ---: |
| Six dynamic split-head variants | 2 | 1 |
| Original dynamic equality workload, equal widths | 2 | 1 |
| Original dynamic equality workload, unequal widths | 2 | 2 |

Counts are generated kernel definitions, not measured launches or speedups.
The final local suite passed 50 tests, with one CPU kernel-count test skipped.
The configured lintrunner checks passed, including Pyrefly.
Tests cover:

- Outputs and random-weight gradients for both equal-first and unequal-first
  sequences, contiguous and stride-2 inputs, with multiple width values.
- Reuse of the two GPU specializations after the initial equal/unequal calls.
- Correctness after Dynamo resets and verified Inductor disk-cache hits.
- Zero rows, singleton rows and singleton widths; disabling the new pass.
- Guard creation for backed widths and refusal for unbacked, frozen and AOT
  cases; missing and unrelated divisibility facts; split-head forward/backward.

With a built version of this branch, run:

```sh
python test/inductor/test_shape_relation_fusion.py -v
```

For the installed-runtime overlay used locally, use an environment containing
the matching installed torch and its test dependencies (including expecttest):

```sh
TORCHINDUCTOR_COMPILE_THREADS=1 \
  python tools/inductor/shape_relation_local_test.py -v
```

The runner prints the actual runtime path/version. If using `PYTHONPATH` to
select a cached wheel, keep the source checkout out of the import search path.
Do not report this overlay as a source build. A full fork build, CUDA execution,
BF16 coverage and runtime/compilation-cost measurements remain outstanding.

Prepared with an AI coding assistant for the project; not submitted upstream.
