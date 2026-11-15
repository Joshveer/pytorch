# CUDA/Triton handoff

This workflow covers the original minimal examples **and** Cyrus's complete
`xsa_unsqueeze`/`xsa_expand` reproductions, including forward/backward execution,
unequal widths, and mixed equal/unequal inputs. CUDA execution is pending: local
validation of these runners used MPS, not simulated CUDA.

## 1. Use the modified native build

Use a Linux NVIDIA machine and a checkout containing these scripts and the XSA
repair (`884f6091ee917ced7fc75881cf68d11ae2c05979` or a descendant). Use the
appropriate CUDA toolkit/compiler for this PyTorch checkout. Check `nvidia-smi`
and `nvcc --version` before building. The scripts do not install software,
allocate a GPU, contact another machine, or fall back to CPU/MPS.

Start in the repository root. In a new environment, for example:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# This repository script installs the Triton revision pinned by this checkout.
env -u USE_XPU bash scripts/install_triton_wheel.sh

git submodule update --init --recursive
EXPECTED_BUILD_COMMIT=$(git rev-parse HEAD)
CMAKE_PREFIX_PATH="$VIRTUAL_ENV" CMAKE_BUILD_TYPE=Release MAX_JOBS=8 \
USE_CUDA=1 USE_MPS=0 USE_DISTRIBUTED=0 BUILD_TEST=0 TORCH_NATIVE_AOT=0 \
  python -m pip install -e . -v --no-build-isolation
```

Adjust `MAX_JOBS` to host memory. Build on the target GPU host when possible;
otherwise set `TORCH_CUDA_ARCH_LIST` for the target GPU. `TORCH_NATIVE_AOT=0`
disables the separate embedded DSL-kernel build, not Inductor's Triton JIT.
Do not use an ordinary released torch wheel: it does not contain these fixes.

The baseline revision must exist locally:

```sh
git cat-file -e c79633996ebc2114482ae5a4ff3cb409d79a8a9c:torch/_inductor/sizevars.py
# If missing in a shallow checkout:
git fetch origin c79633996ebc2114482ae5a4ff3cb409d79a8a9c
```

Keep the expected build commit from the build, rather than silently accepting
whatever torch happens to import. `runtime.py` checks Python/native commit
agreement and that the imported compiler belongs to this checkout. It records
the current compiler source hashes separately, because editable Python files
can change without rebuilding the native library.

## 2. Preflight and full run

Select one physical GPU with `CUDA_VISIBLE_DEVICES`; it is logical device zero
inside the process. Run on an otherwise idle GPU. Use a fresh output directory.

```sh
BENCH=benchmarks/inductor/shape_relations
OUT=agent_space/cuda-shape-relations

CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH .venv/bin/python "$BENCH/runtime.py" \
  --device cuda --expected-build-commit "$EXPECTED_BUILD_COMMIT" \
  --output agent_space/cuda-shape-preflight.json

CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH .venv/bin/python "$BENCH/run_cuda.py" \
  --expected-build-commit "$EXPECTED_BUILD_COMMIT" --output-dir "$OUT"
```

Add `--profile` to save separate warm CUDA profiler traces outside timed blocks.
Use `--tests-only` to omit timing. Use `--fp32-only` for intentional FP32-only
coverage, including on GPUs without native BF16 support. A requested full run
on such a GPU omits BF16 and reports that validation needs review.

The driver runs stages serially, with separate processes/cache directories and
logs. It continues collecting diagnostics after individual test failures.
Benchmarks are gated on passing FP32 regression/reproduction checks. It writes
`run.json` with every command, exit code and log path. Exit codes are:

- `0`: all requested checks passed.
- `1`: compilation, unexpected correctness, or regression-test failure.
- `2`: no hard failure, but results need review (for example a preserved BF16
  eager-gradient discrepancy, a fusion-count mismatch, an unavailable requested profile, or unavailable BF16).

A nonzero exit is not an instruction to loosen tolerances. Inspect the JSON and
logs. No GPU or unusable Triton is a hard preflight failure, not a skipped pass.

## 3. What the original baseline means

The baseline is the compiler at
`c79633996ebc2114482ae5a4ff3cb409d79a8a9c`, before this project's changes. It is
not the older `7231f9e` build from the original upstream issue.

Both variants use the same native build. `runtime.load_baseline` compares the
current `SizeVarAllocator` AST with the pinned parent and restores **all seven
changed methods**, not just `simplify`: the simplifier, two simplification-cache
factories, loop simplification, equality checking, divisibility checking and
the general simplification cache. The guarded-equality pass is disabled as well.
Original method/source hashes are saved. Added/removed or decorated methods
cause the adapter to refuse the comparison rather than silently use a partial
baseline. Fixed mode restores the current methods and enables the pass.

For training, the baseline context encloses initial forward **and backward**
compilation. Both graphs are warmed before timing. The code rejects compilation
inside timed blocks. This is a complete Python-compiler ablation for this patch
set, not a second installation or an old PyTorch-version comparison. The patch
set makes no native C++ changes.

The older archived MPS benchmark predates the XSA repair. At that revision only
the simplifier and equality-pass switch needed reverting. Do not combine those
historical runs with new CUDA runs or present their timings as CUDA results.

## 4. Full XSA reproductions, independently

The workload function bodies are copied unchanged from Cyrus's repository at
`275f764ee84812a8db53ff93e915577f9b97b2bc`. The runner preserves its
`B=1, T=2048, HQ=32, HKV=4, D=64`, dynamic annotations (including VJP weights),
three outputs, and compiled forward/backward closure. Static and dynamic cases
use the same seeded inputs. Only the comparison/capture logic is added outside
the workload.

```sh
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 \
TORCH_LOGS=fusion,recompiles,guards .venv/bin/python "$BENCH/xsa.py" \
  --device cuda --expected-build-commit "$EXPECTED_BUILD_COMMIT" \
  --dtype float32 --benchmark --profile --output agent_space/xsa-cuda-fp32/results.json

CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 \
TORCH_LOGS=fusion,recompiles,guards .venv/bin/python "$BENCH/xsa.py" \
  --device cuda --expected-build-commit "$EXPECTED_BUILD_COMMIT" \
  --dtype bfloat16 --output agent_space/xsa-cuda-bf16/results.json
```

Repeat with `--inference` and a different output directory for forward-only
checks. `--rounds 2 --repeats 2` is a functional smoke test, not a performance
result. Defaults are 16 balanced paired timing rounds and 16 queued repetitions;
training steps clear input gradients before each call. Both input-gradient
comparisons are recorded, alongside output errors and baseline comparisons.

The runner records generated kernel definitions per captured graph and whether
fixed dynamic counts match fixed static counts. It does **not** hardcode the
MPS counts as the CUDA answer. Existing regression tests make the stronger
fusion assertions; a CUDA-only failure needs investigation. Generated
kernel definitions are not necessarily runtime launches or autotuner choices.

### BF16 interpretation

FP32 eager checks use `rtol=atol=2e-4`; BF16 checks use `0.02`. On MPS, the
pre-fix dynamic BF16 gradients already differed from eager and sometimes from
compiled static gradients. CUDA may behave differently. The runner never hides
that discrepancy: it records eager and original-baseline comparisons separately.
A baseline-preserved BF16 eager-gradient discrepancy returns `needs_review`,
not a correctness pass. A new difference from the baseline is a failure.
Timing is skipped for a variant with unresolved numerical checks.

The BF16 regression tests include a simplifier-disabled dynamic control. The
full `xsa.py` runner supplies the complete original baseline described above.
Passing baseline comparisons alone does not establish agreement with eager BF16 gradients.

## 5. Minimal benchmarks, unequal widths and mixed streams

The original 17-case matrix includes split softmax/RMSNorm/LayerNorm, equal and
unequal normalization widths, the no-split control, and a mixed-width stream.
Run two processes with the same GPU/build and different case/compile ordering:

```sh
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python "$BENCH/bench.py" --device cuda \
  --expected-build-commit "$EXPECTED_BUILD_COMMIT" \
  --output agent_space/cuda-minimal/run1/results.json --seed 20261115
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python "$BENCH/bench.py" --device cuda \
  --expected-build-commit "$EXPECTED_BUILD_COMMIT" \
  --output agent_space/cuda-minimal/run2/results.json --seed 20261116
.venv/bin/python "$BENCH/summarize.py" --results-dir agent_space/cuda-minimal
```

The default dtype is FP32. `--dtype bfloat16` is optional; put it in a separate
result directory. To smoke-test the key cases, add:

```text
--rounds 2 --only softmax-original,equality-equal-original,equality-unequal-original,equality-mixed-original
```

The unit tests cover equal-first and unequal-first sequences. The mixed benchmark
uses the original equal-first alternating stream. Warm timings exclude compilation
and input generation. CUDA graphs are explicitly disabled, preserving the ordinary
launch/dispatch comparison; graph replay would be a separate experiment.

Queued timing synchronizes around repeated calls; single-call timing synchronizes
after each call. Both are wall times including dispatch/allocations, not kernel-only
CUDA event times. Code, setup costs, graph counts, tolerances and raw paired samples
are saved. The summary refuses to combine different devices, compiler sources,
Triton/CUDA/driver versions, graph/kernel counts, or shapes. Legacy MPS result
files remain readable and unchanged.

## 6. What to return

Keep the complete output directory, especially:

- `run.json`: environment, baseline hashes and stage outcomes.
- `minimal-tests/`, `xsa-tests*/`: full unit-test logs, including any failures.
- `xsa-*/results.json` and `code/`: both functions, both compiler modes, static
  and dynamic, output/VJP errors, counts and optional paired timings.
- `run1/results.json`, `run2/results.json`, `summary.json`: minimal benchmark data.
- `*-trace.json` and `cuda_device_events`, if profiling was requested. These are
  raw CUDA events and can include copies/memsets; inspect them before claiming
  a compute-kernel launch count. CUPTI/profile failures are recorded separately.

Do not drop failing cases or merge results from different hardware. Report the
actual GPU, native commit, current compiler-source hashes, CUDA and Triton
versions, dtype and training/inference mode with the results.
