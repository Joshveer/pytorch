# Generalized reduction benchmarks: MPS, November 16, 2026

Two independent processes measured the generalized reduction shape checks on
Apple M5 (24 GB), macOS 27.2.1, Python 3.12.12, native PyTorch
`2.16.0a0+git6345ea6` with editable compiler sources. The machine was on
**battery power**, at 67%/66% charge when the two runs started. Do not conflate
these measurements with the earlier AC-powered benchmark archives.

See the [implementation, results and rerun instructions](../../../../../tools/inductor/general_shape_fusion.md),
[full timing report](report.txt) and [machine-readable summary](summary.json).

## Comparison

The baseline restores the complete reduction-guard module from
`01d544c8af54330e035a2eaf38554b7efef3a274` and calls its original mean/sum pass
through the current post-grad entry point. The fixed mode uses the generalized
pass. Both run identical workloads and inputs in the same native installation;
earlier divisibility and matmul changes remain present in both. This isolates
the new extension, rather than comparing against unmodified upstream PyTorch.

The recorded repository HEAD is the parent commit: measurements were taken
before committing the new Python compiler changes. Source hashes and the diff
in each results file identify the measured compiler. The native library remains
at `6345ea65549d76751f09501580074b700b7c56c6`.

Each process ran all 34 cases with 16 balanced baseline-first/fixed-first paired
blocks. Inputs include eight reduction operations, small/medium/large examples,
unequal shapes, equal element counts with unequal axes, mixed streams, and
already-optimized controls. All **168 eager-output comparisons passed** across
the two runs. No compilation occurred during warmup or timed execution.

Queued blocks repeat the input sequence 32 times and synchronize at the block
boundaries. Synchronized blocks repeat it eight times and synchronize after
every call. Both report time per call. They include allocation, dispatch,
synchronization and device execution; compilation and warmup are excluded.
These are FP32 inference measurements, not training, BF16, CUDA or whole-model
results. Shader-definition counts are not profiler launch measurements.

## Included files

- `run*/results.json`: unchanged raw timing pairs, setup times, correctness status,
  graph counts, code fingerprints and environment records.
- `run*/code/*.py.txt`: byte-for-byte generated wrappers and Metal source.
- `run*/baseline_shape_relation_fusion.py.txt`: exact restored baseline module.
- `summary.json`: pooled median block means and baseline/fixed ratios, per-process
  speedups, setup totals and 95% paired bootstrap intervals. Resampling is within
  each process; intervals do not cover cross-machine or power-mode variation.
- `report.txt`: a compact table of all 34 cases, including unequal cases and
  controls, with gains below 1 reported as slowdowns.
- `recorded_reductions.py.txt`, `recorded_runtime.py.txt`, and
  `recorded_summarize_reductions.py.txt`: exact measurement and analysis scripts.
- `new-tests.log.txt`, `existing-shape-tests.log.txt`, `matmul-tests.log.txt`:
  fresh publication checks, totaling 217 passing tests and one skip.
- `fusion-survey.json`: the earlier static/previous-dynamic/fixed-dynamic survey,
  with all 45 output comparisons passing. It is structural evidence, not timing.
- `sha256.json`: SHA-256 checksums of all other files in this directory.

Mixed streams require two fixed graphs instead of one baseline graph and incur
extra setup cost. Small-case and mixed-axis latency gains can be within noise.
Raw per-process results are retained rather than selecting the fastest run.
Future measurements should use a fresh output directory and an otherwise idle
GPU, matching and recording power mode. The earlier XSA BF16 gradient caveat
remains unresolved; CUDA validation is pending.
