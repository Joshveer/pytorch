# Recorded symbolic-matmul MPS results

These are two final publication runs on Apple M5 (24 GB, AC power), using the
native editable PyTorch `2.16.0a0+git6345ea6` build and the compiler changes
published alongside this folder. The native build commit is
`6345ea65549d76751f09501580074b700b7c56c6`. Compiler source hashes were checked
against the publication sources. The `repository_head` in the JSON is the parent
commit because the Python changes were measured before committing them.

See the [usage and interpretation notes](../../../../../tools/inductor/symbolic_batch_linear.md)
for enabling the pass, running tests, the result table, limits and rerun commands.

## Included evidence

- `run1/results.json` and `run2/results.json`: unmodified raw timings, environment,
  source hashes, correctness checks, setup costs and wrapper counts. Seeds are
  51 and 73. Both processes ran all six cases and four modes, with 16 timing
  blocks each. All 48 output comparisons passed.
- `run*/code/*.py.txt`: byte-for-byte generated code, with `.txt` appended to
  prevent source formatters from changing the evidence. These correspond to
  case/mode names in each run's JSON.
- `summary.json`: for each case, mode and timing metric, pool the 32 block
  means from both processes and take their median. Speedup is the original
  median divided by the selected mode's median. `call_counts` retains run 1's
  full record, including its setup time; setup time is not a pooled measurement.
- `recorded_matmul.py.txt` and `recorded_runtime.py.txt`: exact harness and
  environment helper used for these runs.
- `tests-final.log`: 39 new tests passed, including outputs, gradients, unequal
  widths, size-limit crossings and bounded graph reuse.
- `existing-shape-final.log`: 79 existing minimal/XSA tests passed, one skipped.
- `existing-batch-tests.log`: four existing batch-fusion tests passed.
- `sha256.json`: SHA-256 checksums of every other file in this result directory.

The original mode restores the batch-linear rule from `986da9d`, within the same
native installation, using unchanged inputs. It is not a baseline that removes
all earlier shape-relation changes. The proof-only, guarded (1 MiB) and
large-budget (64 MiB) modes use the current compiler. Timings include packing,
allocation and dispatch; compilation and warmup are excluded. Generated wrapper
call counts are not measured GPU launch counts.

The new publication runs supersede the exploratory timings quoted before
publication. Keep future runs in a new output directory. These measurements
cover FP32 inference on six small synthetic cases; they do not establish
training or whole-model speedups. CUDA execution is pending. The earlier XSA
BF16 eager-gradient discrepancy is not resolved by the new linear tests.
