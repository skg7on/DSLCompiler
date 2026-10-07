# `llk-bench`: time the synthetic host workload

`llk-bench` currently times a built-in C++ loop, with work dispatched through the
runtime thread pool. It is a useful smoke check of the timing/threading harness.
It does not load MLIR, use a schedule database, invoke the LLK JIT, or execute a
selected mapped kernel. Its GFLOPS numbers are synthetic-workload throughput,
not SwiGLU/GEMM compiler performance.

This manual describes main revision `bb82038` (2026-10-07), checked against
[`tools/llk-bench/llk-bench.cpp`](../../tools/llk-bench/llk-bench.cpp). A mapped
executable and its numerical tests are separate paths; their presence does not
mean this benchmark invokes them.

## Short first run

Run from the repository root after building:

```sh
build/llk-bench --help
build/llk-bench -M=2 -N=16 -K=16 --threads=1 \
  --warmup-ms=10 --measure-ms=20 --reps=2
```

This bounded command exercises the harness without the large default shape.
Output has two lines in this form, with machine-dependent numeric results:

```text
LLK Bench — M=2 N=16 K=16 threads=1
Best: <value> GFLOPS (<value> ms/iter, 2 reps)
```

The angle-bracket values above are placeholders, not reproducible expected
measurements. To keep a log:

```sh
build/llk-bench -M=2 -N=16 -K=16 --threads=1 \
  --warmup-ms=10 --measure-ms=20 --reps=2 > build/bench-small.txt
```

## Options and defaults

| Option | Default | Meaning |
|---|---|---|
| `-M=<int64>` | `128` | Number of input/output rows and parallel tasks. |
| `-N=<int64>` | `4096` | Number of output columns. |
| `-K=<int64>` | `4096` | Input-row extent and inner reduction length. |
| `--threads=<int>` | `8` | Runtime thread-pool worker count. |
| `--warmup-ms=<int>` | `500` | Minimum elapsed warmup window, in milliseconds. |
| `--measure-ms=<int>` | `2000` | Minimum elapsed window for each measurement repetition. |
| `--reps=<int>` | `5` | Number of measurement windows; the best throughput is printed. |
| `--help`, `--version` | — | LLVM command-line help/version information. |

There is no input file, operation, dtype, target, machine profile, seed, schedule,
or output-format option. All data is F32. Use positive shapes, positive thread
count, and at least one repetition. The tool does not validate these ranges;
negative dimensions, overflowed sizes, or invalid counts can fail allocation or
produce misleading output. Zero warmup/measurement windows still execute at
least one invocation because elapsed time is tested after each invocation.

The defaults imply `128×4096×4096` inner work per invocation. Timing windows
are minimums, so a slow invocation can substantially exceed them. Start small
before increasing shape, threads, or repetitions.

## What is timed and how units are calculated

The tool allocates input of `M×K` floats initialized to one and output of `M×N`
floats initialized to zero. Each invocation creates a new `ThreadPool` and
dispatches rows with grain size one. For each output `(m,n)`, the kernel computes
the same reduction:

```text
sum = 0
for k in [0, K):
  sum += input[m, k] * 0.5
output[m, n] = sum
```

There is no weight matrix; all columns of a row repeat the same reduction. The
source counts one multiply and one add per inner iteration, using
`nominal FLOPs = M × N × 2K`. This nominal count is not an instruction count:
the C++ compiler may optimize the loop, and the harness does not inspect the
generated machine code or validate a numerical result.

After warmup, each repetition invokes the loop until its elapsed measurement
window is reached. It calculates:

```text
ms/iter = total elapsed milliseconds / completed invocations
GFLOPS  = nominal FLOPs / (ms/iter × 1e-3) / 1e9
```

`ms/iter` is average wall time per invocation within the winning repetition.
`Best` chooses the repetition with maximum GFLOPS and prints that repetition's
average time. It is not a median or a percentile, and no per-repetition samples
or statistical dispersion are printed. Thread-pool construction, dispatch, and
destruction are inside the timed invocation; initial input/output allocation is
outside it.

## Interpretation and troubleshooting

When comparing harness runs, keep compiler/build flags, host, shape, thread
count, timing windows, and repetition count fixed. For tiny work, pool/dispatch
overhead can dominate. More threads may slow a small shape, and `M` bounds the
available row tasks. Other host activity and frequency changes affect wall time.

| Symptom | Next step |
|---|---|
| A run appears much slower than its requested window | Windows are checked between invocations. Reduce M/N/K so one invocation is short. |
| Very low GFLOPS for tiny input | Pool construction and dispatch are included; increase work only after the smoke check succeeds. |
| `Best: 0 GFLOPS (0 ms/iter, 0 reps)` | Use `--reps` greater than zero; no measurement window ran. |
| Allocation failure or unstable results with unusual values | Use positive, modest dimensions/counts; the CLI has no range/overflow validation. |
| An MLIR file or mapping flag is rejected | This harness has no MLIR/JIT path. Use the compiler/execution APIs and their tests for that workflow. |

A normal completed run returns zero; LLVM option parsing rejects invalid flag
names or values. There is no custom benchmark-result failure threshold.

For modeled cycles and capacity diagnostics, use
[`micro-perf`](performance.md). For schedule enumeration, use
[`llk-tune`](tuning.md), remembering that its shipped CLI is prediction-based.
Neither converts this benchmark's synthetic GFLOPS into calibrated Micro costs.
Production mapped-kernel measurement and calibration remain separate work
under #51/#52.
