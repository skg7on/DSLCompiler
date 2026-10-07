# `llk-compile`: export and compile kernels

`llk-compile` reads an MLIR module and either exports MicroIR, runs the legacy CPU lowering pipeline, or maps a concrete Micro kernel through a registered target package. It is a compilation and inspection driver: successful JIT compilation does **not** invoke the kernel or check its numerical result.

Start with the [first-kernel tutorial](../tutorials/01-first-kernel.md) and [mapping tutorial](../tutorials/02-mapping.md). All commands below run from the repository root after the [build instructions](../getting-started.md); `build/` is the chosen build directory. See the [documentation index](../README.md) for the other tools.

## Invocation and outputs

```sh
build/llk-compile [options] input.mlir
build/llk-compile --help
mkdir -p build/tutorial
```

The input filename is required. Textual output goes to stdout and diagnostics go to stderr. Use shell redirection to save an artifact; this driver has no `-o` output option. Errors return a nonzero exit status.

Without `--mapping-target`, `--emit` chooses the path:

| Value | Default? | Work performed | Successful stdout |
|---|---|---|---|
| `llvm` | Yes | Full legacy lowering, LLVM IR translation, native ORC JIT compilation, entry-symbol lookup | `Compilation successful` |
| `mlir` | No | Full legacy lowering, stopping before JIT compilation | A module in the **MLIR LLVM dialect**, with operations such as `llvm.func` |
| `micro` | No | Concrete LLK-to-Micro export only | Original source functions plus generated `micro.kernel` operations |
| `micro-search` | No | LLK-to-Micro search-space export only | Original source functions plus `micro.search_space`, parameters, constraints, and objectives |

`--emit=llvm` does not print LLVM assembly, emit an object file, or save a standalone executable. `--emit=mlir` means the fully lowered MLIR module, not an intermediate Linalg or Vector stage. Use [`llk-opt`](optimizer.md) to inspect those intermediate stages.

## Driver options

| Option | Default | Meaning |
|---|---|---|
| `--emit=llvm\|mlir\|micro\|micro-search` | `llvm` | Select the non-mapped path described above. |
| `--M=<integer>` | `0` | Append an M value to the legacy JIT cache key. Does not resize or specialize input IR. |
| `--N=<integer>` | `0` | Append an N value to the legacy JIT cache key. Does not change the tensor shapes. |
| `--K=<integer>` | `0` | Append a K value to the legacy JIT cache key. Does not change the contraction dimension. |
| `--mapping-target=<name>` | Empty | Use a registered mapping package instead of the legacy/export path. Registered names are `x86-avx2` and `generic-ai-accel`. |
| `--mapping-root=<directory>` | `.` | Root containing the package's `machines/` and `mapping/` directories. |
| `--mapping-mode=deterministic\|beam\|exact` | `beam` | Choose the mapping search mode when searching a new plan. |
| `--mapping-stop=mapped-micro\|target-lowered\|lowered\|executable` | `executable` | Choose the mapped compilation stopping point. |
| `--mapping-entry=<symbol>` | Empty | Entry symbol to compile, without `@`. Normally omit it for a module with one kernel. |
| `--plan-report=<path>` | Empty | Read a frozen JSON selection instead of searching. Requires the mapping path. |

The `--M`, `--N`, and `--K` options affect only the legacy JIT cache key. Shapes in the MLIR types drive export and compilation. They do not affect Micro export, mapped search, or mapped compilation. The corresponding shape options on [`llk-tune`](tuning.md) and [`llk-bench`](benchmark.md) have different contracts.

When a nonempty `--mapping-target` is supplied, the mapping path takes precedence over every **valid** `--emit` value. Use `--mapping-stop` to select its output. Invalid `--emit` values still fail option validation before mapping starts.

## Export a concrete kernel and a search space

The tutorial sources use static BF16 matrices and FP32 accumulation. The matmul source exports one kernel; SwiGLU exports two projection MMAs followed by the SiLU/multiply/conversion epilogue.

```sh
build/llk-compile --emit=micro docs/examples/matmul.mlir \
  > build/tutorial/matmul.micro.mlir

build/llk-compile --emit=micro-search docs/examples/swiglu.mlir \
  > build/tutorial/swiglu.search.mlir

build/llk-opt build/tutorial/matmul.micro.mlir -o build/tutorial/matmul.roundtrip.mlir
```

The first artifact contains `micro.kernel @matmul_M16_N64_K64`. The second contains a search space around the schedule selected for the SwiGLU workload. Neither command runs a mapping search, lowers to machine code, or invokes a kernel. The original `func.func` remains beside the export, so seeing `llk.*` in the module is expected; the generated concrete kernel itself contains Micro execution operations.

Both exports read `schedules/schedule_db.json` relative to the current directory. An unreadable database or a missing matching entry selects the built-in conservative schedule; the optimizer export also surfaces a warning. That fallback is useful for experiments, but record whether it was used when comparing results. To choose a different database explicitly, use `llk-opt --llk-to-micro="schedule-db=<path>"` or `--llk-to-micro-search-space="schedule-db=<path>"`; `llk-compile` does not expose a driver-level schedule-database option.

The current shared selector filters entries by operation and M bucket, then chooses the first match. It does not select by exact N/K, target, dtype, or math mode even though the database records those fields. The M buckets are `{1}`, `[2,4]`, `[5,16]`, `[17,64]`, and `≥65`. The no-match fallback uses `BM=8`, `BN=32`, `BK=32`, `VM=1`, `VN=4`, vector width 8, four threads, and grain size 1; resolved tiles can be clamped to smaller problem extents.

Current export support is narrower than the set of operations the parser accepts:

- Roots are `llk.matmul` and `llk.fused_swiglu`; SwiGLU activation must be `silu`.
- Operands and result must have static rank-two shapes. Matrix dimensions must agree: `[M,K] × [K,N] → [M,N]`; SwiGLU's second weight must also be `[K,N]`.
- Supported Micro tile element types are `f32`, `f16`, `bf16`, `i32`, and `i8`; the accumulator attribute must name a Micro tile dtype. Parsing/export support does not imply every target has an executable rule for that dtype.
- The selected tile schedule must be realizable by the exporter. A non-divisible tile needs a supported tail policy; layout kinds requiring unresolved block/swizzle parameters are rejected.
- Each supported root adds its own export. The mapping front doors expect one concrete kernel, so use a module with one supported root for the workflows below.
- A file with no supported root can pass the export without adding a kernel or search space. Check the expected operation in the output before moving on.

`--emit=micro-search` describes choices; it does not automatically choose a `micro.candidate`, instantiate every candidate, or map candidates to target rules. See the [tuning manual](tuning.md) and the optimizer's [`candidate=` workflow](optimizer.md#map-at-an-explicit-candidate).

## Legacy CPU compilation

Use the SwiGLU source for this path:

```sh
build/llk-compile --emit=mlir docs/examples/swiglu.mlir \
  > build/tutorial/swiglu.llvm.mlir

build/llk-compile docs/examples/swiglu.mlir
```

The first command saves MLIR LLVM-dialect IR. The second prints `Compilation successful` after the native JIT has compiled an exported function. The JIT and its cache live only for this process; the command does not leave a callable executable file behind.

The fixed legacy pipeline is:

1. For inputs that reach it with `tt.*` operations, Triton structured-compute and grid lowering.
2. LLK-to-Linalg lowering and canonicalization.
3. Shape classification and schedule-selection pass, double-contraction fusion, canonicalization, weight-packing annotation.
4. Tiling/vectorization, canonicalization, forall linearization, and serial/parallel dispatch selection.
5. One-Shot Bufferize, scratch-allocation audit, remaining Linalg-to-loop conversion, and forall-to-LLRT runtime lowering.
6. Vector, SCF, Arith, Math, UB, Func, and MemRef conversion to LLVM dialect; unrealized-cast reconciliation.
7. For `--emit=llvm`, LLVM IR translation and native ORC JIT compilation.

The legacy LLK-to-Linalg pass currently lowers SwiGLU, RoPE, and Attention. It **does not lower `llk.matmul`**. The tutorial matmul therefore works for Micro export and mapped compilation, while a legacy `llk-compile docs/examples/matmul.mlir` currently fails with `op was not bufferized`. Start from a `linalg.matmul` program or use the mapped path for matmul.

There is no `--scalar`, `--vector`, tile-size, thread-count, or driver-level ISA selector for this fixed pipeline. `TileAndVectorize` currently uses fixed values `BM=32`, `BN=64`, `BK=64`, `VM=4`, `VN=8`. Its `schedule-file` and `target-isa` pass options are exposed by `llk-opt`, but the implementation does not consume them. The legacy `select-schedule` pass reports a selected entry; it does not feed these fixed tile constants. The Micro export path does consume its selected schedule. Use the optimizer's [scalar and vector inspection examples](optimizer.md#inspect-scalar-and-vector-lowering) to compare IR stages.

The parser registry is also intentionally narrower than `llk-opt`'s all-dialects registry. The compiler's Triton detection is a lowering hook, not a general promise to parse arbitrary Triton dumps or GPU programs. Use the explicit staged Triton passes in `llk-opt` for that work.

## Map through a registered package

```sh
build/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
  --mapping-mode=exact --mapping-stop=mapped-micro \
  docs/examples/matmul.mlir > build/tutorial/matmul.mapped.log

sed '1d' build/tutorial/matmul.mapped.log > build/tutorial/matmul.mapped.mlir
build/llk-opt build/tutorial/matmul.mapped.mlir -o build/tutorial/matmul.mapped.roundtrip.mlir
```

The compiler exports a concrete kernel if the input does not already contain one, extracts its workload graph, selects a covering plan, binds and verifies it, and stops at the requested stage. When it exports, it removes the LLK source function from the mapped compilation module. It uses the first selected plan returned by the search.

Every successful mapped invocation prints a status line before any IR:

```text
mapping: target=x86-avx2 target-lowered-ops=0 reference-lowered-ops=0
```

The status line makes raw compiler stdout unsuitable as an MLIR file until it is removed. The `sed` command above removes that first line. For a pipeline that should produce directly parseable IR and a separate report, use `llk-opt --micro-map` instead.

The mapped stages are:

| `--mapping-stop` | Result after binding and target verification |
|---|---|
| `mapped-micro` | Prints the bound `micro.kernel` and selected mapping metadata. No emitters have run, so both counts are zero. |
| `target-lowered` | Runs selected target emitter hooks and prints their rewritten Micro form. Verification-only emitters leave operations for the reference bridge. |
| `lowered` | Applies Micro-to-Linalg, One-Shot Bufferize with identity function-boundary layouts, and Linalg-to-loops. Prints SCF/MemRef/backend input IR. |
| `executable` | Applies the descriptor calling convention and native JIT compilation; prints the counts and `Compilation successful`. |

For example:

```sh
build/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
  --mapping-mode=exact --mapping-stop=lowered \
  docs/examples/matmul.mlir > build/tutorial/matmul.lowered.log

build/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
  --mapping-mode=exact docs/examples/matmul.mlir
```

The shipped matmul example currently reports one target-lowered operation and four reference-lowered operations at the later stages. SwiGLU reports three and six. These count emitter-hook dispatches; they do not certify selected vector width, actual AVX2 instructions, physical resource occupancy, or numerical correctness. Native JIT execution is host-portable, and selected AVX2 width is currently lost by the tensor reference bridge. Read the [current gap assessment](../reviews/2026-10-07-issue67-current-gap-assessment.md) before treating mapping success as complete target-execution acceptance.

`--mapping-root` makes the package load these conventional paths:

| Package | Machine profile | Mapping policy |
|---|---|---|
| `x86-avx2` | `machines/x86-avx2-v2.yaml` | [layouts.llkmap](../../mapping/x86-avx2/layouts.llkmap), [rules.llkmap](../../mapping/x86-avx2/rules.llkmap) |
| `generic-ai-accel` | `machines/generic-ai-accel-v2.yaml` | [layouts.llkmap](../../mapping/generic-ai-accel/layouts.llkmap), [rules.llkmap](../../mapping/generic-ai-accel/rules.llkmap) |

The package supplies its emitter keys in code. The compiler does not accept the optimizer's individual `machine=`, `layouts=`, `rules=`, or `emitters=` pass options. Use [`llk-opt`](optimizer.md) when experimenting with policy files. The accelerator's shipped rules cover a smaller operation set; do not assume it can map the tile-level matmul/SwiGLU export or execute accelerator hardware.

Mapped compilation uses an executable binding contract by default. It rejects decisions the binder cannot materialize; the driver has no partial-binding toggle. That contract does not close the known storage, scheduling, selected-width, and fused-lowering gaps described in the assessment. Use the optimizer's report-only mode for analysis without binding.

## Replay a saved selection

The compiler reads a report; it does not write one. Produce a versioned JSON report with [`llk-opt --micro-map report=...`](optimizer.md#save-an-analysis-report-without-binding), then compile the **original concrete Micro input** it searched:

```sh
# Files created by the optimizer manual's report-only example:
build/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
  --plan-report=build/tutorial/matmul.plan.json \
  --mapping-stop=mapped-micro build/tutorial/matmul.micro.mlir \
  > build/tutorial/matmul.replayed.log
```

This reuses the report's frozen `selectedState` instead of re-running search; `--mapping-mode` is not used. The report version, target-content hash, and original workload-graph hash must match. Changing the source workload or target files can invalidate a report. Replaying against already-bound/materialized IR can change the graph, so retain the pre-mapping input along with the report.

`llk-opt --micro-bind-plan` is a different interface: it reads a plan **id**, re-runs the same search, and finds that id among the retained plans. It does not load a report file.

## Runtime boundary

The CLI provides neither input buffers nor an invocation command. To run a mapped kernel from C++, use the executable returned by `compileMappedKernel` and `MappedExecutable::invoke`; see [MappedExecutable.h](../../include/LLK/Runtime/MappedExecutable.h) and the numerical examples in [mapped_acceptance.cpp](../../test/Execution/mapped_acceptance.cpp).

The mapped API records the kernel's declared input/output ports before type erasure, accepts pointers to `MemRef2D` descriptors in port order, and writes caller-owned output buffers. The Micro matmul export has A/B inputs and one result; its source `%init` argument is not an input port in that exported ABI. A raw function pointer from the legacy `JitCache` uses a separate fixed descriptor signature. `Tensor2D`, `MemRef2D`, and `KernelContext` are distinct structures; avoid casting between their addresses.

Current numerical support should be treated as rank-two, row-major descriptor invocation. The descriptor has no runtime dtype tag, unsupported ranks are not comprehensively rejected, and compiler-created temporary allocations are not deterministically freed between mapped calls. A successful CLI compilation checks neither those invocation inputs nor repeated-call allocation behavior. These limitations are recorded in the [current assessment](../reviews/2026-10-07-issue67-current-gap-assessment.md); do not infer that every ABI claim in a header comment is an enforced runtime check.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `Failed to parse ...` or unknown operation/dialect | Validate syntax with `llk-opt`; the compiler registers a narrower set of dialects. Confirm the file is textual MLIR accepted by the intended path. |
| Micro output contains no kernel | The exporter found no `llk.matmul`/`llk.fused_swiglu` root. Export success can be a no-op. |
| Dynamic-shape or rank diagnostic | Edit tensor types to supported static rank-two shapes. `--M/--N/--K` cannot repair the IR. |
| Conservative-schedule warning | Check the current directory and `schedules/schedule_db.json`. Use the optimizer export option for an explicit database. |
| `op was not bufferized` for `llk.matmul` | Use Micro export plus mapping, or provide an already-structured Linalg matmul for the legacy path. |
| Mapping cannot load files | Run from the repository root or set `--mapping-root` to the directory containing both `machines/` and `mapping/`. |
| No complete plan | Check operation/dtype support and inspect an optimizer mapping report. Beam/candidate caps can limit the search; exact mode still has finite internal caps and known feasibility limitations. |
| No selected vector width / fused emitter failure | Inspect the selected rule and bundle. The shipped fused `convert → silu → mul` rule has a known target-lowering integration gap. |
| Report hash/version mismatch | Use the original pre-mapping Micro input and the exact target configuration that produced the report; regenerate after intentional changes. |
| Saved mapped output does not parse | Remove the leading `mapping:` line, or obtain pure MLIR output through `llk-opt`. |
| JIT symbol/translation error | First inspect `--emit=mlir` or `--mapping-stop=lowered`; compilation requires a valid exported entry and all operations to reach the backend. |

For cost estimates use [`micro-perf`](performance.md). For candidate ranking use [`llk-tune`](tuning.md). [`llk-bench`](benchmark.md) runs the current synthetic host harness; it does not time this compiled kernel. Actual kernel timing requires a C++ caller that supplies descriptors, invokes the executable, checks the result, and measures calls.
