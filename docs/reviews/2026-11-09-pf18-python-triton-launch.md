# PF18 verification record: upstream Triton backend, driver and CPU launcher

Date: 2026-11-09
Issue: [#229](https://github.com/skg7on/DSLCompiler/issues/229)
Parent: [#203](https://github.com/skg7on/DSLCompiler/issues/203)
Plan: `docs/superpowers/plans/2026-10-18-python-frontend-refinement.md` §PF18
Predecessor: [#228](https://github.com/skg7on/DSLCompiler/issues/228) (merged in #285)

## Delivered

`llk.triton` now implements genuine upstream Triton integration:

- `llk.triton._backend.LLKCPUBackend` subclasses upstream `BaseBackend` and
  implements every audited abstract method (`supports_target`, `hash`,
  `parse_options`, `add_stages`, `load_dialects`, `get_module_map`) plus the
  methods upstream `CompiledKernel` consumes (`pack_metadata`,
  `get_codegen_implementation`, `binary_ext`, tensor/int specialization).
- `llk.triton._driver.LLKCPUDriver` subclasses upstream `DriverBase`
  (`is_active`, `map_python_to_cpp_type`, `get_current_target`,
  `get_active_torch_device`, `get_benchmarker`) and supplies the device/stream,
  `utils`, and `launcher_cls` used by upstream's JIT.
- `llk.triton.activate(options)` is a scoped, process-serialized activation of
  the `llk_cpu` target. It saves and restores the previous process-global
  Triton driver on success and on error, and it rejects an unqualified upstream
  version before installing anything.
- `llk.triton.last_launch_report()` and `Activation.last_launch_report()`
  expose launch-time evidence; no native claim is made before a real launch.
- `CPULauncher.bind_triton_template(template, args, grid, options)` binds the
  PF17 `UnboundFrontendTemplate` to concrete tensors, scalars and the runtime
  grid and returns a checked `CompileRequest`.
- `CPULauncher` implements the upstream launcher protocol: it re-exports the
  genuine upstream specialization through PF17, validates the compiled
  artifact against the template, compiles/invokes through the shared PF08/PF09
  service, and records the native report.
- `llk.runtime._buffers` gained an explicit CPU-Torch adapter. Non-CPU devices,
  meta tensors, fake tensors, storage-offset views, non-compact strides and
  unsupported dtypes are rejected before any data pointer is requested.
- `llk.triton._plugin.{compiler,driver}` plus the `triton.backends` entry point
  make the backend discoverable by an installed upstream package. The driver
  reports `is_active() == False`, so a GPU host keeps its own default driver;
  LLK activation is explicit.

## Compiler final bytes are an unbound template

`LLKCPUBackend.binary_ext` is `llktpl`. The final stage emits an
`UnboundLaunchArtifact` (schema version 1) containing:

| Field | Meaning |
|---|---|
| `entry` | upstream entry function |
| `target` | `("llk_cpu", arch, warp_size)`; never a Triton GPU target |
| `upstream_version` | observed Triton version |
| `backend_version` | LLK backend contract version |
| `compile_hash` | upstream compile cache hash |
| `ttir`, `ttir_hash` | genuine TTIR text and its SHA-256 |
| `artifact_identity` | SHA-256 over the canonical body |

`CPUUtils.load_binary` retains that template, validates every field and the
content identity, and reports `machine_code = False` in its metadata. The TTIR
stage runs the same pass pipeline as the qualified PF17 exporter
(`make_ttir`: inliner, tensor-descriptor rewrite, canonicalizer, combine,
reorder-broadcast, CSE, symbol-DCE, loop-unroll), so the compiled artifact's
`.asm["ttir"]` is byte-identical to the exported template TTIR and its hash is
the binding identity between the two halves.

## Launch-time binding

The runtime grid, tensor shapes/strides/dtypes and scalar values are absent at
compile time by design. At a real launch the launcher:

1. Reads the upstream `ASTSource` signature, specialization attributes and
   constexprs.
2. Re-exports the genuine upstream specialization through PF17 in a subprocess
   and validates: source bytes hash, entry, signature order, parameter kinds,
   constexpr values and per-pointer alignment.
3. Validates the compiled artifact against that template (TTIR hash, entry,
   upstream version, schema).
4. Binds each pointer provider: CPU device, rank two, compact row-major,
   zero storage offset, writable output, element dtype, and the artifact's
   deferred alignment constraint. Scalars must fit their artifact integer type.
5. Proves exact launch coverage and normalizes the flat Triton grid into the
   LLK two-dimensional grid:
   - `BLOCK % columns == 0`: block `(rows, columns)`, grid `(rows/BM, 1, 1)`.
   - `columns % BLOCK == 0`: block `(1, BLOCK)`, grid `(rows, columns/BLOCK, 1)`.
6. Translates the accepted flat elementwise dataflow into a checked
   `FrontendProgram`: `program_id`/`range`/`addi`/`muli` become the exact
   affine row/column decomposition, `cmpi` becomes `compare`, `addf`/`mulf`
   become `add`/`mul`, and `tt.load`/`tt.store` become typed `address` +
   `load`/`store` with the original mask.
7. Compiles and invokes through the PF09 `CompilationCache` and the PF08
   checked invocation, holding the caller's providers for the synchronous
   call.

The second cache stage is keyed by the full serialized `CompileRequest`
(program, source identity, template identity, concrete tensor shapes, scalar
values, launch, numerical policy, config snapshot) plus the native environment
identity. Upstream pointer/alignment specialization alone never identifies a
handle: the same upstream artifact was launched on `32x64`, `64x32` and
`16x64` and produced three distinct native identities.

## Native execution evidence

The bound request is the native contract. Because of the coexistence blocker
below, the compatibility lane executes that exact request with the LLK C API in
a Triton-free subprocess and compares real outputs:

```
wide  32x64  grid (2,)  block (16,64)  -> reference, exact x+y
tall  64x32  grid (2,)  block (32,32)  -> reference, exact x+y
short 16x64  grid (1,)  block (16,64)  -> reference, exact x+y
```

All three share one upstream artifact/template and produce three distinct
`native_request_identity` values. This proves the translation, the artifact
validation and the native compile/invoke path; the in-process activation lane
adds the driver/launch wiring when coexistence is available.

## Blocker: native-library coexistence

Upstream Triton 3.6 (the qualified source checkout) statically links its own
LLVM image. The LLK frontend C API links the project's LLVM/MLIR shared
libraries. On this macOS host, loading both images into one process aborts:

| Import order | Fatal error |
|---|---|
| Torch → Triton → LLK C API | `LLVM ERROR: Option 'basic' already exists!` |
| Torch → LLK C API → Triton | `LLVM ERROR: Option 'pbqp' already exists!` |

The same failure occurs when only `libLLVM.24.0git.dylib` is loaded after
`libtriton.so`, so it is the LLVM images, not LLK's entry points. The C API
already hides its own symbols with an export list; the clash is Mach-O weak
symbol coalescing between LLVM's exported weak definitions and Triton's
statically linked copy. The plan explicitly allows this outcome: activation
qualifies both import orders in subprocesses and, on failure, raises
`UnsupportedFeatureError(code="upstream_toolchain_unqualified")` with the exact
observations instead of guessing. The in-process launch tests are skipped with
that reason and the issue remains open for the qualified lane.

This is an environment/build property, not a launcher defect: the launcher and
activation code paths are exercised by the activation/restore tests (with a
qualified-probe stub), and the exact bound request is executed natively in the
cross-process lane.

## Supported boundary (PF18)

- Flat elementwise f32 add/multiply over compact rank-two storage, one scalar
  integer extent and one static zero-based range.
- `BLOCK` must tile the row-major storage into whole rows or whole row
  segments, and the launch grid must cover the declared domain exactly.
- The stock x86-avx2 profile's SRAM capacity limits the block program: on
  `32x64`, `BLOCK=1024`/`2048` compile; `BLOCK=256`/`512` are rejected by the
  native storage plan (a mapping/capacity follow-up, not a launcher bug).
- Padded loads, casts, select dataflow, 2-D program ids, counted loops,
  reductions, atomics and layout conversions remain explicit
  `unsupported_triton_launch` diagnostics. PF19 owns the compatibility corpus.

## Evidence commands

```sh
export LLK_TRITON_PYTHONPATH=/Users/skg7on/Workspace/Projects/triton/python
export LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib
.venv/bin/python -m pytest test/python/compat/test_triton_launch.py -q
# 1 passed (dedicated process; inner lane: 17 passed, 4 skipped)

.venv/bin/python -m pytest test/python/compat/_triton_launch_inner.py -q -rs
# 17 passed, 4 skipped (coexistence gate)

.venv/bin/python -m pytest test/python/compat -q
# 23 passed

.venv/bin/python -m pytest test/python -q
# 301 passed, 9 failed
```

The nine failures are a pre-existing local Inductor C++ compilation issue:
Homebrew `llvm@20`'s libc++ include ordering leaves `NAN`/`INFINITY` undeclared
in Inductor-generated kernels. They are unrelated to this change and pass in
the pinned CI toolchain.

## Handoff

- PF19: build the fixed compatibility corpus on this binder and record the
  capacity/boundary rejections; extend only patterns that reach a verified
  source adapter.
- PF20: register the dedicated launch lane in CI with the qualified Linux
  toolchain, where the two images may coexist; if they do not, the lane keeps
  the `upstream_toolchain_unqualified` diagnostic and the milestone stays open.
