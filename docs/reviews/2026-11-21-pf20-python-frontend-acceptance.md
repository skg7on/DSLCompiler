# PF20 verification record: required Python frontend execution and acceptance

Date: 2026-11-21
Issue: [#231](https://github.com/skg7on/DSLCompiler/issues/231)
Parent: [#203](https://github.com/skg7on/DSLCompiler/issues/203) (closed)
Plan: `docs/superpowers/plans/2026-10-18-python-frontend-refinement.md` §PF20
Design: `docs/superpowers/specs/2026-10-18-python-frontend-refinement-design.md` §10

## Delivered

- **One required-execution declaration.**
  `test/Frontend/acceptance_manifest.json` (schema:
  `test/Frontend/acceptance_manifest.schema.json`) names the five required
  lanes, their pytest marker expressions, required dependencies and evidence
  obligations.
- **Registered markers and evidence collection.**
  `test/python/conftest.py` registers `frontend_unit`, `frontend_native`,
  `torch_integration`, `triton_integration` and `selected_avx2`, counts
  `llk.runtime._ffi.native_invoke` calls when a lane run requests evidence,
  and writes `<evidence-dir>/<lane>.json`. The Triton corpus reports the
  native calls made by its isolated worker and the executed selected backend
  and group counts.
- **Enforcing checker.** `test/python/check_required_python_execution.py`
  joins JUnit XML with per-lane evidence and returns these stable codes:
  `required_execution_missing`, `required_lane_skipped`,
  `required_execution_failed`, `native_invocation_missing`,
  `selected_reference_fallback`.
- **Packaged target policy.** The wheel now carries
  `llk/config/machines/*.yaml` and `llk/config/mapping/*/*.llkmap` with
  `llk/resources.json` SHA-256 hashes. `CompileOptions()` validates and
  snapshots the installed tree; a missing or modified file raises
  `native_configuration_missing` or `native_configuration_corrupt` before a
  request is built. `python/llk/torch/_fx_importer.py` prefers the validated
  installed tree over repository discovery.
- **CMake integration.** `LLK_BUILD_PYTHON_CAPI=ON` now installs the shared
  library, `resources.json`, `compatibility.json` and the configuration tree
  under `share/llk`. `LLK_ENABLE_PYTHON_TESTS=ON` registers the five lanes and
  `PythonFrontendAcceptance` with CTest and makes `check-llk` depend on
  `LLKFrontendCAPI`; both options default to `OFF`.
- **Required CI jobs.** `.github/workflows/ci.yml` adds:
  `python-frontend` (Python 3.11 and 3.12, Torch 2.9.0, unit/native/Torch
  lanes), `python-minimal` (no Torch or Triton installed),
  `triton-frontend` (pinned Triton source revision
  `93f05e19a8024938c320d82854e307d602b5a05d`, Triton and Linux AVX2 selected
  lanes, AVX2 asserted) and `frontend-acceptance` (joins all artifacts and
  runs the checker).
- **Documentation and boundary repair.** The new
  `docs/manual/source/tutorials/python-frontend.rst` tutorial is verified
  against the installed defaults; `features`, `python-frontend`,
  `getting-started`, `index`, `tutorials/index` and `README.md` describe the
  lanes and the supported envelope. The disabled
  `TritonStagedExecution.DISABLED_SwiGLUCorrectness` skeleton is now the active
  expected-rejection test `RejectsUnsupportedTritonInlineAssemblyOnCPU`, which
  asserts the `triton_unsupported_op` boundary.

## Required lanes

| Lane | Marker expression | Requires | Native counter | Selected target |
|---|---|---|---|---|
| `frontend_unit` | `frontend_unit and not frontend_native and not torch_integration and not triton_integration` | none | no | no |
| `frontend_native` | `frontend_native and not torch_integration and not triton_integration` | C ABI | yes | no |
| `torch_integration` | `torch_integration` | C ABI, Torch | yes | no |
| `triton_integration` | `triton_integration` | C ABI, pinned Triton | yes | no |
| `selected_avx2` | `selected_avx2` | C ABI, pinned Triton, AVX2 | yes | yes |

## Local verification

Environment: macOS arm64, LLVM/MLIR `24.0.0git` at
`/Users/skg7on/Workspace/Projects/llvm-project/build`, project `.venv` Python
3.14.8, Torch 2.14.1 (local environment; CI pins the 3.11/3.12 + Torch 2.9.0
baseline), Triton 3.6.0 source checkout at
`93f05e19a8024938c320d82854e307d602b5a05d`, build configured with
`-DLLK_BUILD_PYTHON_CAPI=ON`.

Focused commands and results:

```sh
python -m pytest test/python -m "frontend_unit and not frontend_native and not torch_integration and not triton_integration" -q
# 133 passed
python -m pytest test/python -m "frontend_native and not torch_integration and not triton_integration" -q
# 64 passed, native_invocations = 40 (evidence file)
python -m pytest test/python -m "torch_integration" -q
# 103 passed, 9 failed (pre-existing local Inductor C++ include-order issue,
# reproduced on unmodified main; native_invocations = 81)
python -m pytest test/python -m "triton_integration" -q
# 33 passed, native_invocations = 10 (reported by the corpus worker)
python -m pytest test/python -q
# 333 passed, 9 failed (same local Inductor environment issue)
```

The selected lane is not executable on this host: the corpus records
`selected_target_supported: false`, `selected_backend: unavailable`, and no
reference result is substituted. The `triton-frontend` CI job is the first
environment that can execute `selected_avx2`.

Packaging and installation:

```sh
python -m pytest test/python/test_package_install.py test/python/test_package_resources.py -q
# 17 passed
cmake --install build --prefix /tmp/pf20-install
# installs lib/libLLKFrontendCAPI.dylib, share/llk/{resources,compatibility}.json,
# share/llk/config/{machines,mapping}; 12 files match the recorded hashes
```

Strict documentation build:

```sh
ninja -C build docs-html
# sphinx -b html -n -W --keep-going: build succeeded
```

CTest registration and compiler suite:

```sh
cmake -S . -B build -DLLK_ENABLE_PYTHON_TESTS=ON -DPython3_EXECUTABLE=.../.venv/bin/python
ctest -N -R '^PythonFrontend'
# PythonFrontend_frontend_unit, PythonFrontend_frontend_native,
# PythonFrontend_torch_integration, PythonFrontend_triton_integration,
# PythonFrontend_selected_avx2, PythonFrontendAcceptance
ctest -R '^PythonFrontend_(frontend_unit|frontend_native|triton_integration)$'
# 3/3 passed (4.93 s, 3.18 s, 14.05 s)
```

`check-llk` (Debug build, arm64, run serially with
`LLK_ENABLE_PYTHON_TESTS=OFF` to isolate the compiler suite):

```sh
ninja -C build check-llk
# 99% tests passed, 5 failed out of 1698
```

The five local failures are environmental, not product regressions:

- `MappedAVX2Acceptance`, `GenericElementwiseAVX2Test`,
  `FrameworkPrimitivesAVX2Test` and `DynamicMappedAVX2Acceptance` require an
  x86_64 AVX2 host; this arm64 host cannot execute them. The pinned Linux
  `build-and-test` job runs them.
- `MappedTuningSessionTest.PersistsOnlyVerifiedSamplesAndReusesExactRecords`
  hit its 60 s timeout in a Debug build while the Python lanes ran
  concurrently; it is not part of this change's verification path.

Three legacy JIT cases and `MappedAcceptance` remain the documented skips.
The authoritative full compiler suite is the PR's Linux `build-and-test` job
(LLVM/MLIR 22.1.8, Release): it passed in the final evidence run at
`100e9e2` in 9m15s (and at `71e8c25` in 9m17s).

## CI evidence

Run [44927853997](https://github.com/skg7on/DSLCompiler/actions/runs/44927853997)
at `100e9e2f336fef1d8242196cc1c0893fabf0273b` (PR #288) is green on every
check:

| Check | Result |
|---|---|
| `Python frontend (Python 3.11)` | pass, 5m11s |
| `Python frontend (Python 3.12)` | pass, 5m46s |
| `Python frontend without optional dependencies` | pass, 5m02s |
| `Python frontend (pinned Triton and selected AVX2)` | pass, 47m01s |
| `Python frontend acceptance` | pass, 8s |
| `build-and-test` (LLVM/MLIR 22.1.8, Release) | pass, 9m15s |
| analyze, coverage, html, license-headers, CodeQL | pass |

The `frontend-acceptance` job downloaded the lane artifacts, ran
`check_required_python_execution.py`, and reported `errors: []` with:

| Lane | Executed | Failed | Skipped | Native invocations |
|---|---|---|---|---|
| `frontend_unit` | 133 | 0 | 0 | 0 |
| `frontend_native` | 64 | 0 | 0 | 43 |
| `torch_integration` | 112 | 0 | 0 | 90 |
| `triton_integration` | 33 | 0 | 0 | 20 |
| `selected_avx2` | 2 | 0 | 0 | 20 |

Selected evidence from the Linux x86_64/AVX2 job:
`selected_backend = "selected-target"`, `selected_group_count = 19`,
`reference_group_count = 0` for both the Triton and selected lanes. No required
lane skipped, no expected failure was counted as execution, and no reference
result was accepted for selected execution.

Python 3.11 and 3.12 both ran the unit/native/Torch lanes; the dedicated
dependency-free job ran the unit lane with neither Torch nor Triton installed.
The Triton job built the pinned checkout at
`93f05e19a8024938c320d82854e307d602b5a05d`, asserted AVX2, and executed both
the upstream corpus and the selected lane. The corpus executes all ten genuine
fixtures on the selected target with nonzero selected groups and zero
reference groups.

## In-process launch findings

Enabling the required lane on Linux executed the PF18 in-process upstream launch
tests for the first time; on macOS they skip because Triton's static LLVM and
LLK's shared LLVM cannot coexist in one process. The qualification surfaced and
fixed three boundaries:

- Triton's entry-point discovery re-entered the partially initialized LLK
  backend while Triton was being imported. LLK now registers `llk_cpu` only
  through the documented scoped activation instead of a `triton.backends` entry
  point.
- `parse_options` received kernel constexprs in the same mapping as backend
  options; it now consumes only backend fields and leaves argument validation
  to upstream's binder.
- Upstream specialization requires host objects exposing `data_ptr()`; raw
  `memoryview` providers are rejected explicitly by the in-process lane. The
  C API corpus lane remains the supported entry point for buffer-protocol
  providers.
- Template pointer comparison now normalizes Triton's `fp32`/`fp16` names to
  the LLK `f32`/`f16` dtypes.

## Supported boundary

- Reference execution covers the bounded static rank-two f32 elementwise,
  canonical counted matmul, caller-accumulated matmul and fused SwiGLU subset
  through `@llk.jit`, the strict Dynamo backend, Inductor composition,
  registered operators and the pinned upstream Triton corpus.
- Training covers add/multiply/matmul; SwiGLU training, dynamic-shape Python
  graphs, additional upstream Triton releases and non-AVX2 selected execution
  remain explicit diagnostics or follow-up work.
- Selected execution requires a compatible ISA and never falls back to the
  reference backend. A host without AVX2 records the limitation rather than
  substituting results.
