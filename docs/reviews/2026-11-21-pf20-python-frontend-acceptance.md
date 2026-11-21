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

Compiler suite and CTest lane registration are recorded below after the
verification run. CI evidence is appended from the pull-request run; the
acceptance job's artifact records the exact lane JUnit files, native counters
and selected group counts.

<!-- PF20-EVIDENCE-PENDING -->

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
