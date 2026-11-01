# PF12 verification record: frontend-json in the compiler CLI

Date: 2026-11-01
Issue: [#223](https://github.com/skg7on/DSLCompiler/issues/223)
Parent: [#203](https://github.com/skg7on/DSLCompiler/issues/203)
Plan: `docs/superpowers/plans/2026-10-18-python-frontend-refinement.md` §PF12
Implementation commit: `f3cf8cb22228d50f2a3b7156f8289dfc13a12fe7`
Pull request: [#278](https://github.com/skg7on/DSLCompiler/pull/278)

## Delivered

`llk-compile` gains an explicit `--input-format=mlir|frontend-json` selector
(default `mlir`). A `frontend-json` input is read as one canonical
`CompileRequest`, parsed by `llk::frontend::parseRequest`, and compiled by
`llk::frontend::compileFrontend` — the same native service the Python package
and the versioned C API call. The CLI prints the service's report JSON on
stdout. `--inspect=frontend|llk|micro` prints one registered inspection stage
verbatim instead; the three stages are the ones the report advertises, and an
unknown stage fails with `inspection_unavailable`.

The request is the single source of compile options. Explicit `--emit`,
`--M`/`--N`/`--K`, `--mapping-*`, `--source-adapter`, `--plan-report` and AOT
flags on the `frontend-json` path fail with `conflicting_compile_options`
rather than being silently ignored or overriding the request. Options left at
their defaults do not conflict.

The MLIR path is unchanged. The unreachable implicit Triton detection was
removed from `runCompilationPipeline`; a `tt.*` input now fails with
`unsupported_input`, names `llk-opt --allow-unregistered-dialect` plus the
`--triton-*` passes as the experimental flow, and names
`--input-format=frontend-json` as the production entry path. No
unregistered-dialect escape flag was added to `llk-compile`.

## Entry-path identity

One materialized request (the checked-in `add-v1` program plus the frozen
`machines/x86-avx2-v2.yaml`, `mapping/x86-avx2/layouts.llkmap` and
`mapping/x86-avx2/rules.llkmap` contents) is compiled twice in
`test/Frontend/cli.py`: once through the CLI process and once through
`LLKFrontendCAPI` in the test process. The report fields that identify what was
compiled are compared; handles, timings and the plan-report text are not.

Observed identity for the 8x16 reference-backend add request:

| Field | Value |
|---|---|
| `backend` | `reference` |
| `source_adapter` | `linalg-elementwise-v1` |
| `workload_identity` | `version=1;ports=4;…;numerical_contract=10:strict-f32;…` |
| `target_hash` | `0x17bbdd0bb651e319` |
| `plan_id` | `0xd97be0129d9e1131` |
| `abi_hash` | `0xc4a811ca1d1fa77b` |
| `inspection_stages` | `frontend`, `llk`, `micro` |
| `stages` | `frontend`, `llk`, `micro`, `lowered`, `executable` |

`frontend`, `llk` and `micro` inspection text is byte-identical between the CLI
and the C API. The pinned assertions are exercised literally:

```python
assert cli_report["abi_hash"] == capi_report["abi_hash"]
assert cli_report["plan_id"] == capi_report["plan_id"]
```

## Rejections

The CLI and C API reject the same malformed inputs with the service's own codes
and messages:

| Input | Code |
|---|---|
| truncated/duplicated JSON | `invalid_frontend_schema` |
| schema `version: 2` | `invalid_frontend_schema` (`only schema version 1 is supported`) |
| unknown opcode | `invalid_frontend_schema` (`unsupported frontend opcode`) |
| `--inspect=asm` | `inspection_unavailable` |
| `--emit` / `--mapping-*` / AOT flag beside a request | `conflicting_compile_options` |
| legacy `tt.*` MLIR in production mode | `unsupported_input` |

## Test evidence

Build (worktree-local, LLVM/MLIR 24.0.0git, frontend C API enabled):

```sh
cmake .. -G Ninja \
    -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build \
    -DLLK_BUILD_PYTHON_CAPI=ON -DCMAKE_BUILD_TYPE=Release
ninja llk-compile LLKFrontendCAPI
```

Red step — before the implementation:

```sh
ctest --test-dir build -R FrontendCLI --output-on-failure
# llk-compile: Unknown command line argument '--input-format=frontend-json'
```

Green step — the CLI, C API, inspection, rejection, default-MLIR and `tt.*`
checks:

```sh
ctest --test-dir build -R FrontendCLI --output-on-failure
# 1/1 passed
python3 test/Frontend/cli.py build/bin/llk-compile \
    build/lib/libLLKFrontendCAPI.dylib "$PWD" build/FrontendCLI
# FrontendCLI: 64 checks passed
```

Affected regression set:

```sh
ctest --test-dir build -R \
  'FrontendCLI|LLKCompileMicroSearch|MicroToLinalgJit|MicroMappingCompile|FrontendCAPI' \
  --output-on-failure
# 14/14 passed
```

Full serial suite:

```sh
ctest --test-dir build --output-on-failure
```

1695 registered, 1691 passed, 4 failed. All four failures are the
`selected-avx2` lanes (`MappedAVX2Acceptance`, `GenericElementwiseAVX2Test`,
`FrameworkPrimitivesAVX2Test`, `DynamicMappedAVX2Acceptance`), whose required
execution asserts an x86_64 AVX2 host; the local host is aarch64. They are CI
lanes, not local evidence. Four legacy JIT tests are skipped and one disabled,
as in the recorded baseline. The new `FrontendCLI` case passed in the same run.

## Limits

This task exposes the shared service; it does not extend the supported program,
target or execution envelope. The reference-backend request compiles an
executable but the CLI does not invoke it, so no numerical or invocation claim
is made here. Selected-target execution, dynamic ABI and Python/Torch entry
paths remain with their own milestones. The frontend-json test requires the
`LLK_BUILD_PYTHON_CAPI` library because the C API comparison is part of the
acceptance; it is registered only when that library is built, rather than
silently skipping.
