# PF02 — legacy Triton safety verification

Scope: issue #213, based on main `b0d01f1` after PF01 (#233). The design and
PF02 task in `docs/superpowers/{specs,plans}/2026-10-18-python-frontend-refinement*`
remain authoritative. This repairs experimental quoted generic operations;
it does not deliver a Python executable frontend or upstream TTIR integration.

## Safety contract

- All six legacy pass entry points validate the whole module before rewriting.
  Incorrect arities, array lengths, ranks, shapes, pointer/value/mask types,
  physical strides, layout/offsets and unknown semantic attributes fail with
  `triton_invalid_schema`. Signed/unsigned integer elements are rejected because
  arithmetic padding requires signless integers.
- Block-pointer physical strides belong to the compact base memref. Subview
  steps are one, retaining physical row stride 64 rather than producing 4096.
  Sizes are clipped to the remaining base extent; dynamic offsets must satisfy
  `0 <= offset < extent` through `cf.assert` before any access. Constant violations
  fail preflight. Explicit masks are ANDed with the generated boundary mask;
  masked load lanes are zero padded.
- Grid functions must have a single block, index program IDs on axes 0–2,
  sufficient dimension/block arguments, and void results. Escaping results fail
  with `triton_grid_result_unsupported` before mutation. Positive dynamic grid
  dimensions/block sizes are checked before division.
- Atomics, old `llrt.atomic_*` declarations/calls, shared allocation/asynchronous
  copy/await, advance, range/splat, reduction, broadcast and transpose operations
  fail with `triton_unsupported_op`. Scratch ownership and asynchronous semantics
  have no executable qualification, so those old rewrites were removed.
- `triton-cpu-verify` validates supported experimental input;
  `triton-cpu-verify=reject-residual=true` rejects every residual `tt.*` operation
  at the final experimental execution boundary. The full-pipeline fixture uses
  both boundaries. No success test disables MLIR verification.

## Local evidence

Host: macOS arm64; project compiler build against LLVM/MLIR `24.0.0git` at
`/Users/skg7on/Workspace/Projects/llvm-project/build`; E2E tests enabled.
Each build/artifact belongs to the PF02 worktree. Python checks use the existing
project `.venv` (Python 3.14.8, Torch 2.14.1).

```sh
cmake -S . -B build -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
cmake --build build --target LLKFrontendTests llk-opt -j 8
ctest --test-dir build \
  -R 'FrontendLegacy|BlockPointer|GridToForall|TritonCPU|TritonFullPipeline|AtomicToLLRT|SharedMemToScratch' \
  --output-on-failure
```

The RED run against the old implementation had **19 failures in 24 checks**,
including the stride/return regressions and compiler aborts on short arrays,
malformed dot operands and invalid grid axes. The original numerical regression
checks the lowered stride before invocation, so the unsafe old code is never
executed. The final focused run passes **38/38**, with **zero skips**.

The Frontend category has 28 cases with E2E enabled. Five native copy fixtures
cover zero offsets, nonzero offsets, a two-dimensional tail with an all-true user
mask, partial masks and all-false masks. Each checks every output element,
unchanged inputs, untouched output lanes and 16-element guards on each side.
A sixth native case exercises dynamic offsets, checks tail rows and guards,
and requires SIGABRT for negative/out-of-range offsets before memory access.
These execute through the existing host ORC JIT and expanded memref calling ABI;
they do not use mapped selected-target groups or claim selected AVX2 execution.
Native cases are omitted when `LLK_BUILD_E2E_TESTS=OFF`, consistent with that option.

```sh
cmake --build build --target check-llk -j 8
cmake --build build -j 8
ctest --test-dir build -LE selected-avx2 --output-on-failure
/Users/skg7on/Workspace/Projects/DSLCompiler/.venv/bin/python -m pytest -q
python3 scripts/check-license-headers.py
git diff --check
```

The initial unfiltered `check-llk` run exposed the host-incompatible required
`MappedAVX2Acceptance` check and four acceptance executables omitted from its
build dependencies. Building all default targets supplies those executables.
The portable suite passes; its six existing skips and one disabled staged
Triton skeleton remain explicit. The `selected-avx2` lane is excluded locally
because this host is arm64; it must execute separately on compatible Linux
x86_64 CI. The Python suite passes **21/21**, with zero skips. License and diff
checks pass.

An independent review found the signed/unsigned-integer preflight gap. A failing
regression reproduced post-mutation verifier failure for both `si32` and `ui32`;
rejecting those elements makes the regression pass with the module unchanged.
No outstanding Critical/Important review findings remain.

## Remaining boundaries

The PR's revision-pinned CI checks record LLVM/MLIR 22.1.8 Linux validation,
including required selected AVX2 execution. Local host-native copy evidence is
not substituted for that lane. PF03–PF20 and umbrella #203 remain open; packaging,
legacy lowering safety and eager execution do not establish Python compilation.
