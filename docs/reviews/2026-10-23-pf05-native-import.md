# PF05 checked native structured import verification

**Date:** 2026-10-23 (Asia/Shanghai).

**Scope:** [issue #216](https://github.com/skg7on/DSLCompiler/issues/216), PF05 in
[the frontend plan](../superpowers/plans/2026-10-18-python-frontend-refinement.md).
Base main `c2263348b885e94d0b2b5fdf61e6760891d54fdb` contains merged PF04 PR #243.
Implementation commit `bd6ddb9` is on `feat/issue216-native-importer`, isolated
worktree `.claude/worktrees/issue216-native-importer`.

## Delivered boundary

`LLKFrontend` supplies `llk::frontend::parseRequest` and `importProgram`, with
immutable owned `FrontendRequest` and module-owning `ImportedProgram` records.
LLVM JSON decoding is paired with bounded lexical checks: duplicate decoded
keys, fractional integer tokens, invalid number spelling, non-finite numbers,
signed-64 overflow, unknown/missing fields and unsupported versions reject.
Requests have a 16 MiB / 64-level JSON bound. Decoding never reads configuration
files; the full explicit snapshot is owned and retained in request identity.

Typed validation checks SSA order/uniqueness, nested region scope, operation
arity/attributes, shapes, pointer provenance, exact i1 masks, read/write effects,
loop init/iter/yield types and ordered numerical cast metadata. The finite v1
registry includes PF04 exact `neg`; unsupported import semantics reject rather
than changing the producer grammar.

Import currently accepts only two read inputs plus one write output, equal
positive compact rank-two f32 shapes, exactly one add or multiply per element,
and the complete strict-f32 numerical policy. Checked symbolic affine arithmetic
proves every index intermediate, access bounds, all-active masks, matching read
coordinates and one rectangular output bijection. Constant div/rem from live
PF04 flat offsets is simplified only with proven residue bounds. Grid times
BLOCK must equal each axis, and unused third axes must be singleton. No tail,
flattened lane tile, axis permutation, loop/cast/activation or relaxed numerical
import is claimed.

Recovery creates registered `func`, `arith`, `tensor`, `linalg` operations and
verifies the module. The full canonical request is retained as a module attribute.
The existing `linalg-elementwise-v1` adapter describes the native workload; no
new planner or adapter fork is introduced. Source BLOCK domains are singleton
integer choices with roles `axis:d0` / `axis:d1`. Actual adapter realization and
binding-receipt validation prove source ownership, while a changed BLOCK outside
the domain rejects. This realization emits MicroIR; it does not execute it.

The public/native port map preserves signature order, including a fixture with
output first and reversed input order. Native ports are two read inputs, an
internal no-read destination (2), and the write output (3). The destination is
not a public buffer and PF06 must allocate separate owned scratch for it. The
caller-owned MLIR context must outlive the imported module.

## Validation commands and actual results

Local native toolchain: LLVM/MLIR **24.0.0git**, Homebrew clang, macOS arm64.
Python: project `.venv` **3.14.8**, real Torch **2.14.1**.

```sh
cmake -S . -B build -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
cmake --build build --target LLKFrontendTests -j4
build/bin/LLKFrontendTests --gtest_filter='FrontendProgram.*:FrontendImporter.*'
build/bin/LLKFrontendTests
cmake --build build --target check-llk -j4
/Users/skg7on/Workspace/Projects/DSLCompiler/.venv/bin/python -m pytest -q
```

| Lane | Actual result |
| --- | --- |
| New PF05 parser/importer GTests | 20 passed, zero skips |
| Complete frontend native category, including PF02 | 48 passed, zero skips |
| Full Python suite | 104 passed, zero skips |
| Full native `check-llk` | 1440 registered; 1439 enabled; 3 failed, 6 skipped, 1 disabled |
| License headers / deprecated API / whitespace audit | Passed for new sources |

The **same three native failures occur on the unchanged base** (1420 registered):
`MappedAVX2Acceptance`, `GenericElementwiseAVX2Test`,
`FrameworkPrimitivesAVX2Test`. They require x86_64 AVX2; the local host is aarch64.
They remain required failures here, not relabeled as skips or reference successes.
The six preexisting skips are three legacy JIT fixtures and three selected AVX2
fixtures. `TritonStagedExecution.SwiGLUCorrectness` remains disabled. Full-suite
logs are `build/pf05-baseline-ctest.log` and `build/pf05-full-ctest.log`.

Initial tests failed on the missing native frontend headers (`build/pf05-red.log`).
A later numeric regression reproduced LLVM accepting the invalid JSON spelling
`00`, before lexical number grammar was enforced. Passing tests cover the literal
PF03 request, live PF04 add/multiply and nested-loop casts, MLIR
parse/print/parse/verify, canonical request roundtrip, exact large integers and
negative zero. Negative fixtures cover undefined/duplicate/escaping SSA, bad
masks/effects, bad yields/cast metadata, unsafe/non-affine/overflowing/out-of-bounds
indices, false masks, duplicate stores, incomplete/duplicate launch domains,
relaxed numerical policies and zero-result unsupported loops. No fake `tt.*`
operation survives recovery.

Python 3.11.14 and 3.12.12 independently generate the live add, multiply and
nested-loop fixtures from this worktree using `PYTHONPATH=python` and
`test/Frontend/capture_fixture.py`. No Torch or upstream Triton is involved in
that interop check. Test builds require a Python 3.11+ interpreter; the production
native library has no Python runtime dependency.

## Integration and capability limits

The library is registered through `lib/CMakeLists.txt`, which the root already
includes. This follows the repository library layout rather than adding another
root-level target block.

Python native invocation still reports `native_backend_unavailable`.
PF05 does not compile/invoke frontend kernels, report measured performance,
qualify upstream Triton, or implement PF06–PF20. Linux LLVM/MLIR 22.1.8 CI and
CodeQL must verify the new C++ APIs and required selected AVX2 lane before merge.
