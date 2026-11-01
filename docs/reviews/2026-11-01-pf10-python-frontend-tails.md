# PF10 bounded elementwise tails verification

**Date:** 2026-11-01 (Asia/Shanghai).

**Scope:** [issue #221](https://github.com/skg7on/DSLCompiler/issues/221), PF10
in [the frontend plan](../superpowers/plans/2026-10-18-python-frontend-refinement.md).
Base `main` `7edc153` contains merged PF09 (PR #275). Implementation commit
`fccf7f0` is on `feat/python-frontend-tails` in the isolated worktree
`.claude/worktrees/feat+python-frontend-tails`, which owns its own `build/`.

## Delivered behavior

A launch whose grid rounds the domain up -- 33x65 or 1x7 with 8x16 blocks --
is now importable, but only through a declared and proved tail policy. The
divisibility gate is not removed: a non-dividing tile still rejects unless the
source carries the importer's proof for exactly that block.

`lib/Frontend/Importer.cpp` proves the rounded launch before emitting IR:

- Coverage must be exactly `ceil(extent / block)` on each axis. An
  under-covering, over-provisioned or non-rank-two grid still fails
  `invalid_launch_coverage`.
- Mask facts are no longer only constant `true`/`false`. A comparison of a
  canonical lane expression against a positive bound records
  `program_id[axis] * block + lane < bound`; `select` combines those
  conjunctions with the same `&`/`|` lowering the AST producer emits. A mask
  that is not exactly the rounded domain's in-bounds predicate -- constant
  true, all false, one row short, or one row too wide -- fails
  `unsupported_tail_policy`.
- A `div`/`rem` whose residue can leave `[0, divisor)` is the true
  quotient/remainder only on lanes the mask selects. The recovered address
  carries that residue requirement, and the load/store that consumes it must
  discharge it with an identical mask term. Coordinates that start before the
  domain (a negative offset) remain `invalid_launch_coverage`.
- Both reads and the write must use the canonical rectangular block-lane map,
  so every active lane reads and writes exactly the element the source names.
  Together with the exactly-covering grid and the in-bounds mask, every output
  element is written once and no access leaves the tensor.

The proof is an explicit source fact: the imported `linalg.generic` carries
`llk.tail_policy = "masked-edge"` and `llk.tail_block = [B0, B1]`, and the
frontend search space gains a symbolic source-owned
`frontend_tail_policy = "masked-edge"` parameter.

`lib/Conversion/Workload/ElementwiseSourceAdapter.cpp` realizes that fact:

- The present `source_legality: tile_divisibility` rejection still applies when
  no tail policy is declared, and to any declared policy that does not name the
  proved block.
- A proved block decomposes into the interior rectangle plus the right, bottom
  and corner rectangles. Each edge rectangle keeps the block's compile-time tile
  capacity and states its exact static valid extents -- for 33x65/8x16 they are
  8x16, 8x1, 1x16 and 1x1 -- so `micro.tile_load` copies only in-bounds source
  elements and the clipped `micro.tile_store` writes only the declared domain.
  No view is positioned outside its source and repaired by a later mask.
- A rounded program walks its rectangles with temporal loops: every rectangle
  reuses the previous one's staged, accumulator and carried-output storage.
  This keeps a 5x5 grid of 8x16 tiles inside the AVX2 profile's 8192-byte
  accumulator and 32768-byte SRAM capacities. A divisible source keeps the
  original worker-lane spatial nesting byte for byte.
- The contract requires `micro.tile_load` (instead of `micro.tile_async_copy`)
  for a tail source, and the source receipt includes the tail parameter, so the
  policy is evidenced rather than assumed.

`python/llk/runtime/_jit.py` accepts exactly `ceil(shape / block)` blocks and
still rejects partial or over-provisioned grids. The native importer remains the
authority on whether the declared mask is safe.

## Red/green

The checks were written first. Before the extension, the rounded launches
failed and the unproved-mask cases reported the old coverage code:

```text
$ ./build/bin/LLKFrontendTests --gtest_filter='FrontendTails.*'
3 FAILED TESTS
  FrontendTails.ReferenceExecutesRoundedBlocksWithGuards
  FrontendTails.SelectedTargetRealizesEdgesOrReportsUnavailableHost
  FrontendTails.RejectsUnprovedMasksAndMalformedAccesses

$ "$LLK_PYTHON" -m pytest test/python/test_frontend_tails.py -q
7 failed, 2 passed
```

After the implementation the focused commands pass:

```text
$ ./build/bin/LLKFrontendTests --gtest_filter='FrontendTails.*'
4 tests from 1 test suite, 4 passed

$ LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib \
    "$LLK_PYTHON" -m pytest test/python/test_frontend_tails.py -q
9 passed
```

The new C++ cases are registered as four CTest entries through the audited
`test/Frontend/CMakeLists.txt` category. The fixtures are produced at build
time by the real PF04 producer (`test/Frontend/capture_fixture.py`), not
hand-written JSON.

## Numerical and guard evidence

`test/Frontend/tails.cpp` and `test/python/test_frontend_tails.py` exercise
33x65 and 1x7 f32 add and multiply with 8x16 blocks and the declared
`(rows < M) & (cols < N)` mask. Expected values are computed independently in
the test from the same inputs. Both suites assert:

- exact output values over the whole 33x65 and 1x7 domains (no tolerance);
- guard regions on both sides of every payload still hold their sentinel after
  invocation, so the rounded blocks wrote nothing outside the domain;
- inputs are unchanged;
- `micro_ir` contains `micro.tile_load` and `valid[...]`, including the 8x1,
  1x16 and 1x1 edge rectangles.

The C++ cases additionally invoke twice on one executable (its own scratch is
reused safely), and assert
`source_bindings["frontend_tail_policy"] == "masked-edge"` plus the matching
source receipt entry.

Negative evidence: an unmasked rounded launch, an all-false mask, a mask one
row short, and a mask one row too wide all reject with
`unsupported_tail_policy`; a negative element offset and an under-covering grid
reject with `invalid_launch_coverage`.

## Backend identities

Reference execution (the local arm64 host): `backend = reference`,
`source_adapter = linalg-elementwise-v1`, `selected_group_count = 0`, and the
bound plan contains the bounded movement above.

Selected execution on this `arm64-apple-darwin27.0.0` host is explicitly
unavailable: `compileFrontend` returns `native_backend_unavailable` and never
runs the reference backend under a selected-target request. The C++ test
asserts exactly that on an unqualified host; on the CI x86_64 AVX2 lane it
requires the selected edge realization (`selected_group_count > 0`,
`reference_group_count == 0`, exact values and intact guards) or no
compilation at all.

The qualified CI lane provided that evidence. In PR #276 run `41270469108` the
x86_64 AVX2 host passed
`FrontendTails.SelectedTargetRealizesEdgesOrReportsUnavailableHost` (0.35 s),
so the selected backend realized the rounded 33x65 and 1x7 edge kernels with
exact outputs, intact guards, `selected_group_count > 0` and
`reference_group_count == 0` -- no reference fallback.

## Full suite

The full serial CTest run registers 1,682 cases and passes 1,678 (99%). The four
failures are `MappedAVX2Acceptance`, `GenericElementwiseAVX2Test`,
`FrameworkPrimitivesAVX2Test` and `DynamicMappedAVX2Acceptance` -- the same
`selected-avx2` host-ISA failures PF09 recorded on this arm64 machine. Five
tests did not run: the three documented legacy JIT skips
(`SwiGLUScalar.JitCompilationSmoke`, `SwiGLUScalar.E2EWithAbiWrapper`,
`SwiGLUVectorAVX2.Correctness`), one disabled test
(`TritonStagedExecution.SwiGLUCorrectness`) and the skipped
`MappedAcceptance`. No test outside that pre-existing set regressed. The four
new `FrontendTails.*` cases are registered (CTest 1657-1660) and pass through
`ctest -R '^FrontendTails\.'` as well.

- Full `LLKFrontendTests`: 60 passed (56 before PF10 + 4 new).
- `LLKTuningTests --gtest_filter='ElementwiseSourceAdapter*'`: 11 passed,
  including `NondividingTilesReject`, so the divisibility gate still rejects a
  generic non-dividing source.
- Full Python suite (`python -m pytest -q` with
  `LLK_FRONTEND_LIBRARY`): 169 passed (160 before PF10 + 9 new).

Local toolchain: LLVM/MLIR 24.0.0git, Homebrew clang 20.1.8, macOS arm64
(host triple `arm64-apple-darwin27.0.0`), Python 3.14.8, reference backend and
the AVX2 machine profile. CI is the x86_64 AVX2 lane that can execute the
selected path; the local selected check can only assert the explicit
unavailable-host rejection.

CI run `41270469108` (PR #276, Ubuntu 24.04 x86_64 AVX2, LLVM/MLIR 22.1.8)
passed every job: license headers, docs html, build and test, selected AVX2
acceptance, coverage and CodeQL. Its `ctest` run reported 100% -- 1,683 passed,
0 failed -- with `FrontendTails.*` included as CTest 1658-1661.
