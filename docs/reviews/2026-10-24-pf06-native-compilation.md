# PF06 shared native frontend compilation verification

**Date:** 2026-10-24 (Asia/Shanghai).

**Scope:** [issue #217](https://github.com/skg7on/DSLCompiler/issues/217), PF06 in
[the frontend plan](../superpowers/plans/2026-10-18-python-frontend-refinement.md).
Base main `a7aa88de24ab4a9b9e02c0a82a21ce283827f9ba` contains merged PF05 PR #244.
Implementation commit `b050a61c8fe6bb6a314e37a5fcb0856f17fe734e` is on `feat/issue217-frontend-compiler`, isolated
worktree `.claude/worktrees/issue217-frontend-compiler`.

## Delivered behavior

`compileFrontend(const FrontendRequest &) -> Expected<FrontendCompilation>`
imports the validated request, loads its frozen target configuration through
existing registered target factories, and runs the **WorkloadContract** overload
of `runMappedTuningSession`. It retains one candidate under explicit deterministic
bounds. The elementwise path does not construct synthetic M/N/K facts. Initial
`top_k` is exactly one; wider requests fail explicitly rather than being ignored.

The new shared `compileMappedTuningSelection` helper re-instantiates the retained
source with the actual machine, checks source/binding/graph/target/plan content
identities and complete ownership receipts, and calls `compileMappedKernel` to
produce a real `MappedExecutable`. Existing CLI/replay and CapabilityReceipt v2
schemas remain authoritative. No optimization/search policy is copied into the
frontend or generic `LLKMapping` library.

`FrontendCompilation` owns the executable, immutable report/inspection strings
and an independent f32 destination allocation. Public buffers are supplied in
signature order filtered by direction; the explicit public/native port map adds
the private no-read input at native port 2. Invocation remains synchronous and
serialized per compilation. `MappedExecutable::invoke` enforces the existing
arity, dtype, shape, compact-layout, allocation and disjointness contract before
entering JIT code. No public input copy, implicit cast or output alias is used.

Target factories are file-backed. The service materializes **only** the request's
owned `config_files` in a unique private temporary directory and removes it after
loading. It never reads live `config_root` as a fallback. That original root and
all frozen bytes remain part of request identity. Normalized duplicate/escaping
snapshot paths reject; no compiler configuration is written into the checkout.

Reports retain native source/target/machine/rules/plan/ABI identities, actual
group/search counts and limits, source choices and ownership receipts, adaptation
and the existing capability receipt. The full native request identity retains
canonical request content plus compiler, selected plan, ABI and executable
identity. Two identical compilations have identical full identities despite
separate temporary directories; changed frozen contents change the identity.

Reference and selected-target modes stay distinct. Selected mode checks the
plugin's declared host ISA and never substitutes reference execution. Counter
names keep their native meaning: `referenceGroupsLowered` may be zero when the
target's semantic emitters lower all groups before the portable reference
backend. The service copies native counts and reports actual execution identity;
it does not invent a positive reference group count.

Owned inspections cover `frontend` (canonical request), `llk` (structured source)
and `micro` (verified bound MicroIR). An opt-in shared compiler capture preserves
the exact bound module once before lowering consumes it; no MLIR object borrows
from the service context. Unavailable stages, including `asm`, return
`inspection_unavailable`. The report is a compilation snapshot, so its invocation
capability remains `not_checked`; the tests below separately invoke native code.

## Red/green and validation

The clean Python baseline passed 104 tests. The native baseline required a
default all-target build: `check-llk` alone omits four registered standalone
execution binary dependencies on a fresh build. After building them, both
portable execution binaries pass, and only the same three arm64 ISA failures
remain. This unrelated dependency omission was not changed by PF06.

New tests first failed on the missing `LLK/Frontend/Compiler.h` service
(`build/pf06-red.log`). A report regression separately failed on missing native
source bindings, then passed after copying actual retained bindings/receipts.
The implementation is exercised with literal independent expected f32 outputs,
not eager Python computation or mocks.

Local native toolchain: LLVM/MLIR **24.0.0git**, Homebrew clang, macOS arm64.
Python: project `.venv` **3.14.8**, real Torch **2.14.1**.

```sh
cmake -S . -B build -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
cmake --build build -j4
build/bin/LLKFrontendTests --gtest_filter='FrontendCompiler.*'
build/bin/LLKFrontendTests
cmake --build build --target check-llk -j4
/Users/skg7on/Workspace/Projects/DSLCompiler/.venv/bin/python -m pytest -q
```

| Lane | Actual result |
| --- | --- |
| PF06 native compiler/invocation tests | 8 passed, zero skips |
| Complete frontend native category | 56 passed, zero skips |
| Full Python suite | 104 passed, zero skips |
| Full native `check-llk` | 1448 registered; 1447 enabled; 3 failed, 6 skipped, 1 disabled |
| Shared candidate/tuning, source CLI/replay and portable execution tests | Passed in the full native run |
| New source license/API/whitespace audit | Passed |

The three failures are `MappedAVX2Acceptance`, `GenericElementwiseAVX2Test` and
`FrameworkPrimitivesAVX2Test`, which explicitly require x86_64 AVX2. They also fail
on the unchanged base. The local host is aarch64; required tests remain enabled,
and these results are not described as a green full suite. Six preexisting skips
cover three legacy JIT and three selected AVX2 fixtures; the staged Triton test
remains disabled. Logs: `build/pf06-review-fix-full-ctest.log`,
`build/pf06-review-fix-frontend.log`, `build/pf06-review-fix-focused.log` and
`build/pf06-python-final.log`.

The new selected-target test has **no skips**: on this arm64 host it verifies
explicit `native_backend_unavailable` without fallback. Required Linux LLVM22 CI
must execute its four full/tiled add/multiply cases, assert positive realized
selected groups and zero reference groups, and run the existing required AVX2
acceptance lane. An unavailable-host negative is not selected execution evidence.

### Native numerical and guard coverage

- Full 8x16 and tiled 4x8 add/multiply requests invoke real native code. Every
  output matches an independent 128-element literal expected table exactly. Both
  inputs vary at every position, so reusing the first tile cannot pass. Input
  buffers and both output canaries remain unchanged outside the write domain.
- Repeated invocation uses new public output buffers with the same private
  destination. Success under the unchanged native disjointness validator proves
  the internal destination does not alias public input/output storage.
- Wrong arity, insufficient allocation bytes, wrong dtype and input/output
  aliasing reject before writes. A valid call still succeeds after these errors.
- Changed source, source binding, target machine content, plan id and ownership
  receipt reject a retained selection before JIT compilation.
- A nonexistent live configuration root works from frozen contents; empty
  snapshots and unsupported retention reject. Full identity is stable across
  repeated compilation and changes when frozen bytes change.
- A public output-first/reordered-input request preserves native port mapping
  and executes correctly. Bound MicroIR reparses/verifies with registered
  dialects after the service context has been destroyed.

## Capability limits and handoff

PF06 supplies native C++ compilation and invocation for the PF05 qualified
full-domain rank-two strict-f32 add/multiply subset. This does not expose a C ABI,
Python decorator/Dynamo launch, cache or upstream Triton execution. Python still
reports `native_backend_unavailable`; PF07 adds the C boundary and PF08 the
public Python launch/buffer checks. Tails/contractions/relaxed numerical modes
remain separately qualified later tasks. The umbrella #203 remains open.


## Independent review and validation fix pass

A fresh whole-branch reviewer inspected `a7aa88d..b050a61`, shared APIs/spec and
pending documentation, and independently reran 7/7 initial PF06 tests. It found
no production correctness defect, but identified missing signed-zero/NaN/Inf
coverage (P2) and periodic inputs that could hide repeated first-tile reads (P3).
Both were treated as Important qualification gaps by their possible effect:
incorrect native results could pass the claimed execution acceptance.

One test fix pass addressed both. The original full/tiled test was observed
**passing** with temporary deliberately faulty output behavior (repeating the
first 4x8 tile, canonicalizing zero signs and replacing non-finite results).
Position-distinct inputs plus literal expected tables and a new strict-f32
edge suite both **failed** under those faults (`build/pf06-review-guards-red.log`).
The temporary production mutations were then removed completely; production
`Compiler.cpp` is unchanged from `b050a61`. All 8 focused tests and 56 frontend
tests pass on the unchanged correct implementation. These red results validate
the improved tests; they are not evidence of a preexisting arithmetic defect.

The edge suite covers full/tiled add and multiply in both backend lanes when
available: negative and positive zero checked by exact bits, NaN classification,
positive/negative Inf, f32 precision beyond bf16, and smallest subnormal results.
NaN sign/payload are not promised. Arm64 verifies selected-host rejection; Linux
must run the selected numerical cases. No tolerances hide these results.

The validation fix is committed as `33c1791` (`test: qualify positional and strict-f32 frontend execution`). No review findings remain deferred. The review set aside unchanged Python
binding/grid/provider behavior, cache/close concurrency, FX/upstream integration
and actual Linux execution: the first groups remain their explicit later-task
responsibilities; compatible-host behavior must be established by required CI.
The current C++ lifetime/checked invocation/mutex were reviewed directly.
