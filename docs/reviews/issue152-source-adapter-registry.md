# Issue #152 — source adapter registration and realized bindings (B3)

Original implementation: `f4c91aa119a9868adba30d693ff6100d0236f016`, based on
`0a32836e6830352ea758afb14c98406204087ce8`. Corrective implementation and tested
revision: `b10d7a7c4b5533441f365d8b92da30ee25a21cf3`, in PR #173.

## Delivered boundary

The injected registry rejects null adapters and empty/duplicate keys. Resolution
has explicit unsupported/unique/ambiguous outcomes; an explicit registered key
must accept the selected root. Function symbol and root ordinal select only
accepted top-level roots. Adapter ambiguity is checked at the selected root,
so unrelated roots do not prevent an exact selection. Explicit keys restrict
the eligible roots to that adapter's family.

Every offered parameter has one declared owner and role. Candidate keys and
typed values must match declared domains; duplicate parameter declarations,
unknown/unowned/duplicate-owned parameters, conflicting roles and repeated or
misplaced receipts fail explicitly. Source axes require a named axis role;
mapping placement/layout/path decisions cannot become source decisions through
a port-role override. Source-stage receipts require all source bindings and
leave mapping bindings pending. Complete-stage receipts require both owners.

The generic instantiation path checks the adapter's described contract, preserves
the caller's source through a private clone, and verifies the returned concrete
kernel's ownership, ordered signature, numerical metadata and IR before recording
provenance. An adapter may erase the replaced root/function: source identity is
captured before realization. Full original source, contract, adapter key,
function/root selection and candidate bindings remain available as provenance.

The default production registry stays empty until B4/B5 introduce actual source
families. B6 owns selected-plan receipt production and Complete validation in the
mapped tuner. This increment adds no generic importer, extra generator, operator
dispatch switch, name-based integer inference or runtime ABI extension. The
legacy WorkloadShape path and B2 direct singleton remain supported.

## Audit corrections and regression evidence

The original implementation already had green CI. Independent audit found and
reproduced these gaps before fixes:

- An adapter replacing its source function caused a use-after-free while the
  driver recorded provenance. The regression process exited with signal 11
  (shell exit 139); it now succeeds with the original source symbol retained.
- Foreign kernel pointers, mismatched output dtype/numerical contracts and
  malformed IR were accepted. All variants now fail, preserving both caller
  source and foreign module.
- Empty registration keys, duplicate parameter declarations, unnamed axis roles
  and owner changes hidden behind explicit roles were accepted. Each has a
  failing-then-passing regression with a named diagnostic.
- Unselected ambiguous roots or roots belonging to another unique adapter
  prevented exact ordinal selection. Both cases now resolve only the selected
  root's adapter.
- The original test adapter produced receipts without changing IR. The test-only
  adapter now realizes a declared repetition axis by emitting equivalent repeated
  vector work before recording its receipt. Values 2 and 4 produce 2 and 4 actual
  workload graph nodes under an arbitrarily named parameter. This demonstrates
  stage-specific realization without claiming a production elementwise adapter.

An independent reviewer inspected the corrections and reran the framework tests;
no remaining B3 blockers were found.

## Local verification

Fresh isolated completion worktree: `.claude/worktrees/issue152-completion`,
branch `fix/source-adapter-validation`; the original locked implementation
checkout was left untouched. Darwin arm64, Homebrew Clang 20.1.8, LLVM/MLIR
`24.0.0git`.

```sh
cmake -S . -B build -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
cmake --build build --target FrameworkSourceTest CandidateInstantiationTest LLKPerfTests -j 6
build/bin/FrameworkSourceTest
build/bin/CandidateInstantiationTest
build/bin/LLKPerfTests
cmake --build build --target check-llk -j 4
```

- **FrameworkSourceTest 63/63 passed**: WorkloadContract 19, FrameworkTest 22,
  BindingSchema 12, SourceAdapterRegistry 5, SourceSelection 5.
- **CandidateInstantiationTest 7/7 passed**.
- **LLKPerfTests 188/188 passed**, including existing candidate generator/domain
  behavior. The initial baseline also ran 54 framework cases and 18 selected
  generator/search-space cases successfully before corrective tests were added.
- `git diff --check` and license-header validation passed.
- Full local suite: **1,270 registered; 1,262 passed; one failed CTest target;
  six skipped; one disabled**. The only failure is `MappedAVX2Acceptance`:
  selected x86_64 AVX2 execution cannot run on this aarch64 host. Six of its
  selected-execution GTests fail on that host requirement; its seven portable
  checks pass. The requirement remains mandatory in Linux CI.
- Local skips: `SwiGLUScalar.JitCompilationSmoke`,
  `SwiGLUScalar.E2EWithAbiWrapper`, `SwiGLUVectorAVX2.Correctness`,
  `MappedAcceptance.SelectedAvx2ExecutesThePaddedTailFixtures`,
  `MappedAcceptance.SelectedAvx2BackendExecutesNumerically`, and aggregate
  `MappedAcceptance`. Disabled: `TritonStagedExecution.SwiGLUCorrectness`.

Separate fresh non-JIT configuration:

```sh
cmake -S . -B build/nonjit -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build \
  -DLLK_BUILD_E2E_TESTS=OFF -DLLK_BUILD_TOOLS=OFF
cmake --build build/nonjit --target FrameworkSourceTest -j 4
build/nonjit/bin/FrameworkSourceTest
```

**63/63 passed** with E2E and tools disabled, proving the new registry/schema
checks remain always built.
