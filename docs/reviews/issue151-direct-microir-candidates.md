# Issue #151 — exact direct MicroIR candidates (B2)

Implementation base: `c9cf88b46ead0050887ac628305af56ff2acf38c`.
Implementation/test revision: `5d8932e`.

## Delivered boundary

The `WorkloadContract` overload of `tuning::instantiateCandidate` accepts
parameter-free `ConcreteMicro` sources without `original_workload` or MMA
facts. The legacy `WorkloadShape` overload retains its existing checks.

An explicit symbol selects exactly one top-level kernel; an empty symbol
requires a unique kernel. Unknown/duplicate symbols and a nonzero root ordinal
fail. The private clone retains the full source module, except search directives
are removed from that clone. The selected kernel is checked against every
ordered tensor input/result: static shape, element type, direction, and effect.
Inputs have Read effects; results have Write effects. This does not widen the
runtime ABI or provide source parameter realization.

Numerical semantics use explicit `micro.numerical_contract` string metadata.
Nonempty contracts require an exact match; a present non-string attribute fails
even if the supplied contract is empty. Existing operation-level attributes
remain untouched. A retained `micro.workload_contract` cannot be overwritten
with a conflicting descriptor. Required primitive names must occur in the
selected body and cover all execution/movement names derived from its workload
graph. The existing workload graph still comes from the actual IR.

The kernel clone records `micro.workload_contract`, original full module text
in `source_semantics`, exact `source_symbol`, and source/candidate hashes.
Source identity length-delimits the original module text, canonical contract,
and selected symbol. Changing a diagnostic search-space workload label cannot
dispatch to another operator. Source/contract/selection changes affect identity.
No GEMM substitution, fabricated M/N/K, or replacement workload graph is used.

The plan's example omitted numerical source metadata despite requiring absent
metadata to fail; the fixture explicitly declares it. CMake registration follows
the current per-library layout in `lib/Conversion/MicroMapping/CMakeLists.txt`.
`FrameworkSourceTest` remains outside the E2E guard.

## Verification

Fresh isolated implementation worktree:
`.claude/worktrees/direct-microir-candidates`, branch
`feat/direct-microir-candidates`. Host: Darwin arm64; Clang 20.1.8;
LLVM/MLIR build reports `24.0.0git`.

```sh
cmake -S . -B build -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
cmake --build build --target FrameworkSourceTest CandidateInstantiationTest -j 6
build/bin/FrameworkSourceTest
build/bin/CandidateInstantiationTest
cmake --build build --target check-llk -j 6
```

- Baseline: FrameworkSourceTest and CandidateInstantiationTest passed before
  implementation. New tests failed to compile because the descriptor overload
  was missing. The malformed-numerical-metadata regression separately failed
  before the raw attribute presence check was added.
- FrameworkSourceTest: **33/33 passed** (19 B1 contracts, 14 B2 direct-source
  cases). Includes the named `DirectElementwiseNeedsNoMmaFacts`, ordered port
  mismatches, numerical mismatch/malformed metadata, unknown/ambiguous symbols,
  singleton selection, unrealized bindings, label independence, full identity,
  retained descriptor mismatch, graph-derived primitive completeness, unused
  heterogeneous inputs, and source
  preservation.
- CandidateInstantiationTest: **7/7 passed**, preserving legacy semantic-root,
  source preservation/numerical identity, SwiGLU, Linalg and synthetic cases.
- Full local suite: **1,270 registered; 1,262 passed; one failed CTest target;
  six skipped; one disabled**. `MappedAVX2Acceptance` fails because selected
  execution requires x86_64 AVX2 and this host is aarch64. Its failing GTests
  are `InvokesTheCompilerGeneratedMatmulAndChecksEveryElement`,
  `InvokesTheCompilerGeneratedSwiGLUAndChecksEveryElement`,
  `MapsAndInvokesTwoIndependentVectorOutputs`,
  `LeavesTheCallersBuffersOwnedByTheCaller`,
  `SelectedAvx2ExecutesThePaddedTailFixtures`, and
  `SelectedAvx2BackendExecutesNumerically`.
- Skips: `SwiGLUScalar.JitCompilationSmoke`, `SwiGLUScalar.E2EWithAbiWrapper`,
  `SwiGLUVectorAVX2.Correctness`,
  `MappedAcceptance.SelectedAvx2ExecutesThePaddedTailFixtures`,
  `MappedAcceptance.SelectedAvx2BackendExecutesNumerically`, and the aggregate
  `MappedAcceptance` target. Disabled: `TritonStagedExecution.SwiGLUCorrectness`.

Separate fresh non-JIT configuration:

```sh
cmake -S . -B build-static -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build \
  -DLLK_BUILD_E2E_TESTS=OFF -DLLK_BUILD_TOOLS=OFF
cmake --build build-static --target FrameworkSourceTest -j 6
build-static/bin/FrameworkSourceTest
ctest --test-dir build-static -R '^FrameworkSourceTest$' --output-on-failure
```

**33/33 GTests and 1/1 CTest passed** with E2E and tools disabled.

Source adapters, parameter realization, mapped tuning/CLI integration and frozen
replay additions remain B3–B7. This slice establishes candidate instantiation;
it makes no new hardware execution or general importer claim.

Independent review found the underdeclared-primitives gap; a new test reproduced
it red, then passed with coverage derived from the existing workload graph. The
reviewer inspected the correction and reported no remaining actionable findings.
