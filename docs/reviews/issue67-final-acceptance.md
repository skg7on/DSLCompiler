# Issue #67 final acceptance evidence

Revision-pinned evidence for the twelve normative §29 acceptance criteria of the
[MicroIR-inspired enhancement design](../superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md).
Epic: [#67](https://github.com/skg7on/DSLCompiler/issues/67). Plan:
[improvement plan](../superpowers/plans/2026-10-04-issue67-improvement-plan.md),
[C execution and acceptance](../superpowers/plans/2026-10-04-issue67-c-execution-acceptance.md).
Predecessors: [PR #111 reassessment](2026-10-04-issue67-pr111-status.md),
[post-#108 status](2026-10-04-issue67-post108-status.md).

## Delivery revision

| | |
|---|---|
| delivery revision | `50b8fbe` — the merge of PR #127 ("normalize abstract ownership through target data", Stage C9) |
| applied on top | the C10 tuner robustness fix (see "The defect this stage closed") and this document |
| local configuration | LLVM/MLIR source `24.0.0git` (`llvm-project/build`), Homebrew clang 20.1.8, Darwin arm64 |
| CI configuration | Linux, pinned LLVM 22.1.8 static — PR #127 [`build-and-test` pass](https://github.com/skg7on/DSLCompiler/actions/runs/37487013701/job/112349567338), [`analyze` pass](https://github.com/skg7on/DSLCompiler/actions/runs/37487013389/job/112349566084), [`coverage` pass](https://github.com/skg7on/DSLCompiler/actions/runs/37487013537/job/112349567510), CodeQL pass |

Full registered suite at the revision:

```
100% tests passed, 0 tests failed out of 139
  2 CPU-specific skips  (SwigluScalar, SwiGLUVector)
```

`DocReferences` is the check this stage adds: the workflow and acceptance
documents must reference paths that exist and flags the tools accept, so an
example a reader is told to run is one the tools can actually run.

The two skips are JIT/AVX2-gated: `SwigluScalar.JitCompilationSmoke` reports
"JIT runtime not available" and `SwiGLUVectorAVX2.Correctness` reports "JIT
compilation not available — skipping E2E execution" on this arm64 host. Both run
on the x86 CI runner; neither is a failure, and neither is used as evidence
below for a claim it cannot support.

## How to read a criterion

"The design is realized" hides four different strengths of claim. This document
keeps them apart, because a criterion is only as strong as the weakest level it
actually reaches:

| level | question it answers | evidence class |
|---|---|---|
| **planning** | is a search space emitted, parsed, round-tripped? | FileCheck / round-trip tests |
| **complete materialization** | are *all* selected decisions written into the IR? | strict-binding tests; `require-executable` |
| **target readiness** | can the selected target emitter lower the result? | emitter / lowering tests |
| **actual invocation** | does the compiled program run and compute the right numbers? | numeric execution tests |

**Calibrated prediction** — a fifth level, where the performance model's numbers
match measured hardware — is *not* claimed anywhere here. It is #51/#52's
deliverable, and this document only records that the executable-key hooks exist.

## §29 acceptance matrix

Each row names the registered tests that carry the evidence, the result at the
delivery revision, and the boundary that remains.

| §29 | evidence (registered tests / binaries) | result | limitation |
|---|---|---|---|
| **1.** search space and complete bindings round-trip deterministically | `MicroDialectSearchSpace`, `MicroDialectSearchSpaceInvalid`, `LLKToMicroSearchSpace`, `LLKToMicroSearchSpaceMBucket`, `LLKCompileMicroSearch`, `SearchSpaceLoaderTest`, `MappingSearchBindingTest`, `SearchBindingLoaderTest`, `MicroMappingSearchBinding`, `LLKTuneSearchSpace` | met | Non-MMA and multi-role workloads are evaluated from the facts each constraint names (`Legality.ShapeIndependentConstraintNeedsNoWorkloadFacts`); production measurement is #51/#52. |
| **2.** normalized MachineModel topology | `MachineModelTest`, `MachineModelV2LoaderTest` | met | — |
| **3.** deterministic direct and multi-hop routing | `MappingRoutingTest`, `RouteIdentityTest`, `MappingConnectionsTest` | met | Exact mode branches over the *enumerated* joint connection choices (`CoveringSearch.GatherExactBranchesOverFeedAlternatives`); an exhaustive branch-and-bound oracle is a property test, not a solver guarantee. |
| **4.** target-owned layout legality without generic target branches | `MappingLayoutTest`, `MappingRulesTest`, `MappingGenericAcceleratorTargetTest` (purity + conformance) | met | The generic-Micro purity test asserts the leak set is now **empty** — the closed owner enumeration that carried backend words is gone (Stage C9). |
| **5.** target rules generate `MappingCandidate` objects | `MappingRulesTest`, `MappingCoveringSearchTest`, `GraphRule.*` | met | Bounded fused-graph matching landed in Stage C7; arbitrary subgraph rules beyond the declared pattern stay out of scope. |
| **6.** concrete instances, connections and complete coverings | `MappingPlacementTest`, `MappingConnectionsTest`, `MappingPlanTest`, `MappingPlanBinderTest`, `MappingStoragePlanTest` | met | Per-use layouts survive repeated operand uses (`CoveringSearch.DistinctOperandUsesOfOneValueKeepTheirOwnLayouts`); endpoint ports are carried into the plan (`Connections.CarriesEndpointPorts*`). |
| **7.** deterministic / beam / exact share one interface, honest limits | `MappingCoveringSearchTest`, `MappingPropertiesTest`, `MappingRoutingTest` | met | Caps and unexplored-choice restrictions are reported (`FanOutReportsRouteCapThroughTruncated`, `GatherExactBranchesOverFeedAlternatives`). |
| **8.** selected plan materializes placements, routes, transforms, bundles | `MicroMappingMap`, `MicroMappingVerify`, `MicroMappingUnmaterialized`, `MicroMappingRequireExecutable`, `MicroMappingBarrier`, `MicroMappingBarrierStorage`, `MicroMappingBindPlan*`, `MicroMappingMatmulE2E`, `MicroMappingSwiGLUE2E`, `PlanBinder.MaterializesATileMovementAsATypedTileAsyncCopy`, `PlanBinder.MaterializesASameKindMovementBetweenDistinctNodes` | met | Target readiness is a *separate* gate from materialization: a complete plan containing an unsupported operation fails target readiness rather than claiming compilation. |
| **9.** mapping search and `micro-perf` share MachineModel / route / cost primitives | `MappingCostModelTest`, `MappingPlanReportTest`, `MappingStoragePlanTest`, `MicroPerfCli`, `LLKToMicroPerfRoundTrip`, `MicroMappingReport` | met (static model) | Materialized transforms now produce `EventKind::Transform` events with the shared `estimateTransformCost` estimate (`CostEvent.LayoutTransformEventUsesTheSharedEstimate`). *Calibrated* parity is #51/#52. |
| **10.** AVX2 works while the legacy path remains available | `MappedAcceptance`, `MicroKernelExecution`, `MicroMappingCompileMatmulToExecutable`, `MicroMappingCompileSwiGLUToExecutable`, `LLKToLinalgConversion`, `MicroToLinalgJit`, plus the legacy SwiGLU chains | met at the **host-portable** level; AVX2 *instructions* need an x86 runner | The mapped numeric chains are Linalg/SCF/memref compiled for the running machine, so their numbers are identical on arm64 and x86 — and say nothing about which instruction set ran. AVX2 instruction-level execution is exercised only by `SwiGLUVector` on the x86 CI runner; it skips on this arm64 host. |
| **11.** second target without new generic Micro ODS policy | `MappingGenericAcceleratorTargetTest` (`OwnerVocabularyIsTargetDataNotAnOdsEnum`, `TargetVocabularyStaysOutOfGenericMicroOds`, `DoesNotClaimAnExecutableBackend`), `MicroMappingAcceptancePipeline` (`second-target` chain) | met | The accelerator maps/materials/perf-models without claiming execution on hardware it does not have; that is deliberate, not a gap. |
| **12.** identical inputs produce byte-identical IR and reports | `MappingStableHashTest`, `MicroMappingBindPlanRoundtrip`, `MicroMappingReport`, `MappingPropertiesTest`, `MicroMappingAcceptancePipeline` (repeat-bytes check) | met | The chain runner compares normalized report and IR bytes across two independent searches. |

### Numeric-execution evidence in detail

These are the tests that *invoke* compiled programs and check the answers, not
merely that compilation or symbol lookup succeeded:

| binary | tests | what it proves |
|---|---|---|
| `MappedAcceptance` | 3 | the compiler-generated matmul and fused SwiGLU are mapped onto AVX2, compiled and **invoked**, every output element checked (a full K=64 reduction; `silu(64)×64`); a third test writes the output inside a larger allocation with sentinels to show the kernel touches only its region |
| `MicroKernelExecution` | 8 | the descriptor-pointer ABI: repeat invocation, a buffer starting inside a larger allocation, arity/dtype/stride rejection, missing symbol, and executable lifetime after the module is gone |

### Public-pipeline evidence

`MicroMappingAcceptancePipeline` drives the tools through **public flags only**
(no library import, no internal header) and asserts what one process cannot see:
the export/search round trip, `micro.plan`/`micro.mapping`/`micro.routes` in the
IR, the versioned report naming its machine/layout/rule/target/graph content
hashes, byte-identical repeat runs, `--micro-verify-mapping` acceptance, and
`:micro-perf` events. Its four chains — `vector-add`, `staged-gemm`,
`fused-swiglu`, `second-target` — report ok with configuration:

```
compilerVersion      0.1.0+50b8fbe
machineHash          f0492664a314dc21
layoutLibraryHash    da522d38b5993252
ruleLibraryHash      1e7196d484316e34
targetHash           b3ed17370692dfd1
graphHash            469de4afc4393fc4
```

The four hashes are the content identities that matter: the same machine,
layout, rule, target and graph content reproduce them. `compilerVersion` is
`git describe` of the build; a build of a tree with local modifications appends
`-dirty`, which is why a claim of reproducibility is tied to the revision, not
the string.

The runner's docstring names a fifth chain, `layout-transform`; it is not in the
chain list. That transform is asserted instead by
`PlanBinder.MaterializesALayoutTransformAsATransformOp` and the
`MappingConnectionsTest` transform cases, because its fixture is a graph builder
rather than a `.mlir` file the tools can be pointed at. This is a naming
inconsistency in the runner's prose, recorded here rather than hidden.

## The seven review findings from the PR #111 reassessment

Every finding below was open at `c8f74a9` (Stages A/B/C were still to come) and
is closed at the delivery revision. Stages A (#114/#115) and B (#116) carry the
correctness and materialization fixes; the C stages carry execution.

| finding | status | controlling evidence |
|---|---|---|
| P1 — distinct operand uses of one value lose their layouts | fixed | `CoveringSearch.DistinctOperandUsesOfOneValueKeepTheirOwnLayouts`, `EqualLayoutsOnRepeatedOperandUsesShareOneConnection`, `ALayoutOnAnotherPortIsNotAttributedToTheEdge`, `Connections.CarriesEndpointPorts*`, `Connections.FanOutSplitsConsumersThatNeedDifferentLayouts` |
| P2 — mapped verifier checks names, not full selected legality | fixed | `PlanBinder.RejectsASelectedRuleWhosePredicateNoLongerMatches`, `RejectsTamperedSolvedVectorWidth`, `RejectsTamperedLayoutFamily`, `MappedFixtureVerifiesWithFullRuleLegality` |
| P2 — an arbitrary movement stamp bypasses completeness | fixed | `PlanBinder.ValueStampCannotExemptAnUnmappedComputeOperation` (the negative control), `MaterializedMovementKeepsItsCompletenessExemption`, `RejectsAMovementWhoseConnectionStampIsAbsent` |
| P2 — exact-mode gather choices remain undisclosed | fixed | `CoveringSearch.GatherExactBranchesOverFeedAlternatives`, `FanOutReportsRouteCapThroughTruncated` |
| P2 — shape-independent constraints wrongly require an MMA | fixed | `Legality.ShapeIndependentConstraintNeedsNoWorkloadFacts`, `EveryContractionIsCheckedForMmaCompatibility`, `TailConstraintReadsTheOriginalWorkload` |
| P2 — materialized transforms disappear from performance analysis | fixed | `CostEvent.LayoutTransformEventUsesTheSharedEstimate`, `CostModel.TransformCostScalesWithTheSelectedCapability`, and `micro.transform` handling in `lib/Perf/MicroDAG.cpp` |
| P2 — unsupported offset views lower as identity | fixed | `MicroToLinalg` (`@out_of_bounds_window` rejected; `@explicit_zero_offsets` and nonzero-offset slices lower to `tensor.extract_slice`), `MicroToLinalgInvalid` |

## The defect this stage closed

While running the workflow guide's commands at the delivery revision, `llk-tune`
*aborted* on a compiler-generated search space:

```
error: blocked layout requires a block parameter
Assertion failed: ... verifyInvariants ... StorageUniquerSupport.h:180
```

`llk-compile --emit=micro-search` advertised a bare `blocked` tile-layout choice,
and the binder built `#micro.layout<blocked>` without the block parameter the
kind requires; the attribute's verifier failed and, through the asserting
`LayoutAttr::get`, took the process down. The fix makes an unrealizable layout a
**rejected candidate with a reason** (the C6 contract) instead of a crash, stops
the export advertising a choice nothing can bind, and rejects such a schedule in
the concrete export as unsupported input. Regression:
`CandidateBinding.RejectsALayoutTheBinderCannotRealize`. See the workflow guide
for the runnable tuning invocation and the candidate-cap caveat.

## Supported-mode boundaries

State these plainly so no reader infers a capability from a neighbouring one:

- **Mapping disabled** keeps the legacy LLK→Linalg→Vector→LLVM→JIT path
  unchanged; the acceptance chains assert the mapping-disabled export is
  deterministic and the legacy suite (`LLKToLinalgConversion`, `MicroToLinalgJit`,
  the SwiGLU chains) still passes.
- **Mapped AVX2 execution** is host-portable: the selected plan compiles to
  Linalg/SCF/memref and runs wherever the host can JIT it. This is *not* evidence
  about AVX2 instructions.
- **The generic accelerator** maps, materials and perf-models; it does not
  execute, and no test claims it does.
- **`llk-tune`** enumerates the whole declared space by default
  (`--max-candidates=0`); large problem shapes need a cap. Layout kinds a binder
  cannot realize (`blocked`, `swizzled`) are reported as rejected candidates.

## Schema and version migration

- **MachineModel** is `llk.machine.v2`. A file may declare a *minor* above the
  supported one to carry forward-compatible keys; unknown keys are tolerated only
  then. A target's own owner labels map onto the abstract classes
  `group`/`worker`/`vector`/`matrix`/`transfer` through `refines` on its
  executors, compute capabilities and transfer engines; an undeclared spelling is
  unknown, and two conflicting declarations are ambiguous. `#micro.owner` and
  `#micro.map` are open symbols, so legacy text parses as unresolved data.
- **Plan report** is schema version 2; `readPlanReport` checks the graph and
  target content hashes before any executable binding, and searches reconstruct
  data rather than executable code.
- **Source-graph hash** is defined over canonical pre-materialization workload
  semantics; the versioned report names the machine, layout, rule, target and
  graph hashes it searched.

## #51 / #52 handoff

This epic provides the *executable candidate and key/event hooks*; it does not
claim production measurement, calibration or prediction validation, which remain
#51/#52's work:

- the optional measurement callback accepts a verified `MappedExecutable` plus
  invocation inputs (C6);
- measurement keys carry the full B8 event identity plus target/model/ABI content
  identity;
- `LatencyProvider` has a static-cost fallback, and `lookupCycles` defaults to
  nullopt.

What is *not* done: persisting measured records, fitting calibration, and
validating predictions against measured hardware. A measured number that
disagrees with the static model is #51/#52's to reconcile, not a defect here.

## Reproducing

```sh
cmake -S . -B build -G Ninja -DLLVM_PROJECT_BUILD_DIR=<llvm-project>/build
cmake --build build -j 6
ctest --test-dir build --output-on-failure -j 6

# the numeric chains
build/MappedAcceptance
build/MicroKernelExecution

# the public-flag pipelines (prints its configuration block)
python3 test/Conversion/MicroMapping/acceptance_pipeline.py \
  build/llk-opt build/llk-compile build/micro-perf "$PWD" build/evidence
```

Tracking: issue #67 remains the open tracker until its closure is authorized
separately; this document is evidence, not the closure decision. Where a
criterion is listed as "met" above, that is a claim about the tests named in its
row at the delivery revision — not about the epic's issue text, which is
historical.
