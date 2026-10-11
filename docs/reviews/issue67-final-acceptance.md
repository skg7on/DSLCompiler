# Issue #67 interim acceptance evidence

## Issue #129 worktree update (not a release acceptance)

The 2026-10-07 audited-head results below are historical and remain revision-
pinned to that head. The current `feat/issue129-gap-closure` worktree has since
completed R8, T5 and the T6 public-chain work. At T6 commit `5d9a574`, all five
pipeline chains passed, including required-transform materialization,
SRAM→L2→DRAM routing with a 256-byte L2 allocation, frozen selected-plan-ID
replay, parse/print determinism, and planner versus `micro-perf` cycle/DRAM
parity. The T6 targeted CTest selection passed 4/4.

The current local configuration is LLVM/MLIR 24.0.0git on Darwin arm64. The
portable mapped numeric binary ran four tests successfully and skipped its two
selected-AVX2 tests. The separate required-target binary was also run as a
negative control and failed on this unsupported host, as designed. T7 currently
adds fixed-seed xorshift32 signed inputs rounded to stored BF16 and aligned plus
padded-tail matmul checks; fused random, transform numerical, width-specific
vector, and multi-output invocation cases remain open. T8 adds a required
selected-AVX2 CTest and a JUnit outcome parser to the pinned LLVM 22 CI workflow,
but no CI run on this branch head is recorded. T9's current workflow smoke and
DocReferences both pass locally; the full closure matrix remains incomplete.

Therefore issue #67 and the mandatory #129 release gate remain open. In
particular, selected AVX2 invocation has not passed on the candidate release
head, and the full T7 numeric matrix and final whole-suite CI artifact still
need completion. The `WorkflowSmoke` manifest is an allowlisted argv list; its
commands use checked-in fixtures and controlled output paths.

Interim, revision-pinned evidence for the twelve normative §29 acceptance
criteria of the
[MicroIR-inspired enhancement design](../superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md).
Epic: [#67](https://github.com/skg7on/DSLCompiler/issues/67). Plan:
[improvement plan](../superpowers/plans/2026-10-04-issue67-improvement-plan.md),
[C execution and acceptance](../superpowers/plans/2026-10-04-issue67-c-execution-acceptance.md).
Predecessors: [PR #111 reassessment](2026-10-04-issue67-pr111-status.md),
[post-#108 status](2026-10-04-issue67-post108-status.md).

**This document does not establish final acceptance.** A verification review of
the audited revision found ten open, verified gaps, recorded in
[issue #129](https://github.com/skg7on/DSLCompiler/issues/129) and the
[current gap assessment](2026-10-07-issue67-current-gap-assessment.md). Stage
delivery is not the same as passing every acceptance gate: the tests below pass
and demonstrate the capabilities they name, but several mandatory
selected-plan, resource, execution and acceptance contracts remain unproved.
The [October 7 gap-closure plan](../superpowers/plans/2026-10-07-issue129-gap-closure.md)
tracks the repairs, and its acceptance workstream republishes this matrix at the
repaired revision; issue #67 remains open.

```
Status: mandatory issue #67 acceptance remains open under #129.
Portable reference numerical invocation is verified on the audited tree.
Selected AVX2 execution, complete resource feasibility and mapping-driven
tuning require the release gates in the October 7 plan.
```

## Interim gap status ([issue #129](https://github.com/skg7on/DSLCompiler/issues/129))

The audited tree delivers the mapping subsystem's infrastructure and verifies
portable reference numerical execution. It does not yet satisfy the mandatory
selected-target, complete-resource, mapping-driven-tuning and acceptance
contracts. Issue #129 records ten verified gaps, each reproduced on the audited
revision. The table states what has landed and what the release gates in the
[October 7 plan](../superpowers/plans/2026-10-07-issue129-gap-closure.md) still
require.

| #129 gap | Mandatory contract | Landed at the audited revision | Still open (release gate) |
|---|---|---|---|
| **G1** concrete compute identity | a selected compute instance survives report, bind, replay and event scheduling | `CandidateInstance` records each selected compute node; checked metadata and semantic verification landed | `PlanPlacement` carries no compute field, so complete-plan construction drops the selection before bind/score and events normalize every engine to one id (R1, R6, R8, E4) |
| **G2** feasibility before top-K | physical storage is validated before a plan is retained or ranked | deterministic/beam/exact modes share one interface and cap disclosure landed | top-K trims before storage validation and cannot recover a legal, more expensive covering (R3–R8) |
| **G3** scratch and ABI | every compiler allocation is released per invocation; unsupported descriptors are rejected | descriptor-pointer `MappedExecutable`, a checked descriptor ABI and lifetime tests landed | no deallocation stage, so mapped IR allocates without releasing; non-rank-2 ports skip shape/stride checks; the descriptor carries no runtime dtype tag (E1–E3, E9, T7) |
| **G4** width and ISA realization | the selected vector width/ISA reaches executable code | target bundle lowering rewrites the result layout to the chosen `VW`; reference numerical invocation verified | lowering erases the tile to shape/dtype, so width-4 and width-8 lower identically and mapped code is scalar loops (E6–E9, T8) |
| **G5** fused execution | a selected fused rule is lowered as one group with resolved parameters | bounded fused-graph matching landed; the shipped fused rule is selected and bound | the rule declares `VW` but its group carries no solved width, and the lowering path dispatches one operation at a time (E4–E7, T6/T7) |
| **G6** physical storage | storage is complete per hop, with lifetimes and honest capacity | canonical movement/transform/gather/sync infrastructure and a storage plan landed | storage finalization is skipped when any placement lacks a memory binding, and only the route's last hop is reserved, so an SRAM→L2→DRAM route allocates no L2 (R3–R5, R7/R8, T6) |
| **G7** DMA concurrency | each named DMA node schedules against its own concurrency | one shared scheduler is used by planner and perf | each engine-id pool is given the whole machine's transfer-engine count, so an unused DMA node changes a named node's makespan (R2, R5/R6) |
| **G8** static parity | planner and `micro-perf` agree on normalized events, owners, work, traffic, peak and cycles | shared cost primitives, transform events and the versioned report landed | rule-local estimates versus operation formulas disagree on cycles and DRAM bytes, and their capacity checks disagree (R5–R8, T6) |
| **G9** mapping-driven tuner | each candidate's original semantics is mapped, bound, scored and optionally measured | the measurement callback, event keys and ABI identity landed | `llk-tune` analyses a synthetic kernel; it does not instantiate a mapping problem, search, bind or compile the selected plan (T1–T5) |
| **G10** acceptance and claims | five public chains, frozen replay, random/tail/multi-output numerics and non-skipped x86 selected-target evidence | four public chains, portable mapped numerical invocation and reference correctness landed | the required-transform chain is absent, the second-target chain is not canonical tile/two-hop movement, numerical inputs are fixed all-ones, and selected AVX2 execution is unproved (T0, T6–T9) |

The resolved #106 defects retain their credit: the four original defects are
fixed by #107/#108 and #110–#112, and #126/#127 delivered real portable
numerical GEMM/SwiGLU invocation. The matching, materialization, ABI and
fused-matching APIs delivered across #119–#125 are landed infrastructure too.
Fixing named defects and landing APIs is not the same as passing the acceptance
gates the plan names.

## Delivery revision and exact counts

Two revisions matter, and this document previously conflated them.

| | |
|---|---|
| reviewed baseline | `50b8fbe` — the merge of PR #127 ("normalize abstract ownership through target data", Stage C9). CI `build-and-test` ([run 37560345881 / job 112595937288](https://github.com/skg7on/DSLCompiler/actions/runs/37560345881/job/112595937288)) reports **138** registered tests, 100% passed. |
| audited head | `e1b6efc` — the PR #128 head ("publish verified issue 67 acceptance evidence", Stage C10), which carries the useful tuner fix `c4f088d` and this document; merged to main as `693695f` (the PR #128 merge). **139** registered tests. |
| local configuration | LLVM/MLIR source `24.0.0git` (`llvm-project/build`), Homebrew clang 20.1.8, Darwin arm64 |
| local result at the audited head | 139 registered, **137 passed, two skipped** (`SwigluScalar`, `SwiGLUVector`), zero failed |
| head CI (Linux, pinned LLVM 22.1.8 static) | [CI `build-and-test` pass](https://github.com/skg7on/DSLCompiler/actions/runs/37565145333/job/112611023308), [coverage pass](https://github.com/skg7on/DSLCompiler/actions/runs/37565145284), [CodeQL pass](https://github.com/skg7on/DSLCompiler/actions/runs/37565145311) |

The 139-test figure belongs to the audited head, which already includes the
C10-added test; the #127 baseline is 138. ctest prints "100% tests passed, 0
tests failed out of 139" at the head because a *skip* counts as neither passed
nor failed — that line is not evidence the two SwiGLU tests ran.

The two skips are JIT/AVX2-gated: `SwigluScalar.JitCompilationSmoke` reports
"JIT runtime not available" and `SwiGLUVectorAVX2.Correctness` reports "JIT
compilation not available — skipping E2E execution" on this arm64 host. The
audited head's own CI `build-and-test` log reports **both tests skipped on the
Linux x86 runner as well**; an earlier version of this document claimed they run
there. So the Linux static build and coverage pass, but that success does not
include the two legacy SwiGLU runtime tests — and even a passing legacy
`SwiGLUVector` exercises the legacy SIMD path, not selected mapped-target
execution. Neither skip is used as evidence below for a claim it cannot support.

### What `DocReferences` checks

`DocReferences` is the lint this stage adds. It asserts that every repo path a
reader is told to open exists, and that every documented `--flag` is one the
tool's `--help` accepts. It does **not** run the commands inside shell code
fences, and it does not validate the argument values those commands pass —
"the references resolve" is a weaker claim than "the example runs". T9 adds a
controlled workflow smoke test (`WorkflowSmoke`) that executes a small
allowlisted set of documented commands against checked-in fixtures with real
option values; this document makes no such claim yet.

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

## Interim §29 evidence at the audited head

Each row names the registered tests that carry the evidence, the level that
evidence reaches at the audited head, and the boundary that remains. These rows
are **not** an acceptance decision. They record capabilities that genuinely
work; several criteria stop short of the mandatory contract, and the
[interim gap table](#interim-gap-status-issue-129) above names what remains open
under [issue #129](https://github.com/skg7on/DSLCompiler/issues/129). T9
republishes this matrix at the repaired revision.

| §29 | evidence (registered tests / binaries) | interim result | limitation |
|---|---|---|---|
| **1.** search space and complete bindings round-trip deterministically | `MicroDialectSearchSpace`, `MicroDialectSearchSpaceInvalid`, `LLKToMicroSearchSpace`, `LLKToMicroSearchSpaceMBucket`, `LLKCompileMicroSearch`, `SearchSpaceLoaderTest`, `MappingSearchBindingTest`, `SearchBindingLoaderTest`, `MicroMappingSearchBinding`, `LLKTuneSearchSpace` | met | Non-MMA and multi-role workloads are evaluated from the facts each constraint names (`Legality.ShapeIndependentConstraintNeedsNoWorkloadFacts`); production measurement is #51/#52. |
| **2.** normalized MachineModel topology | `MachineModelTest`, `MachineModelV2LoaderTest` | met | — |
| **3.** deterministic direct and multi-hop routing | `MappingRoutingTest`, `RouteIdentityTest`, `MappingConnectionsTest` | met | Exact mode branches over the *enumerated* joint connection choices (`CoveringSearch.GatherExactBranchesOverFeedAlternatives`); an exhaustive branch-and-bound oracle is a property test, not a solver guarantee. |
| **4.** target-owned layout legality without generic target branches | `MappingLayoutTest`, `MappingRulesTest`, `MappingGenericAcceleratorTargetTest` (purity + conformance) | met | The generic-Micro purity test asserts the leak set is now **empty** — the closed owner enumeration that carried backend words is gone (Stage C9). |
| **5.** target rules generate `MappingCandidate` objects | `MappingRulesTest`, `MappingCoveringSearchTest`, `GraphRule.*` | interim | Bounded fused-graph matching landed in Stage C7; arbitrary subgraph rules beyond the declared pattern stay out of scope. Selected fused *execution* is not yet an end-to-end contract (G5). |
| **6.** concrete instances, connections and complete coverings | `MappingPlacementTest`, `MappingConnectionsTest`, `MappingPlanTest`, `MappingPlanBinderTest`, `MappingStoragePlanTest` | interim | Per-use layouts survive repeated operand uses (`CoveringSearch.DistinctOperandUsesOfOneValueKeepTheirOwnLayouts`); endpoint ports are carried into the plan (`Connections.CarriesEndpointPorts*`). Complete selected compute identity and physical feasibility remain open (G1, G2). |
| **7.** deterministic / beam / exact share one interface, honest limits | `MappingCoveringSearchTest`, `MappingPropertiesTest`, `MappingRoutingTest` | interim | Caps and unexplored-choice restrictions are reported (`FanOutReportsRouteCapThroughTruncated`, `GatherExactBranchesOverFeedAlternatives`). Capacity validation after top-K still discards feasible exact results (G2). |
| **8.** selected plan materializes placements, routes, transforms, bundles | `MicroMappingMap`, `MicroMappingVerify`, `MicroMappingUnmaterialized`, `MicroMappingRequireExecutable`, `MicroMappingBarrier`, `MicroMappingBarrierStorage`, `MicroMappingBindPlan*`, `MicroMappingMatmulE2E`, `MicroMappingSwiGLUE2E`, `PlanBinder.MaterializesATileMovementAsATypedTileAsyncCopy`, `PlanBinder.MaterializesASameKindMovementBetweenDistinctNodes` | interim | Target readiness is a *separate* gate from materialization: a complete plan containing an unsupported operation fails target readiness rather than claiming compilation. Complete physical storage and selected-backend materialization remain open (G6, G4). |
| **9.** mapping search and `micro-perf` share MachineModel / route / cost primitives | `MappingCostModelTest`, `MappingPlanReportTest`, `MappingStoragePlanTest`, `MicroPerfCli`, `LLKToMicroPerfRoundTrip`, `MicroMappingReport` | interim (static model) | Materialized transforms now produce `EventKind::Transform` events with the shared `estimateTransformCost` estimate (`CostEvent.LayoutTransformEventUsesTheSharedEstimate`). Complete static selected-plan/perf parity and correct resource occupancy remain open (G8, G7). *Calibrated* parity is #51/#52. |
| **10.** AVX2 works while the legacy path remains available | `MappedAcceptance`, `MicroKernelExecution`, `MicroMappingCompileMatmulToExecutable`, `MicroMappingCompileSwiGLUToExecutable`, `LLKToLinalgConversion`, `MicroToLinalgJit`, plus the legacy SwiGLU chains | interim at the **host-portable** level; selected AVX2 execution is unproved | The mapped numeric chains are Linalg/SCF/memref compiled for the running machine, so their numbers are identical on arm64 and x86 — and say nothing about which instruction set ran. AVX2 instruction-level execution is claimed only by the legacy `SwiGLUVector` test, which skips both on this arm64 host and on the Linux x86 CI runner (G4). |
| **11.** second target without new generic Micro ODS policy | `MappingGenericAcceleratorTargetTest` (`OwnerVocabularyIsTargetDataNotAnOdsEnum`, `TargetVocabularyStaysOutOfGenericMicroOds`, `DoesNotClaimAnExecutableBackend`), `MicroMappingAcceptancePipeline` (`second-target` chain) | interim | The accelerator maps/materials/perf-models without claiming execution on hardware it does not have; that is deliberate, not a gap. Complete canonical tile/two-hop acceptance remains open (G10, G6). |
| **12.** identical inputs produce byte-identical IR and reports | `MappingStableHashTest`, `MicroMappingBindPlanRoundtrip`, `MicroMappingReport`, `MappingPropertiesTest`, `MicroMappingAcceptancePipeline` (repeat-bytes check) | interim | The chain runner compares normalized report and IR bytes across two independent searches. The missing acceptance chains and the full selected-state contract remain open (G1, G10). |

### Numeric-execution evidence in detail

These are the tests that *invoke* compiled programs and check the answers, not
merely that compilation or symbol lookup succeeded:

| binary | tests | what it proves |
|---|---|---|
| `MappedAcceptance` | 3 | the compiler-generated matmul and fused SwiGLU are mapped onto the AVX2 target, compiled and **invoked** as host-portable reference code, every output element checked (a full K=64 reduction; `silu(64)×64`); a third test writes the output inside a larger allocation with sentinels to show the kernel touches only its region. This is not evidence that AVX2 instructions ran (G4) |
| `MicroKernelExecution` | 8 | the descriptor-pointer ABI: repeat invocation, a buffer starting inside a larger allocation, arity and stride rejection, missing symbol, and executable lifetime after the module is gone. The descriptor carries no runtime dtype tag, so invocation-time *dtype* rejection is not established by these eight tests (G3) |

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
is closed at the audited head. Stages A (#114/#115) and B (#116) carry the
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

While running the workflow guide's commands at the audited head, `llk-tune`
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

Tracking: issue #67 remains the open tracker and is **not** closed; its
mandatory rows stay open until the #129 repairs satisfy the release gates in the
[October 7 plan](../superpowers/plans/2026-10-07-issue129-gap-closure.md). This
document is interim evidence, not the closure decision. Where a criterion is
listed as "interim" above, that is the level the named tests reach at the
audited head, and the gap ids in its row name what remains open — not a claim
about the epic's issue text, which is historical. Closure of #67, #106 or #109
is a separate, authorized decision.
