# Issue #67 current gap assessment — 2026-10-07

The implementation has advanced substantially, but the A/B/C plan's release gates are not all satisfied. The current result is a functioning mapping framework plus host-portable numerical execution. It is not yet complete selected-target execution, physical resource verification, or a mapping-driven tuner. PR #128 should not establish final acceptance in its present form.

## Reviewed state and verification

- Merged main: `50b8fbeed84fead59956b50c3fb338267e0cbd89`, through PR #127.
- Open PR #128: `e1b6efc01e11cc5928d96603bf81c75e02e018dd`, including tuner fix `c4f088d`.
- Read `CLAUDE.md`, issues [#67](https://github.com/skg7on/DSLCompiler/issues/67), [#106](https://github.com/skg7on/DSLCompiler/issues/106), [#109](https://github.com/skg7on/DSLCompiler/issues/109), historical delivery descriptions, the normative September 18 design, the October 3 gap plan, and the October 4 A/B/C improvement plans. Inspected current implementation and relevant tests; historical PR descriptions establish scope, while the current source and probes establish status.
- Fresh isolated configure/build of the exact PR #128 head: Homebrew Clang 20.1.8, LLVM/MLIR `24.0.0git`, Darwin arm64. Full CTest: **139 registered, 137 passed, two skipped, zero failures**. Skips: `SwigluScalar`, `SwiGLUVector`.
- PR #128 Linux static LLVM 22.1.8 [build-and-test](https://github.com/skg7on/DSLCompiler/actions/runs/37565145333/job/112611023308) and coverage passed. The build log also reports both legacy SwiGLU tests **skipped**. CodeQL and analyze also passed on the issue-creation recheck.
- Independently reran the planner/resource probe, width-erasure comparison, selected fused-rule lowering, public planner/perf comparisons, and the new blocked-layout regression. No production source was modified.

## Delivery history and what it establishes

| Delivery | Result carried into current main |
|---|---|
| #81, #83–#88 | Mapping model, MachineModel v2, routing, declarative layouts/rules, placement/connections, covering search. #82 is an issue, not a PR. |
| #89–#97 | Two target packages, binding, tensor movement, shared machine/perf vocabulary, latency-provider hooks and workflow. #93 consolidated the stacked perf/workflow content onto main. |
| #98, #101–#105 | Public passes, objectives/reports/diagnostics, stronger layout/resource legality, persistent binding integration; LLKMap fixes. #100 is the MicroDAG refactor design, not its implementation. |
| #107/#108; #110/#111/#112 | Review evidence and successive correctness fixes. The original four #106 defects are fixed. #111/#112 delivered checked metadata, semantic verification and partial durable integration. |
| #113–#117 | A/B improvement plan and implementations: endpoint identities, legality controls, canonical materialization, storage/sync structures and joint choices. Several complete-resource/search contracts remain incomplete below. |
| #118 | Coverage workflow maintenance. |
| #119–#125 | Explicit kernel contract, emitter hook, structural lowering, descriptor invocation, shared compilation, candidate measurement hook and bounded fused matching. The end-to-end selected-target/tuner contracts remain incomplete below. |
| #126/#127 | Real portable numerical GEMM/SwiGLU invocation and target-owned owner aliases. C8's full acceptance chains were not delivered. |
| #128, open | Useful tuner crash fix and proposed acceptance document. Its completion and CI claims need correction. |

The issue bodies still describe October 4 baselines. Their historical unchecked boxes should be refreshed against current evidence. They should not be blindly retained or blindly marked complete because subsequent PRs merged.

## Current findings

### 1. P1 — concrete compute selections disappear before binding and scoring

[`Placement.cpp:424`](../../lib/Mapping/Placement.cpp#L424) records each selected compute node in `CandidateInstance::computeBindings`. [`PlanPlacement`](../../include/LLK/Mapping/MappingPlan.h#L408) has no corresponding field, and [`CoveringSearch.cpp:2056`](../../lib/Mapping/CoveringSearch.cpp#L2056) drops it when constructing the complete plan. [`CostEvent.cpp:248`](../../lib/Mapping/CostEvent.cpp#L248) selects an engine from the executor instead.

**Reproduced:** two attached vector engines, `vpu.a` and `vpu.b`, produce distinct selected instances, but both complete plans normalize their event resource as `vpu.a`. The concrete selected implementation cannot survive report/bind/replay or drive resource occupancy correctly. B1/B8 require this identity to persist.

### 2. P1 — exact search can discard the available capacity-legal covering

[`CoveringSearch.cpp:2165`](../../lib/Mapping/CoveringSearch.cpp#L2165) trims to top-K before [`MicroMappingCommon.h:576`](../../lib/Conversion/MicroMapping/MicroMappingCommon.h#L576) validates finalized physical storage. The adapter only filters retained plans; it never recovers a more expensive legal plan.

**Reproduced:** a single `8x8xf32` output with known execution multiplicity four; SRAM capacity 512 bytes, DRAM capacity 4096 bytes; SRAM rule cost one, DRAM rule cost two. Exact `topK=1` retains SRAM and then fails its 1024-byte storage footprint. Exact `topK=2` also exposes the legal DRAM covering. Physical feasibility must participate before final acceptance/top-K; merely increasing top-K is not a correctness guarantee. B3/B7 remain incomplete.

### 3. P1 — repeated mapped invocation leaks compiler-allocated buffers

[`MappedCompilation.cpp:162`](../../lib/Conversion/MicroMapping/MappedCompilation.cpp#L162) runs bufferization and loop conversion without ownership-based deallocation. The LLVM pipeline adds no deallocation stage. [`MappedExecutable.cpp:111`](../../runtime/MappedExecutable.cpp#L111) copies the allocated result into caller-owned output but does not release the internal result.

**Reproduced in lowered IR:** mapped `matmul_e2e.mlir` contains seven `memref.alloc` sites and zero `memref.dealloc` sites. Accounting for their loop nesting gives approximately 68 KiB of allocated buffers per invocation, excluding allocator overhead. This is source/IR evidence of unreleased allocations, not an RSS measurement. Repeated-invocation numerical tests do not test deterministic scratch release, which C4 requires.

The descriptor boundary also needs explicit unsupported-rank rejection: [`checkDescriptors:140`](../../runtime/MappedExecutable.cpp#L140) skips shape/stride checks for non-rank-2 ports even though invocation supplies fixed `MemRef2D` descriptors. This was source-reviewed, not invoked with an incompatible descriptor. The current descriptor contains no runtime dtype tag; the acceptance document's claimed invocation-time dtype rejection is not established by the eight runtime tests.

### 4. P1 — selected AVX2 width does not reach executable code

The AVX2 emitter rewrites a result tile's layout to `vectorized` with the chosen width ([`AVX2BundleLowering.cpp:224`](../../lib/Target/X86/Mapping/AVX2BundleLowering.cpp#L224)). [`MicroTypeConverter:95`](../../lib/Conversion/MicroToLinalg/MicroToLinalg.cpp#L95) immediately converts that tile to a plain tensor using shape/dtype only. The compilation path then converts scalar Linalg directly to loops, without explicit vectorization. The JIT targets the native host.

**Reproduced:** width-four and width-eight add kernels produce byte-identical `--micro-to-linalg` output. Mapped matmul's lowered form contains scalar loops/multiply/add and zero `vector.*` operations. This proves loss of selected width at the IR boundary; no assembly inspection was used to claim that LLVM can never auto-vectorize. Host-portable numerical execution works, but C2/C8 require selected implementations to determine vector width/ISA and mapped AVX2 execution evidence.

### 5. P1 — a selected shipped fused rule cannot pass target lowering

Exact search selects `avx2.fused_convert_silu_mul` for a `convert → silu → mul` fixture and successfully binds it. The rule declares `VW` but supplies no layout requirement ([`rules.llkmap:158`](../../mapping/x86-avx2/rules.llkmap#L158)); selected metadata supplies neither a bundle width nor a solved layout width. Target lowering fails:

```text
avx2_lowering: no vector width was selected: bundle
'avx2.fused.convert_silu_mul' carries none and the operation records
no solved layout that has one
```

There is a second integration defect: [`lowerSelectedOperations:111`](../../lib/Conversion/MicroMapping/MappedCompilation.cpp#L111) supplies empty placement/connection arrays and dispatches one operation at a time, rather than the complete selected fused group. The ordinary SwiGLU numerical fixture uses `silu → mul → convert` and selects standalone rules; it does not cover this failure. C7 matching is delivered, but selected fused execution is not.

### 6. P2 — complete physical storage is bypassed or omits intermediates

[`MicroMappingCommon.h:552`](../../lib/Conversion/MicroMapping/MicroMappingCommon.h#L552) skips storage finalization if any placement has no memory binding. All four public acceptance reports carry the note that storage planning was skipped; strict executable mapping can still succeed.

For storage-plannable two-hop routes, [`StoragePlan.cpp:789`](../../lib/Mapping/StoragePlan.cpp#L789) reserves only `route.back()`. The strict SRAM→L2→DRAM fixture emits both copies, but its selected report reserves SRAM/DRAM and no L2 allocation. Routing checks an individual intermediate capacity; it does not establish the missing allocation lifetime or shared occupancy. B3/B4's complete physical storage contract is therefore not satisfied.

### 7. P2 — the shared scheduler overbooks named DMA engines

[`EventSchedule.cpp:49`](../../lib/Mapping/EventSchedule.cpp#L49) gives each separate engine-ID pool the **whole machine's** transfer-engine count.

**Reproduced:** two independent ten-cycle transfers on `dma.a(count=1)` take 20 cycles with that engine alone. Adding an unused `dma.b(count=1)` changes their makespan to ten, both starting at zero. Each pool needs its own node's concurrency/count. Sharing this scheduler between planner and perf does not make its occupancy correct.

### 8. P2 — static planner/perf agreement is still incomplete

Current public acceptance output, with the shipped profiles:

| Fixture | Planner cycles | Bound-kernel `micro-perf` cycles | Planner DRAM bytes | Perf DRAM bytes |
|---|---:|---:|---:|---:|
| vector-add | 4 | 232 | 0 | 256 |
| staged-gemm | 16 | 12144 | 0 | 20480 |
| fused-swiglu | 16 | 14416 | 0 | 36864 |
| second-target | 3 | 318 | 0 | 256 |

[`CostEvent.cpp:258`](../../lib/Mapping/CostEvent.cpp#L258) intentionally uses rule-local compute estimates while perf uses operation formulas. `L1ResourceDag.PlanComputeCyclesUseTheRuleEstimate` explicitly pins five versus eight cycles for the same work. Structural execution multiplicity and workload movement are also not represented equivalently in the public scores.

Perf reports capacity violations on staged GEMM/SwiGLU while planner storage checking was skipped. These reports do **not** establish the true required capacity: perf may itself overestimate live storage. They establish disagreement that must be reconciled. B8/C8 require common selected work/resources/dependencies and static scoring semantics. This is separate from calibrated prediction under #51/#52.

### 9. P2 — C6 still lacks mapping-driven tuning

[`TuningSession.cpp:184`](../../lib/Perf/TuningSession.cpp#L184) constructs a synthetic concrete kernel, analyzes it and ranks it. It does not instantiate a mapping problem, search a target plan, bind the selected covering, or compile that selected plan. [`llk-tune.cpp:371`](../../tools/llk-tune/llk-tune.cpp#L371) supplies no target/rule/layout adapter or measurement provider; output is schedule YAML without the selected mapping report.

The measurement test calls `compileConcreteMicroKernel`, not `compileMappedKernel`. PR #124 explicitly acknowledges the plan's “instantiate/map each candidate through C5” requirement is not implemented. The callback is useful progress, but its input is a module and the caller must supply compilation/invocation. It is not the planned verified executable plus invocation-input contract. Production measurement persistence remains deferred; mapping-driven tuning does not become out of scope because persistence is deferred.

### 10. P2 — C8/C10 evidence is incomplete and overstates CI

- The public runner has four chains; it lacks the required-transform chain. Its second-target chain reuses tensor copy/add rather than canonical tile/two-hop acceptance.
- `llk_compile` is assigned but never invoked. The runner does not round-trip a search space or bind a frozen report. It checks hash/metadata key substrings and nonempty perf stdout, not selected rules/routes, required movement, capacity validity or event/resource parity.
- Numerical acceptance genuinely invokes mapped GEMM/SwiGLU, but only fixed `16×64×64` all-ones inputs plus an output sentinel. Deterministic random inputs, M/N/K tails, multiple outputs and numerical required-transform controls from C8 are absent.
- PR #128's acceptance document says the legacy SwiGLU tests run on x86. Both its cited #127 job and the current #128 job report them skipped. Even a successful legacy `SwiGLUVector` test would not establish selected mapped-target execution.
- The document pins base `50b8fbe` while quoting 139 tests including a C10-added test; base #127 has 138. Pin the exact tested C10 tree/head and its own CI separately.
- `DocReferences` is useful path/help-name lint. It does not execute documented commands, validate pass option values, or validate file arguments inside shell code fences. Its current “examples can actually run” claim is stronger than the check.

## Revised acceptance assessment

| Design §29 | Assessment |
|---|---|
| 1–4 | Foundation delivered: persistent syntax/bindings, normalized topology, deterministic routing, target-owned declarative layouts. This does not establish full executable axis realization. |
| 5 | One-op and bounded fused matching delivered; fused lowering integration remains open. |
| 6 | Plans/instances/connections exist; complete selected compute identity and physical feasibility remain open. |
| 7 | Common modes and cap disclosure exist; capacity validation after top-K still loses feasible exact results. |
| 8 | Canonical movement/transform/gather/sync infrastructure delivered; complete physical storage and selected backend materialization remain open. |
| 9 | Shared primitives/scheduler delivered; complete static selected-plan/perf parity and correct resource occupancy remain open. |
| 10 | Portable mapped numerical execution and legacy availability delivered; selected AVX2 code generation/execution remains unproved. |
| 11 | Target-neutral owner vocabulary and second-target mapping delivered; complete canonical tile/two-hop acceptance remains open under the implementation plan. Accelerator hardware execution is not required. |
| 12 | Byte-identical repeated searches/IR/reports demonstrated for supported fixtures; the missing acceptance chains and full selected-state contract need completion. |

## What should happen next

1. Correct PR #128's matrix, CI statements, exact revision pins and `CLAUDE.md` completion claim. Keep the useful tuner crash fix. Do not close #67/#109 based on this matrix.
2. Repair concrete compute persistence, physical feasibility before top-K, intermediate storage and per-engine scheduling. Require complete resource facts for executable claims; retain explicit partial analysis modes.
3. Release runtime scratch deterministically and reject unsupported ABI ranks before publishing an executable.
4. Carry selected width/layout/resource facts into real backend lowering; dispatch fused instances as groups with their resolved parameters and plan context.
5. Connect `llk-tune` to the same mapping/compilation/objective/report path.
6. Finish all five C8 chains with replay, substantive static parity/capacity checks, random/tail/multiple-output numerical controls, and non-skipped mapped AVX2 execution on x86.
7. Refresh #106/#109/#67 so resolved historical defects remain credited and remaining items point to this evidence.

Production measured-record storage, calibration fitting and prediction validation legitimately remain #51/#52. Generic-accelerator hardware execution, functional emulation and unbounded graph/constraint solving are not blockers for this epic. The gaps above concern mandatory selected-plan/resource/execution/acceptance contracts that the existing plan already requires.

## Probe locations and commands

Probe files live in the isolated review worktree's `build/audit-probes/`; they do not change production code.

```sh
# From the review worktree:
build/audit-probes/mapping_ab

build/llk-opt --micro-to-linalg build/audit-probes/width-4.mlir
build/llk-opt --micro-to-linalg build/audit-probes/width-8.mlir

build/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
  --mapping-mode=exact --mapping-stop=target-lowered \
  build/audit-probes/fused-exact.mlir

build/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
  --mapping-stop=lowered test/Conversion/MicroMapping/matmul_e2e.mlir

build/micro-perf --machine machines/x86-avx2-v2.yaml --level 1 \
  build/AcceptancePipeline/vector-add/mapped_a.mlir

build/L1ResourceDAGTest \
  --gtest_filter=L1ResourceDag.PlanComputeCyclesUseTheRuleEstimate
```
