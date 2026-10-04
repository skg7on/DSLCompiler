# Issue #67 reassessment after PR #108

Reviewed on 2026-10-04 against main commit `749bf2106a597ee3e70fee4d73c1eb2f9696713f`.

- Follow-up tracker: [issue #109](https://github.com/skg7on/DSLCompiler/issues/109)
- Epic: [issue #67](https://github.com/skg7on/DSLCompiler/issues/67)
- Fix: [PR #108](https://github.com/skg7on/DSLCompiler/pull/108), merged at 09:28:06 UTC.
- Normative baseline: [enhancement design](../superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md), including §29 acceptance criteria.
- Historical review: [PR #98 reassessment](2026-10-04-issue67-pr98-status.md). Its revision/status statements and four defect findings describe the earlier revision; this report supersedes them for current main.

## Conclusion

PR #108 fixes all four originally reported connection/resource defects for their original reproduction cases. Those findings should be marked resolved. Issue #67 is nevertheless **not complete**. The implementation provides a substantial, tested mapping/search foundation, but selected plans still cannot be fully executed through the promised binding, target-lowering, and performance pipeline. Additional resource/layout/search correctness holes and public-verifier defects remain.

CI success establishes the currently tested subset, not all design acceptance criteria. No percentage-complete estimate is justified by counting merged PRs or checked boxes.

## Verification

Fresh isolated configure/build against local LLVM/MLIR 24 succeeded. `ctest --test-dir build --output-on-failure -j 6` registered 121 tests: **119 passed, two CPU-specific skips, zero failures**. These include PR #108 regressions and mapping, bind/replay, matmul/SwiGLU, report-only, verification, and second-target tests.

Additional current-main runtime probes:

| Probe | Observed result |
|---|---|
| Graph API: 4-byte and 4096-byte outputs, two 1024-byte memories, exact mode | Accepts one plan. |
| Placement API: two layout requirements of class `t.plain`, ports 11 and 22 | Retains one solution, associated with port 22. |
| Existing tile-route unmaterialized fixture | Exit 0; warning that a connection was not materialized; emits `micro.plan`. |
| Unmapped kernel passed to `micro-verify-mapping` | Exit 0 with no `micro.plan`. |
| Valid vector-add mapping changed to bundle `garbage`, emitter `avx2_mma` | Exit 0. |
| Valid vector-add mapping changed to existing rule `avx2.mma_bf16` | Exit 0. |
| Remove mapping `layouts` and `memories` | Exit 0. |
| Change route engines to `nonexistent` | Exit 0. |
| Set mapping `memories = {operand0 = 42 : i64}` | Aborts, subprocess return code -6, assertion in attribute cast. |
| Set `micro.routes = [42 : i64]` | Aborts, subprocess return code -6, assertion in attribute cast. |
| Map then verify two kernels in one module | Exit 0; two kernels but only one `micro.plan`. |

Inputs/stdout/stderr are retained in the review worktree's `build/review/`. These are additional local probes, not registered regression tests. Source-only findings below are explicitly distinguished from executed probes.

## PR #108: four previous defects

| Previous finding | Current status and evidence |
|---|---|
| Transfer input capacity omitted | Fixed: `CoveringSearch.cpp:737` charges destination input bytes and records their lifetime. Tests cover over-capacity rejection, legal capacity, and sequential reuse. |
| Layout-only fan-out discarded | Fixed: `Placement.cpp` admits `LayoutTransform` when all group consumers can see producer memory. Dedicated fan-out regression passes. |
| Different layout classes shared one replica | Fixed for differing class IDs: fan-out groups by destination memory **and** layout class. Dedicated split-layout regression passes. Solved parameter/map compatibility remains separate unfinished work. |
| Gather checked only representative consumer | Fixed for original case: `GatherKey` includes memory, layout, executor, port type, and port affine map, forming separately validated groups. Dedicated incompatible-consumer regression passes. |

The outdated transfer-accounting comment near `CoveringSearch.cpp:684` should also be corrected; current code does charge the destination buffer. This is documentation debt, not recurrence of the old defect.

## Remaining actionable correctness findings

### 1. Multi-output, multi-memory capacity is an admitted lower bound

**Compiled graph-API probe confirmed; P1.** `lib/Mapping/CoveringSearch.cpp:590-595` charges only the first output to every memory binding when there is more than one binding. The source comment explicitly calls this an admitted lower bound because output-to-memory association is missing. A small first output cannot establish capacity legality for a large second output. For example, a 4-byte first output and 4096-byte second output should not be admitted into two 1024-byte memories merely by charging four bytes to each.

Add explicit output/port-to-memory associations, account every live value in its actual memory, and reject ambiguous placement rather than using an unsafe lower bound. The probe constructs a synthetic multi-output WorkloadGraph directly; it does not demonstrate a shipped single-result vector rule producing multiple MLIR results. Test asymmetric output sizes with multiple memory requirements. This is outside the transfer-input case fixed by PR #108.

### 2. Per-port solved layouts overwrite each other

**Compiled placement-API overwrite confirmed; P1.** `lib/Mapping/Placement.cpp:383-398` stores layout bindings and solutions by layout class. Two requirements for different ports with the same class overwrite the earlier solution, including its `portValue`. `CoveringSearch.cpp:166-178` can consequently find no layout for the earlier value and synthesize a direct connection without its actual requirement.

The probe demonstrates lost port association; the downstream incorrectly direct connection is source reasoning, not a separate runtime reproduction. Use a port/role-aware key. Connection compatibility must compare concrete solved layout maps/parameters as well as class IDs. PR #108's class-based grouping does not solve same-class, different-parameterization compatibility.

### 3. Exact mode does not explore complete connection choices

**Source-confirmed; P2.** `CoveringSearch.cpp:718` selects a connection locally, and fan-out likewise uses a cheapest alternative per destination group in `Placement.cpp`. Aggregate occupancy is checked later (`CoveringSearch.cpp:820-828`). Two individually legal cheapest routes through a shared intermediate can jointly exceed capacity, while more expensive direct alternatives would fit. The search does not branch over those alternatives, so it can reject a feasible covering even in exact mode without a search cap explaining the omission.

Expand the search state over globally relevant connection choices, or expose this restriction explicitly instead of implying exhaustive joint placement/route search. Distinguish this from the already disclosed exact-mode equal-cost tie pruning/truncation limitation.

### 4. Public mapped verification is not a completeness or semantic check

**Runtime-confirmed; P2.** `lib/Mapping/PlanBinder.cpp:431-437` skips operations without mapping metadata; `461` and `480` treat required memory/layout dictionaries as optional. Rule IDs and emitter IDs are independently checked for membership (`446-452`, `492-498`), without matching the selected rule against the operation, bundle, or emitter. Route verification (`508-540`) does not validate engines.

The probes above show successful verification of unmapped IR, a vector operation labeled with an MMA rule, an arbitrary bundle paired with a known but incompatible emitter, and nonexistent route engines. Define mapped completeness at kernel scope, validate required roles and rule constraints, and invoke target bundle verification.

### 5. Malformed generic metadata crashes verification

**Runtime-confirmed; P2.** No dialect structural validator rejects malformed mapping/route entries before `PlanBinder.cpp:464,483,512,517` casts them. Integer memory IDs and integer route entries abort the process. This contradicts design §18.3 layered verification and §25.1 invalid-metadata rejection.

Validate container shape and entry types, using checked casts and ordinary stable diagnostics. Cover malformed plan, mapping, route, layout, and engine metadata.

### 6. Multi-kernel and multi-search-space selection is incomplete

**Two-kernel behavior runtime-confirmed; objective behavior source-confirmed.** `MicroMappingCommon.h:169-177` and `PlanBinder.cpp:106-123` select the first kernel. A two-kernel module maps only one and then verifies successfully. Independently, `MicroMappingCommon.h:185-195` selects the first objective anywhere in the module, rather than the objective belonging to the selected candidate's search space.

Define a kernel/search-space association or explicit selectors. Map all requested kernels or reject ambiguous/incomplete input; derive objective order from the selected space.

## Full design acceptance matrix

“Delivered” below means implemented with evidence for the stated subset, not proof of all future-target behavior.

| Design §29 criterion | Assessment | Remaining boundary |
|---|---|---|
| 1. Persistent search and complete binding round-trip | Delivered representation; partial mapping effect | Domain checks, parameter pinning and provenance work. Global constraints and all bound axes do not fully constrain mapping. |
| 2. Normalized MachineModel | Delivered | Versioned topology and validation exist. |
| 3. Deterministic direct/multi-hop routing | Delivered foundation | Joint route search, allocation lifetime and executable tile-hop materialization remain incomplete. |
| 4. Declarative target layout legality | Delivered solver; partial integration | Finite constraints/quantifiers and bounded solver work; canonical layout-kind to target implementation bridge and port-specific concrete identity remain incomplete. |
| 5. Target rules generate candidates | Delivered for single-op rules | Fused compatible subgraphs remain later work; initial one-op rules are permitted by §14.2. |
| 6. Concrete instances, connections, complete coverings | Partial | Objects and search exist; residual legality/resource/representation findings above remain. |
| 7. Shared deterministic/beam/exact interface, honest truncation | Substantially delivered | Exact local connection selection is not joint exhaustiveness; equal-cost tie pruning is disclosed as truncation. |
| 8. Selected plan materializes concrete execution | **Open** | Tile transfers, transforms, replication/gather/reduction, consumer-specific rewiring and complete target bundle state remain unfinished. |
| 9. Shared mapping/performance machine and cost primitives | Shared foundation delivered; parity partial | Selected executor/layout/bundle decisions do not fully affect perf; cache identity is incomplete. |
| 10. AVX2 target plus legacy path | Mapping fixtures and legacy compatibility delivered | Selected-bundle target lowering/execution is not demonstrated. Mapping compiler-generated matmul/SwiGLU now succeeds. |
| 11. Second target without new generic ODS policy | Fixture delivered | Existing generic warp/wave vocabulary remains architecture debt; do not confuse it with newly added target policy. |
| 12. Byte-identical normalized IR and reports | Delivered for tested subset | Replay determinism tests pass; full executable multi-kernel/transform/perf chains are absent. |

## Remaining integration work

### Persistent binding → mapping problem (§8.3)

`SearchBindingLoader.cpp:128-163` validates domains/completeness, not global `micro.constraint` evaluators. Those evaluators live in `Perf/Legality.cpp`. `MappingRules.cpp:1033-1051` ignores bound names absent from a rule and returns early for rules without constraints. Owner, memory-path, pipeline, tail and other candidate axes can therefore remain provenance rather than enforced execution choices.

`SearchBindingLoader.h:88-100` explicitly documents the layout bridge as veto-only: canonical `blocked` does not equal shipped target ID `avx2.blocked_2d`. The candidate integration test intentionally expects that shipped-target failure. Multiple layout parameters still fail rather than bind per role. Complete the role-aware canonical-kind/target-layout bridge and validate global legality before search.

### Plan binding and persistent selected state (§18)

`PlanBinder.cpp:157-179` serializes names/IDs but discards `TargetBundle.parameters` and concrete solved layout parameters/maps. `248-253` leaves layout transforms, replicas and reductions unmaterialized; `295-300` cannot construct a destination type for custom `!micro.tile`. Transfer-and-transform emits movement while omitting its transform (`400-403`). Existing per-hop copies/waits are useful but do not finish dependency-required barrier/allocation semantics.

Routes are deduplicated by value (`265-274`); all other source uses are rewired to the last copy (`405-412`), without consumer association in the persisted connection representation. Implement consumer-specific dataflow rather than allowing one movement to redirect unrelated readers. Multiple layout-only transforms also need a source-preservation and output-buffer model; class-split metadata alone establishes neither.

`MicroMappingCommon.h:402-406` reports unmaterialized decisions as warnings and installs the incomplete clone. Design §18.2 allows reporting unsupported cases, so the warning is better than silent loss; it does **not** satisfy §29.8 concrete execution. Separate analysis/partial binding from a mode that guarantees executable output, or fail complete binding when execution-affecting decisions remain unresolved.

### Target emission (§19) and tools (§21)

`MappingTarget.h:35-56` provides emitter identity/verification but no real lowering hook. Generic declared emitters and verifier membership checks do not interpret the selected bundle into backend execution. Preserve the opaque resolved bundle and add target-owned verification/lowering.

`llk-tune.cpp:380-388` still performs the perf/legacy schedule workflow rather than complete target mapping/ranking/measurement. `llk-compile` has no mapped-target entry point. Keep legacy execution available, but implement the requested mapping path and prove selected plans reach executable target code.

Reports are explicitly non-executable reproducibility metadata (`PlanReport.h:5-15`), and ID replay reruns search with the original options/files. Frozen report import is absent; this is a declared boundary, not a new syntax bug.

### Performance/cache fidelity (§17)

`MicroDAG.cpp:334-374` consumes concrete legacy hop stamps. It does not consume `micro.mapping` to model all selected executors/layouts/bundles; compute still derives from intrinsic engine/type information. `LatencyProvider.h:44-53` and `CoveringSearch.cpp:438-452` omit operand/result types, shape, relevant attributes, resolved bundle parameters and all concrete layouts from measurement identity. Different workloads can therefore share a key.

Use common events for selected compute, movement, transforms and synchronization, preserve full measurement identity, and test mapped-plan/perf agreement. Production calibration/prediction validation remains #51/#52 work, not a prerequisite for merely having an optional provider interface.

## Epic checklist corrections and completion order

Issue #67 remains open and its live text is stale: PR #98 is now merged; raw hex IDs, replay modes, report-only output, public verification, secondary objectives and compiler-generated matmul/SwiGLU mapping are implemented. Retire those historical “missing” statements. Mark the four PR #108 findings resolved while keeping the genuinely incomplete composite goals open.

Recommended order:

1. Fix capacity/port-layout identity and strengthen structural/semantic mapped verification; add focused rejection regressions.
2. Make selected state durable: per-port layouts, concrete bundle parameters, consumer-specific connections and resource associations.
3. Complete tile movement, transforms, fan-out/fan-in and synchronization/allocation materialization; guarantee completeness for executable output.
4. Add target-owned lowering, integrate compile/tune, and prove actual AVX2 execution.
5. Complete persistent-binding/global-constraint semantics and joint search over connection alternatives.
6. Finish selected-plan perf/cache fidelity and the required fused/transform/two-hop/perf-parity acceptance chains.

Issue #67 should not be closed until its execution contract and all §29 acceptance criteria are met. PR #108 is a successful focused fix, not evidence that the whole enhancement epic is complete.

## Reproduction appendix

To reproduce the public verifier results, first map `test/Conversion/MicroMapping/verify_mapping.mlir` using:

```sh
build/llk-opt test/Conversion/MicroMapping/verify_mapping.mlir \
  '--micro-map=target=x86-avx2 machine=machines/x86-avx2-v2.yaml layouts=mapping/x86-avx2/layouts.llkmap rules=mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store mode=deterministic'
```

Save the output and apply each mutation independently. Run `build/llk-opt` on it with `--micro-verify-mapping` and the same target/machine/layout/rule/emitter options, without the search mode. Exact mutations:

- Replace `bundle = "avx2.vector.add.f32"` with `bundle = "garbage"` and `emitter = "avx2_vector_add"` with `emitter = "avx2_mma"`.
- Replace `rule = "avx2.vector_add"` with `rule = "avx2.mma_bf16"`.
- Remove mapping `layouts = {...},` and `memories = {...},` fields.
- Replace `engines = []` with `engines = ["nonexistent"]`.
- Replace one `memories = {}` with `memories = {operand0 = 42 : i64}`.
- Replace the kernel's complete `micro.routes` array with `[42 : i64]`.

For the completeness probe, verify the original unmapped fixture directly. For the two-kernel probe, duplicate the original kernel under a different symbol, then run mapping and verification in sequence.

The two compiled planner probes reuse helper definitions from `test/Mapping/covering_search.cpp`; their bodies are preserved below. Their passing assertions intentionally pin the observed defective behavior, not desired semantics.

```cpp
TEST(Post108Probe, MultipleMemoryOutputs) {
 mlir::MLIRContext context;
 auto small=mlir::RankedTensorType::get({1},mlir::Float32Type::get(&context));
 auto large=mlir::RankedTensorType::get({1024},mlir::Float32Type::get(&context));
 WorkloadGraph graph;auto a=graph.addValue(WorkloadValue{0,small,"small",false});auto b=graph.addValue(WorkloadValue{0,large,"large",false});
 WorkloadNode n;n.opName="micro.vector";n.attributes=vectorAttributes(context);n.outputs.push_back(WorkloadPort{a,small,std::nullopt});n.outputs.push_back(WorkloadPort{b,large,std::nullopt});graph.addNode(std::move(n));graph.finalize();
 auto machine=searchMachine();for(auto &m:machine.memories)m.capacityBytes=1024;
 auto target=targetWith(machine,R"(rule r { match micro.vector(op = "add"); require executor kind worker; require memory kind sram; require memory kind dram; bundle "b"; emit "e1"; cost 1; })");ASSERT_NE(target,nullptr);
 MappingSearchOptions opts;opts.mode=SearchMode::Exact;CoveringSearch search(graph,*target,context,LayoutContext{},opts);auto result=search.search();ASSERT_TRUE(bool(result))<<llvm::toString(result.takeError());
 std::cout<<"Outputs="<<tileFactsFor(small).bytes<<","<<tileFactsFor(large).bytes<<" bytes; each memory capacity=1024; accepted plans="<<result->plans.size()<<"\n";ASSERT_FALSE(result->plans.empty());
}
TEST(Post108Probe, RepeatedLayoutClass) {
 mlir::MLIRContext context;auto target=targetWithLayouts(searchMachine(),"",kTwoLayouts);ASSERT_NE(target,nullptr);
 MappingCandidate candidate;candidate.id=1;
 LayoutRequirement a;a.layoutClass="t.plain";a.portValue=11;candidate.layoutRequirements.push_back(a);
 LayoutRequirement b;b.layoutClass="t.plain";b.portValue=22;candidate.layoutRequirements.push_back(b);
 auto result=enumeratePlacements(candidate,*target,context,LayoutContext{});ASSERT_TRUE(bool(result))<<llvm::toString(result.takeError());ASSERT_FALSE(result->empty());
 std::cout<<"Required port values=11,22; stored layout solutions="<<result->front().layoutSolutions.size()<<"; retained port value="<<result->front().layoutSolutions.lookup("t.plain").portValue<<"\n";
 EXPECT_EQ(result->front().layoutSolutions.size(),1u);EXPECT_EQ(result->front().layoutSolutions.lookup("t.plain").portValue,22);
}

```
