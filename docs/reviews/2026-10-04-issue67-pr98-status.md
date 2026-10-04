# Issue #67 / PR #98 implementation reassessment

Reviewed on 2026-10-04. Issue: https://github.com/skg7on/DSLCompiler/issues/67
PR: https://github.com/skg7on/DSLCompiler/pull/98
Follow-up tracker: https://github.com/skg7on/DSLCompiler/issues/106

## Revision and conclusion

Production source and full local suite were reviewed at `cae35761821222ca0ffa2b2345150f53b751bb84`.
During review the PR advanced to `797e247dded32729af173d3f562464912ad1197d`.
The intervening commit changes only `micro_map_candidate.sh`; it was reviewed and
successfully exercised under dash. All production findings therefore apply to
`797e247` as well. PR #98 is open, not merged, at the last status check.

The PR incorporates substantial phase-2, phase-3 and phase-4 improvements, but
it does not close all issue #67 design goals. Its original description and its
109-test claim are stale relative to the present 121-test implementation.
The mapping/search foundation and public tooling are much stronger; complete
selected-plan execution and target lowering remain unfinished. Several new
resource/connection correctness defects also need resolution.

## Verification evidence

- Fresh isolated build against local LLVM/MLIR 24.
- `cmake --build build --target check-llk llk-compile -j 6` completed at `cae3576`.
- 121 registered CTests: 119 passed, two CPU-specific skips, zero failures.
- Previous Linux CI and coverage runs failed only `MicroMappingCandidate`.
  Reproduced using dash: high-bit hex shell arithmetic saturated to INT64_MAX
  while bound IR correctly recorded signed i64. This was a test portability bug.
- Latest test-only commit `797e247` fixes the conversion through complement
  arithmetic; the complete candidate script passes under dash.
- Additional capacity probe accepts a 4096-byte transfer into a 1024-byte
  memory when the consumer output is four bytes. This demonstrates a legality bug.
- Additional connection probe: a legal single-consumer layout transform exists,
  but adding a second identical consumer causes fan-out to return no alternatives.
- Unmaterialized tile-route fixture returns exit 0 and emits `micro.plan` with
  a warning that the selected connection was not materialized.
- Compiler-generated matmul mapping command succeeds. Registered matmul/SwiGLU,
  report/rebind and report-only integration tests pass. They establish mapping
  coverage and metadata, not emitted target-code execution.

The probe sources and binaries were retained locally under `build/review/`.
The appendix preserves the focused probe bodies and replay instructions for
remote reviewers. These probes are not part of the committed CTest suite.

## Reassessment of the nine previously recorded gaps

| Prior gap | Current status in PR #98 | Remaining boundary |
|---|---|---|
| 1. Persistent bindings drive mapping | Partially closed | Candidate loading, domain checks, rule-parameter pinning, binding provenance and replay work. Layout-kind binding compares directly against target IDs and fails for shipped namespaced AVX2 layouts. Other bound search axes/global shape constraints are not a complete mapping bridge; tune/compile are not integrated. |
| 2. Objective/search correctness | Mostly closed for implemented subset | Primary/secondary metrics drive beam/final ranking; calibrated instance costs feed bounds; exact maximize disables unsound bound pruning. Exact cost ties can still prune a lower plan ID, with truncation disclosed. Connection alternatives are chosen locally; complete global optimality is not established. |
| 3. Selected-ID workflow | Closed for tested single-kernel subset | Raw report hex IDs, mode/width/top-K replay, candidate replay, report-only and public verification have tests. Multi-kernel selection and completeness verification remain limited. |
| 4. Compiler-generated AVX2 workload coverage | Mapping coverage closed | Tile copy/store and vector convert/silu/mul rules are present; lowered matmul and SwiGLU map successfully. Real mapped target emission/runtime execution is still absent. |
| 5. Layout/resource legality | Substantially improved, still partial | Actual static bytes/element alignment, producer-value expiry, intermediate occupancy, solved parameter/map retention and endpoint layout association work. Destination transfer capacity is undercounted; fan-out/fan-in compatibility has holes; transform maps/hops remain underspecified; replica/gather lifetimes do not expire. |
| 6. Complete materialization | Open | Canonical tile movement, layout transforms, replication/gather/reduction and consumer-specific rewiring are incomplete. Unsupported decisions are warnings followed by successful output. |
| 7. Target bundle/emitter handoff | Open | Typed bundles are transient, but solved layouts and bundle parameters are dropped in bound IR. Default emitter checks names/primitive types; mapped verifier does not invoke emitter verification, and there is no real target lowering hook. |
| 8. Shared cost semantics/cache keys | Partial | Common MachineModel/events and concrete legacy hop attribution exist. Cache keys lack type/shape/attribute/resolved-bundle identity; mapped executor/layout/bundle decisions are not fully consumed by perf. Production calibration remains separate #51/#52 work. |
| 9. Acceptance coverage/architecture debt | Partial | New compiler-generated workload, binding, reports and property tests add real coverage. No complete required-transform/tile-two-hop/target-execution/perf-parity chain. Fused multi-node rules remain later work; existing generic warp/wave vocabulary remains. |

## Concrete findings

### P1: Transfer destination capacity is not charged

`lib/Mapping/CoveringSearch.cpp:583-588` accounts for consumer outputs.
Lines 684-686 deliberately omit a plain transfer destination under the assumption
that it is the consumer's already-accounted tile. That assumption fails when
input and output sizes differ, e.g. reduction or narrowing. The router's capacity
check excludes destination memories because search is expected to own their
occupancy. The compiled review probe produced:

```text
Transferred bytes=4096, destination capacity=1024,
consumer output bytes=4, accepted plans=1
Connection kind=1, route=dram.0 -> acc.0
```

Charge the actual destination input buffer and its lifetime, deduplicating only
when storage aliasing is explicitly established. Add a rejection regression for
this case and a legal sequential-lifetime case.

### P2: Layout-only fan-out is discarded

`lib/Mapping/Placement.cpp:698-710` keeps only replication alternatives after
fan-out enters its non-direct path. A producer and two consumers sharing one
memory but requiring a layout conversion generate legal `LayoutTransform`
alternatives individually; these are discarded as non-replication.
The review probe confirms a legal single-consumer alternative and an empty
fan-out result. Preserve legal shared transforms as well as copies.

### P2: Shared replicas can conflate incompatible layouts

`lib/Mapping/Placement.cpp:645-656` groups by destination memory only.
Lines 714-721 choose one member's transfer/transform and assign every consumer
to it. Individually legal A-to-B and A-to-C consumers cannot share one resulting
layout. This is a source-confirmed path; unlike the preceding two findings,
it was not separately exercised by a compiled probe. Group only compatible
representations or synthesize distinct transformed replicas.

### P2: Gather checks only a representative consumer

`lib/Mapping/CoveringSearch.cpp:822-855` groups gather consumers by memory and
checks producer connections only against the first consumer. Lines 878-880 attach
all consumers to the result. A second consumer's affine/type/layout requirements
are not validated. This is source-confirmed, not a separate runtime reproduction.
Validate each consumer and partition incompatible gather outputs.

## Unfinished design requirements with source evidence

- **Layout kind versus implementation ID:**
  `include/LLK/Conversion/MicroMapping/SearchBindingLoader.h:88-100` documents
  that `blocked` cannot select `avx2.blocked_2d`. The candidate test intentionally
  expects the shipped AVX2 target to fail this combination. A target-declared
  kind-to-implementation association is needed; renaming target IDs to canonical
  kind strings is not the complete multi-layout solution.
- **Tile routes:** `lib/Mapping/PlanBinder.cpp:295-300` only materializes
  `ShapedType` movement; custom `!micro.tile` needs a destination-memory type.
- **Transforms/replication/reduction:** binder lines 248-253 report unsupported
  connection kinds, and 400-403 report unapplied transferred layout transforms.
- **Warning-and-success:** `lib/Conversion/MicroMapping/MicroMappingCommon.h:390-406`
  installs a clone and returns success even with unmaterialized decisions.
  This cannot establish full execution of the selected plan.
- **Persistence:** binder lines 157-179 write layout IDs and bundle name/key,
  but omit `PlanPlacement.layoutSolutions` and `TargetBundle.parameters`.
- **Target phase:** `include/LLK/Mapping/MappingTarget.h:35-56` has verification
  but no lowering API. `lib/Mapping/MappingTarget.cpp:76-105` supplies a generic
  declared-key verifier. `PlanBinder.cpp:492-498` checks key membership without
  invoking the target emitter's verification contract.
- **Perf parity:** `lib/Perf/MicroDAG.cpp:334-374` handles stamped legacy route
  hops, but compute selection/type facts at 803-808 and 847-850 do not consume
  selected `micro.mapping` executor/layout/bundle decisions.
- **Cache identity:** `include/LLK/Mapping/LatencyProvider.h:44-53` lacks operand/
  result types, relevant attributes and bundle parameters. Search fills a name,
  rule version, bundle name, first sorted layout and executor kind at
  `CoveringSearch.cpp:438-452`, leaving distinct workloads with the same key.
- **Exact tie completeness:** `CoveringSearch.cpp:1097-1113` acknowledges that
  cost-only pruning may lose a lower exposed plan ID and discloses truncation.
  Ordering retained plans is improved; globally exact top-K tie selection is not.
- **Tool integration:** `tools/llk-tune/llk-tune.cpp:380-388` remains the legacy
  perf tuning/schedule-record path. `llk-compile` has no mapped-target entry point.
- **Fused coverage:** one-op rules remain the implemented parser/matcher.
  Design section 14.2 permits initial one-op delivery, but the full acceptance
  matrix still asks for fused rules; a lowered multi-op SwiGLU graph is different
  from a fused multi-node mapping candidate.

## Recommendation

Keep issue #67 open. Before calling PR #98 complete, fix the confirmed capacity
and connection defects, validate latest CI, and scope its claims to implemented
mapping/search/public-pass behavior. Remaining execution, target lowering,
layout-kind binding, tuning integration and perf/cache parity work must be
explicitly tracked. Merging PR #98 can deliver substantial progress without
satisfying every epic acceptance criterion.

This report records a revision-pinned review. Later changes must be checked
against these findings; the observations do not automatically apply to future
PR revisions. No implementation changes are included in this documentation.

## Appendix: replaying the extra probes

Use the [reviewed source revision](https://github.com/skg7on/DSLCompiler/tree/797e247dded32729af173d3f562464912ad1197d).
Build `MappingCoveringSearchTest` and `MappingConnectionsTest`. Append each body
below to its indicated test file in an isolated checkout, then rebuild and run
with `--gtest_filter=ReviewCapacity.*` or `--gtest_filter=ReviewProbe.*`. The
bodies reuse that revision's existing fixture helpers.

The capacity probe demonstrates the bug by asserting the observed acceptance;
when promoting it to a regression test, require rejection with
`memory_capacity_exceeded`. The fan-out probe already asserts the desired
behavior and fails on the reviewed revision.

### Capacity probe: test/Mapping/covering_search.cpp

Add `mlir/IR/BuiltinTypes.h` and `<iostream>` if not already included.

```cpp
TEST(ReviewCapacity, InputTransferExceedsDestination) {
  mlir::MLIRContext context;
  auto large =
      mlir::RankedTensorType::get({1024}, mlir::Float32Type::get(&context));
  auto small =
      mlir::RankedTensorType::get({1}, mlir::Float32Type::get(&context));
  WorkloadGraph graph;
  auto input = graph.addValue(WorkloadValue{0, large, "in", true});
  auto middle = graph.addValue(WorkloadValue{0, large, "mid", false});
  auto output = graph.addValue(WorkloadValue{0, small, "out", false});
  WorkloadNode producer;
  producer.opName = "micro.vector";
  producer.attributes = vectorAttributes(context, "produce");
  producer.inputs.push_back(WorkloadPort{input, large, std::nullopt});
  producer.outputs.push_back(WorkloadPort{middle, large, std::nullopt});
  graph.addNode(std::move(producer));
  WorkloadNode consumer;
  consumer.opName = "micro.vector";
  consumer.sourceOrdinal = 1;
  consumer.attributes = vectorAttributes(context, "consume");
  consumer.inputs.push_back(WorkloadPort{middle, large, std::nullopt});
  consumer.outputs.push_back(WorkloadPort{output, small, std::nullopt});
  graph.addNode(std::move(consumer));
  graph.finalize();
  auto machine = fanMachine();
  machine.memories[1].capacityBytes = 1024;
  auto target = targetWith(machine, kFanRules);
  ASSERT_NE(target, nullptr);
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  CoveringSearch search(graph, *target, context, LayoutContext{}, options);
  auto result = search.search();
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  std::cout << "Transferred bytes=" << tileFactsFor(large).bytes
            << ", destination capacity=" << machine.memories[1].capacityBytes
            << ", consumer output bytes=" << tileFactsFor(small).bytes
            << ", accepted plans=" << result->plans.size() << "\n";
  ASSERT_FALSE(result->plans.empty());
  for (auto &connection : result->plans[0].connectionPlans)
    std::cout << "Connection kind=" << int(connection.kind)
              << ", route=" << connection.route.front() << " -> "
              << connection.route.back() << "\n";
  EXPECT_EQ(result->plans[0].connectionPlans.size(), 1u);
}
```

### Fan-out probe: test/Mapping/connections.cpp

```cpp
TEST(ReviewProbe, LayoutOnlyFanOutKeepsLegalAlternative) {
  auto model = connectionMachine();
  TopologyService topology(model);
  auto r = baseRequest();
  r.consumerMemory = r.producerMemory;
  r.consumerLayout = "t.b";
  auto pair = synthesizeConnections(r, model, topology);
  ASSERT_TRUE(bool(pair));
  ASSERT_FALSE(pair->empty());
  auto r2 = r;
  r2.consumer = 3;
  std::vector<ConnectionRequest> consumers{r, r2};
  auto group = synthesizeFanOut(r, consumers, model, topology);
  ASSERT_TRUE(bool(group));
  EXPECT_FALSE(group->empty());
}
```
