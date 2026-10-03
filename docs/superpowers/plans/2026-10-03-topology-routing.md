# Topology Routing Implementation Plan (D2 / epic #67)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deterministic, bounded, cycle-free enumeration of memory-to-memory routes over a `MachineModel`, producing ranked `MemoryRoute` alternatives for D5 (connections) and the #50 plan binder.

**Architecture:** `RouteRequest` / `MemoryRoute` / `TopologyService` in `LLK/Mapping` (design §12), consuming `LLKMachine`'s topology and `LLK/Mapping`'s `Cost` and id aliases. Best-first search over the memory×link graph; results sorted by estimated cost, then hop count, then lexicographic link-id sequence.

**Tech Stack:** C++20, LLVM ADTs, CMake/Ninja, GoogleTest. No MLIR ops — this is plain-data search over the machine graph.

**Spec:** design §12 (multi-hop routing), §12.1 API, §12.2 legality, §12.3 determinism. Plan §4 D2.

## Global Constraints

- Deterministic: identical inputs yield identical route order, independent of container iteration order.
- Cycle-free: no route repeats a memory node.
- Bounded: at most `limit` routes, and path expansion bounded by `maxHops`.
- Never compare target ids against Micro vocabulary; engine legality is a topology query.
- Depends on `LLKMachine` (topology) and D1's `Cost`/ids; must not depend on `LLK/Perf` or a backend.

## Documented deviations from design §12.1

- `RouteRequest` omits `elementType`: v2 memory and link nodes carry no element-type facts, so a type check would be vacuous. Type support is a D4 rule concern (rules declare supported element types on compute capabilities). An `alignmentBytes` requirement is added instead, which v2 memories *do* model.
- `MemoryRoute` carries the summed `Cost` and the ordered node/link/engine sequences (§12.1), with a `hopCount()` helper.

## File Structure

| File | Responsibility |
|---|---|
| `include/LLK/Mapping/Routing.h` | `RouteRequest`, `MemoryRoute`, `RouteOptions`, `TopologyService` |
| `lib/Mapping/Routing.cpp` | Best-first enumeration, legality checks, cost estimation |
| `test/Mapping/routing.cpp` | Direct/multi-hop ordering, cycle freedom, capacity/engine/visibility rejection, determinism |

---

### Task 1: Route types, legality, and deterministic enumeration

**Files:**
- Create: `include/LLK/Mapping/Routing.h`, `lib/Mapping/Routing.cpp`
- Create: `test/Mapping/routing.cpp`
- Modify: `CMakeLists.txt` (`LLKMapping` sources + `add_llk_mapping_test`)

**Interfaces:**
- `struct RouteRequest { MemoryNodeId source, destination; uint64_t bytes; uint64_t alignmentBytes = 1; std::optional<ExecutorId> producerExecutor, consumerExecutor; }`
- `struct MemoryRoute { llvm::SmallVector<MemoryNodeId> nodes; llvm::SmallVector<LinkId> links; llvm::SmallVector<ExecutorId> transferEngines; Cost cost; uint64_t hopCount() const; }`
- `struct RouteOptions { unsigned maxRoutes = 8; unsigned maxHops = 4; }`
- `class TopologyService { TopologyService(const MachineModel &, RouteOptions = {}); llvm::Expected<llvm::SmallVector<MemoryRoute>> enumerateRoutes(const RouteRequest &, unsigned limit) const; }`

**Legality (design §12.2), checked per candidate route:**
1. source and destination exist and are memories;
2. `producerExecutor`/`consumerExecutor`, when given, can see their endpoint memory (`MachineModel::isVisible`);
3. consecutive nodes are joined by a directed link whose `source`/`destination` match;
4. every hop's link declares at least one transfer engine, and at least one of them is attached to an executor that can see the hop's source memory;
5. every memory on the route supports the requested alignment (`memory.alignmentBytes % request.alignmentBytes == 0`);
6. intermediate (non-endpoint) memories have `capacityBytes >= request.bytes`;
7. `bytes > 0` and `alignmentBytes > 0`.

**Cost (design §12.1/§17.2):** `latencyCycles = Σ hops (link.latencyCycles + bytes / link.bandwidthBytesPerCycle)`; `localBytes = bytes`. The route cost is the summed hop cost.

**Search:** best-first over partial paths with a `(cost, hopCount, linkIdSequence)` ordering; expand neighbors in sorted link-id order; skip a node already on the path (cycle freedom); stop after `limit` complete routes. `source == destination` yields one trivial route (no links, zero cost).

- [ ] **Step 1: Write the failing test**

```cpp
// test/Mapping/routing.cpp
#include "LLK/Mapping/Routing.h"
#include <gtest/gtest.h>
using namespace mlir::llk::machine;
using namespace mlir::llk::mapping;

MachineModel diamond();  // mems: dram/l2/sram/acc; links as below
// dram->l2 (bw32, lat100), l2->sram (bw64, lat10), sram->acc (bw128, lat2),
// dram->acc direct (bw8, lat200), sram->dram back edge (bw32, lat100)

TEST(Routing, PrefersCheaperMultiHopOverExpensiveDirect) {
  TopologyService service(diamond());
  RouteRequest request{"dram", "acc", 1024, 64, std::nullopt, std::nullopt};
  auto routes = service.enumerateRoutes(request, 8);
  ASSERT_TRUE(static_cast<bool>(routes));
  ASSERT_GE(routes->size(), 2u);
  EXPECT_EQ((*routes)[0].links, (llvm::SmallVector<LinkId>{"dram_to_l2", "l2_to_sram", "sram_to_acc"}));
  EXPECT_TRUE((*routes)[0].cost.latencyCycles < (*routes)[1].cost.latencyCycles);
}

TEST(Routing, RoutesAreCycleFree) {
  // With the sram->dram back edge, a naive search loops; assert every route
  // has unique nodes.
}

TEST(Routing, RespectsTheLimit) {
  // limit 1 returns exactly one route: the cheapest.
}

TEST(Routing, ReportsNoRoute) {
  // A topology without dram->acc and without the connecting path fails.
}

TEST(Routing, RejectsInvisibleEndpoint) {
  // producerExecutor that cannot see the source memory fails.
}

TEST(Routing, RejectsTooSmallIntermediate) {
  // bytes larger than an intermediate memory's capacity fails.
}

TEST(Routing, RejectsHopWithoutEngine) {
  // A link with no transfer engines cannot carry the hop.
}

TEST(Routing, IdenticalRequestsProduceIdenticalRoutes) {
  // Two enumerations agree on link sequences.
}

TEST(Routing, SameMemoryIsATrivialRoute) {
  // source == destination -> one route, zero links, zero cost.
}
```

- [ ] **Step 2: Run test to verify it fails** — `ninja -C build MappingRoutingTest` fails: header/target missing.
- [ ] **Step 3: Register `lib/Mapping/Routing.cpp` in `LLKMapping` and `MappingRoutingTest` via `add_llk_mapping_test` in `CMakeLists.txt`.**
- [ ] **Step 4: Implement `Routing.h` and `Routing.cpp`.** Data structures per the interfaces above; `enumerateRoutes` validates the request, runs best-first search, checks per-route legality, computes cost, and returns routes sorted by `(cost.latencyCycles, hopCount, link-id sequence)`. Failures return `llvm::Error` naming the reason (`no route from 'dram' to 'acc'`, `executor 'core.0' cannot see memory 'dram'`, ...).
- [ ] **Step 5: Run tests to verify they pass.**
- [ ] **Step 6: Commit** `feat(mapping): add deterministic topology routing (D2)`

---

### Task 2: Full build, suite, and verification record

- [ ] **Step 1:** `ninja -C build` — clean.
- [ ] **Step 2:** `ctest --test-dir build -R "Mapping|Machine" --output-on-failure` — new and neighbouring suites pass.
- [ ] **Step 3:** `ctest --test-dir build --output-on-failure` — record registered/passed/skipped/failed.
- [ ] **Step 4:** Run the deprecated-API audit scoped to the new files.
- [ ] **Step 5: Commit** `docs(mapping): record D2 verification results`

---

## Self-Review

**Plan §4 D2 coverage:** `RouteRequest`/`MemoryRoute` (Task 1); deterministic bounded enumeration (Task 1); link/engine validation, visibility, intermediate capacity (Task 1 legality); route diagnostics (Task 1 error paths). ✅

**Merge criterion:** direct and multi-hop fixtures cycle-free and ordered by cost → hop count → link-id sequence (Task 1 tests). ✅

**Non-scope respected:** no materialization into Micro-IR (§12.4 — that is D5/plan binding); no placement; no `micro-perf` coupling.

**Type consistency:** `MemoryNodeId`/`LinkId`/`ExecutorId` come from `MappingPlan.h`, `Cost` from `CostModel.h`, `MachineModel` from `LLK/Machine` — all defined once and reused.
