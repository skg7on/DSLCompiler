# Micro-IR Phase 3 — Real Layout and Resource Legality (gap #6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the four approximations in the mapping engine's legality model with facts derived from the workload: real per-connection bytes/alignment, value liveness, router occupancy, and the solved layout's concrete assignment.

**Architecture:** Additions inside `lib/Mapping/` — a tile-fact helper, a live-range model on the search's partial plan, occupancy threading into `TopologyService`, and the solved layout parameters carried on the instance/placement. No dialect change and no new target policy.

**Tech Stack:** C++20, CMake + Ninja, LLVM/MLIR 24 local / 22 CI, GoogleTest, FileCheck.

**Spec:** `docs/superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md` — §9.3 (legality: capacities and live-range estimates), §12.2 (route legality: intermediate capacity and liveness), §13.3 (solved layouts), §15.2 (transform selection needs endpoint layouts), §17.2 (cost from real bytes).

**Gap source:** the PR-#98 review, item **#6** ("layout and resource legality remain approximations"), plus the deferred minors from phases 1–2 that name the same sites (`kAssumedValueBytes`, "storage accumulates without lifetime expiry", "layout solving establishes that a solution exists but placement discards its concrete parameter assignment", "connection requests do not receive selected producer/consumer layout IDs").

## Global Constraints

- Branch `feat/micro-ir-legality`, **stacked on `feat/micro-ir-phase2`** (PR #101). CI will not run (stacked base); report the locally verified subset.
- `.claude/rules/preferred-mlir-api.md` and `llvm-version-compatibility.md` apply.
- Tests hand-registered in `CMakeLists.txt`. `LLKMapping` is `-fno-rtti -fno-exceptions`; generic code stays target-neutral.
- **Determinism is a requirement.** Any new iteration order must be sorted before it reaches output, a hash, or a cost.
- **Re-baselining is allowed, weakening is not.** Deriving real bytes changes capacity/cost numbers, so some tests will move. Each changed expectation must be *the correct consequence* of a real tile size — state why, and never relax an assertion to make a number match.
- Do not change the *interface* of `TopologyService::enumerateRoutes` other than through the existing `RouteRequest` fields.

### Deferred (not in this plan)

**#4** binding-driven mapping (the next phase — this plan deliberately leaves `micro.candidate` bindings unused beyond provenance), **#5** full materialization, **#7** cost-contract parity, the remaining **#8** chains, and the `warp` owner-vocabulary decision.

---

## Task 1: Derive per-connection bytes and alignment from the moved value

Today every connection, capacity charge, and staging entry uses `kAssumedValueBytes = 4096` and `kAssumedAlignment = 32` (`lib/Mapping/CoveringSearch.cpp`), regardless of the tile actually moving. Derive them from `WorkloadValue::type`.

**Files:**
- Create: `include/LLK/Mapping/TileFacts.h` + `lib/Mapping/TileFacts.cpp` (a small, context-free helper)
- Modify: `lib/Mapping/CoveringSearch.cpp` (the `makeRequest` lambda and every `kAssumedValueBytes` site), `CMakeLists.txt` (register the new source)
- Test: `test/Mapping/tile_facts.cpp` (new, registered) + `test/Mapping/covering_search.cpp`

**Interfaces:**
- Produces: `struct TileFacts { uint64_t bytes; uint64_t alignment; bool known; };` and `TileFacts tileFactsFor(Type valueType);`.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(TileFacts, DerivesBytesFromTheTileShapeAndElementWidth) {
  // !micro.tile<8x32xf32> -> 8*32*4 = 1024 bytes.
  EXPECT_EQ(tileFactsFor(tileType("8x32xf32")).bytes, 1024u);
  // bf16 halves it; a dynamic dim yields known == false.
}
TEST(TileFacts, AlignmentFollowsTheElementWidth) { /* f32 -> 4, bf16 -> 2, f64 -> 8 */ }
TEST(TileFacts, AnUnknownShapeIsReportedNotGuessed) { EXPECT_FALSE(tileFactsFor(dynamic).known); }
TEST(CoveringSearch, CapacityUsesTheRealValueSizeNotAConstant) {
  // A plan whose two 1024-byte tiles fit, but whose 4096-byte assumption would not.
}
```

- [ ] **Step 2: Run to verify failure** — `ctest -R "MappingTileFacts|MappingCoveringSearchTest"`.

- [ ] **Step 3: Implement**

Unwrap the tile to a `ShapedType` reusing the existing printed-form unwrapping in `lib/Mapping/MappingRules.cpp` (factor the shared part into `TileFacts.cpp` rather than duplicating it — this is the one concept with two answers the phase-2 review flagged). `bytes = numElements * elementByteWidth`; `alignment = elementByteWidth`; a dynamic/unknown shape returns `known == false`. Then replace every `kAssumedValueBytes`/`kAssumedAlignment` use in `CoveringSearch.cpp` with the derived value, and delete the constants. When `known == false`, fall back to the current constants **and record a `diagnostic`** saying the size was assumed — never silently.

- [ ] **Step 4: Run** → PASS, re-baselining any test that pinned 4096, each with a note saying why the new number is right.

- [ ] **Step 5: Commit** — `feat(mapping): derive connection bytes and alignment from the tile`.

---

## Task 2: Model value liveness instead of monotonic accumulation

`partial.memoryBytes[node]` only ever grows: a value's bytes stay charged for the whole partial plan even after its consumers are placed, so a legal sequential program can be rejected. Expire a value's bytes once its last consumer is placed.

**Files:** Modify: `lib/Mapping/CoveringSearch.cpp` (the `extend` lambda's capacity accounting); Test: `test/Mapping/covering_search.cpp`.

- [ ] **Step 1: Failing test** — a fixture where three tiles each fit a memory *sequentially* but not simultaneously (with monotonic accumulation it is rejected; with expiry it is legal).
- [ ] **Step 2: Verify red**.
- [ ] **Step 3: Implement** — charge a value's bytes to its producer's memory when the producer is placed; release them when the value's **last** consumer is placed (the `ValueLink` structure already has producers/consumers); keep the honest conservatism where the graph does not fix an order (a value whose consumers are not all placed stays charged). Update the comment that currently says expiry "is not modelled". The global `memoryBudgetBytes` and per-node `capacityBytes` checks both use the live total.
- [ ] **Step 4: Run** → PASS (including the phase-1/2 capacity tests, which should be unchanged where the values genuinely overlap).
- [ ] **Step 5: Commit** — `feat(mapping): expire a value's bytes when its last consumer is placed`.

---

## Task 3: Feed partial-plan occupancy into route legality

`RouteRequest::liveBytesOnIntermediate` exists (added in phase 1) but `Placement.cpp` leaves it unset, so a route may stage through an already-occupied intermediate memory. Thread the search's live occupancy for each memory into the route request.

**Files:** Modify: `lib/Mapping/Placement.cpp` (the `synthesizeConnections`/`synthesizeFanOut` signatures or the request path), `lib/Mapping/CoveringSearch.cpp` (pass the occupancy), `include/LLK/Mapping/Placement.h`; Test: `test/Mapping/routing.cpp` + `test/Mapping/covering_search.cpp`.

**Interfaces:** `ConnectionRequest` gains `llvm::StringMap<uint64_t> intermediateOccupancy;` (or the search passes a callback); state which you chose and why.

- [ ] **Step 1: Failing test** — a multi-hop route whose only intermediate is already holding another live tile is rejected; the same route with an empty intermediate is legal.
- [ ] **Step 2: Verify red**.
- [ ] **Step 3: Implement** — read the intermediate memory's live bytes from the partial plan and set `liveBytesOnIntermediate` per hop's memory (the router's existing check is `capacityBytes - min(live, capacity) >= bytes`). Keep the empty case unchanged.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `feat(mapping): reject routes through an occupied intermediate memory`.

---

## Task 4: Surface the solved layout's parameters, and give connections their endpoint layouts

`LayoutSolution { std::map<std::string, LayoutValue> values; AffineMap map; }` is produced by the solver and then discarded — the instance keeps only the layout *id*. And connection requests never receive their endpoints' bound layouts, so `synthesizeConnections`' transform path can never fire from the search.

**Files:** Modify: `include/LLK/Mapping/MappingPlan.h` (carry the solved assignment on `CandidateInstance` and `PlanPlacement`), `lib/Mapping/MappingPlan.cpp` (canonical string/hash), `lib/Mapping/Placement.cpp` (record it; populate the request's layouts from the instances), `lib/Mapping/CoveringSearch.cpp` (`makeRequest` sets `producerLayout`/`consumerLayout`); Test: `test/Mapping/placement.cpp`, `test/Mapping/covering_search.cpp`, `test/Mapping/mapping_plan.cpp`.

**Note:** the solved fields must be **excluded from the canonical string** if phase 2's `LayoutRequirement::elementType/rank` precedent applies — but the *bound* layout id (already in `layoutBindings`) stays. Decide deliberately: a different solved parameter set on the same layout id — is that a different instance or not? State your choice and test it.

- [ ] **Step 1: Failing tests** — (a) an instance records the solved parameter assignment (e.g. `VW=8`) and the placement carries it; (b) a producer/consumer pair with different bound layouts produces a `LayoutTransform` (or transfer+transform) alternative from the search, not just from a direct unit call.
- [ ] **Step 2: Verify red**.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** → PASS; confirm plan ids move only if you decided the solved assignment is part of identity.
- [ ] **Step 5: Commit** — `feat(mapping): carry solved layout parameters and endpoint layouts`.

---

## Self-review

- **Coverage:** §9.3 (capacity + live range) = T1, T2; §12.2 (intermediate capacity and liveness) = T3; §13.3/§15.2 (solved layouts in transform selection) = T4. §17.2's "cost from real bytes" is a side effect of T1 (route cost consumes `bytes`).
- **Sequencing:** T1 before T2 and T3 (they consume real bytes); T3 after T2 (occupancy is the live total); T4 is independent but lands last because it is the only one that may perturb plan identity.
- **Re-baselining risk:** T1 changes many numbers. The plan requires each changed expectation to be justified by the real tile size, not relaxed.
