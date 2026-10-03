# Binding a Selected Plan to Micro-IR (#50 revision / epic #67)

**Goal:** Take a ranked `CoveringPlan` from D6 and materialize its selections — without mutating the source — as generic, target-neutral Micro metadata, then verify the result in layers (design §18).

## Structure

| File | Responsibility |
|---|---|
| `include/LLK/Mapping/WorkloadGraph.h` + `lib/.../WorkloadGraph.cpp` | `WorkloadGraphBinding`: node↔op and value↔SSA correspondence |
| `include/LLK/Mapping/MappingPlan.h` | `PlanPlacement` / `PlanConnection`, so a plan carries what §18.1 materializes |
| `include/LLK/Mapping/PlanBinder.h` + `lib/.../PlanBinder.cpp` | `bindPlan`, `verifyMappedMicroIR` |
| `test/Mapping/plan_binder.cpp` | tests |

## The two gaps this had to close first

1. **A graph could not be traced back to its IR.** `WorkloadGraph` is deliberately context-free — it holds stable ids, not MLIR pointers — so binding needs a correspondence that only *extraction* can build, since ids are assigned and then canonicalized there. `extractWorkloadGraph` now optionally fills a `WorkloadGraphBinding`; `finalize` reports its value remap so the binder can follow it.
2. **A `CoveringPlan` did not carry its own selections.** It listed instance and connection ids, which is enough to rank and not enough to materialize. `PlanPlacement` and `PlanConnection` add the rule, bundle, executor, memories, layouts, and route — exactly §18.1's list — and the canonical plan string covers them, so plan ids stay content-derived.

## Decisions

- **Metadata only, no new ops.** §18.2's copies, allocations, transforms, and waits are *not* emitted. A connection's endpoints are inferred, not chosen: D5 falls back to the executor's first visible memory, and D6 assumes a byte count. Materializing copies from inferred placement would bake an assumption into IR that looks authoritative. The metadata makes the same selection inspectable and verifiable until a plan carries chosen memories — at which point emission becomes a small, safe addition on top of the binding helper.
- **The dialect verifier learns nothing.** `micro.plan`, `micro.mapping`, and `micro.routes` are dictionaries of strings and integers. Micro ODS gained no target vocabulary, and `llk-opt` prints and re-parses mapped IR with no target plugin (design §18.3), which a test asserts.
- **Verification is layered as specified**: structural (MLIR verifier) → machine-aware (ids resolve, memories are visible from their executor, route hops are linked) → target (emitters are declared). The first violation wins, walking operations in order.

## Verification Results (2026-10-03)

- **Build:** `ninja -C build` — clean.
- **New tests:** `MappingPlanBinderTest` **4/4**.
- **Full suite:** `ctest --test-dir build --output-on-failure` — 103 registered, **101 passed, 2 skipped, 0 failed**.
- **Deprecated-API audit:** clean.

### Two D4 tests had to change, and one change is a lesson

Adding a copy rule to the shipped AVX2 set (a target must implement movement) broke two assertions in `mapping_rules.cpp`:

1. It hardcoded the emitter key list, so the new rule failed to verify. It now reads `target::avx2::emitterKeys()`, which is the package that owns them — the duplication was the bug, not the new rule.
2. It asserted exactly three shipped rules. It now asserts that specific rules are present, so growing the shipped set does not fail a test whose subject is that the set loads.

### Not done

Emitting tile copies, allocations, transforms, waits, and barriers (design §18.2), for the reason above. Everything §18.1 and §18.3 asked for is implemented and tested.
