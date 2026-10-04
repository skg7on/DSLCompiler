# Micro-IR Phase 2 — Restore-the-Claims Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four review gaps where PR #98's own claims are currently false: the search bound/pruning is not objective-aware or admissible (#1), the report→bind workflow does not round-trip (#2), the shipped AVX2 target cannot map compiler-generated tile programs (#3), and the tooling/acceptance surface is incomplete (#8, partial).

**Architecture:** All changes sit on top of PR #98 (`feat/micro-ir-gap-closure`). The search bound becomes a multi-dimensional, direction-aware `Cost`; the bind pass gains the search options needed to reproduce a reported plan; the AVX2 rule package gains the operations `llk-to-micro` actually emits; and two small entry points (report-only, verify-mapping) plus integration tests close the tooling gap.

**Tech Stack:** C++20, CMake ≥ 3.20 + Ninja, LLVM/MLIR 24 local / 22 CI, GoogleTest, FileCheck, `llvm::json`.

**Spec:** `docs/superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md` (§17.1 ranking, §16.2 caps, §22.1 ordering, §22.2 report, §18.3 verification phases, §21 tools, §14.1 rule contents, §15.2/§18.2 emission). Gap source: the PR-#98 review of 2026-10-04.

## Global Constraints

- Base is `feat/micro-ir-gap-closure` (PR #98). Branch `feat/micro-ir-phase2` is stacked on it; **CI will not run** on a stacked base until #98 merges — report the locally verified subset.
- `.claude/rules/preferred-mlir-api.md` and `llvm-version-compatibility.md` apply: `Op::create(builder, …)`, `mlir::cast<T>`, `isa<T>`; guard any API that differs between LLVM 22 and 24.
- Tests are hand-registered in `CMakeLists.txt` (`add_llk_*_test`, `add_llk_filecheck_test`). A new file must be registered or it never runs.
- `LLKMapping` takes `-fno-rtti -fno-exceptions`; no deprecated API (`-Werror=deprecated-declarations`).
- Determinism is a requirement (§3.3/§22.1): never iterate an unordered container as output order.
- Target policy stays in `mapping/*.llkmap` + `machines/*.yaml`; generic `lib/Mapping` never names a target.
- Every task ends with `ninja` clean and the affected tests green before committing.

### Out of scope for this plan (deferred by ruling)

**#4** binding-driven mapping, **#5** full materialization (layout transforms, replication/reduce emission, routed `!micro.tile` transfers, barriers, per-consumer routes, invoking the emitter), **#6** real per-tile bytes/alignment/liveness and router occupancy, **#7** cost-contract parity, the remaining **#8** E2E chains, and the `warp`/`wave` owner-vocabulary decision. The dialect decision for #5 is already taken and recorded: **add a generic Micro property carrying an opaque target layout id** when #5 is implemented.

---

## Task 1: Make the search bound multi-dimensional, measured, and direction-aware

`lib/Mapping/CoveringSearch.cpp`'s `bound` returns a scalar `double` built from `entry.instance.localCost.latencyCycles` — the *static* cost — while the accumulated `partial.cost` uses `entry.cost`, which a `LatencyProvider` may have lowered (line 355). So a calibrated-down instance makes the bound over-estimate and prune a genuinely better plan; and the bound is latency-only, so it cannot steer a non-latency objective.

**Files:**
- Modify: `lib/Mapping/CoveringSearch.cpp` (`bound`, `Partial::lowerBound`)
- Test: `test/Mapping/covering_search.cpp`

**Interfaces:**
- Produces: `Cost boundCost(const Partial &)` and a `Cost Partial::lowerBound` (was `double`).

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(CoveringSearch, TheBoundUsesTheMeasuredCostNotTheStaticOne) {
  // A provider that reports a cost BELOW the static lower bound. The bound must
  // follow the measurement, or a plan that is genuinely cheapest is pruned.
  // Static = 100/instance; provider returns 1 for one instance.
  // Assert the search still finds the plan that uses the measured-cheap instance.
}

TEST(CoveringSearch, TheBoundIsObjectiveAware) {
  // Two plans: A cheaper on latency, B cheaper on dram_bytes.
  // With objective.primary == DramBytes, beam/exact must not prune B on a
  // latency bound. Assert B survives and ranks first.
}
```

- [ ] **Step 2: Run to verify failure** — `ctest -R MappingCoveringSearchTest`; the measured case prunes (plan absent), the objective case orders by latency.

- [ ] **Step 3: Implement**

```cpp
/// Optimistic completion cost: accumulated cost plus, for every uncovered
/// node, the componentwise best still reachable. "Best" is direction-aware:
/// a minimize objective wants the smallest reachable value, a maximize
/// objective the largest -- an optimistic bound must always favour the
/// branch it is bounding, or it stops being admissible.
Cost boundCost(const Partial &partial) {
  Cost total = partial.cost;
  for (size_t i = 0; i < tables.size(); ++i) {
    if (partial.chosen[i]) continue;
    std::optional<Cost> best;
    for (const InstanceEntry &e : tables[i].instances)
      best = best ? bestCostForObjective(*best, e.cost, options_.objective) : e.cost;
    if (!best) return infiniteCost();
    total = addCost(total, *best);
  }
  return total;
}
```

Use `entry.cost` (measured-or-static), never `instance.localCost`. Direction: for each metric, take min when `objective.minimize`, max otherwise (the order has one direction flag).

- [ ] **Step 4: Run** `./build/MappingCoveringSearchTest` → PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/Mapping/CoveringSearch.cpp test/Mapping/covering_search.cpp
git commit -m "fix(mapping): make the search bound multi-dimensional and measured"
```

---

## Task 2: Order the beam and prune exact by the declared objective

With a `Cost` bound in place, beam ordering and exact pruning must compare through `costLess(..., objective)` — including the **maximize** direction, where today's `branch.lowerBound >= bestCosts.back()` is inverted (it keeps the cheapest, not the best).

**Files:** Modify: `lib/Mapping/CoveringSearch.cpp` (beam `llvm::sort`, exact prune, `bestCosts`); Test: `test/Mapping/covering_search.cpp`.

- [ ] **Step 1: Failing tests** — (a) a maximize-latency objective where exact search must retain the highest-latency plan and not report truncation pruning it; (b) a secondary-metric improvement (equal primary, better secondary) that survives instead of being pruned as a tie.
- [ ] **Step 2: Verify red** — `ctest -R MappingCoveringSearchTest`.
- [ ] **Step 3: Implement** — beam: sort by `costLess(lowerBound)` (direction-aware), then covered desc, then `id`. Exact: keep `std::vector<Cost> bestCosts`, prune only when the bound is **not better than** the worst kept cost under `objective` (i.e. `!costLess(bound, bestCosts.back(), objective)` for minimize; the mirrored test for maximize). Keep the existing "truncation is reported" contract.
- [ ] **Step 4: Run** → PASS, and confirm the §16.4 beam/exact agreement test still holds.
- [ ] **Step 5: Commit** — `fix(mapping): rank the beam and prune exact by the declared objective`.

---

## Task 3: Accept the plan id the report actually prints

`PlanReport` writes the id as hex (`hexId`); `MicroBindPlanPass` parses decimal only (`getAsInteger(10, …)`), so the documented workflow fails on the report's own output.

**Files:** Modify: `lib/Conversion/MicroMapping/MicroBindPlanPass.cpp`; Test: `test/Conversion/MicroMapping/` (FileCheck or the pass test) plus a unit-level helper test if practical.

- [ ] **Step 1: Failing test** — feed the pass a hex id (e.g. `0081ef1286442d39`) produced by the report and assert it is accepted; also accept `0x`-prefixed and decimal, signed and unsigned.
- [ ] **Step 2: Verify red** — the pass emits "must be a 64-bit integer".
- [ ] **Step 3: Implement** — one parser: try `0x`/hex via `getAsInteger(16, …)`, then decimal unsigned, then decimal signed; reject anything else with the same diagnostic. Document that the report's spelling is accepted verbatim.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `fix(mapping): accept the report's hex plan id in --micro-bind-plan`.

---

## Task 4: Make a reported plan id reproducible and select top-K by the exposed id

Binding forces `deterministic` search, which stops at the first legal plan, so an id from a beam/exact `--micro-map` can never be reproduced. Separately, top-K trimming happens before `computePlanId`, so ties are not broken by the documented plan-ID key.

**Files:** Modify: `lib/Conversion/MicroMapping/MicroBindPlanPass.cpp` (add `mode`, `beam-width`), `lib/Conversion/MicroMapping/MicroMappingCommon.h` (`runMappingSearch` honors a requested mode), `lib/Mapping/CoveringSearch.cpp` (top-K selection keyed by the exposed id); Test: `test/Conversion/MicroMapping/`.

- [ ] **Step 1: Failing tests** — (a) `--micro-map mode=beam` then `--micro-bind-plan mode=beam` reproduces the reported id; (b) with tied costs, the retained top-K are the (objective, plan-id)-smallest K.
- [ ] **Step 2: Verify red**.
- [ ] **Step 3: Implement** — `--micro-bind-plan` gains `mode`/`beam-width` (same spelling as `--micro-map`) and stops forcing deterministic; the pass's header documents that reproducing an id requires the same search options. Fix top-K: build each plan's `id` first (or sort the complete set by `(cost, id)` under the objective *before* trimming) so the cap keeps the documented K.
- [ ] **Step 4: Run** → PASS; confirm the property test for `(objective, plan.id)` ordering still holds.
- [ ] **Step 5: Commit** — `fix(mapping): reproduce reported plan ids and trim top-K by plan id`.

---

## Task 5: Map the tile movement ops the lowering emits

`kNodeOps` includes `micro.tile_async_copy` and `micro.tile_store`, but `mapping/x86-avx2/rules.llkmap` matches neither (only `micro.async_copy`), so `llk-to-micro` output fails with `no_matching_rule`.

**Files:** Modify: `mapping/x86-avx2/rules.llkmap`, `lib/Target/X86/Mapping/AVX2MappingTarget.cpp` (emitter keys), `docs/design/llkmap-rule-grammar.md` if the example set changes; Test: `test/Mapping/avx2_target.cpp` + a new FileCheck fixture.

- [ ] **Step 1: Failing test** — load `mapping/x86-avx2/rules.llkmap` and assert rules exist for `micro.tile_async_copy` and `micro.tile_store`, each with a declared emitter key the target accepts.
- [ ] **Step 2: Verify red**.
- [ ] **Step 3: Implement** — add the two rules (bundle + `emit` keys added to the target's declared key list and to `verifyMappingTarget`'s expectations). Keep them target-shaped but target-neutral in *form* (the ids live in the `.llkmap`, not in generic code).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `feat(target): map tile copy and tile store on AVX2`.

---

## Task 6: Map the vector variants and prove compiler-generated matmul/SwiGLU end to end

The vector rule matches only `op = "add"`; the lowering emits `convert`, `silu`, and `mul` (and `micro.mma` for the GEMM).

**Files:** Modify: `mapping/x86-avx2/rules.llkmap`, the target's emitter keys; Create: `test/Conversion/MicroMapping/matmul_e2e.mlir` and `swiglu_e2e.mlir` (FileCheck, registered); Test: `test/Mapping/avx2_target.cpp`.

- [ ] **Step 1: Failing FileCheck tests** — run an existing `llk-to-micro` matmul fixture through `--micro-map … mode=exact` and assert a bound kernel (no `no_matching_rule`); same for SwiGLU. Start from `test/Conversion/LLKToMicro/*.mlir`.
- [ ] **Step 2: Verify red** — both fail today with `no_matching_rule` for the ops named in the review.
- [ ] **Step 3: Implement** — add the vector-op rules (a general rule plus specializations where the lowering distinguishes them) and any missing emitter keys. Do not weaken matching: if an op genuinely has no legal implementation yet, say so rather than adding a permissive rule.
- [ ] **Step 4: Run** `ctest -R MicroMapping` → PASS.
- [ ] **Step 5: Commit** — `feat(target): map the vector variants and prove matmul/SwiGLU end to end`.

---

## Task 7: A report-only mode for `--micro-map`

§21 requires "emitting a plan report without modifying input IR".

**Files:** Modify: `lib/Conversion/MicroMapping/MicroMapPass.cpp`, `MicroMappingCommon.h`; Test: `test/Conversion/MicroMapping/`.

- [ ] Steps: failing test that `report-only=1` writes the report and leaves the IR byte-identical to the input; implement by skipping `bindPlanOntoModule` while still writing the report; verify; commit `feat(mapping): add a report-only mode to --micro-map`.

---

## Task 8: A `--micro-verify-mapping` pass

§18.3 phase 2 / §21 name `verify-micro-mapping --machine=<profile>`: resolve IDs, rules, layouts, and routes in already-mapped Micro-IR.

**Files:** Create: `lib/Conversion/MicroMapping/MicroVerifyMappingPass.cpp`, register in `MicroMappingPasses.h` + `llk-opt.cpp`; Test: `test/Conversion/MicroMapping/verify_mapping.mlir` (FileCheck) plus a negative case.

- [ ] Steps: failing test (a mapped kernel verifies; a kernel whose `micro.plan` names an unknown rule/layout/emitter fails with a stable diagnostic); implement using `verifyMappedMicroIR` + the target's registries; verify; commit `feat(mapping): add --micro-verify-mapping`.

---

## Task 9: An automated `--micro-bind-plan` integration test

The pass is a public entry point with no automated coverage.

**Files:** Create: `test/Conversion/MicroMapping/bind_plan.sh` (or a GTest driving both passes), registered in `CMakeLists.txt`.

- [ ] Steps: run `--micro-map report=… mode=beam` to get a plan id, then `--micro-bind-plan plan-id=<that id> mode=beam` and assert the same plan is bound and the IR matches the first pass's output; verify it fails without Task 3/4's fixes; commit `test(mapping): cover the report → bind round-trip`.

---

## Self-review

- **Coverage:** Task 1–2 = #1; Task 3–4 = #2; Task 5–6 = #3; Task 7–9 = #8 (partial). The deferred items are listed above with reasons.
- **Sequencing:** T1 before T2 (the bound must exist before it is compared); T3 before T4 and T9 (parsing before reproduction); T5 before T6 (movement before compute variants); T4 before T9.
- **Types:** `Cost` flows from `boundCost` into `Partial::lowerBound` (T1) and is consumed by the comparators in T2; the `mode` option added by T4 uses the same spelling as `--micro-map` (T7 reuses it).
- **No placeholders** beyond the flagged investigation in T6 (which op variants the lowering actually emits — resolve by grepping the fixtures, not by guessing).
