# Micro-IR Epic #67 Gap-Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every *actionable* difference between the PR #66 design and the epic-#67 implementation, on `main`, after the unlanded #93–#97 stack merges.

**Architecture:** Pure additions and narrow corrections on top of the existing `LLKMapping` / `LLKMachine` / `LLKTargetMapping` libraries. No new IR stack. Each task is a TDD slice that keeps the legacy AVX2 compile/JIT path untouched and the full `check-llk` suite green.

**Tech Stack:** C++20, CMake ≥ 3.20 + Ninja, LLVM/MLIR 24 (local) / 22 (CI), GoogleTest, FileCheck, LLVM YAML I/O, MLIR affine maps, LLVM `Error`/`Expected`.

**Spec:** `docs/superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md` (cited throughout as §N). Gap source: the epic-#67 conformance review dated 2026-10-03 (eight workstream reviews plus independent verification).

## Global Constraints

- **Base is `main` *after* PR #93 merges.** #93 carries the entire #93–#97 stack (perf-v2 migration, route-hop events, `LatencyProvider`, the #53 workflow, per-hop emission, and the connection-direction fix). Do **not** start Phase 3+ until `git log origin/main --oneline | grep -q 'run LLKPerf on MachineModel v2'`.
- Follow `.claude/rules/preferred-mlir-api.md` exactly: `Op::create(builder, loc, ...)`, `mlir::cast<T>(x)`, `isa<T>(x)` — the build is `-Werror=deprecated-declarations`.
- Follow `.claude/rules/llvm-version-compatibility.md`: gate any API that differs between LLVM 22 (CI) and 24 (local) with `LLVM_VERSION_MAJOR`.
- Tests are **hand-registered** in `CMakeLists.txt`. A new `test/Mapping/*.cpp` is dead unless added via `add_llk_mapping_test(<Name> <path>)`.
- `LLKMapping` uses `-fno-rtti -fno-exceptions` and links `LLVMSupport` + `LLKMachine` (+ `MLIRIR`, `MLIRAsmParser`). New libraries follow the same pattern (see memory: plain libraries need LLVM compile flags).
- Determinism is a requirement, not a nicety (§3.3, §22.1): never iterate an unordered container as output order; sort before hashing or reporting.
- Do not add target words (`avx2`, `nvidia`, `warp`, `ttgir`, …) to generic `micro` ODS. Do not add a second canonical IR.
- Every task ends with `ninja` clean and `ctest -R <touched tests>` green before committing.

---

## Preconditions (blocking gate)

### Task 0: Establish the post-#93 baseline

**Files:** none (gate).

- [ ] **Step 1: Confirm #93 landed on `main`**

Run: `git fetch origin && git log origin/main --oneline -5`
Expected: `Merge pull request #93` present, and these files exist:

```bash
for f in include/LLK/Mapping/LatencyProvider.h include/LLK/Mapping/CostEvent.h \
         docs/design/micro-ir-mapping-workflow.md test/Mapping/e2e_workflow.cpp; do
  git cat-file -e origin/main:$f && echo "OK $f" || echo "MISSING $f"
done
```

- [ ] **Step 2: Rebase this worktree onto the new `main`**

```bash
git fetch origin
git rebase origin/main
```

Expected: clean, or only this plan file as a conflict (it is new, so no conflict).

- [ ] **Step 3: Configure, build, and record the green baseline**

```bash
mkdir -p build && cd build
cmake .. -G Ninja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja
ctest --output-on-failure 2>&1 | tail -20
```

Expected: 0 failures. Record the passing count in the commit that follows Task 1. **If the baseline is not green, stop and report** — do not start Phase 1 on a dirty baseline.

---

## Gap → task traceability

| Gap (review finding) | Design | Task |
|---|---|---|
| ODS conformance test is a false pass; `warp`-class owners in generic Micro | §19.3, §25.3, §29.11 | T1 (test); **logged** (owner removal) |
| `maxInstances` / `maxRoutes` caps don't set `searchTruncated` | §16.2 | T2 |
| `enableLatencyCache` option absent | §16.2 | T3 |
| `sourceBindingHash` dead; plans collide across bindings | §8.3, §9.5 | T4 |
| `globalParameters` never populated | §9.5 | T5 |
| No mapping pass/tool entry point | §21 | T6 |
| §10.2 direct-compatibility unimplemented; `accessMap` never set | §10.1, §10.2 | T7 |
| Capacity is one global byte budget, no live-range | §9.3 | T8 |
| Ranking hard-coded to `latencyCycles`, objective ignored | §17.1 | T9 |
| 4 of 6 `Cost` dimensions never assigned | §17.2 | T10 |
| Route legality misses layout + transaction size; no liveness | §12.2 | T11 |
| Perf attributes hop cost by memory *kind*, first match | §12.4, §23.3 | T12 |
| Layout-transform-only connections silently dropped | §18.2 | T13 |
| No dtype/shape/affine rule predicates | §14.1 | T14 |
| `require <expr>` never evaluated; `resolvedParameters` empty | §14.1 | T15 |
| `TargetBundle` struct dead (bundles are strings) | §14.3 | T16 |
| Port validation incomplete; unknown capability query not load-time | §14.4 | T17 |
| Canonical ordering half-applied | §22.1 | T18 |
| No diagnostic codes; frontier free-text | §22.3 | T19 |
| No versioned plan report | §22.2 | T20 |
| `MappingTarget::createEmitter` absent | §19 | T21 |
| MachineModel minor version discarded; kinds hardcoded as strings | §11.1, §11.6 | T22 |
| MachineModel missing executor scheduling / memory txn gran. / occupancy / link directionality class | §11.3 | T23 |
| No `LayoutSolver` interface; no finite quantification | §13.3 | T24 |
| No LLKMap printer (round-trip untestable) | §25.3 | T25 |
| Placement first-match; solved layout discarded | §15.1 | T26 |
| Five connection alternatives not attempted in order | §15.2 | T27 |
| Fan-out/fan-in test-only, no cost | §15.3 | T28 |
| Symmetry heuristic, not target-declared; ignores concurrency | §15.1 | T29 |
| No fuzz/property tests | §25.6 | T30 |
| No report-determinism or cross-memory direction test | §25.4, §25.5 | T31 |

**Logged, decision-gated (not tasks in this plan):** removal of `warp`/`wave`/`subgroup`/`pe_group`/`pe`/`lane` from the generic `Owner` enum (needs a dialect-ownership decision, since #43 predates #67); layout-transform *emission* (needs a generic Micro property carrying a target layout id, which §13.4 keeps out of `#micro.layout`); `tile_alloc` emission (the binder's `async_copy`-result-as-storage argument must be reconciled with the tile model); `Perf::Candidate` vs `MappingCandidate` unification (§26.3 — needs a compatibility-wrapper decision); #51/#52 measurement & calibration (blocked on the deferred Micro emulator, #54).

---

## Phase 1 — Truthfulness and guardrails

### Task 1: Make the ODS-purity conformance test honest

The current test (`test/Mapping/generic_accelerator_target.cpp`, `TargetVocabularyStaysOutOfGenericMicroOds`) scans only three `.td` files with a word list that omits `warp`, and never scans the hand-written `MicroEnums.h`. It therefore passes while generic Micro already carries `warp`/`wave`/`subgroup`/`lane`/`pe_group`/`pe`. This task makes the test scan **every** Micro dialect source (generated `.td` **and** hand-written `.h`) and pins the currently-present leaks in an explicit, shrink-only allowlist, so a *new* leak fails immediately and removing a known leak also fails until the list is updated.

**Files:**
- Modify: `test/Mapping/generic_accelerator_target.cpp` (the `TargetVocabularyStaysOutOfGenericMicroOds` test, ~lines 117-140)
- Reference: `include/LLK/Dialect/Micro/MicroDialect.td:118-123`, `include/LLK/Dialect/Micro/MicroEnums.h:209-267`

**Interfaces:**
- Consumes: nothing new.
- Produces: a test-local `kKnownGenericLeaks` contract (documentation of tracked debt).

- [ ] **Step 1: Write the failing test**

Replace the body of `TargetVocabularyStaysOutOfGenericMicroOds` with a scan that also allows an explicit known-leak set, and assert set equality:

```cpp
TEST(GenericAcceleratorTarget, TargetVocabularyStaysOutOfGenericMicroOds) {
  // Generated ODS *and* hand-written headers: MicroEnums.h carries the Owner
  // enumerants and is exactly where a target word can hide (CLAUDE.md warns it
  // is hand-written and not tablegen'd).
  const std::vector<std::string> files = {
      "/include/LLK/Dialect/Micro/MicroDialect.td",
      "/include/LLK/Dialect/Micro/MicroTypes.td",
      "/include/LLK/Dialect/Micro/MicroOps.td",
      "/include/LLK/Dialect/Micro/MicroDialect.h",
      "/include/LLK/Dialect/Micro/MicroEnums.h",
  };
  // Target-family words that must never name a generic Micro concept.
  const std::vector<std::string> targetWords = {
      "avx2", "nvidia", "ampere", "sm80", "ttgir", "npu", "accel",
      "mxu", "vpu", "warp", "wave", "subgroup", "lane", "pe_group",
  };
  // Debt, not policy: generic Micro already ships warp-class owner names
  // (Owner::warp, wave, subgroup, pe_group, pe, lane). Tracked so the debt is
  // visible; a NEW leak fails, and removing one fails until this set shrinks.
  const std::set<std::string> kKnownGenericLeaks = {
      "warp", "wave", "subgroup", "pe_group", "lane",
  };

  std::set<std::string> found;
  for (const std::string &file : files) {
    std::string path = std::string(LLK_SOURCE_DIR) + file;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(path);
    ASSERT_TRUE(static_cast<bool>(buffer)) << path;
    std::string lowered = buffer.get()->getBuffer().lower();
    for (const std::string &word : targetWords)
      if (mentionsWord(lowered, word))
        found.insert(word);
  }
  EXPECT_EQ(found, kKnownGenericLeaks)
      << "generic Micro leaks changed; update kKnownGenericLeaks only when the "
         "dialect genuinely changed";
}
```

Add `#include <set>` and `#include <string>`/`<vector>` if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `ninja -C build MappingGenericAcceleratorTargetTest && ./build/MappingGenericAcceleratorTargetTest --gtest_filter='*TargetVocabularyStaysOutOfGenericMicroOds*'`
Expected: **FAIL** — the first run reports the leak set the old test could not see (proving the old test was blind). If it passes immediately, the scan is still not reaching `MicroEnums.h`; fix the path list.

- [ ] **Step 3: Confirm the leak set is exactly the debt**

Temporarily assert `EXPECT_FALSE(found.empty())` to confirm non-empty, run, observe the printed set, then remove the temporary assert. The observed set must equal `kKnownGenericLeaks`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/MappingGenericAcceleratorTargetTest --gtest_filter='*TargetVocabularyStaysOutOfGenericMicroOds*'`
Expected: PASS, with the known-leak debt now visible in the test source.

- [ ] **Step 5: Commit**

```bash
git add test/Mapping/generic_accelerator_target.cpp
git commit -m "test(mapping): scan hand-written Micro headers and pin known ODS leaks"
```

---

### Task 2: Report truncation for the instance and route caps

§16.2: "Reaching a cap shall produce a `search_truncated` diagnostic rather than silently claiming optimality." Two caps violate this today: `Placement.cpp` `break`s at `maxInstances`, and `Routing.cpp` clamps to `maxRoutes`.

**Files:**
- Modify: `include/LLK/Mapping/Placement.h` (`enumeratePlacements`, `synthesizeConnections` signatures)
- Modify: `lib/Mapping/Placement.cpp:168-175, 179-247`
- Modify: `include/LLK/Mapping/Routing.h:87-88` (`enumerateRoutes` signature)
- Modify: `lib/Mapping/Routing.cpp:134-136`
- Modify: `lib/Mapping/CoveringSearch.cpp` (thread the flag into `result.searchTruncated`)
- Test: `test/Mapping/covering_search.cpp`

**Interfaces:**
- Produces: `bool *truncated = nullptr` out-parameters on `enumeratePlacements`, `synthesizeConnections`, and `TopologyService::enumerateRoutes`.

- [ ] **Step 1: Write the failing test**

Add to `test/Mapping/covering_search.cpp`:

```cpp
TEST(CoveringSearch, ReportsTruncationWhenInstanceCapIsHit) {
  // A graph whose single node has more legal placements than maxInstances.
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  options.maxInstancesPerCandidate = 1;   // force the cap
  CoveringSearch search(/*graph*/ fixture.graph, *fixture.target,
                        fixture.context, fixture.layoutContext, options);
  auto result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->searchTruncated);
}
```

(Reuse the existing fixture helper in this file — the same one used by the capacity test at `covering_search.cpp:259`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `ninja -C build MappingCoveringSearchTest && ./build/MappingCoveringSearchTest --gtest_filter='*ReportsTruncationWhenInstanceCapIsHit*'`
Expected: FAIL — `searchTruncated` is false because the cap is silent.

- [ ] **Step 3: Thread the flag**

In `Placement.h`, change the declarations:

```cpp
llvm::Expected<std::vector<CandidateInstance>>
enumeratePlacements(const MappingCandidate &candidate,
                    const MappingTarget &target, mlir::MLIRContext &context,
                    const LayoutContext &layoutContext,
                    const PlacementOptions &options = {},
                    bool *truncated = nullptr);
```

In `lib/Mapping/Placement.cpp`, at the `break`:

```cpp
    if (instances.size() >= options.maxInstances) {
      if (truncated)
        *truncated = true;
      break;
    }
```

In `Routing.h`, add `bool *truncated = nullptr` to `enumerateRoutes`; in `Routing.cpp:134-136`:

```cpp
  unsigned effectiveLimit = std::min(limit, options_.maxRoutes);
  if (effectiveLimit != limit && truncated)
    *truncated = true;
```

In `CoveringSearch.cpp`, at each `enumeratePlacements(...)` call pass `&result.searchTruncated`, and where a `TopologyService` is used, propagate the flag (set `result.searchTruncated = true` when a route enumeration reports truncation). Ensure the pre-existing `result.searchTruncated` assignments at `CoveringSearch.cpp:149,327,367,397` remain.

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/MappingCoveringSearchTest`
Expected: PASS, including the pre-existing truncation tests (`covering_search.cpp:203-240`).

- [ ] **Step 5: Commit**

```bash
git add include/LLK/Mapping/Placement.h include/LLK/Mapping/Routing.h \
        lib/Mapping/Placement.cpp lib/Mapping/Routing.cpp \
        lib/Mapping/CoveringSearch.cpp test/Mapping/covering_search.cpp
git commit -m "fix(mapping): report search_truncated when instance/route caps are hit"
```

---

### Task 3: Add and honor `enableLatencyCache`

§16.2 lists `bool enableLatencyCache = true` in `MappingSearchOptions`; it exists only in the spec. The stack (#95) consults the `LatencyProvider` unconditionally.

**Files:**
- Modify: `include/LLK/Mapping/CoveringSearch.h:53-62`
- Modify: `lib/Mapping/CoveringSearch.cpp` (the latency lookup added by #95)
- Test: `test/Mapping/covering_search.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
TEST(CoveringSearch, LatencyCacheCanBeDisabled) {
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  options.enableLatencyCache = false;
  // ... same fixture as the existing LatencyProvider test at
  // covering_search.cpp:309, whose provider returns a distinctive cycle count.
  // With the cache disabled the provider must never be consulted.
  EXPECT_EQ(fixture.provider->lookupCount, 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ninja -C build MappingCoveringSearchTest && ./build/MappingCoveringSearchTest --gtest_filter='*LatencyCacheCanBeDisabled*'`
Expected: FAIL to compile — `MappingSearchOptions` has no `enableLatencyCache`.

- [ ] **Step 3: Add the option and the guard**

In `CoveringSearch.h`:

```cpp
  uint64_t memoryBudgetBytes = 512ULL << 20;
  bool enableLatencyCache = true;
  bool enableSymmetryReduction = true;
```

In `CoveringSearch.cpp`, wrap the provider lookup:

```cpp
  if (options_.enableLatencyCache)
    if (const LatencyProvider *provider = target_.latencyProvider())
      if (auto measured = provider->lookupCycles(signature, context))
        applyMeasurement(entry, *measured);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/MappingCoveringSearchTest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/LLK/Mapping/CoveringSearch.h lib/Mapping/CoveringSearch.cpp \
        test/Mapping/covering_search.cpp
git commit -m "feat(mapping): add enableLatencyCache option (design 16.2)"
```

---

## Phase 2 — Traceability and the compiler entry point

### Task 4: Record the source binding hash on every plan

§8.3/§9.5: "Every later plan records the source binding hash." Today `CoveringPlan::sourceBindingHash` is never assigned, so `computePlanId` folds `hexId(0)` and two plans that differ only by their search point collide.

**Files:**
- Modify: `include/LLK/Mapping/CoveringSearch.h` (constructor)
- Modify: `lib/Mapping/CoveringSearch.cpp` (set the field on every emitted plan)
- Modify: `lib/Mapping/PlanBinder.cpp` if it constructs plans without the search
- Test: `test/Mapping/covering_search.cpp`

**Interfaces:**
- Produces: `CoveringSearch(..., std::optional<SearchBinding> binding = std::nullopt)`; `CoveringPlan::sourceBindingHash` populated from `binding->stableHash`; `CoveringPlan::globalParameters` populated from `binding->values`.

- [ ] **Step 1: Write the failing test**

```cpp
TEST(CoveringSearch, PlansCarryTheSourceBindingHash) {
  SearchBinding binding = makeSearchBinding(
      "candidate_17", {{"BM", int64_t{64}}, {"tile_layout", std::string("blocked")}});
  CoveringSearch search(fixture.graph, *fixture.target, fixture.context,
                        fixture.layoutContext, options, binding);
  auto result = search.search();
  ASSERT_TRUE(static_cast<bool>(result));
  ASSERT_FALSE(result->plans.empty());
  EXPECT_EQ(result->plans[0].sourceBindingHash, binding.stableHash);
  EXPECT_NE(result->plans[0].id, 0u);
}

TEST(CoveringSearch, DifferentBindingsYieldDifferentPlanIds) {
  // Build the same graph/plan for binding A and binding B; assert ids differ.
  EXPECT_NE(planIdFor(bindingA), planIdFor(bindingB));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ninja -C build MappingCoveringSearchTest && ./build/MappingCoveringSearchTest --gtest_filter='*SourceBindingHash*'`
Expected: FAIL — hash is 0 / signature mismatch.

- [ ] **Step 3: Implement**

In `CoveringSearch.h` add the parameter and a member:

```cpp
  CoveringSearch(const WorkloadGraph &workload, const MappingTarget &target,
                 mlir::MLIRContext &context, const LayoutContext &layoutContext,
                 const MappingSearchOptions &options = {},
                 std::optional<SearchBinding> binding = std::nullopt);
  ...
  std::optional<SearchBinding> binding_;
```

In `CoveringSearch.cpp`, when a plan is finalized and before `plan.id = computePlanId(plan)`:

```cpp
  if (binding_) {
    plan.sourceBindingHash = binding_->stableHash;
    plan.globalParameters = binding_->values;
  }
  plan.id = computePlanId(plan);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./build/MappingCoveringSearchTest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/LLK/Mapping/CoveringSearch.h lib/Mapping/CoveringSearch.cpp \
        test/Mapping/covering_search.cpp
git commit -m "feat(mapping): thread SearchBinding into covering search and record its hash"
```

---

### Task 5: Populate `globalParameters`

Folded into Task 4 (same field, same source). If Task 4 is done, add the explicit assertion below and mark this complete; otherwise split.

- [ ] **Step 1: Add the assertion to the Task 4 test**

```cpp
  EXPECT_EQ(result->plans[0].globalParameters.size(), 2u);
  EXPECT_EQ(std::get<std::string>(result->plans[0].globalParameters["tile_layout"]),
            "blocked");
```

- [ ] **Step 2: Run** `./build/MappingCoveringSearchTest` → PASS.
- [ ] **Step 3: Commit** with Task 4 (amend) or separately as `test(mapping): assert plans carry global parameters`.

---

### Task 6: Register a `--micro-map` pass and a `--micro-bind-plan` pass

§21: the mapping subsystem must have composable entry points. Today `LLKMapping` is linked **only** into the test binary (`CMakeLists.txt:749`), so nothing in a real pipeline can reach it.

**Files:**
- Create: `include/LLK/Conversion/MicroMapping/MicroMappingPasses.h`
- Create: `lib/Conversion/MicroMapping/MicroMapPass.cpp`
- Create: `lib/Conversion/MicroMapping/MicroBindPlanPass.cpp`
- Modify: `CMakeLists.txt` (new `LLKMicroMapping` library linked into `llk-opt`)
- Modify: `tools/llk-opt/llk-opt.cpp` (register both passes)
- Test: `test/Conversion/MicroMapping/micro_map.mlir` (FileCheck)

**Interfaces:**
- Produces: `mlir::llk::createMicroMapPass(MicroMapOptions)`, `createMicroBindPlanPass(MicroBindPlanOptions)`.
- `MicroMapOptions { std::string target, mode, unsigned topK; }` parsed from `--micro-map="target=<name> machine=<path> layouts=<path> rules=<path> emitters=<csv> mode=<beam|exact|deterministic> top-k=<n>"`.

- [ ] **Step 1: Write the failing FileCheck test**

`test/Conversion/MicroMapping/micro_map.mlir`:

```mlir
// RUN: llk-opt %s --micro-map="target=x86-avx2 machine=%p/../../../machines/x86-avx2-v2.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../../mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add mode=deterministic" | FileCheck %s
// CHECK: micro.kernel
// CHECK-SAME: micro.plan
```

(Use the elementwise fixture from `test/Mapping/e2e_workflow.cpp` as the kernel body; a minimal `micro.kernel` with one `micro.vector` is enough.)

- [ ] **Step 2: Register the test and run it (fails)**

Add to `CMakeLists.txt`:

```cmake
add_llk_filecheck_test(MicroMappingMap test/Conversion/MicroMapping/micro_map.mlir)
```

Run: `ninja -C build llk-opt check-llk 2>&1 | tail`
Expected: FAIL — `--micro-map` is not a registered option.

- [ ] **Step 3: Implement the pass skeleton**

`lib/Conversion/MicroMapping/MicroMapPass.cpp`:

```cpp
struct MicroMapPass : public PassWrapper<MicroMapPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MicroMapPass)
  MicroMapOptions options;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto target = mapping::loadMappingTarget(
        options.target, options.machinePath, options.layoutPath,
        options.rulePath, options.emitterKeys);
    if (!target) return signalPassFailure();
    // find the first micro.kernel, extract, search, bind.
    Operation *kernel = findMicroKernel(module);
    if (!kernel) return signalPassFailure();
    WorkloadGraphBinding binding;
    auto graph = mapping::extractWorkloadGraph(kernel, &binding);
    if (!graph) return signalPassFailure();
    LayoutContext layoutContext = /* rank/type from the kernel's tiles */;
    MappingSearchOptions searchOptions;
    searchOptions.mode = parseMode(options.mode);
    searchOptions.topK = options.topK;
    CoveringSearch search(*graph, **target, *kernel->getContext(),
                          layoutContext, searchOptions);
    auto result = search.search();
    if (!result || result->plans.empty()) return signalPassFailure();
    if (llvm::Error e = mapping::bindPlan(module, result->plans[0], **target)
                            .moveInto(/*BoundPlan*/)) {
      emitError(kernel->getLoc()) << llvm::toString(std::move(e));
      return signalPassFailure();
    }
  }
};
```

Register in `llk-opt.cpp` alongside the existing `mlir::registerPass` block, and add the library to the `llk-opt` link line.

- [ ] **Step 4: Run the test to verify it passes**

Run: `ninja -C build llk-opt && ./build/llk-opt test/Conversion/MicroMapping/micro_map.mlir --micro-map="..."`
Expected: the mapped kernel prints with a `micro.plan` attribute; FileCheck passes.

- [ ] **Step 5: Commit**

```bash
git add include/LLK/Conversion/MicroMapping lib/Conversion/MicroMapping \
        tools/llk-opt/llk-opt.cpp CMakeLists.txt \
        test/Conversion/MicroMapping/micro_map.mlir
git commit -m "feat(mapping): add --micro-map and --micro-bind-plan passes (design 21)"
```

> **Note for the executor:** the exact `MicroMapOptions` spelling and the kernel-discovery helper are the only unspecified internals; derive them from `test/Mapping/e2e_workflow.cpp` (the reference pipeline) and `lib/Mapping/PlanBinder.cpp`. Keep the pass free of any target-specific logic — it must load the target by path only.

---

## Phase 3 — Legality and cost correctness

### Task 7: Implement §10.2 direct-compatibility

Connections today are decided by memory-id equality and layout-id string comparison only (`Placement.cpp:179-247`); element type, tile shape, memory visibility, and affine relation are never checked. This is the "ad hoc comparison §10.2 forbids".

**Files:**
- Modify: `include/LLK/Mapping/Placement.h` (`ConnectionRequest` gains type/shape/affine fields)
- Modify: `lib/Mapping/Placement.cpp` (`synthesizeConnections`)
- Modify: `lib/Mapping/WorkloadGraph.cpp` (populate `WorkloadPort::accessMap` where a `tile_view`/`tile_partition` supplies one)
- Modify: `lib/Mapping/CoveringSearch.cpp` (pass the port types into `ConnectionRequest`)
- Test: `test/Mapping/connections.cpp`

**Interfaces:**
- Produces: `bool portsDirectCompatible(const ConnectionRequest &)`; `ConnectionRequest` gains `mlir::Type elementType`, `std::optional<AffineMap> producerMap`, `std::optional<AffineMap> consumerMap`.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(Connections, RejectsElementTypeMismatch) {
  ConnectionRequest r = baseRequest();
  r.elementType = f32Type;            // producer is f32, consumer bf16
  // ... set consumerType = bf16
  EXPECT_TRUE(synthesizeConnections(r, machine, topology)->empty());
}

TEST(Connections, AcceptsAffinelyEquivalentMaps) {
  // producerMap = (d0,d1)->(d0,d1); consumerMap = affine_map<(d0,d1)->(d0,d1)>
  EXPECT_FALSE(synthesizeConnections(equivalent, machine, topology)->empty());
}

TEST(Connections, RejectsMemoryVisibilityMismatch) {
  // consumer cannot see the producer's memory -> not a Direct connection.
}
```

- [ ] **Step 2: Run to verify failure**

Run: `ninja -C build MappingConnectionsTest && ./build/MappingConnectionsTest --gtest_filter='*ElementTypeMismatch*:*AffinelyEquivalent*'`
Expected: FAIL (fields don't exist / mismatches accepted).

- [ ] **Step 3: Implement**

```cpp
static bool portsDirectCompatible(const ConnectionRequest &r,
                                  const machine::MachineModel &machine) {
  if (r.producerLayout && r.consumerLayout && *r.producerLayout != *r.consumerLayout)
    return false;                                   // different layouts: needs a transform
  if (!machine.isVisible(r.producerMemory, r.consumerMemory))
    return false;                                   // visibility (design 10.2)
  if (r.producerMap && r.consumerMap &&
      *r.producerMap != *r.consumerMap)              // MLIR AffineMap operator==
    return false;                                   // affine relation
  return true;                                      // element type/shape checked by caller
}
```

Use MLIR's `AffineMap` equality (and `mlir::simplifyAffineMap` before comparing when needed) rather than string comparison.

- [ ] **Step 4: Run to verify pass**

Run: `./build/MappingConnectionsTest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/LLK/Mapping/Placement.h lib/Mapping/Placement.cpp \
        lib/Mapping/WorkloadGraph.cpp lib/Mapping/CoveringSearch.cpp \
        test/Mapping/connections.cpp
git commit -m "feat(mapping): check element type, shape, visibility, affine relation (10.2)"
```

---

### Task 8: Per-memory capacity and live-range accounting

§9.3 requires each memory's `capacityBytes` not be exceeded. Today covering search sums ALL bytes against one global `memoryBudgetBytes`.

**Files:**
- Modify: `lib/Mapping/CoveringSearch.cpp` (`extend` lambda, ~lines 206-216)
- Modify: `include/LLK/Mapping/CoveringSearch.h` (`FailureFrontier` already has `plansRejectedByCapacity`)
- Test: `test/Mapping/covering_search.cpp`

- [ ] **Step 1: Write the failing test** — a fixture whose two instances each fit globally but together over-subscribe one `sram` node; expect no plan + `plansRejectedByCapacity > 0`.
- [ ] **Step 2: Run** `./build/MappingCoveringSearchTest --gtest_filter='*PerMemoryCapacity*'` → FAIL.
- [ ] **Step 3: Implement** — keep per-memory totals (already partially present on the stack as `partial.memoryBytes`) and compare each `entry.second` against the machine node's `capacityBytes` via `machine.findMemory(id)->capacityBytes`; count each rejection in `result.frontier.plansRejectedByCapacity`. Keep `memoryBudgetBytes` as an additional global ceiling.
- [ ] **Step 4: Run** `./build/MappingCoveringSearchTest` → PASS.
- [ ] **Step 5: Commit** `fix(mapping): enforce per-memory capacity, not one global byte budget`.

---

### Task 9: Rank by the declared `micro.objective`

§17.1/§16.2: `micro.objective` supplies the comparison order. `ObjectiveOrder`/`costLess` have no production caller; ranking is hard-coded to `latencyCycles` (`CoveringSearch.cpp:299,315,321,386,429-430`).

**Files:**
- Modify: `include/LLK/Mapping/CoveringSearch.h` (`MappingSearchOptions` gains `ObjectiveOrder objective = {};`)
- Modify: `lib/Mapping/CoveringSearch.cpp` (replace latency comparisons with `costLess(..., options_.objective)`)
- Create: bridge from `micro.objective` to `ObjectiveOrder` — `lib/Mapping/ObjectiveBridge.cpp` or a helper in `WorkloadGraph`/`Perf`
- Test: `test/Mapping/covering_search.cpp`

- [ ] **Step 1: Write the failing test** — same two plans; with `objective.primary = CostMetric::DramBytes` the plan with lower DRAM wins even though its latency is higher.
- [ ] **Step 2: Run** → FAIL (latency still decides).
- [ ] **Step 3: Implement** — add the option defaulting to latency, replace every direct `lhs.cost.latencyCycles < rhs.cost.latencyCycles` with `costLess(lhs.cost, rhs.cost, options_.objective)`, and add `ObjectiveOrder objectiveOrderFromMicro(ObjectiveOp)` reading `direction` + `metric`.
- [ ] **Step 4: Run** `./build/MappingCoveringSearchTest` → PASS.
- [ ] **Step 5: Commit** `feat(mapping): rank plans by the declared micro.objective`.

---

### Task 10: Populate the remaining `Cost` dimensions

§17.2: four of six dimensions (`dramBytes`, `spillBytes`, `computeUtilization`, `transferUtilization`) are never assigned.

**Files:** `lib/Mapping/Placement.cpp` (route/local bytes), `lib/Mapping/CostModel.cpp`, tests.

- [ ] **Step 1:** failing test asserting `localBytes` and `dramBytes` are non-zero for a plan with a DRAM route.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** set `localBytes` from staging bytes, `dramBytes` from any route touching a DRAM-class memory, and utilization from `computeUtilization = computeCycles / (executorCount * syncPeriod)` using the v2 perf facts (`clockHz`, `workerThreads`, `sync`).
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(mapping): populate transfer/spill/utilization cost dimensions`.

---

### Task 11: Complete route legality (layout, transaction size, liveness)

`MemoryNode::supportedLayouts` and `LinkEdge::transactionBytes` exist and are unused by routing; liveness is absent. `RouteRequest` also dropped §12.1's `elementType`.

**Files:** `include/LLK/Mapping/Routing.h`, `lib/Mapping/Routing.cpp:162-190`, tests.

- [ ] **Step 1:** failing tests — (a) a value whose layout a hop memory doesn't support is rejected; (b) a value larger than the link's `transactionBytes` is rejected; (c) a route staging through an already-live memory is rejected when live bytes are supplied.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** add `std::optional<std::string> layoutClass;` and `uint64_t liveBytesOnNode = 0;` (or a small `DenseMap<MemoryNodeId,uint64_t>`) to `RouteRequest`; check `supportedLayouts.contains(layoutClass)`, `bytes <= link.transferBytes`, and `next->capacityBytes - live < bytes`.
- [ ] **Step 4:** run `./build/MappingRoutingTest` → PASS.
- [ ] **Step 5:** commit `fix(mapping): validate route layout, transaction size, and liveness (12.2)`.

---

### Task 12: Attribute route-hop cost by node id, not memory kind

Post-#93, `MicroDAG::loadRoutes` keys routes by `(srcSpace, dstSpace)` and `routeHops` returns the **first** match, discarding node ids and the parsed `value`/`kind`. On the shipped profile (`sram.0`/`sram.1`, `acc.0`/`acc.1`) copies are charged the wrong link.

**Files:** (post-#93) `lib/Perf/MicroDAG.cpp:244-296`, `test/Perf/l1_resource_dag.cpp`.

- [ ] **Step 1:** failing test — two movements between same-kind endpoints (`l2→sram.0`, `l2→sram.1`) must charge their distinct links; assert the two events have different link-derived cycle counts.
- [ ] **Step 2:** run `./build/L1ResourceDAGTest` → FAIL (both charge `l2_to_sram.0`).
- [ ] **Step 3:** store the concrete `(sourceNodeId, destinationNodeId)` pair (and the full hop link sequence) on `PlannedRoute`; key `routeHops` on the node-id pair; match the op's endpoint memory **ids** (available in the binder's `connection.route` and the `micro.routes` attr) rather than kinds.
- [ ] **Step 4:** run → PASS; confirm the existing hand-written hop test (`l1_resource_dag.cpp:468-505`) still passes.
- [ ] **Step 5:** commit `fix(perf): identify routes by node id, not memory kind`.

---

## Phase 4 — Emission honesty

### Task 13: Stop silently dropping layout-transform-only connections

`PlanBinder.cpp:186-187` skips a `LayoutTransform`-only connection and does **not** add it to `unmaterialized`, so a selected transform vanishes with no diagnostic.

**Files:** (post-#93) `lib/Mapping/PlanBinder.cpp`, `include/LLK/Mapping/PlanBinder.h`, `test/Mapping/plan_binder.cpp`.

- [ ] **Step 1:** failing test — a plan with a `LayoutTransform` connection yields a `BoundPlan` whose `unmaterialized` names that connection and value.
- [ ] **Step 2:** run `./build/MappingPlanBinderTest` → FAIL.
- [ ] **Step 3:** in the skip branch, push a structured entry onto `BoundPlan::unmaterialized` (value id + reason `"layout_transform_requires_dialect_op"`).
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `fix(mapping): report unmaterialized layout-transform connections`.

---

## Phase 5 — Rules and bundles

### Task 14: dtype / shape / affine-map rule predicates

`RulePredicate` supports only scalar attribute equality. Add operand-type, result-shape, and affine-map predicates, matched against `WorkloadPort::type`/`accessMap`.

**Files:** `include/LLK/Mapping/MappingRules.h:43-46`, `lib/Mapping/MappingRules.cpp:157-179,362-378`, `mapping/x86-avx2/rules.llkmap`, `docs/design/llkmap-rule-grammar.md`, `test/Mapping/mapping_rules.cpp`.

- [ ] **Step 1:** failing tests — `match micro.vector(element_type = f32)` matches only an f32 operand; a shape predicate `shape[0] == 64` matches only that shape; a mismatched dtype rejects.
- [ ] **Step 2:** run `./build/MappingRulesTest` → FAIL.
- [ ] **Step 3:** extend the predicate grammar (`element_type`, `shape[i]`, `access_map`) and `predicateMatches` to read `node.inputs[i].type` / `node.outputs[i].accessMap`; update the grammar doc.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(rules): match on element type, shape, and affine map (14.1)`.

### Task 15: Evaluate `require <expr>` constraints; populate `resolvedParameters`

`RuleDef.constraints` are parsed and validated but never evaluated; `MappingCandidate::resolvedParameters` is never written.

**Files:** `lib/Mapping/MappingRules.cpp:253-267,399-449`, `test/Mapping/mapping_rules.cpp`.

- [ ] **Step 1:** failing test — a rule with `require VW == machine.compute("vector_engine").lanes(element_type)` produces a candidate only when the machine satisfies it.
- [ ] **Step 2:** run → FAIL (constraint inert).
- [ ] **Step 3:** evaluate each constraint with the existing LLKMap evaluator against the target machine during `toMappingCandidate`; record evaluated derived parameters into `resolvedParameters`.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(rules): evaluate require-expressions and record resolved parameters`.

### Task 16: Wire the `TargetBundle` contract

`TargetBundle{name, parameters, emitterKey}` is declared but dead; bundles are bare strings end-to-end.

**Files:** `include/LLK/Mapping/MappingTarget.h:37-41`, `include/LLK/Mapping/MappingPlan.h:103`, `lib/Mapping/MappingRules.cpp:404`, `lib/Mapping/PlanBinder.cpp:124`, tests.

- [ ] **Step 1:** failing test — a rule that declares bundle parameters reaches `PlanPlacement::bundle` as a `TargetBundle` (name + non-empty params + emitter key).
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** parse rule bundle parameters into a `DictionaryAttr`; carry `TargetBundle` (not `std::string`) through `MappingCandidate`/`PlanPlacement`; generic code only compares/hashes/prints it.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(rules): carry typed target bundles through the plan (14.3)`.

### Task 17: Complete rule validation

Missing: declared-port adequacy, `require layout <port>` naming a declared port, and load-time rejection of an unknown `machine.compute("<kind>")` subject.

**Files:** `lib/Mapping/MappingRules.cpp:270-285,234-251`, `lib/Mapping/MappingTarget.cpp:63-86`, `lib/Mapping/LlkMap.cpp:405-430`, `test/Mapping/mapping_rules.cpp`.

- [ ] **Step 1:** failing tests — a rule with no ports, a layout requirement naming an undeclared port, and a `require` naming an unknown compute kind each fail at load.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** add the three checks to `verifyMappingTarget`; make `validateExpr` resolve `machine.compute(<name>)` against the machine.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `fix(rules): validate ports and capability queries at load time (14.4)`.

---

## Phase 6 — Determinism

### Task 18: Apply §22.1 canonical ordering everywhere

Rules/candidates aren't sorted by id; placements sort by node only; plans sort by hard-coded latency.

**Files:** `lib/Mapping/MappingRules.cpp:324-331,380-397`, `lib/Mapping/CoveringSearch.cpp:435-438,390-394`, tests.

- [ ] **Step 1:** failing tests — (a) `matchRules` returns candidates sorted by `(covered-node sequence, rule id, id)` regardless of registry order; (b) `RuleRegistry` iterates by rule id; (c) placements sort by `(executor, memory, layout)` tuple; (d) plans sort by the Task 9 objective then id.
- [ ] **Step 2:** run `./build/MappingRulesTest ./build/MappingCoveringSearchTest` → FAIL.
- [ ] **Step 3:** add the sorts; keep unordered maps for lookup only.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `refactor(mapping): apply canonical ordering to rules, candidates, placements, plans`.

---

## Phase 7 — Reports and diagnostics

### Task 19: Stable diagnostic codes

None of §22.3's eleven codes exist; `FailureFrontier` is free-text.

**Files:** Create `include/LLK/Mapping/Diagnostics.h`; modify `include/LLK/Mapping/CoveringSearch.h:66-72`, `lib/Mapping/CoveringSearch.cpp`, `test/Mapping/covering_search.cpp`.

- [ ] **Step 1:** failing test — each failure category carries the specified `DiagnosticCode` (`no_matching_rule`, `no_legal_layout`, `no_legal_executor`, `memory_capacity_exceeded`, `unsupported_compute_fragment`, `no_memory_route`, `no_layout_transform`, `global_constraint_failed`, `search_truncated`, `latency_cache_miss`, `target_bundle_invalid`) with `stringifyDiagnosticCode`.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** add the enum + stringify/symbolize; attach a code to each frontier counter and message; emit `latency_cache_miss` on a provider miss.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(mapping): add stable diagnostic codes (22.3)`.

### Task 20: Versioned plan report

No report writer exists.

**Files:** Create `include/LLK/Mapping/PlanReport.h` + `lib/Mapping/PlanReport.cpp`; wire a `--micro-map-report=<path>` option into Task 6's pass; `test/Mapping/plan_report.cpp`.

- [ ] **Step 1:** failing test — `writePlanReport(result, machine, target, options, moduleHash)` emits JSON containing `version`, `inputModuleHash`, `sourceBindingHash`, `machineHash`, `layoutLibraryHash`, `ruleLibraryHash`, search options, truncation flag, candidate/instance/route/plan counts, rejection counts by code, top-K plans with component costs, and `selectedPlanId`; two runs are byte-identical.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement with `llvm::json` (not `raw_ostream` string building), stable key order, fixed-format doubles (`canonicalCostString`).
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(mapping): emit a versioned plan report (22.2)`.

---

## Phase 8 — Target boundary, MachineModel, LLKMap

### Task 21: `MappingTarget::createEmitter`
Add `class TargetEmitter` (opaque: `StringRef key()`, `llvm::Error verify(...)`) and `virtual std::unique_ptr<TargetEmitter> createEmitter() const = 0;` to `MappingTarget`; implement in `FileMappingTarget` and the AVX2/generic targets. Keep `isKnownEmitter` as a load-time fast path. Test: a target returns an emitter whose `verify` rejects an unknown bundle.
Files: `include/LLK/Mapping/MappingTarget.h`, `lib/Mapping/MappingTarget.cpp`, `lib/Target/*/Mapping/*`, `test/Mapping/*_target.cpp`.

### Task 22: MachineModel schema hygiene
Preserve the minor version (`MachineModel.schemaMinor`) and accept unknown **minor** keys with a default (or reject with a version-aware message); validate compute/transfer `kind` against the Micro vocabulary enums rather than hard-coded `"matrix_engine"|"vector_engine"|"dma"` string lists.
Files: `include/LLK/Machine/MachineModel.h:45`, `lib/Machine/MachineModelLoader.cpp:58-74,349-353,558-573`, `lib/Machine/MachineModel.cpp:349-353`, `test/Machine/*`.

### Task 23: MachineModel §11.3 gaps
Add executor scheduling properties, memory transaction granularity, compute occupancy limit, and link directionality class; load them with sane defaults; validate them. Files: `include/LLK/Machine/MachineModel.h:53-123`, `lib/Machine/MachineModelLoader.cpp`, `machines/*-v2.yaml`, `test/Machine/*`.

### Task 24: LLKMap `LayoutSolver` interface + finite quantification
Introduce `class LayoutSolver { virtual llvm::Expected<LayoutSolution> solve(...) const = 0; }` with the existing bounded solver as one implementation; add `forall`/`exists` over declared executors or dimensions to the grammar and evaluator.
Files: `include/LLK/Mapping/LayoutConstraints.h:121-124`, `lib/Mapping/LayoutConstraints.cpp:408-490`, `lib/Mapping/LlkMap.cpp`, `docs/design/llkmap-layout-grammar.md`, `mapping/x86-avx2/layouts.llkmap`, `test/Mapping/layout_constraints.cpp`.

### Task 25: LLKMap printer
Add `std::string printLayout(const LayoutDef &)` and `printRule(const RuleDef &)` producing text that re-parses to an identical object; test parse→print→parse round-trip equality for both shipped `.llkmap` files. Files: `lib/Mapping/LlkMap.cpp`, `include/LLK/Mapping/{LayoutConstraints,MappingRules}.h`, `test/Mapping/*`.

---

## Phase 9 — Placement depth

### Task 26: Enumerate attachments; bind solved layouts
`enumeratePlacements` picks the first matching compute/memory and discards `solveLayout`'s result. Enumerate all compatible (compute, memory) combinations and set `layoutBindings` to the concrete solved `LayoutId` + parameters.
Files: `lib/Mapping/Placement.cpp:122-163`, `test/Mapping/placement.cpp:184`. **Note:** that test currently locks in the degenerate `class=class` binding — update it as part of this task.

### Task 27: Ordered five-alternative connection synthesis
Replace the two-boolean switch with the §15.2 order: direct → layout-only transform in a **mutually visible** memory → direct transfer → transfer+transform → bounded multi-hop with the transform placed at a legal hop. Each alternative becomes its own `ConnectionPlan`.
Files: `lib/Mapping/Placement.cpp:179-247`, `test/Mapping/connections.cpp`.

### Task 28: Integrate fan-out / fan-in with cost
`CoveringSearch` currently synthesizes one connection per edge, so shared reads/replication/gathers are never exploited. Call `synthesizeFanOut`/`synthesizeFanIn` from the search and give them real replication/intermediate cost and capacity.
Files: `lib/Mapping/CoveringSearch.cpp`, `lib/Mapping/Placement.cpp:249-305`, `test/Mapping/connections.cpp`, `test/Mapping/covering_search.cpp`.

### Task 29: Declared symmetry equivalence
Replace the structural `interchangeable` heuristic with a target-declared equivalence (an `equivalent_to:`/`interchangeable:` field on executors), and include `concurrency`/`coordinates` in the default heuristic so distinct-performance executors are not collapsed.
Files: `include/LLK/Machine/MachineModel.h`, `lib/Machine/MachineModelLoader.cpp`, `lib/Mapping/Placement.cpp:50-73`, `test/Mapping/placement.cpp`.

---

## Phase 10 — Tests

### Task 30: Property/fuzz harness
Bounded randomized tests for: routes never repeat a node; complete plans cover every node exactly once; ids independent of insertion order; cost ordering satisfies deterministic tie-breaks. Seed from a fixed constant so failures reproduce.
Files: Create `test/Mapping/properties.cpp` (+ `add_llk_mapping_test(MappingPropertiesTest ...)`).

### Task 31: Report determinism and cross-memory direction regressions
- A test that runs the full pipeline twice and asserts byte-identical report JSON (depends on Task 20).
- Add a cross-memory fixture to `test/Mapping/covering_search.cpp` asserting connection **direction** (producer/consumer), so the #97 class of bug cannot regress undetected at the search layer.
Files: `test/Mapping/plan_report.cpp`, `test/Mapping/covering_search.cpp`.

---

## Self-review against the gap list

- **Spec coverage:** every actionable row in the traceability table maps to a task. The five logged items are explicitly out-of-plan with reasons.
- **Placeholder scan:** the only intentionally underspecified internals are flagged in Task 6's note (pass option spelling, kernel discovery) and Task 22's exact YAML key names — both are derivable from named reference files, not invented behavior.
- **Type consistency:** `bool *truncated` out-params (T2), `std::optional<SearchBinding> binding_` (T4), `ObjectiveOrder objective` (T9), `ConnectionRequest{producerMap, consumerMap, elementType}` (T7), `RouteRequest{layoutClass, liveBytesOnNode}` (T11), and `TargetBundle` (T16) are each defined once and reused consistently.
- **Sequencing:** T4/T5 before T20 (report needs the binding hash); T9 before T18 (canonical plan order depends on the objective); T7 before T27 (synthesis uses the compatibility check); T2 before T19 (truncation code depends on the flag).

## Execution note

Phases 1–3 are the highest-value, lowest-risk slice and are fully specified above. Phases 4–10 are specified at task granularity (files, interfaces, tests, commit); expand each to the full step template immediately before executing it, reading the named files first. Do not batch-execute across a phase boundary without a green `ctest`.
