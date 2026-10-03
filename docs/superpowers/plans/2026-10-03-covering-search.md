# Covering Search Implementation Plan (D6 / epic #67)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select complete, ranked `CoveringPlan`s from D5's placed instances and connection alternatives — with three modes behind one interface (deterministic, beam, exact), every cap disclosed, and a structured failure frontier when no plan exists (design §16).

**Architecture:** `LLK/Mapping/CoveringSearch.h` + `lib/Mapping/CoveringSearch.cpp`. It consumes the `WorkloadGraph` (D1), the `MappingTarget` (D4), placement and connection synthesis (D5), and produces D1 `CoveringPlan`s for the plan binder (#50 revision / D7).

**Spec:** design §16.1 stages, §16.2 modes and options, §16.3 partial-plan state, §16.4 exact-cover semantics, §16.5 failure reporting; plan §4 D6.

## Global Constraints

- One interface, three modes: `deterministic` (first legal plan in canonical order), `beam` (bounded best-first, the default), `exact` (branch-and-bound).
- Every cap is an explicit option and is **reported**: reaching one sets `searchTruncated`; optimality is never implied after truncation (design §16.2).
- The beam is ordered by lower bound, then covered-node count descending, then stable partial-plan id (design §16.3).
- Exact mode branches on the lowest-id uncovered node and tries candidates in stable order (design §16.4); its bound is admissible, so pruning cannot discard the optimum.
- A complete plan covers every workload node **exactly once** and connects every dataflow edge.
- Cost is the declared lower bound plus synthesized connection cost — #46's evaluator replaces that later (design §16.2's interface dependency).

## File Structure

| File | Responsibility |
|---|---|
| `include/LLK/Mapping/CoveringSearch.h` | modes, options, result, failure frontier, `CoveringSearch` |
| `lib/Mapping/CoveringSearch.cpp` | candidate/instance tables, workload edges, the three searches |
| `test/Mapping/covering_search.cpp` | mode agreement, caps, truncation, failure frontier |

---

### Task 1: Search types, candidate tables, and workload edges

**Interfaces:**
```cpp
enum class SearchMode { Deterministic, Beam, Exact };

struct MappingSearchOptions {
  SearchMode mode = SearchMode::Beam;
  unsigned beamWidth = 64;
  unsigned topK = 8;
  unsigned maxCandidatesPerNode = 64;
  unsigned maxInstancesPerCandidate = 64;
  unsigned maxRoutesPerConnection = 8;
  uint64_t memoryBudgetBytes = 512ULL << 20;
  bool enableSymmetryReduction = true;
};

struct FailureFrontier {
  uint64_t nodesWithoutRules = 0;
  uint64_t candidatesWithoutPlacement = 0;
  uint64_t incompatibleInstancePairs = 0;
  uint64_t plansRejectedByCapacity = 0;
  std::vector<std::string> messages;
};

struct MappingSearchResult {
  std::vector<CoveringPlan> plans;   // ranked best-first, at most topK
  bool searchTruncated = false;
  FailureFrontier frontier;
  uint64_t expandedStates = 0;
};

class CoveringSearch {
public:
  CoveringSearch(const WorkloadGraph &workload, const MappingTarget &target,
                 mlir::MLIRContext &context, const LayoutContext &layoutContext,
                 const MappingSearchOptions &options = {});
  llvm::Expected<MappingSearchResult> search();
};
```

**Tables:** for each workload node in ascending id, the rule matches (`matchRules` → `toMappingCandidate`) capped by `maxCandidatesPerNode`, then each candidate's legal placements capped by `maxInstancesPerCandidate`. **Edges:** an input value produced by another node's output becomes a dataflow edge; values with no producing node are external and have no edge.

- [ ] **Step 1: Write failing tests** — a two-node graph yields the expected candidates and instances; a node with no matching rule lands in `frontier.nodesWithoutRules`; a candidate with no placement lands in `candidatesWithoutPlacement`.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(mapping): build covering-search tables and workload edges (D6)`

---

### Task 2: Deterministic mode

**Behaviour:** walk nodes in ascending id, instances in canonical order, and complete the first assignment whose every edge has a legal connection. Connections are synthesized lazily — only when both endpoints are chosen — and the **cheapest** alternative is taken. The result carries one plan.

- [ ] **Step 1: Write failing tests** — a satisfiable two-node graph returns exactly one plan covering both nodes with one connection; an unsatisfiable one returns none and records the frontier.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(mapping): add deterministic covering search (D6)`

---

### Task 3: Beam and exact modes

**Beam:** keep at most `beamWidth` partial plans, ordered by (lower bound, covered count descending, partial-plan id); expand the best by adding an instance for the lowest uncovered node; stop at `topK` complete plans. Hit the width cap → `searchTruncated`.

**Exact:** branch-and-bound on the lowest-id uncovered node, candidates in stable order, pruned by an admissible bound (accumulated cost plus, for each uncovered node, its cheapest instance's cost). Returns the true best plan(s).

- [ ] **Step 1: Write failing tests** — on a bounded fixture a sufficiently wide beam and exact mode select the **same best plan** (the merge criterion); a narrow beam with a cap reports `searchTruncated`; exact mode on the same fixture does not; plans are ranked by total cost and capped at `topK`.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(mapping): add beam and exact covering search (D6)`

---

### Task 4: Failure frontier, capacity rejection, and verification

**Behaviour:** resource accounting accumulates `executorSlots` and `memoryBytes`; a partial plan exceeding `memoryBudgetBytes` is rejected and counted in `frontier.plansRejectedByCapacity`. When no plan exists the frontier carries counts for every category and a `messages` list naming the first failures in deterministic order.

- [ ] **Step 1: Write failing tests** — a graph whose only candidate exceeds the memory budget reports a capacity rejection and no plans; a node with no rules is reported; determinism (two runs, identical ids); every cap sets `searchTruncated`.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Full build and suite** — record registered/passed/skipped/failed.
- [ ] **Step 6: Commit** `feat(mapping): report covering-search failures and caps (D6)`

---

## Self-Review

**Design §16 coverage:** §16.2's three modes and options → Tasks 2–3; §16.3's partial-plan state and beam ordering → Task 3; §16.4's exact-cover branching → Task 3; §16.5's five frontier categories → Tasks 1 and 4; truncation on every cap → Tasks 3 and 4. ✅

**Merge criterion:** "exact and sufficiently wide beam select the same best plan on bounded fixtures; every cap produces explicit `search_truncated`" is Task 3's first and second tests. ✅

**Non-scope:** binding the plan to Micro-IR (D7 / #50 revision), latency calibration (#52), and the `micro-perf` evaluator (#46) — D6 ranks by declared lower bounds plus connection cost.

**Type consistency:** `CoveringPlan`, `CandidateInstance`, `ConnectionPlan`, `Cost`, `PlanDiagnostics` are D1 types used unchanged; `MappingTarget` is D4's; `enumeratePlacements`/`synthesizeConnections` are D5's.

## Verification Results (2026-10-03)

- **Build:** `ninja -C build` — clean.
- **New tests:** `MappingCoveringSearchTest` **8/8**.
- **Full suite:** `ctest --test-dir build --output-on-failure` — 100 registered, **98 passed, 2 skipped, 0 failed**.
- **Deprecated-API audit:** clean.

### Decisions taken while implementing

1. **The three modes share one table of legal instances.** Candidates and placements are enumerated once, up front; the modes differ only in traversal. That is what makes the beam/exact agreement test meaningful rather than a coincidence.
2. **Connections are synthesized lazily.** Extending a partial plan immediately tests every edge whose other end is already chosen, so a branch dies at the first incompatible pair rather than at completion.
3. **The bound is admissible**: accumulated cost plus, per uncovered node, the cheapest available instance. Connection costs are non-negative, so pruning cannot discard the optimum — which is why a wide beam and exact mode agree.
4. **Capacity accounting counts a bound memory as holding the tile.** Rule `memory` requirements carry no byte count yet, so an instance with no requirement would never trip a budget; accounting `kAssumedValueBytes` per bound memory makes the budget meaningful. The real size arrives with the plan binder.
5. **`topK` pruning is disclosed.** Exact mode prunes against the kth-best cost once K plans exist; the prune is sound (it cannot drop a plan we would keep) but the space was not exhausted, so `searchTruncated` is set. Deterministic mode's early stop is *not* truncation — returning the first legal plan is its definition.
6. **A connection's byte count is an assumption** (`kAssumedValueBytes`), because a `WorkloadPort` does not yet carry the value's size. Every plan is costed the same way, so ranking is unaffected; the plan binder replaces it.
