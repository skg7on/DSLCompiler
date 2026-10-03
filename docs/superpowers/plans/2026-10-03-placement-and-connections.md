# Placement and Connection Synthesis Implementation Plan (D5 / epic #67)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn an unplaced rule match into legal `CandidateInstance`s on concrete machine resources, then synthesize the `ConnectionPlan`s that move values between them — direct, transform, transfer, transfer-plus-transform, multi-hop, and the fan-out/fan-in shapes (design §15).

**Architecture:** `LLK/Mapping/Placement.h` + `lib/Mapping/Placement.cpp`. Placement consumes D1's `MappingCandidate`/`CandidateInstance`/`ConnectionPlan`, D2's `TopologyService`, D3's `LayoutRegistry`, D4's `RuleRegistry`/`MappingTarget`, and `MachineModel`. It is the first workstream that pulls all four together, and the last before covering search (D6), which consumes its output unchanged.

**Spec:** design §15.1 (placement enumeration), §15.2 (connection synthesis), §15.3 (fan-out and fan-in); plan §4 D5.

## Global Constraints

- Placement is deterministic: executors, computes, memories, and layout solutions are enumerated in machine declaration order.
- Requirements name abstract capabilities; a concrete executor id never appears in a rule (design §11.5).
- Every emitted `CandidateInstance` is legal: it exists in the machine, satisfies containment and visibility, and its layouts solve.
- Connection synthesis tries the cheapest shape first and emits *every* alternative it finds; a pair with no alternative is compatible-with-nothing, which must not reject other placements (design §15.2).
- `ConnectionPlan`s carry the route D2 produced, so `micro-perf` observes every hop (design §12.2/§17.2).
- Symmetry reduction is optional and must keep at least one representative (design §15.1).

## Gap this plan closes

D4 landed rule *parsing and validation*; it does not yet turn a rule into a D1 `MappingCandidate`. Design §14.2's one-op matching belongs with the rules, so Task 1 adds it to `MappingRules.h` rather than inventing a second home for it.

## File Structure

| File | Responsibility |
|---|---|
| `include/LLK/Mapping/MappingRules.h` + `lib/Mapping/MappingRules.cpp` | add `matchRules` (one-op matching) and `toMappingCandidate` |
| `include/LLK/Mapping/Placement.h` + `lib/Mapping/Placement.cpp` | placement enumeration; connection synthesis; fan-out/fan-in |
| `mapping/x86-avx2/rules.llkmap` | predicates renamed to the dialect's real attributes |
| `test/Mapping/placement.cpp`, `test/Mapping/connections.cpp` | tests |

---

### Task 1: One-op rule matching to `MappingCandidate`

**Interfaces (added to `MappingRules.h`):**
- `bool predicateMatches(const RulePredicate &, const mlir::DictionaryAttr &attributes);`
- `std::vector<const RuleDef *> matchRules(const WorkloadNode &node, const RuleRegistry &rules);`
- `MappingCandidate toMappingCandidate(const RuleDef &rule, const WorkloadNode &node);`

**Matching:** a rule matches when its `matchOp` equals the node's op name and every predicate holds against the node's `attributes` (an integer attribute equals an integer predicate value; a string attribute equals a string/symbolic value; a missing attribute fails). Rules are returned in registry order.

**Bridging:** ports come from the rule, wired positionally to the node's inputs then outputs; `kindRequirements` split into `executorRequirements` / `memoryRequirements` by role; `layoutRequirements` become `LayoutRequirement{layoutId}`; `costLowerBound` becomes `lowerBound.latencyCycles`.

- [ ] **Step 1: Write failing tests** — a rule matches a node with the attribute it predicates on and does not match one without it; a rule with a different `matchOp` does not match; matching preserves registry order; the produced `MappingCandidate` has the rule id, the covered node, the bundle, the right ports, and the executor/memory/layout requirements.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Update `mapping/x86-avx2/rules.llkmap`** so predicates name the dialect's real attributes (`op = "add"` for `micro.vector`/`micro.reduce`, `input = bf16` for `micro.mma`), and re-run the D4 tests.
- [ ] **Step 5: Commit** `feat(mapping): match rules to workload nodes (D5)`

---

### Task 2: Placement enumeration

**Interfaces (`Placement.h`):**
```cpp
struct PlacementOptions {
  bool reduceSymmetry = true;
  unsigned maxInstances = 64;
};

llvm::Expected<std::vector<CandidateInstance>>
enumeratePlacements(const MappingCandidate &candidate, const MappingTarget &target,
                    mlir::MLIRContext &context, const LayoutContext &layoutContext,
                    const PlacementOptions &options = {});
```

Per design §15.1: resolve each executor requirement against executors the machine offers (owner-kind matching); enumerate compute and memory attachments that satisfy the rule's requirements; solve each layout requirement against the machine; estimate `ResourceUsage`; assign the candidate's lower bound as the local cost. Emit in stable order.

Symmetry reduction keeps one representative per equivalence class of **interchangeable** executors — same kind, same parent, same attached compute and memory *node ids*. That is conservative on purpose: two executors are only collapsed when swapping them cannot change a binding. Disabled by `reduceSymmetry = false`.

- [ ] **Step 1: Write failing tests** — a candidate with an executor requirement places on every matching executor in declaration order; a machine that offers no matching executor yields no placements; a compute requirement filters attachments; a layout requirement that cannot solve yields no placement; `reduceSymmetry` collapses interchangeable executors to one and `false` keeps them all; instances are legal (every binding exists in the machine).
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(mapping): enumerate legal placements (D5)`

---

### Task 3: Connection synthesis

**Interfaces:**
```cpp
struct ConnectionRequest {
  InstanceId producer;
  InstanceId consumer;
  WorkloadValueId value;
  MemoryNodeId producerMemory;
  MemoryNodeId consumerMemory;
  std::optional<LayoutId> producerLayout;
  std::optional<LayoutId> consumerLayout;
  uint64_t bytes;
  uint64_t alignmentBytes = 1;
};

llvm::Expected<std::vector<ConnectionPlan>>
synthesizeConnections(const ConnectionRequest &request,
                      const machine::MachineModel &machine,
                      const TopologyService &topology,
                      const PlacementOptions &options = {});
```

Attempts in the design's order (§15.2): `Direct` (same memory, same layout); `LayoutTransform` (same memory, different layout); `Transfer` (different memory, same layout); `TransferAndTransform` (different memory, different layout). Every route D2 returns becomes one plan, so a multi-hop route appears as a `Transfer` whose `memoryRoute` has more than two nodes. An empty result means the pair is incompatible.

- [ ] **Step 1: Write failing tests** — direct; transform-only; transfer-only; transfer-plus-transform; a two-hop transfer on a chain machine; a pair with no route returns no alternatives without throwing.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** on top of `TopologyService::enumerateRoutes`.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(mapping): synthesize connections and routes (D5)`

---

### Task 4: Fan-out and fan-in

**Interfaces:**
```cpp
llvm::Expected<std::vector<ConnectionPlan>>
synthesizeFanOut(const ConnectionRequest &base, llvm::ArrayRef<InstanceId> consumers,
                 llvm::ArrayRef<MemoryNodeId> consumerMemories,
                 const machine::MachineModel &, const TopologyService &,
                 const PlacementOptions & = {});

llvm::Expected<ConnectionPlan>
synthesizeFanIn(llvm::ArrayRef<InstanceId> producers, InstanceId consumer,
                WorkloadValueId value, MemoryNodeId consumerMemory, uint64_t bytes);
```

`FanOut`: when every consumer can legally read the producer's memory, emit **one** shared-read plan carrying all consumers (§15.3 "shared reads"); otherwise emit one plan per consumer (**replication**). `FanIn`: one `ConnectionKind::Reduce` plan gathering several producers into the consumer.

- [ ] **Step 1: Write failing tests** — two consumers on the producer's memory produce one shared plan with both consumers; a consumer that cannot reach the producer's memory forces replication; a fan-in produces one `Reduce` plan listing every producer.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Full build and suite** — record registered/passed/skipped/failed.
- [ ] **Step 6: Commit** `feat(mapping): synthesize fan-out and fan-in (D5)`

---

## Self-Review

**Design §15 coverage:** §15.1 steps 1–5 and symmetry → Task 2 (step 3's layout solving via D3, step 4's `ResourceUsage`); §15.2's five shapes → Task 3 (the fifth, multi-hop, is D2's route length); §15.3 shared/replicated/gathered → Task 4. ✅

**Merge criterion:** the fixtures named there — direct, transform-only, transfer-only, transfer-plus-transform, two-hop, shared-read, replication — are exactly Tasks 3 and 4's tests. ✅

**Non-scope:** covering search (D6) consumes these outputs; no Micro-IR is materialized (that is the plan binder, #50 revision / D7); no `micro-perf` coupling.

**Type consistency:** `CandidateInstance`/`ConnectionPlan`/`ResourceUsage`/`Cost` are D1 types used unchanged; `MemoryNodeId`/`ExecutorId`/`LayoutId` come from `MappingPlan.h`; `TopologyService` is D2's.
