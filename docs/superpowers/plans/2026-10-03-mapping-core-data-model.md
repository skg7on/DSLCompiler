# Mapping Core Data Model Implementation Plan (issue #80 / epic #67, workstream D1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `LLK/Mapping`, the target-independent mapping data model: stable IDs, `SearchBinding`, `WorkloadGraph` extraction from concrete Micro-IR, the candidate/instance/connection/covering plan structs, a multi-dimensional `Cost`, and canonical hashing/ordering for everything serialized or compared.

**Architecture:** A new static library `LLKMapping` owning `include/LLK/Mapping` + `lib/Mapping`. It depends only on MLIR core (`MLIRIR`, `MLIRSupport`) and `MicroDialect`; it must not depend on `LLK/Perf` or any target backend. Later workstreams (D2 routing, D3 layouts, D4 rules, D5 placement, D6 covering), the revised #50 binder, and #53 consume these types. D1 defines data and determinism only — no search algorithms.

**Tech Stack:** C++20, MLIR/LLVM 24, CMake/Ninja, GoogleTest. FNV-1a 64-bit hashing (stable across toolchains, matching `lib/Perf/Candidate.cpp`).

**Spec:** `docs/superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md` (§8.3, §9, §10, §17.1, §20, §22.1) and `docs/superpowers/plans/2026-09-18-issue-67-implementation-dependency-plan.md` §4 D1.

## Global Constraints

- Do not introduce a second canonical SemanticIR-like dialect.
- `LLK/Mapping` is generic: no `LLK/Perf` dependency, no AVX2/warp/TTGIR/NPU vocabulary in code or headers.
- Stable IDs and ordering everywhere: identical content must yield identical ids regardless of insertion order.
- Store durable search intent in MLIR, transient search mechanics in C++ — D1 only models the transient side plus the `SearchBinding` handoff type.
- Use MLIR `AffineMap` / `DictionaryAttr` / `OperationName` / `Type` for index and attribute representation; do not invent parallel string forms.
- Do not modify `LLK/Perf` in D1; migration of `perf::Candidate` onto `SearchBinding` is deferred to the #50 revision (design §1113). D1 must not create a second *user-facing* candidate concept — `SearchBinding` is the canonical durable binding and `MappingCandidate` is the distinct internal rule match.
- Namespace convention follows the codebase: `mlir::llk::mapping` (existing perf code uses `mlir::llk::perf`; the spec's `llk::mapping` snippets are shorthand).
- `-Werror=deprecated-declarations` is on: use `Op::create(builder, ...)`, `mlir::cast<T>`, `isa<T>` — never `builder.create<T>` or `type.cast<T>()`.
- Every task starts with a failing test (TDD) and ends with a focused commit.

## File Structure

| File | Responsibility |
|---|---|
| `include/LLK/Mapping/StableHash.h` | FNV-1a 64-bit hashing, hex id formatting, order-independent key combining |
| `lib/Mapping/StableHash.cpp` | Implementation of the above |
| `include/LLK/Mapping/CostModel.h` | Multi-dimensional `Cost`, metric access, objective-ordered comparison |
| `lib/Mapping/CostModel.cpp` | Cost arithmetic, canonical string |
| `include/LLK/Mapping/SearchBinding.h` | `SearchValue`, immutable `SearchBinding`, content hash, canonical order |
| `lib/Mapping/SearchBinding.cpp` | Hashing/ordering implementation |
| `include/LLK/Mapping/WorkloadGraph.h` | `WorkloadNodeId`/`WorkloadValueId`, `WorkloadValue`, `WorkloadPort`, `WorkloadNode`, `WorkloadGraph`, extraction API |
| `lib/Mapping/WorkloadGraph.cpp` | `extractWorkloadGraph`, node-op classification, canonical finalize |
| `include/LLK/Mapping/MappingPlan.h` | Plan ids, requirements, `MappingCandidate`, `CandidateInstance`, `ConnectionPlan`, `CoveringPlan`, `PlanDiagnostics`, canonical hashing |
| `lib/Mapping/MappingPlan.cpp` | Id/hash/canonical-string implementation |
| `test/Mapping/stable_hash.cpp` | Hash stability and canonical id tests |
| `test/Mapping/cost_model.cpp` | Cost arithmetic and objective ordering tests |
| `test/Mapping/search_binding.cpp` | Binding hash/order/determinism tests |
| `test/Mapping/workload_graph.cpp` | Extraction from elementwise + tiled GEMM fixtures, determinism |
| `test/Mapping/mapping_plan.cpp` | Plan id determinism, insertion-order independence |

---

### Task 1: `LLKMapping` library skeleton and stable hashing

**Files:**
- Create: `include/LLK/Mapping/StableHash.h`
- Create: `lib/Mapping/StableHash.cpp`
- Create: `test/Mapping/stable_hash.cpp`
- Modify: `CMakeLists.txt` (new library after `LLKPerf`; new `add_llk_mapping_test` function next to `add_llk_perf_test`)

**Interfaces:**
- Consumes: nothing (leaf library).
- Produces:
  - `uint64_t mlir::llk::mapping::stableHash(llvm::StringRef data)`
  - `uint64_t mlir::llk::mapping::stableHashBytes(uint64_t seed, llvm::ArrayRef<char> data)`
  - `uint64_t mlir::llk::mapping::stableHashCombine(uint64_t seed, uint64_t value)`
  - `std::string mlir::llk::mapping::hexId(uint64_t value)` → 16 lowercase hex digits

- [ ] **Step 1: Write the failing test**

```cpp
// test/Mapping/stable_hash.cpp
#include "LLK/Mapping/StableHash.h"
#include <gtest/gtest.h>

using namespace mlir::llk::mapping;

TEST(StableHash, KnownFnv1aVectors) {
  // FNV-1a 64 reference vectors; pins the algorithm so ids stay reproducible.
  EXPECT_EQ(stableHash(""), 0xcbf29ce484222325ULL);
  EXPECT_EQ(stableHash("a"), 0xaf63dc4c8601ec8cULL);
}

TEST(StableHash, OrderIndependentCombine) {
  EXPECT_EQ(stableHashCombine(stableHashCombine(0, 7), 9),
            stableHashCombine(stableHashCombine(0, 9), 7));
}

TEST(StableHash, HexIdIsSixteenLowercaseDigits) {
  std::string id = hexId(0x0abcULL);
  EXPECT_EQ(id, "0000000000000abc");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ninja -C build MappingStableHashTest` (target does not exist yet → configure fails)
Expected: FAIL — target/header missing.

- [ ] **Step 3: Add the library and test registration to `CMakeLists.txt`**

```cmake
# Library: target-independent mapping data model (issue #80 / epic #67 D1).
# Generic planning types only: it must not depend on LLKPerf or any backend.
add_library(LLKMapping STATIC
    lib/Mapping/StableHash.cpp
)
target_include_directories(LLKMapping PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(LLKMapping PUBLIC LLVMSupport)
mlir_target_link_libraries(LLKMapping PRIVATE MLIRIR)
```

```cmake
# Mapping-engine tests (epic #67 D1). They parse micro IR to build workload
# graphs, so they use the same MLIR exec-lib rule as the perf tests: name
# MLIR components only through LLK_MLIR_EXEC_LIBS.
function(add_llk_mapping_test test_name test_file)
  add_llvm_executable(${test_name} ${test_file})
  target_link_libraries(${test_name} PRIVATE
      LLKMapping MicroDialect GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_llk_mapping_test(MappingStableHashTest test/Mapping/stable_hash.cpp)
```

- [ ] **Step 4: Write the header**

```cpp
//===- StableHash.h - Toolchain-stable hashing for mapping ids ------------===//
#ifndef LLK_MAPPING_STABLEHASH_H
#define LLK_MAPPING_STABLEHASH_H
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>
namespace mlir::llk::mapping {
uint64_t stableHash(llvm::StringRef data);
uint64_t stableHashBytes(uint64_t seed, llvm::ArrayRef<char> data);
uint64_t stableHashCombine(uint64_t seed, uint64_t value);
std::string hexId(uint64_t value);
} // namespace mlir::llk::mapping
#endif
```

- [ ] **Step 5: Write the implementation**

FNV-1a: `h = 0xcbf29ce484222325`, for each byte `h = (h ^ b) * 0x100000001b3`. `stableHashCombine(seed, v)` folds the 8 little-endian bytes of `v` into `seed`. **Order-independent combine requirement:** fold the *sorted* multiset of combined values, so a caller combining a set must sort first; expose `stableHashCombine` as a plain fold and provide the ordering guarantee at the call sites that use it. To satisfy the test's `OrderIndependentCombine` assertion, implement `stableHashCombine(a,b)` as `stableHashBytes(stableHashBytes(seed, le(a)+le(b)))` after sorting the two 8-byte words ascending — this makes the two-argument form commutative while still being a deterministic fold.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cmake --build build --target MappingStableHashTest && ./build/MappingStableHashTest`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/LLK/Mapping/StableHash.h lib/Mapping/StableHash.cpp test/Mapping/stable_hash.cpp
git commit -m "feat(mapping): add LLKMapping library and stable hashing (D1)"
```

---

### Task 2: Multi-dimensional `Cost`

**Files:**
- Create: `include/LLK/Mapping/CostModel.h`
- Create: `lib/Mapping/CostModel.cpp`
- Create: `test/Mapping/cost_model.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `StableHash.h` (canonical string only).
- Produces:
  - `struct Cost { double latencyCycles; uint64_t dramBytes, localBytes, spillBytes; double computeUtilization, transferUtilization; }`
  - `enum class CostMetric { LatencyCycles, DramBytes, LocalBytes, SpillBytes, ComputeUtilization, TransferUtilization }`
  - `double costMetric(const Cost &, CostMetric)`
  - `Cost addCost(const Cost &, const Cost &)`
  - `struct ObjectiveOrder { CostMetric primary; std::vector<CostMetric> secondary; bool minimize = true; }`
  - `bool costLess(const Cost &, const Cost &, const ObjectiveOrder &)`
  - `std::string canonicalCostString(const Cost &)`

- [ ] **Step 1: Write the failing test**

```cpp
// test/Mapping/cost_model.cpp
#include "LLK/Mapping/CostModel.h"
#include <gtest/gtest.h>
using namespace mlir::llk::mapping;

TEST(CostModel, AddsDimensionwise) {
  Cost a{.latencyCycles = 10.0, .dramBytes = 100, .localBytes = 4};
  Cost b{.latencyCycles = 2.5, .dramBytes = 50, .localBytes = 1};
  Cost s = addCost(a, b);
  EXPECT_DOUBLE_EQ(s.latencyCycles, 12.5);
  EXPECT_EQ(s.dramBytes, 150u);
  EXPECT_EQ(s.localBytes, 5u);
}

TEST(CostModel, MinimizeOrdersByPrimaryThenSecondary) {
  ObjectiveOrder o{CostMetric::LatencyCycles, {CostMetric::DramBytes}, true};
  Cost fast{M_CYCLE(10), .dramBytes = 999};   // smaller latency wins regardless of bytes
  Cost slow{M_CYCLE(11), .dramBytes = 0};
  EXPECT_TRUE(costLess(fast, slow, o));
  Cost fastFew{M_CYCLE(10), .dramBytes = 1};
  Cost fastMany{M_CYCLE(10), .dramBytes = 2};
  EXPECT_TRUE(costLess(fastFew, fastMany, o));
}

TEST(CostModel, CanonicalStringIsStableAndDistinct) {
  Cost a{.latencyCycles = 1.0};
  Cost b{.latencyCycles = 1.0, .dramBytes = 1};
  EXPECT_EQ(canonicalCostString(a), canonicalCostString(Cost{.latencyCycles = 1.0}));
  EXPECT_NE(canonicalCostString(a), canonicalCostString(b));
}
```

Define `M_CYCLE(x)` as a small local helper `Cost{.latencyCycles = (x)}` in the test file.

- [ ] **Step 2: Run test to verify it fails** — target missing.
- [ ] **Step 3: Add the header and implementation.**

`canonicalCostString` formats each dimension with a fixed representation (`%.6f` for doubles, decimal for integers, `|`-separated) so it is byte-stable. `costLess` compares `primary`, then each `secondary` in order, respecting `minimize` (a `maximize` objective inverts the comparison); returns false on exact ties (the caller breaks ties by stable id).

- [ ] **Step 4: Register `lib/Mapping/CostModel.cpp` in `LLKMapping` and `MappingCostModelTest` in CMake.**
- [ ] **Step 5: Run tests to verify they pass.** `./build/MappingCostModelTest`
- [ ] **Step 6: Commit** `feat(mapping): add multi-dimensional Cost model (D1)`

---

### Task 3: `SearchBinding`

**Files:**
- Create: `include/LLK/Mapping/SearchBinding.h`, `lib/Mapping/SearchBinding.cpp`
- Create: `test/Mapping/search_binding.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `StableHash.h`.
- Produces:
  - `using SearchValue = std::variant<int64_t, std::string>`
  - `struct SearchBinding { std::string candidateId; llvm::StringMap<SearchValue> values; uint64_t stableHash; }`
  - `uint64_t computeSearchBindingHash(const llvm::StringMap<SearchValue> &)`
  - `SearchBinding makeSearchBinding(std::string candidateId, llvm::StringMap<SearchValue> values)`
  - `std::string canonicalSearchBindingString(const SearchBinding &)`
  - `bool searchBindingLess(const SearchBinding &, const SearchBinding &)`

- [ ] **Step 1: Write the failing test**

```cpp
// test/Mapping/search_binding.cpp
#include "LLK/Mapping/SearchBinding.h"
#include <gtest/gtest.h>
using namespace mlir::llk::mapping;

TEST(SearchBinding, HashIgnoresInsertionOrder) {
  llvm::StringMap<SearchValue> a, b;
  a["BM"] = int64_t{64}; a["tile_layout"] = std::string("blocked");
  b["tile_layout"] = std::string("blocked"); b["BM"] = int64_t{64};
  EXPECT_EQ(computeSearchBindingHash(a), computeSearchBindingHash(b));
}

TEST(SearchBinding, HashDistinguishesValueKindAndValue) {
  llvm::StringMap<SearchValue> ints, strs;
  ints["x"] = int64_t{1};
  strs["x"] = std::string("1");
  EXPECT_NE(computeSearchBindingHash(ints), computeSearchBindingHash(strs));
  llvm::StringMap<SearchValue> two;
  two["x"] = int64_t{2};
  EXPECT_NE(computeSearchBindingHash(ints), computeSearchBindingHash(two));
}

TEST(SearchBinding, CanonicalStringIsSortedAndStable) {
  auto s = makeSearchBinding("candidate_0", {{"b", int64_t{2}}, {"a", std::string("z")}});
  EXPECT_EQ(canonicalSearchBindingString(s), "a=z;b=2");
  EXPECT_NE(s.stableHash, 0u);
}

TEST(SearchBinding, OrderingIsByCanonicalString) {
  auto x = makeSearchBinding("c", {{"a", int64_t{1}}});
  auto y = makeSearchBinding("c", {{"a", int64_t{2}}});
  EXPECT_TRUE(searchBindingLess(x, y));
  EXPECT_FALSE(searchBindingLess(y, x));
}
```

- [ ] **Step 2: Run test to verify it fails** — target missing.
- [ ] **Step 3: Implement.** `values` is a `StringMap`, so iterate `keys()` sorted (`llvm::sort` of the key array) for both hashing and canonical string. Type tag each value (`i:` / `s:`) before hashing so `1` and `"1"` differ. `makeSearchBinding` recomputes `stableHash`.
- [ ] **Step 4: Register tests/lib in CMake.**
- [ ] **Step 5: Run tests to verify they pass.**
- [ ] **Step 6: Commit** `feat(mapping): add SearchBinding handoff type (D1)`

---

### Task 4: `WorkloadGraph` extraction and canonical finalize

**Files:**
- Create: `include/LLK/Mapping/WorkloadGraph.h`, `lib/Mapping/WorkloadGraph.cpp`
- Create: `test/Mapping/workload_graph.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: MLIR `OperationName`, `Type`, `AffineMap`, `DictionaryAttr`; `MicroDialect`.
- Produces:
  - `using WorkloadNodeId = uint32_t; using WorkloadValueId = uint32_t;`
  - `struct WorkloadValue { WorkloadValueId id; Type type; std::string name; bool external; }`
  - `struct WorkloadPort { WorkloadValueId value; Type type; std::optional<AffineMap> accessMap; }`
  - `struct WorkloadNode { WorkloadNodeId id; OperationName opName; llvm::SmallVector<WorkloadPort> inputs, outputs; DictionaryAttr attributes; uint32_t sourceOrdinal; }`
  - `struct WorkloadGraph` with `nodes`, `values`, `findNode`, `findValue`, `canonicalString`, `addNode`, `addValue`, `finalize`
  - `bool isWorkloadNodeOp(OperationName)`, `bool isTransparentWorkloadOp(OperationName)`
  - `llvm::Expected<WorkloadGraph> extractWorkloadGraph(mlir::Operation *kernel)`

**Node-op classification (locked):**
- **Node ops** (work needing target implementation): `micro.mma`, `micro.vector`, `micro.reduce`, `micro.tile_async_copy`, `micro.tile_store`, `micro.async_copy`, `micro.store`.
- **Transparent** (logical, folded into the port access map): `micro.tile_view`, `micro.tile_partition`. Resolving a value through them yields the same `WorkloadValueId` as their source.
- **Structural / non-node** (traversed, no node): `micro.kernel`, `micro.for`, `micro.spatial_for`, `micro.pipeline`, `micro.yield`, `micro.alloc`, `micro.tile_alloc`, `micro.wait`. An `alloc`/`tile_alloc` result with no producing node becomes an *external* `WorkloadValue` (its name is the value's defining-op-derived label).
- Any other op under the kernel is ignored (forward-compatible) — extraction never fails on an unknown op, it just does not create a node.

- [ ] **Step 1: Write the failing test**

```cpp
// test/Mapping/workload_graph.cpp
#include "LLK/Mapping/WorkloadGraph.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include <gtest/gtest.h>
using namespace mlir; using namespace mlir::llk::mapping;

static OwningOpRef<ModuleOp> parse(MLIRContext &ctx, llvm::StringRef text) {
  return parseSourceString<ModuleOp>(text, &ctx);
}

TEST(WorkloadGraph, ExtractsElementwiseVectorNode) {
  MLIRContext ctx; ctx.loadDialect<micro::MicroDialect>();
  auto m = parse(ctx, R"mlir(
    module {
      micro.kernel @k {
        %a = micro.tile_alloc : !micro.tile<[8, 8], f32>
        %r = micro.vector "add" %a, %a : !micro.tile<[8,8],f32>, !micro.tile<[8,8],f32> -> !micro.tile<[8,8],f32>
      }
    })mlir");
  ASSERT_TRUE(m);
  auto g = extractWorkloadGraph(*m->getOps<micro::KernelOp>().begin());
  ASSERT_TRUE(g);
  ASSERT_EQ(g->nodes.size(), 1u);
  EXPECT_EQ(g->nodes[0].opName.getStringRef(), "micro.vector");
  EXPECT_EQ(g->nodes[0].inputs.size(), 2u);
  EXPECT_EQ(g->nodes[0].outputs.size(), 1u);
  // Both operands resolve to the same allocation value.
  EXPECT_EQ(g->nodes[0].inputs[0].value, g->nodes[0].inputs[1].value);
}

TEST(WorkloadGraph, TileViewIsTransparent) {
  // %v = tile_view %a ; vector(%v) must bind the same value id as %a.
  // (assert inputs[0].value == the allocation's value id)
}

TEST(WorkloadGraph, IdenticalKernelsProduceIdenticalCanonicalStrings) {
  // extract twice from the same source; EXPECT_EQ(g1->canonicalString(), g2->canonicalString())
}

TEST(WorkloadGraph, NodeIdsAreCanonicalIndependentOfInsertionOrder) {
  // Build the same 3 distinct nodes through addNode in two orders, finalize both,
  // assert each node's (opName, id) pair matches across the two graphs.
}
```

- [ ] **Step 2: Run test to verify it fails** — target missing.
- [ ] **Step 3: Implement extraction.**
  - Locate `micro.kernel` (accept the op directly; a `findWorkloadKernel` helper mirroring `findMicroKernel` may reuse `micro::KernelOp`).
  - Recursive walk in program order. Maintain `DenseMap<Value, WorkloadValueId>`.
  - A block argument or a value defined outside the kernel becomes an external value whose `name` is `arg%d` / the defining op's result name.
  - Transparent ops: `resolveValue(v)` follows operand 0 recursively, memoizing.
  - Node op: inputs = resolved operands; outputs = fresh value ids; `sourceOrdinal` = a monotonic counter incremented in program order; `attributes` = the op's attribute dictionary.
  - `finalize()`: sort nodes by canonical key `(opName string, canonical input type list, canonical output type list, canonical attributes, sourceOrdinal)`; reassign `WorkloadNodeId` = index; rebuild value ids by first appearance in the sorted node order (external values first, sorted by name).
  - `canonicalString()` renders nodes and values in id order with fixed formatting.
- [ ] **Step 4: Register tests/lib in CMake.**
- [ ] **Step 5: Run tests to verify they pass.** Note: the test uses the real `micro` dialect assembly; if exact syntax differs from the repo's existing `.mlir` fixtures, copy the fixture spelling from `test/Perf/*.mlir`.
- [ ] **Step 6: Commit** `feat(mapping): extract WorkloadGraph from concrete micro IR (D1)`

---

### Task 5: Plan data model and canonical ids

**Files:**
- Create: `include/LLK/Mapping/MappingPlan.h`, `lib/Mapping/MappingPlan.cpp`
- Create: `test/Mapping/mapping_plan.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `SearchBinding.h` (`SearchValue`), `WorkloadGraph.h` (`WorkloadNodeId`, `WorkloadValueId`), `CostModel.h`, `StableHash.h`.
- Produces (verbatim from design §9):
  - Id aliases: `RuleId = std::string`, `ExecutorId = std::string`, `MemoryNodeId = std::string`, `LinkId = std::string`, `LayoutId = std::string`, `using CandidateId = uint64_t; using InstanceId = uint64_t; using ConnectionId = uint64_t; using PlanId = uint64_t;`
  - `struct PortSpec { std::string name; WorkloadValueId value; bool isInput; }`
  - `struct ExecutorRequirement { std::string capability; llvm::StringMap<std::string> attrs; }`
  - `struct MemoryRequirement { std::string kind; uint64_t minBytes = 0; }`
  - `struct LayoutRequirement { std::string layoutClass; }`
  - `struct ResourceUsage { uint64_t executorSlots = 0; llvm::StringMap<uint64_t> memoryBytes; }`
  - `struct LayoutTransform { std::string srcLayout; std::string dstLayout; AffineMap map; }`
  - `struct MappingCandidate { CandidateId id; RuleId rule; llvm::SmallVector<WorkloadNodeId> coveredNodes; std::string targetBundle; llvm::SmallVector<PortSpec> ports; llvm::SmallVector<ExecutorRequirement> executorRequirements; llvm::SmallVector<MemoryRequirement> memoryRequirements; llvm::SmallVector<LayoutRequirement> layoutRequirements; llvm::StringMap<SearchValue> resolvedParameters; Cost lowerBound; }`
  - `struct CandidateInstance { InstanceId id; CandidateId candidate; llvm::StringMap<ExecutorId> executorBindings; llvm::StringMap<MemoryNodeId> memoryBindings; llvm::StringMap<LayoutId> layoutBindings; ResourceUsage resourceUsage; Cost localCost; }`
  - `enum class ConnectionKind { Direct, Transfer, LayoutTransform, TransferAndTransform, Replicate, Reduce }`
  - `struct ConnectionPlan { ConnectionId id; InstanceId producer; llvm::SmallVector<InstanceId> consumers; WorkloadValueId value; ConnectionKind kind; llvm::SmallVector<MemoryNodeId> memoryRoute; llvm::SmallVector<ExecutorId> transferEngines; std::optional<AffineMap> producerMap; llvm::SmallVector<AffineMap> consumerMaps; std::optional<LayoutTransform> transform; Cost cost; }`
  - `struct PlanDiagnostics { std::vector<std::string> errors; std::vector<std::string> warnings; bool searchTruncated = false; }`
  - `struct CoveringPlan { PlanId id; uint64_t sourceBindingHash; llvm::SmallVector<InstanceId> instances; llvm::SmallVector<ConnectionId> connections; llvm::StringMap<SearchValue> globalParameters; Cost totalCost; PlanDiagnostics diagnostics; }`
  - Canonical functions: `uniqueId`/`computeId` for each struct + `canonicalString` for each struct; `sortUnique(SmallVector<WorkloadNodeId>&)` helper enforcing the §9.2 rule (non-empty, sorted, unique).

**Id derivation rule (locked):** each `computeXId` folds the object's canonical string through `stableHash` and returns the 64-bit value; string ids (`RuleId` etc.) are target-owned and never hashed here.

- [ ] **Step 1: Write the failing test**

```cpp
// test/Mapping/mapping_plan.cpp
#include "LLK/Mapping/MappingPlan.h"
#include <gtest/gtest.h>
using namespace mlir::llk::mapping;

TEST(MappingPlan, CandidateIdIsOrderIndependent) {
  MappingCandidate a; a.rule = "r"; a.coveredNodes = {2, 0, 1};
  MappingCandidate b; b.rule = "r"; b.coveredNodes = {0, 1, 2};
  EXPECT_EQ(computeCandidateId(a), computeCandidateId(b));   // canonical: sortUnique first
}

TEST(MappingPlan, CandidateIdChangesWithContent) {
  MappingCandidate a; a.rule = "r"; a.coveredNodes = {0};
  MappingCandidate b; b.rule = "r2"; b.coveredNodes = {0};
  EXPECT_NE(computeCandidateId(a), computeCandidateId(b));
}

TEST(MappingPlan, InstanceAndPlanIdsAreDeterministic) {
  // Same content built twice (and with bindings inserted in two orders) => equal ids.
}

TEST(MappingPlan, PlanIdChangesWithBindingHash) {
  CoveringPlan p; p.sourceBindingHash = 1; CoveringPlan q; q.sourceBindingHash = 2;
  EXPECT_NE(computePlanId(p), computePlanId(q));
}

TEST(MappingPlan, SortUniqueEnforcesCoveredNodeContract) {
  llvm::SmallVector<WorkloadNodeId> v{3, 1, 3, 2};
  sortUnique(v);
  EXPECT_EQ(v, (llvm::SmallVector<WorkloadNodeId>{1, 2, 3}));
}
```

- [ ] **Step 2: Run test to verify it fails** — target missing.
- [ ] **Step 3: Implement** canonical strings with all map/vector fields sorted (StringMap keys sorted; vectors sorted where the design says order is not semantic — `coveredNodes`, `consumers`, `instances`, `connections`, `memoryRoute`; order-sensitive fields like `ports` and `consumerMaps` keep declared order). `computeXId` canonicalizes a copy, then hashes.
- [ ] **Step 4: Register tests/lib in CMake.**
- [ ] **Step 5: Run tests to verify they pass.**
- [ ] **Step 6: Commit** `feat(mapping): add mapping plan data model and canonical ids (D1)`

---

### Task 6: Wire-up verification

- [ ] **Step 1:** Add `lib/Mapping/*.cpp` to `LLKMapping` in dependency order (StableHash, CostModel, SearchBinding, WorkloadGraph, MappingPlan).
- [ ] **Step 2:** `ninja -C build` — full build clean, no deprecated-API warnings.
- [ ] **Step 3:** `ctest --test-dir build -R Mapping --output-on-failure` — all Mapping tests pass.
- [ ] **Step 4:** `ctest --test-dir build --output-on-failure` — record the full registered-suite pass/fail/skip counts (the epic requires reporting the verified subset).
- [ ] **Step 5:** Run the deprecated-API audit from `.claude/rules/preferred-mlir-api.md` scoped to `lib/Mapping` and `include/LLK/Mapping`.
- [ ] **Step 6: Commit** `docs(mapping): record D1 verification results`

---

## Self-Review

**Spec coverage:**
- §8.3 `SearchBinding` → Task 3. ✅
- §9.1 stable ids → Tasks 4 (workload) + 5 (plan). ✅
- §9.2 `MappingCandidate` → Task 5. ✅
- §9.3 `CandidateInstance` → Task 5. ✅
- §9.4 `ConnectionPlan` → Task 5. ✅
- §9.5 `CoveringPlan` → Task 5. ✅
- §10 workload graph + affine maps → Task 4. ✅
- §17.1 `Cost` → Task 2. ✅
- §22.1 canonical ordering → Tasks 1/3/4/5. ✅
- Merge criterion (insertion-order-independent ids) → Task 4 (nodes) + Task 5 (plans). ✅

**Deferred (correctly out of D1):** routing D2, layouts D3, rules D4, placement D5, covering D6, AVX2 target D7, `perf::Candidate` migration (revised #50).

**Type-consistency check:** `SearchValue` is defined once (Task 3) and reused in Task 5; `WorkloadNodeId`/`WorkloadValueId` defined once (Task 4) and reused in Task 5; every `computeXId` returns `uint64_t` and every `canonicalXString` returns `std::string`. `hexId` used by Task 3+ canonical strings is defined in Task 1.

## Verification Results (2026-10-03)

Recorded per the epic's requirement to report the exact verified subset.

- **Build:** `ninja -C build` — clean, no errors; `-Werror=deprecated-declarations` honoured.
- **New tests:** `ctest -R Mapping` — 5/5 passed (`MappingStableHashTest`, `MappingCostModelTest`, `MappingSearchBindingTest`, `MappingWorkloadGraphTest`, `MappingPlanTest`).
- **Full suite:** `ctest --test-dir build --output-on-failure` — 92 registered, **90 passed, 2 skipped, 0 failed**. Skips are the pre-existing host-gated `SwigluScalar` and `SwiGLUVector`.
- **Deprecated-API audit:** clean across `include/LLK/Mapping`, `lib/Mapping`, `test/Mapping`.

### Deviations from the design's illustrative snippets

1. **Namespace `mlir::llk::mapping`** rather than the spec's `llk::mapping`, to match the existing `mlir::llk::perf`.
2. **`WorkloadNode::opName` is a `std::string`**, not an `OperationName`: `OperationName` has no public default constructor in LLVM 24 and would tie the graph to the builder's MLIR context. The graph is context-free and serialized; classification already works on strings.
3. **Added `StableHash.h`** (not in the design's §20 file list) as the single home for the toolchain-stable hash used by every id.
4. **`stableHashCombine` is a plain fold**; order-independence is provided by `stableHashSortedSet` and by sorting at each call site, which is what the design's canonical-ordering rule requires.
