# MachineModel v2 Implementation Plan (issue #82 / epic #67)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `LLKMachine` — a versioned, normalized machine topology (executor / memory / compute / transfer-engine nodes; containment, dominance, attachment, and directed-link edges) with a v2 YAML loader, validation, content hashing, and the normalized queries D2–D5 and D7 consume.

**Architecture:** A new static library `LLKMachine` owning `include/LLK/Machine` + `lib/Machine`, plus its own v2 profile files. It depends only on `LLVMSupport` (YAML parsing, diagnostics) and, in the loader `.cpp` only, the header-only Micro vocabulary (`MicroEnums.h`). It must not depend on `LLK/Perf`, `LLK/Mapping`, or any backend. The v1 model in `LLK/Perf` and the legacy AVX2 path stay untouched.

**Tech Stack:** C++20, LLVM/MLIR 24, LLVM YAML node API (`llvm::yaml`), CMake/Ninja, GoogleTest. FNV-1a 64-bit content hashing.

**Spec:** `docs/superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md` §11 (topology and placement), §11.5 (owner-to-placement), §11.6 (schema compatibility), §17.4 (content hash), §20 (layout). Issue #82.

## Global Constraints

- Node kinds (design §11.1): `executor`, `memory`, `compute`, `transfer_engine`. Edge kinds (§11.2): `contains` (parent), `dominates` (visibility), `attached_to`, `link`.
- Every node has a stable string ID; IDs are a single global namespace (duplicate across kinds is an error).
- Owner-to-executor matching (§11.5) is by abstract owner **kind**, never by target executor ID string comparison.
- Reject unknown **major** schema versions; do not reinterpret a v1 file as v2 (§11.6).
- Content hash must be stable across reloads and independent of YAML key order (§17.4).
- No route enumeration (D2), no LLKMap files (D3/D4), no `micro-perf` rewiring.
- `-Werror=deprecated-declarations`: no `builder.create<T>` / `type.cast<T>()` (no MLIR ops are built here anyway).
- TDD: failing test first, then minimal code; commit per task.

## Vocabulary (locked)

- executor `kind`: a Micro owner name — `cluster, core, warp, wave, subgroup, pe_group, pe, lane, worker, matrix_engine, vector_engine, dma` (`micro::symbolizeOwner`).
- memory `kind`: a Micro memory space — `dram, l2, sram, rf, acc, scratch` (`micro::symbolizeMemorySpace`).
- compute `kind`: capability kind — `matrix_engine` or `vector_engine`.
- transfer-engine `kind`: `dma`.

An executor may declare `refines:` — additional owner kinds it satisfies (transitively closed at load), which is how an accelerator's `pe` can satisfy a kernel's `worker` owner.

## File Structure

| File | Responsibility |
|---|---|
| `include/LLK/Machine/MachineModel.h` | Node/edge structs, `MachineModel`, normalized query API, content hash + verify declarations |
| `lib/Machine/MachineModel.cpp` | Queries, canonical string, content hash, `verifyMachineModel` |
| `lib/Machine/MachineHash.h` | Internal FNV-1a helpers (see note) |
| `include/LLK/Machine/MachineModelLoader.h` | `parseMachineModel(StringRef, name)`, `loadMachineModel(path)` |
| `lib/Machine/MachineModelLoader.cpp` | v2 YAML reader + per-field diagnostics |
| `machines/x86-avx2-v2.yaml` | AVX2 v2 profile |
| `machines/generic-ai-accel-v2.yaml` | generic accelerator v2 profile |
| `test/Machine/machine_model.cpp` | Queries, hash, verify (in-memory) |
| `test/Machine/machine_model_loader.cpp` | Loader positive + error cases, shipped profiles |

**Hash note:** D1 (PR #81) put `StableHash` under `LLK/Mapping`, which `LLK/Machine` must not depend on. This plan keeps a small self-contained hash in `lib/Machine/MachineHash.h`; a follow-up can lift both into `LLK/Support`. The duplication is deliberate and recorded, not accidental.

---

### Task 1: `LLKMachine` library, model types, queries, content hash

**Files:**
- Create: `include/LLK/Machine/MachineModel.h`, `lib/Machine/MachineModel.cpp`, `lib/Machine/MachineHash.h`
- Create: `test/Machine/machine_model.cpp`
- Modify: `CMakeLists.txt` (library + `add_llk_machine_test`)

**Interfaces:**
- `struct ExecutorNode { std::string id, kind; std::optional<std::string> parent; std::vector<int64_t> coordinates; uint32_t concurrency = 1; std::vector<std::string> refines; }`
- `struct MemoryNode { std::string id, kind, visibleFrom; uint64_t capacityBytes = 0, alignmentBytes = 1; std::vector<std::string> supportedLayouts; std::optional<uint32_t> banks; }`
- `struct ComputeNode { std::string id, kind, attachedTo; std::vector<std::string> elementTypes, supportedLayouts; std::vector<std::vector<int64_t>> shapes; uint64_t issueCycles = 1, latencyCycles = 0; std::optional<double> throughputPerCycle; uint32_t concurrency = 1; }`
- `struct TransferEngineNode { std::string id, kind, attachedTo; uint32_t count = 1, maxOutstanding = 1; }`
- `struct LinkEdge { std::string id, source, destination; double bandwidthBytesPerCycle = 0; uint64_t latencyCycles = 0, transactionBytes = 1; std::vector<std::string> transferEngines; uint32_t concurrency = 1; }`
- `struct MachineModel` with the five vectors, `target`, `description`, `schemaMajor`, `contentHash`, and:
  - finders `findExecutor/findMemory/findCompute/findTransferEngine/findLink(StringRef)`
  - `bool isWithin(StringRef nodeId, StringRef ancestorId) const` — containment
  - `bool ownerMatches(StringRef ownerKind, StringRef executorId) const`
  - `bool isVisible(StringRef memoryId, StringRef executorId) const`
  - `std::vector<const ComputeNode *> computesFor(StringRef executorId) const`
  - `std::vector<const TransferEngineNode *> transferEnginesFor(StringRef executorId) const`
- `uint64_t computeContentHash(const MachineModel &)`, `std::string canonicalMachineString(const MachineModel &)`

- [ ] **Step 1: Write the failing test**

```cpp
// test/Machine/machine_model.cpp
#include "LLK/Machine/MachineModel.h"
#include <gtest/gtest.h>
using namespace mlir::llk::machine;

MachineModel twoCoreMachine() {
  MachineModel m;
  m.target = "test";
  m.executors = {{"package.0", "worker", std::nullopt, {}, 1, {}},
                 {"core.0", "core", std::string("package.0"), {0}, 1, {"worker"}},
                 {"core.1", "core", std::string("package.0"), {1}, 1, {"worker"}}};
  m.memories = {{"dram.0", "dram", "package.0", 1u << 30, 64, {"row_major"}, std::nullopt},
                {"sram.0", "sram", "core.0", 1u << 15, 64, {"row_major", "blocked"}, 8u}};
  m.computes = {{"avx2.0", "vector_engine", "core.0", {"f32"}, {"row_major", "vectorized"}, {{8}}, 1, 5, 16.0, 1}};
  m.transferEngines = {{"dma.0", "dma", "core.0", 1, 8}};
  m.links = {{"dram_to_sram.0", "dram.0", "sram.0", 32.0, 20, 64, {"dma.0"}, 1}};
  return m;
}

TEST(MachineModel, ContainmentIsAncestorWalk) {
  MachineModel m = twoCoreMachine();
  EXPECT_TRUE(m.isWithin("core.0", "package.0"));
  EXPECT_FALSE(m.isWithin("package.0", "core.0"));
  EXPECT_TRUE(m.isWithin("core.0", "core.0"));
}

TEST(MachineModel, OwnerMatchesKindOrRefinement) {
  MachineModel m = twoCoreMachine();
  EXPECT_TRUE(m.ownerMatches("core", "core.0"));
  EXPECT_TRUE(m.ownerMatches("worker", "core.0"));   // declared refines
  EXPECT_FALSE(m.ownerMatches("pe", "core.0"));
}

TEST(MachineModel, VisibilityFollowsContainment) {
  MachineModel m = twoCoreMachine();
  EXPECT_TRUE(m.isVisible("sram.0", "core.0"));
  EXPECT_FALSE(m.isVisible("sram.0", "core.1"));   // visible_from core.0 only
  EXPECT_TRUE(m.isVisible("dram.0", "core.1"));    // visible_from package.0
}

TEST(MachineModel, AttachmentQueries) {
  MachineModel m = twoCoreMachine();
  ASSERT_EQ(m.computesFor("core.0").size(), 1u);
  EXPECT_EQ(m.computesFor("core.0")[0]->id, "avx2.0");
  EXPECT_TRUE(m.computesFor("core.1").empty());
  ASSERT_EQ(m.transferEnginesFor("core.0").size(), 1u);
}

TEST(MachineModel, ContentHashIsOrderIndependent) {
  MachineModel a = twoCoreMachine();
  MachineModel b = twoCoreMachine();
  std::swap(b.executors[1], b.executors[2]);
  EXPECT_EQ(computeContentHash(a), computeContentHash(b));
}
```

- [ ] **Step 2: Run test to verify it fails** — `ninja -C build MachineModelTest` fails: header/target missing.

- [ ] **Step 3: Add the library and test function to `CMakeLists.txt`**

```cmake
# Library: versioned machine topology for mapping legality (issue #82, epic #67).
# Target facts only -- no LLKPerf, no LLKMapping, no backend.
add_library(LLKMachine STATIC
    lib/Machine/MachineModel.cpp
)
target_include_directories(LLKMachine PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(LLKMachine PUBLIC LLVMSupport)
```

```cmake
function(add_llk_machine_test test_name test_file)
  add_llvm_executable(${test_name} ${test_file})
  target_link_libraries(${test_name} PRIVATE LLKMachine GTest::gtest_main)
  target_compile_definitions(${test_name} PRIVATE
      LLK_MACHINE_DIR="${CMAKE_SOURCE_DIR}/machines")
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_llk_machine_test(MachineModelTest test/Machine/machine_model.cpp)
```

- [ ] **Step 4: Implement the header and the query/hash/canonical-string bodies.** `canonicalMachineString` renders each node section sorted by id with fixed formatting; `computeContentHash` folds it through FNV-1a. `isWithin` walks `parent` with a visited-set guard (a malformed in-memory model must not loop forever). `ownerMatches` checks `executor->kind == ownerKind` or `llvm::is_contained(executor->refines, ownerKind)`. `isVisible` is `isWithin(executorId, memory.visibleFrom)`.

- [ ] **Step 5: Run test to verify it passes** — `./build/MachineModelTest`.

- [ ] **Step 6: Commit** `feat(machine): add LLKMachine model, queries, and content hash (v2)`

---

### Task 2: `verifyMachineModel`

**Files:** Modify `include/LLK/Machine/MachineModel.h`, `lib/Machine/MachineModel.cpp`; extend `test/Machine/machine_model.cpp`.

**Interfaces:** `llvm::Error verifyMachineModel(const MachineModel &)` — returns the first violation, deterministically ordered (check the model in declaration order), with message paths mirroring the YAML (`executors[1].parent`).

Rules: schema major matches; non-empty target; every ID non-empty and globally unique; executor parent exists and containment is acyclic; `refines` entries are known owner kinds; memory `kind` known and `capacityBytes > 0`, `alignmentBytes > 0`; memory `visibleFrom` references an executor; compute `kind` known, `attachedTo` references an executor, shapes non-empty and positive; transfer-engine `attachedTo` references an executor; link `source`/`destination` reference memories, `bandwidthBytesPerCycle > 0`, `transactionBytes > 0`, `transferEngines` reference transfer-engine nodes.

- [ ] **Step 1: Write the failing tests** — one per rule, each built by mutating `twoCoreMachine()`:

```cpp
TEST(MachineModel, VerifyRejectsDuplicateIds) {
  MachineModel m = twoCoreMachine();
  m.memories[0].id = "core.0";                 // collides with an executor
  EXPECT_THAT_ERROR(verifyMachineModel(m), Succeeded() == false);
}
TEST(MachineModel, VerifyRejectsMissingParent) {
  MachineModel m = twoCoreMachine();
  m.executors[1].parent = "nope.0";
  EXPECT_THAT_ERROR(verifyMachineModel(m), Failed());
}
TEST(MachineModel, VerifyRejectsContainmentCycle) {
  MachineModel m = twoCoreMachine();
  m.executors[0].parent = "core.0";            // package.0 -> core.0 -> package.0
  EXPECT_THAT_ERROR(verifyMachineModel(m), Failed());
}
TEST(MachineModel, VerifyRejectsDanglingLink) {
  MachineModel m = twoCoreMachine();
  m.links[0].destination = "l1.9";
  EXPECT_THAT_ERROR(verifyMachineModel(m), Failed());
}
TEST(MachineModel, VerifyRejectsZeroCapacity) {
  MachineModel m = twoCoreMachine();
  m.memories[0].capacityBytes = 0;
  EXPECT_THAT_ERROR(verifyMachineModel(m), Failed());
}
TEST(MachineModel, VerifyRejectsUnknownSchemaMajor) {
  MachineModel m = twoCoreMachine();
  m.schemaMajor = 1;
  EXPECT_THAT_ERROR(verifyMachineModel(m), Failed());
}
```

Use `llvm::Error` checks via `llvm::toString(std::move(error))` and `EXPECT_FALSE(...empty())`; the above `EXPECT_THAT_ERROR` shorthand is illustrative — write the concrete form in the test file.

- [ ] **Step 2: Run tests to verify they fail** (rules not implemented).
- [ ] **Step 3: Implement `verifyMachineModel`** with a small builder returning `llvm::Error`, mirroring the v1 loader's `node.path` message style.
- [ ] **Step 4: Run tests to verify they pass.**
- [ ] **Step 5: Commit** `feat(machine): verify machine topology (v2)`

---

### Task 3: v2 YAML loader

**Files:** Create `include/LLK/Machine/MachineModelLoader.h`, `lib/Machine/MachineModelLoader.cpp`; create `test/Machine/machine_model_loader.cpp`; modify `CMakeLists.txt`.

**Interfaces:**
- `llvm::Expected<MachineModel> parseMachineModel(llvm::StringRef yamlText, llvm::StringRef sourceName = "<memory>")`
- `llvm::Expected<MachineModel> loadMachineModel(llvm::StringRef path)`

YAML shape (design §11.4):

```yaml
schema: llk.machine.v2
target: x86-avx2
description: ...
executors:
  - {id: package.0, kind: worker}
  - {id: core.0, kind: core, parent: package.0, coordinates: [0], refines: [worker]}
memories:
  - {id: dram.0, kind: dram, visible_from: package.0, capacity_bytes: 1073741824, alignment_bytes: 64, supported_layouts: [row_major]}
compute:
  - {id: avx2.0, kind: vector_engine, attached_to: core.0, element_types: [f32], shapes: [[8]], issue_cycles: 1, latency_cycles: 5, supported_layouts: [row_major, vectorized]}
transfer_engines:
  - {id: dma.0, kind: dma, attached_to: core.0, count: 1, max_outstanding: 8}
links:
  - {id: dram_to_sram.0, source: dram.0, destination: sram.0, bandwidth_bytes_per_cycle: 32, latency_cycles: 20, transaction_bytes: 64, transfer_engines: [dma.0]}
```

- [ ] **Step 1: Write the failing tests** — a valid inline fixture loads and matches the expected node counts; a missing `schema` fails; `schema: llk.machine.v1` (or any other major) fails; an unknown key fails; a non-list where a list is expected fails; a missing required field fails; duplicate ids fail (via verification).
- [ ] **Step 2: Run tests to verify they fail.**
- [ ] **Step 3: Implement the loader** by adapting the v1 loader's `Loader` helper pattern (`lib/Perf/MachineModelLoader.cpp`): a `Loader` holding `llvm::SourceMgr`, `yaml::Stream`, and a saved `Error`; helpers `readText/readUInt/readStringList/forEachEntry/requireKey`; per-section parsers; call `verifyMachineModel` at the end. Unknown keys are errors, matching the v1 loader's strictness.
- [ ] **Step 4: Run tests to verify they pass.**
- [ ] **Step 5: Commit** `feat(machine): load MachineModel v2 from YAML`

---

### Task 4: Shipped v2 profiles, wiring, and verification

**Files:** Create `machines/x86-avx2-v2.yaml`, `machines/generic-ai-accel-v2.yaml`; extend `test/Machine/machine_model_loader.cpp`; modify `CMakeLists.txt`.

- [ ] **Step 1: Write the failing test** — load both shipped profiles by `LLK_MACHINE_DIR` path, assert they verify and have non-empty executors/memories/links; assert the AVX2 profile's `worker` executor matches a `worker` owner and the generic profile's `pe` matches `pe` and (via `refines`) `worker`.
- [ ] **Step 2: Run tests to verify they fail** (files absent).
- [ ] **Step 3: Author both profiles.** AVX2: `package.0` (worker) containing eight `core.N` (core, refines worker) with `dram.0`/`l2.0`/`sram.0`/`acc.0` memories, `avx2.N` vector_engine computes, a `dma.N` transfer engine, and `dram→l2`, `l2→sram`, `sram→acc`, `sram→dram` links. Generic: `chip.0` (worker) → `core.N` (core) → `pe_group.N` → `pe.N` (pe, refines worker), with `mxu.N` matrix_engine and `vpu.N` vector_engine computes, `dram`/`sram`/`acc` memories, DMA engines, and a two-hop `dram→sram→acc` path that D2's routing fixture will need.
- [ ] **Step 4: Run tests to verify they pass.**
- [ ] **Step 5: Full build and suite**

Run: `ninja -C build && ctest --test-dir build --output-on-failure`
Expected: clean; record the registered/passed/skipped/failed counts.

- [ ] **Step 6: Commit** `feat(machine): ship AVX2 and generic-accelerator v2 profiles`

---

## Self-Review

**Issue #82 coverage:** node kinds + edges (Task 1); properties (Task 1 structs); normalized queries (Task 1); versioning + content hash + validation (Tasks 1–2); AVX2 + generic profiles (Task 4); load-time rejection of duplicate/missing-parent/dangling/cycle/version/capacity (Tasks 2–3). ✅

**Non-scope respected:** no routing (D2), no LLKMap (D3/D4), no `micro-perf` rewiring, v1 files untouched.

**Type consistency:** node structs are defined once in Task 1 and referenced by the loader (Task 3) and profiles tests (Task 4); `verifyMachineModel` and `computeContentHash` are declared in Task 1's header and implemented in Task 1/Task 2 respectively.

## Verification Results (2026-10-03)

- **Build:** `ninja -C build` — clean, no warnings.
- **New tests:** `MachineModelTest` 21/21, `MachineModelV2LoaderTest` 12/12.
- **Full suite:** `ctest --test-dir build --output-on-failure` — 89 registered, **87 passed, 2 skipped, 0 failed**. Skips are the pre-existing host-gated `SwigluScalar`/`SwiGLUVector`.
- **Deprecated-API audit:** clean across `include/LLK/Machine`, `lib/Machine`, `test/Machine`.

### Decisions taken while implementing

1. **Node `kind` is a Micro vocabulary name** — executor kinds are `micro::Owner` names, memory kinds are `micro::MemorySpace` names (`MicroEnums.h`), validated at load. This is what lets owner matching be a kind comparison (§11.5) with no target vocabulary in generic code.
2. **Refinement is declared per executor** (`refines:`), transitively closed at load. The generic code only compares kinds; target policy stays in the profile.
3. **The loader accepts `llk.machine.v2.<minor>`** but rejects any other major (§11.6).
4. **A small self-contained FNV-1a hash** lives in `lib/Machine/MachineHash.h`; D1's equivalent is under `LLK/Mapping`, which `LLK/Machine` must not depend on. A follow-up should lift both into `LLK/Support`.
5. **The v1 model and profiles are untouched**; v2 ships as sibling files (`*-v2.yaml`).
