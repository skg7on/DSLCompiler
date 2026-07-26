# Final Whole-Branch Review: Post-M6 Enhancement

**Branch:** `feat/post-m6-enhancement` (c526994..5c8191a)
**Reviewer:** Cross-cutting audit (all 9 per-task reviews were clean)
**Date:** 2026-07-26

---

## Overall Verdict: APPROVED_WITH_MINOR

The branch is clean, well-structured, and safe to merge. Two minor documentation issues noted below; neither blocks merge. All 9 per-task reviews were independently approved, and no cross-cutting interface, build, or logic issues were found.

---

## Cross-Cutting Findings

### F1: ARCHITECTURE.md Status Labels Out of Date (Minor)

**Location:** `ARCHITECTURE.md`, lines 87-95 (ops table) and line 95 (attributes line)

The ops table marks `llk.assume` and `llk.dispatch` as "Planned (post-M6)", and the attributes line marks `#llk.layout` as "Planned (post-M6)". These are now fully implemented in this branch:

- TableGen definitions: `LLKOps.td` (AssumeOp at line 139, DispatchOp at line 179), `LLKDialect.td` (LLK_Layout at line 60)
- C++ verifiers: `LLKOps.cpp` (AssumeOp::verify at line 264, DispatchOp::verify at line 278)
- Hand-written enum: `LLKEnums.h` (Layout enum at line 90 with stringify/symbolize at lines 96-110)
- Round-trip FileCheck tests: `test/Dialect/assume_roundtrip.mlir`, `test/Dialect/dispatch_roundtrip.mlir`

**Recommendation:** Update status labels to "Implemented" before merge, or in an immediate follow-up.

### F2: Missing `--op` CLI Flag Referenced in Plan (Minor)

**Location:** `tools/llk-compile/llk-compile.cpp`

Task 7.3's implementation plan called for adding:
```cpp
static cl::opt<std::string>
    optOp("op", cl::desc("Operation kind (swiglu, rope, attention)"),
          cl::init("swiglu"));
```

The actual code does not include this flag. The operation type is implicitly captured via the module content hash in the cache key (line 278-286). This is not a functional defect -- the hash correctly distinguishes operation types -- but it is a plan-vs-implementation discrepancy worth noting for traceability.

**Recommendation:** Either add the flag or update the plan to reflect the implicit approach. No functional impact.

---

## Global Constraint Compliance

| Constraint | Status | Notes |
|---|---|---|
| Zero SwiGLU leak in shared code | PASS | `grep -r "swiglu" runtime/JitCache.cpp tools/llk-compile/llk-compile.cpp` returns empty |
| LLVM version guards | PASS | Guards in `llk-compile.cpp:66,157`, `JitCache.cpp:136`, `ForallToOpenMP.cpp:51`, TileAndVectorize.cpp (4 sites), plus existing test files |
| C++20 | PASS | All new code uses C++20 features (namespaced enums through TableGen, structured bindings in cache code) |
| Naming conventions | PASS | `llk` prefix, `llk-*` tools, CamelCase passes, snake_case files |
| Worktree isolation | PASS | All writes in `.claude/worktrees/feat+post-m6-enhancement/` |

---

## Architecture Integrity

### Pipeline Order (22 stages, llk-compile.cpp:86-183)

The pipeline is correctly ordered:
1. LLK->Linalg (domain lowering first)
2. Canonicalize
3. Shape specialization (M-bucket classification)
4. Schedule selection (data-driven, not embedded in passes)
5. Fuse double contraction (SwiGLU-specific; no-op for RoPE/Attention)
6. Canonicalize
7. Pack weights
8. Tile and vectorize
9. Canonicalize
10. Linearize forall (2D->1D)
11. Serial/parallel dispatch
12. **One-Shot Bufferize** (correctly after all tensor transformations)
13. Scratch analysis (post-bufferization audit)
14. Residual linalg->loops (dynamic lookup, handles surviving linalg.generic/fill)
15. Forall->LLRT (ThreadPool runtime)
16. Vector->LLVM (SIMD)
17-21. Standard MLIR conversion chain: SCF->CF, Arith->LLVM, Math->LLVM, Func->LLVM, MemRef->LLVM
22. Reconcile unrealized casts

**Assessment:** Pipeline respects the key architectural rule -- all tensor transformations before bufferization. LLVM version guards on the SCF->CF pass name (stage 17). Stage 14 (`convert-linalg-to-loops`) uses `PassInfo::lookup` for graceful degradation if the pass isn't available, acceptable for a research compiler.

### Dialect Consistency

The new ops (`llk.assume`, `llk.dispatch`) follow the same patterns as existing ops:
- DestinationStyleOpInterface on AssumeOp (matches existing ops)
- SymbolUserOpInterface on DispatchOp (appropriate for dispatch)
- Custom verifiers with meaningful error messages
- `LLK_ReturnOp` used as terminator within dispatch variants

The `#llk.layout` attribute follows the exact same pattern as `#llk.activation`, `#llk.math_mode`, and `#llk.softmax_mode` -- I32EnumAttr with `genSpecializedAttr = 0` and hand-written stringification in LLKEnums.h.

### JitCache Multi-Level Architecture

The 4-level cache (L1: IR, L2: Optimized MLIR, L3: Object Code, L4: Packed Weights) is correctly implemented with:
- Per-level `shared_mutex` for concurrent read access
- LRU eviction with MRU-update on access (`splice` to front)
- Thread-safe stats accumulation
- Clear/reset functionality
- `WeightKey` struct for packed weight caching (serialized to 32-byte binary key)

### ForallToOpenMP Prototype Path

The pass correctly chains `scf.forall -> scf.parallel -> omp.parallel + omp.wsloop`:
- Uses `OpPassManager` (scoped to ModuleOp) for the nested pipeline, not a flat `PassManager`
- LLVM version guard on `createForallToParallelLoopPass` vs. fallback to canonicalizer
- Registered in both `llk-opt` (line 83) and `CMakeLists.txt` (ForallToOpenMP test, line 325-329)
- Correctly declared in `Passes.td` with dependent dialects

---

## Test Coverage

| Area | Tests | Status |
|---|---|---|
| New ops round-trip | `assume_roundtrip.mlir`, `dispatch_roundtrip.mlir` | Covered |
| Cache eviction/LRU | `cache_eviction.cpp` (7 test cases) | Covered |
| Cache miss | `jit_cache_miss.cpp` (3 test cases) | Covered |
| Schedule determinism | `schedule_determinism.cpp` (3 test cases) | Covered |
| M-bucket classification | `specialization_dispatch.cpp` (2 test cases, 17 shape combos) | Covered |
| Packed weight equivalence | `packed_vs_unpacked.cpp` (2 test cases) | Covered |
| Thread affinity | `affinity_test.cpp` (4 test cases) | Covered |
| Schedule regression | `schedule_regression.cpp` (3 test cases, placeholder thresholds) | Covered |
| OpenMP lowering | `forall_to_openmp.mlir` | Covered |
| RoPE broadcast | `rope_broadcast.mlir` | Covered |

Broader test coverage note: No single test exercises the full 22-stage pipeline end-to-end for all three ops (the existing `shared_pipeline.cpp` test from M6 exercises a subset). This is expected at this stage -- full integration tests would be a natural next milestone deliverable.

---

## Merge Readiness Assessment

**Status: READY TO MERGE**

The branch is clean across all cross-cutting dimensions:
- **No interface mismatches:** All 9 tasks produce consistent interfaces; no conflicting assumptions found
- **No build regressions:** CMakeLists.txt properly adds all new libraries, links them correctly, and registers all new tests
- **No constraint violations:** LLVM version guards present, naming conventions followed, no SwiGLU leak, C++20 compliant
- **Architecture consistent:** Pipeline order is correct; dialect additions follow established patterns; cache design is internally consistent
- **Test coverage adequate:** All new functionality has corresponding tests

The two findings (F1: ARCHITECTURE.md labels, F2: missing `--op` flag) are documentation-level issues that do not affect correctness or functionality.

**Recommendation:** Fix F1 (update ARCHITECTURE.md status labels) before merging if possible; F2 can be addressed in a follow-up. If neither fix is applied before merge, open a tracking issue for the documentation alignment gap.

### Fix: Minor Doc Corrections

**Date:** 2026-07-26

**F1 resolved:** Updated `ARCHITECTURE.md` line 92, 93, 95 status labels from "🔨 Planned (post-M6)" to "✅ Implemented" for three items:

| Item | Old Status | New Status |
|------|-----------|------------|
| `llk.assume` (line 92) | 🔨 Planned (post-M6) | ✅ Implemented |
| `llk.dispatch` (line 93) | 🔨 Planned (post-M6) | ✅ Implemented |
| `#llk.layout` (line 95 attr line) | 🔨 Planned (post-M6) | ✅ Implemented |

Verification confirmed all three are fully implemented in this branch:
- TableGen definitions: `LLKOps.td` (AssumeOp line 139, DispatchOp line 179), `LLKDialect.td` (LLK_Layout line 62)
- C++ verifiers: `LLKOps.cpp` (AssumeOp::verify line 264, DispatchOp::verify line 278)
- Hand-written enum: `LLKEnums.h` (Layout enum line 90 with stringify/symbolize at lines 96-110)
- Round-trip FileCheck tests: `test/Dialect/assume_roundtrip.mlir`, `test/Dialect/dispatch_roundtrip.mlir`

**F2:** Not applied per review recommendation — the `--op` flag absence is a plan-vs-code discrepancy only, and the hash-based approach is correct.
