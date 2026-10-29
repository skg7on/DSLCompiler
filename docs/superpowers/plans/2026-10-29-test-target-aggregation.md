# Test Target Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the `test/` build from 27 GTest executables to 15 while making `check-llk` structurally unable to miss a test binary, compiling the shared 1,308-line mapping fixture once instead of four times, and giving every CTest entry that external tooling depends on a stable name.

**Architecture:** Three stacked PRs. PR 1 makes target registration automatic (derived from the CMake directory graph, not a hand-written list) and introduces a shared `OBJECT` library for test-only fixtures. PR 2 folds eight single-purpose binaries into their category binaries and gives Tuning a real `LLKTuningTests`. PR 3 collapses the three reference/selected-target binary pairs into single binaries driven by a runtime switch instead of the `LLK_REQUIRE_SELECTED_TARGET` compile-time macro.

**Tech Stack:** CMake 3.20, Ninja, GoogleTest via `gtest_discover_tests`, CTest, GitHub Actions.

**Spec:** Issue #266, which records the review findings (F1-F7) that this plan implements. No separate spec document exists; the findings are restated in the Global Constraints section below.

## Global Constraints

- The repository builds against CMake **3.20**. `target_link_libraries()` with an `OBJECT` library target requires CMake 3.21, so object libraries MUST be consumed through `$<TARGET_OBJECTS:...>` in the consumer's `SOURCES` list.
- `LLVM_PROJECT_BUILD_DIR` is mandatory; every configure needs `-DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build`.
- `add_library` targets that reference LLVM types MUST link `LLKBuildOptions` (`-fno-rtti -fno-exceptions`) or `llvm::Error` links fail with an undefined typeinfo.
- CTest entry **names** are a contract. `.github/workflows/ci.yml:133` validates JUnit output against `REQUIRED = {"MappedAVX2Acceptance", "MappedAllocationLifetimeTest"}` in `test/Execution/check_required_execution.py:8`. The names `FrameworkSourceTest`, `DynamicShapeContractTest`, and `DynamicShapeMappingTest` appear as verification gates in `docs/superpowers/plans/2026-10-17-dynamic-shape-llk-microir.md`.
- `.github/workflows/ci.yml:123` invokes `./MappedAVX2Acceptance` **by path** from the build root.
- Every task ends with `ninja check-llk` green (or its documented equivalent) before committing.
- Commit messages end with the project's Claude Code attribution line.

## Baseline facts (re-measured 2026-10-29 on `main` @ `02e7e25`, after PR 1 and #265 merged)

| Fact | Value |
|---|---|
| Test executables | 29 (10 category + 19 standalone), 27 with `LLK_BUILD_PYTHON_CAPI=OFF` |
| `MappedAcceptance.*` registrations | 1 suite entry (PR 1 removed the 13 duplicate per-case entries) |
| `resource_regression_fixture.cpp` compiles | 1 (PR 1's `LLKTestFixtures` OBJECT library) |
| Binaries missing from `check-llk` deps | 0 (PR 1 derives the set from the directory graph) |
| Reference/selected pairs | 4: `MappedAcceptance`, `GenericElementwise`, `FrameworkPrimitives`, `DynamicMappedAcceptance` |

The figures recorded for `main` @ `1a20d69` (27 executables, 1,505 CTest cases, 4 fixture compiles, 5 missing dependencies) describe the state PR 1 fixed; they are kept in the PR 1 sections as the "before" side of those tasks and are no longer the branch baseline.

`#265` added a fourth reference/selected pair, `DynamicMappedAcceptance` / `DynamicMappedAVX2Acceptance`, built from `test/Execution/dynamic_mapped_acceptance.cpp` with the same `LLK_REQUIRE_SELECTED_TARGET` macro and custom `main` as the other three. It belongs to PR 3, not PR 2. `#265` also changed `test/Execution/check_selected_target_flag.py` to check **two** pairs (2 or 4 positional arguments) and added a second CI invocation of `./DynamicMappedAVX2Acceptance`.

Cases per standalone source (needed to size the alias duplication in PR 2): `mapped_allocation_lifetime.cpp` 4, `dynamic_mapped_invocation.cpp` 13, `mapped_invocation.cpp` 7, `mapped_codegen_requirements.cpp` 5, `selected_group_lowering.cpp` 5, `dynamic_shape_mapping.cpp` 9.

---

# PR 1 — `refactor(build): register test targets automatically and share test fixtures`

Branch: `refactor/test-target-aggregation-pass01`
Title: `refactor(build): auto-register check-llk test deps and share fixtures`

## File Structure

| File | Responsibility |
|---|---|
| `test/CMakeLists.txt` | Derive `LLK_TEST_TARGETS` from the CMake directory graph; define `LLKTestFixtures` |
| `test/Execution/CMakeLists.txt` | Drop `mapped_acceptance.cpp` from the category binary (kills the duplicate registration) |
| `test/Tuning/CMakeLists.txt` | Give the four standalone Tuning binaries the `Tuning` label |
| `test/Conversion/CMakeLists.txt` | Consume `$<TARGET_OBJECTS:LLKTestFixtures>` |
| `test/Perf/CMakeLists.txt` | Consume `$<TARGET_OBJECTS:LLKTestFixtures>` |
| `test/Mapping/CMakeLists.txt` | Consume `$<TARGET_OBJECTS:LLKTestFixtures>` in both targets |

---

### Task 1: Derive `check-llk` dependencies from the build graph

**Files:**
- Modify: `test/CMakeLists.txt:17-23` (function body), `test/CMakeLists.txt:50-74` (the `check-llk` target)

**Interfaces:**
- Consumes: nothing.
- Produces: global property `LLK_TEST_TARGETS` containing every `EXECUTABLE` target created anywhere under `test/`, populated automatically after all `add_subdirectory()` calls.

- [ ] **Step 1: Write the failing check**

Run from the worktree root:

```bash
ninja -C build -t query test/check-llk | grep -cE 'GenericElementwiseTest|FrameworkPrimitivesTest|DynamicShapeContractTest'
```

Expected before the change: **`0`** (exit 1 from `grep -c`). These five binaries are absent from `check-llk` dependencies today. Note the query target is `test/check-llk`, not `check-llk`: `ninja -t query check-llk` reports only an alias and silently hides the dependency list.

- [ ] **Step 2: Add the recursive collector to `test/CMakeLists.txt`**

Insert after `include(GoogleTest)`:

```cmake
# Collect every executable created under test/ into LLK_TEST_TARGETS. Walking
# the directory graph instead of maintaining a list is what makes it
# impossible for a new test binary to be omitted from check-llk; five binaries
# were missing from the hand-written list before this replaced it.
function(_llk_collect_test_targets dir)
  get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(_subdir IN LISTS _subdirs)
    _llk_collect_test_targets("${_subdir}")
  endforeach()

  get_property(_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  set(_executables)
  foreach(_target IN LISTS _targets)
    get_target_property(_target_type ${_target} TYPE)
    if(_target_type STREQUAL "EXECUTABLE")
      list(APPEND _executables ${_target})
    endif()
  endforeach()
  if(_executables)
    set_property(GLOBAL APPEND PROPERTY LLK_TEST_TARGETS ${_executables})
  endif()
endfunction()
```

- [ ] **Step 3: Delete the append from `add_llk_test_category`**

In `test/CMakeLists.txt`, remove line 22:

```cmake
  set_property(GLOBAL APPEND PROPERTY LLK_TEST_TARGETS ${test_target})
```

The collector in Step 2 supersedes it, and leaving both would append duplicates.

- [ ] **Step 4: Replace the hand-written dependency list**

Replace `test/CMakeLists.txt:50-60` (from `get_property(_llk_check_deps GLOBAL PROPERTY LLK_TEST_TARGETS)` through `endforeach()`) with:

```cmake
# Tools are not test executables but CLI tests invoke them by path.
set(_llk_check_deps llk-opt micro-perf llk-tune llk-compile llk-bench llk-validate)
foreach(_llk_tool IN LISTS _llk_check_deps)
  if(NOT TARGET ${_llk_tool})
    list(REMOVE_ITEM _llk_check_deps ${_llk_tool})
  endif()
endforeach()

_llk_collect_test_targets("${CMAKE_CURRENT_SOURCE_DIR}")
get_property(_llk_test_targets GLOBAL PROPERTY LLK_TEST_TARGETS)
list(APPEND _llk_check_deps ${_llk_test_targets})
list(REMOVE_DUPLICATES _llk_check_deps)
```

- [ ] **Step 5: Reconfigure and verify the check now passes**

```bash
cmake -S . -B build -GNinja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja -C build -t query test/check-llk | grep -cE 'GenericElementwiseTest|GenericElementwiseAVX2Test|FrameworkPrimitivesTest|FrameworkPrimitivesAVX2Test|DynamicShapeContractTest'
```

Expected: `5` when `LLK_BUILD_E2E_TESTS=ON` (the default). If the configure used `-DLLK_BUILD_E2E_TESTS=OFF`, this is `1` and that is correct: `GenericElementwise{,AVX2}Test` and `FrameworkPrimitives{,AVX2}Test` are declared inside `if(LLK_BUILD_E2E_TESTS)`, so only `DynamicShapeContractTest` remains. The pattern must name all five binaries. A pattern naming only the three non-AVX2 names also matches their AVX2 siblings as substrings, so it would print `3` and look like a failure.

- [ ] **Step 6: Verify the tool subset still resolves**

```bash
ninja -C build -t query test/check-llk | grep -cE 'bin/(llk-opt|micro-perf|llk-tune|llk-compile|llk-bench|llk-validate)$'
```

Expected: `6` when `LLK_BUILD_TOOLS=ON` (the default). If the configure used `-DLLK_BUILD_TOOLS=OFF`, this is `0` and that is correct.

- [ ] **Step 7: Commit**

```bash
git add test/CMakeLists.txt
git commit -m "refactor(build): derive check-llk dependencies from the test directory graph"
```

---

### Task 2: Stop registering `MappedAcceptance` twice

**Files:**
- Modify: `test/Execution/CMakeLists.txt:2-8`

**Interfaces:**
- Consumes: nothing.
- Produces: `LLKExecutionTests` no longer defines `main`; it links `GTest::gtest_main` like every other category binary. `MappedAcceptance` remains the single owner of that suite.

- [ ] **Step 1: Write the failing check**

`gtest_discover_tests` registers cases POST_BUILD, so the binary must exist before `ctest -N` reports them.

```bash
ninja -C build LLKExecutionTests
ctest --test-dir build -N 2>/dev/null | grep -cE 'Test +#.*: MappedAcceptance\.'
```

Expected before: `13` (per-case entries discovered from `LLKExecutionTests`). Expected after: `0`.

- [ ] **Step 2: Remove the source from the category list**

In `test/Execution/CMakeLists.txt`, delete the `mapped_acceptance.cpp` line from `_llk_execution_jit_sources` (line 6). The resulting list is:

```cmake
set(_llk_execution_jit_sources
      swiglu_scalar.cpp
      triton_swiglu_staged.cpp
      swiglu_vector_avx2.cpp
      micro_kernel_execution.cpp
      swiglu_fused_memory.cpp)
```

- [ ] **Step 3: Reconfigure and confirm both halves**

```bash
cmake -S . -B build -GNinja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja -C build LLKExecutionTests
ctest --test-dir build -N 2>/dev/null | grep -cE 'Test +#.*: MappedAcceptance'
ctest --test-dir build -N 2>/dev/null | grep -E 'Test +#.*: MappedAcceptance$'
```

Expected: the first prints `1`; the second prints exactly `  Test #NNN: MappedAcceptance`.

The relink is not optional. `gtest_discover_tests` writes its case list POST_BUILD, so after a configure alone the stale 13 case entries survive and the count still reads `14`. The count drops only when `LLKExecutionTests` actually relinks.

- [ ] **Step 4: Build and run the affected binaries**

```bash
ninja -C build LLKExecutionTests MappedAcceptance MappedAVX2Acceptance
ctest --test-dir build -R '^MappedAcceptance$' --output-on-failure
```

Expected: the entry reports **Skipped** on a host without AVX2 — the `SelectedAvx2*` cases skip, and `SKIP_REGULAR_EXPRESSION` makes CTest report the entry as Skipped — with exit code 0, and **Passed** on an x86_64 host with AVX2.

- [ ] **Step 5: Commit**

```bash
git add test/Execution/CMakeLists.txt
git commit -m "fix(build): register MappedAcceptance once instead of twice"
```

---

### Task 3: Label the Tuning and Execution standalone tests

**Files:**
- Modify: `test/Tuning/CMakeLists.txt:10,19,27,35`
- Modify: `test/Execution/CMakeLists.txt:85,101,160,192`

**Interfaces:**
- Consumes: nothing.
- Produces: `ctest -L Tuning` selects all four Tuning binaries; every standalone Execution binary carries the `Execution` label (line 192 already does).

- [ ] **Step 1: Write the failing check**

```bash
ctest --test-dir build -N -L Tuning | tail -2
```

Expected before: `Total Tests: 0`.

- [ ] **Step 2: Add the label in Tuning**

Append one of these lines immediately after each of the four `add_test` calls (lines 10, 19, 27, 35 of `test/Tuning/CMakeLists.txt`):

```cmake
set_tests_properties(FrameworkSourceTest PROPERTIES LABELS "Tuning")
set_tests_properties(DynamicShapeContractTest PROPERTIES LABELS "Tuning")
set_tests_properties(MappedTuningSessionTest PROPERTIES LABELS "Tuning")
set_tests_properties(CandidateInstantiationTest PROPERTIES LABELS "Tuning")
```

Place each line inside the same `if()` block as its `add_test`.

- [ ] **Step 3: Add the label in Execution**

After `add_test(NAME MappedInvocationTest COMMAND MappedInvocationTest)` (line 85):

```cmake
set_tests_properties(MappedInvocationTest PROPERTIES LABELS "Execution")
```

After `add_test(NAME MappedCodegenRequirementsTest COMMAND MappedCodegenRequirementsTest)` (lines 101-102):

```cmake
set_tests_properties(MappedCodegenRequirementsTest PROPERTIES LABELS "Execution")
```

After `add_test(NAME SelectedGroupLoweringTest COMMAND SelectedGroupLoweringTest)` (line 160):

```cmake
set_tests_properties(SelectedGroupLoweringTest PROPERTIES LABELS "Execution")
```

- [ ] **Step 4: Reconfigure and verify**

```bash
cmake -S . -B build -GNinja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ctest --test-dir build -N -L Tuning | tail -2
```

Expected: `Total Tests: 4`.

- [ ] **Step 5: Commit**

```bash
git add test/Tuning/CMakeLists.txt test/Execution/CMakeLists.txt
git commit -m "fix(build): label standalone Tuning and Execution test entries"
```

---

### Task 4: Compile the mapping fixture once

**Files:**
- Modify: `test/CMakeLists.txt` (add `LLKTestFixtures` after the `include(GoogleTest)` block)
- Modify: `test/Conversion/CMakeLists.txt:11`
- Modify: `test/Perf/CMakeLists.txt:3`
- Modify: `test/Mapping/CMakeLists.txt:7,57`

**Interfaces:**
- Consumes: `LLK::MLIR`, `LLKBuildOptions`, `LLKMachine`, `LLKMapping`, `LLKMicroMapping`, `LLKTargetMapping` (all defined before `add_subdirectory(test)` runs).
- Produces: `LLKTestFixtures`, an `OBJECT` library whose objects consumers list as `$<TARGET_OBJECTS:LLKTestFixtures>`.

- [ ] **Step 1: Write the failing check**

```bash
ninja -C build -t targets all | grep -c 'resource_regression_fixture\.cpp\.o'
```

Expected before: `4` object targets. Expected after: `1`. Do not use `ninja -t commands` here: it prints link lines too, and every link line that mentions the object inflates the count.

- [ ] **Step 2: Define the object library**

In `test/CMakeLists.txt`, after `include(GoogleTest)` and before `add_subdirectory(CMake)`:

```cmake
# Test-only fixtures shared by several categories. An OBJECT library compiles
# them once without packaging a fixture-only TU into a production library.
add_library(LLKTestFixtures OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/Mapping/resource_regression_fixture.cpp)
target_link_libraries(LLKTestFixtures PRIVATE
    LLKBuildOptions
    LLKMachine
    LLKMapping
    LLKMicroMapping
    LLKTargetMapping
    LLK::MLIR)
target_include_directories(LLKTestFixtures PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping
    ${CMAKE_CURRENT_SOURCE_DIR}/Mapping)
```

`PRIVATE` on the link libraries is deliberate: they supply include directories and compiler flags for the one translation unit but are not propagated to consumers, which link the libraries they already need.

- [ ] **Step 3: Consume it from Conversion**

In `test/Conversion/CMakeLists.txt`, replace the source line `../Mapping/resource_regression_fixture.cpp` (line 11) with:

```cmake
        $<TARGET_OBJECTS:LLKTestFixtures>
```

- [ ] **Step 4: Consume it from Perf**

In `test/Perf/CMakeLists.txt`, replace the source line `../Mapping/resource_regression_fixture.cpp` (line 3) with:

```cmake
        $<TARGET_OBJECTS:LLKTestFixtures>
```

- [ ] **Step 5: Consume it from both Mapping targets**

In `test/Mapping/CMakeLists.txt`, replace `resource_regression_fixture.cpp` at line 7 (inside `add_executable(DynamicShapeMappingTest ...)`) and at line 57 (inside `add_llk_test_category(Mapping ...)`) with:

```cmake
    $<TARGET_OBJECTS:LLKTestFixtures>
```

- [ ] **Step 6: Reconfigure, rebuild, and confirm a single compile**

```bash
cmake -S . -B build -GNinja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja -C build -t clean LLKTestFixtures >/dev/null 2>&1 || true
ninja -C build LLKConversionTests LLKPerfTests LLKMappingTests DynamicShapeMappingTest
ninja -C build -t targets all | grep -c 'resource_regression_fixture\.cpp\.o'
```

Expected: `1`.

- [ ] **Step 7: Run the four affected binaries**

```bash
ctest --test-dir build --output-on-failure -L 'Conversion|Perf|Mapping' -j 1
ctest --test-dir build --output-on-failure -R '^DynamicShapeMappingTest$'
```

Expected: both pass. (`-j 1` avoids the known `ScheduleLoader` parallel flake; that test lives in `LLKTransformsTests` but the ordering rule is cheapest to apply here.) The second command is required because `DynamicShapeMappingTest` is registered without a `LABELS` property, so the label filter does not select it even though it is the fourth consumer of the fixture. Task 7 adds the label; until then it must be named explicitly.

- [ ] **Step 8: Run the whole suite**

```bash
ninja -C build check-llk
```

Expected: the same failure set as the pre-change baseline. On a non-x86_64 host that baseline already includes the three `selected-avx2` entries (`MappedAVX2Acceptance`, `GenericElementwiseAVX2Test`, `FrameworkPrimitivesAVX2Test`) failing with "selected target requires architecture 'x86_64' but host architecture is 'aarch64'" — those binaries are built with `LLK_REQUIRE_SELECTED_TARGET` and fail rather than skip when AVX2 is absent, which is their purpose. Compare against a baseline run, not against zero failures. On x86_64 CI the whole suite passes.

- [ ] **Step 9: Commit**

```bash
git add test/CMakeLists.txt test/Conversion/CMakeLists.txt test/Perf/CMakeLists.txt test/Mapping/CMakeLists.txt
git commit -m "refactor(build): compile the mapping regression fixture once"
```

---

### Task 5: Open PR 1

- [ ] **Step 1: Push and open**

```bash
git push -u origin refactor/test-target-aggregation-pass01
gh pr create --title "refactor(build): auto-register check-llk test deps and share fixtures" --body "$(cat <<'EOF'
Four build-hygiene fixes in `test/`:

- `check-llk` now derives its dependencies from the CMake directory graph
  instead of a hand-maintained list. Five binaries
  (`GenericElementwise{,AVX2}Test`, `FrameworkPrimitives{,AVX2}Test`,
  `DynamicShapeContractTest`) were missing from that list, so `ninja check-llk`
  could run without building them.
- `mapped_acceptance.cpp` was compiled into `LLKExecutionTests` as well as its
  own binary, so each of its 13 cases ran twice. The category binary no longer
  includes it.
- `resource_regression_fixture.cpp` is now an `OBJECT` library compiled once
  instead of four times (Conversion, Perf, Mapping, DynamicShapeMappingTest).
- The four Tuning binaries and three unlabelled Execution binaries now carry
  category labels.

Each of the 13 `MappedAcceptance.*` cases folds from its own CTest entry into the single named `MappedAcceptance` entry; every externally referenced CTest name is preserved.

Part of #266.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR 2 — `refactor(build): fold single-purpose test binaries into categories`

Branch: `refactor/test-target-aggregation-pass2`
Base: `main` after PR 1 merges.
Title: `refactor(build): fold single-purpose test binaries into category binaries`

## File Structure

| File | Responsibility |
|---|---|
| `test/Tuning/CMakeLists.txt` | Becomes a single `LLKTuningTests` with four named gate entries |
| `test/Mapping/CMakeLists.txt` | `DynamicShapeMappingTest` folds into `LLKMappingTests` behind an alias |
| `test/Execution/CMakeLists.txt` | Five standalone binaries fold into `LLKExecutionTests`; one alias kept |

**Alias rule used throughout this PR:** a former standalone binary keeps a CTest entry under its old name **only if** that name is referenced outside `test/CMakeLists.txt` (CI, a validation script, or a plan document). Names referenced only by the old `check-llk` list need no alias because PR 1 made that list automatic. Accepted cost: an aliased suite is registered twice (once per-case via discovery, once as the alias) and therefore runs twice.

---

### Task 6: Create `LLKTuningTests`

**Files:**
- Modify: `test/Tuning/CMakeLists.txt` (whole file)

**Interfaces:**
- Consumes: `add_llk_test_category()` from `test/CMakeLists.txt`.
- Produces: executable `LLKTuningTests`; CTest entries `FrameworkSourceTest` and `DynamicShapeContractTest` preserved as filtered gate entries; new per-case entries for every Tuning suite.

- [ ] **Step 1: Write the failing check**

```bash
ninja -C build -t targets all | grep -c 'bin/LLKTuningTests'
```

Expected before: `0`.

- [ ] **Step 2: Replace the file**

Write `test/Tuning/CMakeLists.txt` as:

```cmake
# One binary for the whole category. Two suites are also reachable under their
# historical binary names because the dynamic-shape plan cites them as gates.
set(_llk_tuning_sources
    workload_contract.cpp
    generic_source_candidates.cpp
    source_adapter_registry.cpp
    elementwise_source_adapter.cpp
    measurement_store.cpp
    dynamic_shape_contract.cpp)
if(LLK_BUILD_E2E_TESTS)
  list(APPEND _llk_tuning_sources
      mapped_tuning_session.cpp
      candidate_instantiation.cpp)
endif()

add_llk_test_category(Tuning
    SOURCES ${_llk_tuning_sources}
    LIBRARIES
        LLKWorkloadContracts
        LLKSourceCandidates
        LLKMappedTuning
        LLKMicroMapping
        LLKMicroToLinalg
        LLKShape
        LLKKernel
        LLKToMicro
        LLKDialect
        MicroDialect
        LLK::MLIR
    DEFINITIONS
        LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

# Historical gate names. See docs/superpowers/plans/2026-10-17-dynamic-shape-llk-microir.md.
add_test(NAME FrameworkSourceTest COMMAND LLKTuningTests
    --gtest_filter=WorkloadContract.*:FrameworkTest.*:BindingSchema.*:SourceAdapterRegistry.*:SourceSelection.*:ElementwiseSourceAdapter.*:ElementwiseSourceAdapterContext.*:MeasurementStoreTest.*)
add_test(NAME DynamicShapeContractTest COMMAND LLKTuningTests
    --gtest_filter=ShapeContract.*)
set_tests_properties(FrameworkSourceTest DynamicShapeContractTest
    PROPERTIES LABELS "Tuning")
```

Keep the remaining CLI `add_test` blocks (`MappedTuneCLI`, `MeasurementStoreCLI`, `GenericTuneCLI`, `PredictionValidationCLI`, `PredictionValidationMappedCLI`) exactly as they are, including their `if()` guards.

- [ ] **Step 3: Reconfigure and verify the entries**

```bash
cmake -S . -B build -GNinja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja -C build LLKTuningTests
ctest --test-dir build -N -L Tuning | grep -cE 'FrameworkSourceTest|DynamicShapeContractTest|ShapeContract\.|WorkloadContract\.'
```

Expected: at least `4` (the two aliases plus discovered cases).

- [ ] **Step 4: Run the category**

```bash
ctest --test-dir build --output-on-failure -L Tuning -j 1
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add test/Tuning/CMakeLists.txt
git commit -m "refactor(build): fold the four Tuning binaries into LLKTuningTests"
```

---

### Task 7: Fold `DynamicShapeMappingTest` into `LLKMappingTests`

**Files:**
- Modify: `test/Mapping/CMakeLists.txt:1-25`

**Interfaces:**
- Consumes: nothing new; `LLKMappingTests` already links `LLKMapping`, `LLKTargetMapping`, `LLKMicroMapping`, `LLKPerf`, `MicroDialect`, `LLKShape` is **not** linked today and must be added.
- Produces: `DynamicShapeMappingTest` CTest entry backed by `LLKMappingTests`.

- [ ] **Step 1: Write the failing check**

```bash
ctest --test-dir build -N | grep -cE 'Test +#.*: DynamicShapeMapping\.'
```

Expected before: `0` (the standalone binary registers one entry named `DynamicShapeMappingTest`). After: `9` discovered cases plus 1 alias.

- [ ] **Step 2: Add the source and library to the category target**

In `test/Mapping/CMakeLists.txt`, add `dynamic_shape_mapping.cpp` to the `SOURCES` list of `add_llk_test_category(Mapping ...)` (after `stable_hash.cpp`), and add `LLKShape` to its `LIBRARIES` list (after `LLKPerf`).

- [ ] **Step 3: Delete the standalone target**

Remove lines 1-25 of `test/Mapping/CMakeLists.txt` (the comment block and the `add_executable(DynamicShapeMappingTest ...)` block including its `add_test`).

- [ ] **Step 4: Re-add the gate name**

Append after the `add_llk_test_category(Mapping ...)` call:

```cmake
# Historical gate name from docs/superpowers/plans/2026-10-17-dynamic-shape-llk-microir.md.
add_test(NAME DynamicShapeMappingTest COMMAND LLKMappingTests
    --gtest_filter=DynamicShapeMapping.*)
set_tests_properties(DynamicShapeMappingTest PROPERTIES LABELS "Mapping")
```

- [ ] **Step 5: Reconfigure, build, and verify**

```bash
cmake -S . -B build -GNinja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja -C build LLKMappingTests
ctest --test-dir build -N | grep -cE 'Test +#.*: (DynamicShapeMappingTest|DynamicShapeMapping\.)'
```

Expected: `10` (9 discovered + 1 alias).

- [ ] **Step 6: Run the category**

```bash
ctest --test-dir build --output-on-failure -L Mapping -j 1
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add test/Mapping/CMakeLists.txt
git commit -m "refactor(build): fold DynamicShapeMappingTest into LLKMappingTests"
```

---

### Task 8: Fold five Execution binaries into `LLKExecutionTests`

**Files:**
- Modify: `test/Execution/CMakeLists.txt`

**Interfaces:**
- Consumes: `add_llk_test_category()`.
- Produces: `LLKExecutionTests` gains suites `DynamicMappedInvocation`, `MappedInvocation`, `MappedCodegenRequirements`, `MappedAllocationLifetime`, `SelectedGroupLowering`. CTest entry `MappedAllocationLifetimeTest` is preserved as an alias because `test/Execution/check_required_execution.py:8` requires it.

- [ ] **Step 1: Write the failing check**

```bash
ctest --test-dir build -N | grep -cE 'Test +#.*: (DynamicMappedInvocation|MappedInvocation|MappedCodegenRequirements|MappedAllocationLifetime|SelectedGroupLowering)\.'
```

Expected before: `0`. After: `34` (`13 + 7 + 5 + 4 + 5`).

- [ ] **Step 2: Add the five sources**

In `test/Execution/CMakeLists.txt`, append to the `SOURCES` list of `add_llk_test_category(Execution ...)`, after `${_llk_execution_e2e_sources}`:

```cmake
        dynamic_mapped_invocation.cpp
        mapped_invocation.cpp
        mapped_codegen_requirements.cpp
        mapped_allocation_lifetime.cpp
        selected_group_lowering.cpp
```

- [ ] **Step 3: Add the libraries they need**

Append to the `LIBRARIES` list of the same call:

```cmake
        LLKMappedCompilation
        LLKWorkloadContracts
```

`LLKMapping`, `LLKTargetMapping`, `MicroDialect`, `LLKDialect`, `LLKToMicro`, and `LLK::MLIR` are already present.

- [ ] **Step 4: Delete the five standalone targets**

Remove these blocks from `test/Execution/CMakeLists.txt`:

- `add_executable(MappedInvocationTest ...)` through its `add_test` (lines 82-85)
- `add_executable(MappedAllocationLifetimeTest ...)` through `set_tests_properties(...)` (lines 87-95)
- `add_executable(MappedCodegenRequirementsTest ...)` through its `add_test` (lines 97-102)
- `add_executable(SelectedGroupLoweringTest ...)` through its `add_test` (lines 157-160)
- `add_executable(DynamicMappedInvocationTest ...)` through `set_tests_properties(...)` (lines 186-193)

Keep `MappedAcceptance`, `MappedAVX2Acceptance`, `DynamicMappedAcceptance`, `DynamicMappedAVX2Acceptance`, `GenericElementwise*`, and `FrameworkPrimitives*` untouched; PR 3 handles those. `DynamicMappedAcceptance` is the reference half of the pair added by #265 — do not fold it into the category binary, because PR 3 collapses it with its selected half.

- [ ] **Step 5: Re-add the one required gate name**

Immediately after the `add_llk_test_category(Execution ...)` call, add:

```cmake
# check_required_execution.py requires this name in the CI JUnit evidence.
add_test(NAME MappedAllocationLifetimeTest COMMAND LLKExecutionTests
    --gtest_filter=MappedAllocationLifetime.*)
set_tests_properties(MappedAllocationLifetimeTest PROPERTIES LABELS "Execution")
```

- [ ] **Step 6: Reconfigure, build, and verify**

```bash
cmake -S . -B build -GNinja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja -C build LLKExecutionTests
ctest --test-dir build -N | grep -cE 'Test +#.*: (DynamicMappedInvocation|MappedInvocation|MappedCodegenRequirements|MappedAllocationLifetime|SelectedGroupLowering)\.'
ctest --test-dir build -N | grep -E 'Test +#.*: MappedAllocationLifetimeTest$'
```

Expected: `34`, then exactly one line naming `MappedAllocationLifetimeTest`.

- [ ] **Step 7: Run the category and the evidence parser**

```bash
ctest --test-dir build --output-on-failure -L Execution -j 1
```

Expected: all pass except documented JIT skips.

- [ ] **Step 8: Confirm the binaries are gone**

```bash
ninja -C build -t targets all | grep -cE 'bin/(DynamicMappedInvocationTest|MappedInvocationTest|MappedCodegenRequirementsTest|MappedAllocationLifetimeTest|SelectedGroupLoweringTest)'
```

Expected: `0`.

- [ ] **Step 9: Commit**

```bash
git add test/Execution/CMakeLists.txt
git commit -m "refactor(build): fold five single-purpose Execution binaries into LLKExecutionTests"
```

---

### Task 9: Full suite and PR 2

- [ ] **Step 1: Clean rebuild of the test targets and full run**

```bash
ninja -C build check-llk
```

Expected: all pass; total registered count is the baseline minus the removed standalone entries plus the new per-case entries.

- [ ] **Step 2: Confirm the binary count**

```bash
ninja -C build -t targets all | grep -cE 'bin/LLK[A-Za-z]*Tests$'
```

Expected: `11` (the ten previous categories plus `LLKTuningTests`).

- [ ] **Step 3: Push and open**

```bash
git push -u origin refactor/test-target-aggregation-pass2
gh pr create --title "refactor(build): fold single-purpose test binaries into category binaries" --body "$(cat <<'EOF'
Ten single-purpose binaries become part of their category binary:

- the four Tuning binaries -> `LLKTuningTests` (new; Tuning previously had no
  category binary at all)
- `DynamicShapeMappingTest` -> `LLKMappingTests`
- `DynamicMappedInvocationTest`, `MappedInvocationTest`,
  `MappedCodegenRequirementsTest`, `MappedAllocationLifetimeTest`,
  `SelectedGroupLoweringTest` -> `LLKExecutionTests`

Every CTest name referenced outside `test/CMakeLists.txt` is preserved as a
filtered gate entry: `FrameworkSourceTest`, `DynamicShapeContractTest`,
`DynamicShapeMappingTest`, and `MappedAllocationLifetimeTest` (required by
`check_required_execution.py`). Suites without such a reference now appear as
per-case entries via `gtest_discover_tests`.

Accepted trade-off: an aliased suite is registered twice and runs twice under
`ctest`. That is 4 duplicated cases for `MappedAllocationLifetime` and 9 for
`DynamicShapeMapping`.

Part of #266.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR 3 — `refactor(build): collapse the reference/selected test binaries`

Branch: `refactor/test-target-aggregation-pass3`
Base: `main` after PR 2 merges.
Title: `refactor(build): collapse reference and selected-target test binaries`

**Why this changes test sources:** `LLK_REQUIRE_SELECTED_TARGET` is not just a default-backend switch — `mapped_acceptance.cpp:264,462,660,1106,1130` use it to decide skip-vs-fail and which `executionIdentity` substring to expect. The runtime flag `--require-selected-target` is already parsed by every one of these `main`s, so the switch becomes a process-global set from `argv`.

---

### Task 10: Make `MappedAcceptance` a single binary

**Files:**
- Modify: `test/Execution/mapped_acceptance.cpp:264,462,660,1106,1130,1170-1200`
- Modify: `test/Execution/CMakeLists.txt:50-80`
- Modify: `test/Execution/check_selected_target_flag.py`
- Modify: `.github/workflows/ci.yml:123`

**Interfaces:**
- Consumes: nothing.
- Produces: one executable `MappedAcceptance` at the build root; CTest entries `MappedAcceptance` and `MappedAVX2Acceptance` both run it, the second with `--require-selected-target`.

- [ ] **Step 1: Write the failing check**

```bash
ninja -C build -t targets all | grep -c 'CXX_EXECUTABLE_LINKER__MappedAVX2Acceptance_'
```

Expected before: `1`. After: `0`. Match the linker rule, not the bare name: the bare name also appears in `cmake_object_order_depends_target_*` rules and in `.o` targets.

- [ ] **Step 2: Add the runtime switch in `mapped_acceptance.cpp`**

Before the first use (above `compileChain`), add:

```cpp
namespace {
/// Set from --require-selected-target. Selecting the backend at run time keeps
/// one binary serving both the reference and the required-AVX2 gates.
bool requireSelectedTarget = false;
} // namespace
```

Replace the five `#ifdef LLK_REQUIRE_SELECTED_TARGET` / `#else` / `#endif` blocks with runtime conditionals:

At line 462 (default argument of `compileChain`):

```cpp
             llk::MappedBackend backend =
                 requireSelectedTarget ? llk::MappedBackend::SelectedTarget
                                       : llk::MappedBackend::Reference,
```

At line 660:

```cpp
  options.backend =
      (requireSelectedTarget && !(requireTransform || vectorWidth == 4))
          ? llk::MappedBackend::SelectedTarget
          : llk::MappedBackend::Reference;
```

At lines 264, 1106, 1130 (the skip-vs-fail guards):

```cpp
  if (!selectedAvx2Available()) {
    if (requireSelectedTarget) {
      if (avx2DisabledByTestOverride())
        FAIL() << "test override disabled required selected AVX2 execution";
      else
        FAIL() << "required selected AVX2 execution needs an x86_64 AVX2 host";
    } else {
      GTEST_SKIP() << "selected AVX2 invocation requires an x86_64 host";
    }
  }
```

At the `executionIdentity` expectation near line 660:

```cpp
  EXPECT_NE((*executable)->executionIdentity().find(
                requireSelectedTarget ? "backend=selected-target"
                                      : "backend=reference"),
            llvm::StringRef::npos);
```

- [ ] **Step 3: Simplify `main`**

Replace the flag-parsing block (lines 1170-1196) with:

```cpp
int main(int argc, char **argv) {
  for (int index = 1; index < argc;) {
    if (std::strcmp(argv[index], "--require-selected-target") == 0) {
      requireSelectedTarget = true;
      for (int move = index; move + 1 < argc; ++move)
        argv[move] = argv[move + 1];
      --argc;
      continue;
    }
    ++index;
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```

- [ ] **Step 4: Delete the second target from `test/Execution/CMakeLists.txt`**

Replace the `MappedAVX2Acceptance` block (lines 67-75) and the `MappedAcceptance` block (lines 56-65) with one target and two entries:

```cmake
  add_executable(MappedAcceptance mapped_acceptance.cpp)
  target_link_libraries(MappedAcceptance PRIVATE ${_llk_mapped_libs})
  target_compile_definitions(MappedAcceptance PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  set_target_properties(MappedAcceptance PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME MappedAcceptance COMMAND MappedAcceptance
      --gtest_filter=MappedAcceptance.*)
  set_tests_properties(MappedAcceptance PROPERTIES
      LABELS "mapped-reference"
      SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")
  # Same binary, required-selected-target mode: a missing AVX2 lane is a
  # failure, not a skip, so CI evidence cannot silently pass on the wrong host.
  add_test(NAME MappedAVX2Acceptance COMMAND MappedAcceptance
      --require-selected-target --gtest_filter=MappedAcceptance.*)
  set_tests_properties(MappedAVX2Acceptance PROPERTIES LABELS "selected-avx2")
```

- [ ] **Step 5: Rewrite the flag contract test**

Replace `test/Execution/check_selected_target_flag.py` with:

```python
#!/usr/bin/env python3
"""Exercise the required selected-target command-line contract."""

import os
import subprocess
import sys


def run(binary, *arguments, env=None):
    environment = os.environ.copy()
    environment.update(env or {})
    return subprocess.run(
        [binary, *arguments], check=False, capture_output=True, text=True,
        env=environment
    )


def check_pair(binary, suite, gtest_filter):
    reference = run(binary, "--gtest_list_tests",
                    env={"LLK_TEST_DISABLE_AVX2": "1"})
    assert reference.returncode == 0, reference.stderr
    assert suite in reference.stdout

    required = run(binary, "--require-selected-target", "--gtest_list_tests")
    assert required.returncode == 0, required.stderr

    unsupported = run(
        binary,
        "--require-selected-target",
        f"--gtest_filter={gtest_filter}",
        env={"LLK_TEST_DISABLE_AVX2": "1"},
    )
    assert unsupported.returncode != 0, (
        "required selected execution accepted a disabled AVX2 feature"
    )
    assert "test override disabled required selected AVX2 execution" in (
        unsupported.stdout + unsupported.stderr
    )


def main():
    arguments = sys.argv[1:]
    assert len(arguments) in (1, 2), "expected one or two collapsed binaries"
    check_pair(
        arguments[0],
        "MappedAcceptance.",
        "MappedAcceptance.SelectedAvx2BackendExecutesNumerically",
    )
    if len(arguments) == 2:
        check_pair(
            arguments[1],
            "DynamicMappedAcceptance.",
            "DynamicMappedAcceptance.CompilesOnceAndExecutesUnrelatedMatmulShapes",
        )


if __name__ == "__main__":
    main()
```

Each argument is now **one** binary serving both modes. The second `check_pair` call is dormant until Task 13 collapses the `DynamicMappedAcceptance` pair; passing that pair here before then would fail, because its reference half rejects `--require-selected-target` by design.

- [ ] **Step 6: Update the CMake invocation of the contract test**

In `test/Execution/CMakeLists.txt`, change the `SelectedTargetFlagContract` test to pass one path:

```cmake
  add_test(NAME SelectedTargetFlagContract
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/check_selected_target_flag.py
          $<TARGET_FILE:MappedAcceptance>)
  set_tests_properties(SelectedTargetFlagContract PROPERTIES LABELS "selected-avx2")
```

Before #265 this test passed four paths (two pairs). Task 10 drops it to one binary because the `DynamicMappedAcceptance` pair has not been collapsed yet and its reference half rejects the flag. Task 13 restores the second argument. The gap exists only between those two commits in this PR, and the merged state checks both pairs.

- [ ] **Step 7: Update CI**

In `.github/workflows/ci.yml`, in the step "Run selected AVX2 acceptance", change the first invocation `./MappedAVX2Acceptance --require-selected-target \` to:

```yaml
          ./MappedAcceptance --require-selected-target \
```

Leave the second invocation, `./DynamicMappedAVX2Acceptance --require-selected-target \`, untouched — Task 13 changes it.

- [ ] **Step 8: Verify**

```bash
cmake -S . -B build -GNinja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja -C build -t targets all | grep -c 'CXX_EXECUTABLE_LINKER__MappedAVX2Acceptance_'
ninja -C build MappedAcceptance
ctest --test-dir build --output-on-failure -R 'MappedAcceptance|SelectedTargetFlagContract'
```

Expected: `0`, then PASS for both entries and the contract test. Do not grep for the bare name `MappedAVX2Acceptance`: it also matches `DynamicMappedAVX2Acceptance`, `cmake_object_order_depends_target_*` rules, and `.o` targets.

- [ ] **Step 9: Commit**

```bash
git add test/Execution/mapped_acceptance.cpp test/Execution/CMakeLists.txt \
        test/Execution/check_selected_target_flag.py .github/workflows/ci.yml
git commit -m "refactor(build): serve both acceptance gates from one binary"
```

---

### Task 11: Collapse the `GenericElementwise` pair

**Files:**
- Modify: `test/Execution/generic_elementwise.cpp:621,669,679-687`
- Modify: `test/Execution/CMakeLists.txt:104-128`

**Interfaces:**
- Consumes: the runtime-switch pattern from Task 10.
- Produces: one `GenericElementwiseTest` binary; CTest entries `GenericElementwiseTest` (all cases) and `GenericElementwiseAVX2Test` (`--require-selected-target --gtest_filter=GenericElementwise.SelectedAvx2*`).

- [ ] **Step 1: Write the failing check**

```bash
ninja -C build -t targets all | grep -c 'CXX_EXECUTABLE_LINKER__GenericElementwiseAVX2Test_'
```

Expected before: `1`. After: `0`.

- [ ] **Step 2: Apply the runtime switch**

Replace `#ifdef LLK_REQUIRE_SELECTED_TARGET` guards with runtime checks on a file-scope `requireSelectedTarget` global, exactly as in Task 10 Step 2. Then replace the `main` flag handling with:

```cpp
int main(int argc, char **argv) {
  for (int index = 1; index < argc;) {
    if (std::strcmp(argv[index], "--require-selected-target") == 0) {
      requireSelectedTarget = true;
      for (int move = index; move + 1 < argc; ++move)
        argv[move] = argv[move + 1];
      --argc;
      continue;
    }
    ++index;
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```

- [ ] **Step 3: Replace the two CMake targets with one**

In `test/Execution/CMakeLists.txt`, replace the `GenericElementwiseTest` and `GenericElementwiseAVX2Test` blocks with:

```cmake
  add_executable(GenericElementwiseTest generic_elementwise.cpp)
  target_link_libraries(GenericElementwiseTest PRIVATE
      ${_llk_generic_elementwise_libs})
  target_compile_definitions(GenericElementwiseTest PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME GenericElementwiseTest COMMAND GenericElementwiseTest)
  set_tests_properties(GenericElementwiseTest PROPERTIES LABELS "mapped-reference")
  add_test(NAME GenericElementwiseAVX2Test COMMAND GenericElementwiseTest
      --require-selected-target
      --gtest_filter=GenericElementwise.SelectedAvx2*)
  set_tests_properties(GenericElementwiseAVX2Test PROPERTIES LABELS "selected-avx2")
```

- [ ] **Step 4: Verify**

```bash
cmake -S . -B build -GNinja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja -C build GenericElementwiseTest
ctest --test-dir build --output-on-failure -R 'GenericElementwise'
```

Expected: both entries pass (the AVX2 one skips its cases only if AVX2 is genuinely absent; it does not fail in reference mode).

- [ ] **Step 5: Commit**

```bash
git add test/Execution/generic_elementwise.cpp test/Execution/CMakeLists.txt
git commit -m "refactor(build): serve both generic elementwise gates from one binary"
```

---

### Task 12: Collapse the `FrameworkPrimitives` pair

**Files:**
- Modify: `test/Execution/framework_primitives.cpp:442,488,498-506`
- Modify: `test/Execution/CMakeLists.txt:130-155`

**Interfaces:**
- Consumes: the runtime-switch pattern from Task 10.
- Produces: one `FrameworkPrimitivesTest` binary; CTest entries `FrameworkPrimitivesTest` and `FrameworkPrimitivesAVX2Test`.

- [ ] **Step 1: Write the failing check**

```bash
ninja -C build -t targets all | grep -c 'CXX_EXECUTABLE_LINKER__FrameworkPrimitivesAVX2Test_'
```

Expected before: `1`. After: `0`.

- [ ] **Step 2: Apply the runtime switch**

Same transformation as Task 10 Step 2 and Task 11 Step 2: a file-scope `requireSelectedTarget` global, runtime guards replacing `#ifdef LLK_REQUIRE_SELECTED_TARGET`, and the simplified `main`.

- [ ] **Step 3: Replace the two CMake targets with one**

In `test/Execution/CMakeLists.txt`, replace both blocks with:

```cmake
  add_executable(FrameworkPrimitivesTest framework_primitives.cpp)
  target_link_libraries(FrameworkPrimitivesTest PRIVATE
      ${_llk_framework_primitives_libs})
  target_compile_definitions(FrameworkPrimitivesTest PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  target_include_directories(FrameworkPrimitivesTest PRIVATE
      ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping)
  add_test(NAME FrameworkPrimitivesTest COMMAND FrameworkPrimitivesTest)
  set_tests_properties(FrameworkPrimitivesTest PROPERTIES LABELS "mapped-reference")
  add_test(NAME FrameworkPrimitivesAVX2Test COMMAND FrameworkPrimitivesTest
      --require-selected-target
      --gtest_filter=FrameworkPrimitives.SelectedAvx2*)
  set_tests_properties(FrameworkPrimitivesAVX2Test PROPERTIES LABELS "selected-avx2")
```

- [ ] **Step 4: Verify**

```bash
cmake -S . -B build -GNinja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja -C build FrameworkPrimitivesTest
ctest --test-dir build --output-on-failure -R 'FrameworkPrimitives'
```

Expected: both entries pass.

- [ ] **Step 5: Confirm no stale macro references remain**

```bash
grep -rn 'LLK_REQUIRE_SELECTED_TARGET' test/ || echo "none"
```

Expected: `none`.

- [ ] **Step 6: Commit**

```bash
git add test/Execution/framework_primitives.cpp test/Execution/CMakeLists.txt
git commit -m "refactor(build): serve both framework primitives gates from one binary"
```

---

### Task 13: Collapse the `DynamicMappedAcceptance` pair

**Files:**
- Modify: `test/Execution/dynamic_mapped_acceptance.cpp` (macro sites at 127, 162, 1269, and `main` at 1623)
- Modify: `test/Execution/CMakeLists.txt:170-193`
- Modify: `test/Execution/check_selected_target_flag.py` (the second `check_pair` call becomes live)
- Modify: `.github/workflows/ci.yml` (the second invocation in "Run selected AVX2 acceptance")

**Interfaces:**
- Consumes: the runtime switch established in Task 10 and the `check_pair(binary, suite, gtest_filter)` signature.
- Produces: one `DynamicMappedAcceptance` binary; CTest entries `DynamicMappedAcceptance` and `DynamicMappedAVX2Acceptance`.

`#265` added this pair after the plan was written, so it is the same transformation as Task 10 with one structural difference: the two cases in suite `DynamicMappedSelected` are declared **inside** `#ifdef LLK_REQUIRE_SELECTED_TARGET`, so in the reference binary they do not exist at all. After the collapse they always exist and must skip instead of fail when the flag is absent.

- [ ] **Step 1: Write the failing check**

```bash
ninja -C build -t targets all | grep -c 'CXX_EXECUTABLE_LINKER__DynamicMappedAVX2Acceptance_'
```

Expected before: `1`. After: `0`.

- [ ] **Step 2: Replace the default-backend constant**

At `test/Execution/dynamic_mapped_acceptance.cpp:162-166` the backend is chosen at compile time:

```cpp
#ifdef LLK_REQUIRE_SELECTED_TARGET
constexpr llk::MappedBackend kDefaultBackend =
    llk::MappedBackend::SelectedTarget;
#else
constexpr llk::MappedBackend kDefaultBackend = llk::MappedBackend::Reference;
#endif
```

Replace it with a mutable global set from `argv` during `main`, declared before its first use:

```cpp
namespace {
/// Set from --require-selected-target. One binary serves the reference and the
/// required-AVX2 gates.
bool requireSelectedTarget = false;
} // namespace

llk::MappedBackend defaultBackend() {
  return requireSelectedTarget ? llk::MappedBackend::SelectedTarget
                               : llk::MappedBackend::Reference;
}
```

Then replace every read of `kDefaultBackend` with a call to `defaultBackend()`.

- [ ] **Step 3: Make the helper functions unconditional**

Delete the `#ifdef LLK_REQUIRE_SELECTED_TARGET` / `#endif` pair around `avx2DisabledByTestOverride()` and the `selectedAvx2Available()` block at lines 127-160. Both are host feature queries, not selected-mode behaviour, and the reference binary needs them to decide skip-vs-run once the selected cases exist in it.

- [ ] **Step 4: Make the `DynamicMappedSelected` suite unconditional**

Remove the `#ifdef LLK_REQUIRE_SELECTED_TARGET` / `#endif` at lines 1269-1290 and change each of the two cases to skip instead of fail when selected mode is not requested:

```cpp
TEST(DynamicMappedSelected, SelectedAvx2ExecutesCompileOnceDynamicAcceptance) {
  if (!selectedAvx2Available()) {
    if (requireSelectedTarget) {
      if (avx2DisabledByTestOverride())
        FAIL() << "test override disabled required selected AVX2 execution";
      else
        FAIL() << "required selected AVX2 execution needs an x86_64 AVX2 host";
    } else {
      GTEST_SKIP() << "selected AVX2 invocation requires an x86_64 host";
    }
  }
  ...
```

Apply the same guard shape to the second `DynamicMappedSelected` case.

- [ ] **Step 5: Simplify `main`**

In `main` (line 1623), keep the `--require-selected-target` parsing loop but delete both compile-time branches that follow it. The runtime contract becomes:

```cpp
int main(int argc, char **argv) {
  for (int index = 1; index < argc;) {
    if (std::strcmp(argv[index], "--require-selected-target") == 0) {
      requireSelectedTarget = true;
      for (int move = index; move + 1 < argc; ++move)
        argv[move] = argv[move + 1];
      --argc;
      continue;
    }
    ++index;
  }

  // A required selected run on a host that cannot execute AVX2 is a failure,
  // never a skip. Listing tests runs nothing, so it stays host-independent.
  if (requireSelectedTarget) {
    bool listingTests = false;
    for (int index = 1; index < argc; ++index)
      listingTests |= std::strcmp(argv[index], "--gtest_list_tests") == 0;
    if (!listingTests && !selectedAvx2Available()) {
      if (avx2DisabledByTestOverride())
        std::fprintf(stderr,
                     "test override disabled required selected AVX2 execution\n");
      else
        std::fprintf(
            stderr,
            "required selected AVX2 execution needs an x86_64 AVX2 host\n");
      return 1;
    }
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```

- [ ] **Step 6: Replace the two CMake targets with one**

In `test/Execution/CMakeLists.txt`, replace the two blocks at lines 170-193 with:

```cmake
  add_executable(DynamicMappedAcceptance dynamic_mapped_acceptance.cpp)
  target_link_libraries(DynamicMappedAcceptance PRIVATE
      ${_llk_dynamic_acceptance_libs})
  target_compile_definitions(DynamicMappedAcceptance PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  set_target_properties(DynamicMappedAcceptance PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME DynamicMappedAcceptance COMMAND DynamicMappedAcceptance
      --gtest_filter=DynamicMappedAcceptance.*)
  set_tests_properties(DynamicMappedAcceptance PROPERTIES
      LABELS "mapped-reference"
      SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")
  add_test(NAME DynamicMappedAVX2Acceptance COMMAND DynamicMappedAcceptance
      --require-selected-target --gtest_filter=DynamicMapped*.*)
  set_tests_properties(DynamicMappedAVX2Acceptance PROPERTIES
      LABELS "selected-avx2")
```

The reference entry now filters `DynamicMappedAcceptance.*` rather than running the whole binary, because the binary always contains the `DynamicMappedSelected` suite; those cases would otherwise run in reference mode where the previous reference binary did not have them at all.

- [ ] **Step 7: Restore the second pair in the contract test**

In `test/Execution/CMakeLists.txt`, change the `SelectedTargetFlagContract` invocation back to two arguments:

```cmake
  add_test(NAME SelectedTargetFlagContract
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/check_selected_target_flag.py
          $<TARGET_FILE:MappedAcceptance> $<TARGET_FILE:DynamicMappedAcceptance>)
```

The `check_pair` call for the second argument is already present in the script from Task 10.

- [ ] **Step 8: Update CI**

In `.github/workflows/ci.yml`, in the step "Run selected AVX2 acceptance", change the second invocation from `./DynamicMappedAVX2Acceptance --require-selected-target \` to:

```yaml
          ./DynamicMappedAcceptance --require-selected-target \
```

- [ ] **Step 9: Verify**

```bash
cmake -S . -B build -GNinja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
ninja -C build DynamicMappedAcceptance
ninja -C build -t targets all | grep -c 'CXX_EXECUTABLE_LINKER__DynamicMappedAVX2Acceptance_'
ctest --test-dir build --output-on-failure -R 'DynamicMapped|SelectedTargetFlagContract'
```

Expected: `0`, then PASS for `DynamicMappedAcceptance`, Skipped or Passed for `DynamicMappedAVX2Acceptance` depending on host (exit 0 either way is not expected on aarch64 — the required entry fails by design there), and PASS for `SelectedTargetFlagContract`.

- [ ] **Step 10: Confirm no macro references remain**

```bash
grep -rn 'LLK_REQUIRE_SELECTED_TARGET' test/ || echo "none"
```

Expected: `none`.

- [ ] **Step 11: Commit**

```bash
git add test/Execution/dynamic_mapped_acceptance.cpp test/Execution/CMakeLists.txt \
        test/Execution/check_selected_target_flag.py .github/workflows/ci.yml
git commit -m "refactor(build): serve both dynamic acceptance gates from one binary"
```

---

### Task 14: Full suite and PR 3

- [ ] **Step 1: Full clean verification**

```bash
ninja -C build check-llk
```

Expected: all pass, documented JIT skips unchanged.

- [ ] **Step 2: Confirm the final binary count**

```bash
ninja -C build -t targets all | grep -cE 'CXX_EXECUTABLE_LINKER__([A-Za-z0-9]*Tests|[A-Za-z0-9]*Test|MappedAcceptance)_'
```

Expected: `16` with `LLK_BUILD_PYTHON_CAPI=ON` — `LLKBenchmarkTests`, `LLKCAPITests`, `LLKConversionTests`, `LLKExecutionTests`, `LLKFrontendTests`, `LLKMachineTests`, `LLKMappingTests`, `LLKNumericalTests`, `LLKPerfTests`, `LLKTransformsTests`, `LLKTuningTests`, `LLKCAPIFrontendTests`, `GenericElementwiseTest`, `FrameworkPrimitivesTest`, `MappedAcceptance`, `DynamicMappedAcceptance`. With `LLK_BUILD_PYTHON_CAPI=OFF` (the default) the CAPI pair is absent and the count is `14`.

The count went up by one relative to the original plan because `#265` added a fourth reference/selected pair; the reduction is still 29 → 16 (29 → 14 with CAPI off).

- [ ] **Step 3: Confirm the evidence contract still holds**

```bash
ctest --test-dir build --output-junit /tmp/ctest-results.xml >/dev/null
python3 test/Execution/check_required_execution.py /tmp/ctest-results.xml
```

Expected: `required selected execution evidence passed`.

- [ ] **Step 4: Push and open**

```bash
git push -u origin refactor/test-target-aggregation-pass3
gh pr create --title "refactor(build): collapse reference and selected-target test binaries" --body "$(cat <<'EOF'
`LLK_REQUIRE_SELECTED_TARGET` selected the backend at compile time, so each
reference/selected pair was built twice from the same source:
`MappedAcceptance`/`MappedAVX2Acceptance`, `GenericElementwise{,AVX2}Test`,
`FrameworkPrimitives{,AVX2}Test`, and — added by #265 —
`DynamicMappedAcceptance`/`DynamicMappedAVX2Acceptance`.

The binaries already parsed `--require-selected-target`; the macro only supplied
a default and chose skip-vs-fail. It is now a process global set from `argv`, so
one binary serves both CTest entries per pair.

CI runs `./MappedAcceptance` and `./DynamicMappedAcceptance` with
`--require-selected-target` instead of the two `*AVX2Acceptance` binaries.
`check_selected_target_flag.py` now proves the required mode enforces AVX2 by
running each collapsed binary twice with `LLK_TEST_DISABLE_AVX2=1`.

CTest names are unchanged: `MappedAcceptance`, `MappedAVX2Acceptance`,
`GenericElementwiseTest`, `GenericElementwiseAVX2Test`,
`FrameworkPrimitivesTest`, `FrameworkPrimitivesAVX2Test`.

Fixes #266.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review

**Spec coverage.** F1 → Task 1. F2 → Task 2. F3 → Tasks 3 and 6. F4 → Task 4. F5 → Tasks 10-12. F6 → Tasks 7 and 8. F7 (mixed `python3` versus `find_package(Python3)`) is deliberately **out of scope**: it is unrelated to aggregation and would add churn to three PRs. Recorded here so it is not silently dropped.

**Placeholder scan.** No `TBD`, no "similar to Task N" without code. Task 11 Step 2 and Task 12 Step 2 reference Task 10 Step 2's transformation, but the exact `main` bodies are reproduced in each task so each can be read standalone; the guarded sites differ per file and are listed by line number.

**Type consistency.** `requireSelectedTarget` is used consistently as a lowercase file-scope global (not `gRequireSelectedTarget`) because the existing `main` has a local `bool requireSelectedTarget` in the standalone binaries; hoisting the same name avoids two spellings of one concept. In all three files the local declaration in `main` is removed in favor of the global.

**Risk not covered by a task.** PR 3's CI edit cannot be validated locally. Mitigation: Task 13 Step 3 reproduces the CI evidence check locally against the JUnit file, and the `SelectedTargetFlagContract` CTest entry exercises the flag both ways.
