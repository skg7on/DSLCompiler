# Framework PR review fixes implementation plan

> **For agentic workers:** Execute the bounded review fixes using test-driven development and isolated worktrees. Independent C1/C3/evidence work may run in parallel; integrate forward without rewriting the PR stack.

**Goal:** Fix all findings and additional coverage gaps in the 2026-10-18 review of PRs #205–#208.
**Architecture:** Keep artifact validation above generic Mapping/Machine. Validate target operation coverage in every shared mapped compilation path. Bind receipt claims and evidence rows to the artifacts actually validated. Preserve separate reference, selected, preparation and invocation facts.
**Tech Stack:** C++20/MLIR, CMake/Ninja, GTest/CTest, standard Python evidence scripts.
**Spec:** docs/superpowers/plans/2026-10-12-microir-framework-conformance.md and docs/superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md.
**Review:** build/2026-10-18-pr205-208-review.md (original review artifact).

## Global constraints
- All implementation writes use existing isolated worktrees under .claude/worktrees/.
- Generic machine and mapping libraries must not branch on target names or interpret opaque emitter/bundle names as policy.
- LLKMapping remains independent of performance, runtime and JIT libraries.
- A status is Supported only after the applicable stage succeeds for this artifact.
- Portable reference execution is separately identified.
- Unsupported rank-one invocation must remain explicit; never skip required selected tests on an unsupported host and call them passed.
- Add regression tests first and observe their intended failure. Do not weaken existing completeness checks.
- Preserve remote branch history. C1's local worktree contains a C2 cherry-pick absent from PR205: transplant only its fix commits onto the exact remote C1 head.

## Review focus
- Changed source or topology with old successful evidence must not get support credited to the changed artifact (task 1).
- Selected-only lowering must not establish reference support; preparation must not imply a measured invocation (tasks 1/3).
- bf16 reduction/multiply must be rejected through CLI, explicit plans and replay as well as evaluator-based search (task 2).
- Missing rows, duplicate rows, zero backend realization, incomplete report fields, mode/identity mismatches must fail the evidence gate (task 3).
- A real added logical view must preserve aliases/dependencies and add zero traffic/storage (task 4).

## Task 1: C1 receipts and runtime distinctions
Files: FrameworkCapabilities.h/.cpp, MappedCompilation.h/.cpp as necessary for verified provenance; framework_capabilities.cpp; relevant CLI/tuning tests and documentation.
1. Add real-artifact regressions using add vs mul sources and a valid changed target, with old plan/analysis/compilation evidence; assert stale evidence does not establish supported stages.
2. Add selected-only and reference-only lowering status assertions; test absent contract/preparation/host/invocation distinctions without inventing successful execution.
3. Run intended red, implement provenance checks using original-source/bound metadata correctly (bound extraction is not the original graph), and actual backend-mode stage evidence.
4. Run receipt, compilation, selected-group and generic CLI tests; commit only C1 fixes. Return commit SHAs and test evidence.

## Task 2: C3 coverage admission and build configuration
Files: shared mapped compilation/validation path, coverage helper, CompletePlanEvaluation as needed; focused execution/CLI/replay tests; Conversion CMake and configuration tests.
1. Add regressions demonstrating bf16 reduction/multiply rejection through ordinary CLI, explicit-plan mapped-micro, lowered compilation and frozen replay.
2. Retain the evaluator's pre-ranking hook; apply read-only operation validation at the common mapped compilation boundary before early returns.
3. Add/run a tools-disabled + E2E-disabled configuration regression; guard tool-dependent tests and later dependent manifest registration.
4. Run intended red/green and affected mapping/compilation/CLI tests; commit C3 fixes. Evidence producer changes are task 3's ownership.

## Task 3: C3 evidence truthfulness and C4 validators/docs
Files: framework_selected_evidence.py, framework_manifest_validator.py, framework_boundary_scan.py, their regression tests/registrations, acceptance reproduction documentation. If structured report parsing requires a report serialization seam, keep it generic and dependency-free in CI.
1. Add negative regressions for omitted/duplicate required rows, zero selected verification/backend realization, actual complete=false disguised by comments, wrong mode/identity, unsupported rows without reasons, and actual opaque AVX2/operator tokens.
2. Fix producer: reference support only with reference evidence; no numeric verification claim without the corresponding run; classify composed rank-one reduction invocation explicitly unsupported with its actual eight-result semantics.
3. Fix validator: enforce expected unique cases and artifact/mode/plan association, positive selected realization with zero fallback, and parse actual report fields rather than substrings. Gracefully reject malformed input.
4. Fix boundary token coverage and add positive/negative scan cases.
5. Correct selected execution command to label selection with --no-tests=error. Update changed evidence/schema descriptions.
6. Run intended red/green, actual producer+validator+boundary tests and docs checks; commit separated C3/C4 fix commits when practical. Coordinate producer/schema changes in this task so task 2 does not touch the producer.

## Task 4: C2 logical view, integration and completion
Files: resource_regression_fixture.cpp, framework_conformance.cpp; integration ledger and acceptance evidence.
1. Add no-view and added-view source fixtures between producer and consumer. Preserve semantics and zero additional movement/allocation; include literal traffic/peak assertions and alias/dependency checks.
2. Run intended regression/green and C2 resource tests; commit C2 fix.
3. Transplant C1-only changes onto its exact remote head; propagate C2 into C3, C1 and C3 into C4 by merges, resolving ownership overlaps without dropping tests.
4. Run affected suites and full check-llk on integrated C4; record actual counts and ARM-host selected failures separately. Run strict documentation checks and tools-disabled configuration.
5. Independently review the fixes; repair any material regression. Push the four PR branches as fast-forwards, inspect CI eligibility/status, and record remaining external host/CI requirements honestly.

