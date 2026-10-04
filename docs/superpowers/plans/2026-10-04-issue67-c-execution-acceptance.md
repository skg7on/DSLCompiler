# Issue #67 Stage C — Target Execution and Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute selected target implementations numerically and provide complete evidence for issue #67 acceptance.

**Architecture:** Introduce an explicit kernel ABI, target-owned lowering and a shared mapped compilation driver. Use caller-owned output descriptors, preserve selected resources/layouts through lowering, and integrate the same driver with compiler/tuner acceptance fixtures.

**Tech Stack:** C++20, LLVM/MLIR, LLKMap, GoogleTest, FileCheck, CMake/Ninja/CTest.

**Spec:** [Normative design](../specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md), [master contracts/dependencies](2026-10-04-issue67-improvement-plan.md), [review evidence](../../reviews/2026-10-04-issue67-pr111-status.md).

## Global Constraints

Apply every global constraint and interface decision in the master plan, including isolated worktrees, canonical Micro-IR, target-owned policy, LLVM 22/24 portability, bounded-solver soundness and legacy compatibility. Proposed APIs below are implemented by their owning task before dependent tasks use them.

## Review Focus

Apply the master's five review-focus cases. Each owning task specifies the negative input, positive control and verification command. Source snippets are implementation/test contracts; adapt includes and registration to existing file conventions without weakening the assertions.

## File responsibilities

Kernel signature semantics belong to Micro ODS/verifiers and LLKToMicro export. Target bundle interpretation belongs to AVX2 target files, not the generic bridge. New MappedCompilation owns the clone/verify/materialize/lower pipeline; MappedExecutable owns runtime invocation and JIT lifetime. The tuner calls these interfaces rather than constructing a second mapping path.

### C1: Declare kernel inputs and results explicitly

**Dependencies:** B1; A8

**Files:**
- Modify: include/LLK/Dialect/Micro/MicroOps.td
- Modify: lib/Dialect/Micro/MicroOps.cpp
- Modify: lib/Conversion/LLKToMicro/LLKToMicro.cpp
- Modify: lib/Conversion/MicroToLinalg/MicroToLinalg.cpp
- Modify: docs/design/m9-micro-ir-core-concepts.md
- Test: test/Dialect/Micro/tile_ops.mlir
- Test: test/Dialect/Micro/tile_ops_invalid.mlir
- Test: test/Execution/micro_kernel_execution.cpp

**Interfaces:** KernelOp acquires function_type, typed block arguments/results and typed yield verification. Public helpers expose its function signature. Legacy argumentless syntax remains parseable for analysis; no tensor.empty is guessed to be an external input.

- [ ] **Step 1 — add the regression and legal controls.**

Add this proposed explicit ABI fixture:
```mlir
micro.kernel @add(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  micro.yield %r : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
}
```
The logical yielded tile converts to the declared tensor result under the kernel's explicit interface. Test input/result count, dtype/shape mismatch, multiple outputs, internal tensor.empty allocations and invalid captured values.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target llk-opt MicroKernelExecution -j 6
ctest --test-dir build -R 'MicroDialect|LLKToMicro' --output-on-failure
build/MicroKernelExecution
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Use a function-style region/signature contract; verify block argument types and result logical shape/dtype explicitly. LLKToMicro preserves semantic function arguments and results, rather than creating entry tensor.empty placeholders and inferring a result from the last store. Lower kernel block arguments/yields through the TypeConverter; internal allocations stay internal.
```text
kernel signature -> func signature
kernel block arguments -> func block arguments
typed micro.yield -> typed func.return
internal tensor.empty -> internal allocation, never a function parameter
```
Retain legacy fixtures as analysis/reference inputs; executing an ambiguous legacy signature requires an explicit adapter/migration, not silent inference.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Input/output ownership and arity are explicit and stable; internal scratch cannot be mistaken for external buffers. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Dialect/Micro/MicroOps.td lib/Dialect/Micro/MicroOps.cpp lib/Conversion/LLKToMicro/LLKToMicro.cpp lib/Conversion/MicroToLinalg/MicroToLinalg.cpp docs/design/m9-micro-ir-core-concepts.md test/Dialect/Micro/tile_ops.mlir test/Dialect/Micro/tile_ops_invalid.mlir test/Execution/micro_kernel_execution.cpp
git commit -m "feat(micro): define explicit kernel argument and result contracts"
```

### C2: Give selected target emitters a real lowering contract

**Dependencies:** B1/B4/B5, C1

**Files:**
- Create: include/LLK/Mapping/MappingLowering.h
- Modify: include/LLK/Mapping/MappingTarget.h
- Modify: lib/Mapping/MappingTarget.cpp
- Create: lib/Target/X86/Mapping/AVX2BundleLowering.cpp
- Modify: lib/Target/X86/Mapping/AVX2MappingTarget.cpp
- Modify: CMakeLists.txt
- Test: test/Mapping/avx2_target.cpp
- Test: test/Mapping/generic_accelerator_target.cpp

**Interfaces:** Implements TargetLoweringContext/TargetEmitter::lower from master. Default file-backed emitter rejects lowering; AVX2 emitters own bundle semantics and typed parameter requirements.

- [ ] **Step 1 — add the regression and legal controls.**

Supply two legal selected bundles for the same operation but different vector widths/implementation strategies; inspect the lowered IR to prove selection changes generated implementation. Unknown bundle, invalid typed parameter and unsupported dtype must reject. Verify the generic-accelerator fixture can map/materialize without claiming it has an executable backend.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingAvx2TargetTest MappingGenericAcceleratorTargetTest -j 6
build/MappingAvx2TargetTest
build/MappingGenericAcceleratorTargetTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Validate the entire resolved bundle and selected port/resource/layout contract before rewriting any operation. Lower on a private clone so failure cannot leave half-rewritten source IR. AVX2 target code owns arithmetic vector width, packing/layout and movement implementation; generic code only dispatches the emitter key.
```cpp
// Default TargetEmitter::lower:
return llvm::createStringError(llvm::inconvertibleErrorCode(),
                              "target_lowering_unsupported: emitter has no lowerer");
```
For executable AVX2 emitters, apply target-owned patterns to the complete covered-op group. Preserve selected physical layout/maps until target lowering has implemented them; do not erase them in the generic bridge first. Unsupported operations or incomplete bundle fields are target-readiness errors.

- [ ] **Step 4 — rerun the focused command and review the gate.**

A selected bundle demonstrably controls emitted code; a verify-only factory cannot be treated as executable. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Mapping/MappingLowering.h include/LLK/Mapping/MappingTarget.h lib/Mapping/MappingTarget.cpp lib/Target/X86/Mapping/AVX2BundleLowering.cpp lib/Target/X86/Mapping/AVX2MappingTarget.cpp CMakeLists.txt test/Mapping/avx2_target.cpp test/Mapping/generic_accelerator_target.cpp
git commit -m "feat(avx2): lower selected mapping bundles through target emitters"
```

### C3: Lower structural, MMA, reduction, view and movement semantics

**Dependencies:** A8, B4–B6, C1/C2

**Files:**
- Modify: lib/Conversion/MicroToLinalg/MicroToLinalg.cpp
- Modify: lib/Target/X86/Mapping/AVX2BundleLowering.cpp
- Modify: include/LLK/Conversion/MicroToLinalg.h
- Test: test/Conversion/MicroToLinalg/micro_to_linalg.mlir
- Test: test/Conversion/MicroToLinalg/micro_to_linalg_invalid.mlir
- Test: test/Conversion/MicroMapping/matmul_e2e.mlir
- Test: test/Conversion/MicroMapping/swiglu_e2e.mlir

**Interfaces:** Reference Micro-to-Linalg supports well-defined logical semantics; selected target lowerers consume physical layouts/bundles. Structural loops preserve bounds/iteration arguments/results and target synchronization semantics.

- [ ] **Step 1 — add the regression and legal controls.**

Require mapped matmul and SwiGLU to traverse the full conversion without residual Micro ops. Add in-bounds dynamic/windowed view, rectangular MMA with dtype conversion, sum/max reduction, tile store output, pipeline and spatial-loop tests. Example view control:
```mlir
%z = arith.constant 0 : index
%one = arith.constant 1 : index
%v = micro.tile_view %a[%one, %z] {shape = array<i64: 7, 8>} : tensor<8x8xf32> -> !micro.tile<7x8xf32, memory = #micro.memory<sram>>
```
Output must use tensor.extract_slice with offset [1,0], size [7,8], unit strides; C4/C8 numerically validate it. Out-of-bounds static views reject.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target llk-opt llk-compile -j 6
ctest --test-dir build -R 'MicroToLinalg|MicroMappingMatmul|MicroMappingSwiGLU' --output-on-failure
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Implement one semantic family at a time, committing each tested pattern within this task: loop structure; MMA/accumulation; vector conversion/silu/mul under math_mode; reduction/gather; views/stores; selected synchronization. Use Linalg/Tensor/SCF and existing target transformations rather than custom backend loops.
```text
reference tile_view -> bounds-aware tensor.extract_slice
reference MMA -> linalg contraction with explicit accumulator/dtype semantics
reference reduce/gather -> declared reduction/assembly semantics
structural loops -> corresponding SCF loops, preserving carried values
selected transform -> target physical reorder, or logical reference copy with
                       explicit reference-mode label; never silently discard target layout
```
Pipeline scheduling must preserve dependency/order; unsupported dynamic/control-flow schedules reject rather than approximate execution. Ensure every compiler-generated operation has either a supported lowerer or a specific target-readiness diagnostic.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Mapped compiler-generated matmul/SwiGLU no longer fail at micro.spatial_for, and views/reductions retain exact semantics. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add lib/Conversion/MicroToLinalg/MicroToLinalg.cpp lib/Target/X86/Mapping/AVX2BundleLowering.cpp include/LLK/Conversion/MicroToLinalg.h test/Conversion/MicroToLinalg/micro_to_linalg.mlir test/Conversion/MicroToLinalg/micro_to_linalg_invalid.mlir test/Conversion/MicroMapping/matmul_e2e.mlir test/Conversion/MicroMapping/swiglu_e2e.mlir
git commit -m "feat(micro): lower complete tiled compute and structural semantics"
```

### C4: Invoke compiled kernels through a tested descriptor-pointer ABI

**Dependencies:** C1–C3, B3

**Files:**
- Create: include/LLK/Runtime/MappedExecutable.h
- Create: runtime/MappedExecutable.cpp
- Modify: include/LLK/Runtime/JitCache.h
- Modify: runtime/JitCache.cpp
- Modify: CMakeLists.txt
- Test: test/Execution/micro_kernel_execution.cpp

**Interfaces:** Implements MappedExecutable::invoke(inputs, outputs) and createMappedExecutable(ModuleOp bufferedModule, StringRef entrySymbol) -> llvm::Expected<std::unique_ptr<MappedExecutable>> from the master, in namespace llk. The factory consumes buffered function/memref IR, preserves its typed ABI contract and owns LLVM translation/JIT/wrapper creation. Generated llvm.emit_c_interface wrappers accept descriptor pointers; results are converted to caller-owned out-parameters. No borrowed source operation is retained.

- [ ] **Step 1 — add the regression and legal controls.**

Compile and **call** add with A[i]=i and B[i]=1, checking all 64 output values equal i+1. Add nonzero descriptor offset/strides, two outputs, dtype/arity mismatch, missing symbol, repeat invocation and executable destruction after use. Use the public interface:
```cpp
TEST(MicroKernelExecution, InvokesDescriptorPointersAndWritesEveryResult) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kAddKernel, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(lowerToLoops(*module));
  auto executable = llk::createMappedExecutable(*module, "add");
  ASSERT_TRUE(bool(executable)) << llvm::toString(executable.takeError());
  module = nullptr; // creation cannot retain borrowed operations
  std::vector<float> a(64), b(64, 1.0f), out(64, -1.0f);
  for (size_t i = 0; i < a.size(); ++i) a[i] = float(i);
  MemRef2D da{a.data(), a.data(), 0, 8, 8, 8, 1};
  MemRef2D db{b.data(), b.data(), 0, 8, 8, 8, 1};
  MemRef2D dout{out.data(), out.data(), 0, 8, 8, 8, 1};
  auto error = (*executable)->invoke({&da, &db}, {&dout});
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  for (size_t i = 0; i < out.size(); ++i) EXPECT_EQ(out[i], float(i + 1));
}
```
Replace kAddKernel with C1's explicit two-argument/one-result fixture, keeping the existing buildRegistry/lowerToLoops helpers. This isolates runtime ABI behavior; C5/C8 additionally exercise selected-target compilation.
A compiler/invocation error is FAIL, not an unconditional JIT-unavailable skip.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MicroKernelExecution JitCacheHit JitCacheMiss -j 6
build/MicroKernelExecution
ctest --test-dir build -R 'JitCache|MicroKernelExecution' --output-on-failure
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

The factory converts buffered function results to out-parameters, records the input/output memref contract before LLVM type erasure, and emits the C-interface wrapper. Adapt descriptor pointers through MLIR invokePacked/C-interface calling support (the local ExecutionEngine header documents packed invocation) or generate/test an equivalent ORC adapter. Do not cast an expanded LLVM entry point to KernelFn or descriptor aggregates.
Validate input/output count, rank/dtype/stride and alias permissions before invocation. Avoid ownership transfer of caller buffers; release internal scratch deterministically. Cache compiled entry points with ABI/signature and plan/content identity. Keep existing legacy KernelFn behavior isolated and covered by its old tests.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Numeric invocation works on supported hosts, descriptors with offsets/strides are exercised, and no unchecked typed cast stands in for ABI validation. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Runtime/MappedExecutable.h runtime/MappedExecutable.cpp include/LLK/Runtime/JitCache.h runtime/JitCache.cpp CMakeLists.txt test/Execution/micro_kernel_execution.cpp
git commit -m "feat(runtime): execute mapped kernels through verified descriptor wrappers"
```

### C5: Integrate a shared mapped compilation path and binding instantiation

**Dependencies:** A7, B1–B8, C1–C4

**Files:**
- Create: include/LLK/Conversion/MappedCompilation.h
- Create: lib/Conversion/MicroMapping/MappedCompilation.cpp
- Modify: tools/llk-compile/llk-compile.cpp
- Modify: lib/Perf/CandidateBinding.cpp
- Modify: lib/Conversion/LLKToMicro/LLKToMicro.cpp
- Modify: CMakeLists.txt
- Test: test/Mapping/e2e_workflow.cpp

**Interfaces:** Produces MappedCompileOptions and compileMappedKernel as declared in the master. Compiler and tuner use this same verified pipeline. Bindings instantiate executable schedules before target mapping.

- [ ] **Step 1 — add the regression and legal controls.**

Add `llk-compile --mapping-target=x86-avx2` with machine/layout/rule/candidate options and optional `--plan-report=<path>` frozen-selection replay and stops at mapped Micro, target-lowered IR and executable output. Test invalid target/bundle/readiness, v1 ambiguity, wrong candidate space and unchanged mapping-disabled output. For BM/BN/BK, pipeline/tail and owner/path axes, require either a changed concrete schedule/placement or a specific unsupported-binding error.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target llk-compile llk-opt MappingE2EWorkflowTest -j 6
build/MappingE2EWorkflowTest
ctest --test-dir build -R 'LLKCompile|MicroMapping' --output-on-failure
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Build the shared sequence:
```text
clone semantic input -> instantiate selected schedule/binding -> export concrete Micro
-> enforce scoped constraints -> CoveringSearch -> strict materialize
-> semantic verify -> selected emitter readiness/lowering
-> bufferize/loop lowering -> createMappedExecutable (output ABI wrapper -> LLVM -> owned JIT)
```
LLK/Linalg inputs can instantiate new tiling from the bound schedule before export. Already concrete Micro input cannot silently claim a different BM/BN/BK: require matching declared facts or an explicit transform/template capable of realizing that binding. Tail/pipeline settings alter executable loops/dependencies; otherwise reject them. Expose clear stop modes and preserve complete selected metadata until its consumers have interpreted it. Mapping disabled keeps the legacy path.

- [ ] **Step 4 — rerun the focused command and review the gate.**

The compiler uses the selected target path end to end, and binding provenance reflects real executable choices. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Conversion/MappedCompilation.h lib/Conversion/MicroMapping/MappedCompilation.cpp tools/llk-compile/llk-compile.cpp lib/Perf/CandidateBinding.cpp lib/Conversion/LLKToMicro/LLKToMicro.cpp CMakeLists.txt test/Mapping/e2e_workflow.cpp
git commit -m "feat(compile): integrate verified target mapping and executable schedules"
```

### C6: Rank and optionally measure mapped tuner candidates

**Dependencies:** B8, C4/C5

**Files:**
- Modify: include/LLK/Perf/TuningSession.h
- Modify: lib/Perf/TuningSession.cpp
- Modify: tools/llk-tune/llk-tune.cpp
- Modify: include/LLK/Mapping/LatencyProvider.h
- Modify: lib/Mapping/LatencyProvider.cpp
- Test: test/Perf/tuning_session.cpp
- Test: test/Mapping/plan_report.cpp

**Interfaces:** TuningSession consumes compileMappedKernel, selected CoveringPlan and common objective validation. Optional measurement callback accepts a verified MappedExecutable plus invocation inputs; production record persistence remains #51/#52.

- [ ] **Step 1 — add the regression and legal controls.**

Use a tiny candidate domain whose valid candidates produce distinct plan IDs/resources/costs; verify ranking by a non-latency primary metric and a secondary metric. Unknown metrics must error, not return zero. Add compile failure retention as a candidate diagnostic, measurement-provider miss/static fallback and an actual timed/numeric invocation for one supported top candidate.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target TuningSessionTest llk-tune -j 6
build/TuningSessionTest
ctest --test-dir build -R LLKTune --output-on-failure
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Enumerate complete persistent bindings, instantiate/map each candidate through C5, rank with the declared mapping objective, and retain deterministic rejection reasons. Optionally compile/invoke top candidates; exclude warmup from measured timing and verify numeric correctness before accepting measurements.
```text
unsupported objective -> stable usage error
provider miss -> static score, unchanged legality
compile/verification failure -> rejected candidate with reason
measurement -> full B8 key + target/model/ABI content identity
```
Expose the selected plan/report alongside legacy schedule output with versioned provenance. Do not implement a second synthetic-only mapping pipeline. Provide the callback/record identity needed by #51/#52 without claiming production calibration is completed.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Tuner choices correspond to materialized selected plans and supported objectives; optional measured results come from invoked, numerically checked code. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Perf/TuningSession.h lib/Perf/TuningSession.cpp tools/llk-tune/llk-tune.cpp include/LLK/Mapping/LatencyProvider.h lib/Mapping/LatencyProvider.cpp test/Perf/tuning_session.cpp test/Mapping/plan_report.cpp
git commit -m "feat(tune): rank and measure executable mapped candidates"
```

### C7: Add bounded fused-rule matching and target-owned group lowering

**Dependencies:** A1/A3, B1/B7, C2/C3

**Files:**
- Modify: include/LLK/Mapping/MappingRules.h
- Modify: lib/Mapping/LlkMap.cpp
- Modify: lib/Mapping/MappingRules.cpp
- Modify: lib/Mapping/WorkloadGraph.cpp
- Modify: lib/Mapping/CoveringSearch.cpp
- Modify: lib/Target/X86/Mapping/AVX2BundleLowering.cpp
- Modify: docs/design/llkmap-rule-grammar.md
- Test: test/Mapping/mapping_rules.cpp
- Test: test/Mapping/covering_search.cpp

**Interfaces:** Adds RulePattern nodes/edges and bounded matchRulePattern, with stable coveredNodes and explicit external PortRefs. TargetEmitter::lower already accepts a covered-op array, so no second emission interface is needed.

- [ ] **Step 1 — add the regression and legal controls.**

Define/round-trip a graph rule for convert -> silu -> mul and verify one candidate covers exactly its declared nodes and external ports. Proposed syntax:
```text
match graph {
  node cv: micro.vector(op = "convert");
  node act: micro.vector(op = "silu");
  node gate: micro.vector(op = "mul");
  edge cv.result -> act.operand0;
  edge act.result -> gate.operand0;
}
```
Test dangling/duplicate node names, undeclared ports, overlap between selected candidates, a synchronization boundary, different regions and an intervening side effect. Fused numeric output must match the unfused reference.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target MappingRulesTest MappingCoveringSearchTest MappingAvx2TargetTest -j 6
build/MappingRulesTest
build/MappingCoveringSearchTest
build/MappingAvx2TargetTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Add explicit formal grammar/lexer tokens and scope rules; existing single-op syntax stays unchanged. Record structural scope/effect boundaries during extraction. Match bounded DAG patterns in canonical anchor/node order, with all predicates checked via A3. Compute external boundary ports from matched edges, deduplicate matches and report matching caps.
Generate candidate.coveredNodes and feed existing complete-cover search; selected candidates cover each required node once. Reject boundary/effect crossings unless the rule explicitly implements the exact semantics. Target-owned AVX2 group lowering implements the declared fused bundle; no generic code interprets its opaque name.

- [ ] **Step 4 — rerun the focused command and review the gate.**

One real compatible subgraph can be selected/lowered as a fused bundle; illegal fusion boundaries and incomplete coverage reject. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Mapping/MappingRules.h lib/Mapping/LlkMap.cpp lib/Mapping/MappingRules.cpp lib/Mapping/WorkloadGraph.cpp lib/Mapping/CoveringSearch.cpp lib/Target/X86/Mapping/AVX2BundleLowering.cpp docs/design/llkmap-rule-grammar.md test/Mapping/mapping_rules.cpp test/Mapping/covering_search.cpp
git commit -m "feat(mapping): match bounded fused implementation rules"
```

### C8: Register the complete design acceptance fixture chains

**Dependencies:** A1–A9, B1–B8, C1–C7

**Files:**
- Create: test/Conversion/MicroMapping/acceptance_pipeline.py
- Create: test/Execution/mapped_acceptance.cpp
- Modify: test/Conversion/MicroMapping/matmul_e2e.mlir
- Modify: test/Conversion/MicroMapping/swiglu_e2e.mlir
- Modify: test/Mapping/e2e_workflow.cpp
- Modify: CMakeLists.txt
- Modify: docs/design/micro-ir-mapping-workflow.md

**Interfaces:** Uses public map/bind/verify/perf/compile/tune interfaces. The runner emits revision/configuration evidence and checks every fixture stage; it does not import an emulator.

- [ ] **Step 1 — add the regression and legal controls.**

Register these five chains: vector add, staged tiled GEMM, fused SwiGLU, required layout transform, generic-accelerator tile two-hop. Each asserts search-space round-trip, selected rule/placement, concrete routes/ops, byte-identical repeat report/IR, perf success and legacy compilation when mapping is disabled.
```python
# Runner behavior: subprocess.run(command, check=True, capture_output=True).
# Compare normalized repeat bytes; fail if required ops/metadata are missing.
# Numeric executable output is checked by mapped_acceptance.cpp, not grep.
```
Numeric AVX2 controls: add(i,1)=i+1; ones GEMM output=K; deterministic random GEMM/SwiGLU against the existing reference math with dtype/math_mode tolerances. Include M/N/K tails and multiple outputs. Generic-accelerator tests prove route/materialization/perf without requiring unavailable hardware; do not label them numeric execution tests.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build -j 6
ctest --test-dir build -R 'MappedAcceptance|MappingE2E|MicroKernelExecution' --output-on-failure
ctest --test-dir build --output-on-failure -j 6
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Make acceptance_pipeline.py drive tools only through public flags and serialized files in its supplied temporary directory. Validate hashes/versions and completeness, compare repeat bytes, and assert resource/event parity under B8. Add C++ invocation tests using C4, checking every result element and output ownership.
Separate host-portable numeric reference tests from AVX2-instruction execution tests on AVX2 CI runners. A required stage's error is a failure; intentional hardware skips state their reason. Avoid compiling an unmapped fixture and calling it proof of target-selected execution. Preserve exact options/profile/rule/layout hashes in test diagnostics.

- [ ] **Step 4 — rerun the focused command and review the gate.**

All five normative chains have stage-by-stage evidence; supported AVX2 chains also invoke selected code and check numerical results. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add test/Conversion/MicroMapping/acceptance_pipeline.py test/Execution/mapped_acceptance.cpp test/Conversion/MicroMapping/matmul_e2e.mlir test/Conversion/MicroMapping/swiglu_e2e.mlir test/Mapping/e2e_workflow.cpp CMakeLists.txt docs/design/micro-ir-mapping-workflow.md
git commit -m "test(mapping): cover complete design acceptance workflows"
```

### C9: Keep generic owner vocabulary independent of backend policy

**Dependencies:** A7/B2, C8

**Files:**
- Modify: include/LLK/Dialect/Micro/MicroDialect.td
- Modify: include/LLK/Dialect/Micro/MicroEnums.h
- Modify: lib/Dialect/Micro/MicroOps.cpp
- Modify: lib/Mapping/Placement.cpp
- Modify: lib/Machine/MachineModelLoader.cpp
- Modify: machines/x86-avx2-v2.yaml
- Modify: docs/design/m9-micro-ir-core-concepts.md
- Test: test/Dialect/Micro/tile_ops.mlir
- Test: test/Mapping/generic_accelerator_target.cpp

**Interfaces:** Abstract owner classes become generic worker/group/vector/matrix/transfer roles. Legacy owner labels are normalized through a target-provided alias table, not hard-coded warp/wave semantics in generic ODS.

- [ ] **Step 1 — add the regression and legal controls.**

Test generic group ownership on two target fixtures with different hierarchy/width data; neither needs a new ODS enum. Legacy owner label migration succeeds only with an explicit target alias; ambiguous/missing aliases reject executable verification. Round-trip generic canonical output and compare legacy disabled-mapping behavior.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build --target llk-opt MappingGenericAcceleratorTargetTest MappingAvx2TargetTest -j 6
ctest --test-dir build -R MicroDialect --output-on-failure
build/MappingGenericAcceleratorTargetTest
build/MappingAvx2TargetTest
```

Expected before the change: the new behavioral assertion fails (or the new interface test cannot compile). Keep existing legal controls green; record the observed failure.

- [ ] **Step 3 — implement this contract.**

Remove backend policy from generic owner/map enum declarations. Make owner class names symbolic at the syntax boundary, with structural verification independent of target IDs. Target machine/package data maps accepted legacy class spellings to generic abstract classes and validates hierarchy/extent.
```text
parse owner class symbol -> target data resolves class/legacy alias
-> normalize generic owner role -> place concrete executor in selected metadata
```
Use the schema minor-version compatibility mechanism for alias/hierarchy metadata; do not change the machine major version unnecessarily. Keep legacy text parseable as unresolved analysis data, but require explicit normalization before target execution. Update hand-written stringifiers/parsers together and preserve concrete executor IDs exclusively in selected metadata.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Canonical generic owner policy no longer requires backend-specific ODS cases, and legacy ambiguity cannot silently change placement semantics. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add include/LLK/Dialect/Micro/MicroDialect.td include/LLK/Dialect/Micro/MicroEnums.h lib/Dialect/Micro/MicroOps.cpp lib/Mapping/Placement.cpp lib/Machine/MachineModelLoader.cpp machines/x86-avx2-v2.yaml docs/design/m9-micro-ir-core-concepts.md test/Dialect/Micro/tile_ops.mlir test/Mapping/generic_accelerator_target.cpp
git commit -m "refactor(micro): normalize abstract ownership through target data"
```

### C10: Publish the final acceptance matrix and completion evidence

**Dependencies:** A/B/C gates through C9

**Files:**
- Create: docs/reviews/issue67-final-acceptance.md
- Modify: docs/design/micro-ir-mapping-workflow.md
- Modify: CLAUDE.md

**Interfaces:** No code API. Deliver revision-pinned §29 evidence matrix, supported-mode boundaries, schema migration guide and #51/#52 handoff. External tracker changes occur only when separately requested.

- [ ] **Step 1 — add the regression and legal controls.**

Run the complete suite and public workflow examples at the delivery head. Required evidence rows:
```text
criterion | fixture/test | exact revision and configuration | result | limitation
```
A criterion without supporting execution/replay/verification evidence remains open. Add a documentation command/link check so example paths/flags exist and unsupported modes are not presented as completed workflows.

- [ ] **Step 2 — run the focused test before changing behavior.**

```sh
cmake --build build -j 6
ctest --test-dir build --output-on-failure -j 6
git diff --check
```

Expected before the documentation update: the evidence matrix is absent or has missing/stale revision references. This task records the verified delivery; it does not require deliberately breaking a passing suite.

- [ ] **Step 3 — implement this contract.**

Replace stale counts and historical claims with revision-pinned results. Link each §29 criterion to registered tests and numeric/runtime evidence where required. Distinguish planning, complete materialization, target readiness, actual invocation and calibrated prediction.
Describe supported schema migration and machine/layout/rule/model version matching. Record #51/#52's remaining measurement/calibration work separately. Do not close #67 based on PR count or compile success; final closure requires no unresolved mandatory criterion and explicit tracker authorization.

- [ ] **Step 4 — rerun the focused command and review the gate.**

Every mandatory design criterion has current reproducible evidence; limitations are explicit rather than hidden by checked boxes. Run relevant neighboring tests when the interface changes. No broad success claim follows merely from a new schema parsing.

- [ ] **Step 5 — commit the independently testable change.**

```sh
git add docs/reviews/issue67-final-acceptance.md docs/design/micro-ir-mapping-workflow.md CLAUDE.md
git commit -m "docs(mapping): publish verified issue 67 acceptance evidence"
```

## Stage C release gate

- [ ] Confirm Linux static-library and local supported LLVM builds pass.
- [ ] Execute numeric tests on supported hosts; run AVX2 target-instruction tests on an AVX2 runner.
- [ ] Complete schema/legacy migration, malformed-input and unsupported-target controls.
- [ ] Every §29 criterion has an evidence row, including deterministic repeat outputs and second-target neutrality.
- [ ] Keep #51/#52 handoff explicit; no production calibration claim follows from providing optional measurement interfaces.
