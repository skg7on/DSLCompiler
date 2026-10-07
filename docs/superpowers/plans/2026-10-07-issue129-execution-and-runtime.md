# Issue #129 Selected Execution and Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close G3/G4/G5 with a checked invocation boundary, balanced scratch ownership and executable realization of selected AVX2/fused decisions.

**Architecture:** Separate reference execution from selected-target execution explicitly. Apply caller-owned output parameters before the MLIR ownership/deallocation pipeline. Dispatch complete selected instance groups with their plan facts; preserve opaque provenance through Micro-to-Linalg and use a target-owned backend pipeline to emit selected-width Vector IR before bufferization.

**Tech Stack:** C++20, MLIR Linalg/Vector/SCF/Bufferization, target packages, LLVM ORC, GTest and FileCheck.

**Spec:** [Master plan](2026-10-07-issue129-gap-closure.md), normative design §§5, 14, 18–19, 25–29, `ARCHITECTURE.md` explicit SIMD/math-mode rules, October 4 C2/C4/C5/C7/C8 contracts.

## Global Constraints

- “`micro` remains the only canonical execution IR.”
- “Target policy lives in target packages.”
- “Existing LLK-to-AVX2 lowering remains the default until end-to-end correctness and performance validation meet roadmap criteria.”
- All master-plan constraints apply. Generic Micro ODS and generic compiler control flow cannot acquire AVX2 names, lane constants or target-specific layout fields.
- Caller-owned input/output memory is borrowed; scratch belongs to one invocation. The checked ABI initially supports static rank two, compact row-major layout, descriptor offset zero and at most **12 total descriptors**, matching the current dispatcher. An aligned pointer inside a larger allocation is supported with a checked base/range; padded rows/nonzero descriptor offsets need a separately supported strided boundary and are rejected in this repair.
- Explicit vectorization precedes One-Shot Bufferization. Strict arithmetic does not gain FMA/approximate exp through compiler flags; preserve declared `math_mode`.

## Review Focus

1. Non-rank-2 and wrong-dtype inputs must be rejected before machine code runs: E1.
2. A returned input, shared internal result or nested allocation must not leak or free caller storage: E2/E3.
3. A fused group with live external intermediate uses must preserve those outputs or reject before rewriting: E4/E5.
4. VW4/VW8, transformed/padded internal layouts and M/N/K tails must affect valid emitted code; unsupported boundary strides must fail: E6/E7/E9.
5. An arm64 or x86 host missing required features must not run code labeled selected AVX2: E8 and T8.

---

## File responsibilities and interface additions

`MappedInvocation.h/.cpp` owns typed host descriptors and validation. `MappedKernelAbi.h/.cpp` owns ABI preparation/ownership passes before LLVM lowering. `MappedExecutable.cpp` owns JIT lifetime and call dispatch. `MappedCompilation.cpp` owns shared compilation sequencing and dispatch by selected instance, not target semantics. `AVX2BackendLowering.cpp` owns target-specific Linalg/Vector lowering. MicroToLinalg copies opaque selected metadata but never interprets AVX2 fields.

Proposed interfaces introduced in the owning tasks. E1 introduces the plain data headers for ABI/codegen/JIT options and the backend enum so E2 has complete types; E6/E8 implement the target hooks and their behavior:

```cpp
// E1: namespace llk, MappedInvocation.h.
enum class InvocationElementType { F32, BF16, F16, I32, I8 };
struct InvocationBuffer2D {
  MemRef2D descriptor{};
  InvocationElementType elementType = InvocationElementType::F32;
  uint64_t allocationBytes = 0; // bytes from descriptor.allocated
};
llvm::Error validateMappedInvocation(
    const KernelAbi &abi, llvm::ArrayRef<InvocationBuffer2D> inputs,
    llvm::ArrayRef<InvocationBuffer2D> outputs);
// MappedExecutable gains this checked overload:
llvm::Error invoke(llvm::ArrayRef<InvocationBuffer2D> inputs,
                   llvm::ArrayRef<InvocationBuffer2D> outputs);
inline constexpr uint32_t kMappedAbiVersion = 1;
uint64_t computeKernelAbiHash(const KernelAbi &abi);
// KernelAbi gains version=1; MappedExecutable gains uint64_t abiHash() const.

// E2: namespace llk, MappedKernelAbi.h.
struct PreparedMappedKernel {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  KernelAbi abi;
  std::string entrySymbol;
};
llvm::Expected<PreparedMappedKernel>
prepareMappedKernelForInvocation(mlir::ModuleOp bufferedModule,
                                  llvm::StringRef entrySymbol);
// Runtime factory overload consumes the prepared ownership-safe module;
// raw-module overload delegates to preparation, preserving source compatibility.
llvm::Expected<std::unique_ptr<MappedExecutable>>
createMappedExecutable(PreparedMappedKernel prepared,
                       const MappedJitOptions &options = {});

// E1 data declaration, E6 selected behavior: MappedCompilation.h, namespace llk.
enum class MappedBackend { Reference, SelectedTarget };
// MappedCompileOptions gains MappedBackend backend = MappedBackend::Reference;

// E1 data header: Mapping/CodegenRequirements.h, namespace mlir::llk::mapping.
struct TargetCodegenRequirements {
  std::string architecture;
  std::string cpu;
  std::vector<std::string> requiredFeatures;
};
// E6/E8 hooks in MappingTarget.h:
virtual llvm::Expected<TargetCodegenRequirements>
codegenRequirements() const;
virtual llvm::Error buildSelectedBackendPipeline(mlir::OpPassManager &pm) const;
// Base implementations reject selected execution; configuration-only targets
// retain mapping/materialization/perf support and explicit reference mode.

// E1 data header, E8 JIT behavior: Runtime/MappedJitOptions.h, namespace llk.
struct MappedJitOptions {
  std::optional<mlir::llk::mapping::TargetCodegenRequirements> selectedTarget;
};
```

Resolve header ownership to avoid `LLKRuntime -> LLKMappedCompilation`: move existing KernelAbi data to `include/LLK/Runtime/KernelAbi.h`, place PreparedMappedKernel data in `include/LLK/Runtime/PreparedMappedKernel.h`, and expose preparation in the conversion header as a library LLKRuntime can call. Add `LLKMappedKernelAbi` in root CMake; it links MLIR libraries and LLK dialect utilities, **not LLKRuntime or LLKMappedCompilation**. LLKMappedCompilation and LLKRuntime both depend on it. CodegenRequirements.h and MappedJitOptions.h contain plain data; Runtime never includes the full target registry. Before E6, requesting the newly declared SelectedTarget mode explicitly fails as unsupported, preserving Reference default behavior.

Test support created in E1/E2: `test/Execution/mapped_execution_fixture.h/.cpp` defines `checkedBuffer(std::vector<float> &, int64_t rows, int64_t cols)` returning an InvocationBuffer2D, and `compileReferenceFixture(llvm::StringRef file)` returning `llvm::Expected<MappedCompilation>` using compileConcreteMicroKernel in Reference mode. E4 adds `compileSelectedFixture(file, backend, stop)` using the registered target, canonical search/evaluation and compileMappedKernel; use R1/R3/R4/R7 when strict selected plans are required. Fixture names always refer to checked-in files under `test/Execution/Inputs/issue129/`; helper constructors do not compute expected numeric outputs.

### Task E1: Define and check a typed rank-two invocation contract

**Files:** Create MappedInvocation header/source, `include/LLK/Runtime/{KernelAbi,MappedJitOptions}.h`, `include/LLK/Mapping/CodegenRequirements.h`; modify `include/LLK/Runtime/MappedExecutable.h`, `include/LLK/Conversion/MappedCompilation.h`, `runtime/MappedExecutable.cpp`; create `test/Execution/mapped_invocation.cpp` (`MappedInvocationTest`), helper files and fixtures `rank3.mlir`, `dynamic_rank2.mlir`, `add.mlir`; root CMake.

**Interfaces:** Produces the typed descriptor/validation API above. Keep the old pointer overload as a deprecated compatibility entry, document that it cannot check element type or backing allocation size, and rename its implementation `invokeUncheckedLegacy`; migrate all new mapped/tuner/acceptance callers to the checked overload. Do not claim dtype detection from untagged `void *`.

- [ ] **Step 1: Add factory rejection and pure descriptor tests.** Compile static rank3, dynamic rank2, unsupported dtype and 13-descriptor functions; all fail construction before wrapper/JIT. The pure checker can be tested without a JIT or supported AVX2 host.

```cpp
KernelAbi abi;
abi.inputs.push_back({{8, 8}, "f32"});
abi.outputs.push_back({{8, 8}, "f32"});
std::vector<float> input(64), output(64);
auto in = checkedBuffer(input, 8, 8);
auto out = checkedBuffer(output, 8, 8);
EXPECT_FALSE(validateMappedInvocation(abi, {in}, {out}));
in.elementType = InvocationElementType::BF16;
auto error = validateMappedInvocation(abi, {in}, {out});
ASSERT_TRUE(static_cast<bool>(error));
EXPECT_NE(llvm::toString(std::move(error)).find("dtype"), std::string::npos);
```

- [ ] **Step 2: Run MappedInvocationTest.** The initial compile exposes the missing typed API; the rank3 test must fail on current factory behavior once helper scaffolding is present.
- [ ] **Step 3: Validate the ABI at construction.** Require rank=2, static positive extents, supported element type and total arity≤12. Introduce the first explicit checked ABI version, `kMappedAbiVersion=1`, independently of report schema v3, and a deterministic content hash (shape/type/port order/stride and alias policy/descriptor layout); do not alter legacy KernelFn's calling convention. Backend requirements travel separately in codegen identity.
- [ ] **Step 4: Implement checked range arithmetic.** Verify arity, tag, shape, non-null allocation/aligned base, offset=0, column stride=1 and row stride=columns for the identity-layout boundary. Compute maximum touched element/address with checked multiply/add and require it within allocationBytes from allocated base. Allow an aligned base inside a larger allocation, retaining that allocation's actual beginning/size; reject padded rows, nonzero offsets, negative/non-unit strides explicitly. Do not assume an identity-layout memref reads dynamic stride fields. A typed tag is the caller's declared type, not a way to inspect actual bytes.

```cpp
// For a rank-two positive-stride port:
// last = offset + (rows - 1) * stride0 + (cols - 1) * stride1;
// byteEnd = (aligned - allocated) + (last + 1) * elementBytes;
// Every operation is overflow-checked before address comparison or calling.
```

- [ ] **Step 5: Define alias behavior.** Inputs may overlap because they are read-only. Outputs must not overlap each other or any input span in this first checked ABI; diagnose which ports overlap. A kernel returning an input is supported by copying into a distinct caller output, tested in E2. Refuse kernels that require input mutation unless an explicit read/write port contract is added in a separately reviewed extension; bufferization must not make a read-only input mutable implicitly.
- [ ] **Step 6: Add null, extent, arity, offset, subview, short allocation, overflow, row-padding and alias controls.** Wrong metadata never reaches a test entry function that increments a call counter. An aligned subview with offset zero computes correct add values; nonzero descriptor offsets/padded rows reject before the call. Preserve the existing larger-allocation test via its supported aligned-subview form. Run MicroKernelExecution and MappedInvocationTest; commit with `feat: validate typed mapped invocation buffers`.

### Task E2: Prepare output arguments before deterministic buffer deallocation

**Files:** Create MappedKernelAbi header/source, `include/LLK/Runtime/PreparedMappedKernel.h` and LLKMappedKernelAbi CMake target; modify `MappedCompilation.cpp`, `MappedExecutable.cpp` and header dependencies; test `test/Execution/micro_kernel_execution.cpp`; create `borrowed_return.mlir`, `shared_results.mlir`, `nested_alloc.mlir` under execution Inputs.

**Interfaces:** Produces prepareMappedKernelForInvocation and PreparedMappedKernel. Both compileMappedKernel and compileConcreteMicroKernel use the same preparation. The direct raw-module factory delegates once; the prepared factory never repeats ABI conversion/deallocation. The Lowered inspection stop retains its documented pre-ABI signature.

- [ ] **Step 1: Add allocation/deallocation IR tests.** Inspect a compiled GEMM at pre-LLVM prepared form: every compiler-owned allocation has a valid release on its return path; returned memrefs have become borrowed output arguments. Include a returned input, two outputs aliasing one internal value, and alloc inside SCF loops.

```cpp
auto prepared = prepareMappedKernelForInvocation(*bufferedModule, "kernel");
ASSERT_TRUE(prepared);
auto function = prepared->module->lookupSymbol<mlir::func::FuncOp>("kernel");
EXPECT_EQ(function.getFunctionType().getNumResults(), 0u);
EXPECT_EQ(prepared->abi.outputs.size(), 2u);
// Walk the prepared IR in the test: ownership/deallocation checks are about
// every allocated path, not the number of textual alloc/dealloc operations.
```

- [ ] **Step 2: Run the new preparation tests and inspect current lowered matmul.** Current baseline has seven memref.alloc sites and no releases; preserve this as a regression input, not an expectation that exactly seven allocations must remain after optimization.
- [ ] **Step 3: Move result conversion into shared preparation on a clone.** Capture input/output ABI from original memref types, add caller-owned output parameters, copy each returned value to its matching output, replace returns with void and mark input/output arguments borrowed. Shared internal results are copied before one ownership release; returning an input never transfers its ownership.
- [ ] **Step 4: Run the standard ownership/deallocation pipeline after the ABI rewrite.** The API is `mlir::bufferization::buildBufferDeallocationPipeline(pm)` from `mlir/Dialect/Bufferization/Pipelines/Passes.h`, verified in both the local LLVM24 header and [LLVM22.1.8's header](https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/mlir/include/mlir/Dialect/Bufferization/Pipelines/Passes.h). Register required SCF/Func/BufferDeallocation external models and link `MLIRBufferizationPipelines` plus its declared dependencies explicitly. The pinned static build validates the complete pipeline/dependency contract; use a small version adapter only if actual compiler diagnostics require it.

```cpp
// Preparation sequence after cloning and changing the function contract:
mlir::PassManager pm(module.getContext());
mlir::bufferization::buildBufferDeallocationPipeline(pm);
if (mlir::failed(pm.run(module)))
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapped ownership preparation failed");
```

- [ ] **Step 5: Keep ownership sound through loops and aliases.** Test copies/subviews/casts and loop-carried scratch; input/result aliases and borrowed boundary descriptors are excluded from frees. Reject unsupported ownership/control-flow constructs instead of emitting a leak. The LLVM lowering consumes a deallocation-lowered module; its alloc/free runtime symbols must be resolvable on both static Linux and local Darwin builds.
- [ ] **Step 6: Run preparation, MicroKernelExecution and MappedAcceptance tests and a static LLVM22 link build.** Commit with `fix: prepare mapped output ABI before buffer deallocation`. Review gate: both public compilation APIs and direct factory calls pass through ownership preparation exactly once.

### Task E3: Prove per-invocation allocation balance and caller ownership

**Files:** Create `test/Execution/mapped_allocation_lifetime.cpp` (`MappedAllocationLifetimeTest`), test allocator support; add a narrow internal test factory seam in `runtime/MappedExecutable.cpp` if required; modify root CMake. No global process allocator replacement.

**Interfaces:** Test-only factory in `test/Execution/mapped_jit_test_factory.h`:

```cpp
struct TestAllocatorHooks {
  void *(*allocate)(size_t) = nullptr;
  void *(*alignedAllocate)(size_t, size_t) = nullptr;
  void (*release)(void *) = nullptr;
};
llvm::Expected<std::unique_ptr<llk::MappedExecutable>>
createMappedExecutableForTest(llk::PreparedMappedKernel prepared,
                              const llk::MappedJitOptions &options,
                              TestAllocatorHooks hooks);
```

The seam overrides only the JIT's emitted allocation symbols. Test hooks track pointer/size sets and real allocation/free calls; JIT/compiler allocations are not counted. Default production behavior is unchanged. Verify the actual emitted allocator spellings (malloc/aligned_alloc/free) rather than guessing. Implement a test-only `AllocationTracker` with `std::set<void *> outstandingPointers() const`, `bool freedAny(llvm::ArrayRef<void *> callerPointers) const`, and `uint64_t doubleFreeCount() const`; its hook handlers insert/erase pointer-size records under a mutex, call the actual allocator, and record an unknown release as a failure. Add numeric output verification immediately after each call. Obtain PreparedMappedKernel by the same Lowered -> shared preparation sequence used by production compilation.

- [ ] **Step 1: Add a repeated-call regression with scratch-heavy GEMM.** Invoke at least 100 times with changing inputs, validate outputs each time, and require the outstanding allocation set to return to its pre-call state after every invocation.

```cpp
for (unsigned iteration = 0; iteration < 100; ++iteration) {
  auto before = tracker.outstandingPointers();
  auto error = executable->invoke(inputs, outputs);
  ASSERT_FALSE(static_cast<bool>(error));
  EXPECT_EQ(tracker.outstandingPointers(), before);
  EXPECT_FALSE(tracker.freedAny(callerPointers));
}
EXPECT_EQ(tracker.doubleFreeCount(), 0u);
```

- [ ] **Step 2: Run MappedAllocationLifetimeTest before E2 integration to capture the leak; run it again after E2.** Avoid RSS-based thresholds or assuming allocation-site count equals dynamic allocation count.
- [ ] **Step 3: Add borrowed-return, shared-results, transformed-copy, nested-loop and repeated-executable controls.** Exercise every compiler allocation family actually emitted. Record maximum live bytes and balanced call counts; all caller allocations remain owned by the test. Validate output after the original MLIR module has been destroyed.
- [ ] **Step 4: Add a rejected invocation control.** Wrong dtype/range causes zero emitted allocation calls and no machine-code entry. Destruction of an executable after calls does not release caller memory or re-free invocation scratch.
- [ ] **Step 5: Run lifetime/ABI/runtime tests on both local host and pinned static x86 CI.** Commit with `test: prove deterministic mapped scratch release`. Review gate: each real call returns with no outstanding JIT scratch, regardless of numeric success.

### Task E4: Dispatch each selected instance as one complete group

**Files:** Modify `lib/Conversion/MicroMapping/MappedCompilation.cpp`, `include/LLK/Mapping/MappingLowering.h`, `lib/Mapping/PlanVerification.cpp`; tests `test/Mapping/e2e_workflow.cpp`, new `test/Execution/selected_group_lowering.cpp` (`SelectedGroupLoweringTest`); extend execution fixture helper.

**Interfaces:** Change lowerSelectedOperations to receive `const CoveringPlan &`; fill existing TargetLoweringContext with real placements/connections, plus `InstanceId instance=0` and a read-only group ordering/port mapping if needed. Consumes R1 persisted compute and R3/R4 complete physical decisions; strict fixture helper uses R7 evaluation when available. Produces one TargetEmitter::lower call per selected instance.

- [ ] **Step 1: Add a recording emitter test.** Three selected convert/silu/mul nodes share one instance; it must receive exactly one group of three in topological order and nonempty selected context. Two independent instances receive two calls. The test emitter verifies operands/ports and changes nothing until validation succeeds.

```cpp
EXPECT_EQ(recordingEmitter.calls.size(), 1u);
EXPECT_EQ(recordingEmitter.calls.front().coveredNames,
          (std::vector<std::string>{"micro.vector", "micro.vector", "micro.vector"}));
EXPECT_EQ(recordingEmitter.calls.front().instance, selectedInstance);
EXPECT_EQ(recordingEmitter.calls.front().placements.size(), 3u);
EXPECT_EQ(recordingEmitter.calls.front().computeId, "vpu.b");
```

- [ ] **Step 2: Run SelectedGroupLoweringTest.** Baseline sends singleton operations with empty placement/connection arrays.
- [ ] **Step 3: Collect stable groups before mutation.** Decode metadata, group operations by instance ID, check consistent bundle/rule/compute/layout facts, topologically order groups and members, and resolve endpoint/value mappings. Include connections touching the group. Preserve source operation mapping through moves/rewrites with explicit replacement tracking; do not keep dangling Operation pointers after an earlier emitter erases them.
- [ ] **Step 4: Preflight all groups before any destructive rewrite.** Verify target support, complete parameters, legal region/effect boundaries and external-use contracts. Compile on a private clone so any later lowering failure leaves the caller's source intact. A missing group member, conflicting same-instance bundle or region crossing rejects with the instance/rule ID.
- [ ] **Step 5: Count evidence by instances and operations separately.** Add explicit counters for selected groups verified, backend groups realized and reference groups lowered. Existing targetLowered compatibility counts cannot establish selected machine code. Test a group with a live external intermediate and an effectful/region-crossing negative.
- [ ] **Step 6: Run group/workflow/binder tests.** Commit with `fix: lower complete selected instances with plan context`. Review gate: one group call realizes the same instance the planner selected, using its recorded resources.

### Task E5: Complete the shipped fused rule's parameter and output contract

**Files:** Modify `mapping/x86-avx2/{rules,layouts}.llkmap`, `lib/Target/X86/Mapping/AVX2BundleLowering.cpp`; tests `test/Mapping/{mapping_rules,avx2_target}.cpp`; create `test/Execution/Inputs/issue129/fused_convert_silu_mul.mlir` and `test/Conversion/MicroMapping/fused_selected_lowering.py`.

**Interfaces:** Consumes E4 group dispatch and R1/R3 selected facts. Produces a complete `avx2.fused_convert_silu_mul` bundle with one selected VW and solved endpoint layouts, and a verified fused group output contract. Retain the finite graph matcher and existing effect/region limits.

- [ ] **Step 1: Check in the audit's exact convert→silu→mul fixture.** Search must choose the shipped fused rule; strict bind and target lowering must succeed. Assert selected rule/instance count, resolved VW and physical endpoint bindings, not merely presence of `micro.mapping`.

```python
report = json.loads(report_path.read_text())
selected = report["selectedState"]  # existing selected-state key retained in v3
assert any(p["rule"] == "avx2.fused_convert_silu_mul"
           for p in selected["placements"])
# Also assert every placement in that instance shares the same resolved VW.
```

- [ ] **Step 2: Run the public llk-compile exact search to target-lowered.** Baseline fails with “no vector width was selected”; keep that failure text in the negative regression history.
- [ ] **Step 3: Add target-owned layout requirements that solve VW and physical endpoint maps.** Reject unresolved/conflicting width before emission. Support only explicitly declared conversion dtype pairs; a no-op f32 convert is legal but must not conceal unsupported bf16/f16 conversion. Supply named memory requirements and capability checks consistent with R3.
- [ ] **Step 4: Verify fused semantics and exposed results.** Preserve every live external use of intermediate results, either by returning those results from the combined Linalg body or by rejecting that candidate during matching/verification so standalone rules remain searchable. Do not silently change an externally used conversion result. Keep strict versus bounded_fast math-mode semantics.
- [ ] **Step 5: Add positive/numeric and negative controls.** VW4/VW8 exact selections; missing/conflicting VW; bf16→f32 conversion; effect/region boundary; repeated operand; intermediate external use. Numerical fused execution is finalized after E6; compile/target-lowered success alone does not close G5.
- [ ] **Step 6: Run rule/target/group/public fused tests.** Commit with `fix: resolve fused AVX2 bundle layout and group outputs`. Review gate: the shipped rule itself is executable-ready under its declared supported types, not an unrelated standalone SwiGLU fixture.

### Task E6: Preserve selected provenance and emit explicit vector arithmetic

**Files:** Modify `MappingTarget.h`, `MappedCompilation.h/.cpp`, `lib/Conversion/MicroToLinalg/MicroToLinalg.cpp`, `AVX2BundleLowering.cpp`; create AVX2BackendLowering header/source; root CMake; tests `test/Conversion/MicroMapping/selected_vector_width.py`, execution fixtures `width4.mlir`, `width8.mlir` and arithmetic/fused cases.

**Interfaces:** Produces MappedBackend and target backend hook above. AVX2 implements `buildSelectedBackendPipeline`; its pass interprets preserved opaque bundle/instance metadata on Linalg. Generic MicroToLinalg only transfers that metadata. Reference is the compatibility default; selected mode rejects unimplemented selected compute bundles.

- [ ] **Step 1: Add the width-erasure regression at the actual selected backend boundary.** Map equivalent kernels to VW4/VW8, stop after selected lowering/vectorization, and require `vector<4xf32>` versus `vector<8xf32>` transfer/arithmetic. Also ensure the final lowered body retains explicit Vector operations until Vector-to-LLVM conversion. Reference MicroToLinalg output equality alone is not the repaired backend test.

```python
assert "vector<4xf32>" in width4_ir
assert "vector<8xf32>" in width8_ir
assert width4_ir != width8_ir
assert "vector.transfer_read" in width8_ir
assert "vector.transfer_write" in width8_ir
```

- [ ] **Step 2: Run the new selected-width test.** Baseline has no selected backend mode/hook and loses the width in tensor conversion.
- [ ] **Step 3: Preserve generic provenance while building Linalg.** Transfer selected instance/bundle/layout/compute facts to the corresponding Linalg operation(s). Explicitly map one fused group to a fused Linalg generic body when legal. Mark non-compute structural conversions separately. Do not add AVX2 width decoding to MicroTypeConverter or generic ODS.
- [ ] **Step 4: Implement AVX2 vectorization for f32 add/mul/convert/silu and the fused group using Linalg/Vector transforms.** Tile the vector dimension by the selected width and emit transfer_read/write with masks or a scalar cleanup path. A fused group performs conversion/SILU/multiply in one vector body rather than merely rewriting each tile layout. Use existing math approximation utilities for declared bounded_fast; strict mode uses mathematically faithful exp lowering/runtime support and never enables global unsafe flags.

```cpp
// Selected compilation sequencing (generic orchestrator):
pm.addPass(mlir::llk::createMicroToLinalgPass());
if (options.backend == MappedBackend::SelectedTarget) {
  if (auto error = target.buildSelectedBackendPipeline(pm))
    return std::move(error);
}
// Only then One-Shot Bufferization, remaining Linalg-to-loops, ABI ownership,
// and Vector/SCF/LLVM conversion. Target pass rejects unrealized compute tags.
```

- [ ] **Step 5: Validate realization and reporting.** Selected mode checks every selected compute group has a backend realization record; unknown dtype/op/VW fails with bundle/instance context. Metadata-only rewrites and scalar reference bridging cannot increment backendGroupsRealized. Explicitly reported scalar tail cleanup is part of a vector implementation, not an unsupported whole-group fallback.
- [ ] **Step 6: Run selected-width/arithmetic/group tests plus MicroToLinalg and reference runtime controls.** Commit with `feat: realize selected AVX2 widths in Vector IR`. Review gate: width affects actual executable IR and fused selection changes the computation body, without generic target policy.

### Task E7: Realize staged contraction, movement and required transforms

**Files:** Modify AVX2BackendLowering, `AVX2BundleLowering.cpp`, target rules/profile and selected provenance adapters; tests `test/Execution/Inputs/issue129/{matmul_bf16,matmul_f32,required_transform}.mlir`, `test/Conversion/MicroMapping/selected_contraction.py`.

**Interfaces:** Consumes E6 hook, R4 per-hop/storage metadata and solved physical layouts. Produces selected AVX2 MMA/reduce/copy/store/transform support needed by shipped staged GEMM/SwiGLU/transform acceptance; unsupported declared emitters remain explicitly unsupported. An abstract `micro.mma` on this CPU lowers to vector contraction/reduction; it does not imply an AMX or native bf16-matrix instruction.

- [ ] **Step 1: Add explicit selected GEMM/transform IR checks.** bf16 operands are widened correctly to f32, accumulation preserves the initialized accumulator, vector width is selected, staged memory/copy order and the required map transform survive. A fixture whose transform changes observable data order must fail if the transform is removed.

```python
assert "vector.contract" in selected_ir or "vector.reduction" in selected_ir
assert "vector<8xf32>" in selected_ir
assert selected_manifest["backend_groups_realized"] > 0
assert selected_manifest["unrealized_compute_groups"] == 0
# Exact numeric transform/permutation assertion belongs to E9/T7 as well.
```

- [ ] **Step 2: Run selected_contraction.py.** Baseline carries these emitters through scalar reference lowering.
- [ ] **Step 3: Vectorize Linalg contractions using the selected fragment/layout.** Keep K reduction order/precision under the math contract; strict mode does not fuse multiply-add implicitly. If a bundle requests FMA, declare `fma` as a required feature and permit it only under a compatible math contract. Implement bf16 widening as an exact bit conversion before f32 arithmetic on AVX2; no unsupported bf16 opcode is assumed.
- [ ] **Step 4: Lower selected physical maps and movement.** Copies and stores use recorded allocations/routes; layout transform realizes its source/destination affine maps and physical padding, with explicit Vector transpose/shuffle or correct strided transfers. For reference host execution, target memory kinds map to host storage while retaining logical provenance; do not claim accelerator hardware behavior.
- [ ] **Step 5: Cover reduce and tail cases required by the shipped rules.** Reject unsupported reduction semantics, mixed layout maps or fragment shapes during verification. Add initialized-accumulator, nonidentity transform, same-kind distinct memory and two-stage copy controls. Ensure R6 static provenance remains traceable before backend lowering.
- [ ] **Step 6: Run selected contraction/transform and existing materialization/structural tests.** Commit with `feat: lower selected AVX2 contractions and physical layouts`. Review gate: selected staged GEMM and required-transform chains have no whole-compute reference fallback.

### Task E8: Enforce selected ISA requirements and truthful compilation evidence

**Files:** Modify AVX2 target, MappingTarget interface, MappedJitOptions header, `runtime/MappedExecutable.cpp`, `MappedCompilation.cpp`, `tools/llk-compile/llk-compile.cpp`; tests `test/Execution/mapped_codegen_requirements.cpp` (`MappedCodegenRequirementsTest`); root CMake.

**Interfaces:** Completes TargetCodegenRequirements/codegenRequirements and prepared factory JIT options. Produces `--mapping-backend=reference|selected-target` on llk-compile and a compile manifest with selected target identity, CPU/features, ABI identity, verified/realized/reference counts and reached stop. Reference manifest records actual native host/reference mode.

- [ ] **Step 1: Add policy-only host-feature tests.** Factor feature checking into `llvm::Error checkHostRequirements(const TargetCodegenRequirements &, llvm::StringRef hostArchitecture, const llvm::StringMap<bool> &hostFeatures)` in the runtime-visible codegen header/source. Inject arm64, x86 without avx2, x86 with avx2, and optional fma cases; no illegal instruction is executed in negative tests.

```cpp
llvm::StringMap<bool> features;
features["avx2"] = false;
auto rejected = checkHostRequirements(requirements, "x86_64", features);
ASSERT_TRUE(static_cast<bool>(rejected));
EXPECT_NE(llvm::toString(std::move(rejected)).find("avx2"), std::string::npos);
features["avx2"] = true;
EXPECT_FALSE(checkHostRequirements(requirements, "x86_64", features));
auto wrongArchitecture = checkHostRequirements(requirements, "aarch64", features);
ASSERT_TRUE(static_cast<bool>(wrongArchitecture));
llvm::consumeError(std::move(wrongArchitecture));
```

- [ ] **Step 2: Run the codegen requirements test.** Baseline JITBuilder only targets the native host and has no selected-target feature contract.
- [ ] **Step 3: Apply requirements to JITTargetMachineBuilder.** Start from detected host triple/data layout, verify architecture and features, then set deterministic selected CPU/features. Reference may use the native host; selected mode uses the target's declared requirements. Reject incompatible host before adding/calling IR. Avoid adding X86 codegen libraries to arm64 builds solely to pretend the selected kernel can run there.
- [ ] **Step 4: Version compiler/ABI/codegen identity.** Include backend, required CPU/features, math mode, target content and ABI hash in executable/measurement identity. Require both mapped semantic verification and actual backend realization before reporting selected execution. `TargetLowered`/`Lowered` stops report compilation stages only; a lookup is not invocation evidence.
- [ ] **Step 5: Add one supported-x86 invocation and object/LLVM evidence control.** Save Vector/LLVM IR and CPU/features manifest; inspect vector instructions where stable, avoiding a brittle mandatory exact assembly mnemonic. A real numerical call plus explicit Vector IR and declared ISA establishes selected-target evidence. Host-dependent skipping is allowed only in portable local tests; T8's selected-x86 acceptance fails if it skips.
- [ ] **Step 6: Run codegen/CLI/reference tests.** Commit with `feat: enforce mapped target ISA and backend identity`. Review gate: arm64 reference success cannot appear as selected AVX2 execution.

### Task E9: Implement bounded tails and verify multiple-result execution

**Files:** Modify shared LLKToMicro/CandidateBinding tiling helpers, AVX2BackendLowering and structural MicroToLinalg lowering where needed; tests execution fixtures `tail_{m,n,k,all}.mlir`, `multi_output.mlir`, `transform_output.mlir`, `borrowed_return.mlir`; extend MappedAcceptance/MappedInvocation/MappedAllocationLifetime.

**Interfaces:** Consumes R3/R4 physical shape/maps and E1–E8. Supports `tail_policy=pad` in the first repair: valid extents recorded generically, source reads masked/zero-filled, fragment computes full tiles, final stores restricted to valid extents. `tail_policy=none` still rejects non-divisible shapes; unsupported `mask`/other policies are explicit rejections until separately implemented. Generic valid-extent metadata carries no target-specific lane/ISA names.

- [ ] **Step 1: Add tail correctness tests that currently cannot be represented.** Use M/N/K `(5,9,7)`, `(1,17,3)`, `(17,65,63)` with chosen tile/fragment widths, nonzero initial accumulators, random signed inputs and guard canaries. Use bf16-rounded inputs for bf16 reference math. Multi-output returns two different values; returning an input copies into a separate output.

```cpp
// Reference for C = init + A * B, using the actual stored input values.
for (int64_t m = 0; m < M; ++m)
  for (int64_t n = 0; n < N; ++n) {
    float expected = init[m * N + n];
    for (int64_t k = 0; k < K; ++k)
      expected += a[m * K + k] * b[k * N + n];
    EXPECT_NEAR(output[m * N + n], expected,
                2e-4f + 2e-4f * std::abs(expected));
  }
EXPECT_EQ(leftGuard, expectedGuard);
EXPECT_EQ(rightGuard, expectedGuard);
```

- [ ] **Step 2: Run tail/multi-output tests.** Existing export/binder rejects non-divisible shapes or silently lacks the requested masked/padded behavior; do not mark unsupported tails passed by skipping.
- [ ] **Step 3: Share tail realization between source export and candidate instantiation.** Move the common bounded tiling/padding algorithm to a focused existing conversion utility or new `include/LLK/Conversion/MicroMapping/TileExtent.h` / `lib/Conversion/MicroMapping/TileExtent.cpp`. Preserve original tensor extents, math mode, operand roles and accumulator. Pad K inputs with zeros, pad M/N work, and suppress out-of-range stores. Feed physical padded footprints and valid extents to R5/R6.
- [ ] **Step 4: Realize vector tails consistently.** In-bounds full lanes use selected width; partial vectors use masks or explicit scalar cleanup. Required layout transforms preserve the valid-region mapping. Structural lowering cannot clamp a schedule into a different shape without recording it. Negative tail-none and unsupported policy cases receive stable candidate rejection reasons.
- [ ] **Step 5: Verify two different outputs, shared internal values and aligned subviews numerically; reject row padding/nonzero offsets before invocation.** Apply checked E1 invocation and E3 allocation tracker. Use the same fixture on reference and selected x86 mode where supported; verify declared approximate math tolerance separately from strict GEMM/add. Compare external intermediate results of fused groups or prove the candidate was rejected before selection.
- [ ] **Step 6: Run targeted tails/runtime/ownership plus selected x86 tests, then the full suite at Gate E.** Commit with `feat: realize bounded mapped tails and multiple outputs`. Review gate: selected execution handles the declared positive tail contract and rejects unsupported policies without memory corruption.
