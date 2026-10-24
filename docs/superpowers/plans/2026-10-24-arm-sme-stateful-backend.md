# Arm SME Stateful Executable Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the accepted f32 16x16x16 initialized contraction executable on `arm-sme` through a native Apple M5 runner (tier T3), with QEMU SME1 as the CI-automated secondary tier (T2).

**Architecture:** The `arm-sme` package becomes a selective executable target: an emitter claims `sme_mma_za` plus the three movement keys, a pre-bufferization pass tiles/vectorizes the selected `linalg.matmul` into scalable Vector IR, and a post-bufferization pass runs MLIR's production ArmSME sequence with caller-managed streaming (`aarch64_pstate_sm_enabled` + `aarch64_new_za`). A tiny assembly shim is the only code that changes PSTATE.SM. Generic framework additions are target-neutral: a streaming-vector-length field, invocation attributes, and a post-bufferization target hook.

**Tech Stack:** C++20, MLIR/LLVM 22.1.8 (CI) and 24 (local), MLIR ArmSME dialect and conversion passes, `llk-compile` AOT object emission, Python evidence drivers, QEMU linux-user 8.2, Apple clang, GTest, CTest.

**Spec:** `docs/superpowers/specs/2026-10-24-arm-sme-stateful-backend-design.md`

## Global Constraints

- Work on a `feat/arm-sme-stateful-backend` branch cut from the merged design PR; CI only runs for PRs targeting `main`, so retarget after the design PR merges. Never write in the repository root checkout.
- C++20, CMake >= 3.20, LLVM/MLIR 20+ API guards via `llvm/Config/llvm-config.h` (`LLVM_VERSION_MAJOR >= 21`).
- Use `Op::create(builder, ...)`, `mlir::cast<T>`, `isa<T>`, `dyn_cast<T>`; `-Werror=deprecated-declarations` makes the deprecated forms build errors.
- Generic `lib/Mapping`, `lib/Machine`, `lib/Conversion/MicroMapping` and `runtime/` code must never branch on `sme`, `arm-sme` or any target name. Target policy lives in the `arm-sme` package, the `.llkmap` files and the machine profile.
- The f32 16x16x16 initialized contraction is the only executable signature. bf16, i8, other `sme_*` keys, SME2, multi-tile, non-512-bit SVL and ZA preservation across calls are out of scope.
- `referenceGroupsLowered` must be zero for every accepted selected-target run; `backendGroupsRealized` must be greater than zero.
- Numerical data is integer-valued f32 so every product and partial sum is exact; expected results are literal bit patterns. No fast-math.
- TDD: write the failing test first, run it to see the failure, implement minimally, run it to pass, commit.
- Build: `cmake -S . -B build -G Ninja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build -DLLK_BUILD_TOOLS=ON -DLLK_BUILD_E2E_TESTS=ON`; `ninja -C build check-llk`.
- The repository currently links LLVM/MLIR through the monolithic `LLK::MLIR` target. Do not add component archives.
- The native evidence test runs only on an Apple arm64 host whose `sysctl hw.optional.arm.FEAT_SME` is 1; it is not part of CI. The QEMU test runs in CI and needs `aarch64-linux-gnu-gcc` and `qemu-aarch64`.

---

## File Map

| File | Responsibility |
|---|---|
| `include/LLK/Mapping/CodegenRequirements.h`, `lib/Mapping/CodegenRequirements.cpp` | Add streaming vector length and invocation attributes; check SVL against the host |
| `include/LLK/Mapping/MappingTarget.h` | Add the post-bufferization target hook |
| `lib/Conversion/MicroMapping/MappedCompilation.cpp` | Invoke the post-bufferization hook after One-Shot Bufferize |
| `include/LLK/Runtime/MappedObject.h`, `runtime/MappedObject.cpp` | Apply wrapper attributes; record SVL and attributes in the identity sidecar |
| `runtime/MappedExecutable.cpp` | Probe host SVL and apply wrapper attributes on the JIT path |
| `tools/llk-compile/llk-compile.cpp` | Pass the target's invocation attributes into AOT emission and the identity record |
| `include/LLK/Target/AArch64/Mapping/SMEMappingTarget.h`, `lib/Target/AArch64/Mapping/SMEMappingTarget.cpp` | SME target wrapper: contract, emitters, both pipelines |
| `lib/Target/AArch64/Mapping/SMEBundleLowering.cpp` | `SMEEmitter`: `sme_mma_za` and movement handoff |
| `lib/Target/AArch64/Mapping/SMEVectorLowering.cpp` | Pre-bufferization tiling/vectorization and contract-to-outerproduct |
| `lib/Target/AArch64/Mapping/SMEArmLowering.cpp` | Post-bufferization ArmSME conversion and realization accounting |
| `lib/Target/CMakeLists.txt` | Build the three new SME translation units |
| `test/Mapping/codegen_requirements.cpp`, `test/Execution/mapped_codegen_requirements.cpp` | Framework unit tests |
| `test/Mapping/backend_pipeline_hooks.cpp` | Post-bufferization hook ordering test |
| `test/Mapping/arm_sme_backend.cpp` | SME emitter, lowering, refusal and receipt tests |
| `test/Mapping/arm_targets.cpp`, `test/Mapping/arm_placement.cpp` | Narrow the old SME static-only assertions |
| `test/Mapping/CMakeLists.txt` | Register the new test sources |
| `test/Execution/aot_object_emission.py` | Extend identity checks |
| `test/Execution/arm_sme_evidence_common.py` | Shared SME harness/shim/disassembly/identity helpers |
| `test/Execution/arm_sme_native_evidence.py` | Native Apple M5 T3 driver |
| `test/Execution/arm_sme_qemu_evidence.py` | QEMU SME1 T2 driver |
| `test/Execution/arm_sme_evidence_test.py` | Host-independent parser/regex tests |
| `test/FileCheck.cmake`, `test/Execution/CMakeLists.txt` | Register the new tests |
| `.github/workflows/ci.yml`, `.github/workflows/coverage.yml` | Run the QEMU SME evidence in CI |
| `docs/reviews/2026-10-24-arm-sme-stateful-backend-acceptance.md` | Revision-pinned T2/T3 ledger |
| `docs/reviews/2026-10-24-arm-sve-executable-acceptance.md`, `docs/reviews/arm-mapping-targets-acceptance.md`, `docs/manual/source/concepts.rst` | Correct superseded static-only claims |

---

## Task 1: Extend the codegen contract with streaming vector length and invocation attributes

**Files:**
- Modify: `include/LLK/Mapping/CodegenRequirements.h`
- Modify: `lib/Mapping/CodegenRequirements.cpp`
- Test: `test/Mapping/codegen_requirements.cpp`

**Interfaces:**
- Produces: `TargetCodegenRequirements{architecture, cpu, requiredFeatures, vectorLengthBits, streamingVectorLengthBits, invocationAttributes}`
- Produces: `checkHostRequirements(requirements, hostArchitecture, hostFeatures, hostVectorLengthBits, hostStreamingVectorLengthBits = std::nullopt)`

- [ ] **Step 1: Write the failing tests**

Append to `test/Mapping/codegen_requirements.cpp`:

```cpp
TEST(CodegenRequirements, ChecksStreamingVectorLengthIndependently) {
  // SME has no non-streaming vector-length requirement; its SVL is separate.
  TargetCodegenRequirements requirements{"aarch64", "", {"sme"},
                                         std::nullopt, 512};
  llvm::StringMap<bool> features{{"sme", true}};
  EXPECT_FALSE(static_cast<bool>(checkHostRequirements(
      requirements, "aarch64", features, std::nullopt, 512)));
  // A non-streaming length of 128 is irrelevant to an SME-only requirement.
  EXPECT_FALSE(static_cast<bool>(checkHostRequirements(
      requirements, "aarch64", features, 128, 512)));
  EXPECT_TRUE(static_cast<bool>(checkHostRequirements(
      requirements, "aarch64", features, std::nullopt, std::nullopt)));
  EXPECT_TRUE(static_cast<bool>(checkHostRequirements(
      requirements, "aarch64", features, std::nullopt, 256)));
}

TEST(CodegenRequirements, ReportsStreamingVectorLengthRefusals) {
  TargetCodegenRequirements requirements{"aarch64", "", {"sme"},
                                         std::nullopt, 512};
  llvm::StringMap<bool> features{{"sme", true}};

  llvm::Error unknown = checkHostRequirements(
      requirements, "aarch64", features, std::nullopt, std::nullopt);
  ASSERT_TRUE(static_cast<bool>(unknown));
  std::string unknownText = llvm::toString(std::move(unknown));
  EXPECT_NE(unknownText.find("host streaming vector length is unknown"),
            std::string::npos) << unknownText;
  EXPECT_NE(unknownText.find("requires 512 bits"), std::string::npos)
      << unknownText;

  llvm::Error mismatch = checkHostRequirements(
      requirements, "aarch64", features, std::nullopt, 256);
  ASSERT_TRUE(static_cast<bool>(mismatch));
  std::string mismatchText = llvm::toString(std::move(mismatch));
  EXPECT_NE(mismatchText.find("host streaming vector length is 256 bits"),
            std::string::npos) << mismatchText;
}

TEST(CodegenRequirements, KeepsInvocationAttributesAsTargetPolicy) {
  TargetCodegenRequirements requirements{
      "aarch64", "", {"sme"}, std::nullopt, 512,
      {"aarch64_pstate_sm_enabled"}};
  EXPECT_EQ(requirements.streamingVectorLengthBits, 512);
  EXPECT_EQ(requirements.invocationAttributes,
            std::vector<std::string>{"aarch64_pstate_sm_enabled"});
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ninja -C build LLKMappingTests
./build/bin/LLKMappingTests --gtest_filter=CodegenRequirements.*
```

Expected: compile failure (`streamingVectorLengthBits` and the 5-argument call do not exist).

- [ ] **Step 3: Add the fields and the check**

In `include/LLK/Mapping/CodegenRequirements.h`, replace the struct with:

```cpp
struct TargetCodegenRequirements {
  std::string architecture;
  std::string cpu;
  std::vector<std::string> requiredFeatures;
  /// Exact implemented vector length in bits, when the target's code assumes
  /// one. Nullopt means the target makes no vector-length claim.
  std::optional<int64_t> vectorLengthBits = std::nullopt;
  /// Exact implemented *streaming* vector length in bits (Arm SME SVL). This is
  /// independent of vectorLengthBits: a machine can have no non-streaming SVE
  /// and still implement a streaming length.
  std::optional<int64_t> streamingVectorLengthBits = std::nullopt;
  /// LLVM function attributes the exported C-interface wrapper must carry.
  /// Empty for targets whose invocation entry needs no ABI annotation.
  std::vector<std::string> invocationAttributes{};
};

llvm::Error checkHostRequirements(const TargetCodegenRequirements &requirements,
                                  llvm::StringRef hostArchitecture,
                                  const llvm::StringMap<bool> &hostFeatures,
                                  std::optional<int64_t> hostVectorLengthBits,
                                  std::optional<int64_t>
                                      hostStreamingVectorLengthBits =
                                          std::nullopt);
```

In `lib/Mapping/CodegenRequirements.cpp`, add the SVL check after the existing `vectorLengthBits` block:

```cpp
  if (requirements.streamingVectorLengthBits) {
    if (!hostStreamingVectorLengthBits)
      return requirementError(
          "host streaming vector length is unknown but '" +
          requirements.architecture + "' requires " +
          std::to_string(*requirements.streamingVectorLengthBits) + " bits");
    if (*hostStreamingVectorLengthBits !=
        *requirements.streamingVectorLengthBits)
      return requirementError(
          "host streaming vector length is " +
          std::to_string(*hostStreamingVectorLengthBits) +
          " bits but the selected target requires " +
          std::to_string(*requirements.streamingVectorLengthBits) + " bits");
  }
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
ninja -C build LLKMappingTests
./build/bin/LLKMappingTests --gtest_filter=CodegenRequirements.*
```

Expected: PASS, including every pre-existing `CodegenRequirements` case (the fifth argument defaults to `std::nullopt`).

- [ ] **Step 5: Commit**

```bash
git add include/LLK/Mapping/CodegenRequirements.h \
        lib/Mapping/CodegenRequirements.cpp \
        test/Mapping/codegen_requirements.cpp
git commit -m "feat(mapping): model streaming vector length in codegen requirements (#253)"
```

---

## Task 2: Apply invocation attributes to the exported wrapper and record them

**Files:**
- Modify: `include/LLK/Runtime/MappedObject.h`
- Modify: `runtime/MappedObject.cpp`
- Modify: `runtime/MappedExecutable.cpp`
- Modify: `tools/llk-compile/llk-compile.cpp`
- Test: `test/Execution/mapped_codegen_requirements.cpp`
- Test: `test/Execution/aot_object_emission.py`

**Interfaces:**
- Produces: `MappedObjectOptions::invocationAttributes`
- Produces: `MappedObjectIdentity::streamingVectorLengthBits` and `invocationAttributes`
- Produces: `llvm::Error llk::applyInvocationAttributes(llvm::Module &module, llvm::StringRef wrapperSymbol, llvm::ArrayRef<std::string> attributes)`
- Consumes: `TargetCodegenRequirements::invocationAttributes` and `streamingVectorLengthBits` from Task 1

- [ ] **Step 1: Write the failing tests**

Append to `test/Execution/mapped_codegen_requirements.cpp` (add the includes):

```cpp
#include "LLK/Runtime/MappedObject.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
```

```cpp
TEST(MappedCodegenRequirements, AppliesInvocationAttributesToTheWrapper) {
  llvm::LLVMContext context;
  llvm::Module module("m", context);
  auto *wrapper = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), false),
      llvm::GlobalValue::ExternalLinkage, "_mlir_ciface_k", module);
  ASSERT_FALSE(wrapper->hasFnAttribute("aarch64_pstate_sm_enabled"));

  llvm::Error error = llk::applyInvocationAttributes(
      module, "_mlir_ciface_k", {"aarch64_pstate_sm_enabled"});
  ASSERT_FALSE(static_cast<bool>(error));
  EXPECT_TRUE(wrapper->hasFnAttribute("aarch64_pstate_sm_enabled"));

  // An empty attribute list is a no-op even when the wrapper is absent.
  EXPECT_FALSE(static_cast<bool>(
      llk::applyInvocationAttributes(module, "_mlir_ciface_missing", {})));

  // Naming a missing wrapper with attributes is an error, not a silent skip.
  llvm::Error missing = llk::applyInvocationAttributes(
      module, "_mlir_ciface_missing", {"aarch64_pstate_sm_enabled"});
  ASSERT_TRUE(static_cast<bool>(missing));
  EXPECT_NE(llvm::toString(std::move(missing)).find("_mlir_ciface_missing"),
            std::string::npos);
}
```

In `test/Execution/aot_object_emission.py`, extend the required key list and add the reference/selected assertions:

```python
    for key in ("schemaVersion", "compilerVersion", "llvmVersion",
                "targetTriple", "targetFeatures", "entrySymbol",
                "wrapperSymbol", "abiVersion", "abiHash", "sourceGraphHash",
                "planId", "machineHash", "targetHash", "selectedGroupsVerified",
                "backendGroupsRealized", "referenceGroupsLowered",
                "streamingVectorLengthBits", "invocationAttributes",
                "executionIdentity"):
        assert key in record, "identity has no " + key
    assert record["schemaVersion"] == 2
    assert record["streamingVectorLengthBits"] is None
    assert record["invocationAttributes"] == []
```

And in the selected-record block:

```python
    assert selected_record["schemaVersion"] == 2
    # arm-sve is a non-streaming target: no SVL claim, no wrapper attributes.
    assert selected_record["streamingVectorLengthBits"] is None
    assert selected_record["invocationAttributes"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ninja -C build MappedCodegenRequirementsTest AOTObjectEmission
ctest --test-dir build -R '^(MappedCodegenRequirementsTest|AOTObjectEmission)$' --output-on-failure
```

Expected: compile failure (`applyInvocationAttributes` missing) and identity-key failures.

- [ ] **Step 3: Add the helper, options, identity fields and emission**

In `include/LLK/Runtime/MappedObject.h`:

```cpp
#include "llvm/IR/Module.h"
#include <optional>
```

```cpp
struct MappedObjectOptions {
  std::string triple;
  std::string cpu;
  std::vector<std::string> features;
  /// LLVM function attributes to attach to the exported `_mlir_ciface_*`
  /// wrapper. Supplied by the target's codegen contract; empty for targets
  /// whose entry needs no ABI annotation.
  std::vector<std::string> invocationAttributes;
};
```

```cpp
struct MappedObjectIdentity {
  uint32_t schemaVersion = 2;
  // ... existing fields unchanged ...
  std::optional<int64_t> streamingVectorLengthBits;
  std::vector<std::string> invocationAttributes;
  std::string executionIdentity;
};

/// Attach `attributes` to `wrapperSymbol` in `module`.
///
/// An empty attribute list is a no-op and does not require the wrapper to
/// exist. A nonempty list naming a missing wrapper is an error: silently
/// dropping a required ABI attribute would emit an object whose entry runs in
/// the wrong streaming state.
llvm::Error applyInvocationAttributes(llvm::Module &module,
                                      llvm::StringRef wrapperSymbol,
                                      llvm::ArrayRef<std::string> attributes);
```

In `runtime/MappedObject.cpp` (add `#include "llvm/IR/Function.h"` if not already transitively present):

```cpp
llvm::Error applyInvocationAttributes(llvm::Module &module,
                                      llvm::StringRef wrapperSymbol,
                                      llvm::ArrayRef<std::string> attributes) {
  if (attributes.empty())
    return llvm::Error::success();
  llvm::Function *wrapper = module.getFunction(wrapperSymbol);
  if (!wrapper)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invocation wrapper '" + wrapperSymbol.str() +
                                       "' is missing from the LLVM module");
  for (const std::string &attribute : attributes)
    wrapper->addFnAttr(attribute);
  return llvm::Error::success();
}
```

After the wrapper lookup in `emitMappedObject` (immediately after the `if (!llvmModule->getFunction(wrapper))` check):

```cpp
  if (llvm::Error error = applyInvocationAttributes(
          *llvmModule, wrapper, options.invocationAttributes))
    return std::move(error);
```

In `writeMappedObjectIdentity`, add after the existing group counters:

```cpp
  if (identity.streamingVectorLengthBits)
    record["streamingVectorLengthBits"] = *identity.streamingVectorLengthBits;
  else
    record["streamingVectorLengthBits"] = nullptr;
  llvm::json::Array invocationAttributes;
  for (const std::string &attribute : identity.invocationAttributes)
    invocationAttributes.push_back(attribute);
  record["invocationAttributes"] = std::move(invocationAttributes);
```

In `runtime/MappedExecutable.cpp`, add the host streaming-length probe next to
the existing SVE probe and pass it to the host check. Without this wiring the
new contract field would be inert and every selected SME JIT run would refuse
with "host streaming vector length is unknown".

In the existing `#if defined(__linux__) && defined(__aarch64__)` block at the
top of the file, next to `kPrSveGetVl`:

```cpp
#ifndef HWCAP2_SME
#define HWCAP2_SME (1UL << 23)
#endif
/// prctl() request code for PR_SME_GET_VL.
constexpr int kPrSmeGetVl = 64;
```

After `detectHostVectorLengthBits()`:

```cpp
std::optional<int64_t> detectHostStreamingVectorLengthBits() {
#if defined(__linux__) && defined(__aarch64__)
  unsigned long hwcap2 = getauxval(AT_HWCAP2);
  if (!(hwcap2 & HWCAP2_SME))
    return std::nullopt;
  long value = prctl(kPrSmeGetVl, 0, 0, 0, 0);
  if (value < 0)
    return std::nullopt;
  return mlir::llk::mapping::vectorLengthBitsFromPrctl(value);
#else
  return std::nullopt;
#endif
}
```

Replace the four-argument host check call with:

```cpp
    if (llvm::Error error = mlir::llk::mapping::checkHostRequirements(
            *options.selectedTarget, hostArchitecture, hostFeatures,
            detectHostVectorLengthBits(),
            detectHostStreamingVectorLengthBits()))
      return error;
```

Add `#include "LLK/Runtime/MappedObject.h"` to `runtime/MappedExecutable.cpp`
(if it is not already included transitively). Then, after
`translateModuleToLLVMIR` succeeds (`if (!llvmModule) { ... }` block) and
before the evidence sink:

```cpp
  if (options.selectedTarget) {
    std::string wrapper = "_mlir_ciface_" + prepared.entrySymbol;
    if (llvm::Error error = applyInvocationAttributes(
            *llvmModule, wrapper, options.selectedTarget->invocationAttributes))
      return std::move(error);
  }
```

In `tools/llk-compile/llk-compile.cpp`, after `objectOptions.features = std::move(features);`:

```cpp
    if (compiled->codegenRequirements) {
      objectOptions.invocationAttributes =
          compiled->codegenRequirements->invocationAttributes;
      record.streamingVectorLengthBits =
          compiled->codegenRequirements->streamingVectorLengthBits;
      record.invocationAttributes = objectOptions.invocationAttributes;
    }
```

The `record` is declared later in the function; move the two `record` assignments to the block where the other `record` fields are set:

```cpp
    record.streamingVectorLengthBits =
        compiled->codegenRequirements
            ? compiled->codegenRequirements->streamingVectorLengthBits
            : std::nullopt;
    record.invocationAttributes = objectOptions.invocationAttributes;
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
ninja -C build MappedCodegenRequirementsTest AOTObjectEmission llk-compile
ctest --test-dir build -R '^(MappedCodegenRequirementsTest|AOTObjectEmission)$' --output-on-failure
```

Expected: PASS. The AOT test's sidecar now records `schemaVersion = 2`, `streamingVectorLengthBits = null` and an empty `invocationAttributes` for both the reference and selected SVE objects.

- [ ] **Step 5: Commit**

```bash
git add include/LLK/Runtime/MappedObject.h runtime/MappedObject.cpp \
        runtime/MappedExecutable.cpp tools/llk-compile/llk-compile.cpp \
        test/Execution/mapped_codegen_requirements.cpp \
        test/Execution/aot_object_emission.py
git commit -m "feat(runtime): apply target invocation attributes to the exported wrapper (#253)"
```

---

## Task 3: Add the post-bufferization selected-backend hook

**Files:**
- Modify: `include/LLK/Mapping/MappingTarget.h`
- Modify: `lib/Conversion/MicroMapping/MappedCompilation.cpp`
- Create: `test/Mapping/backend_pipeline_hooks.cpp`
- Modify: `test/Mapping/CMakeLists.txt`

**Interfaces:**
- Produces: `virtual llvm::Error MappingTarget::buildPostBufferizationBackendPipeline(mlir::OpPassManager &) const` (default `llvm::Error::success()`)
- Consumes: the existing `buildSelectedBackendPipeline` and `arm-sve` fixture helpers from `test/Mapping/arm_target_fixture.h`

- [ ] **Step 1: Write the failing test**

Create `test/Mapping/backend_pipeline_hooks.cpp`:

```cpp
//===- backend_pipeline_hooks.cpp - Target pipeline ordering -------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include "arm_target_fixture.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <gtest/gtest.h>

#include <memory>

namespace {

struct ProbeState {
  bool ran = false;
  bool sawMemref = false;
  bool sawLinalg = false;
};

class ProbePass : public mlir::PassWrapper<ProbePass,
                                           mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProbePass)
  explicit ProbePass(ProbeState *state) : state_(state) {}
  mlir::StringRef getArgument() const final { return "llk-test-hook-probe"; }

  void runOnOperation() override {
    state_->ran = true;
    getOperation().walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::linalg::LinalgOp>(op))
        state_->sawLinalg = true;
      for (mlir::Value value : op->getOperands())
        state_->sawMemref |= mlir::isa<mlir::MemRefType>(value.getType());
      for (mlir::Value value : op->getResults())
        state_->sawMemref |= mlir::isa<mlir::MemRefType>(value.getType());
    });
  }

private:
  ProbeState *state_;
};

/// Delegates to the real arm-sve target and records where the second hook runs.
class HookProbeTarget : public mlir::llk::mapping::MappingTarget {
public:
  HookProbeTarget(std::unique_ptr<mlir::llk::mapping::MappingTarget> inner,
                  ProbeState *state)
      : inner_(std::move(inner)), state_(state) {}

  mlir::StringRef name() const override { return inner_->name(); }
  const mlir::llk::machine::MachineModel &machine() const override {
    return inner_->machine();
  }
  const mlir::llk::mapping::LayoutRegistry &layouts() const override {
    return inner_->layouts();
  }
  const mlir::llk::mapping::RuleRegistry &rules() const override {
    return inner_->rules();
  }
  bool isKnownEmitter(mlir::StringRef key) const override {
    return inner_->isKnownEmitter(key);
  }
  std::optional<mlir::llk::mapping::TargetCodegenRequirements>
  codegenRequirements() const override {
    return inner_->codegenRequirements();
  }
  const mlir::llk::mapping::LatencyProvider *latencyProvider() const override {
    return inner_->latencyProvider();
  }
  std::unique_ptr<mlir::llk::mapping::TargetEmitter>
  createEmitter(mlir::StringRef key) const override {
    return inner_->createEmitter(key);
  }
  std::unique_ptr<mlir::llk::mapping::TargetEmitter>
  createEmitter() const override {
    return inner_->createEmitter();
  }
  llvm::Error
  buildSelectedBackendPipeline(mlir::OpPassManager &pm) const override {
    return inner_->buildSelectedBackendPipeline(pm);
  }
  llvm::Error buildPostBufferizationBackendPipeline(
      mlir::OpPassManager &pm) const override {
    pm.addPass(std::make_unique<ProbePass>(state_));
    return llvm::Error::success();
  }

private:
  std::unique_ptr<mlir::llk::mapping::MappingTarget> inner_;
  ProbeState *state_;
};

TEST(BackendPipelineHooks, PostBufferizationHookSeesMemrefsAndOpenLinalg) {
  auto fixture = arm_test::mapFixture(
      "arm-sve",
      "test/Conversion/MicroMapping/Inputs/arm/vector_add_f32.mlir");
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  ProbeState state;
  HookProbeTarget probe(std::move(fixture->target), &state);
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_add_f32";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled =
      llk::compileMappedKernel(*source, probe, fixture->plan, options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  EXPECT_TRUE(state.ran) << "the post-bufferization hook never ran";
  EXPECT_TRUE(state.sawMemref) << "the hook ran before bufferization";
  EXPECT_TRUE(state.sawLinalg)
      << "the hook ran after convert-linalg-to-loops";
}
```

Add `backend_pipeline_hooks.cpp` to `test/Mapping/CMakeLists.txt` after `arm_sve_backend.cpp`.

- [ ] **Step 2: Run the test to verify it fails**

```bash
ninja -C build LLKMappingTests
./build/bin/LLKMappingTests --gtest_filter=BackendPipelineHooks.*
```

Expected: compile failure (`buildPostBufferizationBackendPipeline` does not exist and is not virtual).

- [ ] **Step 3: Add the hook and invoke it**

In `include/LLK/Mapping/MappingTarget.h`, after `buildSelectedBackendPipeline`:

```cpp
  /// Add this target's passes that must run after One-Shot Bufferize and
  /// before `convert-linalg-to-loops`. Targets that lower before bufferization
  /// leave this as the default no-op. A target that needs memrefs (Arm SME tile
  /// operations) overrides it rather than making a target pass own the generic
  /// pipeline's bufferization.
  virtual llvm::Error
  buildPostBufferizationBackendPipeline(mlir::OpPassManager &) const {
    return llvm::Error::success();
  }
```

In `lib/Conversion/MicroMapping/MappedCompilation.cpp`, between the bufferize pass and `createConvertLinalgToLoopsPass`:

```cpp
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
  if (backend == MappedBackend::SelectedTarget && target)
    if (llvm::Error error = target->buildPostBufferizationBackendPipeline(pm))
      return error;
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
ninja -C build LLKMappingTests
./build/bin/LLKMappingTests --gtest_filter='BackendPipelineHooks.*:ArmSVEBackend.*'
```

Expected: PASS. The probe sees memrefs (bufferization ran) and at least one
Linalg op (loops have not been rolled yet).

The committed test adds a test-only `MarkerLinalgPass` after the inner target's
pre-bufferization pipeline. The arm-sve pass consumes its own selected
`linalg.generic` before bufferization, so without a dead marker
`convert-linalg-to-loops` would have nothing to act on and the hook's position
relative to it would be unobservable. The marker is dead code, is not part of
any selected group, and does not affect the realized count; the reviewer should
confirm it preserves the ordering claim without weakening the test.

- [ ] **Step 5: Commit**

```bash
git add include/LLK/Mapping/MappingTarget.h \
        lib/Conversion/MicroMapping/MappedCompilation.cpp \
        test/Mapping/backend_pipeline_hooks.cpp test/Mapping/CMakeLists.txt
git commit -m "feat(mapping): add the post-bufferization selected-backend hook (#253)"
```

---

## Task 4: Make `arm-sme` a selective executable target with `sme_mma_za` and movement handoff

**Files:**
- Modify: `include/LLK/Target/AArch64/Mapping/SMEMappingTarget.h`
- Modify: `lib/Target/AArch64/Mapping/SMEMappingTarget.cpp`
- Create: `lib/Target/AArch64/Mapping/SMEBundleLowering.cpp`
- Modify: `lib/Target/CMakeLists.txt`
- Create: `test/Mapping/arm_sme_backend.cpp`
- Modify: `test/Mapping/CMakeLists.txt`
- Modify: `test/Mapping/arm_targets.cpp`
- Modify: `test/Mapping/arm_placement.cpp`

**Interfaces:**
- Produces: `mlir::llk::target::sme::createSMEEmitter(llvm::StringRef key)`
- Produces: `mlir::llk::target::sme::buildSMESelectedBackendPipeline(mlir::OpPassManager &)` (declared here; defined in Task 5)
- Produces: `mlir::llk::target::sme::buildSMEPostBufferizationPipeline(mlir::OpPassManager &)` (declared here; defined in Task 5)
- Produces: emitter handoff attribute `llk.sme.selected_handoff` (unit)
- Consumes: Task 1's `TargetCodegenRequirements` fields and Task 3's hook

- [ ] **Step 1: Write the failing tests**

Create `test/Mapping/arm_sme_backend.cpp`:

```cpp
//===- arm_sme_backend.cpp - Selected SME lowering -----------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Framework E11 (issue #253): `arm-sme` is executable for exactly one f32
// initialized 16x16x16 contraction plus the movement keys its fixture needs.
// These tests prove the emitter boundary, the codegen contract, and that an
// unimplemented dtype is refused rather than reference-lowered.

#include "LLK/Conversion/FrameworkCapabilities.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"

#include "arm_target_fixture.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

namespace {

constexpr const char *kContraction =
    "test/Conversion/MicroMapping/Inputs/arm/contraction_f32.mlir";
constexpr const char *kContractionBf16 =
    "test/Conversion/MicroMapping/Inputs/arm/contraction_bf16.mlir";

TEST(ArmSMEBackend, OnlyContractionAndMovementKeysLower) {
  auto fixture = arm_test::mapFixture("arm-sme", kContraction);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  for (const char *key :
       {"sme_mma_za", "sme_copy", "sme_tile_copy", "sme_tile_store"})
    EXPECT_TRUE(fixture->target->createEmitter(key)->hasLowering()) << key;
  for (const char *key :
       {"sme_vector_add", "sme_vector_mul", "sme_reduce", "sme_vector_silu"})
    EXPECT_FALSE(fixture->target->createEmitter(key)->hasLowering()) << key;

  auto requirements = fixture->target->codegenRequirements();
  ASSERT_TRUE(requirements.has_value());
  EXPECT_EQ(requirements->architecture, "aarch64");
  EXPECT_EQ(requirements->requiredFeatures,
            std::vector<std::string>{"sme"});
  EXPECT_EQ(requirements->vectorLengthBits, std::nullopt);
  EXPECT_EQ(requirements->streamingVectorLengthBits, 512);
  EXPECT_EQ(requirements->invocationAttributes,
            std::vector<std::string>{"aarch64_pstate_sm_enabled"});
}

TEST(ArmSMEBackend, EmitterValidatesTheWholeContractionGroup) {
  auto fixture = arm_test::mapFixture("arm-sme", kContraction);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  mlir::Operation *mma = nullptr;
  fixture->bound.module->walk([&](mlir::Operation *op) {
    if (!mma && op->getName().getStringRef() == "micro.mma")
      mma = op;
  });
  ASSERT_NE(mma, nullptr);

  auto emitter = fixture->target->createEmitter("sme_mma_za");
  ASSERT_NE(emitter, nullptr);
  EXPECT_TRUE(emitter->hasLoweringFor({mma}));

  mlir::IRRewriter rewriter(fixture->bound.module->getContext());
  mlir::llk::mapping::TargetLoweringContext context{
      fixture->target->machine(), {}, {}};
  mlir::llk::mapping::TargetBundle bundle{"sme.mma.f32", {}, "sme_mma_za"};
  ASSERT_FALSE(static_cast<bool>(
      emitter->lower({mma}, bundle, context, rewriter)));
  EXPECT_TRUE(mma->hasAttr("llk.sme.selected_handoff"));
}

TEST(ArmSMEBackend, SelectedBf16ContractionIsRefused) {
  auto fixture = arm_test::mapFixture("arm-sme", kContractionBf16);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_contraction";  // the bf16 fixture keeps this name
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
  ASSERT_FALSE(static_cast<bool>(compiled));
  const std::string message = llvm::toString(compiled.takeError());
  EXPECT_NE(message.find("realized no selected compute groups"),
            std::string::npos) << message;
}
```

Add `arm_sme_backend.cpp` to `test/Mapping/CMakeLists.txt`.

In `test/Mapping/arm_targets.cpp`, replace `SmeHasNoExecutableBackend` with:

```cpp
// SME is executable for exactly one signature plus the fixture's movement
// keys. Every other key keeps the explicit rejection.
TEST(ArmTargets, SmeExecutableSubsetIsExplicit) {
  mlir::MLIRContext context;
  auto target = loadRegistered(kConfigurations[1]); // arm-sme
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto requirements = (*target)->codegenRequirements();
  ASSERT_TRUE(requirements.has_value());
  EXPECT_EQ(requirements->architecture, "aarch64");
  EXPECT_EQ(requirements->requiredFeatures,
            std::vector<std::string>{"sme"});
  EXPECT_EQ(requirements->streamingVectorLengthBits, 512);
  for (const RuleDef &rule : (*target)->rules().all()) {
    bool executable = rule.emitter == "sme_mma_za" ||
                      rule.emitter == "sme_copy" ||
                      rule.emitter == "sme_tile_copy" ||
                      rule.emitter == "sme_tile_store";
    auto emitter = (*target)->createEmitter(rule.emitter);
    ASSERT_NE(emitter, nullptr);
    EXPECT_EQ(emitter->hasLowering(), executable) << rule.emitter;
    if (!executable) {
      mlir::IRRewriter rewriter(&context);
      TargetLoweringContext lowering{(*target)->machine(), {}, {}};
      TargetBundle bundle{rule.bundle, {}, rule.emitter};
      EXPECT_NE(llvm::toString(emitter->lower({}, bundle, lowering, rewriter))
                    .find("target_lowering_unsupported"),
                std::string::npos);
    }
  }
}
```

In `test/Mapping/arm_placement.cpp`, replace
`SelectedTargetCompilationRefusesUnimplementedSignatures` with:

```cpp
// arm-sve still rejects the contraction (`sve_mma` has no lowering); arm-sme
// executes the f32 contraction but rejects bf16. Every refusal comes from the
// shared group accounting, not from a missing code-generation contract.
TEST(ArmPlacement, SelectedTargetCompilationRefusesUnimplementedSignatures) {
  struct RefusalCase {
    const char *targetName;
    const char *inputPath;
  };
  const RefusalCase cases[] = {{"arm-sve", kF32Path.data()},
                               {"arm-sve", kBf16Path.data()},
                               {"arm-sme", kBf16Path.data()}};
  for (const RefusalCase &testCase : cases) {
    SCOPED_TRACE(std::string(testCase.targetName) + "/" + testCase.inputPath);
    llvm::Expected<arm_test::MappedFixture> mapped =
        arm_test::mapFixture(testCase.targetName, testCase.inputPath);
    ASSERT_TRUE(bool(mapped)) << llvm::toString(mapped.takeError());

    ::llk::MappedCompileOptions options;
    options.backend = ::llk::MappedBackend::SelectedTarget;
    options.stop = ::llk::MappedStop::Lowered;
    options.requireExecutable = false;

    llvm::Expected<::llk::MappedCompilation> compiled =
        ::llk::compileMappedKernel(*mapped->source, *mapped->target,
                                   mapped->plan, options);
    ASSERT_FALSE(bool(compiled))
        << "a target accepted selected-target compilation for an unimplemented "
           "signature";
    const std::string message = llvm::toString(compiled.takeError());
    EXPECT_NE(message.find("realized no selected compute groups"),
              std::string::npos) << message;
  }
}
```

Replace `SelectedBundlesVerifyButCannotLower` with:

```cpp
// Every bundle the map selected verifies, and every key this target does not
// implement keeps the explicit `target_lowering_unsupported` refusal. Keys
// that do implement a lowering are proven end to end in ArmSMEBackend.
TEST(ArmPlacement, SelectedBundlesVerifyAndOnlyImplementedKeysLower) {
  for (const ArmContractionCase &testCase : kCases) {
    SCOPED_TRACE(std::string(testCase.targetName) + "/" + testCase.dtype);
    llvm::Expected<arm_test::MappedFixture> mapped =
        arm_test::mapFixture(testCase.targetName, testCase.inputPath);
    ASSERT_TRUE(bool(mapped)) << llvm::toString(mapped.takeError());
    // Both Arm targets now publish a code-generation contract.
    EXPECT_TRUE(mapped->target->codegenRequirements().has_value());

    IRRewriter rewriter(mapped->context.get());
    ASSERT_FALSE(mapped->plan.placements.empty());
    for (const PlanPlacement &placement : mapped->plan.placements) {
      SCOPED_TRACE("rule " + placement.rule);
      std::unique_ptr<TargetEmitter> emitter =
          mapped->target->createEmitter(placement.bundle.emitterKey);
      ASSERT_NE(emitter, nullptr) << placement.bundle.emitterKey;
      EXPECT_EQ(llvm::toString(emitter->verify(placement.bundle)), "");
      if (emitter->hasLowering())
        continue;
      TargetLoweringContext lowering{
          mapped->target->machine(), mapped->plan.placements,
          mapped->plan.connectionPlans, placement.instance};
      const std::string refusal = llvm::toString(
          emitter->lower({}, placement.bundle, lowering, rewriter));
      EXPECT_NE(refusal.find("target_lowering_unsupported"), std::string::npos)
          << refusal;
    }
  }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ninja -C build LLKMappingTests
./build/bin/LLKMappingTests --gtest_filter='ArmSMEBackend.*:ArmTargets.*'
```

Expected: compile failure (`createSMEEmitter` missing) and the old
`SmeHasNoExecutableBackend` expectation failing once the target gains a contract.

- [ ] **Step 3: Implement the SME target and emitter**

`include/LLK/Target/AArch64/Mapping/SMEMappingTarget.h` becomes:

```cpp
#ifndef LLK_TARGET_AARCH64_MAPPING_SMEMAPPINGTARGET_H
#define LLK_TARGET_AARCH64_MAPPING_SMEMAPPINGTARGET_H

#include "LLK/Mapping/MappingTarget.h"

#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir::llk::target::sme {

/// Declared mapping emitter keys. `sme_mma_za` and the three movement keys
/// carry a lowering; every other key keeps its explicit rejection.
llvm::ArrayRef<llvm::StringRef> emitterKeys();

/// An emitter for one declared key. Only `sme_mma_za`, `sme_copy`,
/// `sme_tile_copy` and `sme_tile_store` lower their covered groups.
std::unique_ptr<mapping::TargetEmitter> createSMEEmitter(llvm::StringRef key);

/// Adds the pre-bufferization scalable-vector lowering for the selected
/// contraction. Defined in SMEVectorLowering.cpp.
void buildSMESelectedBackendPipeline(mlir::OpPassManager &pm);

/// Adds the post-bufferization ArmSME conversion and realization accounting.
/// Defined in SMEArmLowering.cpp.
void buildSMEPostBufferizationPipeline(mlir::OpPassManager &pm);

/// Loads machines/arm-sme-v2.yaml and mapping/arm-sme/{layouts,rules}.llkmap
/// below configurationRoot. Returns the target with the f32 contraction
/// executable subset and the AArch64/SME streaming contract.
llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath = {});

} // namespace mlir::llk::target::sme

#endif // LLK_TARGET_AARCH64_MAPPING_SMEMAPPINGTARGET_H
```

`lib/Target/AArch64/Mapping/SMEMappingTarget.cpp` becomes:

```cpp
#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"

#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::target::sme {

namespace {

/// Wraps the configuration target with the executable f32 contraction
/// subset, the streaming ABI contract, and both lowering hooks. The
/// configuration loader knows a bundle's shape but nothing about SME code.
class SMETarget : public mapping::MappingTarget {
public:
  explicit SMETarget(std::unique_ptr<mapping::MappingTarget> configuration)
      : configuration_(std::move(configuration)) {}

  llvm::StringRef name() const override { return configuration_->name(); }
  const machine::MachineModel &machine() const override {
    return configuration_->machine();
  }
  const mapping::LayoutRegistry &layouts() const override {
    return configuration_->layouts();
  }
  const mapping::RuleRegistry &rules() const override {
    return configuration_->rules();
  }
  const mapping::LatencyProvider *latencyProvider() const override {
    return configuration_->latencyProvider();
  }
  bool isKnownEmitter(llvm::StringRef key) const override {
    return configuration_->isKnownEmitter(key);
  }
  std::optional<mapping::TargetCodegenRequirements>
  codegenRequirements() const override {
    return mapping::TargetCodegenRequirements{
        "aarch64",
        "",
        {"sme"},
        /*vectorLengthBits=*/std::nullopt,
        /*streamingVectorLengthBits=*/512,
        /*invocationAttributes=*/{"aarch64_pstate_sm_enabled"}};
  }
  llvm::Error
  buildSelectedBackendPipeline(mlir::OpPassManager &pm) const override {
    buildSMESelectedBackendPipeline(pm);
    return llvm::Error::success();
  }
  llvm::Error buildPostBufferizationBackendPipeline(
      mlir::OpPassManager &pm) const override {
    buildSMEPostBufferizationPipeline(pm);
    return llvm::Error::success();
  }
  std::unique_ptr<mapping::TargetEmitter>
  createEmitter(llvm::StringRef key) const override {
    if (!configuration_->isKnownEmitter(key))
      return nullptr;
    return createSMEEmitter(key);
  }
  std::unique_ptr<mapping::TargetEmitter> createEmitter() const override {
    llvm::ArrayRef<llvm::StringRef> keys = emitterKeys();
    return keys.empty() ? nullptr : createEmitter(keys.front());
  }

private:
  std::unique_ptr<mapping::MappingTarget> configuration_;
};

} // namespace

llvm::ArrayRef<llvm::StringRef> emitterKeys() {
  static const llvm::StringRef keys[] = {
      "sme_vector_add", "sme_vector_convert", "sme_vector_silu",
      "sme_vector_mul", "sme_reduce",         "sme_copy",
      "sme_tile_copy",  "sme_tile_store",     "sme_fused_convert_silu_mul",
      "sme_mma_za"};
  return keys;
}

llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath) {
  std::string root = configurationRoot.str();
  std::string machine = machinePath.empty() ? root + "/machines/arm-sme-v2.yaml"
                                            : machinePath.str();
  std::vector<std::string> keys;
  for (llvm::StringRef key : emitterKeys())
    keys.push_back(key.str());
  llvm::Expected<std::unique_ptr<mapping::MappingTarget>> configuration =
      mapping::loadMappingTarget(
          "arm-sme", machine, root + "/mapping/arm-sme/layouts.llkmap",
          root + "/mapping/arm-sme/rules.llkmap", std::move(keys));
  if (!configuration)
    return configuration.takeError();
  return std::make_unique<SMETarget>(std::move(*configuration));
}

} // namespace mlir::llk::target::sme
```

Create `lib/Target/AArch64/Mapping/SMEBundleLowering.cpp`:

```cpp
//===- SMEBundleLowering.cpp - Selected SME bundle lowering --------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The SME target's lowering boundary. Verification asks whether a bundle is
// complete; these emitters answer what code the selected group is. The f32
// 16x16x16 initialized contraction and the fixture's movement keys are the
// only supported signatures. Movement groups are carried by the generic
// Micro-to-Linalg and Linalg-to-loops bridge; the emitter's claim keeps them
// selected rather than reference-lowered. Every other key keeps the base
// emitter's explicit rejection.

#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"

#include "LLK/Dialect/Micro/MicroDialect.h" // IWYU pragma: keep
#include "LLK/Dialect/Micro/MicroEnums.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

namespace mlir::llk::target::sme {
namespace {

llvm::Error smeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "sme_lowering_unsupported: " + message);
}

bool isMovementKey(llvm::StringRef key) {
  return key == "sme_copy" || key == "sme_tile_copy" ||
         key == "sme_tile_store";
}

class SMEEmitter : public mapping::DeclaredTargetEmitter {
public:
  SMEEmitter(std::string key, std::vector<std::string> declaredKeys)
      : DeclaredTargetEmitter(std::move(key), std::move(declaredKeys)) {}

  bool hasLowering() const override {
    return key() == "sme_mma_za" || isMovementKey(key());
  }

  bool
  hasLoweringFor(llvm::ArrayRef<mlir::Operation *> coveredOps) const override {
    if (!hasLowering() || coveredOps.empty())
      return false;
    return !static_cast<bool>(validateGroup(coveredOps));
  }

  llvm::Error lower(llvm::ArrayRef<mlir::Operation *> coveredOps,
                    const mapping::TargetBundle &bundle,
                    const mapping::TargetLoweringContext &context,
                    mlir::RewriterBase &rewriter) const override {
    if (llvm::Error error = verify(bundle))
      return error;
    if (!hasLowering())
      return DeclaredTargetEmitter::lower(coveredOps, bundle, context,
                                          rewriter);
    if (llvm::Error error = validateGroup(coveredOps))
      return error;
    for (mlir::Operation *op : coveredOps)
      op->setAttr("llk.sme.selected_handoff",
                  mlir::UnitAttr::get(rewriter.getContext()));
    return llvm::Error::success();
  }

private:
  llvm::Error
  validateGroup(llvm::ArrayRef<mlir::Operation *> coveredOps) const {
    if (coveredOps.empty())
      return smeError("no covered operations to lower");
    if (isMovementKey(key())) {
      llvm::StringRef expected = key() == "sme_copy"      ? "micro.async_copy"
                                 : key() == "sme_tile_copy"
                                     ? "micro.tile_async_copy"
                                     : "micro.tile_store";
      for (mlir::Operation *op : coveredOps)
        if (op->getName().getStringRef() != expected)
          return smeError("emitter '" + key() +
                          "' cannot lower covered operation '" +
                          op->getName().getStringRef() + "'");
      return llvm::Error::success();
    }

    if (coveredOps.size() != 1)
      return smeError("the contraction group must cover exactly one micro.mma");
    auto mma = mlir::dyn_cast<micro::MmaOp>(coveredOps.front());
    if (!mma)
      return smeError("emitter 'sme_mma_za' needs a micro.mma");
    if (mma.getShape() != llvm::ArrayRef<int64_t>{16, 16, 16})
      return smeError("emitter 'sme_mma_za' supports the 16x16x16 shape only");
    if (mma.getInput() != micro::DType::f32 ||
        mma.getAccumulator() != micro::DType::f32)
      return smeError("emitter 'sme_mma_za' supports f32 inputs and "
                      "accumulator only");
    auto result = mlir::dyn_cast<micro::TileType>(mma.getResult().getType());
    if (!result || result.getShape() != llvm::ArrayRef<int64_t>{16, 16} ||
        !result.getElementType().isF32())
      return smeError("emitter 'sme_mma_za' needs a static 16x16 f32 tile "
                      "result");
    return llvm::Error::success();
  }
};

} // namespace

std::unique_ptr<mapping::TargetEmitter> createSMEEmitter(llvm::StringRef key) {
  std::vector<std::string> declared;
  for (llvm::StringRef declaredKey : emitterKeys())
    declared.push_back(declaredKey.str());
  return std::make_unique<SMEEmitter>(key.str(), std::move(declared));
}

} // namespace mlir::llk::target::sme
```

In `lib/Target/CMakeLists.txt`, add to `LLKTargetMapping`:

```cmake
    AArch64/Mapping/SMEBundleLowering.cpp
    AArch64/Mapping/SMEVectorLowering.cpp
    AArch64/Mapping/SMEArmLowering.cpp
```

For this task, create empty stubs so the target links; Task 5 fills them:

```cpp
// lib/Target/AArch64/Mapping/SMEVectorLowering.cpp (Task 4 stub)
#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"
namespace mlir::llk::target::sme {
void buildSMESelectedBackendPipeline(mlir::OpPassManager &) {}
} // namespace mlir::llk::target::sme
```

```cpp
// lib/Target/AArch64/Mapping/SMEArmLowering.cpp (Task 4 stub)
#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"
namespace mlir::llk::target::sme {
void buildSMEPostBufferizationPipeline(mlir::OpPassManager &) {}
} // namespace mlir::llk::target::sme
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
ninja -C build LLKMappingTests
./build/bin/LLKMappingTests --gtest_filter='ArmSMEBackend.*:ArmTargets.*'
```

Expected: PASS for the emitter and contract tests. The `SelectedBf16ContractionIsRefused` test passes because no target pass realizes a group yet; the f32 end-to-end test arrives in Task 5.

- [ ] **Step 5: Commit**

```bash
git add include/LLK/Target/AArch64/Mapping/SMEMappingTarget.h \
        lib/Target/AArch64/Mapping/SMEMappingTarget.cpp \
        lib/Target/AArch64/Mapping/SMEBundleLowering.cpp \
        lib/Target/AArch64/Mapping/SMEVectorLowering.cpp \
        lib/Target/AArch64/Mapping/SMEArmLowering.cpp \
        lib/Target/CMakeLists.txt test/Mapping/arm_sme_backend.cpp \
        test/Mapping/CMakeLists.txt test/Mapping/arm_targets.cpp \
        test/Mapping/arm_placement.cpp
git commit -m "feat(target): make arm-sme a selective executable target (#253)"
```

---

## Task 5: Lower the selected contraction through MLIR ArmSME

**Files:**
- Modify: `include/LLK/Mapping/MappingTarget.h` (target-neutral late hook)
- Modify: `include/LLK/Target/AArch64/Mapping/SMEMappingTarget.h` (late builder decl)
- Modify: `lib/Target/AArch64/Mapping/SMEMappingTarget.cpp` (late hook override)
- Modify: `lib/Target/AArch64/Mapping/SMEVectorLowering.cpp` (replace the stub)
- Modify: `lib/Target/AArch64/Mapping/SMEArmLowering.cpp` (replace the stub; add the late pass)
- Modify: `include/LLK/Conversion/MappedCompilation.h`,
  `lib/Conversion/MicroMapping/MappedCompilation.cpp` (`runLateBackendPipeline`
  and its call on the executable path)
- Modify: `tools/llk-compile/llk-compile.cpp` (AOT object path)
- Modify: `test/Mapping/arm_sme_backend.cpp`

**Interfaces:**
- Consumes: `createSMEEmitter`, the target's three pipeline hooks, and the
  selected `linalg.matmul` provenance
- Produces (post-bufferization): module attribute `llk.sme.selected_groups`
  (i64) and `llk.backend_groups_realized` (i64)
- Produces (post-bufferization): kernel function attributes
  `llvm.arm_streaming` and `llvm.arm_new_za` (translated to LLVM IR
  `aarch64_pstate_sm_enabled` / `aarch64_new_za`)
- Produces (late): flattened `cf` control flow and the ArmSME dialect's LLVM
  intrinsic operations (`arm_sme.intr.*`, translated to `llvm.aarch64.sme.*`)

**Why three stages.** The shared ABI/ownership preparation
(`prepareMappedKernelForInvocation`) runs ownership-based buffer deallocation,
which accepts **only structured control flow**. ArmSME tile allocation inside
`convert-arm-sme-to-llvm` needs **flattened** `cf` control flow. Both cannot run
in the same pipeline position: flattening in the post-bufferization pass makes
every ArmSME artifact fail preparation with "Only structured control-flow loops
are supported." The lowering is therefore split at the ownership boundary:

1. pre-bufferization — select and vectorize the contraction;
2. post-bufferization (before ownership) — Vector lowering + the ArmSME sequence
   through `enable-arm-streaming`, stopping with structured SCF;
3. late (after ownership) — `convert-scf-to-cf`, `convert-arm-sme-to-llvm`,
   cleanup.

- [ ] **Step 1: Write the failing end-to-end tests**

Append to `test/Mapping/arm_sme_backend.cpp`:

```cpp
TEST(ArmSMEBackend, SelectedContractionLowersToArmSME) {
  auto fixture = arm_test::mapFixture("arm-sme", kContraction);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_contraction";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  EXPECT_GE(compiled->backendGroupsRealized, 1u);
  EXPECT_EQ(compiled->referenceGroupsLowered, 0u);

  std::string text;
  llvm::raw_string_ostream stream(text);
  compiled->module->print(stream);
  // The ArmSME conversion consumes the vector contraction and leaves the
  // dialect's high-level tile operations; the movement copies are already
  // loops. The SCF flattening and ArmSME-to-LLVM conversion are a late step,
  // after the shared ownership preparation.
  EXPECT_NE(text.find("arm_sme.outerproduct"), std::string::npos) << text;
  EXPECT_EQ(text.find("vector.contract"), std::string::npos) << text;
  EXPECT_EQ(text.find("vector.outerproduct"), std::string::npos) << text;
  // enable-arm-streaming records MLIR's streaming/ZA attributes; the LLVM
  // translation names them aarch64_pstate_sm_enabled / aarch64_new_za.
  EXPECT_NE(text.find("llvm.arm_streaming"), std::string::npos) << text;
  EXPECT_NE(text.find("llvm.arm_new_za"), std::string::npos) << text;
}

TEST(ArmSMEBackend, SelectedContractionLatePipelineFinishesArmSME) {
  auto fixture = arm_test::mapFixture("arm-sme", kContraction);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_contraction";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  ASSERT_TRUE(static_cast<bool>(compiled->module));

  // In the executable path this runs on the ownership-prepared clone; the
  // lowered module is the same structured form.
  ASSERT_FALSE(static_cast<bool>(
      llk::runLateBackendPipeline(*fixture->target, *compiled->module)));

  std::string text;
  llvm::raw_string_ostream stream(text);
  compiled->module->print(stream);
  EXPECT_NE(text.find("cf.br"), std::string::npos) << text;
  EXPECT_EQ(text.find("scf.for"), std::string::npos) << text;
  EXPECT_NE(text.find("arm_sme.intr.mopa"), std::string::npos) << text;
  EXPECT_EQ(text.find("arm_sme.outerproduct"), std::string::npos) << text;
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ninja -C build LLKMappingTests
./build/bin/LLKMappingTests --gtest_filter='ArmSMEBackend.*'
```

Expected: FAIL with `selected-target backend realized no selected compute
groups` (both pipeline hooks are still stubs), and the late test fails to run
the pipeline.

- [ ] **Step 3: Implement the pre-bufferization vector lowering**

Replace `lib/Target/AArch64/Mapping/SMEVectorLowering.cpp` with
`SMEVectorLoweringPass` (`llk-sme-vector-lowering`, module pass):

- selects only `micro.mapping` with `emitter == "sme_mma_za"` **and**
  `bundle == "sme.mma.f32"` (the bundle is the discriminator: `MicroToLinalg`
  casts bf16 inputs to the f32 accumulator type, so operand types cannot
  distinguish the unsupported bf16 selection);
- requires `micro.op == "mma"` and a static 16x16 f32 `linalg.matmul` result,
  failing with `SME selected lowering needs a static 16x16x16 f32 contraction`
  otherwise;
- tiles with scalable `[vscale*4, vscale*4, 1]` via `scf::tileUsingSCF` and
  vectorizes with `linalg::vectorize(..., {4,4,1}, {true,true,false},
  createNamedContraction=true)`;
- sets `llk.sme.selected_groups` on the module (0 when nothing matched).

`buildSMESelectedBackendPipeline` = pass + canonicalize + CSE.

- [ ] **Step 4: Implement the post-bufferization pass and the late pass**

`SMEArmLoweringPass` (`llk-sme-arm-lowering`, module pass) in
`lib/Target/AArch64/Mapping/SMEArmLowering.cpp`:

1. reads `llk.sme.selected_groups`; returns when absent or zero;
2. applies the Vector patterns (`populateVectorTransferPermutationMapLowering`,
   `populateVectorTransferDropUnitDims`, contract→outerproduct with
   `disableOuterProductLowering`, mask lowering);
3. runs, via `runPipeline` on a fresh module pass manager:

   ```text
   arm-sme-vector-legalization
   canonicalize, cse
   convert-arith-to-arm-sme
   convert-vector-to-arm-sme
   convert-arm-sme-to-scf
   convert-vector-to-scf{full-unroll}
   func.func(enable-arm-streaming{streaming-mode=streaming za-mode=new-za
                                  if-required-by-ops})
   ```

   `enable-arm-streaming` is func-scoped and must be added with
   `addNestedPass<func::FuncOp>`. Every dialect the nested pipeline creates must
   also be declared by the enclosing pass (`getDependentDialects`), because a
   pipeline run from inside a pass does not load its passes' dependencies;
   otherwise ops such as `cf.br` are created unregistered.
4. counts `arm_sme.*` ops and fails with `selected SME lowering produced no
   ArmSME tile operations` when none exist (this check must run before the late
   pass does the intrinsic conversion);
5. sets `llk.backend_groups_realized`.

`SMELateLoweringPass` (`llk-sme-late-lowering`, module pass, same file):

1. reads `llk.sme.selected_groups`; returns when absent or zero;
2. runs, via `runPipeline` on a fresh module pass manager:

   ```text
   convert-scf-to-cf                      # createSCFToControlFlowPass (LLVM >= 21)
   func.func(convert-arm-sme-to-llvm)
   canonicalize, cse
   ```

   `createSCFToControlFlowPass()` (LLVM >= 21) or `createConvertSCFToCFPass()`
   picks the SCF-to-CF factory behind the existing version guard. Tile
   allocation happens inside `convert-arm-sme-to-llvm` and requires the
   flattened form.
3. declares CF, LLVM, func, scf, vector, memref, arith and arm_sme as dependent
   dialects.

- [ ] **Step 5: Add the framework hook and call it after ownership preparation**

In `include/LLK/Mapping/MappingTarget.h`:

```cpp
/// Runs after the shared ABI and ownership preparation, before code emission.
/// The shared preparation owns structured control flow (buffer deallocation
/// rejects backedges); a target whose instruction selection needs a flattened
/// form changes it here. Default: no-op.
virtual llvm::Error
buildLateBackendPipeline(mlir::OpPassManager &) const {
  return llvm::Error::success();
}
```

In `include/LLK/Conversion/MappedCompilation.h` / `MappedCompilation.cpp`:

```cpp
llvm::Error runLateBackendPipeline(const MappingTarget &target,
                                   mlir::ModuleOp module);
```

It builds a `PassManager`, calls `target.buildLateBackendPipeline(pm)`, skips an
empty pipeline, runs it and returns an error on failure.

Call it:

- `finishCompilation`: after `prepareMappedKernelForInvocation` succeeds and
  before `createMappedExecutable`, only for `MappedBackend::SelectedTarget`, on
  `*prepared->module`;
- `tools/llk-compile/llk-compile.cpp`, `emitMode == "object"`: after
  preparation and before `emitMappedObject`, on `*prepared->module`.

`SMETarget` overrides `buildLateBackendPipeline` to add `SMELateLoweringPass`
through `buildSMELateBackendPipeline`, declared in `SMEMappingTarget.h`.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
ninja -C build
./build/bin/LLKMappingTests --gtest_filter='ArmSMEBackend.*:ArmTargets.*:ArmPlacement.*'
./build/bin/LLKMappingTests
ctest --test-dir build
```

Expected: the two new tests pass, `SelectedBf16ContractionIsRefused` still
passes, and the full suite shows no new failure. `llk-compile --emit=object`
then reaches LLVM translation (the ArmSME translation-interface registration and
the native harness stubs are Task 6).

- [ ] **Step 7: Commit**

```bash
git add include/LLK/Mapping/MappingTarget.h \
        include/LLK/Target/AArch64/Mapping/SMEMappingTarget.h \
        lib/Target/AArch64/Mapping/SMEMappingTarget.cpp \
        lib/Target/AArch64/Mapping/SMEVectorLowering.cpp \
        lib/Target/AArch64/Mapping/SMEArmLowering.cpp \
        include/LLK/Conversion/MappedCompilation.h \
        lib/Conversion/MicroMapping/MappedCompilation.cpp \
        tools/llk-compile/llk-compile.cpp \
        test/Mapping/arm_sme_backend.cpp
git commit -m "feat(target): lower the selected SME contraction through ArmSME (#253)"
```

---

## Task 6: Add the native Apple M5 evidence runner (tier T3, gating)

**Files:**
- Create: `test/Execution/arm_sme_evidence_common.py`
- Create: `test/Execution/arm_sme_evidence_test.py`
- Create: `test/Execution/arm_sme_native_evidence.py`
- Modify: `test/Execution/CMakeLists.txt`
- Modify: `test/FileCheck.cmake`

**Interfaces:**
- Consumes: `llk-compile --emit=object` with the `arm-sme` target, Task 2's identity fields, and Task 5's lowering
- Produces: `emit_object`, `check_object_format`, `check_identity`, `check_sme_instructions`, `parse_hex_identity`, `sha256_file`, `render_shim`, `render_native_harness`, `render_tables`
- Produces: CTest `ArmSMENativeEvidence` on an SME-capable arm64 macOS host and `ArmSMEEvidenceParser` on every host

- [ ] **Step 1: Write the failing parser test**

Create `test/Execution/arm_sme_evidence_test.py`:

```python
import unittest

from arm_sme_evidence_common import (
    has_sme_instructions, parse_hex_identity)


class ArmSMEEvidenceParserTest(unittest.TestCase):
    def test_sme_instructions_are_required(self):
        sme = """
        0: 2598e3e0  ptrue p0.s
        4: 80812000  fmopa za0.s, p0/m, p1/m, z0.s, z1.s
        8: c00800ff  zero {za}
        c: d503457f  smstart za
        """
        self.assertTrue(has_sme_instructions(sme))

    def test_neon_and_scalar_objects_are_not_sme(self):
        scalar = """
        0: d503201f  nop
        4: 8b010000  add x0, x0, x1
        8: 4e21d800  fadd v0.4s, v0.4s, v1.4s
        c: 4e21d800  fadd v0.4s, v0.4s, v1.4s
        """
        self.assertFalse(has_sme_instructions(scalar))
        # A vector contraction without ZA enablement is not enough.
        outer = "0: 80812000  fmopa za0.s, p0/m, p1/m, z0.s, z1.s"
        self.assertFalse(has_sme_instructions(outer))

    def test_hex_identity_parses_both_forms(self):
        self.assertEqual(parse_hex_identity("deadbeef"), 0xDEADBEEF)
        self.assertEqual(parse_hex_identity("0x10"), 0x10)
        self.assertEqual(parse_hex_identity("0"), 0)
        self.assertEqual(parse_hex_identity("0x0"), 0)

    def test_absent_or_malformed_identity_is_rejected(self):
        for value in (None, "", "not-a-hash", 17):
            self.assertIsNone(parse_hex_identity(value))


if __name__ == "__main__":
    unittest.main()
```

Add to `test/Execution/CMakeLists.txt`, next to `ArmSVEEvidenceParser`:

```cmake
  add_test(NAME ArmSMEEvidenceParser
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/arm_sme_evidence_test.py)
  set_tests_properties(ArmSMEEvidenceParser PROPERTIES LABELS "mapped-reference")
```

- [ ] **Step 2: Run the parser test to verify it fails**

```bash
ninja -C build   # re-runs CMake and registers the new test
ctest --test-dir build -R '^ArmSMEEvidenceParser$' --output-on-failure
```

Expected: FAIL with `ModuleNotFoundError: arm_sme_evidence_common`.

- [ ] **Step 3: Create the shared evidence module**

Create `test/Execution/arm_sme_evidence_common.py`:

```python
"""Shared Arm SME evidence helpers (E11).

The numerical data is integer-valued f32: every product and every partial sum
is exactly representable, so the expected result is independent of the
hardware's internal accumulation order. The harness compares literal bits.
"""

import hashlib
import json
import re
import struct
import subprocess

ROWS = 16
COLS = 16
K = 16
CANARY = 32
SENTINEL = 0x5A5A5A5A
CASE = ("contraction_f32", "arm_contraction")
TRIPLE_DARWIN = "arm64-apple-macosx14.0.0"
TRIPLE_LINUX = "aarch64-unknown-linux-gnu"
FEATURES = ["+sme"]

SME_FMOPA = re.compile(r"\bfmopa\b")
SME_ZERO_ZA = re.compile(r"\bzero\s*\{\s*za\s*\}")
SME_SMSTART_ZA = re.compile(r"\bsmstart\s+za\b")


def parse_hex_identity(value):
    if not isinstance(value, str):
        return None
    try:
        return int(value, 16)
    except ValueError:
        return None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_sme_instructions(disassembly):
    """True only for code that accumulates, zeroes ZA and enters streaming ZA."""
    return (bool(SME_FMOPA.search(disassembly)) and
            bool(SME_ZERO_ZA.search(disassembly)) and
            bool(SME_SMSTART_ZA.search(disassembly)))


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _dataset(lhs_base, rhs_base, init):
    a = [[lhs_base + i for _ in range(COLS)] for i in range(ROWS)]
    b = [[rhs_base + j for _ in range(COLS)] for _ in range(ROWS)]
    initial = [[init for _ in range(COLS)] for _ in range(ROWS)]
    # C[i][j] = init + sum_k A[i][k] * B[k][j]
    #         = init + K * (lhs_base + i) * (rhs_base + j)
    expected = [[init + K * (lhs_base + i) * (rhs_base + j)
                 for j in range(COLS)] for i in range(ROWS)]
    return a, b, initial, expected


def render_tables():
    a1, b1, i1, e1 = _dataset(1, 1, 1)
    a2, b2, i2, e2 = _dataset(2, 2, 2)
    names = {"A1": a1, "B1": b1, "I1": i1, "E1": e1,
             "A2": a2, "B2": b2, "I2": i2, "E2": e2}
    rendered = []
    for name in ("A1", "B1", "I1", "E1", "A2", "B2", "I2", "E2"):
        values = [_f32_bits(v) for row in names[name] for v in row]
        body = ",\n  ".join(
            ", ".join("0x%08xu" % v for v in values[i:i + COLS])
            for i in range(0, len(values), COLS))
        rendered.append(
            "static const uint32_t %s[ROWS * COLS] = {\n  %s\n};" % (name, body))
    return "\n\n".join(rendered)


def render_shim(symbol_prefix, arch_directive=""):
    # The ELF path passes `.arch armv9-a+sme` because GNU as defaults to the
    # base architecture; the Mach-O path passes an empty directive and enables
    # SME with `clang -march=armv9-a+sme`.
    return arch_directive + r"""
	.text
	.p2align 2
	.globl	%(p)ssme_call
%(p)ssme_call:
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	mov	x16, x0
	mov	x0, x1
	mov	x1, x2
	mov	x2, x3
	mov	x3, x4
	smstart	sm
	blr	x16
	smstop	sm
	ldp	x29, x30, [sp], #16
	ret

	.globl	%(p)ssme_streaming_bytes
%(p)ssme_streaming_bytes:
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	smstart	sm
	cntb	x0
	smstop	sm
	ldp	x29, x30, [sp], #16
	ret
""" % {"p": symbol_prefix}


_HARNESS_HEAD = r"""
#include <stdint.h>
#include <stddef.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size0;
  int64_t size1;
  int64_t stride0;
  int64_t stride1;
} MemRef2D;

typedef void (*KernelFn)(MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern void %(wrapper)s(MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern void sme_call(KernelFn, MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern unsigned long sme_streaming_bytes(void);

#define ROWS 16
#define COLS 16
#define CANARY 32
#define SENTINEL 0x5a5a5a5au

%(tables)s

static float a[ROWS * COLS] __attribute__((aligned(64)));
static float b[ROWS * COLS] __attribute__((aligned(64)));
static float init[ROWS * COLS] __attribute__((aligned(64)));
static float out[ROWS * COLS + CANARY] __attribute__((aligned(64)));

static void load(float *dst, const uint32_t *src) {
  for (size_t i = 0; i < ROWS * COLS; ++i) {
    uint32_t bits = src[i];
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    dst[i] = value;
  }
}

static int matches(const float *result, const uint32_t *expected) {
  for (size_t i = 0; i < ROWS * COLS; ++i) {
    uint32_t bits;
    __builtin_memcpy(&bits, &result[i], sizeof(bits));
    if (bits != expected[i])
      return 0;
  }
  return 1;
}

static int canary_holds(void) {
  for (size_t i = 0; i < CANARY; ++i) {
    uint32_t bits;
    __builtin_memcpy(&bits, &out[ROWS * COLS + i], sizeof(bits));
    if (bits != SENTINEL)
      return 0;
  }
  return 1;
}

static void sentinel_fill(void) {
  for (size_t i = 0; i < ROWS * COLS + CANARY; ++i)
    out[i] = 0.0f;
  for (size_t i = 0; i < ROWS * COLS + CANARY; ++i) {
    uint32_t bits = SENTINEL;
    __builtin_memcpy(&out[i], &bits, sizeof(bits));
  }
}

static MemRef2D descriptor(float *buffer) {
  MemRef2D result = {buffer, buffer, 0, ROWS, COLS, COLS, 1};
  return result;
}
"""


def render_native_harness(wrapper, without_shim=False):
    call = ("%s(&da, &db, &di, &dout);" % wrapper if without_shim
            else "sme_call(&%s, &da, &db, &di, &dout);" % wrapper)
    body = r"""
#include <sys/sysctl.h>
#include <unistd.h>

#define SAY(text) write(1, (text), sizeof(text) - 1)

static int host_has_sme(void) {
  int value = 0;
  size_t size = sizeof(value);
  if (sysctlbyname("hw.optional.arm.FEAT_SME", &value, &size, NULL, 0) != 0)
    return 0;
  return value != 0;
}

int main(void) {
  if (!host_has_sme()) {
    SAY("unsupported: no sme\n");
    return 70;
  }
  if (sme_streaming_bytes() != 64) {
    SAY("unsupported: svl is not 512\n");
    return 71;
  }
  MemRef2D da = descriptor(a), db = descriptor(b);
  MemRef2D di = descriptor(init), dout = descriptor(out);

  load(a, A1); load(b, B1); load(init, I1); sentinel_fill();
  %(call)s
  if (!matches(out, E1)) return 73;
  if (!canary_holds()) return 74;

  load(a, A2); load(b, B2); load(init, I2); sentinel_fill();
  %(call)s
  if (!matches(out, E2)) return 73;
  if (!canary_holds()) return 74;

  SAY("pass\n");
  return 0;
}
""" % {"call": call}
    return _HARNESS_HEAD % {"wrapper": wrapper,
                            "tables": render_tables()} + body


def emit_object(llk_compile, source_dir, work_dir, triple, features):
    source = ("%s/test/Conversion/MicroMapping/Inputs/arm/%s.mlir" %
              (source_dir, CASE[0]))
    obj = "%s/kernel.o" % work_dir
    identity = "%s/identity.json" % work_dir
    command = [
        llk_compile, "--mapping-target=arm-sme",
        "--mapping-root=" + source_dir,
        "--mapping-backend=selected-target", "--mapping-stop=lowered",
        "--emit=object", "--target-triple=" + triple,
        "--target-features=" + ",".join(features),
        "--output=" + obj, "--emit-identity=" + identity, source,
    ]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=300)
    if result.returncode:
        raise SystemExit("object emission failed:\n%s\n%s" %
                         (result.stdout, result.stderr))
    with open(identity, encoding="utf-8") as handle:
        record = json.load(handle)
    return obj, record, "_mlir_ciface_" + CASE[1]


def check_object_format(path, platform):
    data = open(path, "rb").read()
    if platform == "darwin":
        if data[:4] != b"\xcf\xfa\xed\xfe":
            raise SystemExit("output is not a 64-bit Mach-O object")
        if struct.unpack_from("<I", data, 4)[0] != 0x0100000C:
            raise SystemExit("Mach-O object is not CPU_TYPE_ARM64")
    else:
        if data[:4] != b"\x7fELF":
            raise SystemExit("output is not an ELF object")
        if struct.unpack_from("<H", data, 18)[0] != 183:
            raise SystemExit("ELF object is not EM_AARCH64")


def check_identity(record, triple, wrapper, failures, label):
    if record.get("schemaVersion") != 2:
        failures.append("%s: identity schemaVersion is not 2" % label)
    if record.get("targetTriple") != triple:
        failures.append("%s: identity triple %r != %r" %
                        (label, record.get("targetTriple"), triple))
    if record.get("targetFeatures") != FEATURES:
        failures.append("%s: identity features %r != %r" %
                        (label, record.get("targetFeatures"), FEATURES))
    if record.get("streamingVectorLengthBits") != 512:
        failures.append("%s: identity streamingVectorLengthBits is %r" %
                        (label, record.get("streamingVectorLengthBits")))
    if record.get("invocationAttributes") != ["aarch64_pstate_sm_enabled"]:
        failures.append("%s: identity invocationAttributes is %r" %
                        (label, record.get("invocationAttributes")))
    if record.get("referenceGroupsLowered") != 0:
        failures.append("%s: selected lowering fell back (%r reference groups)" %
                        (label, record.get("referenceGroupsLowered")))
    if record.get("backendGroupsRealized", 0) <= 0:
        failures.append("%s: selected backend realized no groups" % label)
    if record.get("wrapperSymbol") != wrapper:
        failures.append("%s: identity wrapper %r != %r" %
                        (label, record.get("wrapperSymbol"), wrapper))
    for key in ("planId", "sourceGraphHash", "abiHash"):
        value = parse_hex_identity(record.get(key))
        if value is None or value == 0:
            failures.append("%s: identity %s is not a nonzero hex hash: %r" %
                            (label, key, record.get(key)))


def check_sme_instructions(disassembler, obj, failures, label):
    result = subprocess.run([disassembler, "-d", obj], capture_output=True,
                            text=True, timeout=120)
    if result.returncode:
        failures.append("%s: disassembly failed\n%s%s" %
                        (label, result.stdout, result.stderr))
        return
    if not has_sme_instructions(result.stdout):
        failures.append(
            "%s: object has no fmopa + zero {za} + smstart za sequence\n%s" %
            (label, result.stdout))
```

- [ ] **Step 4: Create the native driver**

Create `test/Execution/arm_sme_native_evidence.py`:

```python
#!/usr/bin/env python3
"""Native Apple M5 SME contraction evidence (E11, tier T3).

This is the gating run. It cross-emits a Mach-O object, links it with an
assembly shim that is the only code that changes PSTATE.SM, and executes it on
real SME hardware. It is not reproducible in CI; the ledger pins the host,
toolchain and artifact digests.
"""

import os
import signal
import subprocess
import sys

import arm_sme_evidence_common as common


def _run(command, timeout=300):
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=timeout)
    if result.returncode:
        raise SystemExit("command failed: %s\n%s\n%s" %
                         (" ".join(command), result.stdout, result.stderr))
    return result


def main():
    if len(sys.argv) != 6:
        print("usage: arm_sme_native_evidence.py <llk-compile> <source-dir> "
              "<work-dir> <clang> <llvm-objdump>", file=sys.stderr)
        return 2
    llk_compile, source_dir, work_dir, clang, objdump = sys.argv[1:]
    os.makedirs(work_dir, exist_ok=True)
    failures = []

    obj, record, wrapper = common.emit_object(
        llk_compile, source_dir, work_dir, common.TRIPLE_DARWIN,
        common.FEATURES)
    common.check_object_format(obj, "darwin")
    common.check_identity(record, common.TRIPLE_DARWIN, wrapper, failures,
                          "contraction_f32")
    common.check_sme_instructions(objdump, obj, failures, "contraction_f32")

    shim = os.path.join(work_dir, "shim.S")
    harness_c = os.path.join(work_dir, "harness.c")
    with open(shim, "w", encoding="utf-8") as handle:
        handle.write(common.render_shim("_"))
    with open(harness_c, "w", encoding="utf-8") as handle:
        handle.write(common.render_native_harness(wrapper))
    shim_o = os.path.join(work_dir, "shim.o")
    harness_o = os.path.join(work_dir, "harness.o")
    binary = os.path.join(work_dir, "harness")
    _run([clang, "-march=armv9-a+sme", "-c", shim, "-o", shim_o])
    _run([clang, "-O2", "-c", harness_c, "-o", harness_o])
    _run([clang, harness_o, shim_o, obj, "-o", binary])

    run = subprocess.run([binary], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        failures.append("native run exit %d\n%s%s" %
                        (run.returncode, run.stdout, run.stderr))

    # Negative control 1: smstart sm alone does not enable ZA.
    probe_c = os.path.join(work_dir, "za_probe.c")
    with open(probe_c, "w", encoding="utf-8") as handle:
        handle.write("""
int main(void) {
  __asm__ volatile("smstart sm");
  __asm__ volatile("zero {za}");
  __asm__ volatile("smstop sm");
  return 0;
}
""")
    probe = os.path.join(work_dir, "za_probe")
    _run([clang, "-O2", "-march=armv9-a+sme", probe_c, "-o", probe])
    za = subprocess.run([probe], capture_output=True, text=True, timeout=30)
    if za.returncode != -signal.SIGILL:
        failures.append(
            "ZA probe exited %d; smstart sm alone must leave ZA disabled and "
            "trap with SIGILL" % za.returncode)

    # Negative control 2: the streaming wrapper cannot be called with SM off.
    off_c = os.path.join(work_dir, "harness_without_shim.c")
    with open(off_c, "w", encoding="utf-8") as handle:
        handle.write(common.render_native_harness(wrapper, without_shim=True))
    off_o = os.path.join(work_dir, "harness_without_shim.o")
    off_bin = os.path.join(work_dir, "harness_without_shim")
    _run([clang, "-O2", "-c", off_c, "-o", off_o])
    _run([clang, off_o, shim_o, obj, "-o", off_bin])
    try:
        off = subprocess.run([off_bin], capture_output=True, text=True,
                             timeout=30)
        if off.returncode == 0:
            failures.append("the wrapper ran successfully with SM off")
    except subprocess.TimeoutExpired:
        pass  # did not produce a passing result; counted as a refusal

    version = _run([clang, "--version"]).stdout.splitlines()[0]
    os_version = _run(["sw_vers", "-productVersion"]).stdout.strip()
    model = _run(["sysctl", "-n", "hw.model"]).stdout.strip()
    print(
        "arm sme native evidence: host=%s macos=%s clang=%r wrapper=%s "
        "plan=%s graph=%s abi=%s svl=512 selected-groups=%s "
        "reference-groups=%s object-sha256=%s harness-sha256=%s "
        "shim-sha256=%s za-probe=SIGILL" % (
            model, os_version, version, wrapper, record.get("planId"),
            record.get("sourceGraphHash"), record.get("abiHash"),
            record.get("selectedGroupsVerified"),
            record.get("referenceGroupsLowered"), common.sha256_file(obj),
            common.sha256_file(harness_c), common.sha256_file(shim)))

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("arm sme native evidence: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Register the tests**

In `test/FileCheck.cmake`, inside the `if(LLK_BUILD_TOOLS)` block after `AOTObjectEmission`:

```cmake
  # Framework E11 (#253): the native Apple M5 T3 evidence. Register only on an
  # arm64 macOS host that advertises SME; a host without SME does not register
  # the test, and a registered run fails rather than skips.
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    execute_process(COMMAND sysctl -n hw.optional.arm.FEAT_SME
        OUTPUT_VARIABLE LLK_HOST_SME
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(LLK_HOST_SME STREQUAL "1")
      # On an SME host a missing disassembler is a configuration failure, not
      # a reason to quietly drop the evidence test. The harness and shim use
      # the SDK's clang rather than a Homebrew LLVM that may be the project's
      # C compiler: the harness links libSystem.
      find_program(LLK_LLVM_OBJDUMP NAMES llvm-objdump
          HINTS "${LLVM_PROJECT_BUILD_DIR}/bin" REQUIRED)
      find_program(LLK_APPLE_CLANG NAMES clang
          HINTS /usr/bin NO_DEFAULT_PATH REQUIRED)
      add_test(NAME ArmSMENativeEvidence
          COMMAND ${LLK_TEST_PYTHON}
              ${CMAKE_SOURCE_DIR}/test/Execution/arm_sme_native_evidence.py
              $<TARGET_FILE:llk-compile> ${CMAKE_SOURCE_DIR}
              ${CMAKE_CURRENT_BINARY_DIR}/ArmSMENativeEvidence
              ${LLK_APPLE_CLANG} ${LLK_LLVM_OBJDUMP})
    endif()
  endif()
```

- [ ] **Step 6: Run the evidence**

```bash
ninja -C build llk-compile
ctest --test-dir build -R '^(ArmSMEEvidenceParser|ArmSMENativeEvidence)$' --output-on-failure
```

Expected: `ArmSMEEvidenceParser` passes everywhere. `ArmSMENativeEvidence` passes on the M5 and prints a record with `svl=512`, nonzero plan/graph/abi hashes, `reference-groups=0`, and the three artifact digests.

- [ ] **Step 7: Commit**

```bash
git add test/Execution/arm_sme_evidence_common.py \
        test/Execution/arm_sme_evidence_test.py \
        test/Execution/arm_sme_native_evidence.py \
        test/Execution/CMakeLists.txt test/FileCheck.cmake
git commit -m "test(execution): run the selected SME contraction natively on Apple M5 (#253)"
```

---

## Task 7: Add the QEMU SME1 evidence runner and CI job (tier T2, secondary)

**Files:**
- Modify: `test/Execution/arm_sme_evidence_common.py` (verify the existing `render_qemu_harness` and streaming ABI shims)
- Create: `test/Execution/arm_sme_qemu_evidence.py`
- Modify: `test/FileCheck.cmake`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/coverage.yml`

**Interfaces:**
- Consumes: Task 6's `emit_object`, `check_object_format`, `check_identity`, `check_sme_instructions`, `render_shim`
- Produces: CTest `ArmSMEExecutionEvidence` under `LLK_BUILD_ARM_EVIDENCE`

- [ ] **Step 1: Verify the freestanding QEMU harness renderer**

The common module already contains `render_qemu_harness` and the
`__arm_sc_memcpy`/`__arm_sc_memmove` streaming ABI shims, added with Task 6's
H25 fix. Verify the committed functions match the reference below (especially
the `%(shims)s` placeholder and the shim calls); do not add a second copy.
Task 7 owns the driver, CTest registration and CI only.

```python
_QEMU_PREAMBLE = r"""
#include <stdint.h>
#include <stddef.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size0;
  int64_t size1;
  int64_t stride0;
  int64_t stride1;
} MemRef2D;

typedef void (*KernelFn)(MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern void %(wrapper)s(MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern void sme_call(KernelFn, MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern unsigned long sme_streaming_bytes(void);

#define ROWS 16
#define COLS 16
#define CANARY 32
#define SENTINEL 0x5a5a5a5au

static unsigned char heap[1 << 20] __attribute__((aligned(64)));
static size_t used;

void *malloc(size_t n) {
  size_t start = (used + 63) & ~(size_t)63;
  if (start + n > sizeof(heap)) for (;;) { }
  used = start + n;
  return heap + start;
}
void *aligned_alloc(size_t alignment, size_t n) {
  size_t start = (used + alignment - 1) & ~(alignment - 1);
  if (start + n > sizeof(heap)) for (;;) { }
  used = start + n;
  return heap + start;
}
void free(void *p) { (void)p; }
void *memset(void *dst, int value, size_t n) {
  unsigned char *bytes = dst;
  for (size_t i = 0; i < n; ++i) bytes[i] = (unsigned char)value;
  return dst;
}
void *memcpy(void *dst, const void *src, size_t n) {
  unsigned char *out = dst;
  const unsigned char *in = src;
  for (size_t i = 0; i < n; ++i) out[i] = in[i];
  return dst;
}
static long syscall3(long, long, long, long);
void memrefCopy(int64_t element_size, void *source, void *destination) {
  (void)element_size; (void)source; (void)destination;
  syscall3(93, 75, 0, 0);
  for (;;) { }
}
/* new_za consults TPIDR2_EL0 and calls this only when it is nonzero. At the
   exported boundary it is zero; a call means the ABI assumption broke. */
void __arm_tpidr2_save(void) { syscall3(93, 76, 0, 0); for (;;) { } }
void __arm_tpidr2_restore(void) { syscall3(93, 76, 0, 0); for (;;) { } }

static long syscall3(long n, long a, long b, long c) {
  register long x8 __asm__("x8") = n;
  register long x0 __asm__("x0") = a;
  register long x1 __asm__("x1") = b;
  register long x2 __asm__("x2") = c;
  __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2)
                   : "memory");
  return x0;
}
#define SAY(text) syscall3(64, 1, (long)(text), sizeof(text) - 1)
#define EXIT(code) do { syscall3(93, (code), 0, 0); for (;;) { } } while (0)

__asm__(".globl _start\n"
        "_start:\n"
        "  mov x0, sp\n"
        "  bl harness_main\n");

%(tables)s

static float a[ROWS * COLS] __attribute__((aligned(64)));
static float b[ROWS * COLS] __attribute__((aligned(64)));
static float init[ROWS * COLS] __attribute__((aligned(64)));
static float out[ROWS * COLS + CANARY] __attribute__((aligned(64)));

static void load(float *dst, const uint32_t *src) {
  for (size_t i = 0; i < ROWS * COLS; ++i) {
    uint32_t bits = src[i];
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    dst[i] = value;
  }
}
static int matches(const float *result, const uint32_t *expected) {
  for (size_t i = 0; i < ROWS * COLS; ++i) {
    uint32_t bits;
    __builtin_memcpy(&bits, &result[i], sizeof(bits));
    if (bits != expected[i]) return 0;
  }
  return 1;
}
static int canary_holds(void) {
  for (size_t i = 0; i < CANARY; ++i) {
    uint32_t bits;
    __builtin_memcpy(&bits, &out[ROWS * COLS + i], sizeof(bits));
    if (bits != SENTINEL) return 0;
  }
  return 1;
}
static void sentinel_fill(void) {
  for (size_t i = 0; i < ROWS * COLS + CANARY; ++i) {
    uint32_t bits = SENTINEL;
    __builtin_memcpy(&out[i], &bits, sizeof(bits));
  }
}
static MemRef2D descriptor(float *buffer) {
  MemRef2D result = {buffer, buffer, 0, ROWS, COLS, COLS, 1};
  return result;
}
"""

_QEMU_BODY = r"""
static int has_hwcap2_sme(unsigned long *stack) {
  unsigned long argc = stack[0];
  unsigned long *p = stack + 1 + argc + 1;
  while (*p) ++p;   /* envp */
  ++p;              /* envp NULL */
  for (; p[0]; p += 2)
    if (p[0] == 26) return (p[1] & (1ul << 23)) != 0;  /* AT_HWCAP2 */
  return 0;
}

void harness_main(unsigned long *stack) {
  if (!has_hwcap2_sme(stack)) {
    SAY("unsupported: no sme\n");
    EXIT(70);
  }
  if (sme_streaming_bytes() != 64) {
    SAY("unsupported: svl is not 512\n");
    EXIT(71);
  }
  MemRef2D da = descriptor(a), db = descriptor(b);
  MemRef2D di = descriptor(init), dout = descriptor(out);

  load(a, A1); load(b, B1); load(init, I1); sentinel_fill();
  %(call)s
  if (!matches(out, E1)) EXIT(73);
  if (!canary_holds()) EXIT(74);

  load(a, A2); load(b, B2); load(init, I2); sentinel_fill();
  %(call)s
  if (!matches(out, E2)) EXIT(73);
  if (!canary_holds()) EXIT(74);

  SAY("pass\n");
  EXIT(0);
}
"""


def render_qemu_harness(wrapper, without_shim=False):
    call = ("%s(&da, &db, &di, &dout);" % wrapper if without_shim
            else "sme_call(&%s, &da, &db, &di, &dout);" % wrapper)
    preamble = _QEMU_PREAMBLE % {"wrapper": wrapper,
                                 "tables": render_tables()}
    return preamble + _QEMU_BODY % {"call": call}
```

- [ ] **Step 2: Create the QEMU driver**

Create `test/Execution/arm_sme_qemu_evidence.py`:

```python
#!/usr/bin/env python3
"""QEMU SME1 contraction evidence (E11, tier T2).

Emulated functional evidence only; it never calibrates native timing. QEMU's
`sme-default-vector-length` is in bytes and defaults to 32 (256 bits), so the
positive run requests 64 bytes explicitly (512 bits).
"""

import os
import signal
import subprocess
import sys

import arm_sme_evidence_common as common

POSITIVE_CPU = "max,sme=on,sme-default-vector-length=64"
NO_SME_CPU = "max,sme=off"
WRONG_SVL_CPU = "max,sme=on,sme-default-vector-length=32"

ZA_PROBE = r"""
	.arch	armv9-a+sme
	.text
	.p2align 2
	.globl	_start
_start:
	smstart	sm
	zero	{za}
	smstop	sm
	mov	x0, #0
	mov	x8, #93
	svc	#0
"""


def _run(command, timeout=300):
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=timeout)
    if result.returncode:
        raise SystemExit("command failed: %s\n%s\n%s" %
                         (" ".join(command), result.stdout, result.stderr))
    return result


def main():
    if len(sys.argv) != 7:
        print("usage: arm_sme_qemu_evidence.py <llk-compile> <source-dir> "
              "<work-dir> <qemu-aarch64> <aarch64-gcc> <aarch64-objdump>",
              file=sys.stderr)
        return 2
    (llk_compile, source_dir, work_dir, qemu, cross_gcc,
     objdump) = sys.argv[1:]
    os.makedirs(work_dir, exist_ok=True)
    failures = []

    obj, record, wrapper = common.emit_object(
        llk_compile, source_dir, work_dir, common.TRIPLE_LINUX,
        common.FEATURES)
    common.check_object_format(obj, "linux")
    common.check_identity(record, common.TRIPLE_LINUX, wrapper, failures,
                          "contraction_f32")
    common.check_sme_instructions(objdump, obj, failures, "contraction_f32")

    shim = os.path.join(work_dir, "shim.S")
    harness_c = os.path.join(work_dir, "harness.c")
    with open(shim, "w", encoding="utf-8") as handle:
        handle.write(common.render_shim("", ".arch\tarmv9-a+sme\n"))
    with open(harness_c, "w", encoding="utf-8") as handle:
        handle.write(common.render_qemu_harness(wrapper))
    shim_o = os.path.join(work_dir, "shim.o")
    binary = os.path.join(work_dir, "harness")
    _run([cross_gcc, "-c", shim, "-o", shim_o])
    _run([cross_gcc, "-static", "-O2", "-nostdlib", "-ffreestanding",
          "-fno-builtin", "-fno-stack-protector", "-Wl,-e,_start",
          harness_c, shim_o, obj, "-o", binary])

    run = subprocess.run([qemu, "-cpu", POSITIVE_CPU, binary],
                         capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        failures.append("positive run exit %d\n%s%s" %
                        (run.returncode, run.stdout, run.stderr))

    for label, cpu, expected in (("no-sme", NO_SME_CPU, 70),
                                 ("wrong-svl", WRONG_SVL_CPU, 71)):
        result = subprocess.run([qemu, "-cpu", cpu, binary],
                                capture_output=True, text=True, timeout=120)
        if result.returncode != expected:
            failures.append("%s: exit %d, expected %d\n%s%s" %
                            (label, result.returncode, expected,
                             result.stdout, result.stderr))

    # QEMU may or may not model "ZA disabled" as a trap. Record what it did;
    # the native T3 control is authoritative for this distinction.
    probe_s = os.path.join(work_dir, "za_probe.S")
    with open(probe_s, "w", encoding="utf-8") as handle:
        handle.write(ZA_PROBE)
    probe_o = os.path.join(work_dir, "za_probe.o")
    probe = os.path.join(work_dir, "za_probe")
    _run([cross_gcc, "-c", probe_s, "-o", probe_o])
    _run([cross_gcc, "-static", "-nostdlib", "-Wl,-e,_start", probe_o,
          "-o", probe])
    za = subprocess.run([qemu, "-cpu", POSITIVE_CPU, probe],
                        capture_output=True, text=True, timeout=60)
    if za.returncode in (-signal.SIGILL, 132):
        za_note = "SIGILL"
    elif za.returncode == 0:
        za_note = "not-modeled-by-qemu"
    else:
        failures.append("ZA probe exited %d" % za.returncode)
        za_note = "unexpected"

    off_c = os.path.join(work_dir, "harness_without_shim.c")
    with open(off_c, "w", encoding="utf-8") as handle:
        handle.write(common.render_qemu_harness(wrapper, without_shim=True))
    off_bin = os.path.join(work_dir, "harness_without_shim")
    _run([cross_gcc, "-static", "-O2", "-nostdlib", "-ffreestanding",
          "-fno-builtin", "-fno-stack-protector", "-Wl,-e,_start",
          off_c, shim_o, obj, "-o", off_bin])
    try:
        off = subprocess.run([qemu, "-cpu", POSITIVE_CPU, off_bin],
                             capture_output=True, text=True, timeout=30)
        if off.returncode == 0:
            failures.append("the wrapper ran successfully with SM off")
    except subprocess.TimeoutExpired:
        pass

    version = _run([qemu, "--version"]).stdout.splitlines()[0]
    print(
        "arm sme qemu evidence: qemu=%r cpu=%s wrapper=%s plan=%s graph=%s "
        "abi=%s svl=512 selected-groups=%s reference-groups=%s "
        "object-sha256=%s harness-sha256=%s shim-sha256=%s za-probe=%s" % (
            version, POSITIVE_CPU, wrapper, record.get("planId"),
            record.get("sourceGraphHash"), record.get("abiHash"),
            record.get("selectedGroupsVerified"),
            record.get("referenceGroupsLowered"), common.sha256_file(obj),
            common.sha256_file(harness_c), common.sha256_file(shim),
            za_note))

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("arm sme qemu evidence: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Register the CTest and CI job**

In `test/FileCheck.cmake`, extend the `if(LLK_BUILD_ARM_EVIDENCE)` block:

```cmake
    find_program(LLK_AARCH64_OBJDUMP
        NAMES aarch64-linux-gnu-objdump REQUIRED)
    add_test(NAME ArmSMEExecutionEvidence
        COMMAND ${LLK_TEST_PYTHON}
            ${CMAKE_SOURCE_DIR}/test/Execution/arm_sme_qemu_evidence.py
            $<TARGET_FILE:llk-compile> ${CMAKE_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/ArmSMEExecutionEvidence
            ${LLK_QEMU_AARCH64} ${LLK_AARCH64_GCC}
            ${LLK_AARCH64_OBJDUMP})
```

In `.github/workflows/ci.yml`, after the Arm SVE evidence step:

```yaml
      - name: Run Arm SME emulated evidence
        run: |
          cd build
          set -o pipefail
          ctest -R '^ArmSMEExecutionEvidence$' --no-tests=error --output-on-failure 2>&1 | tee arm-sme-evidence.log
```

Add the evidence paths to the existing upload step:

```yaml
            build/arm-sme-evidence.log
            build/test/ArmSMEExecutionEvidence/
```

In `.github/workflows/coverage.yml`, mirror the evidence run after the SVE step:

```yaml
      - name: Run Arm SME emulated evidence
        run: |
          cd build
          set -o pipefail
          ctest -R '^ArmSMEExecutionEvidence$' --no-tests=error --output-on-failure 2>&1 | tee arm-sme-evidence.log
```

- [ ] **Step 4: Run the QEMU evidence**

On a Linux host with the cross toolchain (or in CI):

```bash
ninja -C build llk-compile
ctest --test-dir build -R '^ArmSMEExecutionEvidence$' --output-on-failure
```

Expected: positive exit 0; `no-sme` exit 70; `wrong-svl` exit 71; the ZA probe
either SIGILL or `not-modeled-by-qemu` (recorded, not silently passed); the
without-shim run never exits 0.

- [ ] **Step 5: Commit**

```bash
git add test/Execution/arm_sme_evidence_common.py \
        test/Execution/arm_sme_qemu_evidence.py test/FileCheck.cmake \
        .github/workflows/ci.yml .github/workflows/coverage.yml
git commit -m "test(execution): run the selected SME contraction under qemu (#253)"
```

---

## Task 8: Capability receipt, documentation corrections and acceptance ledger

**Files:**
- Modify: `lib/Conversion/MicroMapping/FrameworkCapabilities.cpp`
- Modify: `test/Mapping/arm_sme_backend.cpp`
- Modify: `test/Mapping/arm_sve_backend.cpp` (one bf16 receipt case)
- Create: `docs/reviews/2026-10-24-arm-sme-stateful-backend-acceptance.md`
- Modify: `docs/reviews/2026-10-24-arm-sve-executable-acceptance.md`
- Modify: `docs/reviews/arm-mapping-targets-acceptance.md`
- Modify: `docs/manual/source/concepts.rst`
- Modify: `docs/manual/source/tools/compiler.rst`

**Interfaces:**
- Consumes: Task 4's emitters, Task 5's compilation results, Task 6/7's evidence records
- Produces: per-group capability receipts and the revision-pinned E11 ledger

- [ ] **Step 1: Write the failing receipt tests**

Append to `test/Mapping/arm_sme_backend.cpp`:

```cpp
TEST(ArmSMEBackend, CapabilityReceiptDrawsTheExecutableBoundary) {
  using mlir::llk::framework::CapabilityReceipt;
  using mlir::llk::framework::CapabilityStatus;

  auto fixture = arm_test::mapFixture("arm-sme", kContraction);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::Operation *kernel = nullptr;
  fixture->source->walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  ASSERT_NE(kernel, nullptr);
  mlir::llk::workload::WorkloadContract contract;
  CapabilityReceipt planned = mlir::llk::framework::describeCapabilities(
      kernel, contract, *fixture->target, &fixture->analysis, &fixture->plan);
  EXPECT_EQ(planned.selectedLowering, CapabilityStatus::NotChecked);

  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_contraction";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  CapabilityReceipt realized = mlir::llk::framework::describeCapabilities(
      kernel, contract, *fixture->target, &fixture->analysis, &fixture->plan,
      &*compiled);
  EXPECT_EQ(realized.selectedLowering, CapabilityStatus::Supported);

  auto bf16 = arm_test::mapFixture("arm-sme", kContractionBf16);
  ASSERT_TRUE(static_cast<bool>(bf16)) << llvm::toString(bf16.takeError());
  mlir::Operation *bf16Kernel = nullptr;
  bf16->source->walk([&](mlir::Operation *op) {
    if (!bf16Kernel && op->getName().getStringRef() == "micro.kernel")
      bf16Kernel = op;
  });
  ASSERT_NE(bf16Kernel, nullptr);
  CapabilityReceipt refused = mlir::llk::framework::describeCapabilities(
      bf16Kernel, contract, *bf16->target, &bf16->analysis, &bf16->plan);
  EXPECT_EQ(refused.selectedLowering, CapabilityStatus::Unsupported);
  const std::string reasons = llvm::join(refused.reasons, "\n");
  EXPECT_NE(reasons.find("sme_mma_za"), std::string::npos) << reasons;
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
ninja -C build LLKMappingTests
./build/bin/LLKMappingTests --gtest_filter=ArmSMEBackend.CapabilityReceiptDrawsTheExecutableBoundary
```

Expected: FAIL. The receipt currently asks only `hasLowering()` on the key,
which is true for `sme_mma_za`, so the bf16 plan is reported `NotChecked`
instead of `Unsupported`.

- [ ] **Step 3: Make the receipt ask the same per-group question as lowering**

In `lib/Conversion/MicroMapping/FrameworkCapabilities.cpp`, inside the
`selected lowering` block, build the instance groups once before the placement
loop:

```cpp
    llvm::DenseMap<uint64_t, llvm::SmallVector<mlir::Operation *>>
        opsByInstance;
    if (kernel)
      kernel->walk([&](mlir::Operation *op) {
        auto mapping =
            op->getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
        if (!mapping)
          return;
        if (auto instance = mapping.getAs<mlir::IntegerAttr>("instance"))
          opsByInstance[instance.getInt()].push_back(op);
      });
```

Replace the key-level lowering check:

```cpp
        else if (!emitter->hasLowering())
          rejectSelected("emitter '" + key +
                         "' only verifies its bundle, so the selected group "
                         "would fall back to the reference bridge");
```

with the per-group check:

```cpp
        else if (!emitter->hasLoweringFor(opsByInstance.lookup(
                     placement.instance)))
          rejectSelected("emitter '" + key +
                         "' cannot lower the selected group's operations, so "
                         "it would fall back to the reference bridge");
```

Add `#include "llvm/ADT/DenseMap.h"` and
`#include "llvm/ADT/SmallVector.h"` to the file's includes.

- [ ] **Step 4: Run the receipts and the full mapping suite**

```bash
ninja -C build LLKMappingTests
./build/bin/LLKMappingTests --gtest_filter='ArmSMEBackend.*:ArmSVEBackend.*:ArmTargets.*'
```

Expected: PASS. The f32 contraction is `Supported` after its selected compile;
the bf16 contraction is `Unsupported` before any compile, with a reason naming
`sme_mma_za`; the SVE f32 add and SiLU receipts are unchanged.

- [ ] **Step 5: Correct the superseded documentation**

In `docs/manual/source/concepts.rst`, replace the sentence at lines 269-272:

```rst
The generic accelerator is an architecture exploration target rather than an
accelerator hardware backend. The Arm profiles model VL=SVL=512 for static
mapping and analysis. The executable ``arm-sve`` f32 add/multiply subset has an
emulated acceptance path, and the executable ``arm-sme`` f32 16x16x16
initialized contraction has a native Apple M5 acceptance path with a QEMU SME1
secondary run. Other SME signatures, bf16/i8 SME execution, ZA preservation
across calls and timing remain non-executable.
```

In `docs/reviews/2026-10-24-arm-sve-executable-acceptance.md`, replace this
exact bullet:

```markdown
- No SME execution, no ZA realization, no bf16/i8/SiLU/reduction/conversion
  execution, no calibrated timing and no CPU-schedule retargeting.
```

with:

```markdown
- E11 (#253) supersedes the SME half of this boundary: `arm-sme` now executes
  the f32 16x16x16 initialized contraction on a native Apple M5 (T3) and under
  QEMU SME1 (T2). bf16/i8 SME execution, ZA preservation across calls, SME2 and
  calibrated timing remain out of scope. No ZA realization beyond the one
  contraction tile, no non-SME signature execution and no CPU-schedule
  retargeting is claimed. This ledger's SVE evidence is unchanged.
```

Then replace this exact final bullet:

```markdown
- #175 stays open until a native executable Arm acceptance is actually met;
  E11 (#253) covers the SME stateful backend.
```

with:

```markdown
- #175 closes with E11's ledger
  (`docs/reviews/2026-10-24-arm-sme-stateful-backend-acceptance.md`), which
  records the native T3 run and the QEMU T2 run.
```

In `docs/reviews/arm-mapping-targets-acceptance.md`, replace this exact bullet:

```markdown
- Every `arm-sve` signature outside the executable f32 add/multiply subset, and
  every `arm-sme` signature, remains static mapping and analysis. Their selected
  lowering is rejected rather than emulated.
```

with:

```markdown
- Every `arm-sve` signature outside the executable f32 add/multiply subset
  remains static mapping and analysis. On `arm-sme`, the f32 16x16x16
  initialized contraction is executable as of E11; every other signature
  remains static and is rejected rather than emulated.
```

In `docs/manual/source/tools/compiler.rst`, update the identity-sidecar prose
(around lines 321-324) so it lists `schemaVersion`,
`streamingVectorLengthBits` (null for non-SME targets) and
`invocationAttributes` (the LLVM function attributes applied to the exported
wrapper), in addition to the existing fields.

- [ ] **Step 6: Write the revision-pinned acceptance ledger**

Create `docs/reviews/2026-10-24-arm-sme-stateful-backend-acceptance.md` after
the runs complete. Copy every value from the printed evidence records; do not
invent a hash or a version.

```markdown
# Arm SME stateful executable acceptance (E11)

Implements `docs/superpowers/specs/2026-10-24-arm-sme-stateful-backend-design.md`
and `docs/superpowers/plans/2026-10-24-arm-sme-stateful-backend.md`.

**Scope:** executable `arm-sme` f32 16x16x16 initialized contraction through a
native Apple M5 run (T3, gating) and a QEMU linux-user SME1 run (T2, CI).

## Identity

| Item | Value |
|---|---|
| Implementation head | `<commit>` |
| Design head | `<design commit>` |
| Native host | `<hw.model>`, macOS `<version>`, `FEAT_SME=1`, `FEAT_SME2=1`, SVL=512 |
| Native harness compiler | `<clang --version first line>` |
| QEMU | `<qemu --version first line>` |
| QEMU CPU | `max,sme=on,sme-default-vector-length=64` |
| CI LLVM/MLIR | 22.1.8, `LLVM_TARGETS_TO_BUILD="X86;AArch64"` |

## Accepted executable domain

| Item | Value |
|---|---|
| Target | `arm-sme` |
| Operation | `micro.mma` 16x16x16, f32 inputs and accumulator, initialized from `%init` |
| Streaming | `aarch64_pstate_sm_enabled` kernel and wrapper; assembly `smstart sm`/`smstop sm` shim |
| ZA | `aarch64_new_za`: `smstart za`, `zero {za}`, slice extraction, `smstop za`; not preserved across calls |
| ABI | `_mlir_ciface_<kernel>`, inputs then borrowed outputs, rank-two descriptors |
| SVL | exactly 512 bits; `CNTB` = 64 bytes checked before the first call |

## Evidence records

Paste the native and QEMU evidence lines here verbatim:

```text
<arm sme native evidence: ...>
<arm sme qemu evidence: ...>
```

`selectedGroupsVerified` covers the contraction group plus the fixture's
movement groups (the staged DRAM-to-SRAM and SRAM-to-accumulator copies and the
result copy), which the emitter claims and the generic Micro-to-Linalg bridge
carries. `backendGroupsRealized` counts the contraction group the ArmSME pass
realized; `referenceGroupsLowered` is zero. The two numbers are not expected to
be equal.

## Disassembly

The object contains `fmopa`, `zero {za}` and `smstart za`. A non-SME object
fails the evidence test; it is not skipped.

## Negative controls

| Control | Required | Native | QEMU |
|---|---|---|---|
| bf16 contraction selected compile | refuses; no reference fallback | pass | pass |
| i8 contraction mapping | retains no plan | pass | pass |
| missing SME | exit 70 | sysctl branch | `sme=off` exit 70 |
| `smstart sm` then `zero {za}` | SIGILL | pass | `<SIGILL or not-modeled-by-qemu>` |
| wrapper without the streaming shim | never a matching pass | pass | pass |
| wrong SVL | exit 71 | fixed at 512, checked | `sme-default-vector-length=32` exit 71 |
| reference-group fallback | zero groups | pass | pass |

## Claims and limits

- T3 is native functional evidence on the recorded M5; T2 is emulated functional
  evidence. Neither calibrates cycles, bytes, storage or timing; `micro-perf`
  numbers remain modeled predictions.
- The native T3 run is not reproducible in CI. CI proves T2 only; the T3 record
  is pinned by the host, toolchain and artifact digests above.
- No bf16/i8 SME execution, no SME2, no multi-tile kernels, no ZA preservation
  across calls, no physical SRAM/L2/DMA realization, no calibrated timing.
- The JIT path refuses selected `arm-sme` on hosts without an SVL probe
  (`checkHostRequirements` never assumes SVL=512). On macOS,
  `llvm::sys::getHostCPUFeatures()` also does not report `sme`, so the JIT
  route refuses with "selected target requires host feature 'sme'" even on an
  SME-capable M5; the AOT path does not consult the host check. The Python frontend's
  selected-target check uses the no-vector-length overload, so it also refuses
  `arm-sve` and `arm-sme` even on capable Linux hardware; that is fail-closed
  and is left to the later frontend integration increment. The AOT object plus
  the native/QEMU harness is the executable route.
- The E10 SVE evidence and the AVX2 baseline are unchanged.
- #175 closes with this ledger: a non-AVX2 selected-target backend has
  reproducible numerical execution evidence, including initialized
  accumulation, a documented runner identity and explicit omissions.
```

- [ ] **Step 7: Run the full suite and the documentation build**

```bash
ninja -C build check-llk
ctest --test-dir build -R '^ArmSMEEvidenceParser$'
ninja -C build docs
```

Expected: the full suite passes (excluding the documented selected-AVX2
host-unsupported cases on macOS), and the Sphinx manual builds with
`-n -W --keep-going` in CI.

- [ ] **Step 8: Commit**

```bash
git add lib/Conversion/MicroMapping/FrameworkCapabilities.cpp \
        test/Mapping/arm_sme_backend.cpp test/Mapping/arm_sve_backend.cpp \
        docs/reviews/2026-10-24-arm-sme-stateful-backend-acceptance.md \
        docs/reviews/2026-10-24-arm-sve-executable-acceptance.md \
        docs/reviews/arm-mapping-targets-acceptance.md \
        docs/manual/source/concepts.rst
git commit -m "docs(target): accept the stateful Arm SME backend and close #175 (#253)"
```

---

## Self-Review

**Spec coverage**

- §1 scope and f32-only boundary: Tasks 4, 5 and 8; bf16 refusal in Tasks 4 and 8.
- §2 audited behavior and probes: inputs to the design; no code task needed.
- §3 native-first runner: Tasks 6 (T3 gating) and 7 (T2 CI); §7/§8 omissions in Task 8's ledger.
- §4 facts/assumptions: encoded in the 512-bit checks, the machine profile's unchanged `za.0`, and the canonical/backend split (no canonical change anywhere in the plan).
- §5 streaming/ZA ABI: Task 5's `enable-arm-streaming` call, the shim in Task 6, the ZA probe controls in Tasks 6 and 7, and the two-call harness data in Task 6.
- §6 selected lowering contract: Tasks 4 and 5, including the post-bufferization hook from Task 3 and the invocation attributes from Tasks 1-2.
- §7 artifacts/runners: Tasks 6 and 7 (Mach-O and ELF emission, harnesses, disassembly, QEMU CPU properties).
- §8 evidence tiers/identities: Task 2's identity fields, Task 6/7 records, Task 8's ledger.
- §9 numerical contract and negative controls: Task 6's exact integer tables and two calls; controls in Tasks 4, 6, 7.
- §10 acceptance fixtures/evidence: Task 5's end-to-end test, Tasks 6/7 runners, Task 8's receipt.
- §11 implementation sequence: Tasks 1-8 follow the seven-step shape.
- §12 limits/#175 closure: Task 8's ledger.

**Placeholder scan**

No `TBD`/`TODO` implementation steps. The ledger's angled values and the
`<commit>` fields are evidence to be copied from the actual runs, not code
placeholders; the plan states that explicitly.

**Type consistency**

- `streamingVectorLengthBits` and `invocationAttributes` are spelled the same in
  the contract, identity, JSON and Python checks.
- `applyInvocationAttributes(module, wrapper, attributes)` is used by the AOT
  and JIT paths with the same argument order.
- `buildPostBufferizationBackendPipeline` is the hook name in the base class,
  `MappedCompilation` and `SMETarget`.
- `sme_mma_za`, `llk.sme.selected_groups` and `llk.backend_groups_realized` are
  spelled identically across the emitter, both passes and the tests.
- `render_native_harness(wrapper, without_shim=False)` and
  `render_qemu_harness(wrapper, without_shim=False)` share the `wrapper` and
  `without_shim` names used by both drivers.

