# Arm SVE fixed-VL executable backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `arm-sve` target lower its f32 vector add/multiply subset to AArch64 SVE code, emit a cross-lowered object with a complete identity record, and prove the result numerically under QEMU in CI.

**Architecture:** Selected `sve_vector_add`/`sve_vector_mul` emitters hand their covered Micro operations to a new SVE backend pass, which vectorizes the generated Linalg operations with scalable vectors derived from the selected `VW`. `llk-compile` gains a target-neutral AOT object path that reuses the existing C-interface ABI, and a generated freestanding harness runs the object under `qemu-aarch64` with VL=512 forcing.

**Tech Stack:** C++20, MLIR/LLVM 22.1.8 (CI) and 24 (local), CMake/Ninja, CTest, Python 3 standard library, QEMU linux-user 8.2.

**Spec:** `docs/superpowers/specs/2026-10-24-arm-sve-executable-backend-design.md`

## Global Constraints

- The project builds against LLVM 22.1.8 (CI) and LLVM 24 (local). Gate any API difference with `#if LLVM_VERSION_MAJOR >= 21` and verify both configurations before merging.
- Use modern MLIR/LLVM APIs only (`Op::create`, `mlir::cast<T>`, `isa<T>`); the build uses `-Werror=deprecated-declarations`.
- Generic `lib/Mapping`, `lib/Machine` and `lib/Conversion` code must never branch on a target name. All `arm-sve` behavior lives in `lib/Target/AArch64/Mapping/`.
- No MachineModel or Micro dialect schema change. Adding fields to the internal `TargetCodegenRequirements` struct is allowed; give every new field a default member initializer.
- Keep every existing static Arm test active. Narrow a static-only assertion only where this increment legitimately changes it, and never delete a negative control.
- The executable subset is exactly: `arm-sve`, f32, `micro.vector "add"` and `"mul"`, direct-MicroIR fixtures. Everything else keeps rejecting.
- No silent reference fallback: a selected-target compilation must report `referenceGroupsLowered == 0` for the covered group.
- Modeled costs stay separate from evidence. Emulated results never become timing claims.
- TDD: every task starts with a failing test, then minimal code, then a commit. Work in an isolated worktree branch named `feat/arm-sve-executable-backend`.
- `#175` stays open. This plan implements E10 only; E11 (SME) is separate.

---

## File map

| File | Responsibility |
|---|---|
| `include/LLK/Runtime/MappedObject.h` (new) | AOT object emission and identity-record interfaces |
| `runtime/MappedObject.cpp` (new) | LLVM-dialect lowering reuse, AArch64 object emission, JSON identity writer |
| `tools/llk-compile/llk-compile.cpp` | `--emit=object`, `--target-triple`, `--target-features`, `--output`, `--emit-identity` |
| `include/LLK/Target/AArch64/Mapping/SVEMappingTarget.h` | Executable emitter/pipeline declarations |
| `lib/Target/AArch64/Mapping/SVEMappingTarget.cpp` | SVE target wrapper: codegen requirements, emitters, pipeline |
| `lib/Target/AArch64/Mapping/SVEBackendLowering.cpp` (new) | Scalable-vector backend pass for the selected subset |
| `lib/Target/AArch64/Mapping/SVEBundleLowering.cpp` (new) | `sve_vector_add`/`sve_vector_mul` bundle emitters |
| `lib/Target/CMakeLists.txt` | Add the two new sources to `LLKTargetMapping` |
| `include/LLK/Mapping/CodegenRequirements.h` | Optional vector-length requirement |
| `lib/Mapping/CodegenRequirements.cpp` | Vector-length host check |
| `runtime/MappedExecutable.cpp` | Probe and pass the host vector length |
| `test/Mapping/arm_sve_backend.cpp` (new) | Emitter, vectorization and negative GTest cases |
| `test/Mapping/codegen_requirements.cpp` (new) | Vector-length requirement unit tests |
| `test/Execution/aot_object_emission.py` (new) | ELF/identity acceptance for Task 1 |
| `test/Execution/arm_sve_evidence.py` (new) | Harness generation, qemu execution and identity checks |
| `test/Conversion/MicroMapping/Inputs/arm/vector_{add,mul}_f32_full.mlir` (new) | Full-width fixtures |
| `test/FileCheck.cmake` | Register `AOTObjectEmission`, option-gated `ArmSVEExecutionEvidence` |
| `.github/workflows/ci.yml` | AArch64 LLVM target, qemu/cross toolchain, evidence test |

---

### Task 1: Target-neutral AOT object emission with an identity sidecar

**Files:**
- Create: `include/LLK/Runtime/MappedObject.h`
- Create: `runtime/MappedObject.cpp`
- Modify: `runtime/CMakeLists.txt`
- Modify: `tools/llk-compile/llk-compile.cpp`
- Create: `test/Execution/aot_object_emission.py`
- Modify: `test/FileCheck.cmake`

**Interfaces:**
- Consumes: `llk::PreparedMappedKernel` (`include/LLK/Runtime/PreparedMappedKernel.h`), `llk::addKernelToLLVMPasses` (`include/LLK/Runtime/JitCache.h`), `llk::MappedCompilation` fields.
- Produces: `llk::emitMappedObject(...)`, `llk::writeMappedObjectIdentity(...)`, the `--emit=object` CLI contract and the identity JSON schema used by Task 4.

- [ ] **Step 1: Write the failing acceptance test**

Create `test/Execution/aot_object_emission.py`:

```python
#!/usr/bin/env python3
"""AOT object emission: ELF identity and a complete sidecar (E10 task 1)."""

import json
import os
import struct
import subprocess
import sys


def run(command, expect_success=True):
    result = subprocess.run(command, capture_output=True, text=True)
    if expect_success and result.returncode != 0:
        raise SystemExit("command failed: %s\n%s\n%s" %
                         (" ".join(command), result.stdout, result.stderr))
    return result


def main():
    llk_compile, source_dir, work_dir = sys.argv[1:]
    os.makedirs(work_dir, exist_ok=True)
    source = os.path.join(source_dir,
                          "test/Conversion/MicroMapping/Inputs/arm",
                          "vector_add_f32.mlir")
    obj = os.path.join(work_dir, "arm_vector_add_f32.o")
    identity = os.path.join(work_dir, "identity.json")
    common = [
        llk_compile,
        "--mapping-target=arm-sve",
        "--mapping-root=" + source_dir,
        "--mapping-backend=reference",
        "--mapping-stop=lowered",
        "--emit=object",
        "--target-features=+sve",
    ]
    run(common + ["--target-triple=aarch64-unknown-linux-gnu",
                  "--output=" + obj, "--emit-identity=" + identity, source])

    data = open(obj, "rb").read()
    assert data[:4] == b"\x7fELF", "output is not an ELF object"
    assert struct.unpack_from("<H", data, 18)[0] == 183, "not EM_AARCH64"

    record = json.load(open(identity))
    for key in ("schemaVersion", "compilerVersion", "llvmVersion",
                "targetTriple", "targetFeatures", "entrySymbol",
                "wrapperSymbol", "abiVersion", "abiHash", "sourceGraphHash",
                "planId", "machineHash", "targetHash", "selectedGroupsVerified",
                "backendGroupsRealized", "referenceGroupsLowered",
                "executionIdentity"):
        assert key in record, "identity has no " + key
    assert record["targetTriple"] == "aarch64-unknown-linux-gnu"
    assert record["targetFeatures"] == ["+sve"]
    assert record["entrySymbol"] == "arm_vector_add_f32"
    assert record["wrapperSymbol"] == "_mlir_ciface_arm_vector_add_f32"
    assert record["wrapperSymbol"].encode() in data, "wrapper symbol not exported"
    assert record["referenceGroupsLowered"] == 1

    missing = run(common[:-2] + ["--output=" + obj, "--emit-identity=" + identity,
                                source], expect_success=False)
    assert missing.returncode != 0
    assert "--target-triple" in missing.stderr, missing.stderr
    print("aot object emission: ok")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Register and run the test to verify it fails**

Add to `test/FileCheck.cmake` beside the other Python pipelines (inside the `LLK_BUILD_TOOLS` block):

```cmake
  add_test(NAME AOTObjectEmission
      COMMAND ${LLK_TEST_PYTHON} ${CMAKE_SOURCE_DIR}/test/Execution/aot_object_emission.py
          $<TARGET_FILE:llk-compile> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/AOTObjectEmission
  )
```

Run: `cmake -S . -B build && ctest --test-dir build -R '^AOTObjectEmission$' --output-on-failure`
Expected: FAIL with `Unsupported --emit=object`.

- [ ] **Step 3: Declare the object-emission interface**

Create `include/LLK/Runtime/MappedObject.h`:

```cpp
#ifndef LLK_RUNTIME_MAPPEDOBJECT_H
#define LLK_RUNTIME_MAPPEDOBJECT_H

#include "LLK/Runtime/PreparedMappedKernel.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llk {

struct MappedObjectOptions {
  std::string triple;
  std::string cpu;
  std::vector<std::string> features;
};

struct MappedObjectIdentity {
  uint32_t schemaVersion = 1;
  std::string compilerVersion;
  std::string llvmVersion;
  std::string targetTriple;
  std::string cpu;
  std::vector<std::string> targetFeatures;
  std::string entrySymbol;
  std::string wrapperSymbol;
  uint32_t abiVersion = 0;
  uint64_t abiHash = 0;
  uint64_t sourceGraphHash = 0;
  uint64_t planId = 0;
  uint64_t machineHash = 0;
  uint64_t targetHash = 0;
  unsigned selectedGroupsVerified = 0;
  unsigned backendGroupsRealized = 0;
  unsigned referenceGroupsLowered = 0;
  std::string executionIdentity;
};

/// Lowers `prepared` through the same LLVM pipeline the JIT uses and writes an
/// object for `options.triple`. Returns the C-interface wrapper symbol.
llvm::Expected<std::string>
emitMappedObject(PreparedMappedKernel &prepared,
                 const MappedObjectOptions &options,
                 llvm::raw_ostream &objectStream);

llvm::Error writeMappedObjectIdentity(const MappedObjectIdentity &identity,
                                      llvm::raw_ostream &jsonStream);

} // namespace llk
#endif
```

- [ ] **Step 4: Implement object emission**

Create `runtime/MappedObject.cpp` with this core (include `LLK/Runtime/JitCache.h`, `mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h`, `mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h`, `mlir/Target/LLVMIR/ModuleTranslation.h`, `llvm/IR/LegacyPassManager.h`, `llvm/MC/TargetRegistry.h`, `llvm/Support/TargetSelect.h`, `llvm/Target/TargetMachine.h`, `llvm/Support/JSON.h`):

```cpp
llvm::Expected<std::string>
emitMappedObject(PreparedMappedKernel &prepared,
                 const MappedObjectOptions &options,
                 llvm::raw_ostream &objectStream) {
  if (options.triple.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AOT emission requires --target-triple");
  mlir::ModuleOp module = prepared.module.get();
  auto function = module.lookupSymbol<mlir::func::FuncOp>(prepared.entrySymbol);
  if (!function)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "AOT emission cannot find kernel '" + prepared.entrySymbol + "'");
  function->setAttr("llvm.emit_c_interface",
                    mlir::UnitAttr::get(module.getContext()));

  mlir::registerBuiltinDialectTranslation(*module.getContext());
  mlir::registerLLVMDialectTranslation(*module.getContext());
  mlir::PassManager pm(module.getContext());
  addKernelToLLVMPasses(pm);
  if (mlir::failed(pm.run(module)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "lowering kernel '" + prepared.entrySymbol +
                                       "' to the LLVM dialect failed");

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "translating kernel '" + prepared.entrySymbol +
                                       "' to LLVM IR failed");
  std::string wrapper = "_mlir_ciface_" + prepared.entrySymbol;
  if (!llvmModule->getFunction(wrapper))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no C-interface wrapper '" + wrapper + "'");

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(options.triple, error);
  if (!target)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), error);
  std::string features;
  for (const std::string &feature : options.features) {
    if (!features.empty())
      features += ',';
    features += feature;
  }
  llvm::TargetOptions targetOptions;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      options.triple, options.cpu, features, targetOptions, std::nullopt));
  if (!machine)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot create a target machine for " +
                                       options.triple);
  llvmModule->setTargetTriple(options.triple);
  llvmModule->setDataLayout(machine->createDataLayout());
  llvm::legacy::PassManager codegen;
  if (machine->addPassesToEmitFile(codegen, objectStream, nullptr,
                                   llvm::CodeGenFileType::ObjectFile))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "target cannot emit an object file");
  codegen.run(*llvmModule);
  return wrapper;
}
```

Implement `writeMappedObjectIdentity` with `llvm::json::Object`, writing hashes in the project's existing `llvm::formatv("{0:x}", value)` form (a `0x`-prefixed hex string, not zero-padded) and `targetFeatures` as an array. Consumers parse them with `int(value, 16)`, never by fixed width. Add `MappedObject.cpp` to `runtime/CMakeLists.txt`.

- [ ] **Step 5: Wire `--emit=object` into the compiler**

In `tools/llk-compile/llk-compile.cpp` add:

```cpp
static cl::opt<std::string> emitOutput(
    "output", cl::desc("Output path for --emit=object"), cl::init(""));
static cl::opt<std::string> emitTargetTriple(
    "target-triple", cl::desc("Target triple for --emit=object"), cl::init(""));
static cl::opt<std::string> emitTargetFeatures(
    "target-features",
    cl::desc("Comma-separated LLVM features for --emit=object, e.g. +sve"),
    cl::init(""));
static cl::opt<std::string> emitIdentity(
    "emit-identity",
    cl::desc("Write the AOT identity JSON record to this path"),
    cl::init(""));
```

Extend the `--emit` validation to accept `object`, and reject it outside the mapped path:

```cpp
  if (emit == "object" && mappingTargetName.empty()) {
    llvm::errs() << "--emit=object requires --mapping-target\n";
    return 1;
  }
  if (emit == "object" && (emitOutput.empty() || emitTargetTriple.empty() ||
                           emitIdentity.empty())) {
    llvm::errs() << "--emit=object requires --output, --target-triple and "
                    "--emit-identity\n";
    return 1;
  }
```

In `runMappedCompilation`, force `options.stop = llk::MappedStop::Lowered` when `emitMode == "object"`, and before the final `compiled->module->print(...)` branch emit the object:

```cpp
  if (emitMode == "object") {
    llvm::Expected<llk::PreparedMappedKernel> prepared =
        llk::prepareMappedKernelForInvocation(*compiled->module,
                                              options.entrySymbol);
    if (!prepared) {
      llvm::errs() << llvm::toString(prepared.takeError()) << "\n";
      return 1;
    }
    std::vector<std::string> features;
    llvm::SmallVector<llvm::StringRef> split;
    llvm::StringRef(emitTargetFeatures).split(split, ',');
    for (llvm::StringRef feature : split)
      if (!feature.empty())
        features.push_back(feature.str());
    llk::MappedObjectOptions objectOptions;
    objectOptions.triple = emitTargetTriple;
    objectOptions.features = std::move(features);
    std::error_code error;
    llvm::raw_fd_ostream objectFile(emitOutput, error);
    if (error) { llvm::errs() << error.message() << "\n"; return 1; }
    llvm::Expected<std::string> wrapper =
        llk::emitMappedObject(*prepared, objectOptions, objectFile);
    if (!wrapper) {
      llvm::errs() << llvm::toString(wrapper.takeError()) << "\n";
      return 1;
    }
    llk::MappedObjectIdentity record;
    record.compilerVersion = LLK_COMPILER_VERSION;
    record.llvmVersion = LLVM_VERSION_STRING;
    record.targetTriple = emitTargetTriple;
    record.targetFeatures = objectOptions.features;
    record.entrySymbol = prepared->entrySymbol;
    record.wrapperSymbol = *wrapper;
    record.abiVersion = prepared->abi.version;
    record.abiHash = llk::computeKernelAbiHash(prepared->abi);
    record.sourceGraphHash = sourceGraphHash;
    record.planId = compiled->planId;
    record.machineHash = compiled->machineHash;
    record.targetHash = compiled->targetHash;
    record.selectedGroupsVerified = compiled->selectedGroupsVerified;
    record.backendGroupsRealized = compiled->backendGroupsRealized;
    record.referenceGroupsLowered = compiled->referenceGroupsLowered;
    record.executionIdentity = compiled->executionIdentity;
    llvm::raw_fd_ostream identityFile(emitIdentity, error);
    if (error) { llvm::errs() << error.message() << "\n"; return 1; }
    if (llvm::Error writeError =
            llk::writeMappedObjectIdentity(record, identityFile)) {
      llvm::errs() << llvm::toString(std::move(writeError)) << "\n";
      return 1;
    }
    llvm::outs() << "aot: object=" << emitOutput << " wrapper=" << *wrapper
                 << " triple=" << emitTargetTriple << "\n";
    return 0;
  }
```

Add `#include "LLK/Runtime/MappedObject.h"`,
`#include "LLK/Conversion/MappedKernelAbi.h"` (for
`prepareMappedKernelForInvocation` and `computeKernelAbiHash`),
`#include "LLK/Version.h"`, and `#include "llvm/Config/llvm-config.h"` as
needed. Add `LLKMappedKernelAbi` to `llk-compile`'s link libraries in
`tools/CMakeLists.txt` so the preparation symbols are linked directly rather
than relying on a transitive static-library order.

- [ ] **Step 6: Build and run the test to verify it passes**

Run: `ninja -C build llk-compile && ctest --test-dir build -R '^AOTObjectEmission$' --output-on-failure`
Expected: PASS, printing `aot object emission: ok`.

- [ ] **Step 7: Commit**

```bash
git add include/LLK/Runtime/MappedObject.h runtime/MappedObject.cpp \
        runtime/CMakeLists.txt tools/llk-compile/llk-compile.cpp \
        test/Execution/aot_object_emission.py test/FileCheck.cmake
git commit -m "feat(compile): emit cross-target AOT objects with identity records (#184)"
```

---

### Task 2: Selected SVE emitters and scalable-vector backend pass

**Files:**
- Modify: `include/LLK/Target/AArch64/Mapping/SVEMappingTarget.h`
- Create: `lib/Target/AArch64/Mapping/SVEBundleLowering.cpp`
- Create: `lib/Target/AArch64/Mapping/SVEBackendLowering.cpp`
- Modify: `lib/Target/AArch64/Mapping/SVEMappingTarget.cpp`
- Modify: `lib/Target/CMakeLists.txt`
- Create: `test/Mapping/arm_sve_backend.cpp`
- Modify: `test/Mapping/CMakeLists.txt`

**Interfaces:**
- Consumes: `mapping::DeclaredTargetEmitter`, `mapping::TargetEmitter::lower`, `mapping::TargetLoweringContext`, `micro::VectorOp`, the `micro.op`/`micro.mapping` attributes Micro→Linalg leaves on generated Linalg ops.
- Produces: `sve::createSVEEmitter(key)`, `sve::buildSVESelectedBackendPipeline(pm, f32Width)`, and an `arm-sve` target with `codegenRequirements = {aarch64, "", {sve}}`. Task 4 relies on the lowered artifact containing `vector<vscale x 4 x f32>`.

- [ ] **Step 1: Write the failing tests**

Create `test/Mapping/arm_sve_backend.cpp`:

```cpp
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include "arm_target_fixture.h"

#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

namespace {

constexpr const char *kVectorAdd =
    "test/Conversion/MicroMapping/Inputs/arm/vector_add_f32.mlir";
constexpr const char *kContraction =
    "test/Conversion/MicroMapping/Inputs/arm/contraction_f32.mlir";

TEST(ArmSVEBackend, OnlyArithmeticKeysLower) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorAdd);
  ASSERT_TRUE(static_cast<bool>(fixture)) << llvm::toString(fixture.takeError());
  EXPECT_TRUE(fixture->target->createEmitter("sve_vector_add")->hasLowering());
  EXPECT_TRUE(fixture->target->createEmitter("sve_vector_mul")->hasLowering());
  EXPECT_FALSE(fixture->target->createEmitter("sve_mma")->hasLowering());
  EXPECT_FALSE(fixture->target->createEmitter("sve_reduce")->hasLowering());
  auto requirements = fixture->target->codegenRequirements();
  ASSERT_TRUE(requirements.has_value());
  EXPECT_EQ(requirements->architecture, "aarch64");
  EXPECT_EQ(requirements->requiredFeatures,
            std::vector<std::string>{"sve"});
}

TEST(ArmSVEBackend, SelectedVectorAddVectorizesToScalableSVE) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorAdd);
  ASSERT_TRUE(static_cast<bool>(fixture)) << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_add_f32";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled =
      llk::compileMappedKernel(*source, *fixture->target, fixture->plan,
                               options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  EXPECT_GT(compiled->backendGroupsRealized, 0u);
  EXPECT_EQ(compiled->referenceGroupsLowered, 0u);
  std::string text;
  llvm::raw_string_ostream stream(text);
  compiled->module->print(stream);
  EXPECT_NE(text.find("vector<8x[4]xf32>"), std::string::npos) << text;
}

TEST(ArmSVEBackend, ContractionStillHasNoSelectedLowering) {
  auto fixture = arm_test::mapFixture("arm-sve", kContraction);
  ASSERT_TRUE(static_cast<bool>(fixture)) << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_contraction";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled =
      llk::compileMappedKernel(*source, *fixture->target, fixture->plan,
                               options);
  EXPECT_FALSE(static_cast<bool>(compiled));
}

} // namespace
```

MLIR prints a scalable dimension in brackets, so the 8×8 tile vectorizes to
`vector<8x[4]xf32>` (8 rows × vscale·4 f32 lanes). Add `arm_sve_backend.cpp` to
the `SOURCES` list in `test/Mapping/CMakeLists.txt`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `ninja -C build LLKMappingTests && ./build/bin/LLKMappingTests --gtest_filter='ArmSVEBackend.*'`
Expected: FAIL — `sve_vector_add` reports `hasLowering() == false` and `codegenRequirements()` is empty.

- [ ] **Step 3: Add the executable emitter and pipeline declarations**

In `include/LLK/Target/AArch64/Mapping/SVEMappingTarget.h`, add:

```cpp
/// An emitter that lowers the fixed-VL SVE vector add/multiply subset and
/// keeps every other key's explicit rejection.
std::unique_ptr<mapping::TargetEmitter>
createSVEEmitter(llvm::StringRef key);

/// Adds the SVE scalable-vector lowering after generic Micro to Linalg
/// conversion. `f32VectorWidth` is the selected 512-bit lane width (16).
void buildSVESelectedBackendPipeline(mlir::OpPassManager &pm,
                                     int64_t f32VectorWidth);
```

Include `mlir/Pass/PassManager.h`.

- [ ] **Step 4: Implement the bundle emitters**

Create `lib/Target/AArch64/Mapping/SVEBundleLowering.cpp`:

```cpp
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"
#include "LLK/Dialect/Micro/MicroOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace mlir::llk::target::sve {
namespace {

llvm::Error sveError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "sve_lowering_unsupported: " + message);
}

class SVEEmitter : public mapping::DeclaredTargetEmitter {
public:
  SVEEmitter(std::string key, std::vector<std::string> declaredKeys)
      : DeclaredTargetEmitter(std::move(key), std::move(declaredKeys)) {}

  bool hasLowering() const override {
    return key() == "sve_vector_add" || key() == "sve_vector_mul";
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
    if (coveredOps.empty())
      return sveError("no covered operations to lower");
    llvm::StringRef expected = key() == "sve_vector_add" ? "add" : "mul";
    for (mlir::Operation *op : coveredOps) {
      auto vector = mlir::dyn_cast<micro::VectorOp>(op);
      if (!vector || vector.getOp() != expected)
        return sveError("emitter '" + key() + "' cannot lower covered "
                        "operation '" + op->getName().getStringRef() + "'");
      auto result = mlir::dyn_cast<micro::TileType>(vector.getResult().getType());
      if (!result || !result.getElementType().isF32())
        return sveError("emitter '" + key() + "' supports f32 tiles only");
      op->setAttr("llk.sve.selected_handoff",
                  mlir::UnitAttr::get(rewriter.getContext()));
    }
    return llvm::Error::success();
  }
};

} // namespace

std::unique_ptr<mapping::TargetEmitter>
createSVEEmitter(llvm::StringRef key) {
  std::vector<std::string> declared;
  for (llvm::StringRef declaredKey : emitterKeys())
    declared.push_back(declaredKey.str());
  return std::make_unique<SVEEmitter>(key.str(), std::move(declared));
}

} // namespace mlir::llk::target::sve
```

- [ ] **Step 5: Implement the scalable-vector pass**

Create `lib/Target/AArch64/Mapping/SVEBackendLowering.cpp`:

```cpp
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Error.h"

namespace mlir::llk::target::sve {
namespace {

class SVEBackendLoweringPass
    : public PassWrapper<SVEBackendLoweringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SVEBackendLoweringPass)
  SVEBackendLoweringPass() = default;
  SVEBackendLoweringPass(const SVEBackendLoweringPass &other)
      : PassWrapper(other) {}
  explicit SVEBackendLoweringPass(int64_t f32Width) : f32Width_(f32Width) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> covered;
    module.walk([&](Operation *op) {
      if (op->hasAttr("llk.sve.selected_handoff"))
        covered.push_back(op);
    });
    if (covered.empty()) {
      module.emitError("SVE backend found no selected handoff operations");
      return signalPassFailure();
    }
    IRRewriter rewriter(&getContext());
    for (Operation *op : covered) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      auto name = op->getAttrOfType<StringAttr>("micro.op");
      if (!linalgOp || !name ||
          (name.getValue() != "add" && name.getValue() != "mul")) {
        op->emitError("SVE backend only lowers add/multiply vector groups");
        return signalPassFailure();
      }
      auto output = dyn_cast<ShapedType>(linalgOp->getResult(0).getType());
      if (!output || output.getRank() != 2 || !output.hasStaticShape() ||
          !output.getElementType().isF32()) {
        op->emitError("SVE backend needs a static rank-two f32 result");
        return signalPassFailure();
      }
      if (f32Width_ != 16) {
        op->emitError("SVE backend supports the modeled 16-lane f32 width");
        return signalPassFailure();
      }
      // One 512-bit vector is vscale*4 f32 lanes; the row is masked when the
      // tile is narrower than 16, which covers the 8x8 fixtures.
      SmallVector<int64_t> sizes(output.getShape().begin(),
                                 output.getShape().end());
      sizes.back() = 4;
      SmallVector<bool> scalable(sizes.size(), false);
      scalable.back() = true;
      if (failed(linalg::vectorize(rewriter, op, sizes, scalable))) {
        op->emitError("SVE scalable vectorization failed");
        return signalPassFailure();
      }
      ++realized_;
    }
    module->setAttr(
        "llk.backend_groups_realized",
        IntegerAttr::get(IntegerType::get(&getContext(), 64), realized_));
  }

private:
  int64_t f32Width_ = 16;
  unsigned realized_ = 0;
};

} // namespace

void buildSVESelectedBackendPipeline(mlir::OpPassManager &pm,
                                     int64_t f32VectorWidth) {
  pm.addPass(std::make_unique<SVEBackendLoweringPass>(f32VectorWidth));
}

} // namespace mlir::llk::target::sve
```

- [ ] **Step 6: Wrap the target with codegen requirements and the pipeline**

In `lib/Target/AArch64/Mapping/SVEMappingTarget.cpp`, add an `SVETarget` class that mirrors `AVX2Target` (see `lib/Target/X86/Mapping/AVX2MappingTarget.cpp`):

```cpp
class SVETarget : public mapping::MappingTarget {
public:
  explicit SVETarget(std::unique_ptr<mapping::MappingTarget> configuration)
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
  bool isKnownEmitter(llvm::StringRef key) const override {
    return configuration_->isKnownEmitter(key);
  }
  std::optional<mapping::TargetCodegenRequirements>
  codegenRequirements() const override {
    return mapping::TargetCodegenRequirements{"aarch64", "", {"sve"}};
  }
  llvm::Error
  buildSelectedBackendPipeline(mlir::OpPassManager &pm) const override {
    const machine::ComputeNode *engine = machine().findCompute("svp");
    if (!engine)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "arm-sve selected backend has no vector engine 'svp'");
    auto lanes = engine->lanes.find("f32");
    if (lanes == engine->lanes.end())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "arm-sve selected backend has no f32 lane width");
    buildSVESelectedBackendPipeline(pm, lanes->second);
    return llvm::Error::success();
  }
  std::unique_ptr<mapping::TargetEmitter>
  createEmitter(llvm::StringRef key) const override {
    if (!configuration_->isKnownEmitter(key))
      return nullptr;
    return createSVEEmitter(key);
  }
  std::unique_ptr<mapping::TargetEmitter> createEmitter() const override {
    return createEmitter(emitterKeys().front());
  }

private:
  std::unique_ptr<mapping::MappingTarget> configuration_;
};
```

and return `std::make_unique<SVETarget>(std::move(*configuration))` from `createMappingTarget`.

- [ ] **Step 7: Narrow the two static-only tests to the unimplemented subset**

`arm-sve` now legitimately has a codegen contract, so the blanket assertions
fail. Split them without deleting any negative control.

In `test/Mapping/arm_targets.cpp`, replace `ArmTargets.NoExecutableBackend`
with two tests:

```cpp
// SME remains a configuration-only package.
TEST(ArmTargets, SmeHasNoExecutableBackend) {
  auto target = loadRegistered(kConfigurations[1]);  // arm-sme
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  EXPECT_FALSE((*target)->codegenRequirements().has_value());
  mlir::OpPassManager pipeline;
  EXPECT_NE(llvm::toString((*target)->buildSelectedBackendPipeline(pipeline))
                .find("unsupported"),
            std::string::npos);
  for (const RuleDef &rule : (*target)->rules().all()) {
    auto emitter = (*target)->createEmitter(rule.emitter);
    ASSERT_NE(emitter, nullptr);
    EXPECT_FALSE(emitter->hasLowering());
  }
}

// SVE declares exactly the executable arithmetic subset; every other key keeps
// the explicit rejection.
TEST(ArmTargets, SveExecutableSubsetIsExplicit) {
  auto target = loadRegistered(kConfigurations[0]);  // arm-sve
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto requirements = (*target)->codegenRequirements();
  ASSERT_TRUE(requirements.has_value());
  EXPECT_EQ(requirements->architecture, "aarch64");
  mlir::OpPassManager pipeline;
  EXPECT_FALSE(static_cast<bool>(
      (*target)->buildSelectedBackendPipeline(pipeline)));
  for (const RuleDef &rule : (*target)->rules().all()) {
    bool executable = rule.emitter == "sve_vector_add" ||
                      rule.emitter == "sve_vector_mul";
    auto emitter = (*target)->createEmitter(rule.emitter);
    ASSERT_NE(emitter, nullptr);
    EXPECT_EQ(emitter->hasLowering(), executable);
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

In `test/Mapping/arm_placement.cpp`, rename
`SelectedTargetCompilationRejectsMissingIsa` to
`SelectedTargetCompilationRefusesUnimplementedSignatures` and assert per target:
`arm-sme` still fails with `ISA requirements`; `arm-sve` fails with
`realized no selected compute groups` (the contraction's `sve_mma` has no
lowering, so nothing selected is realized). Keep the same fixture loop and the
`ASSERT_FALSE` shape.

- [ ] **Step 8: Add the sources, build and run the tests**

Add `AArch64/Mapping/SVEBundleLowering.cpp` and `AArch64/Mapping/SVEBackendLowering.cpp` to `LLKTargetMapping` in `lib/Target/CMakeLists.txt`.

Run: `ninja -C build LLKMappingTests && ./build/bin/LLKMappingTests --gtest_filter='ArmSVEBackend.*:ArmTargets.*:ArmPlacement.*'`
Expected: all pass. `ArmTargets` now proves both the executable subset and the
retained refusals; `ArmPlacement` proves the contraction is still refused for
the right reason.

- [ ] **Step 9: Commit**

```bash
git add include/LLK/Target/AArch64/Mapping/SVEMappingTarget.h \
        lib/Target/AArch64/Mapping/SVEBundleLowering.cpp \
        lib/Target/AArch64/Mapping/SVEBackendLowering.cpp \
        lib/Target/AArch64/Mapping/SVEMappingTarget.cpp \
        lib/Target/CMakeLists.txt test/Mapping/arm_sve_backend.cpp \
        test/Mapping/arm_targets.cpp test/Mapping/arm_placement.cpp \
        test/Mapping/CMakeLists.txt
git commit -m "feat(target): lower the SVE f32 vector subset to scalable vectors (#184)"
```

---

### Task 3: Vector-length requirement and host check

**Files:**
- Modify: `include/LLK/Mapping/CodegenRequirements.h`
- Modify: `lib/Mapping/CodegenRequirements.cpp`
- Modify: `runtime/MappedExecutable.cpp`
- Modify: `lib/Target/AArch64/Mapping/SVEMappingTarget.cpp`
- Create: `test/Mapping/codegen_requirements.cpp`
- Modify: `test/Mapping/CMakeLists.txt`

**Interfaces:**
- Produces: `TargetCodegenRequirements::vectorLengthBits` (`std::optional<int64_t>`, default `std::nullopt`) and the four-argument `checkHostRequirements`. `arm-sve` requires 512 bits.

- [ ] **Step 1: Write the failing tests**

Create `test/Mapping/codegen_requirements.cpp`:

```cpp
#include "LLK/Mapping/CodegenRequirements.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include <gtest/gtest.h>

using mlir::llk::mapping::TargetCodegenRequirements;
using mlir::llk::mapping::checkHostRequirements;

TEST(CodegenRequirements, AcceptsMatchingVectorLength) {
  TargetCodegenRequirements requirements{"aarch64", "", {"sve"}, 512};
  llvm::StringMap<bool> features{{"sve", true}};
  EXPECT_FALSE(checkHostRequirements(requirements, "aarch64", features, 512));
}

TEST(CodegenRequirements, RejectsMissingOrDifferentVectorLength) {
  TargetCodegenRequirements requirements{"aarch64", "", {"sve"}, 512};
  llvm::StringMap<bool> features{{"sve", true}};
  EXPECT_TRUE(checkHostRequirements(requirements, "aarch64", features,
                                    std::nullopt));
  EXPECT_TRUE(checkHostRequirements(requirements, "aarch64", features, 256));
  features["sve"] = false;
  EXPECT_TRUE(checkHostRequirements(requirements, "aarch64", features, 512));
}
```

Add `codegen_requirements.cpp` to `test/Mapping/CMakeLists.txt`.

- [ ] **Step 2: Run it to verify it fails**

Run: `ninja -C build LLKMappingTests && ./build/bin/LLKMappingTests --gtest_filter='CodegenRequirements.*'`
Expected: compile error — `TargetCodegenRequirements` has no fourth field and `checkHostRequirements` takes three arguments.

- [ ] **Step 3: Add the requirement and the check**

In `include/LLK/Mapping/CodegenRequirements.h`:

```cpp
struct TargetCodegenRequirements {
  std::string architecture;
  std::string cpu;
  std::vector<std::string> requiredFeatures;
  /// Exact implemented vector length in bits, when the target's code assumes
  /// one. Nullopt means the target makes no vector-length claim.
  std::optional<int64_t> vectorLengthBits = std::nullopt;
};

llvm::Error
checkHostRequirements(const TargetCodegenRequirements &requirements,
                      llvm::StringRef hostArchitecture,
                      const llvm::StringMap<bool> &hostFeatures,
                      std::optional<int64_t> hostVectorLengthBits);
```

Keep the existing three-argument declaration as an inline overload forwarding `std::nullopt` so every current caller keeps compiling. In `lib/Mapping/CodegenRequirements.cpp`, after the architecture and feature checks:

```cpp
  if (requirements.vectorLengthBits) {
    if (!hostVectorLengthBits)
      return requirementError("host vector length is unknown but '" +
                              requirements.architecture +
                              "' requires " +
                              std::to_string(*requirements.vectorLengthBits) +
                              " bits");
    if (*hostVectorLengthBits != *requirements.vectorLengthBits)
      return requirementError(
          "host vector length is " + std::to_string(*hostVectorLengthBits) +
          " bits but the selected target requires " +
          std::to_string(*requirements.vectorLengthBits) + " bits");
  }
```

- [ ] **Step 4: Probe the host vector length and pass it to the check**

In `runtime/MappedExecutable.cpp`, add a helper (guard Linux/AArch64; other hosts return `std::nullopt`):

```cpp
std::optional<int64_t> detectHostVectorLengthBits() {
#if defined(__linux__) && defined(__aarch64__)
  unsigned long hwcap = getauxval(AT_HWCAP);
  if (!(hwcap & HWCAP_SVE))
    return std::nullopt;
  // PR_SVE_GET_VL = 51; the low 16 bits are VL/16.
  long value = prctl(51, 0, 0, 0, 0);
  if (value < 0)
    return std::nullopt;
  return static_cast<int64_t>((value & 0xffff) * 16);
#else
  return std::nullopt;
#endif
}
```

Include `<sys/auxv.h>` and `<sys/prctl.h>` under the same guard, and pass `detectHostVectorLengthBits()` as the fourth argument of `checkHostRequirements`.

- [ ] **Step 5: Require exactly 512 bits on `arm-sve`**

Change the SVE `codegenRequirements()` return to:

```cpp
    return mapping::TargetCodegenRequirements{"aarch64", "", {"sve"}, 512};
```

The existing `ArmTargets.SveExecutableSubsetIsExplicit` and
`ArmSVEBackend.OnlyArithmeticKeysLower` tests keep passing; the requirement is
now the exact modeled length rather than merely "SVE present".

- [ ] **Step 6: Build and run the focused tests**

Run: `ninja -C build LLKMappingTests && ./build/bin/LLKMappingTests --gtest_filter='CodegenRequirements.*:ArmSVEBackend.*:ArmTargets.*:ArmPlacement.*'`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add include/LLK/Mapping/CodegenRequirements.h lib/Mapping/CodegenRequirements.cpp \
        runtime/MappedExecutable.cpp lib/Target/AArch64/Mapping/SVEMappingTarget.cpp \
        test/Mapping/codegen_requirements.cpp test/Mapping/arm_targets.cpp \
        test/Mapping/arm_placement.cpp test/Mapping/CMakeLists.txt
git commit -m "feat(mapping): require the modeled 512-bit vector length for SVE (#184)"
```

---

### Task 4: Freestanding harness, QEMU runner and CI evidence job

**Files:**
- Create: `test/Execution/arm_sve_evidence.py`
- Modify: `test/FileCheck.cmake`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `--emit=object` from Task 1, the selected SVE lowering from Task 2, the ABI descriptor convention (`_mlir_ciface_<kernel>`), and the identity JSON.
- Produces: CTest `ArmSVEExecutionEvidence`, gated by `-DLLK_BUILD_ARM_EVIDENCE=ON`; the script fails (never skips) when qemu or the cross linker is missing.

- [ ] **Step 1: Write the failing evidence script**

Create `test/Execution/arm_sve_evidence.py`. It takes
`<llk-compile> <source-dir> <work-dir> <qemu-aarch64> <aarch64-gcc>` and, for each
case, generates a harness, compiles it with the cross toolchain, runs it under
qemu and validates the identity record.

```python
#!/usr/bin/env python3
"""Run the selected SVE object under qemu and check its numbers (E10)."""

import json
import os
import struct
import subprocess
import sys

CASES = [
    ("vector_add_f32", "arm_vector_add_f32", "add", 8, 8),
    ("vector_mul_f32", "arm_vector_mul_f32", "mul", 8, 8),
]

HARNESS = r"""
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

extern void WRAPPER(MemRef2D *out, MemRef2D *a, MemRef2D *b);

static unsigned char heap[1 << 20] __attribute__((aligned(64)));
static size_t used;

void *malloc(size_t n) {
  size_t start = (used + 63) & ~(size_t)63;
  if (start + n > sizeof(heap))
    for (;;) { }
  used = start + n;
  return heap + start;
}
void *aligned_alloc(size_t alignment, size_t n) {
  size_t start = (used + alignment - 1) & ~(alignment - 1);
  if (start + n > sizeof(heap))
    for (;;) { }
  used = start + n;
  return heap + start;
}
void free(void *p) { (void)p; }
void *memset(void *dst, int value, size_t n) {
  unsigned char *bytes = dst;
  for (size_t i = 0; i < n; ++i)
    bytes[i] = (unsigned char)value;
  return dst;
}
void *memcpy(void *dst, const void *src, size_t n) {
  unsigned char *out = dst;
  const unsigned char *in = src;
  for (size_t i = 0; i < n; ++i)
    out[i] = in[i];
  return dst;
}
static long syscall3(long, long, long, long);
void memrefCopy(int64_t, void *, void *) {
  syscall3(93, 75, 0, 0);  // this subset must not reach a generic copy
  for (;;) { }
}

static const uint32_t LHS[ROWS * COLS] = { LHS_DATA };
static const uint32_t RHS[ROWS * COLS] = { RHS_DATA };
static const uint32_t EXPECTED[ROWS * COLS] = { EXPECTED_DATA };

static long syscall3(long n, long a, long b, long c) {
  register long x8 __asm__("x8") = n;
  register long x0 __asm__("x0") = a;
  register long x1 __asm__("x1") = b;
  register long x2 __asm__("x2") = c;
  __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2) : "memory");
  return x0;
}

static void write_all(const char *text, long length) {
  syscall3(64, 1, (long)text, length);
}
#define WRITE_LITERAL(text) write_all((text), (long)(sizeof(text) - 1))

static float bits(uint32_t value) {
  union { uint32_t u; float f; } cast;
  cast.u = value;
  return cast.f;
}

static uint32_t as_bits(float value) {
  union { uint32_t u; float f; } cast;
  cast.f = value;
  return cast.u;
}

static int is_nan32(uint32_t value) {
  return (value & 0x7f800000u) == 0x7f800000u && (value & 0x7fffffu);
}

void _start(void) {
  unsigned long hwcap = 0;
  for (int i = 0; i < 2; ++i) { }
  /* getauxval is unavailable without libc; qemu sets HWCAP in the auxv the
     kernel passes on the stack. Parsing it here keeps the harness libc-free. */
  unsigned long *stack;
  __asm__("mov %0, sp" : "=r"(stack));
  unsigned long argc = stack[0];
  unsigned long *aux = stack + 1 + argc + 1;  // envp[0]
  while (*aux)
    ++aux;                                    // skip envp
  ++aux;                                      // skip its NULL
  while (*aux) {
    if (aux[0] == 16) { hwcap = aux[1]; break; }
    aux += 2;
  }
  if (!(hwcap & (1UL << 22))) {
    WRITE_LITERAL("unsupported: no sve hwcap\n");
    syscall3(93, 70, 0, 0);
  }
  /* PR_SVE_GET_VL = 51; VL = (low 16 bits) * 16 bits. */
  long vl = syscall3(167, 51, 0, 0);
  if (vl < 0 || (vl & 0xffff) * 16 != 512) {
    WRITE_LITERAL("unsupported: vector length is not 512\n");
    syscall3(93, 71, 0, 0);
  }

  static float lhs[ROWS * COLS] __attribute__((aligned(64)));
  static float rhs[ROWS * COLS] __attribute__((aligned(64)));
  static unsigned char out[ROWS * COLS * 4 + 128];
  for (int i = 0; i < ROWS * COLS; ++i) {
    lhs[i] = bits(LHS[i]);
    rhs[i] = bits(RHS[i]);
  }
  for (unsigned i = 0; i < sizeof(out); ++i)
    out[i] = 0x5a;
  MemRef2D a = {lhs, lhs, 0, ROWS, COLS, COLS, 1};
  MemRef2D b = {rhs, rhs, 0, ROWS, COLS, COLS, 1};
  MemRef2D result = {0, 0, 0, 0, 0, 0, 0};
  WRAPPER(&result, &a, &b);
  if (!result.aligned) {
    WRITE_LITERAL("failed: no result descriptor\n");
    syscall3(93, 72, 0, 0);
  }
  float *values = result.aligned + result.offset;
  for (int i = 0; i < ROWS * COLS; ++i) {
    uint32_t got = as_bits(values[i]);
    if (is_nan32(EXPECTED[i]) ? !is_nan32(got) : got != EXPECTED[i]) {
      WRITE_LITERAL("failed: numeric mismatch\n");
      syscall3(93, 73, 0, 0);
    }
  }
  for (int i = 0; i < 128; ++i) {
    if (out[i] != 0x5a) {
      WRITE_LITERAL("failed: canary written\n");
      syscall3(93, 74, 0, 0);
    }
  }
  WRITE_LITERAL("pass\n");
  syscall3(93, 0, 0, 0);
}
"""


def float_bits(values):
    return ", ".join("0x%08xu" % struct.unpack("<I", struct.pack("<f", v))[0]
                     for v in values)


def expected_data(name, rows, cols):
    lhs, rhs = [], []
    for i in range(rows * cols):
        lhs.append(struct.unpack("<f", struct.pack("<I", 0x3f800000 + i * 7))[0])
        rhs.append(struct.unpack("<f", struct.pack("<I", 0x40000000 + i * 3))[0])
    if name == "add":
        values = [struct.unpack("<f", struct.pack("<f", a + b))[0]
                  for a, b in zip(lhs, rhs)]
    else:
        values = [struct.unpack("<f", struct.pack("<f", a * b))[0]
                  for a, b in zip(lhs, rhs)]
    return lhs, rhs, values


def main():
    llk_compile, source_dir, work_dir, qemu, cross_gcc = sys.argv[1:]
    os.makedirs(work_dir, exist_ok=True)
    failures = []
    for fixture, symbol, name, rows, cols in CASES:
        case_dir = os.path.join(work_dir, fixture)
        os.makedirs(case_dir, exist_ok=True)
        source = os.path.join(source_dir, "test/Conversion/MicroMapping",
                              "Inputs/arm", fixture + ".mlir")
        obj = os.path.join(case_dir, "kernel.o")
        identity = os.path.join(case_dir, "identity.json")
        command = [
            llk_compile, "--mapping-target=arm-sve",
            "--mapping-root=" + source_dir,
            "--mapping-backend=selected-target",
            "--mapping-stop=lowered", "--emit=object",
            "--target-triple=aarch64-unknown-linux-gnu",
            "--target-features=+sve", "--output=" + obj,
            "--emit-identity=" + identity, source,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            failures.append("%s: object emission failed\n%s" %
                            (fixture, result.stderr))
            continue
        record = json.load(open(identity))
        if record["referenceGroupsLowered"] != 0 or \
                record["backendGroupsRealized"] == 0:
            failures.append("%s: selected lowering fell back\n%s" %
                            (fixture, record))
            continue
        lhs, rhs, expected = expected_data(name, rows, cols)
        harness = HARNESS.replace("WRAPPER", record["wrapperSymbol"])
        harness = harness.replace("ROWS", str(rows)).replace("COLS", str(cols))
        harness = harness.replace("LHS_DATA", float_bits(lhs))
        harness = harness.replace("RHS_DATA", float_bits(rhs))
        harness = harness.replace("EXPECTED_DATA", float_bits(expected))
        harness_path = os.path.join(case_dir, "harness.c")
        open(harness_path, "w").write(harness)
        binary = os.path.join(case_dir, "harness")
        subprocess.run([cross_gcc, "-static", "-O2", "-nostdlib",
                        "-ffreestanding", "-fno-builtin",
                        "-Wl,-e,_start", harness_path, obj, "-o", binary],
                       check=True)
        run = subprocess.run([qemu, "-cpu", "max,sve-max-vq=4", binary],
                             capture_output=True, text=True)
        if run.returncode != 0:
            failures.append("%s: qemu exit %d\n%s%s" %
                            (fixture, run.returncode, run.stdout, run.stderr))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("arm sve evidence: %d cases passed" % len(CASES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Register the test and run it to verify it fails**

Add to `test/FileCheck.cmake`:

```cmake
  option(LLK_BUILD_ARM_EVIDENCE
      "Build and run the cross-object Arm SVE evidence under qemu" OFF)
  if(LLK_BUILD_ARM_EVIDENCE)
    find_program(LLK_QEMU_AARCH64 NAMES qemu-aarch64 qemu-aarch64-static REQUIRED)
    find_program(LLK_AARCH64_GCC NAMES aarch64-linux-gnu-gcc REQUIRED)
    add_test(NAME ArmSVEExecutionEvidence
        COMMAND ${LLK_TEST_PYTHON} ${CMAKE_SOURCE_DIR}/test/Execution/arm_sve_evidence.py
            $<TARGET_FILE:llk-compile> ${CMAKE_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/ArmSVEExecutionEvidence
            ${LLK_QEMU_AARCH64} ${LLK_AARCH64_GCC}
    )
  endif()
```

Run locally with the option on only when qemu/cross gcc exist; otherwise run
the cross-object half by hand:

`cmake -S . -B build -DLLK_BUILD_ARM_EVIDENCE=ON` (expect configure error naming
the missing program on macOS). This is the expected RED state for a host with no
runner; the CI job below supplies both tools.

- [ ] **Step 3: Wire the evidence job into CI**

In `.github/workflows/ci.yml`:

1. Change the LLVM configure to
   `-DLLVM_TARGETS_TO_BUILD=X86;AArch64` and bump the cache key to
   `llvm-build-${LLVM_VERSION}-shared-v4`.
2. Extend "Install build dependencies" with
   `qemu-user gcc-aarch64-linux-gnu`.
3. Pass `-DLLK_BUILD_ARM_EVIDENCE=ON` to the project configure.
4. Apply the same `X86;AArch64` target change and cache-key bump to
   `.github/workflows/coverage.yml`, which also builds LLVM with `Native` and
   runs the full CTest suite. `AOTObjectEmission` needs the AArch64 backend at
   runtime, so the coverage job fails without it.
5. Add a step after the build:

```yaml
      - name: Run Arm SVE emulated evidence
        run: |
          cd build
          set -o pipefail
          ctest -R '^ArmSVEExecutionEvidence$' --output-on-failure 2>&1 | tee arm-sve-evidence.log
```

CI must fail, not skip, when qemu or the cross toolchain is missing; the CMake
`REQUIRED` lookups enforce that.

- [ ] **Step 4: Run the focused local checks and commit**

Run: `ninja -C build llk-compile LLKMappingTests`
Run: `ctest --test-dir build -R '^(ArmSVEExecutionEvidence|AOTObjectEmission)$' --output-on-failure`
Expected: `AOTObjectEmission` passes; `ArmSVEExecutionEvidence` is absent on a
host without qemu (option OFF) or passes in CI.

```bash
git add test/Execution/arm_sve_evidence.py test/FileCheck.cmake \
        .github/workflows/ci.yml
git commit -m "test(execution): run the SVE object under qemu in CI (#184)"
```

---

### Task 5: Full-width fixtures, tails, canaries and non-finite values

**Files:**
- Create: `test/Conversion/MicroMapping/Inputs/arm/vector_add_f32_full.mlir`
- Create: `test/Conversion/MicroMapping/Inputs/arm/vector_mul_f32_full.mlir`
- Modify: `test/Execution/arm_sve_evidence.py`
- Modify: `test/Mapping/arm_sve_backend.cpp`

**Interfaces:**
- Produces: 16×16 fixtures whose rows fill one VL=512 f32 vector, and expected
  tables covering signed zero, infinity and NaN in both the 8×8 (masked tail)
  and 16×16 (full-width) cases.

- [ ] **Step 1: Add the full-width fixtures**

Copy the 8×8 fixtures and change every `8x8` / `array<i64: 8, 8>` to `16x16` /
`array<i64: 16, 16>`, keeping the kernel comment that this is a bounded
conformance program. Names: `@arm_vector_add_f32_full`,
`@arm_vector_mul_f32_full`.

- [ ] **Step 2: Extend the evidence cases and expectations**

In `test/Execution/arm_sve_evidence.py`, add:

```python
CASES = [
    ("vector_add_f32", "arm_vector_add_f32", "add", 8, 8),
    ("vector_mul_f32", "arm_vector_mul_f32", "mul", 8, 8),
    ("vector_add_f32_full", "arm_vector_add_f32_full", "add", 16, 16),
    ("vector_mul_f32_full", "arm_vector_mul_f32_full", "mul", 16, 16),
]
```

and extend `expected_data` so the first four elements are fixed special values
in both operands:

```python
SPECIAL = [0x00000000, 0x80000000, 0x7f800000, 0x7fc00000]  # +0, -0, +inf, NaN
```

with the rule that any expected NaN is compared by classification (already in
the harness) and every other element by exact bits. Assert the identity record's
`planId` differs between the 8×8 and 16×16 runs and that
`backendGroupsRealized > 0` in all four.

- [ ] **Step 3: Add a backend test for the full-width shape**

In `test/Mapping/arm_sve_backend.cpp` add a test that maps
`vector_mul_f32_full.mlir`, compiles with `SelectedTarget`/`Lowered`, and
asserts `vector<16x[4]xf32>` appears and `referenceGroupsLowered == 0`.

- [ ] **Step 4: Run the tests and commit**

Run: `ninja -C build LLKMappingTests && ./build/bin/LLKMappingTests --gtest_filter='ArmSVEBackend.*'`
Run in CI: `ctest -R '^ArmSVEExecutionEvidence$' --output-on-failure`
Expected: all cases pass; the tail case masks the 8 columns and the full-width
case has no mask.

```bash
git add test/Conversion/MicroMapping/Inputs/arm/vector_*_f32_full.mlir \
        test/Execution/arm_sve_evidence.py test/Mapping/arm_sve_backend.cpp
git commit -m "test(mapping): cover full-width SVE vectors and masked tails (#184)"
```

---

### Task 6: Negative controls, receipts and documentation corrections

**Files:**
- Modify: `test/Mapping/arm_sve_backend.cpp`
- Modify: `test/Execution/arm_sve_evidence.py`
- Modify: `docs/reviews/arm-mapping-targets-acceptance.md`
- Modify: `docs/manual/source/tools/compiler.rst`
- Modify: `docs/manual/source/features.rst`
- Create: `docs/reviews/2026-10-24-arm-sve-executable-acceptance.md` (only after
  CI evidence exists; fill it from the actual artifacts in this task)

**Interfaces:**
- Consumes: everything above.
- Produces: the narrow, qualified static-support wording and the executable
  acceptance record; `#175` still open.

- [ ] **Step 1: Add the failing negative-control tests**

In `test/Mapping/arm_sve_backend.cpp` add:

```cpp
TEST(ArmSVEBackend, UnsupportedDtypeAndOperationAreRefused) {
  // Compile the bf16 add fixture with the selected backend.
  auto fixture = arm_test::mapFixture("arm-sve",
      "test/Conversion/MicroMapping/Inputs/arm/vector_add_bf16.mlir");
  ASSERT_TRUE(static_cast<bool>(fixture)) << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_add_bf16";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  EXPECT_FALSE(static_cast<bool>(
      llk::compileMappedKernel(*source, *fixture->target, fixture->plan,
                               options)));
}
```

The diagnostic must be one of the emitter/backend errors, not a reference
fallback. Add the same refusal assertion for `vector_silu_f32.mlir`.

In `test/Execution/arm_sve_evidence.py`, after the passing cases, run each
object once under `qemu -cpu max,sve=off` and once under
`qemu -cpu max,sve-max-vq=8`, and require the harness's distinct exit codes 70
(no SVE) and 71 (wrong VL). A zero exit in either run is a failure.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `ninja -C build LLKMappingTests && ./build/bin/LLKMappingTests --gtest_filter='ArmSVEBackend.*'`
Expected: the bf16/SiLU cases currently fail for an unexpected reason if the
emitter still accepts them, and the evidence script has no negative runs yet.
After adding the expectations, both must fail on the current code and pass once
the emitter's typed guard and the harness exit codes are correct.

- [ ] **Step 3: Update the static claims to the qualified wording**

- `docs/reviews/arm-mapping-targets-acceptance.md`: in "Static-only boundary"
  and "Claims and limits", replace the unqualified "selected-target compilation
  is rejected" for `arm-sve` with "rejected for signatures without an
  implemented emitter; the f32 add/multiply subset is executable as of E10".
  Keep every historical and AVX2 claim unchanged.
- `docs/manual/source/tools/compiler.rst`: same qualification in the
  registered-target paragraph.
- `docs/manual/source/features.rst`: in the Arm row, change the limit to say the
  f32 add/multiply subset has an emulated AOT acceptance path, while SME,
  execution of other signatures, native SVE and timing remain unimplemented.

- [ ] **Step 4: Write the E10 acceptance ledger from actual CI artifacts**

Run the CI evidence job on the implementation head and record in
`docs/reviews/2026-10-24-arm-sve-executable-acceptance.md`: source head, plan
ids, ABI hashes, compiler/LLVM/triple/features/VL, qemu version and CPU model,
harness hash, the four case results, the two negative qemu runs, and the
selected/reference group counts. Do not claim native execution or timing. If CI
evidence is not yet available, leave the task open — the ledger is not written
from predictions.

- [ ] **Step 5: Run the full verification and commit**

Run: `ninja -C build check-llk`
Run: the documented Sphinx build with `-n -W --keep-going`
Run: `git diff --check`
Expected: the full suite matches the E8 baseline plus the new tests; the manual
builds; whitespace is clean.

```bash
git add test/Mapping/arm_sve_backend.cpp test/Execution/arm_sve_evidence.py \
        docs/reviews/arm-mapping-targets-acceptance.md \
        docs/reviews/2026-10-24-arm-sve-executable-acceptance.md \
        docs/manual/source/tools/compiler.rst docs/manual/source/features.rst
git commit -m "docs: record the executable SVE subset and its negative controls (#184)"
```

---

## Self-review

**Spec coverage**

| Spec section | Plan tasks |
|---|---|
| §1 scope, exclusions | Tasks 2, 6 |
| §2 audited behavior | Tasks 1–3 |
| §3 choice and alternatives | Spec only (decision record) |
| §4 VL/VW contract and host checks | Task 3 |
| §5 selected lowering contract | Task 2 |
| §6 AOT object, harness, runner, CI | Tasks 1, 4 |
| §7 evidence tiers and identities | Tasks 1, 4, 6 |
| §8 numerical contract and negatives | Tasks 4, 5, 6 |
| §9 fixtures and expected evidence | Tasks 4, 5 |
| §10 implementation sequence | Tasks 1–6 |
| §11 follow-on issues and limits | Task 6, plus the issue-creation step below |

**Placeholder scan:** no TBD/TODO steps; every test and implementation step
names a file, a command and its expected result. Scalable vector types are
asserted in MLIR's printed spelling (`vector<8x[4]xf32>`,
`vector<16x[4]xf32>`), not left to the implementer to guess.

**Type consistency:** `TargetCodegenRequirements` gains
`vectorLengthBits` with a default initializer; every existing three-field
initializer remains valid. `emitMappedObject` returns the wrapper symbol used by
the identity record and the harness. `createSVEEmitter` and
`buildSVESelectedBackendPipeline` are the only new target entry points.
`MappedObjectIdentity` field names match the identity JSON keys asserted in
Tasks 1 and 4.

**Issue creation (after approval, before execution):** open E10 ("Implement the
fixed-VL SVE executable subset", depends on #184 approval, acceptance = Tasks
1–6 evidence) and E11 ("Design and implement the SME stateful backend", blocked
by E10) under #175; keep #175 open.
