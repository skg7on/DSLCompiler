# ARM SVE and SME Mapping Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `arm-sve` and `arm-sme` as static-conformance mapping targets — machine profiles, LLKMap policy, thin target packages and registration — with contraction bound to the vector engine on SVE and to a ZA matrix engine on SME.

**Architecture:** Each target is pure data plus a ~40-line loader. A machine profile (`machines/arm-*-v2.yaml`) declares topology, lanes and memory spaces; LLKMap files (`mapping/arm-*/`) declare layouts whose vector width is read back from the machine and rules that name bundles and emit keys; a package (`lib/Target/AArch64/Mapping/`) calls `mapping::loadMappingTarget` and returns the resulting `FileMappingTarget` unchanged. Nothing in `lib/Mapping`, `lib/Machine`, the `micro` dialect or the search changes.

**Tech Stack:** C++20, MLIR/LLVM 22–24, CMake + Ninja, GoogleTest, the project's own `LLKMap` declarative grammar and `MachineModel` v2 YAML schema.

**Spec:** `docs/superpowers/specs/2026-10-17-arm-sve-sme-mapping-targets-design.md`

## Global Constraints

- **Worktree only.** All work happens in the isolated worktree; the repository root is read-only.
- **Build dir:** `build/` at the worktree root, configured with `-DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build`. Configure once; `ninja` thereafter.
- **Test invocation:** `ninja check-llk` runs the whole registered suite. For a single case use the built binary, e.g. `./build/bin/LLKMappingTests --gtest_filter=ArmSveTarget.*`.
- **License header:** every new `.h`/`.cpp` starts with the project banner, verbatim:
  ```
  //===- <file> - <description> ----------*- C++ -*-===//
  //
  // Part of the LLK Compiler Project, under the Apache License v2.0 with
  // LLVM Exceptions.
  // See https://llvm.org/LICENSE.txt for license information.
  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  //
  //===----------------------------------------------------------------------===//
  ```
  Copy the banner from `lib/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.cpp`. Do not clang-format it (commit #169 stopped the formatter wrapping banners).
- **Deprecated API is a hard error** (`-Werror=deprecated-declarations`): use `Op::create(builder, loc, ...)`, `mlir::cast<T>(x)`, `isa<T>(x)`. No `.create<T>()`, `.cast<T>()`, `.isa<T>()`.
- **No target-name branching in generic code.** `lib/Mapping` and `lib/Machine` must not learn the words `sve`, `sme` or `arm`. Everything target-specific stays in the new files.
- **No dialect or schema change.** Do not touch `include/LLK/Dialect/Micro/`, `MicroEnums.h`, `MicroDialect.td`, or the MachineModel loader/schema.
- **Emitter keys are declared, not implemented.** Neither package subclasses `MappingTarget`; both must leave `codegenRequirements()` returning `nullopt` and lowering rejecting with `target_lowering_unsupported`.
- **Comment density:** match the surrounding files. The shipped profiles and policy files carry substantive header comments explaining *why*; new files are expected to do the same.
- **Commit per task**, with the trailer `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

---

### Task 1: `arm-sve` machine profile

**Files:**
- Create: `machines/arm-sve-v2.yaml`
- Modify: `test/Machine/machine_model_loader.cpp` (add one test near the end; add `#include "llvm/Support/MemoryBuffer.h"` to the include block at the top)

**Interfaces:**
- Consumes: nothing.
- Produces: a profile at `machines/arm-sve-v2.yaml` whose `target` is `arm-sve`, with executor kinds `cluster` (refines `group`), `worker`, `sve_engine` (refines `vector`); a compute node with **id `svp`**, kind `vector_engine`, `lanes: {f32: 16, bf16: 32, i8: 64}`; memory kinds `dram`, `l2`, `sram`, `rf`; and **no** memory of kind `acc`. Tasks 2, 5 and 6 depend on every one of these names.

- [ ] **Step 1: Write the failing test**

Append to `test/Machine/machine_model_loader.cpp`:

```cpp
TEST(MachineModelLoader, LoadsTheShippedArmSveProfile) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(std::string(LLK_MACHINE_DIR) +
                                  "/arm-sve-v2.yaml");
  ASSERT_TRUE(static_cast<bool>(buffer)) << "arm-sve profile must ship";
  llvm::Expected<MachineModel> model =
      parseMachineModel(buffer.get()->getBuffer(), "arm-sve-v2.yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());

  EXPECT_EQ(model->target, "arm-sve");

  // The hierarchy a rule's `require executor kind worker` resolves against.
  EXPECT_NE(model->findExecutor("worker.0"), nullptr);
  EXPECT_TRUE(model->ownerMatches("worker", "worker.0"));
  EXPECT_TRUE(model->ownerMatches("vector", "sve_eng.0"));
  EXPECT_EQ(model->ownerClass("cluster"), std::optional<std::string>("group"));
  EXPECT_EQ(model->ownerClass("sve_engine"),
            std::optional<std::string>("vector"));

  // A modeled 512-bit SVE core: a vector engine, and no accumulator space.
  const ComputeNode *svp = model->findCompute("svp");
  ASSERT_NE(svp, nullptr);
  EXPECT_EQ(svp->kind, "vector_engine");
  EXPECT_EQ(svp->lanes.at("f32"), 16u);
  EXPECT_EQ(svp->lanes.at("bf16"), 32u);
  EXPECT_EQ(svp->lanes.at("i8"), 64u);
  EXPECT_EQ(model->findMemoryOfKind("acc"), nullptr);
}
```

Add to the includes at the top of that file:

```cpp
#include "llvm/Support/MemoryBuffer.h"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
ninja LLKMachineTests && ./build/bin/LLKMachineTests --gtest_filter=MachineModelLoader.LoadsTheShippedArmSveProfile
```

Expected: FAIL — the assertion `arm-sve profile must ship` fires because the file does not exist.

- [ ] **Step 3: Write the profile**

Create `machines/arm-sve-v2.yaml`:

```yaml
# Machine model (v2): ARMv9-A core with SVE2, at a pinned modeled vector length.
#
# SVE's defining property is a vector length the implementation chooses at
# runtime; this profile pins one so the model carries concrete numbers. Nothing
# downstream hard-codes it: `sve.blocked_2d` requires
# `VW == machine.compute("vector_engine").lanes(element_type)`, so a profile
# with a different length is a new file rather than a schema change. Per the
# Arm Architecture Reference Manual, SVL is a multiple of 128 within
# [128, 2048]; 512 is also the longest length SME permits, which keeps this
# profile and arm-sme-v2.yaml describing the same core.
#
# The numbers are calibration seeds for a class of ARMv9-A cores, not
# measurements of any part.

schema: llk.machine.v2
target: arm-sve
description: >-
  ARMv9-A core with SVE2 at a modeled 512-bit vector length, behind a private
  L1, a shared L2 and DRAM, with one DMA engine.
clock_hz: 3000000000
worker_threads: 8
sync:
  barrier_cycles: 100
  wait_cycles: 0

executors:
  # `worker` is already the abstract spelling, so it refines nothing; the
  # rules that ask for `kind worker` resolve against it directly. `cluster`
  # refines `group` and `sve_engine` refines `vector`. The kernel writes these
  # labels as owner *symbols*; this table is what turns a symbol into a class.
  - id: cluster.0
    kind: cluster
    refines: [group]
    concurrency: 1
  - id: worker.0
    kind: worker
    parent: cluster.0
    concurrency: 8
  - id: sve_eng.0
    kind: sve_engine
    refines: [vector]
    parent: worker.0
    concurrency: 8

memories:
  - id: dram.0
    kind: dram
    visible_from: cluster.0
    capacity_bytes: 68719476736
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 220
  - id: l2.0
    kind: l2
    visible_from: cluster.0
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major, vectorized]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
  - id: sram.0
    kind: sram
    visible_from: worker.0
    capacity_bytes: 65536
    alignment_bytes: 64
    supported_layouts: [row_major, vectorized, blocked]
    banks: 1
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
  - id: rf.0
    kind: rf
    visible_from: worker.0
    # 32 SVE Z registers of SVL bits each, at SVL=512: 32 x 64 B = 2048 B.
    capacity_bytes: 2048
    alignment_bytes: 64
    supported_layouts: [row_major, vectorized]
    banks: 32
    bandwidth_bytes_per_cycle: 256
    latency_cycles: 1

compute:
  # The vector engine is attached to the worker scope a placement picks; the
  # `sve_eng.0` executor above exists so an abstract owner of vector kind, and
  # an owner path through it, still resolve.
  - id: svp
    kind: vector_engine
    refines: [vector]
    attached_to: worker.0
    element_types: [f32, bf16, i8]
    supported_layouts: [row_major, vectorized]
    # One instruction of the modeled length: 512 / 32, / 16 and / 8 bits.
    shapes: [[16], [32], [64]]
    lanes: {f32: 16, bf16: 32, i8: 64}
    issue_cycles: 1
    latency_cycles: 4
    concurrency: 8

transfer_engines:
  - id: dma.0
    kind: dma
    refines: [transfer]
    attached_to: worker.0
    count: 1
    max_outstanding: 8
    setup_cycles: 0

links:
  - id: dram_to_l2.0
    source: dram.0
    destination: l2.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: l2_to_sram.0
    source: l2.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: dram_to_sram.0
    source: dram.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: sram_to_rf.0
    source: sram.0
    destination: rf.0
    bandwidth_bytes_per_cycle: 256
    latency_cycles: 4
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: rf_to_sram.0
    source: rf.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: sram_to_dram.0
    source: sram.0
    destination: dram.0
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.0]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
ninja LLKMachineTests && ./build/bin/LLKMachineTests --gtest_filter=MachineModelLoader.LoadsTheShippedArmSveProfile
```

Expected: PASS.

- [ ] **Step 5: Run the whole machine suite for regressions**

```bash
./build/bin/LLKMachineTests
```

Expected: all tests pass. If `LoadsTheShippedArmSveProfile` fails on a field name, read `include/LLK/Machine/MachineModel.h` for the real accessor and fix the test — do not change the profile to suit the test.

- [ ] **Step 6: Commit**

```bash
git add machines/arm-sve-v2.yaml test/Machine/machine_model_loader.cpp
git commit -m "feat(machine): add the arm-sve v2 profile

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: `arm-sve` mapping target

**Files:**
- Create: `mapping/arm-sve/layouts.llkmap`
- Create: `mapping/arm-sve/rules.llkmap`
- Create: `include/LLK/Target/AArch64/Mapping/SVEMappingTarget.h`
- Create: `lib/Target/AArch64/Mapping/SVEMappingTarget.cpp`
- Modify: `lib/Target/CMakeLists.txt` (add one source line to `LLKTargetMapping`)
- Modify: `lib/Target/MappingTargets.cpp` (one include, one registration)
- Create: `test/Mapping/arm_sve_target.cpp`
- Modify: `test/Mapping/CMakeLists.txt` (add `arm_sve_target.cpp` to `SOURCES`)

**Interfaces:**
- Consumes: `machines/arm-sve-v2.yaml` from Task 1 — it must declare executor kind `worker`, compute id `svp` of kind `vector_engine` with 16 f32 lanes, and memory kinds `dram`/`sram`.
- Produces: namespace `mlir::llk::target::sve` with `llvm::ArrayRef<llvm::StringRef> emitterKeys()` returning the ten keys listed in Step 3, and `llvm::Expected<std::unique_ptr<mapping::MappingTarget>> createMappingTarget(llvm::StringRef configurationRoot, llvm::StringRef machinePath = {})`. Registered under the name `arm-sve`. Tasks 5 and 6 consume the emitted keys `sve_mma` and the rule ids `sve.vector_add` / `sve.mma_f32`.

- [ ] **Step 1: Write the failing test**

Create `test/Mapping/arm_sve_target.cpp`. Use the banner from the Global Constraints, with `- the ARM SVE mapping target ` as the description.

```cpp
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

using namespace mlir::llk::mapping;

/// Aliased because the tests also have locals named `target`.
namespace sve_mapping = mlir::llk::target::sve;

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

mlir::DictionaryAttr vectorAttributes(mlir::MLIRContext &context,
                                      llvm::StringRef op) {
  return mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                                      mlir::StringAttr::get(&context, op))});
}

WorkloadGraph vectorGraph(mlir::MLIRContext &context) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in", /*external=*/true});
  WorkloadValueId output =
      graph.addValue(WorkloadValue{0, mlir::Type(), "out", /*external=*/false});
  WorkloadNode node;
  node.opName = "micro.vector";
  node.attributes = vectorAttributes(context, "add");
  node.inputs.push_back(WorkloadPort{input, mlir::Type(), std::nullopt});
  node.outputs.push_back(WorkloadPort{output, mlir::Type(), std::nullopt});
  graph.addNode(std::move(node));
  graph.finalize();
  return graph;
}

llvm::Expected<std::unique_ptr<MappingTarget>> loadTarget() {
  return sve_mapping::createMappingTarget(LLK_SOURCE_DIR);
}

/// A bundle naming `emitterKey` with one well-typed integer parameter.
TargetBundle makeBundle(llvm::StringRef emitterKey, mlir::MLIRContext &context) {
  TargetBundle bundle;
  bundle.name = "sve.vector.add.f32";
  bundle.emitterKey = emitterKey.str();
  bundle.parameters = mlir::DictionaryAttr::get(
      &context,
      {mlir::NamedAttribute(
          mlir::StringAttr::get(&context, "tile"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 16))});
  return bundle;
}

} // namespace

TEST(ArmSveTarget, LoadsAndVerifiesItsConfiguration) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_EQ((*target)->name(), "arm-sve");
  EXPECT_NE((*target)->rules().find("sve.vector_add"), nullptr);
  EXPECT_NE((*target)->layouts().find("sve.row_major"), nullptr);
  EXPECT_NE((*target)->machine().findExecutor("worker.0"), nullptr);
  EXPECT_TRUE((*target)->machine().ownerMatches("worker", "worker.0"));
  EXPECT_TRUE((*target)->machine().ownerMatches("vector", "sve_eng.0"));
}

TEST(ArmSveTarget, EveryRuleEmitterIsDeclared) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  for (const RuleDef &rule : (*target)->rules().all())
    EXPECT_TRUE((*target)->isKnownEmitter(rule.emitter)) << rule.id;
  EXPECT_FALSE(sve_mapping::emitterKeys().empty());
}

TEST(ArmSveTarget, ExposesAnEmitterForEachDeclaredKey) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  std::unique_ptr<TargetEmitter> primary = (*target)->createEmitter();
  ASSERT_NE(primary, nullptr);
  EXPECT_TRUE((*target)->isKnownEmitter(primary->key()));

  for (llvm::StringRef key : sve_mapping::emitterKeys()) {
    std::unique_ptr<TargetEmitter> emitter = (*target)->createEmitter(key);
    ASSERT_NE(emitter, nullptr) << key.data();
    EXPECT_EQ(emitter->key(), key);
  }
  EXPECT_EQ((*target)->createEmitter("sve_missing"), nullptr);
}

TEST(ArmSveTarget, EmitterVerifiesBundleCompleteness) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  std::unique_ptr<TargetEmitter> emitter = (*target)->createEmitter();
  ASSERT_NE(emitter, nullptr);
  mlir::MLIRContext context;

  EXPECT_FALSE(
      static_cast<bool>(emitter->verify(makeBundle(emitter->key(), context))));

  // A parameterless bundle carries a null DictionaryAttr; that is the common
  // shape and must be accepted, not dereferenced.
  TargetBundle parameterless = makeBundle(emitter->key(), context);
  parameterless.parameters = {};
  EXPECT_FALSE(static_cast<bool>(emitter->verify(parameterless)));

  llvm::Error unknown = emitter->verify(makeBundle("sve_missing", context));
  ASSERT_TRUE(static_cast<bool>(unknown));
  EXPECT_NE(llvm::toString(std::move(unknown)).find("sve_missing"),
            std::string::npos);

  llvm::ArrayRef<llvm::StringRef> keys = sve_mapping::emitterKeys();
  ASSERT_GE(keys.size(), 2u);
  std::unique_ptr<TargetEmitter> first = (*target)->createEmitter(keys[0]);
  ASSERT_NE(first, nullptr);
  llvm::Error otherKey = first->verify(makeBundle(keys[1], context));
  ASSERT_TRUE(static_cast<bool>(otherKey));
  EXPECT_NE(llvm::toString(std::move(otherKey)).find("handles"),
            std::string::npos);

  TargetBundle malformed = makeBundle(emitter->key(), context);
  malformed.parameters = mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(mlir::StringAttr::get(&context, "tile"),
                                      mlir::UnitAttr::get(&context))});
  llvm::Error badParameters = emitter->verify(malformed);
  ASSERT_TRUE(static_cast<bool>(badParameters));
  EXPECT_NE(llvm::toString(std::move(badParameters)).find("tile"),
            std::string::npos);
}

TEST(ArmSveTarget, MapsAVectorNodeEndToEnd) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  mlir::MLIRContext context;
  WorkloadGraph graph = vectorGraph(context);

  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;

  CoveringSearch search(graph, **target, context, layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->plans.size(), 1u);
  EXPECT_EQ(result->plans[0].instances.size(), 1u);
}

/// The SVE half of the target pair: a vector engine and no accumulator space.
/// Task 5 contrasts this with arm-sme.
TEST(ArmSveTarget, ModelsAVectorEngineAndNoAccumulatorSpace) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  const machine::MachineModel &model = (*target)->machine();

  ASSERT_NE(model.findCompute("svp"), nullptr);
  EXPECT_EQ(model.findCompute("svp")->kind, "vector_engine");
  EXPECT_EQ(model.findMemoryOfKind("acc"), nullptr);
  EXPECT_EQ(model.findCompute("za"), nullptr);
  EXPECT_EQ(model.ownerClass("sve_engine"),
            std::optional<std::string>("vector"));
}

TEST(ArmSveTarget, DoesNotClaimAnExecutableBackend) {
  // This package ships no lowering: SVE execution is out of scope, and saying
  // so is the honest answer. A lowering hook that returned success and emitted
  // nothing would be worse, because the compiler could not then tell a
  // configured target from an executable one.
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      sve_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::ArrayRef<llvm::StringRef> keys = sve_mapping::emitterKeys();
  ASSERT_FALSE(keys.empty());

  mlir::MLIRContext context;
  mlir::DictionaryAttr noParameters;
  TargetBundle bundle;
  bundle.name = "sve.vector.add.f32";
  bundle.emitterKey = keys.front().str();
  bundle.parameters = noParameters;

  for (llvm::StringRef key : keys) {
    std::unique_ptr<TargetEmitter> emitter = (*target)->createEmitter(key);
    ASSERT_TRUE(emitter) << key.str();
    bundle.emitterKey = key.str();
    EXPECT_FALSE(static_cast<bool>(emitter->verify(bundle)))
        << "a well-formed bundle must still verify for " << key.str();

    TargetLoweringContext lowering{(*target)->machine(), {}, {}};
    mlir::IRRewriter rewriter(&context);
    llvm::Error error =
        emitter->lower(/*coveredOps=*/{}, bundle, lowering, rewriter);
    ASSERT_TRUE(static_cast<bool>(error)) << key.str();
    EXPECT_NE(
        llvm::toString(std::move(error)).find("target_lowering_unsupported"),
        std::string::npos)
        << key.str();
  }
}
```

Add `arm_sve_target.cpp` to `SOURCES` in `test/Mapping/CMakeLists.txt`, after `generic_accelerator_target.cpp`.

- [ ] **Step 2: Run the test to verify it fails**

```bash
ninja LLKMappingTests
```

Expected: FAIL to compile — `LLK/Target/AArch64/Mapping/SVEMappingTarget.h` does not exist.

- [ ] **Step 3: Write the package, the policy and the wiring**

Create `include/LLK/Target/AArch64/Mapping/SVEMappingTarget.h` — banner, then:

```cpp
// Part of the target package layer. The first ARM mapping target.
//
// SVE is modelled at the profile's pinned vector length, read back through
// `machine.compute("vector_engine").lanes(...)`; nothing here knows that
// number. The package is deliberately the same shape as the generic
// accelerator's: it loads configuration and returns it, and it claims no
// backend.

#ifndef LLK_TARGET_AARCH64_MAPPING_SVEMAPPINGTARGET_H
#define LLK_TARGET_AARCH64_MAPPING_SVEMAPPINGTARGET_H

#include "LLK/Mapping/MappingTarget.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir::llk::target::sve {

/// The emitter keys this target declares.
llvm::ArrayRef<llvm::StringRef> emitterKeys();

/// Loads the SVE mapping target from a configuration root, which must contain
/// `machines/arm-sve-v2.yaml`, `mapping/arm-sve/layouts.llkmap`, and
/// `mapping/arm-sve/rules.llkmap`.
llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath = {});

} // namespace mlir::llk::target::sve

#endif // LLK_TARGET_AARCH64_MAPPING_SVEMAPPINGTARGET_H
```

Create `lib/Target/AArch64/Mapping/SVEMappingTarget.cpp` — banner, then:

```cpp
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include <string>
#include <vector>

namespace mlir::llk::target::sve {

llvm::ArrayRef<llvm::StringRef> emitterKeys() {
  static const llvm::StringRef kKeys[] = {
      "sve_vector_add",     "sve_vector_convert", "sve_vector_silu",
      "sve_vector_mul",     "sve_mma",            "sve_reduce",
      "sve_copy",           "sve_tile_copy",      "sve_tile_store",
      "sve_fused_convert_silu_mul"};
  return kKeys;
}

llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath) {
  std::string root = configurationRoot.str();
  std::string machine = machinePath.empty()
                            ? root + "/machines/arm-sve-v2.yaml"
                            : machinePath.str();
  std::vector<std::string> keys;
  for (llvm::StringRef key : emitterKeys())
    keys.push_back(key.str());
  return mapping::loadMappingTarget("arm-sve", machine,
                                    root + "/mapping/arm-sve/layouts.llkmap",
                                    root + "/mapping/arm-sve/rules.llkmap",
                                    std::move(keys));
}

} // namespace mlir::llk::target::sve
```

Create `mapping/arm-sve/layouts.llkmap`:

```
// ARM SVE target layouts.
//
// Layout ids are target-owned mapping results, not `#micro.layout` enumerants.
// The vector width is read from the machine, so these declarations follow the
// profile's modeled vector length instead of hard-coding 16.

// Plain row-major: the logical index is the physical index.
layout sve.row_major(int N) {
  param N in [1..8];
  require rank == 2;
  implements row_major;
  map (m, n) -> (m, n);
}

// Two-dimensional blocking with a vector inner dimension. `VW in [4..16]`
// spans the 16 f32 lanes of the 512-bit modeled machine; the equality below
// pins it to whatever the profile declares, so a profile at another vector
// length needs no edit here -- only a wider or narrower domain.
layout sve.blocked_2d(int M, int N, int VW) {
  param M in [4..16];
  param N in [4..16];
  param VW in [4..16];
  require rank == 2;
  require VW == machine.compute("vector_engine").lanes(element_type);
  require N % VW == 0;
  implements blocked;
  map (m, n) -> (m, floordiv(n, VW), mod(n, VW));
}
```

Create `mapping/arm-sve/rules.llkmap`:

```
// ARM SVE mapping rules.
//
// A rule names one Micro operation pattern and the bundle that implements it.
// `bundle` and `emit` are opaque: this file names them, the SVE package gives
// them meaning. Layout ids refer to layouts.llkmap and are never redefined
// here. Capability kinds must be ones machines/arm-sve-v2.yaml declares --
// `verifyMappingTarget` rejects the file otherwise.
//
// `micro.mma` binds to the *vector* engine here. SVE has no matrix engine, so
// the contraction is modelled as vector FMLA/BFDOT work. That is a real
// implementation strategy for SVE, and it is what lets this target be
// exercised on the project's matmul fixture; arm-sme places the same
// operation on a ZA matrix engine instead.

rule sve.vector_add v1 {
  match micro.vector(op = "add");
  param VW in [4..16];
  require executor kind worker;
  require compute kind vector_engine;
  require layout operand0 satisfies sve.blocked_2d;
  input "operand0";
  input "operand1";
  output "result";
  bundle "sve.vector.add.f32";
  emit "sve_vector_add";
  cost 4;
}

rule sve.vector_convert v1 {
  match micro.vector(op = "convert");
  require executor kind worker;
  require compute kind vector_engine;
  require layout operand0 satisfies sve.blocked_2d;
  input "operand0";
  output "result";
  bundle "sve.vector.convert.f32";
  emit "sve_vector_convert";
  cost 2;
}

// The GEMM initializer is read as a row-major tile before promotion into the
// accumulator dtype. Keep this legal path explicit alongside the blocked
// accumulator/result conversion above.
rule sve.vector_convert_row_major v1 {
  match micro.vector(op = "convert");
  param VW in [4..16];
  require executor kind worker;
  require compute kind vector_engine;
  require VW == machine.compute("vector_engine").lanes("f32");
  require layout operand0 satisfies sve.row_major;
  input "operand0";
  output "result";
  bundle "sve.vector.convert.f32" { VW = VW };
  emit "sve_vector_convert";
  cost 2;
}

rule sve.vector_silu v1 {
  match micro.vector(op = "silu");
  require executor kind worker;
  require compute kind vector_engine;
  require layout operand0 satisfies sve.blocked_2d;
  input "operand0";
  output "result";
  bundle "sve.vector.silu.f32";
  emit "sve_vector_silu";
  cost 4;
}

rule sve.vector_mul v1 {
  match micro.vector(op = "mul");
  require executor kind worker;
  require compute kind vector_engine;
  require layout operand0 satisfies sve.blocked_2d;
  input "operand0";
  input "operand1";
  output "result";
  bundle "sve.vector.mul.f32";
  emit "sve_vector_mul";
  cost 4;
}

rule sve.mma_bf16 v1 {
  match micro.mma(input[0].element_type = bf16);
  param VW in [4..16];
  require VW == machine.compute("vector_engine").lanes("f32");
  require executor kind worker;
  require compute kind vector_engine;
  require layout lhs satisfies sve.row_major;
  input "lhs";
  input "rhs";
  input "acc";
  output "result";
  bundle "sve.mma.bf16" { VW = VW };
  emit "sve_mma";
  cost 16;
}

rule sve.mma_f32 v1 {
  match micro.mma(input[0].element_type = f32);
  param VW in [4..16];
  require VW == machine.compute("vector_engine").lanes("f32");
  require executor kind worker;
  require compute kind vector_engine;
  require layout lhs satisfies sve.row_major;
  input "lhs";
  input "rhs";
  input "acc";
  output "result";
  bundle "sve.mma.f32" { VW = VW };
  emit "sve_mma";
  cost 16;
}

rule sve.reduce_sum_f32 v1 {
  match micro.reduce(op = "sum");
  param VW in [4..16];
  require VW == machine.compute("vector_engine").lanes("f32");
  require executor kind worker;
  require compute kind vector_engine;
  input "operand0";
  output "result";
  bundle "sve.reduce.sum.f32" { VW = VW };
  emit "sve_reduce";
  cost 3;
}

// `micro.async_copy` is the one shipped operation whose boundary values are
// plain `tensor`s: it moves into a low-level memory rather than operating on a
// tile that already states one, so its endpoint occurrences carry no explicit
// memory kind. The rule therefore names the ports it governs -- source DRAM to
// destination SRAM, this profile's load path.
rule sve.async_copy v1 {
  match micro.async_copy();
  require executor kind worker;
  require memory input "source" kind dram;
  require memory output "result" kind sram;
  input "source";
  output "result";
  bundle "sve.copy";
  emit "sve_copy";
  cost 2;
}

rule sve.tile_async_copy v1 {
  match micro.tile_async_copy();
  require executor kind worker;
  input "source";
  output "result";
  bundle "sve.tile.copy";
  emit "sve_tile_copy";
  cost 2;
}

rule sve.tile_store v1 {
  match micro.tile_store();
  require executor kind worker;
  input "source";
  bundle "sve.tile.store";
  emit "sve_tile_store";
  cost 2;
}

// The SwiGLU epilogue's shape, implemented as one unit: a convert feeding a
// SiLU feeding the gating multiply. A graph rule matches a *subgraph*, so the
// three operations are placed and emitted together.
rule sve.fused_convert_silu_mul v1 {
  match graph {
    node cv: micro.vector(op = "convert");
    node act: micro.vector(op = "silu");
    node gate: micro.vector(op = "mul");
    edge cv.result -> act.operand0;
    edge act.result -> gate.operand0;
  }
  param VW in [4..16];
  require VW == machine.compute("vector_engine").lanes(element_type);
  require executor kind worker;
  require compute kind vector_engine;
  require layout cv.operand0 satisfies sve.blocked_2d;
  require layout gate.operand1 satisfies sve.blocked_2d;
  require layout gate.result satisfies sve.blocked_2d;
  input "cv.operand0";
  input "gate.operand1";
  output "gate.result";
  bundle "sve.fused.convert_silu_mul" { VW = VW };
  emit "sve_fused_convert_silu_mul";
  cost 3;
}
```

In `lib/Target/CMakeLists.txt`, add to the `LLKTargetMapping` source list, after the `GenericAccelerator/...` line:

```
    AArch64/Mapping/SVEMappingTarget.cpp
```

In `lib/Target/MappingTargets.cpp`, add the include next to the existing two:

```cpp
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"
```

and inside `registerAllMappingTargets()`:

```cpp
  mapping::registerMappingTarget("arm-sve", &sve::createMappingTarget);
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
ninja LLKMappingTests && ./build/bin/LLKMappingTests --gtest_filter=ArmSveTarget.*
```

Expected: 7 tests PASS.

If `LoadsAndVerifiesItsConfiguration` fails with a load error, the message names the offending file and line (the loader reports unknown layout ids, unknown capability kinds and undeclared emitters). Read it rather than guessing.

If `MapsAVectorNodeEndToEnd` finds zero plans, the cause is the layout domain: `sve.blocked_2d` requires `N % VW == 0` with `VW` pinned to 16 and `N in [4..16]`, so `N` must be 16. Widen `param N` to `[16..32]` in `mapping/arm-sve/layouts.llkmap` if the solver needs more than one candidate.

- [ ] **Step 5: Run the full mapping suite**

```bash
./build/bin/LLKMappingTests
```

Expected: all tests pass, including the pre-existing AVX2 and generic-accelerator target tests.

- [ ] **Step 6: Commit**

```bash
git add mapping/arm-sve include/LLK/Target/AArch64/Mapping/SVEMappingTarget.h \
        lib/Target/AArch64/Mapping/SVEMappingTarget.cpp lib/Target/CMakeLists.txt \
        lib/Target/MappingTargets.cpp test/Mapping/arm_sve_target.cpp \
        test/Mapping/CMakeLists.txt
git commit -m "feat(target): add the arm-sve mapping target

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: `arm-sme` machine profile

**Files:**
- Create: `machines/arm-sme-v2.yaml`
- Modify: `test/Machine/machine_model_loader.cpp` (add one test)

**Interfaces:**
- Consumes: nothing (it is written to mirror `machines/arm-sve-v2.yaml` from Task 1 and must agree with it on every shared name).
- Produces: a profile at `machines/arm-sme-v2.yaml` whose `target` is `arm-sme`, containing everything the SVE profile has **plus** executor `za_eng.0` (kind `za_engine`, refines `matrix`), compute node **id `za`** (kind `matrix_engine`, `shapes: [[16, 16, 16]]`), and a memory of kind `acc` (id `za.0`, 4096 bytes, 4 banks). Tasks 4, 5 and 6 depend on these names.

- [ ] **Step 1: Write the failing test**

Append to `test/Machine/machine_model_loader.cpp`:

```cpp
TEST(MachineModelLoader, LoadsTheShippedArmSmeProfile) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(std::string(LLK_MACHINE_DIR) +
                                  "/arm-sme-v2.yaml");
  ASSERT_TRUE(static_cast<bool>(buffer)) << "arm-sme profile must ship";
  llvm::Expected<MachineModel> model =
      parseMachineModel(buffer.get()->getBuffer(), "arm-sme-v2.yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());

  EXPECT_EQ(model->target, "arm-sme");

  // SME is SVE plus a matrix engine: the vector side must be identical in
  // shape, or the two profiles describe different cores.
  const ComputeNode *svp = model->findCompute("svp");
  ASSERT_NE(svp, nullptr);
  EXPECT_EQ(svp->lanes.at("f32"), 16u);
  EXPECT_TRUE(model->ownerMatches("worker", "worker.0"));
  EXPECT_TRUE(model->ownerMatches("matrix", "za_eng.0"));
  EXPECT_EQ(model->ownerClass("za_engine"), std::optional<std::string>("matrix"));

  // The ZA accumulator array, as an `acc`-kind memory of SVL x SVL bits.
  const MemoryNode *za = model->findMemoryOfKind("acc");
  ASSERT_NE(za, nullptr);
  EXPECT_EQ(za->id, "za.0");
  EXPECT_EQ(za->capacityBytes, 4096u);
  EXPECT_EQ(za->banks, 4u);

  const ComputeNode *matrix = model->findCompute("za");
  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->kind, "matrix_engine");
  ASSERT_EQ(matrix->shapes.size(), 1u);
  EXPECT_EQ(matrix->shapes[0],
            std::vector<int64_t>({16, 16, 16}));
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
ninja LLKMachineTests && ./build/bin/LLKMachineTests --gtest_filter=MachineModelLoader.LoadsTheShippedArmSmeProfile
```

Expected: FAIL — `arm-sme profile must ship`.

- [ ] **Step 3: Write the profile**

Create `machines/arm-sme-v2.yaml` — the SVE profile plus the SME additions. Copy the whole of `machines/arm-sve-v2.yaml` (Task 1) and change the header, `target`, `description`, and add the three marked blocks:

```yaml
# Machine model (v2): ARMv9-A core with SVE2 and SME, at a pinned modeled
# vector length.
#
# This is the SVE profile plus the scalable matrix extension. SVL is shared
# between SVE and SME on a real core, which is why both profiles pin 512 bits:
# the pair differs in *capability*, not in vector length, so a ranking
# difference between them is attributable to the matrix engine rather than to
# a different width.
#
# ZA geometry is arithmetic, not estimation. Per the Arm Architecture
# Reference Manual (B1.4.8) ZA is an (SVL/8) x (SVL/8) byte array: at SVL=512
# that is 64 x 64 = 4096 bytes. For 32-bit elements the non-widening FMOPA
# computes an SVL_S x SVL_S sum of outer products with SVL_S = SVL/32, so
# ZA.S is four (SVL/32) x (SVL/32) = 16x16 f32 quadrants, ZA0.S..ZA3.S. The
# matrix engine below models one quadrant per issue; the four quadrants are
# the memory's `banks`.
#
# The numbers are calibration seeds for a class of ARMv9-A cores, not
# measurements of any part.

schema: llk.machine.v2
target: arm-sme
description: >-
  ARMv9-A core with SVE2 and SME at a modeled 512-bit vector length, with a
  ZA matrix engine alongside the SVE vector engine.
clock_hz: 3000000000
worker_threads: 8
sync:
  barrier_cycles: 100
  wait_cycles: 0

executors:
  - id: cluster.0
    kind: cluster
    refines: [group]
    concurrency: 1
  - id: worker.0
    kind: worker
    parent: cluster.0
    concurrency: 8
  - id: sve_eng.0
    kind: sve_engine
    refines: [vector]
    parent: worker.0
    concurrency: 8
  # The matrix capability. It exists so an abstract owner of matrix kind, and
  # an owner path through it, still resolve.
  - id: za_eng.0
    kind: za_engine
    refines: [matrix]
    parent: worker.0
    concurrency: 8

memories:
  - id: dram.0
    kind: dram
    visible_from: cluster.0
    capacity_bytes: 68719476736
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 220
  - id: l2.0
    kind: l2
    visible_from: cluster.0
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major, vectorized]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
  - id: sram.0
    kind: sram
    visible_from: worker.0
    capacity_bytes: 65536
    alignment_bytes: 64
    supported_layouts: [row_major, vectorized, blocked]
    banks: 1
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
  - id: rf.0
    kind: rf
    visible_from: worker.0
    capacity_bytes: 2048
    alignment_bytes: 64
    supported_layouts: [row_major, vectorized]
    banks: 32
    bandwidth_bytes_per_cycle: 256
    latency_cycles: 1
  # ZA: SVL/8 x SVL/8 bytes = 64 x 64 at SVL=512. `acc` is already a member
  # of the Micro memory-space vocabulary, so ZA needs no new word.
  - id: za.0
    kind: acc
    visible_from: worker.0
    capacity_bytes: 4096
    alignment_bytes: 64
    supported_layouts: [row_major, blocked]
    banks: 4
    bandwidth_bytes_per_cycle: 256
    latency_cycles: 1

compute:
  - id: svp
    kind: vector_engine
    refines: [vector]
    attached_to: worker.0
    element_types: [f32, bf16, i8]
    supported_layouts: [row_major, vectorized]
    shapes: [[16], [32], [64]]
    lanes: {f32: 16, bf16: 32, i8: 64}
    issue_cycles: 1
    latency_cycles: 4
    concurrency: 8
  # One ZA quadrant per issue: (SVL/32) x (SVL/32) = 16x16 f32, accumulating
  # over a modeled K of 16.
  - id: za
    kind: matrix_engine
    refines: [matrix]
    attached_to: worker.0
    element_types: [f32, bf16]
    accumulator_dtypes: [f32]
    supported_layouts: [row_major, blocked]
    shapes: [[16, 16, 16]]
    issue_cycles: 1
    latency_cycles: 5
    throughput_per_cycle: 16
    concurrency: 8

transfer_engines:
  - id: dma.0
    kind: dma
    refines: [transfer]
    attached_to: worker.0
    count: 1
    max_outstanding: 8
    setup_cycles: 0

links:
  - id: dram_to_l2.0
    source: dram.0
    destination: l2.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: l2_to_sram.0
    source: l2.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 12
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: dram_to_sram.0
    source: dram.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: sram_to_rf.0
    source: sram.0
    destination: rf.0
    bandwidth_bytes_per_cycle: 256
    latency_cycles: 4
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: rf_to_sram.0
    source: rf.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: sram_to_za.0
    source: sram.0
    destination: za.0
    bandwidth_bytes_per_cycle: 256
    latency_cycles: 4
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: za_to_sram.0
    source: za.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: sram_to_dram.0
    source: sram.0
    destination: dram.0
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 220
    transaction_bytes: 64
    transfer_engines: [dma.0]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
ninja LLKMachineTests && ./build/bin/LLKMachineTests --gtest_filter=MachineModelLoader.LoadsTheShippedArmSmeProfile
```

Expected: PASS. If a field name in the test does not exist (`za->id`, `za->capacityBytes`, `za->banks`, `matrix->shapes`), read `include/LLK/Machine/MachineModel.h` for the real member and fix the test — do not reshape the profile.

- [ ] **Step 5: Commit**

```bash
git add machines/arm-sme-v2.yaml test/Machine/machine_model_loader.cpp
git commit -m "feat(machine): add the arm-sme v2 profile with a ZA matrix engine

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: `arm-sme` mapping target

**Files:**
- Create: `mapping/arm-sme/layouts.llkmap`
- Create: `mapping/arm-sme/rules.llkmap`
- Create: `include/LLK/Target/AArch64/Mapping/SMEMappingTarget.h`
- Create: `lib/Target/AArch64/Mapping/SMEMappingTarget.cpp`
- Modify: `lib/Target/CMakeLists.txt` (one more source line)
- Modify: `lib/Target/MappingTargets.cpp` (one include, one registration)
- Create: `test/Mapping/arm_sme_target.cpp`
- Modify: `test/Mapping/CMakeLists.txt` (add `arm_sme_target.cpp`)

**Interfaces:**
- Consumes: `machines/arm-sme-v2.yaml` from Task 3 (compute id `za` of kind `matrix_engine`, memory of kind `acc`, executor kind `worker`).
- Produces: namespace `mlir::llk::target::sme` with `emitterKeys()` (the ten keys in Step 3) and `createMappingTarget(configurationRoot, machinePath = {})`, registered as `arm-sme`. Task 5 consumes the emitted key `sme_mma_za` and the rule id `sme.mma_f32`.

- [ ] **Step 1: Write the failing test**

Copy `test/Mapping/arm_sve_target.cpp` (created in Task 2), save it as `test/Mapping/arm_sme_target.cpp`, and apply these substitutions. That file is on disk by the time this task runs; read it and edit it rather than reconstructing 200 lines from the table:

| In `arm_sve_target.cpp` | In `arm_sme_target.cpp` |
|---|---|
| `namespace sve_mapping = mlir::llk::target::sve;` | `namespace sme_mapping = mlir::llk::target::sme;` |
| `#include ".../SVEMappingTarget.h"` | `#include ".../SMEMappingTarget.h"` |
| `Test(ArmSveTarget, ...)` | `Test(ArmSmeTarget, ...)` |
| `sve_mapping::` | `sme_mapping::` |
| `"arm-sve"` | `"arm-sme"` |
| `"sve.vector_add"`, `"sve.row_major"` | `"sme.vector_add"`, `"sme.row_major"` |
| `"sve_missing"` | `"sme_missing"` |
| `bundle.name = "sve.vector.add.f32"` | `bundle.name = "sme.vector.add.f32"` |
| `"sve_engine"`, `"sve_eng.0"` | `"za_engine"`, `"za_eng.0"` for matrix, plus keep `"sve_eng.0"` for the vector check |
| `"vector"` in the `ownerMatches`/`ownerClass` assertions | keep the vector assertions **and** add `matrix` for `za_eng.0` |

Also replace the `ModelsAVectorEngineAndNoAccumulatorSpace` test with its SME counterpart:

```cpp
/// The SME half of the target pair: the SVE vector engine is still present,
/// and so is a ZA matrix engine backed by an `acc` memory. Task 5 contrasts
/// this with arm-sve.
TEST(ArmSmeTarget, ModelsAVectorEngineAndAZaMatrixEngine) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  const machine::MachineModel &model = (*target)->machine();

  ASSERT_NE(model.findCompute("svp"), nullptr);
  EXPECT_EQ(model.findCompute("svp")->kind, "vector_engine");
  ASSERT_NE(model.findCompute("za"), nullptr);
  EXPECT_EQ(model.findCompute("za")->kind, "matrix_engine");
  EXPECT_NE(model.findMemoryOfKind("acc"), nullptr);
  EXPECT_EQ(model.ownerClass("za_engine"),
            std::optional<std::string>("matrix"));
}
```

Add `arm_sme_target.cpp` to `SOURCES` in `test/Mapping/CMakeLists.txt`.

- [ ] **Step 2: Run the test to verify it fails**

```bash
ninja LLKMappingTests
```

Expected: FAIL to compile — `SMEMappingTarget.h` does not exist.

- [ ] **Step 3: Write the package, the policy and the wiring**

Copy `include/LLK/Target/AArch64/Mapping/SVEMappingTarget.h` (Task 2) and apply these replacements, keeping the banner: guard `LLK_TARGET_AARCH64_MAPPING_SMEMAPPINGTARGET_H`, namespace `mlir::llk::target::sme`, doc `machines/arm-sme-v2.yaml` / `mapping/arm-sme/`, and a comment noting that this target adds the ZA matrix engine to the SVE vector model.

Copy `lib/Target/AArch64/Mapping/SVEMappingTarget.cpp` (Task 2), keep the banner, and replace `emitterKeys()` with:

```cpp
llvm::ArrayRef<llvm::StringRef> emitterKeys() {
  static const llvm::StringRef kKeys[] = {
      "sme_vector_add",     "sme_vector_convert", "sme_vector_silu",
      "sme_vector_mul",     "sme_mma_za",         "sme_reduce",
      "sme_copy",           "sme_tile_copy",      "sme_tile_store",
      "sme_fused_convert_silu_mul"};
  return kKeys;
}
```

and a `createMappingTarget` that loads `arm-sme`, `machines/arm-sme-v2.yaml`, `mapping/arm-sme/layouts.llkmap`, `mapping/arm-sme/rules.llkmap`.

Copy `mapping/arm-sve/layouts.llkmap` to `mapping/arm-sme/layouts.llkmap`, rename `sve.` to `sme.` throughout, update the header comment, and append:

```
// A ZA-resident tiling for the accumulator.
//
// This layout deliberately carries no `implements` clause. `implements`
// bridges a Micro layout *kind* to a target layout id, and two layouts
// claiming one kind is rejected as ambiguous -- so only `sme.blocked_2d`
// implements `blocked`, and this one is referenced by id from the matrix rule.
layout sme.za_blocked_2d(int M, int N) {
  param M in [4..16];
  param N in [4..16];
  require rank == 2;
  require N % 16 == 0;
  map (m, n) -> (m, n);
}
```

Create `mapping/arm-sme/rules.llkmap`: the twelve `sve.*` rules from Task 2, renamed `sme.*` with `sme_*` emit keys, except that the two contraction rules bind the matrix engine:

```
rule sme.mma_bf16 v1 {
  match micro.mma(input[0].element_type = bf16);
  require executor kind worker;
  require compute kind matrix_engine;
  require layout acc satisfies sme.za_blocked_2d;
  require memory output "result" kind acc;
  input "lhs";
  input "rhs";
  input "acc";
  output "result";
  bundle "sme.mma.bf16";
  emit "sme_mma_za";
  cost 8;
}

rule sme.mma_f32 v1 {
  match micro.mma(input[0].element_type = f32);
  require executor kind worker;
  require compute kind matrix_engine;
  require layout acc satisfies sme.za_blocked_2d;
  require memory output "result" kind acc;
  input "lhs";
  input "rhs";
  input "acc";
  output "result";
  bundle "sme.mma.f32";
  emit "sme_mma_za";
  cost 8;
}
```

Add a header comment stating that this file is the SVE rule set with the contraction re-bound to the ZA matrix engine, and that the vector rules are byte-for-byte the SVE ones apart from their `sme.` prefix.

In `lib/Target/CMakeLists.txt`, add:

```
    AArch64/Mapping/SMEMappingTarget.cpp
```

In `lib/Target/MappingTargets.cpp`, add the include and:

```cpp
  mapping::registerMappingTarget("arm-sme", &sme::createMappingTarget);
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
ninja LLKMappingTests && ./build/bin/LLKMappingTests --gtest_filter=ArmSmeTarget.*
```

Expected: 7 tests PASS. A load error names the offending file and line; read it rather than guessing.

- [ ] **Step 5: Verify both registrations resolve**

```bash
./build/bin/llk-compile --help 2>&1 | head -20
```

Then confirm the registry accepts the new names by running the full mapping suite (the target tests load through the packages, and Task 6 exercises the registry path directly):

```bash
./build/bin/LLKMappingTests
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add mapping/arm-sme include/LLK/Target/AArch64/Mapping/SMEMappingTarget.h \
        lib/Target/AArch64/Mapping/SMEMappingTarget.cpp lib/Target/CMakeLists.txt \
        lib/Target/MappingTargets.cpp test/Mapping/arm_sme_target.cpp \
        test/Mapping/CMakeLists.txt
git commit -m "feat(target): add the arm-sme mapping target

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: Cross-target placement and engine difference

**Files:**
- Create: `test/Mapping/arm_placement.cpp`
- Modify: `test/Mapping/CMakeLists.txt` (add `arm_placement.cpp`)

**Interfaces:**
- Consumes: both packages from Tasks 2 and 4 — `mlir::llk::target::sve` and `mlir::llk::target::sme`, their `createMappingTarget(LLK_SOURCE_DIR)` entry points, and the rule ids `sve.mma_f32` / `sme.mma_f32`.
- Produces: nothing consumed by later tasks.

This is the task that pins the design's central claim: the two targets map the same contraction on different engines, so a ranking difference between them cannot be a relabelled one.

- [ ] **Step 1: Write the test**

Create `test/Mapping/arm_placement.cpp` — banner, then:

```cpp
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <memory>

using namespace mlir::llk::mapping;

namespace sve_mapping = mlir::llk::target::sve;
namespace sme_mapping = mlir::llk::target::sme;

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

/// Both targets, or a failed assertion that says which one did not load.
struct Pair {
  std::unique_ptr<MappingTarget> sve;
  std::unique_ptr<MappingTarget> sme;
};

Pair loadPair() {
  llvm::Expected<std::unique_ptr<MappingTarget>> sve =
      sve_mapping::createMappingTarget(LLK_SOURCE_DIR);
  EXPECT_TRUE(static_cast<bool>(sve)) << llvm::toString(sve.takeError());
  llvm::Expected<std::unique_ptr<MappingTarget>> sme =
      sme_mapping::createMappingTarget(LLK_SOURCE_DIR);
  EXPECT_TRUE(static_cast<bool>(sme)) << llvm::toString(sme.takeError());
  return Pair{std::move(*sve), std::move(*sme)};
}

} // namespace

/// SME is SVE plus a matrix engine. If this ever stops holding, the two
/// profiles have drifted apart and the placement comparison below is void.
TEST(ArmPlacement, SveHasNoMatrixEngineAndSmeDoes) {
  Pair pair = loadPair();
  ASSERT_TRUE(pair.sve);
  ASSERT_TRUE(pair.sme);

  // The vector engine is common to both, at the same modeled width.
  ASSERT_NE(pair.sve->machine().findCompute("svp"), nullptr);
  ASSERT_NE(pair.sme->machine().findCompute("svp"), nullptr);
  EXPECT_EQ(pair.sve->machine().findCompute("svp")->lanes.at("f32"), 16u);
  EXPECT_EQ(pair.sme->machine().findCompute("svp")->lanes.at("f32"), 16u);

  // Only SME has an accumulator space and a matrix engine.
  EXPECT_EQ(pair.sve->machine().findMemoryOfKind("acc"), nullptr);
  EXPECT_EQ(pair.sve->machine().findCompute("za"), nullptr);
  EXPECT_NE(pair.sme->machine().findMemoryOfKind("acc"), nullptr);
  ASSERT_NE(pair.sme->machine().findCompute("za"), nullptr);
  EXPECT_EQ(pair.sme->machine().findCompute("za")->kind, "matrix_engine");
}

/// The same Micro operation, bound to a different engine on each target. The
/// rule text carries the binding (`require compute kind ...`); these are the
/// observable consequences -- the rule exists on both, matches the same
/// operation, and names a different implementation.
TEST(ArmPlacement, ContractionBindsDifferentImplementationsOnEachTarget) {
  Pair pair = loadPair();
  ASSERT_TRUE(pair.sve);
  ASSERT_TRUE(pair.sme);

  const RuleDef *sveMma = pair.sve->rules().find("sve.mma_f32");
  const RuleDef *smeMma = pair.sme->rules().find("sme.mma_f32");
  ASSERT_NE(sveMma, nullptr) << "arm-sve must map micro.mma";
  ASSERT_NE(smeMma, nullptr) << "arm-sme must map micro.mma";

  EXPECT_EQ(sveMma->matchOp, "micro.mma");
  EXPECT_EQ(smeMma->matchOp, "micro.mma");

  // Different emitter keys, so the two targets cannot collapse into the same
  // implementation.
  EXPECT_EQ(sveMma->emitter, "sve_mma");
  EXPECT_EQ(smeMma->emitter, "sme_mma_za");
  EXPECT_NE(sveMma->emitter, smeMma->emitter);

  // Each emitter is declared by its own target and by no other.
  EXPECT_TRUE(pair.sve->isKnownEmitter("sve_mma"));
  EXPECT_FALSE(pair.sme->isKnownEmitter("sve_mma"));
  EXPECT_TRUE(pair.sme->isKnownEmitter("sme_mma_za"));
  EXPECT_FALSE(pair.sve->isKnownEmitter("sme_mma_za"));
}
```

Add `arm_placement.cpp` to `SOURCES` in `test/Mapping/CMakeLists.txt`.

- [ ] **Step 2: Run the test**

```bash
ninja LLKMappingTests && ./build/bin/LLKMappingTests --gtest_filter=ArmPlacement.*
```

Expected: 2 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add test/Mapping/arm_placement.cpp test/Mapping/CMakeLists.txt
git commit -m "test(mapping): pin the arm-sve/arm-sme contraction placement difference

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: `--micro-map` pass-spec path and full-suite verification

**Files:**
- Create: `test/Mapping/arm_sve_micro_map.sh` (or a `.mlir` FileCheck test if the pre-existing mapping tests use that form — check `test/Conversion/MicroMapping/` for the shipped convention first and follow it)
- Modify: the matching registration in `test/FileCheck.cmake` if a FileCheck test is used, or `test/Mapping/CMakeLists.txt` if a script test is used

**Interfaces:**
- Consumes: the policy files from Tasks 2 and 4, and the emitter key lists in those packages.
- Produces: nothing.

The registry path was covered by Tasks 2 and 4. This task covers the *other* dispatch path — `--micro-map target=... machine=... layouts=... rules=... emitters=...` — which consults no registry and needs only configuration. It is the path the shipped mapping shell tests use.

- [ ] **Step 1: Find the shipped convention**

```bash
ls test/Conversion/MicroMapping/ && sed -n '1,60p' test/Conversion/MicroMapping/micro_bind_plan_id.sh
```

Read whichever test drives `--micro-map` with explicit `target=`/`machine=`/`layouts=`/`rules=`/`emitters=` options, and copy its structure exactly.

- [ ] **Step 2: Write the test**

Write the new test against `mapping/arm-sve/` and `machines/arm-sve-v2.yaml`, running `llk-opt --micro-map` with:

```
target=arm-sve machine=machines/arm-sve-v2.yaml layouts=mapping/arm-sve/layouts.llkmap rules=mapping/arm-sve/rules.llkmap emitters=sve_vector_add,sve_vector_convert,sve_vector_silu,sve_vector_mul,sve_mma,sve_reduce,sve_copy,sve_tile_copy,sve_tile_store,sve_fused_convert_silu_mul
```

The input `.mlir` is the same minimal MicroIR kernel the shipped mapping tests use; take it from the test read in Step 1 rather than writing a new one. Register it the same way the shipped test is registered.

- [ ] **Step 3: Run it**

```bash
ninja check-llk
```

Expected: the new test passes and appears in the CTest listing.

- [ ] **Step 4: Run the entire suite and confirm no regressions**

```bash
ninja check-llk
```

Expected: every test that passed before this branch still passes, plus the new ones. Record the pass/fail/skip counts. Do not report the work complete on a partial run.

- [ ] **Step 5: Commit**

```bash
git add test/Conversion/MicroMapping test/FileCheck.cmake test/Mapping/CMakeLists.txt
git commit -m "test(mapping): drive the arm-sve target through the micro-map pass spec

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/manual/source/features.rst` (the delivered-support matrix)
- Modify: `ARCHITECTURE.md` §3 / §5 if either enumerates the shipped targets

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

The repository's review culture distinguishes design intent from delivered support, and `features.rst` is named in `CLAUDE.md` as a source for delivered support. Two new targets must appear there, described with the same boundary language the generic accelerator uses.

- [ ] **Step 1: Find every place the shipped targets are enumerated**

```bash
grep -rn "generic-ai-accel" --include='*.rst' --include='*.md' docs/ ARCHITECTURE.md README.md 2>/dev/null
```

- [ ] **Step 2: Update each**

For each hit that enumerates target support, add `arm-sve` and `arm-sme` and state plainly that they are **mapping/analysis targets**: they model, rank and report plans, and they ship no executable backend — the same boundary sentence `features.rst` already carries for the generic accelerator. Do not claim SVE/SME execution, and do not describe the pinned vector length as a hardware fact.

- [ ] **Step 3: Commit**

```bash
git add docs/manual/source/features.rst ARCHITECTURE.md
git commit -m "docs: record the arm-sve and arm-sme mapping targets

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: §5 profiles → Tasks 1 and 3; §6 layouts → Tasks 2 and 4; §7 rules → Tasks 2 and 4; §8 packaging and both dispatch paths → Tasks 2, 4 and 6; §9 acceptance, including `DoesNotClaimAnExecutableBackend` and the placement difference → Tasks 2, 4 and 5; §10 limits → Task 7's wording constraints; §11 sequencing → the task order. §3's non-goals are enforced by the Global Constraints (no dialect, no schema, no `llk-tune` default machine, no hook cleanup).

**Placeholder scan.** No `TBD`/`TODO`. Task 4 does not repeat Task 2's near-identical files verbatim: it names each file to copy and gives a substitution table covering every value that changes, with the one genuinely different test written out in full. This is a deliberate deviation from "repeat the code", because the copied files are committed and on disk by the time Task 4 runs, so a fresh executor reads them rather than reconstructing them — and repeating ~400 lines of near-duplicate policy would obscure the handful of lines that actually differ. Task 6 Step 1 directs the implementer to read the shipped convention before writing, because the repository's existing mapping shell tests are the authority on that form and this plan should not invent a second one.

**Type consistency.** `createMappingTarget(configurationRoot, machinePath = {})` matches the shipped `GenericAcceleratorMappingTarget` signature in both packages. `emitterKeys()` returns `llvm::ArrayRef<llvm::StringRef>` in both. The emitter keys declared in Tasks 2 and 4 are exactly the keys named by the `emit` directives in the matching `rules.llkmap`, and Task 5 asserts that correspondence rather than assuming it. Profile names are consistent across tasks: compute ids `svp`/`za`, memory kind `acc`, executor kinds `worker`/`sve_engine`/`za_engine`, rule ids `sve.*`/`sme.*`.
