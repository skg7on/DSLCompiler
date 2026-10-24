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
  EXPECT_EQ(requirements->requiredFeatures, std::vector<std::string>{"sme"});
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
  ASSERT_FALSE(
      static_cast<bool>(emitter->lower({mma}, bundle, context, rewriter)));
  EXPECT_TRUE(mma->hasAttr("llk.sme.selected_handoff"));
}

TEST(ArmSMEBackend, SelectedBf16ContractionIsRefused) {
  auto fixture = arm_test::mapFixture("arm-sme", kContractionBf16);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_contraction"; // the bf16 fixture keeps this name
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
  ASSERT_FALSE(static_cast<bool>(compiled));
  const std::string message = llvm::toString(compiled.takeError());
  EXPECT_NE(message.find("realized no selected compute groups"),
            std::string::npos)
      << message;
}

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
  // dialect's high-level tile operations (`arm_sme.outerproduct` carries the
  // contraction); the movement copies are already loops. `convert-arm-sme-to-llvm`
  // and the SCF flattening are a late backend step -- they need owned,
  // structured control flow first -- so the intrinsics appear only after
  // `runLateBackendPipeline` (see the late test below).
  EXPECT_NE(text.find("arm_sme.outerproduct"), std::string::npos) << text;
  EXPECT_EQ(text.find("vector.contract"), std::string::npos) << text;
  EXPECT_EQ(text.find("vector.outerproduct"), std::string::npos) << text;
  // The kernel owns PSTATE.SM and a fresh ZA array. `enable-arm-streaming`
  // records them as MLIR attributes; depending on the MLIR version they print
  // as `llvm.arm_streaming`/`llvm.arm_new_za` or `arm_streaming`/`arm_new_za`.
  // The stable suffix is the attribute name; the LLVM translation names the
  // same two facts `aarch64_pstate_sm_enabled` and `aarch64_new_za`.
  EXPECT_NE(text.find("arm_streaming"), std::string::npos) << text;
  EXPECT_NE(text.find("arm_new_za"), std::string::npos) << text;
}

// The late backend step is what the executable path runs after shared ownership
// preparation: it flattens the structured loops (tile allocation needs CF) and
// converts the ArmSME tile operations to the dialect's LLVM intrinsic
// operations.
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
  // lowered module is the same structured form, so running it directly
  // exercises the same passes.
  ASSERT_FALSE(static_cast<bool>(
      llk::runLateBackendPipeline(*fixture->target, *compiled->module)));

  std::string text;
  llvm::raw_string_ostream stream(text);
  compiled->module->print(stream);
  // Tile allocation needed the flattened form, and the contraction is now the
  // dialect's LLVM MOPA intrinsic operation.
  EXPECT_NE(text.find("cf.br"), std::string::npos) << text;
  EXPECT_EQ(text.find("scf.for"), std::string::npos) << text;
  EXPECT_NE(text.find("arm_sme.intr.mopa"), std::string::npos) << text;
  EXPECT_EQ(text.find("arm_sme.outerproduct"), std::string::npos) << text;
}

// The receipt asks the same per-group question the lowering dispatch asks. The
// contraction key `sme_mma_za` implements one signature; the bf16 contraction
// is mapped by the same key, so a key-level `hasLowering()` would report the
// bf16 plan as merely unchecked. Before any compile it must already be
// `unsupported`, and the f32 group must still be credited by its own selected
// compilation.
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

} // namespace
