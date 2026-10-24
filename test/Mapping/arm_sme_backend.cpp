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

} // namespace
