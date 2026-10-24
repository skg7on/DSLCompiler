//===- arm_sve_backend.cpp - Selected SVE lowering -----------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Framework E10 (issue #184): the `arm-sve` target is executable for exactly
// one subset -- f32 `micro.vector "add"` and `"mul"`. These tests prove the
// three properties that make that claim meaningful:
//
//  * the two arithmetic emitters declare a lowering and the target publishes
//    the AArch64/SVE code-generation contract, while every other key (and the
//    contraction signature) keeps the explicit rejection;
//  * a selected add compiles with no reference-lowered group and reaches
//    scalable SVE Vector IR;
//  * a selected contraction still refuses to compile, so "executable subset"
//    does not silently widen to the whole target.

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include "arm_target_fixture.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

constexpr const char *kVectorAdd =
    "test/Conversion/MicroMapping/Inputs/arm/vector_add_f32.mlir";
constexpr const char *kContraction =
    "test/Conversion/MicroMapping/Inputs/arm/contraction_f32.mlir";

TEST(ArmSVEBackend, OnlyArithmeticKeysLower) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorAdd);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  EXPECT_TRUE(fixture->target->createEmitter("sve_vector_add")->hasLowering());
  EXPECT_TRUE(fixture->target->createEmitter("sve_vector_mul")->hasLowering());
  EXPECT_FALSE(fixture->target->createEmitter("sve_mma")->hasLowering());
  EXPECT_FALSE(fixture->target->createEmitter("sve_reduce")->hasLowering());
  auto requirements = fixture->target->codegenRequirements();
  ASSERT_TRUE(requirements.has_value());
  EXPECT_EQ(requirements->architecture, "aarch64");
  EXPECT_EQ(requirements->requiredFeatures, std::vector<std::string>{"sve"});
}

TEST(ArmSVEBackend, SelectedVectorAddVectorizesToScalableSVE) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorAdd);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_add_f32";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
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
  EXPECT_FALSE(static_cast<bool>(compiled));
}

} // namespace
