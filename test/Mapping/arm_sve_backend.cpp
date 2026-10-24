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
#include "LLK/Conversion/MappedKernelAbi.h"
#include "LLK/Mapping/MappingLowering.h"
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include "arm_target_fixture.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

constexpr const char *kVectorAdd =
    "test/Conversion/MicroMapping/Inputs/arm/vector_add_f32.mlir";
constexpr const char *kVectorMulFull =
    "test/Conversion/MicroMapping/Inputs/arm/vector_mul_f32_full.mlir";
constexpr const char *kVectorBf16Add =
    "test/Conversion/MicroMapping/Inputs/arm/vector_add_bf16.mlir";
constexpr const char *kVectorI8Mul =
    "test/Conversion/MicroMapping/Inputs/arm/vector_mul_i8.mlir";
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

// A row of an 8x8 f32 tile is half of the modeled 512-bit vector, so the
// scalable form needs a mask. A 16x16 row is exactly one 512-bit vector: it is
// the full-width shape the executable subset must also realize, and it lowers
// to the same `vector<16x[4]xf32>` form without any fallback group.
TEST(ArmSVEBackend, SelectedFullWidthVectorMulVectorizesToScalableSVE) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorMulFull);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_mul_f32_full";
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
  EXPECT_NE(text.find("vector<16x[4]xf32>"), std::string::npos) << text;
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

// The executable subset is a dtype boundary, not a key boundary. A reference
// compilation is the portable path and must still lower the bf16 and i8 rules
// the same keys cover; the per-group capability query keeps them on the
// reference bridge instead of letting the f32-only emitter fail the run.
TEST(ArmSVEBackend, ReferenceBackendKeepsUnimplementedDtypesPortable) {
  struct Case {
    const char *path;
    const char *entry;
  };
  for (const Case &testCase : {Case{kVectorBf16Add, "arm_vector_add_bf16"},
                               Case{kVectorI8Mul, "arm_vector_mul_i8"}}) {
    SCOPED_TRACE(testCase.path);
    auto fixture = arm_test::mapFixture("arm-sve", testCase.path);
    ASSERT_TRUE(static_cast<bool>(fixture))
        << llvm::toString(fixture.takeError());
    mlir::OwningOpRef<mlir::ModuleOp> source =
        mlir::cast<mlir::ModuleOp>(fixture->source->clone());
    llk::MappedCompileOptions options;
    options.entrySymbol = testCase.entry;
    options.backend = llk::MappedBackend::Reference;
    options.stop = llk::MappedStop::Lowered;
    llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
        *source, *fixture->target, fixture->plan, options);
    ASSERT_TRUE(static_cast<bool>(compiled))
        << llvm::toString(compiled.takeError());
    // The honest accounting: a group the target cannot lower is carried by the
    // reference bridge, not quietly claimed as target code.
    EXPECT_GT(compiled->referenceGroupsLowered, 0u);
    EXPECT_EQ(compiled->targetLowered, 0u);
    EXPECT_EQ(compiled->backendGroupsRealized, 0u);
  }
}

// The same bf16 signature selected for the target is refused: the shared
// contract reports that no selected compute group was realized rather than
// silently falling back to reference lowering and calling it target execution.
TEST(ArmSVEBackend, SelectedBackendRefusesUnimplementedDtypes) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorBf16Add);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_add_bf16";
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

// The prepared artifact is what the AOT object and the JIT both consume.
// Ownership-based buffer deallocation runs there, and it inserts a constant
// into every region it traverses -- which a `vector.mask` region (exactly one
// masked operation) cannot hold. The selected pass dissolves the mask and
// rank-reduces the scalable transfers, so preparation must succeed and the
// whole-tile scalable arithmetic must survive.
TEST(ArmSVEBackend, SelectedVectorAddPreparesForInvocation) {
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

  llvm::Expected<llk::PreparedMappedKernel> prepared =
      llk::prepareMappedKernelForInvocation(*compiled->module,
                                            "arm_vector_add_f32");
  ASSERT_TRUE(static_cast<bool>(prepared))
      << llvm::toString(prepared.takeError());

  bool hasMask = false;
  prepared->module->walk([&](mlir::Operation *op) {
    hasMask |= op->getName().getStringRef() == "vector.mask";
  });
  EXPECT_FALSE(hasMask) << "a vector.mask region survives kernel preparation";
  std::string text;
  llvm::raw_string_ostream stream(text);
  prepared->module->print(stream);
  EXPECT_NE(text.find("vector<8x[4]xf32>"), std::string::npos) << text;
}

// The emitter's pre-write contract: a covered group is validated as a whole, so
// a later unsupported operation cannot leave earlier operations marked as
// handed off. The per-group capability query answers the same question without
// mutating anything, which is what keeps a portable reference run off the
// f32-only emitter.
TEST(ArmSVEBackend, EmitterValidatesTheWholeGroupBeforeMutating) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorAdd);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  mlir::Operation *vector = nullptr;
  mlir::Operation *other = nullptr;
  fixture->bound.module->walk([&](mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (!vector && name == "micro.vector")
      vector = op;
    if (!other && name == "micro.yield")
      other = op;
  });
  ASSERT_NE(vector, nullptr);
  ASSERT_NE(other, nullptr);

  auto emitter = fixture->target->createEmitter("sve_vector_add");
  ASSERT_NE(emitter, nullptr);
  mlir::IRRewriter rewriter(fixture->context.get());
  mlir::llk::mapping::TargetLoweringContext lowering{
      fixture->target->machine(), {}, {}};
  mlir::llk::mapping::TargetBundle bundle{
      "sve.vector_add_f32", {}, "sve_vector_add"};

  EXPECT_TRUE(emitter->hasLoweringFor({vector}));
  EXPECT_FALSE(emitter->hasLoweringFor({vector, other}));

  llvm::Error error =
      emitter->lower({vector, other}, bundle, lowering, rewriter);
  EXPECT_TRUE(static_cast<bool>(error));
  EXPECT_FALSE(vector->hasAttr("llk.sve.selected_handoff"))
      << "a rejected group left an earlier operation mutated";

  // The valid group lowers and is marked only after the whole group passed.
  EXPECT_FALSE(
      static_cast<bool>(emitter->lower({vector}, bundle, lowering, rewriter)));
  EXPECT_TRUE(vector->hasAttr("llk.sve.selected_handoff"));
}

} // namespace
