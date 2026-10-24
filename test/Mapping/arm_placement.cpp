//===- arm_placement.cpp - Real ARM contraction placement and storage ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Framework E5 (issue #180). The same typed, directly authored M=N=K=16
// contraction is mapped on `arm-sve` and `arm-sme` through the public pipeline
// -- registry target, canonical complete-plan evaluation before retention,
// materialization, mapped-IR verification and the strict selected-kernel
// analysis. What differs between the two runs is only the selected compute
// resource: a vector engine on SVE, the matrix engine on SME. The initializer
// is copied into accumulator storage, consumed by the MMA, and the result is
// copied out and yielded.
//
// These tests deliberately assert the *plan* and the *bound IR*, never an
// emitter key or a rule id alone: a target that merely declares emitters has
// not placed anything.

#include "arm_target_fixture.h"

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Mapping/MappingLowering.h"
#include "LLK/Mapping/MappingPlan.h"

#include "mlir/Bytecode/BytecodeOpInterface.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h" // IWYU pragma: keep
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "LLK/Dialect/Micro/MicroEnums.h" // IWYU pragma: keep

// The generated Micro attribute, type and op classes: this test inspects the
// bound IR's concrete operations, not their printed spelling.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::llk::mapping;

namespace {

/// One target/dtype combination: the exact compute resource it must select,
/// the accumulator memory the plan must reserve, and the budget that memory
/// declares.
struct ArmContractionCase {
  const char *targetName;
  const char *dtype;
  const char *inputPath;
  const char *rule;
  const char *computeKind;
  const char *engine;
  const char *otherKind;
  const char *accumulator;
  uint64_t accumulatorBudget;
  /// Bytes per staged bf16/f32 operand tile (16x16).
  uint64_t operandBytes;
};

constexpr llvm::StringLiteral kF32Path =
    "test/Conversion/MicroMapping/Inputs/arm/contraction_f32.mlir";
constexpr llvm::StringLiteral kBf16Path =
    "test/Conversion/MicroMapping/Inputs/arm/contraction_bf16.mlir";

const ArmContractionCase kCases[] = {
    {"arm-sve", "f32", kF32Path.data(), "sve.mma_f32", "vector_engine", "svp",
     "matrix_engine", "acc.0", 2048, 1024},
    {"arm-sme", "f32", kF32Path.data(), "sme.mma_f32", "matrix_engine", "za",
     "vector_engine", "za.0", 4096, 1024},
    {"arm-sve", "bf16", kBf16Path.data(), "sve.mma_bf16", "vector_engine",
     "svp", "matrix_engine", "acc.0", 2048, 512},
    {"arm-sme", "bf16", kBf16Path.data(), "sme.mma_bf16", "matrix_engine", "za",
     "vector_engine", "za.0", 4096, 512},
};

/// The bound kernel's contraction, or null when the materialized IR no longer
/// contains exactly one.
micro::MmaOp findContraction(ModuleOp module) {
  micro::MmaOp found;
  bool ambiguous = false;
  module.walk([&](micro::MmaOp op) {
    if (found)
      ambiguous = true;
    found = op;
  });
  return ambiguous ? micro::MmaOp() : found;
}

/// The stored accumulator must be the caller's initializer, and the yielded
/// result must be the MMA's, moved back out to SRAM. This is what separates a
/// real initialized contraction from an uninitialized accumulator and an
/// implicit output.
void expectInitializedContractionAndYieldedResult(
    const arm_test::MappedFixture &fixture, micro::MmaOp mma) {
  // acc <- copy(sram -> acc) <- copy(dram -> sram) <- tile_view(init)
  auto intoAccumulator = mma.getAcc().getDefiningOp<micro::TileAsyncCopyOp>();
  ASSERT_NE(intoAccumulator, nullptr)
      << "the MMA accumulator is not a materialized movement";
  EXPECT_EQ(intoAccumulator.getDstMemory(), micro::MemorySpace::acc);
  auto intoSram =
      intoAccumulator.getSource().getDefiningOp<micro::TileAsyncCopyOp>();
  ASSERT_NE(intoSram, nullptr)
      << "the accumulator copy does not read a staged initializer";
  EXPECT_EQ(intoSram.getDstMemory(), micro::MemorySpace::sram);
  auto initView = intoSram.getSource().getDefiningOp<micro::TileViewOp>();
  ASSERT_NE(initView, nullptr)
      << "the staged initializer does not read the caller's argument";
  EXPECT_EQ(initView.getSource(),
            fixture.bound.kernel->getRegion(0).front().getArgument(2));

  // yield <- copy(acc -> sram) <- mma result
  auto yield = dyn_cast<micro::YieldOp>(
      fixture.bound.kernel->getRegion(0).front().getTerminator());
  ASSERT_TRUE(yield);
  ASSERT_EQ(yield.getNumOperands(), 1u);
  auto extract = yield.getOperand(0).getDefiningOp<micro::TileAsyncCopyOp>();
  ASSERT_NE(extract, nullptr)
      << "the yielded value is not an explicit extraction";
  EXPECT_EQ(extract.getDstMemory(), micro::MemorySpace::sram);
  EXPECT_EQ(extract.getSource(), mma.getResult());
}

/// Every selected storage slot the plan decided, checked against the three
/// memories this fixture may use. The accumulator's two values are 16x16 f32
/// each; the staged operands live in SRAM; the caller's inputs stay borrowed
/// DRAM descriptors.
void expectMovementStorage(const arm_test::MappedFixture &fixture,
                           const ArmContractionCase &testCase) {
  unsigned accumulators = 0;
  unsigned sram = 0;
  unsigned dram = 0;
  std::vector<uint64_t> sramBytes;
  for (const StorageAllocation &allocation : fixture.plan.allocations) {
    if (allocation.memory == testCase.accumulator) {
      ++accumulators;
      EXPECT_FALSE(allocation.borrowed);
      EXPECT_EQ(allocation.bytes, 1024u) << allocation.memory;
      continue;
    }
    if (allocation.memory == "sram.0") {
      ++sram;
      EXPECT_FALSE(allocation.borrowed);
      sramBytes.push_back(allocation.bytes);
      continue;
    }
    // The caller's three input tensors are borrowed DRAM descriptors, never
    // compiler scratch.
    EXPECT_EQ(allocation.memory, "dram.0")
        << "unexpected selected storage '" << allocation.memory << "'";
    EXPECT_TRUE(allocation.borrowed);
    ++dram;
  }
  // The copied initializer and the MMA result: both in accumulator storage.
  EXPECT_EQ(accumulators, 2u);
  // as, bs, is and the extracted result are all staged in SRAM.
  EXPECT_EQ(sram, 4u);
  EXPECT_EQ(dram, 3u);
  // The two operand stages carry the operand width; the initializer and the
  // extracted result are f32.
  llvm::sort(sramBytes);
  EXPECT_EQ(sramBytes,
            (std::vector<uint64_t>{testCase.operandBytes, testCase.operandBytes,
                                   1024, 1024}));
}

} // namespace

// The acceptance test: the same sources select different concrete compute
// resources, and each target's plan is physically complete with the exact
// 1024-byte live accumulator peak the refined design requires.
TEST(ArmPlacement, SameTypedContractionSelectsDifferentEngines) {
  std::map<std::string, uint64_t> sourceHashes;
  for (const ArmContractionCase &testCase : kCases) {
    SCOPED_TRACE(std::string(testCase.targetName) + "/" + testCase.dtype);
    llvm::Expected<arm_test::MappedFixture> mapped =
        arm_test::mapFixture(testCase.targetName, testCase.inputPath);
    ASSERT_TRUE(bool(mapped)) << llvm::toString(mapped.takeError());

    const PlanPlacement *placement = arm_test::contractionPlacement(*mapped);
    ASSERT_NE(placement, nullptr);
    EXPECT_EQ(placement->rule, testCase.rule);
    EXPECT_EQ(placement->executor, "worker.0");
    // The concrete engine the plan selected, by requirement kind -- not the
    // machine's first attachment and not an emitter string.
    EXPECT_EQ(placement->computeBindings.lookup(testCase.computeKind),
              testCase.engine);
    EXPECT_TRUE(placement->computeBindings.lookup(testCase.otherKind).empty())
        << "the plan selected a resource of the wrong kind";
    ASSERT_EQ(placement->computeRequirements.size(), 1u);
    EXPECT_EQ(placement->computeRequirements.front(), testCase.computeKind);

    // The rule's named accumulator and result ports both resolve to the
    // target's accumulator storage, at the occurrences the rule names.
    bool accumulatorInput = false;
    bool accumulatorOutput = false;
    for (const PortMemoryBinding &binding : placement->portMemoryBindings) {
      if (binding.memory != testCase.accumulator) {
        ADD_FAILURE() << "the contraction bound a named port to unexpected "
                         "memory '"
                      << binding.memory << "'";
        continue;
      }
      if (binding.port.direction == PortDirection::Input &&
          binding.port.index == 2)
        accumulatorInput = true;
      if (binding.port.direction == PortDirection::Output &&
          binding.port.index == 0)
        accumulatorOutput = true;
    }
    EXPECT_TRUE(accumulatorInput) << "the MMA accumulator input is not in "
                                     "accumulator storage";
    EXPECT_TRUE(accumulatorOutput) << "the MMA result is not in accumulator "
                                      "storage";

    // Physical completeness is asserted separately from executable readiness:
    // every selected value's memory, footprint and live interval resolved.
    EXPECT_TRUE(mapped->plan.diagnostics.physicalComplete);
    EXPECT_TRUE(mapped->plan.diagnostics.physicalReasons.empty())
        << llvm::join(mapped->plan.diagnostics.physicalReasons, "; ");

    // Complete, plan-liveness analysis with no incomplete reason.
    EXPECT_TRUE(mapped->analysis.complete);
    EXPECT_TRUE(mapped->analysis.incompleteReasons.empty())
        << llvm::join(mapped->analysis.incompleteReasons, "; ");
    EXPECT_TRUE(mapped->analysis.storageLivenessFromPlan);
    auto peak = mapped->analysis.peakBytes.find(testCase.accumulator);
    ASSERT_NE(peak, mapped->analysis.peakBytes.end());
    EXPECT_EQ(peak->second, 1024u);
    EXPECT_LE(peak->second, testCase.accumulatorBudget);

    // Nothing the plan selected was left unmaterialized.
    EXPECT_TRUE(mapped->bound.unmaterialized.empty())
        << llvm::join(mapped->bound.unmaterialized, "; ");

    // The identical source input is what both targets mapped.
    auto [entry, inserted] =
        sourceHashes.emplace(testCase.inputPath, mapped->plan.graphHash);
    if (!inserted)
      EXPECT_EQ(entry->second, mapped->plan.graphHash)
          << "the source identity changed between targets";

    micro::MmaOp mma = findContraction(*mapped->bound.module);
    ASSERT_TRUE(mma);
    expectInitializedContractionAndYieldedResult(*mapped, mma);
    expectMovementStorage(*mapped, testCase);
  }
}

// A physically complete static plan is still not executable code: these
// targets declare no ISA requirements, so asking for selected-target
// compilation must refuse instead of silently falling back to reference
// lowering.
TEST(ArmPlacement, SelectedTargetCompilationRejectsMissingIsa) {
  for (const ArmContractionCase &testCase : kCases) {
    SCOPED_TRACE(std::string(testCase.targetName) + "/" + testCase.dtype);
    llvm::Expected<arm_test::MappedFixture> mapped =
        arm_test::mapFixture(testCase.targetName, testCase.inputPath);
    ASSERT_TRUE(bool(mapped)) << llvm::toString(mapped.takeError());

    ::llk::MappedCompileOptions options;
    options.backend = ::llk::MappedBackend::SelectedTarget;
    options.stop = ::llk::MappedStop::MappedMicro;
    // The binding contract is not what is under test here; the target's
    // declared capability is. The plan is fully materialized, so the partial
    // contract still binds the same IR.
    options.requireExecutable = false;

    llvm::Expected<::llk::MappedCompilation> compiled =
        ::llk::compileMappedKernel(*mapped->source, *mapped->target,
                                   mapped->plan, options);
    ASSERT_FALSE(bool(compiled))
        << "a target with no code generation accepted selected-target "
           "compilation";
    const std::string message = llvm::toString(compiled.takeError());
    EXPECT_NE(message.find("ISA requirements"), std::string::npos) << message;
  }
}

// The helper's failures are `llvm::Expected` errors, so a bad path or an
// unregistered target is a diagnosable failure rather than a null dereference.
TEST(ArmPlacement, FailedFixturePropagatesError) {
  llvm::Expected<arm_test::MappedFixture> missing = arm_test::mapFixture(
      "arm-sve", "test/Conversion/MicroMapping/Inputs/arm/missing.mlir");
  ASSERT_FALSE(bool(missing));
  EXPECT_NE(llvm::toString(missing.takeError()).find("missing.mlir"),
            std::string::npos);

  llvm::Expected<arm_test::MappedFixture> unknown =
      arm_test::mapFixture("arm-unknown", kF32Path);
  ASSERT_FALSE(bool(unknown));
  EXPECT_NE(llvm::toString(unknown.takeError()).find("arm-unknown"),
            std::string::npos);
}

// Every bundle the map actually selected is a well-formed opaque bundle the
// target's emitter verifies -- and every one of those emitters still refuses to
// lower it. Verifying a bundle is not implementing it, and this is the boundary
// that keeps the static pair honest.
TEST(ArmPlacement, SelectedBundlesVerifyButCannotLower) {
  for (const ArmContractionCase &testCase : kCases) {
    SCOPED_TRACE(std::string(testCase.targetName) + "/" + testCase.dtype);
    llvm::Expected<arm_test::MappedFixture> mapped =
        arm_test::mapFixture(testCase.targetName, testCase.inputPath);
    ASSERT_TRUE(bool(mapped)) << llvm::toString(mapped.takeError());
    EXPECT_FALSE(mapped->target->codegenRequirements().has_value());

    IRRewriter rewriter(mapped->context.get());
    ASSERT_FALSE(mapped->plan.placements.empty());
    for (const PlanPlacement &placement : mapped->plan.placements) {
      SCOPED_TRACE("rule " + placement.rule);
      std::unique_ptr<TargetEmitter> emitter =
          mapped->target->createEmitter(placement.bundle.emitterKey);
      ASSERT_NE(emitter, nullptr) << placement.bundle.emitterKey;
      EXPECT_EQ(llvm::toString(emitter->verify(placement.bundle)), "");
      EXPECT_FALSE(emitter->hasLowering());
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
