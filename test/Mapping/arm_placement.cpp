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

#include "MicroMappingCommon.h"

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Mapping/LayoutConstraints.h"
#include "LLK/Mapping/LlkMap.h"
#include "LLK/Mapping/MappingLowering.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingRules.h"
#include "LLK/Target/MappingTargets.h"

#include "mlir/Bytecode/BytecodeOpInterface.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"

#include <gtest/gtest.h>

#include <map>
#include <optional>
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

// A physically complete static plan is still not executable code. SME declares
// no ISA requirements at all, so selected-target compilation refuses before any
// lowering. SVE declares a contract for its f32 vector subset only: this
// contraction selects `sve_mma`, which has no lowering, so the shared contract
// reports that no selected compute group was realized rather than falling back
// to reference lowering and calling the result target execution.
TEST(ArmPlacement, SelectedTargetCompilationRefusesUnimplementedSignatures) {
  for (const ArmContractionCase &testCase : kCases) {
    SCOPED_TRACE(std::string(testCase.targetName) + "/" + testCase.dtype);
    llvm::Expected<arm_test::MappedFixture> mapped =
        arm_test::mapFixture(testCase.targetName, testCase.inputPath);
    ASSERT_TRUE(bool(mapped)) << llvm::toString(mapped.takeError());

    ::llk::MappedCompileOptions options;
    options.backend = ::llk::MappedBackend::SelectedTarget;
    options.stop = ::llk::MappedStop::Lowered;
    // The binding contract is not what is under test here; the target's
    // declared capability is. The plan is fully materialized, so the partial
    // contract still binds the same IR.
    options.requireExecutable = false;

    llvm::Expected<::llk::MappedCompilation> compiled =
        ::llk::compileMappedKernel(*mapped->source, *mapped->target,
                                   mapped->plan, options);
    ASSERT_FALSE(bool(compiled))
        << "a target accepted selected-target compilation for an unimplemented "
           "signature";
    const std::string message = llvm::toString(compiled.takeError());
    const llvm::StringRef expected =
        llvm::StringRef(testCase.targetName) == "arm-sve"
            ? "realized no selected compute groups"
            : "ISA requirements";
    EXPECT_NE(message.find(expected), std::string::npos) << message;
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
// target's emitter verifies -- and the contraction emitter still refuses to
// lower it. Verifying a bundle is not implementing it, and this is the boundary
// that keeps the executable subset honest.
TEST(ArmPlacement, SelectedBundlesVerifyButCannotLower) {
  for (const ArmContractionCase &testCase : kCases) {
    SCOPED_TRACE(std::string(testCase.targetName) + "/" + testCase.dtype);
    llvm::Expected<arm_test::MappedFixture> mapped =
        arm_test::mapFixture(testCase.targetName, testCase.inputPath);
    ASSERT_TRUE(bool(mapped)) << llvm::toString(mapped.takeError());
    // `arm-sve` publishes a code-generation contract for its vector subset, so
    // only SME is contract-free now. The contraction case below still proves
    // that a declared emitter is not an implementation.
    if (llvm::StringRef(testCase.targetName) == "arm-sve")
      EXPECT_TRUE(mapped->target->codegenRequirements().has_value());
    else
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

//===----------------------------------------------------------------------===//
// Framework E6 (#181): typed signatures, infeasibility and tampered identity
//===----------------------------------------------------------------------===//
//
// The E5 acceptance ran one contraction shape. This section covers the rest of
// the advertised table through the same public pipeline, then asks the same
// pipeline to refuse: unsupported signatures, machine-profile variants that
// break one storage or routing fact, and plans whose selected compute identity
// was edited. Every negative searches with topK=1 and asserts the frontier, so
// "no plan" is attributable to the changed fact and cannot be a search cap.

namespace {

/// One advertised vector signature: the fixture, the rule that must cover it,
/// and the width its blocked layout must solve at the named operand.
struct VectorSignatureCase {
  const char *inputPath;
  const char *ruleSuffix;
  const char *opName;
  const char *opAttribute;
  /// Unset when the rule declares no layout requirement (the f32 reduction).
  std::optional<int64_t> width;
};

const VectorSignatureCase kVectorSignatures[] = {
    {"test/Conversion/MicroMapping/Inputs/arm/vector_add_f32.mlir",
     "vector_add_f32", "micro.vector", "add", 16},
    {"test/Conversion/MicroMapping/Inputs/arm/vector_add_bf16.mlir",
     "vector_add_bf16", "micro.vector", "add", 32},
    {"test/Conversion/MicroMapping/Inputs/arm/vector_add_i8.mlir",
     "vector_add_i8", "micro.vector", "add", 64},
    {"test/Conversion/MicroMapping/Inputs/arm/vector_mul_f32.mlir",
     "vector_mul_f32", "micro.vector", "mul", 16},
    {"test/Conversion/MicroMapping/Inputs/arm/vector_mul_bf16.mlir",
     "vector_mul_bf16", "micro.vector", "mul", 32},
    {"test/Conversion/MicroMapping/Inputs/arm/vector_mul_i8.mlir",
     "vector_mul_i8", "micro.vector", "mul", 64},
    // Conversion solves the operand's width, not the result's: bf16 -> 32 and
    // f32 -> 16, so the two directions must not share one graph-wide dtype.
    {"test/Conversion/MicroMapping/Inputs/arm/vector_convert_bf16_f32.mlir",
     "vector_convert_bf16_f32", "micro.vector", "convert", 32},
    {"test/Conversion/MicroMapping/Inputs/arm/vector_convert_f32_bf16.mlir",
     "vector_convert_f32_bf16", "micro.vector", "convert", 16},
    {"test/Conversion/MicroMapping/Inputs/arm/vector_silu_f32.mlir",
     "vector_silu_f32", "micro.vector", "silu", 16},
    {"test/Conversion/MicroMapping/Inputs/arm/reduce_sum_f32.mlir",
     "reduce_sum_f32", "micro.reduce", "sum", std::nullopt},
};

/// The unique graph node matching `opName` and, when given, its `op`
/// attribute. A second match makes the fixture ambiguous, so this returns null.
const WorkloadNode *findWorkloadNode(const arm_test::MappedFixture &fixture,
                                     llvm::StringRef opName,
                                     llvm::StringRef opAttribute = {}) {
  const WorkloadNode *found = nullptr;
  for (const WorkloadNode &node : fixture.graph.getNodes()) {
    if (node.opName != opName)
      continue;
    if (!opAttribute.empty()) {
      auto op = node.attributes.getAs<StringAttr>("op");
      if (!op || op.getValue() != opAttribute)
        continue;
    }
    if (found)
      return nullptr;
    found = &node;
  }
  return found;
}

/// The width the selected plan solved for `port`, or null when no solution
/// names it. The solution records the occurrence it was solved for, so two
/// uses of one value are not conflated.
std::optional<int64_t> solvedWidth(const arm_test::MappedFixture &fixture,
                                   const PortRef &port) {
  for (const PlanPlacement &placement : fixture.plan.placements)
    for (const auto &entry : placement.layoutSolutions) {
      const SolvedLayout &solution = entry.second;
      if (!solution.port || !(*solution.port == port))
        continue;
      auto value = solution.parameters.find("VW");
      if (value == solution.parameters.end())
        return std::nullopt;
      if (const auto *width = std::get_if<int64_t>(&value->second))
        return *width;
      return std::nullopt;
    }
  return std::nullopt;
}

bool hasDiagnostic(const MappingSearchResult &result, DiagnosticCode code,
                   llvm::StringRef fragment = {}) {
  for (const mlir::llk::mapping::Diagnostic &diagnostic :
       result.frontier.diagnostics) {
    if (diagnostic.code != code)
      continue;
    if (fragment.empty() ||
        llvm::StringRef(diagnostic.message).contains(fragment))
      return true;
  }
  return false;
}

/// True when some recorded diagnostic names every one of `fragments`,
/// regardless of its stable code. Used where the refusal may legitimately
/// surface as a route failure or as a storage-materialization failure.
bool diagnosticsMention(const MappingSearchResult &result,
                        std::initializer_list<llvm::StringRef> fragments) {
  for (const mlir::llk::mapping::Diagnostic &diagnostic :
       result.frontier.diagnostics) {
    llvm::StringRef message(diagnostic.message);
    bool all = true;
    for (llvm::StringRef fragment : fragments)
      if (!message.contains(fragment)) {
        all = false;
        break;
      }
    if (all)
      return true;
  }
  return false;
}

std::string describeDiagnostics(const MappingSearchResult &result) {
  std::string text;
  for (const mlir::llk::mapping::Diagnostic &diagnostic :
       result.frontier.diagnostics) {
    text += stringifyDiagnosticCode(diagnostic.code).str();
    text += ": " + diagnostic.message + "\n";
  }
  return text;
}

llvm::Expected<std::string> readTextFile(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::errorCodeToError(buffer.getError());
  return (*buffer)->getBuffer().str();
}

llvm::Expected<std::string> replacedOnce(std::string text, llvm::StringRef from,
                                         llvm::StringRef to) {
  size_t offset = text.find(from.str());
  if (offset == std::string::npos)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "machine profile does not contain '" +
                                       from.str() + "'");
  text.replace(offset, from.size(), to.str());
  return text;
}

/// Drop one `  - id: <id>` YAML entry and its nested keys. The next sibling
/// entry or the next top-level key ends the block, so a removed memory can be
/// removed together with the links that name it while leaving valid YAML.
llvm::Expected<std::string> removedEntry(llvm::StringRef text,
                                         llvm::StringRef id) {
  llvm::SmallVector<llvm::StringRef> lines;
  text.split(lines, '\n');
  const std::string marker = "- id: " + id.str();
  size_t begin = lines.size();
  for (size_t index = 0; index < lines.size(); ++index)
    if (lines[index].ltrim().starts_with(marker)) {
      begin = index;
      break;
    }
  if (begin == lines.size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "machine profile has no entry '- id: " + id.str() + "'");
  size_t end = begin + 1;
  while (end < lines.size() && lines[end].starts_with("    "))
    ++end;
  std::string result;
  for (size_t index = 0; index < lines.size(); ++index) {
    if (index >= begin && index < end)
      continue;
    result += lines[index].str();
    if (index + 1 < lines.size())
      result += '\n';
  }
  return result;
}

llvm::Expected<std::string> insertedBefore(std::string text,
                                           llvm::StringRef anchor,
                                           llvm::StringRef block) {
  size_t offset = text.find(anchor.str());
  if (offset == std::string::npos)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "machine profile does not contain '" +
                                       anchor.str() + "'");
  text.insert(offset, block.str());
  return text;
}

llvm::Expected<std::string> shippedMachine(llvm::StringRef targetName) {
  return readTextFile(std::string(LLK_SOURCE_DIR) + "/machines/" +
                      targetName.str() + "-v2.yaml");
}

/// The one profile fact this target's contraction depends on.
struct ArmStorageFacts {
  const char *targetName;
  const char *accumulator;
  const char *intoAccumulatorLink;
  const char *outOfAccumulatorLink;
  uint64_t declaredCapacity;
};

const ArmStorageFacts kStorageFacts[] = {
    {"arm-sve", "acc.0", "sram_to_acc.0", "acc_to_sram.0", 2048},
    {"arm-sme", "za.0", "sram_to_za.0", "za_to_sram.0", 4096},
};

/// A rank-two f64 elementwise graph. It is built at the workload-graph layer
/// because f64 is not a representable `!micro.tile` element type, so no MicroIR
/// source can carry this signature; the rule table is what must reject it.
WorkloadGraph f64VectorGraph(mlir::MLIRContext &context) {
  mlir::Type f64 = mlir::Float64Type::get(&context);
  mlir::Type tile = mlir::RankedTensorType::get({8, 8}, f64);
  WorkloadGraph graph;
  WorkloadValueId lhs =
      graph.addValue(WorkloadValue{0, tile, "lhs", /*external=*/true});
  WorkloadValueId rhs =
      graph.addValue(WorkloadValue{0, tile, "rhs", /*external=*/true});
  WorkloadValueId result =
      graph.addValue(WorkloadValue{0, tile, "result", /*external=*/false});

  WorkloadNode node;
  node.opName = "micro.vector";
  node.attributes = DictionaryAttr::get(
      &context, {NamedAttribute(StringAttr::get(&context, "op"),
                                StringAttr::get(&context, "add"))});
  node.inputs.push_back(WorkloadPort{lhs, tile, std::nullopt});
  node.inputs.push_back(WorkloadPort{rhs, tile, std::nullopt});
  node.outputs.push_back(WorkloadPort{result, tile, std::nullopt});
  graph.addNode(std::move(node));
  graph.finalize();
  return graph;
}

/// Searches a hand-built graph with the registered target. No evaluator is
/// installed: this control only asks whether any rule covers the signature, and
/// an uncovered node produces no proposal at all.
llvm::Expected<MappingSearchResult> searchGraph(llvm::StringRef targetName,
                                                const WorkloadGraph &graph,
                                                mlir::MLIRContext &context) {
  mlir::llk::target::registerAllMappingTargets();
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      createRegisteredMappingTarget(targetName, LLK_SOURCE_DIR);
  if (!target)
    return target.takeError();

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  options.topK = 1;
  options.requireCompleteEvaluation = false;
  CoveringSearch search(
      graph, **target, context,
      mlir::llk::micro_mapping_detail::deriveLayoutContext(graph), options);
  return search.search();
}

} // namespace

// Every advertised vector signature maps completely on both targets, and each
// blocked layout retains the width of the port it was solved for.
TEST(ArmDtypeCoverage, TypedSignaturesMapCompletely) {
  for (llvm::StringRef targetName : {"arm-sve", "arm-sme"}) {
    llvm::StringRef prefix = targetName == "arm-sve" ? llvm::StringRef("sve")
                                                     : llvm::StringRef("sme");
    for (const VectorSignatureCase &testCase : kVectorSignatures) {
      SCOPED_TRACE((targetName + "/" + testCase.inputPath).str());
      llvm::Expected<arm_test::MappedFixture> mapped =
          arm_test::mapFixture(targetName, testCase.inputPath);
      ASSERT_TRUE(bool(mapped)) << llvm::toString(mapped.takeError());

      EXPECT_TRUE(mapped->plan.diagnostics.physicalComplete);
      EXPECT_TRUE(mapped->plan.diagnostics.physicalReasons.empty())
          << llvm::join(mapped->plan.diagnostics.physicalReasons, "; ");
      EXPECT_TRUE(mapped->analysis.complete);
      EXPECT_TRUE(mapped->analysis.incompleteReasons.empty())
          << llvm::join(mapped->analysis.incompleteReasons, "; ");
      EXPECT_TRUE(mapped->bound.unmaterialized.empty());

      const WorkloadNode *node =
          findWorkloadNode(*mapped, testCase.opName, testCase.opAttribute);
      ASSERT_NE(node, nullptr);
      const PlanPlacement *placement = nullptr;
      for (const PlanPlacement &candidate : mapped->plan.placements)
        if (candidate.node == node->id) {
          ASSERT_EQ(placement, nullptr) << "the node was placed twice";
          placement = &candidate;
        }
      ASSERT_NE(placement, nullptr);
      EXPECT_EQ(placement->rule, (prefix + "." + testCase.ruleSuffix).str());
      EXPECT_EQ(placement->executor, "worker.0");
      EXPECT_EQ(placement->computeBindings.lookup("vector_engine"), "svp");

      const PortRef operand{node->id, PortDirection::Input, 0};
      if (testCase.width) {
        EXPECT_EQ(solvedWidth(*mapped, operand), testCase.width)
            << "the blocked layout did not retain the port's own width";
      } else {
        // The reduce rule declares no layout requirement, so there is no width
        // to solve -- and no layout solution was invented for it.
        EXPECT_TRUE(placement->layoutSolutions.empty());
      }
    }
  }
}

// The mixed-dtype epilogue maps and analyzes completely, each node's own
// operand layout is solved at that operand's width, and the fused rule the
// policy declares is applicable to this same extracted graph with its three
// boundary ports solved independently: bf16 input 32, f32 gate and result 16.
//
// The selected covering in the deterministic search is the three single-op
// rules: a finalized plan is ranked by its analysis, not by the rule's cost
// lower bound, so both coverings describe the same work. The fused rule is
// therefore checked where its per-port contract lives -- the candidate derived
// from the fixture's real extracted graph -- rather than by forcing a choice
// the search does not make.
TEST(ArmDtypeCoverage, FusedEpilogueSolvesEachPortAtItsOwnWidth) {
  constexpr llvm::StringLiteral kFusedPath =
      "test/Conversion/MicroMapping/Inputs/arm/fused_convert_silu_mul.mlir";
  for (llvm::StringRef targetName : {"arm-sve", "arm-sme"}) {
    llvm::StringRef prefix = targetName == "arm-sve" ? llvm::StringRef("sve")
                                                     : llvm::StringRef("sme");
    SCOPED_TRACE(targetName.str());
    llvm::Expected<arm_test::MappedFixture> mapped =
        arm_test::mapFixture(targetName, kFusedPath);
    ASSERT_TRUE(bool(mapped)) << llvm::toString(mapped.takeError());
    EXPECT_TRUE(mapped->analysis.complete);
    EXPECT_TRUE(mapped->analysis.incompleteReasons.empty());
    EXPECT_TRUE(mapped->plan.diagnostics.physicalComplete);
    EXPECT_TRUE(mapped->plan.diagnostics.physicalReasons.empty());
    EXPECT_TRUE(mapped->bound.unmaterialized.empty());

    const WorkloadNode *convert =
        findWorkloadNode(*mapped, "micro.vector", "convert");
    const WorkloadNode *silu =
        findWorkloadNode(*mapped, "micro.vector", "silu");
    const WorkloadNode *multiply =
        findWorkloadNode(*mapped, "micro.vector", "mul");
    ASSERT_NE(convert, nullptr);
    ASSERT_NE(silu, nullptr);
    ASSERT_NE(multiply, nullptr);

    struct PlacedSignature {
      const WorkloadNode *node;
      llvm::StringRef ruleSuffix;
      int64_t width;
    };
    for (const PlacedSignature &expected :
         {PlacedSignature{convert, "vector_convert_bf16_f32", 32},
          PlacedSignature{silu, "vector_silu_f32", 16},
          PlacedSignature{multiply, "vector_mul_f32", 16}}) {
      const PlanPlacement *placement = nullptr;
      for (const PlanPlacement &candidate : mapped->plan.placements)
        if (candidate.node == expected.node->id) {
          ASSERT_EQ(placement, nullptr);
          placement = &candidate;
        }
      ASSERT_NE(placement, nullptr) << expected.node->opName;
      EXPECT_EQ(placement->rule, (prefix + "." + expected.ruleSuffix).str());
      EXPECT_EQ(placement->computeBindings.lookup("vector_engine"), "svp");
      EXPECT_EQ(solvedWidth(*mapped, PortRef{expected.node->id,
                                             PortDirection::Input, 0}),
                expected.width);
    }

    // The fused candidate over the same extracted graph.
    std::vector<RulePatternMatch> matches =
        matchRulePatterns(mapped->graph, mapped->target->rules());
    const RulePatternMatch *fused = nullptr;
    for (const RulePatternMatch &match : matches)
      if (match.rule->id == (prefix + ".fused_convert_silu_mul").str())
        fused = &match;
    ASSERT_NE(fused, nullptr) << "the fused rule does not match the fixture";
    ASSERT_EQ(fused->coveredNodes.size(), 3u);

    LayoutContext fallback;
    fallback.rank = 2;
    fallback.elementType = "bf16";
    std::string reason;
    std::optional<MappingCandidate> candidate =
        toFusedMappingCandidate(*fused->rule, *fused, mapped->graph,
                                mapped->target->machine(), fallback, &reason);
    ASSERT_TRUE(candidate.has_value()) << reason;
    ASSERT_EQ(candidate->layoutRequirements.size(), 3u);

    struct BoundaryPort {
      PortRef port;
      llvm::StringRef elementType;
      int64_t width;
    };
    const BoundaryPort boundaries[] = {
        {PortRef{convert->id, PortDirection::Input, 0}, "bf16", 32},
        {PortRef{multiply->id, PortDirection::Input, 1}, "f32", 16},
        {PortRef{multiply->id, PortDirection::Output, 0}, "f32", 16},
    };
    for (size_t index = 0; index < 3; ++index) {
      const LayoutRequirement &requirement =
          candidate->layoutRequirements[index];
      EXPECT_EQ(requirement.elementType, boundaries[index].elementType.str());
      ASSERT_TRUE(requirement.port.has_value());
      EXPECT_TRUE(*requirement.port == boundaries[index].port);
      const LayoutDef *layout =
          mapped->target->layouts().find(requirement.layoutClass);
      ASSERT_NE(layout, nullptr);
      LayoutContext context;
      context.rank = requirement.rank;
      context.elementType = requirement.elementType;
      llvm::Expected<LayoutSolveResult> solved = solveLayout(
          *layout, mapped->target->machine(), *mapped->context, context);
      ASSERT_TRUE(bool(solved)) << llvm::toString(solved.takeError());
      EXPECT_FALSE(solved->undecided);
      ASSERT_EQ(solved->solutions.size(), 1u);
      auto value = solved->solutions.front().values.find("VW");
      ASSERT_NE(value, solved->solutions.front().values.end());
      const auto *width = std::get_if<int64_t>(&value->second);
      ASSERT_NE(width, nullptr);
      EXPECT_EQ(*width, boundaries[index].width);
    }
  }
}

// A legal MicroIR signature that the typed table does not advertise retains no
// complete plan. The refusal names the rule table, not a search cap.
TEST(ArmDtypeCoverage, UnsupportedSignaturesRetainNoPlan) {
  constexpr llvm::StringLiteral kI8Contraction =
      "test/Conversion/MicroMapping/Inputs/arm/mma_i8.mlir";
  constexpr llvm::StringLiteral kBf16Silu =
      "test/Conversion/MicroMapping/Inputs/arm/vector_silu_bf16.mlir";

  for (llvm::StringRef targetName : {"arm-sve", "arm-sme"}) {
    SCOPED_TRACE(targetName.str());
    for (llvm::StringRef path : {kI8Contraction, kBf16Silu}) {
      SCOPED_TRACE(path.str());
      llvm::Expected<arm_test::SearchedFixture> searched =
          arm_test::searchFixture(targetName, path, /*machineOverride=*/{},
                                  /*topK=*/1);
      ASSERT_TRUE(bool(searched)) << llvm::toString(searched.takeError());
      EXPECT_TRUE(searched->result.plans.empty());
      EXPECT_FALSE(searched->result.searchTruncated)
          << "no plan must be a rejection, not a cap";
      EXPECT_GT(searched->result.frontier.nodesWithoutRules, 0u);
      bool explained =
          hasDiagnostic(searched->result, DiagnosticCode::NoMatchingRule,
                        "no rule satisfies its constraints") ||
          hasDiagnostic(searched->result, DiagnosticCode::NoMatchingRule,
                        "no matching rule");
      EXPECT_TRUE(explained) << "the frontier did not name the rule table";
    }

    // f64 is outside the canonical tile element types, so the signature is
    // exercised where the advertised table actually lives: the workload graph.
    MLIRContext context;
    WorkloadGraph graph = f64VectorGraph(context);
    llvm::Expected<MappingSearchResult> result =
        searchGraph(targetName, graph, context);
    ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
    EXPECT_TRUE(result->plans.empty());
    EXPECT_FALSE(result->searchTruncated);
    EXPECT_TRUE(hasDiagnostic(*result, DiagnosticCode::NoMatchingRule,
                              "no rule satisfies its constraints") ||
                hasDiagnostic(*result, DiagnosticCode::NoMatchingRule,
                              "no matching rule"));
  }
}

// A profile with no lane fact for a dtype cannot solve the blocked layout, so
// the otherwise-supported f32 add yields no plan and the search reports the
// layout fact, not a rule match. This is the "unknown vector width" control
// from the design's negative list: the solve fails closed rather than picking
// 16 or another dtype's assignment. Because the machine query itself cannot be
// evaluated, the layout solve errors and the search propagates that error --
// no retained plan can exist.
TEST(ArmDtypeCoverage, MissingLaneWidthRetainsNoPlan) {
  constexpr llvm::StringLiteral kAddPath =
      "test/Conversion/MicroMapping/Inputs/arm/vector_add_f32.mlir";
  for (llvm::StringRef targetName : {"arm-sve", "arm-sme"}) {
    SCOPED_TRACE(targetName.str());
    llvm::Expected<std::string> shipped = shippedMachine(targetName);
    ASSERT_TRUE(bool(shipped)) << llvm::toString(shipped.takeError());
    llvm::Expected<std::string> text =
        replacedOnce(*shipped, "lanes: {f32: 16, bf16: 32, i8: 64}",
                     "lanes: {bf16: 32, i8: 64}");
    ASSERT_TRUE(bool(text)) << llvm::toString(text.takeError());
    llvm::Expected<std::string> path = arm_test::writeMachineVariant(*text);
    ASSERT_TRUE(bool(path)) << llvm::toString(path.takeError());
    llvm::FileRemover cleanup(*path);
    llvm::Expected<mlir::llk::machine::MachineModel> model =
        mlir::llk::machine::loadMachineModel(*path);
    ASSERT_TRUE(bool(model)) << llvm::toString(model.takeError());

    llvm::Expected<arm_test::SearchedFixture> searched =
        arm_test::searchFixture(targetName, kAddPath, *path, /*topK=*/1);
    ASSERT_FALSE(bool(searched))
        << "a profile with no f32 lane width produced a plan";
    const std::string message = llvm::toString(searched.takeError());
    EXPECT_NE(message.find("does not model dtype 'f32'"), std::string::npos)
        << message;
  }
}

// A same-target profile that changes one storage or routing fact must refuse
// the contraction for that fact before retention. topK=1 makes the observation
// sharp: nothing survived, and no cap was hit.
TEST(ArmFeasibility, VariantsRejectForTheirChangedFact) {
  constexpr llvm::StringLiteral kF32Path =
      "test/Conversion/MicroMapping/Inputs/arm/contraction_f32.mlir";
  constexpr llvm::StringLiteral kBf16Path =
      "test/Conversion/MicroMapping/Inputs/arm/contraction_bf16.mlir";

  for (const ArmStorageFacts &facts : kStorageFacts) {
    SCOPED_TRACE(facts.targetName);
    llvm::Expected<std::string> shipped = shippedMachine(facts.targetName);
    ASSERT_TRUE(bool(shipped)) << llvm::toString(shipped.takeError());

    // (a) Capacity below the 1024-byte live accumulator peak.
    {
      llvm::Expected<std::string> text = replacedOnce(
          *shipped, "capacity_bytes: " + std::to_string(facts.declaredCapacity),
          "capacity_bytes: 512");
      ASSERT_TRUE(bool(text)) << llvm::toString(text.takeError());
      llvm::Expected<std::string> path = arm_test::writeMachineVariant(*text);
      ASSERT_TRUE(bool(path)) << llvm::toString(path.takeError());
      llvm::FileRemover cleanup(*path);
      // Structurally valid: it is the capacity fact alone that is infeasible.
      llvm::Expected<mlir::llk::machine::MachineModel> model =
          mlir::llk::machine::loadMachineModel(*path);
      ASSERT_TRUE(bool(model)) << llvm::toString(model.takeError());

      for (llvm::StringRef input : {kF32Path, kBf16Path}) {
        SCOPED_TRACE(input.str());
        llvm::Expected<arm_test::SearchedFixture> searched =
            arm_test::searchFixture(facts.targetName, input, *path,
                                    /*topK=*/1);
        ASSERT_TRUE(bool(searched)) << llvm::toString(searched.takeError());
        EXPECT_TRUE(searched->result.plans.empty());
        EXPECT_FALSE(searched->result.searchTruncated);
        EXPECT_GT(searched->result.frontier.plansRejectedByCapacity, 0u);
        EXPECT_TRUE(hasDiagnostic(searched->result,
                                  DiagnosticCode::MemoryCapacityExceeded));
      }
    }

    // (b) The accumulator memory and the links into and out of it are gone.
    // The profile is still well formed, so the refusal is the target's declared
    // need for that memory kind rather than a YAML parse error.
    {
      llvm::Expected<std::string> text =
          removedEntry(*shipped, facts.accumulator);
      ASSERT_TRUE(bool(text)) << llvm::toString(text.takeError());
      text = removedEntry(*text, facts.intoAccumulatorLink);
      ASSERT_TRUE(bool(text)) << llvm::toString(text.takeError());
      text = removedEntry(*text, facts.outOfAccumulatorLink);
      ASSERT_TRUE(bool(text)) << llvm::toString(text.takeError());
      llvm::Expected<std::string> path = arm_test::writeMachineVariant(*text);
      ASSERT_TRUE(bool(path)) << llvm::toString(path.takeError());
      llvm::FileRemover cleanup(*path);
      llvm::Expected<mlir::llk::machine::MachineModel> model =
          mlir::llk::machine::loadMachineModel(*path);
      ASSERT_TRUE(bool(model)) << llvm::toString(model.takeError());

      llvm::Expected<arm_test::SearchedFixture> searched =
          arm_test::searchFixture(facts.targetName, kF32Path, *path,
                                  /*topK=*/1);
      ASSERT_FALSE(bool(searched))
          << "a profile without accumulator storage produced a plan";
      const std::string message = llvm::toString(searched.takeError());
      EXPECT_NE(message.find("no memory of kind 'acc'"), std::string::npos)
          << message;
    }

    // (c) Only the SRAM -> accumulator transfer is removed. Storage resolution
    // can no longer place the initialized-accumulator movement.
    {
      llvm::Expected<std::string> text =
          removedEntry(*shipped, facts.intoAccumulatorLink);
      ASSERT_TRUE(bool(text)) << llvm::toString(text.takeError());
      llvm::Expected<std::string> path = arm_test::writeMachineVariant(*text);
      ASSERT_TRUE(bool(path)) << llvm::toString(path.takeError());
      llvm::FileRemover cleanup(*path);
      llvm::Expected<mlir::llk::machine::MachineModel> model =
          mlir::llk::machine::loadMachineModel(*path);
      ASSERT_TRUE(bool(model)) << llvm::toString(model.takeError());

      llvm::Expected<arm_test::SearchedFixture> searched =
          arm_test::searchFixture(facts.targetName, kF32Path, *path,
                                  /*topK=*/1);
      ASSERT_TRUE(bool(searched)) << llvm::toString(searched.takeError());
      EXPECT_TRUE(searched->result.plans.empty());
      EXPECT_FALSE(searched->result.searchTruncated);
      // Removing the link leaves the initialized-accumulator movement without
      // a hop. A mapped movement with no routed hop falls back to the machine
      // link between its endpoint kinds; with that link gone there is none, and
      // the strict event stream refuses the abstract transfer pool. Either
      // refusal is a property of this plan, reported before retention.
      EXPECT_TRUE(
          hasDiagnostic(searched->result, DiagnosticCode::NoMemoryRoute) ||
          hasDiagnostic(searched->result,
                        DiagnosticCode::UnsupportedMaterialization,
                        "transfer engine") ||
          diagnosticsMention(searched->result, {"sram.0", facts.accumulator}))
          << describeDiagnostics(searched->result);
    }
  }

  // (d) A supported signature whose tiles request a memory kind no ARM profile
  // models. Storage resolution must refuse it, never remap to SRAM.
  constexpr llvm::StringLiteral kRfPath =
      "test/Conversion/MicroMapping/Inputs/arm/vector_add_rf.mlir";
  for (llvm::StringRef targetName : {"arm-sve", "arm-sme"}) {
    SCOPED_TRACE(targetName.str());
    llvm::Expected<arm_test::SearchedFixture> searched =
        arm_test::searchFixture(targetName, kRfPath, /*machineOverride=*/{},
                                /*topK=*/1);
    ASSERT_TRUE(bool(searched)) << llvm::toString(searched.takeError());
    EXPECT_TRUE(searched->result.plans.empty());
    EXPECT_FALSE(searched->result.searchTruncated);
    EXPECT_TRUE(diagnosticsMention(searched->result, {"'rf' memory"}))
        << "the frontier did not name the absent memory kind";
  }
}

// A tampered selected compute identity is a rejection, never a silent
// replacement with some other engine. Each case changes one fact on a copy of
// the selected plan and runs the same complete-plan evaluator the search uses.
TEST(ArmComputeIdentity, TamperedSelectionIsNeverReplaced) {
  constexpr llvm::StringLiteral kF32Path =
      "test/Conversion/MicroMapping/Inputs/arm/contraction_f32.mlir";

  for (const ArmStorageFacts &facts : kStorageFacts) {
    SCOPED_TRACE(facts.targetName);
    llvm::Expected<std::string> shipped = shippedMachine(facts.targetName);
    ASSERT_TRUE(bool(shipped)) << llvm::toString(shipped.takeError());

    // A valid profile with one extra engine attached to a different executor.
    // The search still selects `svp`/`za`; this node exists only to be named by
    // a tampered plan, where its attachment must be detected.
    llvm::Expected<std::string> variant =
        insertedBefore(*shipped, "\ntransfer_engines:",
                       "\n"
                       R"(  - id: svp.remote
    kind: vector_engine
    refines: [vector]
    attached_to: cluster.0
    element_types: [f32, bf16, i8]
    accumulator_dtypes: [f32]
    shapes: [[16], [32], [64]]
    lanes: {f32: 16, bf16: 32, i8: 64}
    issue_cycles: 1
    latency_cycles: 4
    throughput_per_cycle: 32
    concurrency: 8
  - id: za.remote
    kind: matrix_engine
    refines: [matrix]
    attached_to: cluster.0
    element_types: [f32, bf16]
    accumulator_dtypes: [f32]
    shapes: [[16, 16, 16]]
    issue_cycles: 1
    latency_cycles: 5
    throughput_per_cycle: 512
    concurrency: 8
)");
    ASSERT_TRUE(bool(variant)) << llvm::toString(variant.takeError());
    llvm::Expected<std::string> machinePath =
        arm_test::writeMachineVariant(*variant);
    ASSERT_TRUE(bool(machinePath)) << llvm::toString(machinePath.takeError());
    llvm::FileRemover cleanup(*machinePath);

    llvm::Expected<arm_test::MappedFixture> mapped =
        arm_test::mapFixture(facts.targetName, kF32Path, *machinePath);
    ASSERT_TRUE(bool(mapped)) << llvm::toString(mapped.takeError());
    const PlanPlacement *placement = arm_test::contractionPlacement(*mapped);
    ASSERT_NE(placement, nullptr);
    ASSERT_EQ(placement->computeRequirements.size(), 1u);
    const std::string kind = placement->computeRequirements.front();
    const std::string engine = placement->computeBindings.lookup(kind);
    ASSERT_FALSE(engine.empty());

    const std::string otherKind =
        kind == "vector_engine" ? "matrix_engine" : "vector_engine";
    const std::string otherEngine = otherKind == "matrix_engine" ? "za" : "svp";

    auto expectRejected = [&](const CoveringPlan &plan,
                              llvm::StringRef fragment) {
      llvm::Expected<CompletePlanEvaluation> evaluated =
          arm_test::evaluatePlan(*mapped, plan);
      ASSERT_TRUE(bool(evaluated)) << llvm::toString(evaluated.takeError());
      EXPECT_FALSE(evaluated->plan.has_value())
          << "a tampered identity retained a complete plan";
      ASSERT_TRUE(evaluated->rejection.has_value());
      EXPECT_EQ(evaluated->rejection->code,
                DiagnosticCode::UnsupportedMaterialization);
      EXPECT_NE(evaluated->rejection->message.find(fragment.str()),
                std::string::npos)
          << evaluated->rejection->message;
    };

    // Control: the untouched plan is accepted, so the rejections below are
    // caused by the tampering and not by an evaluator that refuses this shape.
    {
      llvm::Expected<CompletePlanEvaluation> accepted =
          arm_test::evaluatePlan(*mapped, mapped->plan);
      ASSERT_TRUE(bool(accepted)) << llvm::toString(accepted.takeError());
      EXPECT_TRUE(accepted->plan.has_value());
      EXPECT_FALSE(accepted->rejection.has_value());
    }

    // Unknown engine id.
    {
      CoveringPlan tampered = mapped->plan;
      for (PlanPlacement &candidate : tampered.placements)
        if (candidate.node == placement->node)
          candidate.computeBindings[kind] = "ghost.engine";
      expectRejected(tampered, "ghost.engine");
    }

    // Two selected engines of different kinds: the plan cannot mean both.
    {
      CoveringPlan tampered = mapped->plan;
      for (PlanPlacement &candidate : tampered.placements)
        if (candidate.node == placement->node) {
          candidate.computeBindings[kind] = engine;
          candidate.computeBindings[otherKind] = otherEngine;
        }
      expectRejected(tampered, "selects both");
    }

    // A real engine of the required kind that is not attached to the selected
    // executor.
    {
      const std::string foreign =
          kind == "matrix_engine" ? "za.remote" : "svp.remote";
      CoveringPlan tampered = mapped->plan;
      for (PlanPlacement &candidate : tampered.placements)
        if (candidate.node == placement->node)
          candidate.computeBindings[kind] = foreign;
      expectRejected(tampered, "attached");
    }
  }
}
