//===- framework_primitives.cpp - Composed vector/reduction acceptance ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework C3 (issue #159). Two primitive paths are checked end to end:
//
//   * a direct 8x8 -> 8 sum reduction, which is rank-neutral through static
//     analysis and selected lowering but outside the rank-two mapped ABI; and
//   * an add -> multiply composition, which is rank-two and executes
//     numerically on the reference backend everywhere and on the selected AVX2
//     backend on a supported host.
//
// The two claims stay separate: selected lowering is IR evidence, and numeric
// execution is what makes the selected width a computation rather than a
// contract. A reference run is never counted as selected evidence.

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Perf/MicroDAG.h"
#include "LLK/Perf/SelectedKernelAnalysis.h"
#include "LLK/Runtime/MappedExecutable.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "CompletePlanEvaluation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

// The Micro attribute, type and op classes. Only the tile type is read here,
// but its layout attribute is part of the generated type definition.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

using mlir::llk::mapping::BindContract;
using mlir::llk::mapping::CoveringPlan;
using mlir::llk::mapping::MappingSearchOptions;
using mlir::llk::mapping::MappingSearchResult;
using mlir::llk::mapping::MappingTarget;
using mlir::llk::mapping::SearchMode;

mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  return registry;
}

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
parseFixture(mlir::MLIRContext &context, llvm::StringRef name) {
  std::string path = std::string(LLK_SOURCE_DIR) +
                     "/test/Execution/Inputs/framework/" + name.str();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
      llvm::MemoryBuffer::getFile(path);
  if (!input)
    return llvm::createStringError(input.getError(), "cannot read " + path);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>((*input)->getBuffer(), &context);
  if (!module)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot parse " + path);
  return module;
}

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

struct DirectPlan {
  mlir::llk::mapping::WorkloadGraph graph;
  MappingSearchResult search;
  CoveringPlan plan;
};

/// Searches one directly authored MicroIR fixture with the shared complete-plan
/// evaluator, so the retained plan has a strict analysis of the kernel it binds.
llvm::Expected<DirectPlan> searchFixture(mlir::ModuleOp module,
                                         MappingTarget &target,
                                         llvm::StringRef symbol) {
  llvm::Expected<mlir::Operation *> kernel =
      mlir::llk::perf::findMicroKernel(module, symbol);
  if (!kernel)
    return kernel.takeError();
  llvm::Expected<mlir::llk::mapping::WorkloadGraph> graph =
      mlir::llk::mapping::extractWorkloadGraph(*kernel);
  if (!graph)
    return graph.takeError();

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 1;
  options.requireCompleteEvaluation = true;
  options.evaluateCompletePlan = [&](const CoveringPlan &proposal) {
    return mlir::llk::mapping::evaluateCompletePlan(
        module, *graph, target, proposal, BindContract::Partial, std::nullopt,
        mlir::llk::mapping::AnalysisRequirement::Complete);
  };
  mlir::llk::mapping::LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  mlir::llk::mapping::CoveringSearch search(
      *graph, target, *module.getContext(), layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  if (result->plans.empty()) {
    std::string reason = "the fixture produced no complete plan";
    if (!result->frontier.diagnostics.empty())
      reason += ": " + result->frontier.diagnostics.front().message;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), reason);
  }
  DirectPlan plan;
  plan.graph = std::move(*graph);
  plan.plan = result->plans.front();
  plan.search = std::move(*result);
  return plan;
}

llvm::Expected<llk::MappedCompilation>
compileFixture(mlir::ModuleOp module, MappingTarget &target,
               llvm::StringRef symbol, const CoveringPlan &plan,
               llk::MappedBackend backend, llk::MappedStop stop) {
  llk::MappedCompileOptions options;
  options.entrySymbol = symbol.str();
  options.backend = backend;
  options.stop = stop;
  options.requireExecutable = stop == llk::MappedStop::Executable;
  return llk::compileMappedKernel(module, target, plan, options);
}

std::unique_ptr<MappingTarget> avx2Target() {
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  if (!target) {
    ADD_FAILURE() << llvm::toString(target.takeError());
    return nullptr;
  }
  return std::move(*target);
}

bool avx2DisabledByTestOverride() {
  const char *disabled = std::getenv("LLK_TEST_DISABLE_AVX2");
  return disabled && std::strcmp(disabled, "1") == 0;
}

bool selectedAvx2Available() {
  if (avx2DisabledByTestOverride())
    return false;
  const std::string triple = llvm::sys::getDefaultTargetTriple();
  if (triple.rfind("x86_64-", 0) != 0)
    return false;
  const auto features = llvm::sys::getHostCPUFeatures();
  auto avx2 = features.find("avx2");
  return avx2 != features.end() && avx2->second;
}

/// One rank-two f32 ABI buffer.
struct FloatBuffer {
  std::vector<float> storage;
  llk::InvocationBuffer2D buffer;

  explicit FloatBuffer(size_t count, float fill)
      : storage(count, fill),
        buffer{MemRef2D{storage.data(), storage.data(), 0, 8, 8, 8, 1},
               llk::InvocationElementType::F32,
               storage.size() * sizeof(float)} {}
};

/// Runs add -> multiply and checks every element against the scalar formula.
void expectAddMulResult(llk::MappedExecutable &executable) {
  constexpr size_t kCount = 64;
  FloatBuffer left(kCount, 0.0f);
  FloatBuffer right(kCount, 0.0f);
  for (size_t i = 0; i < kCount; ++i) {
    left.storage[i] = static_cast<float>(i) / 8.0f;
    right.storage[i] = static_cast<float>(63 - i) / 16.0f;
  }
  const std::vector<float> originalLeft = left.storage;
  const std::vector<float> originalRight = right.storage;

  FloatBuffer output(kCount, -1.0f);
  llvm::Error error =
      executable.invoke({left.buffer, right.buffer}, {output.buffer});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  for (size_t i = 0; i < kCount; ++i) {
    const float expected =
        (left.storage[i] + right.storage[i]) * left.storage[i];
    EXPECT_NEAR(output.storage[i], expected, 1.0e-5f) << "element " << i;
  }
  EXPECT_EQ(left.storage, originalLeft);
  EXPECT_EQ(right.storage, originalRight);
}

} // namespace

//===----------------------------------------------------------------------===//
// Reduction: static analysis, selected lowering, invocation refusal
//===----------------------------------------------------------------------===//

TEST(FrameworkPrimitives, RowSumVerifiesAndAnalyzesStatically) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseFixture(context, "reduce.mlir");
  ASSERT_TRUE(static_cast<bool>(module)) << llvm::toString(module.takeError());
  mlir::ModuleOp moduleOp = **module;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(moduleOp)));

  mlir::Operation *reduce = nullptr;
  moduleOp->walk([&](mlir::Operation *op) {
    if (!reduce && op->getName().getStringRef() == "micro.reduce")
      reduce = op;
  });
  ASSERT_NE(reduce, nullptr);
  auto resultTile =
      mlir::dyn_cast<mlir::micro::TileType>(reduce->getResult(0).getType());
  ASSERT_TRUE(resultTile);
  ASSERT_EQ(resultTile.getShape().size(), 1u);
  EXPECT_EQ(resultTile.getShape()[0], 8);
  EXPECT_TRUE(resultTile.getElementType().isF32());

  auto target = avx2Target();
  ASSERT_NE(target, nullptr);
  auto searched = searchFixture(moduleOp, *target, "reduce_sum");
  ASSERT_TRUE(static_cast<bool>(searched))
      << llvm::toString(searched.takeError());
  ASSERT_EQ(searched->graph.getNodes().size(), 1u);
  EXPECT_EQ(searched->graph.getNodes().front().opName, "micro.reduce");
  ASSERT_EQ(searched->plan.placements.size(), 1u);
  const auto &placement = searched->plan.placements.front();
  EXPECT_EQ(placement.rule, "avx2.reduce_sum_f32");
  ASSERT_NE(placement.computeBindings.find("vector_engine"),
            placement.computeBindings.end());
  EXPECT_NE(target->machine().findCompute(
                placement.computeBindings.lookup("vector_engine")),
            nullptr);
  EXPECT_TRUE(searched->plan.diagnostics.physicalComplete);

  // Bind the selected plan and run the *strict* shared analysis over the
  // materialized kernel: completeness is the contract, not an optimistic flag.
  llk::MappedCompileOptions compile;
  compile.stop = llk::MappedStop::MappedMicro;
  compile.requireExecutable = false;
  auto bound =
      llk::compileMappedKernel(moduleOp, *target, searched->plan, compile);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  auto kernel =
      mlir::llk::perf::findMicroKernel(bound->module.get(), "reduce_sum");
  ASSERT_TRUE(static_cast<bool>(kernel)) << llvm::toString(kernel.takeError());
  auto analysis = mlir::llk::perf::analyzeSelectedKernel(
      *kernel, target->machine(), /*requireComplete=*/true, searched->plan,
      searched->graph);
  ASSERT_TRUE(static_cast<bool>(analysis))
      << llvm::toString(analysis.takeError());
  EXPECT_TRUE(analysis->complete);
  EXPECT_DOUBLE_EQ(analysis->cost.latencyCycles,
                   searched->plan.totalCost.latencyCycles);

  // The reduction processes all 64 input elements; the output tile has 8.
  // The scheduled event is where the work count is charged.
  const mlir::llk::mapping::PlanCostEvent *reduceEvent = nullptr;
  for (const auto &event : analysis->events.events)
    if (event.event.kind == mlir::llk::mapping::CostEventKind::Compute)
      reduceEvent = &event;
  ASSERT_NE(reduceEvent, nullptr);
  EXPECT_EQ(reduceEvent->workItems, 64u);
  EXPECT_EQ(reduceEvent->event.resource,
            placement.computeBindings.lookup("vector_engine"));
}

TEST(FrameworkPrimitives, RowSumSelectedLoweringRealizesVectorReduction) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseFixture(context, "reduce.mlir");
  ASSERT_TRUE(static_cast<bool>(module)) << llvm::toString(module.takeError());
  mlir::ModuleOp moduleOp = **module;
  auto target = avx2Target();
  ASSERT_NE(target, nullptr);
  auto searched = searchFixture(moduleOp, *target, "reduce_sum");
  ASSERT_TRUE(static_cast<bool>(searched))
      << llvm::toString(searched.takeError());

  auto lowered = compileFixture(moduleOp, *target, "reduce_sum", searched->plan,
                                llk::MappedBackend::SelectedTarget,
                                llk::MappedStop::Lowered);
  ASSERT_TRUE(static_cast<bool>(lowered))
      << llvm::toString(lowered.takeError());
  EXPECT_GT(lowered->targetLowered, 0u);
  EXPECT_GT(lowered->backendGroupsRealized, 0u);
  EXPECT_EQ(lowered->referenceGroupsLowered, 0u);
  const std::string ir = printModule(*lowered->module);
  EXPECT_TRUE(ir.find("vector.multi_reduction") != std::string::npos ||
              ir.find("vector.reduction") != std::string::npos)
      << ir;
  EXPECT_NE(ir.find("vector<8xf32>"), std::string::npos) << ir;
  EXPECT_EQ(ir.find("micro.reduce"), std::string::npos) << ir;
}

TEST(FrameworkPrimitives, RowSumExecutablePreparationRefusesRankOneAbi) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseFixture(context, "reduce.mlir");
  ASSERT_TRUE(static_cast<bool>(module)) << llvm::toString(module.takeError());
  mlir::ModuleOp moduleOp = **module;
  auto target = avx2Target();
  ASSERT_NE(target, nullptr);
  auto searched = searchFixture(moduleOp, *target, "reduce_sum");
  ASSERT_TRUE(static_cast<bool>(searched))
      << llvm::toString(searched.takeError());

  auto executable = compileFixture(
      moduleOp, *target, "reduce_sum", searched->plan,
      llk::MappedBackend::SelectedTarget, llk::MappedStop::Executable);
  ASSERT_FALSE(static_cast<bool>(executable));
  const std::string message = llvm::toString(executable.takeError());
  EXPECT_NE(message.find("runtime_capability:"), std::string::npos) << message;
  EXPECT_NE(message.find("rank-two"), std::string::npos) << message;
}

TEST(FrameworkPrimitives, InvalidReductionShapesFailTheVerifier) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
      llvm::MemoryBuffer::getFile(
          std::string(LLK_SOURCE_DIR) +
          "/test/Execution/Inputs/framework/reduce.mlir");
  ASSERT_TRUE(static_cast<bool>(input)) << input.getError().message();
  const std::string source = (*input)->getBuffer().str();

  auto verifyVariant = [&](std::string text, llvm::StringRef what) {
    SCOPED_TRACE(what.str());
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    // The kernel verifier may reject the variant during parsing (the yielded
    // type is checked against the declared result) or at explicit verify; both
    // are the existing verifier refusing it, and neither may produce a module
    // that verifies.
    ASSERT_TRUE(!module || mlir::failed(mlir::verify(*module)))
        << "the invalid reduction must be refused";
  };

  std::string badAxis = source;
  badAxis.replace(badAxis.find("axis = 1 : i64"), 14, "axis = 2 : i64");
  verifyVariant(badAxis, "axis out of range");

  std::string badShape = source;
  for (size_t at = badShape.find("!micro.tile<8xf32"); at != std::string::npos;
       at = badShape.find("!micro.tile<8xf32")) {
    badShape.replace(at, std::strlen("!micro.tile<8xf32"),
                     "!micro.tile<8x1xf32");
  }
  verifyVariant(badShape, "result keeps the reduced axis");

  std::string badDtype = source;
  for (size_t at = badDtype.find("!micro.tile<8xf32"); at != std::string::npos;
       at = badDtype.find("!micro.tile<8xf32")) {
    badDtype.replace(at, std::strlen("!micro.tile<8xf32"), "!micro.tile<8xi32");
  }
  verifyVariant(badDtype, "result dtype differs");
}

//===----------------------------------------------------------------------===//
// Composed add -> multiply: reference and selected numeric execution
//===----------------------------------------------------------------------===//

TEST(FrameworkPrimitives, AddMulReferenceExecutesLiteralFormula) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseFixture(context, "add_mul.mlir");
  ASSERT_TRUE(static_cast<bool>(module)) << llvm::toString(module.takeError());
  mlir::ModuleOp moduleOp = **module;
  auto target = avx2Target();
  ASSERT_NE(target, nullptr);
  auto searched = searchFixture(moduleOp, *target, "add_mul");
  ASSERT_TRUE(static_cast<bool>(searched))
      << llvm::toString(searched.takeError());
  ASSERT_EQ(searched->graph.getNodes().size(), 2u);

  auto compiled = compileFixture(moduleOp, *target, "add_mul", searched->plan,
                                 llk::MappedBackend::Reference,
                                 llk::MappedStop::Executable);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  ASSERT_TRUE(compiled->executable);
  expectAddMulResult(*compiled->executable);
}

TEST(FrameworkPrimitives, SelectedAvx2AddMulExecutesLiteralFormula) {
  if (!selectedAvx2Available()) {
#ifdef LLK_REQUIRE_SELECTED_TARGET
    if (avx2DisabledByTestOverride())
      FAIL() << "test override disabled required selected AVX2 execution";
    else
      FAIL() << "required selected AVX2 execution needs an x86_64 AVX2 host";
#else
    GTEST_SKIP() << "selected AVX2 execution requires an x86_64 AVX2 host";
#endif
  }

  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseFixture(context, "add_mul.mlir");
  ASSERT_TRUE(static_cast<bool>(module)) << llvm::toString(module.takeError());
  mlir::ModuleOp moduleOp = **module;
  auto target = avx2Target();
  ASSERT_NE(target, nullptr);
  auto searched = searchFixture(moduleOp, *target, "add_mul");
  ASSERT_TRUE(static_cast<bool>(searched))
      << llvm::toString(searched.takeError());

  auto compiled = compileFixture(moduleOp, *target, "add_mul", searched->plan,
                                 llk::MappedBackend::SelectedTarget,
                                 llk::MappedStop::Executable);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  ASSERT_TRUE(compiled->executable);
  EXPECT_GT(compiled->backendGroupsRealized, 0u);
  EXPECT_EQ(compiled->referenceGroupsLowered, 0u);
  expectAddMulResult(*compiled->executable);

  // The lowered form is separate evidence: it names the width the selected
  // backend reached, which the callable ABI never exposes.
  auto lowered = compileFixture(moduleOp, *target, "add_mul", searched->plan,
                                llk::MappedBackend::SelectedTarget,
                                llk::MappedStop::Lowered);
  ASSERT_TRUE(static_cast<bool>(lowered))
      << llvm::toString(lowered.takeError());
  const std::string ir = printModule(*lowered->module);
  EXPECT_NE(ir.find("vector<8xf32>"), std::string::npos) << ir;
}

int main(int argc, char **argv) {
  bool requireSelectedTarget = false;
  for (int index = 1; index < argc;) {
    if (std::strcmp(argv[index], "--require-selected-target") == 0) {
      requireSelectedTarget = true;
      for (int move = index; move + 1 < argc; ++move)
        argv[move] = argv[move + 1];
      --argc;
      continue;
    }
    ++index;
  }

#ifdef LLK_REQUIRE_SELECTED_TARGET
  if (!requireSelectedTarget) {
    std::fprintf(stderr, "FrameworkPrimitivesAVX2Test requires "
                         "--require-selected-target\n");
    return 2;
  }
#else
  if (requireSelectedTarget) {
    std::fprintf(stderr, "--require-selected-target is only valid for the "
                         "selected-target executable\n");
    return 2;
  }
#endif

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
