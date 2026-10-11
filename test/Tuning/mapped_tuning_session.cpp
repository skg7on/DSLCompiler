#include "LLK/Conversion/MicroMapping/MappedTuningSession.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include "gtest/gtest.h"

#include <algorithm>

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

mlir::llk::perf::SearchSpace oneCandidateSpace() {
  using namespace mlir::llk::perf;
  SearchSpace space;
  space.name = "mapped_test";
  space.workload = "matmul";
  space.params = {
      {"BM",
       "integer",
       {SearchChoice(int64_t{8}), SearchChoice(int64_t{256}),
        SearchChoice(int64_t{3})}},
      {"BN", "integer", {SearchChoice(int64_t{16})}},
      {"BK", "integer", {SearchChoice(int64_t{32})}},
      {"fragment", "fragment_shape", {SearchChoice("8x16x32")}},
      {"tail", "tail_policy", {SearchChoice("none")}},
      {"pipeline_stages", "integer", {SearchChoice(int64_t{1})}},
      {"vector_width", "integer", {SearchChoice(int64_t{8})}},
      {"num_threads", "integer", {SearchChoice(int64_t{1})}},
      {"grain_size", "integer", {SearchChoice(int64_t{1})}},
  };
  space.constraints = {
      {ConstraintKind::MmaCompatible, {"BM", "BN", "BK"}, {}},
      {ConstraintKind::AccCapacity, {"BM", "BN"}, {}},
  };
  return space;
}

class AnalysisOnlyTarget final : public mlir::llk::mapping::MappingTarget {
public:
  explicit AnalysisOnlyTarget(const mlir::llk::mapping::MappingTarget &base)
      : base_(base) {}
  llvm::StringRef name() const override { return base_.name(); }
  const mlir::llk::machine::MachineModel &machine() const override {
    return base_.machine();
  }
  const mlir::llk::mapping::LayoutRegistry &layouts() const override {
    return base_.layouts();
  }
  const mlir::llk::mapping::RuleRegistry &rules() const override {
    return base_.rules();
  }
  bool isKnownEmitter(llvm::StringRef key) const override {
    return base_.isKnownEmitter(key);
  }
  std::unique_ptr<mlir::llk::mapping::TargetEmitter>
  createEmitter(llvm::StringRef key) const override {
    return base_.createEmitter(key);
  }

private:
  const mlir::llk::mapping::MappingTarget &base_;
};

mlir::OwningOpRef<mlir::ModuleOp> parseSource(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral sourceText = R"mlir(
    module {
      func.func @matmul(%a: tensor<8x32xf32>, %b: tensor<32x16xf32>,
                        %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
        %y = llk.matmul ins(%a, %b : tensor<8x32xf32>, tensor<32x16xf32>)
            outs(%init : tensor<8x16xf32>)
            {accumulator_type = f32, math_mode = #llk.math_mode<strict>}
            -> tensor<8x16xf32>
        return %y : tensor<8x16xf32>
      }
    }
  )mlir";
  return mlir::parseSourceString<mlir::ModuleOp>(sourceText, &context);
}

mlir::llk::perf::WorkloadShape testShape() {
  mlir::llk::perf::WorkloadShape shape;
  shape.M = 8;
  shape.N = 16;
  shape.K = 32;
  shape.inputDType = "f32";
  shape.weightDType = "f32";
  shape.accumulatorDType = "f32";
  shape.outputDType = "f32";
  return shape;
}

TEST(MappedTuningSessionTest, RanksARealMappedSourceCandidate) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = parseSource(context);
  ASSERT_TRUE(source);
  auto target = mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  mlir::llk::perf::WorkloadShape shape = testShape();
  auto space = oneCandidateSpace();
  auto generated = mlir::llk::perf::generateCandidates(space, {});
  ASSERT_EQ(generated.size(), 3u);
  mlir::llk::tuning::CandidateInstantiationOptions sourceOptions;
  sourceOptions.sourceSymbol = "matmul";
  sourceOptions.sourceRootOrdinal = 0;
  sourceOptions.machine = &(*target)->machine();
  auto instance = mlir::llk::tuning::instantiateCandidate(
      *source, space, generated.front(), shape, sourceOptions);
  ASSERT_TRUE(bool(instance)) << llvm::toString(instance.takeError());
  ASSERT_TRUE(instance->kernel->getAttr("source_argument_memories"));
  EXPECT_EQ(instance->kernel->getAttrOfType<mlir::StringAttr>("source_symbol")
                .getValue(),
            "matmul");
  auto kernelType = mlir::dyn_cast<mlir::FunctionType>(
      instance->kernel->getAttrOfType<mlir::TypeAttr>("function_type")
          .getValue());
  ASSERT_TRUE(kernelType);
  ASSERT_EQ(kernelType.getNumInputs(), 3u);
  EXPECT_EQ(mlir::cast<mlir::ShapedType>(kernelType.getInput(0)).getDimSize(1),
            32);
  EXPECT_EQ(mlir::cast<mlir::ShapedType>(kernelType.getInput(1)).getDimSize(0),
            32);
  EXPECT_EQ(mlir::cast<mlir::ShapedType>(kernelType.getInput(2)).getDimSize(0),
            8);
  mlir::llk::tuning::MappedTuningOptions options;
  options.source.sourceSymbol = "matmul";
  options.source.sourceRootOrdinal = 0;
  options.generator.mode = mlir::llk::perf::SearchMode::Grid;
  options.mapping.mode = mlir::llk::mapping::SearchMode::Exact;
  options.mapping.topK = 4;
  auto report = mlir::llk::tuning::runMappedTuningSession(*source, space, shape,
                                                          **target, options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  EXPECT_EQ(report->generated, 3u);
  ASSERT_EQ(report->ranked.size(), 1u)
      << (report->rejected.empty() ? ""
                                   : report->rejected.front().rejectionReason);
  EXPECT_EQ(report->rejected.size(), 2u);
  EXPECT_EQ(report->rejectedByLegality, 2u);
  bool sawMmaRejection = false;
  bool sawCapacityRejection = false;
  for (const auto &rejected : report->rejected) {
    sawMmaRejection |=
        rejected.rejectionReason.find("mma_compatible") != std::string::npos;
    sawCapacityRejection |=
        rejected.rejectionReason.find("acc_capacity") != std::string::npos;
  }
  EXPECT_TRUE(sawMmaRejection) << report->rejected[0].rejectionReason;
  EXPECT_TRUE(sawCapacityRejection) << report->rejected[1].rejectionReason;
  const auto &selected = report->ranked.front();
  EXPECT_TRUE(selected.executable);
  EXPECT_TRUE(selected.materializationReady);
  EXPECT_NE(selected.plan.sourceBindingHash, 0u);
  EXPECT_NE(selected.sourceGraphHash, 0u);
  EXPECT_EQ(selected.plan.sourceBindingHash, selected.bindingHash);
  EXPECT_FALSE(selected.planReport.empty());
  EXPECT_NE(selected.planReport.find(selected.ranking.candidate.id),
            std::string::npos);
  EXPECT_EQ(selected.ranking.metrics.predictedCycles,
            static_cast<uint64_t>(selected.plan.totalCost.latencyCycles));
}

TEST(MappedTuningSessionTest, AnalysisOnlyTargetsDoNotClaimExecutableSupport) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = parseSource(context);
  ASSERT_TRUE(source);
  auto base = mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(base)) << llvm::toString(base.takeError());
  AnalysisOnlyTarget target(**base);

  mlir::llk::tuning::MappedTuningOptions options;
  options.source.sourceSymbol = "matmul";
  options.mapping.mode = mlir::llk::mapping::SearchMode::Exact;
  auto shape = testShape();
  auto missingBackend = mlir::llk::tuning::runMappedTuningSession(
      *source, oneCandidateSpace(), shape, target, options);
  ASSERT_TRUE(bool(missingBackend))
      << llvm::toString(missingBackend.takeError());
  EXPECT_EQ(missingBackend->generated, 3u);
  EXPECT_EQ(missingBackend->rejectedByCompile, 3u);
  EXPECT_TRUE(missingBackend->ranked.empty());
  EXPECT_TRUE(std::all_of(
      missingBackend->rejected.begin(), missingBackend->rejected.end(),
      [](const auto &candidate) {
        return candidate.rejectionReason.starts_with("backend_unavailable:");
      }));

  options.executable = false;
  auto analysis = mlir::llk::tuning::runMappedTuningSession(
      *source, oneCandidateSpace(), shape, target, options);
  ASSERT_TRUE(bool(analysis)) << llvm::toString(analysis.takeError());
  ASSERT_EQ(analysis->ranked.size(), 1u)
      << (analysis->rejected.empty()
              ? ""
              : analysis->rejected.front().rejectionReason);
  EXPECT_FALSE(analysis->ranked.front().executable);
  EXPECT_TRUE(analysis->ranked.front().materializationReady);
}

} // namespace
