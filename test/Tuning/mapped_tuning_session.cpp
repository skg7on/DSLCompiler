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

  auto measuredSpace = oneCandidateSpace();
  measuredSpace.objective.primaryMetric = "measured_ns";
  auto missingMeasurement = mlir::llk::tuning::runMappedTuningSession(
      *source, measuredSpace, testShape(), target, options);
  ASSERT_FALSE(bool(missingMeasurement));
  EXPECT_NE(llvm::toString(missingMeasurement.takeError())
                .find("requires a mapped measurement provider"),
            std::string::npos);
}

TEST(MappedTuningSessionTest, MeasuresExactExecutableAndKeepsIdentity) {
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

  using namespace mlir::llk::tuning;
  MappedTuningOptions options;
  options.source.sourceSymbol = "matmul";
  options.source.sourceRootOrdinal = 0;
  options.mapping.mode = mlir::llk::mapping::SearchMode::Exact;
  options.mapping.topK = 4;
  // The ARM test host can exercise the exact mapped executable contract with
  // the reference backend. x86 exercises the target's selected AVX2 backend.
#if defined(__x86_64__) || defined(_M_X64)
  options.backend = ::llk::MappedBackend::SelectedTarget;
#else
  options.backend = ::llk::MappedBackend::Reference;
#endif
  bool sawExecutable = false;
  options.measurement.inputs = [](const ::llk::KernelAbi &abi)
      -> llvm::Expected<OwnedInvocationBuffers> {
    OwnedInvocationBuffers owned;
    auto allocate = [&](const ::llk::KernelAbi::Port &port,
                        bool initialize) -> ::llk::InvocationBuffer2D {
      const size_t count = static_cast<size_t>(port.shape[0] * port.shape[1]);
      owned.storage.emplace_back(count * sizeof(float));
      auto *data = reinterpret_cast<float *>(owned.storage.back().data());
      for (size_t i = 0; i < count; ++i)
        data[i] = initialize ? 3.0f : 0.0f;
      return {{data, data, 0, port.shape[0], port.shape[1], port.shape[1], 1},
              ::llk::InvocationElementType::F32,
              count * sizeof(float)};
    };
    for (size_t i = 0; i < abi.inputs.size(); ++i) {
      auto buffer = allocate(abi.inputs[i], i == 2);
      if (i == 0)
        std::fill_n(static_cast<float *>(buffer.descriptor.aligned),
                    buffer.descriptor.size0 * buffer.descriptor.size1, 1.0f);
      if (i == 1)
        std::fill_n(static_cast<float *>(buffer.descriptor.aligned),
                    buffer.descriptor.size0 * buffer.descriptor.size1, 2.0f);
      owned.inputs.push_back(buffer);
    }
    for (const auto &port : abi.outputs)
      owned.outputs.push_back(allocate(port, false));
    return owned;
  };
  options.measurement.measure = [&](const MappedMeasurementRequest &request)
      -> llvm::Expected<std::optional<mlir::llk::perf::CandidateMetrics>> {
    sawExecutable = true;
    EXPECT_EQ(request.identity.planId, request.candidate.plan.id);
    EXPECT_NE(request.identity.abiHash, 0u);
    EXPECT_FALSE(request.identity.operationKeys.empty());
    EXPECT_FALSE(request.identity.contentHash.empty());
    EXPECT_FALSE(request.identity.canonical.empty());
    EXPECT_EQ(request.executable.abiHash(), request.identity.abiHash);
    if (llvm::Error error =
            request.executable.invoke(request.inputs, request.outputs))
      return std::move(error);
    mlir::llk::perf::CandidateMetrics measured;
    measured.measuredNs = 100.0;
    return std::optional<mlir::llk::perf::CandidateMetrics>(measured);
  };
  options.measurement.verify =
      [](const MappedTuningCandidate &,
         llvm::ArrayRef<::llk::InvocationBuffer2D>,
         llvm::ArrayRef<::llk::InvocationBuffer2D> outputs) -> llvm::Error {
    for (int64_t i = 0; i < outputs.front().descriptor.size0 *
                                outputs.front().descriptor.size1;
         ++i)
      if (static_cast<float *>(outputs.front().descriptor.aligned)[i] != 67.0f)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "scalar reference mismatch");
    return llvm::Error::success();
  };
  auto report = runMappedTuningSession(*source, oneCandidateSpace(),
                                       testShape(), **target, options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  EXPECT_TRUE(sawExecutable)
      << (report->rejected.empty() ? ""
                                   : report->rejected.back().rejectionReason);
  ASSERT_EQ(report->ranked.size(), 1u)
      << (report->rejected.empty() ? ""
                                   : report->rejected.back().rejectionReason);
  ASSERT_TRUE(report->ranked.front().ranking.metrics.measuredNs);
  EXPECT_EQ(*report->ranked.front().ranking.metrics.measuredNs, 100.0);
  ASSERT_TRUE(report->ranked.front().measurementIdentity);

  options.measurement.measure = [](const MappedMeasurementRequest &)
      -> llvm::Expected<std::optional<mlir::llk::perf::CandidateMetrics>> {
    return std::optional<mlir::llk::perf::CandidateMetrics>();
  };
  auto unavailable = runMappedTuningSession(*source, oneCandidateSpace(),
                                            testShape(), **target, options);
  ASSERT_TRUE(bool(unavailable)) << llvm::toString(unavailable.takeError());
  ASSERT_EQ(unavailable->ranked.size(), 1u);
  EXPECT_TRUE(unavailable->ranked.front().ranking.legal);
  EXPECT_FALSE(unavailable->ranked.front().ranking.metrics.measuredNs);
  EXPECT_FALSE(unavailable->ranked.front().measurementIdentity);

  options.measurement.measure = [](const MappedMeasurementRequest &request)
      -> llvm::Expected<std::optional<mlir::llk::perf::CandidateMetrics>> {
    if (llvm::Error error =
            request.executable.invoke(request.inputs, request.outputs))
      return std::move(error);
    mlir::llk::perf::CandidateMetrics measured;
    measured.measuredNs = 0.0;
    return std::optional<mlir::llk::perf::CandidateMetrics>(measured);
  };
  auto invalidMetric = runMappedTuningSession(*source, oneCandidateSpace(),
                                              testShape(), **target, options);
  ASSERT_TRUE(bool(invalidMetric)) << llvm::toString(invalidMetric.takeError());
  EXPECT_TRUE(invalidMetric->ranked.empty());
  ASSERT_EQ(invalidMetric->rejected.size(), 3u);
  EXPECT_TRUE(std::any_of(
      invalidMetric->rejected.begin(), invalidMetric->rejected.end(),
      [](const auto &rejected) {
        return rejected.rejectionReason.find("measurement metric:") !=
               std::string::npos;
      }));

  auto measuredSpace = oneCandidateSpace();
  measuredSpace.objective.primaryMetric = "measured_ns";
  options.measurement.measure = [](const MappedMeasurementRequest &)
      -> llvm::Expected<std::optional<mlir::llk::perf::CandidateMetrics>> {
    return std::optional<mlir::llk::perf::CandidateMetrics>();
  };
  auto noMeasuredResult = runMappedTuningSession(
      *source, measuredSpace, testShape(), **target, options);
  ASSERT_TRUE(bool(noMeasuredResult))
      << llvm::toString(noMeasuredResult.takeError());
  EXPECT_TRUE(noMeasuredResult->ranked.empty());
  ASSERT_EQ(noMeasuredResult->unrankable.size(), 1u);
  EXPECT_TRUE(noMeasuredResult->unrankable.front().ranking.legal);
  EXPECT_FALSE(noMeasuredResult->hasMeasuredResult);
  EXPECT_TRUE(noMeasuredResult->measuredCohortOnly);
}

} // namespace
