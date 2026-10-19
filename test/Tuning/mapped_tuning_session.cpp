//===- mapped_tuning_session.cpp - Tests for mapped tuning sessions -------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/MappedTuningSession.h"
#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "LLK/Conversion/MicroMapping/MeasurementStore.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <set>
#include <string>

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

class RulesOverrideTarget final : public mlir::llk::mapping::MappingTarget {
public:
  RulesOverrideTarget(const mlir::llk::mapping::MappingTarget &base,
                      mlir::llk::mapping::RuleRegistry rules)
      : base_(base), rules_(std::move(rules)) {}
  llvm::StringRef name() const override { return base_.name(); }
  const mlir::llk::machine::MachineModel &machine() const override {
    return base_.machine();
  }
  const mlir::llk::mapping::LayoutRegistry &layouts() const override {
    return base_.layouts();
  }
  const mlir::llk::mapping::RuleRegistry &rules() const override {
    return rules_;
  }
  bool isKnownEmitter(llvm::StringRef key) const override {
    return base_.isKnownEmitter(key);
  }
  std::unique_ptr<mlir::llk::mapping::TargetEmitter>
  createEmitter(llvm::StringRef key) const override {
    return base_.createEmitter(key);
  }
  std::optional<mlir::llk::mapping::TargetCodegenRequirements>
  codegenRequirements() const override {
    return base_.codegenRequirements();
  }

private:
  const mlir::llk::mapping::MappingTarget &base_;
  mlir::llk::mapping::RuleRegistry rules_;
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

constexpr llvm::StringLiteral elementwiseAdapterKey = "linalg-elementwise-v1";

/// The bounded structured source fixture: a rank-two f32 elementwise add.
mlir::OwningOpRef<mlir::ModuleOp>
parseElementwiseSource(mlir::MLIRContext &context) {
  auto buffer = llvm::MemoryBuffer::getFile(
      std::string(LLK_SOURCE_DIR) +
      "/test/Tuning/Inputs/framework/elementwise.mlir");
  if (!buffer)
    return {};
  return mlir::parseSourceString<mlir::ModuleOp>((*buffer)->getBuffer(),
                                                 &context);
}

/// Two source-owned tile axes: d0 (8) by T0 and d1 (16) by T1.
mlir::llk::perf::SearchSpace elementwiseTiledSpace() {
  using namespace mlir::llk::perf;
  SearchSpace space;
  space.name = "elementwise";
  SearchParam t0;
  t0.name = "T0";
  t0.kind = "integer";
  t0.role = "axis:d0";
  t0.choices = {SearchChoice(int64_t{4}), SearchChoice(int64_t{8})};
  SearchParam t1;
  t1.name = "T1";
  t1.kind = "integer";
  t1.role = "axis:d1";
  t1.choices = {SearchChoice(int64_t{8}), SearchChoice(int64_t{16})};
  space.params = {std::move(t0), std::move(t1)};
  return space;
}

llvm::Expected<mlir::llk::workload::WorkloadContract>
describeElementwise(mlir::ModuleOp source) {
  using namespace mlir::llk;
  workload::SourceAdapterRegistry registry =
      workload::makeDefaultSourceAdapters();
  auto root = workload::selectSourceRoot(source, {"elementwise", 0}, registry,
                                         elementwiseAdapterKey);
  if (!root)
    return root.takeError();
  auto adapter = registry.resolve(*root, elementwiseAdapterKey);
  if (!adapter)
    return adapter.takeError();
  return (*adapter)->describe(*root);
}

mlir::llk::tuning::MappedTuningOptions elementwiseOptions() {
  mlir::llk::tuning::MappedTuningOptions options;
  options.source.sourceSymbol = "elementwise";
  options.source.adapterKey = elementwiseAdapterKey.str();
  options.mapping.mode = mlir::llk::mapping::SearchMode::Exact;
  options.mapping.topK = 8;
  return options;
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

/// The contract-based overload takes a source family that is not a contraction:
/// four tile bindings realize four schedules of one source and all enter the
/// existing complete-plan search.
TEST(MappedTuningSessionTest, ContractOverloadRanksElementwiseTiledSchedules) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = parseElementwiseSource(context);
  ASSERT_TRUE(source);
  auto target = mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto contract = describeElementwise(*source);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());

  auto options = elementwiseOptions();
  options.executable = false;
  auto report = mlir::llk::tuning::runMappedTuningSession(
      *source, elementwiseTiledSpace(), *contract, **target, options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  EXPECT_EQ(report->generated, 4u);
  ASSERT_EQ(report->ranked.size(), 4u)
      << (report->rejected.empty() ? ""
                                   : report->rejected.front().rejectionReason);

  std::set<uint64_t> sourceIdentities;
  std::set<uint64_t> planIdentities;
  std::set<std::string> bindings;
  for (const auto &selected : report->ranked) {
    sourceIdentities.insert(selected.sourceGraphHash);
    planIdentities.insert(selected.plan.id);
    bindings.insert(selected.ranking.candidate.id);
    EXPECT_EQ(selected.receipt.source, (std::vector<std::string>{"T0", "T1"}));
    EXPECT_TRUE(selected.receipt.mapping.empty());
    EXPECT_EQ(selected.plan.sourceBindingHash, selected.bindingHash);
    EXPECT_TRUE(selected.materializationReady);
  }
  EXPECT_EQ(sourceIdentities.size(), 1u);
  EXPECT_EQ(planIdentities.size(), 4u);
  EXPECT_EQ(bindings.size(), 4u);
}

/// The same four schedules compile through the selected target down to lowered
/// IR (not executable code), which is the integration point this increment
/// claims. Execution stays B7.
TEST(MappedTuningSessionTest,
     ContractOverloadLowersElementwiseForSelectedTarget) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = parseElementwiseSource(context);
  ASSERT_TRUE(source);
  auto target = mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto contract = describeElementwise(*source);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());

  auto options = elementwiseOptions();
  options.executable = true;
  options.backend = ::llk::MappedBackend::SelectedTarget;
  auto report = mlir::llk::tuning::runMappedTuningSession(
      *source, elementwiseTiledSpace(), *contract, **target, options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  EXPECT_EQ(report->generated, 4u);
  ASSERT_FALSE(report->ranked.empty())
      << (report->rejected.empty() ? ""
                                   : report->rejected.front().rejectionReason);
  for (const auto &selected : report->ranked)
    EXPECT_TRUE(selected.executable);
}

/// A mapping-owned choice that the selected plan does not consume is an
/// unrealized axis. It rejects the candidate inside the completion evaluator,
/// before top-K retention, rather than being filtered after the search.
TEST(MappedTuningSessionTest, UnconsumedMappingChoiceRejectsBeforeRetention) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = parseElementwiseSource(context);
  ASSERT_TRUE(source);
  auto target = mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto contract = describeElementwise(*source);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());

  auto space = elementwiseTiledSpace();
  mlir::llk::perf::SearchParam layout;
  layout.name = "L";
  layout.kind = "layout";
  layout.role = "operand9";
  layout.choices = {mlir::llk::perf::SearchChoice("row_major")};
  space.params.push_back(layout);

  auto options = elementwiseOptions();
  options.executable = false;
  auto report = mlir::llk::tuning::runMappedTuningSession(
      *source, space, *contract, **target, options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  EXPECT_EQ(report->generated, 4u);
  EXPECT_TRUE(report->ranked.empty());

  bool sawReceipt = false;
  for (const auto &rejected : report->rejected) {
    if (rejected.rejectionReason.find("mapping_receipt") == std::string::npos)
      continue;
    if (rejected.rejectionReason.find("axis_not_realized: L") !=
        std::string::npos)
      sawReceipt = true;
  }
  EXPECT_TRUE(sawReceipt) << (report->rejected.empty()
                                  ? ""
                                  : report->rejected.front().rejectionReason);
}

TEST(MappedTuningSessionTest, OwnerReceiptResolvesIntermediateOwners) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = parseElementwiseSource(context);
  ASSERT_TRUE(source);
  auto target = mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto contract = describeElementwise(*source);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());

  struct OwnerCase {
    const char *path;
    size_t ranked;
  };
  // The target models worker.0 -> lane.0 -> veng.0. Merely selecting both
  // compute kinds, or seeing the same owners in a different order, is not a
  // realization of that hierarchy.
  for (const OwnerCase &test :
       {OwnerCase{"worker/lane/vector_engine", 1},
        OwnerCase{"worker/vector_engine/lane", 0},
        OwnerCase{"worker/matrix_engine/vector_engine", 0}}) {
    SCOPED_TRACE(test.path);
    auto space = elementwiseTiledSpace();
    space.params[0].choices = {mlir::llk::perf::SearchChoice(int64_t{4})};
    space.params[1].choices = {mlir::llk::perf::SearchChoice(int64_t{8})};
    mlir::llk::perf::SearchParam owners;
    owners.name = "Owners";
    owners.kind = "owner_mapping";
    owners.choices = {mlir::llk::perf::SearchChoice(test.path)};
    space.params.push_back(owners);
    auto options = elementwiseOptions();
    options.executable = false;
    options.mapping.topK = 1;
    auto report = mlir::llk::tuning::runMappedTuningSession(
        *source, space, *contract, **target, options);
    ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
    ASSERT_EQ(report->ranked.size(), test.ranked)
        << (report->rejected.empty()
                ? ""
                : report->rejected.front().rejectionReason);
    if (test.ranked)
      EXPECT_EQ(report->ranked.front().receipt.mapping,
                (std::vector<std::string>{"Owners"}));
  }
}

TEST(MappedTuningSessionTest, MeasurementReplayRetainsReceiptValidPlan) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = parseElementwiseSource(context);
  ASSERT_TRUE(source);
  auto base = mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(base)) << llvm::toString(base.takeError());
  auto contract = describeElementwise(*source);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());

  // This competing rule implements add without consuming the bound layout.
  // Initial selection and replay must both reject it before retaining top-1.
  mlir::llk::mapping::RuleRegistry rules = (*base)->rules();
  const auto *add = rules.find("avx2.vector_add");
  ASSERT_NE(add, nullptr);
  auto alternate = *add;
  alternate.id = "review.no_layout_0";
  alternate.layoutRequirements.clear();
  std::string error;
  ASSERT_TRUE(rules.add(std::move(alternate), error)) << error;
  RulesOverrideTarget target(**base, std::move(rules));

  auto space = elementwiseTiledSpace();
  space.params[0].choices = {mlir::llk::perf::SearchChoice(int64_t{4})};
  space.params[1].choices = {mlir::llk::perf::SearchChoice(int64_t{8})};
  mlir::llk::perf::SearchParam layout;
  layout.name = "L";
  layout.kind = "layout";
  layout.role = "operand0";
  layout.choices = {mlir::llk::perf::SearchChoice("blocked")};
  space.params.push_back(layout);
  auto options = elementwiseOptions();
  options.executable = false;
  options.backend = ::llk::MappedBackend::Reference;
  options.mapping.topK = 1;
  auto initial = mlir::llk::tuning::runMappedTuningSession(
      *source, space, *contract, target, options);
  ASSERT_TRUE(bool(initial)) << llvm::toString(initial.takeError());
  ASSERT_EQ(initial->ranked.size(), 1u);
  const uint64_t selectedId = initial->ranked.front().plan.id;
  EXPECT_EQ(initial->ranked.front().receipt.mapping,
            (std::vector<std::string>{"L"}));

  using namespace mlir::llk::tuning;
  options.measurement.inputs = [](const ::llk::KernelAbi &abi)
      -> llvm::Expected<OwnedInvocationBuffers> {
    OwnedInvocationBuffers owned;
    auto allocate = [&](const ::llk::KernelAbi::Port &port,
                        float value) -> ::llk::InvocationBuffer2D {
      const size_t count = static_cast<size_t>(port.shape[0] * port.shape[1]);
      owned.storage.emplace_back(count * sizeof(float));
      auto *data = reinterpret_cast<float *>(owned.storage.back().data());
      std::fill_n(data, count, value);
      return {{data, data, 0, port.shape[0], port.shape[1], port.shape[1], 1},
              ::llk::InvocationElementType::F32,
              count * sizeof(float)};
    };
    for (size_t i = 0; i < abi.inputs.size(); ++i)
      owned.inputs.push_back(allocate(abi.inputs[i], i == 0 ? 1.0f : 2.0f));
    for (const auto &port : abi.outputs)
      owned.outputs.push_back(allocate(port, 0.0f));
    return owned;
  };
  bool measured = false;
  options.measurement.measure = [&](const MappedMeasurementRequest &request)
      -> llvm::Expected<std::optional<mlir::llk::perf::CandidateMetrics>> {
    measured = true;
    EXPECT_EQ(request.identity.planId, selectedId);
    EXPECT_EQ(request.candidate.receipt.mapping,
              (std::vector<std::string>{"L"}));
    if (llvm::Error invalid =
            request.executable.invoke(request.inputs, request.outputs))
      return std::move(invalid);
    mlir::llk::perf::CandidateMetrics metrics;
    metrics.measuredNs = 100.0;
    return std::optional<mlir::llk::perf::CandidateMetrics>(metrics);
  };
  options.measurement.verify =
      [](const MappedTuningCandidate &,
         llvm::ArrayRef<::llk::InvocationBuffer2D>,
         llvm::ArrayRef<::llk::InvocationBuffer2D> outputs) -> llvm::Error {
    for (const auto &output : outputs)
      for (int64_t i = 0; i < output.descriptor.size0 * output.descriptor.size1;
           ++i)
        if (static_cast<float *>(output.descriptor.aligned)[i] != 3.0f)
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "elementwise reference mismatch");
    return llvm::Error::success();
  };
  auto report = mlir::llk::tuning::runMappedTuningSession(
      *source, space, *contract, target, options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  EXPECT_TRUE(measured);
  ASSERT_EQ(report->ranked.size(), 1u)
      << (report->rejected.empty() ? ""
                                   : report->rejected.front().rejectionReason);
  EXPECT_EQ(report->ranked.front().plan.id, selectedId);
  ASSERT_TRUE(report->ranked.front().measurementIdentity);
  EXPECT_EQ(report->ranked.front().measurementIdentity->planId, selectedId);
  ASSERT_TRUE(report->ranked.front().ranking.metrics.measuredNs);
  EXPECT_EQ(*report->ranked.front().ranking.metrics.measuredNs, 100.0);
}

/// A rejected candidate does not end the search: the nondividing tile choice
/// rejects while the dividing one still ranks.
TEST(MappedTuningSessionTest, SourceRejectionDoesNotEndTheSearch) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = parseElementwiseSource(context);
  ASSERT_TRUE(source);
  auto target = mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto contract = describeElementwise(*source);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());

  auto space = elementwiseTiledSpace();
  space.params.front().choices = {mlir::llk::perf::SearchChoice(int64_t{3}),
                                  mlir::llk::perf::SearchChoice(int64_t{4})};
  auto options = elementwiseOptions();
  options.executable = false;
  auto report = mlir::llk::tuning::runMappedTuningSession(
      *source, space, *contract, **target, options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  EXPECT_EQ(report->generated, 4u);
  EXPECT_EQ(report->rejectedBySource, 2u);
  ASSERT_EQ(report->ranked.size(), 2u);
  for (const auto &rejected : report->rejected)
    EXPECT_NE(rejected.rejectionReason.find("tile_divisibility"),
              std::string::npos)
        << rejected.rejectionReason;
}

/// A source that offers no tunable choice is serialized as an explicit
/// singleton space. The loaded space yields its one empty binding and searches
/// the single schedule; an ordinary empty space would yield no candidates.
TEST(MappedTuningSessionTest, ExplicitSingletonSearchesItsOnlySchedule) {
  constexpr llvm::StringLiteral text = R"mlir(
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> tensor<8x16xf32> {
    %init = tensor.empty() : tensor<8x16xf32>
    %out = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
  micro.search_space @direct attributes {singleton = true, workload = "elementwise"} {
  }
}
)mlir";

  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  ASSERT_TRUE(source);
  auto space = mlir::llk::perf::loadSearchSpace(*source, "direct");
  ASSERT_TRUE(bool(space)) << llvm::toString(space.takeError());
  EXPECT_TRUE(space->singleton);
  EXPECT_TRUE(space->params.empty());

  auto target = mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto contract = describeElementwise(*source);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());
  auto options = elementwiseOptions();
  options.executable = false;
  auto report = mlir::llk::tuning::runMappedTuningSession(
      *source, *space, *contract, **target, options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  EXPECT_EQ(report->generated, 1u);
  ASSERT_EQ(report->ranked.size(), 1u)
      << (report->rejected.empty() ? ""
                                   : report->rejected.front().rejectionReason);
  EXPECT_TRUE(report->ranked.front().receipt.source.empty());
  EXPECT_TRUE(report->ranked.front().receipt.mapping.empty());
}

/// A contraction-only constraint has no facts on a source that performs no MMA:
/// it rejects instead of silently passing or fabricating M/N/K. A generic
/// shape-independent constraint over a mapping-owned choice still evaluates.
TEST(MappedTuningSessionTest, ContractionConstraintRequiresContractionFacts) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = parseElementwiseSource(context);
  ASSERT_TRUE(source);
  auto target = mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto contract = describeElementwise(*source);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());
  auto options = elementwiseOptions();
  options.executable = false;

  auto mmaSpace = elementwiseTiledSpace();
  mmaSpace.constraints = {
      {mlir::llk::perf::ConstraintKind::MmaCompatible, {"T0", "T1"}, {}}};
  auto rejected = mlir::llk::tuning::runMappedTuningSession(
      *source, mmaSpace, *contract, **target, options);
  ASSERT_TRUE(bool(rejected)) << llvm::toString(rejected.takeError());
  EXPECT_TRUE(rejected->ranked.empty());
  EXPECT_EQ(rejected->rejectedByLegality, 4u);
  ASSERT_FALSE(rejected->rejected.empty());
  EXPECT_NE(rejected->rejected.front().rejectionReason.find("mma_compatible"),
            std::string::npos);
  EXPECT_NE(rejected->rejected.front().rejectionReason.find("no contraction"),
            std::string::npos);

  // `layout_supported` needs only the bound symbolic choice and the machine, so
  // it remains evaluable for a non-contraction source.
  auto layoutSpace = elementwiseTiledSpace();
  mlir::llk::perf::SearchParam layout;
  layout.name = "L";
  layout.kind = "layout";
  layout.role = "operand0";
  layout.choices = {mlir::llk::perf::SearchChoice("blocked")};
  layoutSpace.params.push_back(layout);
  layoutSpace.constraints = {
      {mlir::llk::perf::ConstraintKind::LayoutSupported, {"L"}, {}}};
  auto accepted = mlir::llk::tuning::runMappedTuningSession(
      *source, layoutSpace, *contract, **target, options);
  ASSERT_TRUE(bool(accepted)) << llvm::toString(accepted.takeError());
  ASSERT_EQ(accepted->ranked.size(), 4u)
      << (accepted->rejected.empty()
              ? ""
              : accepted->rejected.front().rejectionReason);
  for (const auto &selected : accepted->ranked)
    EXPECT_EQ(selected.receipt.mapping, (std::vector<std::string>{"L"}));
}

/// The generic accelerator's audited package implements `micro.vector "add"`
/// but not the movement primitives realization emits. The report names the
/// exact unimplemented boundary instead of pretending portability.
TEST(MappedTuningSessionTest, GenericAcceleratorReportsPrimitiveBoundary) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = parseElementwiseSource(context);
  ASSERT_TRUE(source);
  auto target =
      mlir::llk::target::generic_accel::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  auto contract = describeElementwise(*source);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());

  auto options = elementwiseOptions();
  options.executable = false;
  auto report = mlir::llk::tuning::runMappedTuningSession(
      *source, elementwiseTiledSpace(), *contract, **target, options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  EXPECT_TRUE(report->ranked.empty());
  ASSERT_FALSE(report->rejected.empty());

  bool sawNoRule = false;
  bool sawPrimitive = false;
  for (const auto &rejected : report->rejected) {
    sawNoRule |=
        rejected.rejectionReason.find("no_matching_rule") != std::string::npos;
    sawPrimitive |=
        rejected.rejectionReason.find("micro.tile_async_copy") !=
            std::string::npos ||
        rejected.rejectionReason.find("micro.tile_store") != std::string::npos;
  }
  EXPECT_TRUE(sawNoRule) << report->rejected.front().rejectionReason;
  EXPECT_TRUE(sawPrimitive) << report->rejected.front().rejectionReason;
}

TEST(MappedTuningSessionTest,
     PersistsOnlyVerifiedSamplesAndReusesExactRecords) {
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
  std::vector<std::string> calls;
  options.measurement.inputs = [&](const ::llk::KernelAbi &abi)
      -> llvm::Expected<OwnedInvocationBuffers> {
    calls.push_back("inputs");
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

  llvm::SmallString<128> directory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("llk-d3", directory));
  struct Cleanup {
    llvm::StringRef directory;
    ~Cleanup() { llvm::sys::fs::remove_directories(directory); }
  } cleanup{directory};
  MeasurementStore store(directory.str().str());
  options.measurement.store = &store;
  options.measurement.reuseExact = true;
  options.measurement.protocol = {1, 1, 3};
  options.measurement.environment = {
      "test-host", "test-cpu", "test-isa", 1, "unpinned", "observed-default"};
  options.measurement.verification = {"inputs=1,2,3;outputs=67;tol=0",
                                      "scalar-f32-v1", "strict-f32"};
  options.measurement.measureDetailed = [&](const MappedMeasurementRequest &r)
      -> llvm::Expected<std::optional<MappedMeasurementSamples>> {
    calls.push_back("measure");
    if (auto error = r.executable.invoke(r.inputs, r.outputs))
      return std::move(error);
    MappedMeasurementSamples samples;
    samples.metrics.measuredNs = 90.0;
    samples.samplesNs = {80.0, 100.0, 90.0};
    return std::optional<MappedMeasurementSamples>(samples);
  };
  options.measurement.verify =
      [&](const MappedTuningCandidate &,
          llvm::ArrayRef<::llk::InvocationBuffer2D>,
          llvm::ArrayRef<::llk::InvocationBuffer2D> outputs) -> llvm::Error {
    calls.push_back("verify");
    for (int64_t i = 0; i < outputs.front().descriptor.size0 *
                                outputs.front().descriptor.size1;
         ++i)
      if (static_cast<float *>(outputs.front().descriptor.aligned)[i] != 67.0f)
        return llvm::createStringError("wrong output");
    return llvm::Error::success();
  };
  options.measurement.recordSink =
      [&](const mlir::llk::perf::MeasurementRecord &r) -> llvm::Error {
    calls.push_back("persist");
    EXPECT_TRUE(r.verified);
    EXPECT_EQ(r.samplesNs, (std::vector<double>{80.0, 100.0, 90.0}));
    EXPECT_EQ(r.medianNs, 90.0);
    EXPECT_FALSE(r.prediction.measuredNs);
    MeasurementLookup key{r.executableIdentity, r.environment, r.protocol,
                          r.verification};
    auto hit = store.lookup(key);
    EXPECT_TRUE(bool(hit));
    if (!hit)
      return hit.takeError();
    EXPECT_TRUE(hit->has_value()); // Observer runs after the durable write.
    return llvm::Error::success();
  };
  auto space = oneCandidateSpace();
  space.objective.primaryMetric = "measured_ns";
  auto run = [&] {
    return runMappedTuningSession(*source, space, testShape(), **target,
                                  options);
  };
  auto fresh = run();
  ASSERT_TRUE(bool(fresh)) << llvm::toString(fresh.takeError());
  ASSERT_EQ(fresh->ranked.size(), 1u);
  EXPECT_EQ(calls, (std::vector<std::string>{"inputs", "measure", "verify",
                                             "persist"}));
  EXPECT_EQ(fresh->measurementRecordsWritten, 1u);
  EXPECT_EQ(fresh->measurementRecordMisses, 1u);
  EXPECT_EQ(fresh->measurementAttempts, 1u);
  EXPECT_EQ(fresh->ranked.front().measurementOrigin, "measured_runtime");
  calls.clear();
  auto reused = run();
  ASSERT_TRUE(bool(reused)) << llvm::toString(reused.takeError());
  ASSERT_EQ(reused->ranked.size(), 1u);
  EXPECT_EQ(calls, (std::vector<std::string>{"inputs", "verify"}));
  EXPECT_EQ(reused->measurementRecordHits, 1u);
  EXPECT_EQ(reused->measurementAttempts, 0u);
  EXPECT_EQ(reused->measurementRecordsWritten, 0u);
  EXPECT_EQ(reused->ranked.front().measurementOrigin, "measured_record");
  EXPECT_EQ(reused->ranked.front().ranking.metrics.measuredNs, 90.0);
  for (const auto &placement : reused->ranked.front().plan.placements)
    EXPECT_FALSE(placement.measuredCycles);

  // A whole-kernel latency record cannot fabricate a missing secondary metric.
  space.objective.secondaryMetrics = {"measured_gflops"};
  auto missing = run();
  ASSERT_TRUE(bool(missing)) << llvm::toString(missing.takeError());
  EXPECT_TRUE(missing->ranked.empty());
  EXPECT_EQ(missing->unrankable.size(), 1u);
  space.objective.secondaryMetrics.clear();

  calls.clear();
  options.measurement.environment.threads = 2;
  auto changed = run();
  ASSERT_TRUE(bool(changed)) << llvm::toString(changed.takeError());
  EXPECT_EQ(changed->measurementRecordHits, 0u);
  EXPECT_EQ(changed->measurementRecordMisses, 1u);
  EXPECT_EQ(changed->measurementRecordsWritten, 1u);

  options.measurement.verify =
      [&](const MappedTuningCandidate &,
          llvm::ArrayRef<::llk::InvocationBuffer2D>,
          llvm::ArrayRef<::llk::InvocationBuffer2D>) -> llvm::Error {
    calls.push_back("verify");
    return llvm::createStringError("bad numerical output");
  };
  calls.clear();
  auto badHit = run();
  ASSERT_TRUE(bool(badHit)) << llvm::toString(badHit.takeError());
  EXPECT_TRUE(badHit->ranked.empty());
  EXPECT_EQ(badHit->measurementRecordHits, 0u);
  EXPECT_EQ(badHit->measurementRecordsWritten, 0u);
  EXPECT_EQ(calls, (std::vector<std::string>{"inputs", "verify"}));
  options.measurement.verification.fixtureContent += ";changed";
  calls.clear();
  auto failed = run();
  ASSERT_TRUE(bool(failed)) << llvm::toString(failed.takeError());
  EXPECT_TRUE(failed->ranked.empty());
  EXPECT_EQ(failed->measurementRecordsWritten, 0u);
  EXPECT_EQ(calls, (std::vector<std::string>{"inputs", "measure", "verify"}));
}

} // namespace
