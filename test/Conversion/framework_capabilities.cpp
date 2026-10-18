//===- framework_capabilities.cpp - Stage-specific capability receipts ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Framework C1 (issue #157). A known emitter key and a complete static mapping
// do not prove that a target lowers or runs a kernel. These tests drive the
// receipt from real fixtures -- the B2 direct elementwise source, a complete
// evaluated plan and the compiler's own compilation result -- and check that
// each stage reports only what was actually established.

#include "LLK/Conversion/FrameworkCapabilities.h"

#include "CompletePlanEvaluation.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Perf/MicroDAG.h"
#include "LLK/Perf/SelectedKernelAnalysis.h"
#include "LLK/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include "gtest/gtest.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

using mlir::llk::framework::CapabilityReceipt;
using mlir::llk::framework::CapabilityStatus;
using mlir::llk::mapping::BindContract;
using mlir::llk::mapping::CoveringPlan;
using mlir::llk::mapping::MappingSearchOptions;
using mlir::llk::mapping::MappingTarget;
using mlir::llk::mapping::SearchMode;
using mlir::llk::workload::PortDirection;
using mlir::llk::workload::PortEffect;
using mlir::llk::workload::WorkloadContract;

constexpr llvm::StringLiteral kAddSource = R"mlir(
module {
  micro.kernel @add(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> tensor<8x8xf32> attributes {micro.numerical_contract = "strict-f32"} {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>}
      : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>}
      : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %s = micro.vector "add" %ta, %tb
      : !micro.tile<8x8xf32, memory = #micro.memory<sram>>,
        !micro.tile<8x8xf32, memory = #micro.memory<sram>>
        -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield %s : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
  }
}
)mlir";

WorkloadContract addContract() {
  WorkloadContract contract;
  contract.numericalContract = "strict-f32";
  contract.ports = {
      {"input0", PortDirection::Input, PortEffect::Read, {8, 8}, "f32"},
      {"input1", PortDirection::Input, PortEffect::Read, {8, 8}, "f32"},
      {"output0", PortDirection::Output, PortEffect::Write, {8, 8}, "f32"}};
  contract.requiredPrimitives = {"micro.vector"};
  return contract;
}

enum class TargetKind { GenericAccelerator, AVX2 };

/// A real static-mapping fixture: the parsed B2 source, the target it maps
/// against, the complete plan the shared evaluator finalized, the strict
/// analysis of the kernel that plan bound, and that bound kernel itself.
struct Fixture {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> source;
  mlir::llk::mapping::WorkloadGraph graph;
  std::unique_ptr<MappingTarget> target;
  WorkloadContract contract;
  CoveringPlan plan;
  mlir::llk::perf::SelectedKernelAnalysis analysis;
  mlir::OwningOpRef<mlir::ModuleOp> boundModule;
  mlir::Operation *kernel = nullptr;
};

llvm::Error fixtureError(std::string message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 std::move(message));
}

llvm::Expected<std::unique_ptr<MappingTarget>> createTarget(TargetKind kind) {
  if (kind == TargetKind::GenericAccelerator)
    return mlir::llk::target::generic_accel::createMappingTarget(
        LLK_SOURCE_DIR);
  return mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
}

llvm::Expected<Fixture> buildFixture(TargetKind kind,
                                     llvm::StringRef source = kAddSource) {
  Fixture fixture;
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  fixture.context = std::make_unique<mlir::MLIRContext>(registry);
  fixture.context->loadAllAvailableDialects();
  fixture.source =
      mlir::parseSourceString<mlir::ModuleOp>(source, fixture.context.get());
  if (!fixture.source)
    return fixtureError("cannot parse the direct elementwise fixture");

  fixture.contract = addContract();
  if (llvm::Error error =
          mlir::llk::workload::validateWorkloadContract(fixture.contract))
    return std::move(error);

  mlir::Operation *sourceKernel = nullptr;
  fixture.source->walk([&](mlir::Operation *op) {
    if (!sourceKernel && op->getName().getStringRef() == "micro.kernel")
      sourceKernel = op;
  });
  if (!sourceKernel)
    return fixtureError("the fixture has no micro.kernel");

  auto graph = mlir::llk::mapping::extractWorkloadGraph(sourceKernel);
  if (!graph)
    return graph.takeError();
  fixture.graph = std::move(*graph);

  auto target = createTarget(kind);
  if (!target)
    return target.takeError();
  fixture.target = std::move(*target);

  MappingSearchOptions searchOptions;
  searchOptions.mode = SearchMode::Exact;
  searchOptions.topK = 1;
  searchOptions.requireCompleteEvaluation = true;
  searchOptions.evaluateCompletePlan = [&](const CoveringPlan &proposal) {
    return mlir::llk::mapping::evaluateCompletePlan(
        *fixture.source, fixture.graph, *fixture.target, proposal,
        BindContract::Partial, std::nullopt,
        mlir::llk::mapping::AnalysisRequirement::Complete);
  };
  mlir::llk::mapping::LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  mlir::llk::mapping::CoveringSearch search(fixture.graph, *fixture.target,
                                            *fixture.context, layoutContext,
                                            searchOptions);
  auto found = search.search();
  if (!found)
    return found.takeError();
  if (found->plans.empty()) {
    std::string reason = "the fixture produced no complete plan";
    if (!found->frontier.diagnostics.empty())
      reason += ": " + found->frontier.diagnostics.front().message;
    return fixtureError(std::move(reason));
  }
  fixture.plan = found->plans.front();

  llk::MappedCompileOptions compile;
  compile.entrySymbol = "add";
  compile.stop = llk::MappedStop::MappedMicro;
  compile.requireExecutable = false;
  auto bound = llk::compileMappedKernel(*fixture.source, *fixture.target,
                                        fixture.plan, compile);
  if (!bound)
    return bound.takeError();
  fixture.boundModule = std::move(bound->module);

  auto kernel =
      mlir::llk::perf::findMicroKernel(fixture.boundModule.get(), "add");
  if (!kernel)
    return kernel.takeError();
  fixture.kernel = *kernel;

  auto analysis = mlir::llk::perf::analyzeSelectedKernel(
      fixture.kernel, fixture.target->machine(), /*requireComplete=*/true,
      fixture.plan, fixture.graph);
  if (!analysis)
    return analysis.takeError();
  fixture.analysis = std::move(*analysis);
  return std::move(fixture);
}

llvm::Expected<llk::MappedCompilation>
compileAt(Fixture &fixture, llk::MappedStop stop, llk::MappedBackend backend) {
  llk::MappedCompileOptions options;
  options.entrySymbol = "add";
  options.stop = stop;
  options.backend = backend;
  options.requireExecutable = true;
  return llk::compileMappedKernel(*fixture.source, *fixture.target,
                                  fixture.plan, options);
}

std::string joinedReasons(const CapabilityReceipt &receipt) {
  return llvm::join(receipt.reasons, "\n");
}

TEST(FrameworkCapabilitiesTest,
     GenericAcceleratorReportsModeledSupportWithoutExecution) {
  auto fixture = buildFixture(TargetKind::GenericAccelerator);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  CapabilityReceipt receipt = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, *fixture->target, &fixture->analysis,
      &fixture->plan);

  EXPECT_EQ(receipt.structural, CapabilityStatus::Supported);
  EXPECT_EQ(receipt.interpretation, CapabilityStatus::Supported);
  EXPECT_EQ(receipt.mapping, CapabilityStatus::Supported);
  EXPECT_EQ(receipt.physicalPlan, CapabilityStatus::Supported);
  EXPECT_EQ(receipt.referenceLowering, CapabilityStatus::NotChecked);
  EXPECT_EQ(receipt.selectedLowering, CapabilityStatus::Unsupported);
  EXPECT_EQ(receipt.invocation, CapabilityStatus::NotChecked);
  EXPECT_NE(joinedReasons(receipt).find("generic-ai-accel"), std::string::npos);
  EXPECT_FALSE(receipt.contractIdentity.empty());
  EXPECT_FALSE(receipt.targetIdentity.empty());
  EXPECT_NE(receipt.planId, 0u);
}

TEST(FrameworkCapabilitiesTest,
     Avx2MappedMicroIsNotCheckedUntilSelectedLoweringRuns) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  auto mapped = compileAt(*fixture, llk::MappedStop::MappedMicro,
                          llk::MappedBackend::Reference);
  ASSERT_TRUE(static_cast<bool>(mapped)) << llvm::toString(mapped.takeError());

  CapabilityReceipt atMapping = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, *fixture->target, &fixture->analysis,
      &fixture->plan, &*mapped);
  EXPECT_EQ(atMapping.selectedLowering, CapabilityStatus::NotChecked);
  EXPECT_EQ(atMapping.referenceLowering, CapabilityStatus::NotChecked);
  EXPECT_EQ(atMapping.invocation, CapabilityStatus::NotChecked);

  auto lowered = compileAt(*fixture, llk::MappedStop::Lowered,
                           llk::MappedBackend::SelectedTarget);
  ASSERT_TRUE(static_cast<bool>(lowered))
      << llvm::toString(lowered.takeError());
  ASSERT_GT(lowered->backendGroupsRealized, 0u);
  ASSERT_EQ(lowered->referenceGroupsLowered, 0u);

  CapabilityReceipt atLowered = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, *fixture->target, &fixture->analysis,
      &fixture->plan, &*lowered);
  EXPECT_EQ(atLowered.selectedLowering, CapabilityStatus::Supported);
  EXPECT_EQ(atLowered.referenceLowering, CapabilityStatus::NotChecked);
  EXPECT_EQ(atLowered.invocation, CapabilityStatus::NotChecked);
}

TEST(FrameworkCapabilitiesTest,
     ReferenceLoweringDoesNotCreditSelectedLowering) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(bool(fixture)) << llvm::toString(fixture.takeError());
  auto lowered = compileAt(*fixture, llk::MappedStop::Lowered,
                           llk::MappedBackend::Reference);
  ASSERT_TRUE(bool(lowered)) << llvm::toString(lowered.takeError());
  auto receipt = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, *fixture->target, &fixture->analysis,
      &fixture->plan, &*lowered);
  EXPECT_EQ(receipt.referenceLowering, CapabilityStatus::Supported);
  EXPECT_EQ(receipt.selectedLowering, CapabilityStatus::NotChecked);
}

// A valid alternate topology retains the same rule vocabulary but changes the
// concrete engines. Old successful evidence must not describe these engines.
class ReplacedEnginesTarget : public MappingTarget {
public:
  explicit ReplacedEnginesTarget(const MappingTarget &base)
      : base(base), model(base.machine()) {
    for (auto &compute : model.computes)
      compute.id += ".replacement";
    model.contentHash = mlir::llk::machine::computeContentHash(model);
  }
  llvm::StringRef name() const override { return base.name(); }
  const mlir::llk::machine::MachineModel &machine() const override {
    return model;
  }
  const mlir::llk::mapping::LayoutRegistry &layouts() const override {
    return base.layouts();
  }
  const mlir::llk::mapping::RuleRegistry &rules() const override {
    return base.rules();
  }
  bool isKnownEmitter(llvm::StringRef key) const override {
    return base.isKnownEmitter(key);
  }
  std::unique_ptr<mlir::llk::mapping::TargetEmitter>
  createEmitter(llvm::StringRef key) const override {
    return base.createEmitter(key);
  }
  std::optional<mlir::llk::mapping::TargetCodegenRequirements>
  codegenRequirements() const override {
    return base.codegenRequirements();
  }

private:
  const MappingTarget &base;
  mlir::llk::machine::MachineModel model;
};

TEST(FrameworkCapabilitiesTest, ChangedTargetRejectsOldSuccessfulEvidence) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(bool(fixture)) << llvm::toString(fixture.takeError());
  auto lowered = compileAt(*fixture, llk::MappedStop::Lowered,
                           llk::MappedBackend::SelectedTarget);
  ASSERT_TRUE(bool(lowered)) << llvm::toString(lowered.takeError());
  ReplacedEnginesTarget changed(*fixture->target);
  llvm::Error invalid = mlir::llk::mapping::verifyMappingTarget(changed);
  ASSERT_FALSE(bool(invalid)) << llvm::toString(std::move(invalid));
  auto receipt = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, changed, &fixture->analysis,
      &fixture->plan, &*lowered);
  auto original = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, *fixture->target, &fixture->analysis,
      &fixture->plan, &*lowered);
  EXPECT_NE(receipt.targetIdentity, original.targetIdentity);
  EXPECT_EQ(receipt.interpretation, CapabilityStatus::Unsupported);
  EXPECT_EQ(receipt.mapping, CapabilityStatus::Unsupported);
  EXPECT_EQ(receipt.physicalPlan, CapabilityStatus::Unsupported);
  EXPECT_EQ(receipt.selectedLowering, CapabilityStatus::Unsupported);
  EXPECT_NE(joinedReasons(receipt).find("target"), std::string::npos);
}

TEST(FrameworkCapabilitiesTest, ChangedSourceRejectsOldSuccessfulEvidence) {
  auto original = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(bool(original)) << llvm::toString(original.takeError());
  auto lowered = compileAt(*original, llk::MappedStop::Lowered,
                           llk::MappedBackend::SelectedTarget);
  ASSERT_TRUE(bool(lowered)) << llvm::toString(lowered.takeError());
  std::string mul = kAddSource.str();
  mul.replace(mul.find("micro.vector \"add\""), 18, "micro.vector \"mul\"");
  auto changed = buildFixture(TargetKind::AVX2, mul);
  ASSERT_TRUE(bool(changed)) << llvm::toString(changed.takeError());

  auto stale = mlir::llk::framework::describeCapabilities(
      changed->kernel, changed->contract, *changed->target, &original->analysis,
      &original->plan, &*lowered);
  auto current = mlir::llk::framework::describeCapabilities(
      changed->kernel, changed->contract, *changed->target, &changed->analysis,
      &changed->plan);
  EXPECT_EQ(stale.sourceIdentity, current.sourceIdentity);
  EXPECT_EQ(stale.interpretation, CapabilityStatus::Unsupported);
  EXPECT_EQ(stale.mapping, CapabilityStatus::Unsupported);
  EXPECT_EQ(stale.physicalPlan, CapabilityStatus::Unsupported);
  EXPECT_EQ(stale.selectedLowering, CapabilityStatus::Unsupported);

  // Even a current plan cannot lend its identity to old analysis/compilation.
  auto mixed = mlir::llk::framework::describeCapabilities(
      changed->kernel, changed->contract, *changed->target, &original->analysis,
      &changed->plan, &*lowered);
  EXPECT_EQ(mixed.mapping, CapabilityStatus::Supported);
  EXPECT_EQ(mixed.interpretation, CapabilityStatus::Unsupported);
  EXPECT_EQ(mixed.selectedLowering, CapabilityStatus::Unsupported);
}

TEST(FrameworkCapabilitiesTest, FrozenPlanRetainsVerifiedSourceProvenance) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(bool(fixture)) << llvm::toString(fixture.takeError());
  mlir::llk::mapping::MappingSearchResult result;
  result.plans.push_back(fixture->plan);
  result.workloadHash = fixture->plan.graphHash;
  std::string report = mlir::llk::mapping::writePlanReport(
      result, fixture->target->machine(), *fixture->target,
      MappingSearchOptions{}, fixture->plan.graphHash);
  auto replay = mlir::llk::mapping::readPlanReport(report, *fixture->target,
                                                   fixture->graph);
  ASSERT_TRUE(bool(replay)) << llvm::toString(replay.takeError());
  auto receipt = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, *fixture->target, &fixture->analysis,
      &*replay);
  EXPECT_EQ(receipt.mapping, CapabilityStatus::Supported)
      << joinedReasons(receipt);
  EXPECT_EQ(receipt.physicalPlan, CapabilityStatus::Supported);
}

TEST(FrameworkCapabilitiesTest, PreparationAndHostChecksDoNotCreditInvocation) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(bool(fixture)) << llvm::toString(fixture.takeError());
  auto compiled = compileAt(*fixture, llk::MappedStop::Executable,
                            llk::MappedBackend::Reference);
  ASSERT_TRUE(bool(compiled)) << llvm::toString(compiled.takeError());
  ASSERT_TRUE(compiled->executable);
  auto describe = [&](const WorkloadContract &contract) {
    return mlir::llk::framework::describeCapabilities(
        fixture->kernel, contract, *fixture->target, &fixture->analysis,
        &fixture->plan, &*compiled);
  };
  auto expectJsonStage = [](const CapabilityReceipt &receipt,
                            llvm::StringRef key, llvm::StringRef expected) {
    auto json = mlir::llk::framework::capabilityReceiptToJson(receipt);
    ASSERT_TRUE(json.getString(key).has_value()) << key.str();
    EXPECT_EQ(*json.getString(key), expected) << key.str();
  };
  auto prepared = describe(fixture->contract);
  EXPECT_EQ(prepared.invocation, CapabilityStatus::NotChecked);
  expectJsonStage(prepared, "abi_eligibility", "supported");
  expectJsonStage(prepared, "abi_preparation", "supported");
  expectJsonStage(prepared, "host_compatibility", "supported");
  auto absentContract = describe({});
  EXPECT_EQ(absentContract.invocation, CapabilityStatus::NotChecked);
  expectJsonStage(absentContract, "abi_eligibility", "not_checked");
  expectJsonStage(absentContract, "abi_preparation", "supported");

  std::array<float, 64> left, right, output{};
  left.fill(1.0f);
  right.fill(2.0f);
  auto buffer = [](std::array<float, 64> &data) -> llk::InvocationBuffer2D {
    return {{data.data(), data.data(), 0, 8, 8, 8, 1},
            llk::InvocationElementType::F32,
            sizeof(data)};
  };
  std::array<llk::InvocationBuffer2D, 2> inputs{buffer(left), buffer(right)};
  std::array<llk::InvocationBuffer2D, 1> outputs{buffer(output)};
  llvm::Error invalid = compiled->executable->invoke(
      llvm::ArrayRef<llk::InvocationBuffer2D>{}, outputs);
  ASSERT_TRUE(bool(invalid));
  llvm::consumeError(std::move(invalid));
  EXPECT_EQ(describe(fixture->contract).invocation,
            CapabilityStatus::NotChecked);
  llvm::Error error = compiled->executable->invoke(inputs, outputs);
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  EXPECT_EQ(describe(fixture->contract).invocation,
            CapabilityStatus::Supported);
  for (float actual : output)
    EXPECT_FLOAT_EQ(actual, 3.0f);

  ReplacedEnginesTarget changed(*fixture->target);
  auto stale = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, changed, &fixture->analysis,
      &fixture->plan, &*compiled);
  EXPECT_EQ(stale.referenceLowering, CapabilityStatus::Unsupported);
  EXPECT_EQ(stale.abiPreparation, CapabilityStatus::Unsupported);
  EXPECT_EQ(stale.hostCompatibility, CapabilityStatus::Unsupported);
  EXPECT_EQ(stale.invocation, CapabilityStatus::Unsupported);
}

std::vector<WorkloadContract> differentPortContracts() {
  std::vector<WorkloadContract> contracts(4, addContract());
  contracts[0].ports[0].shape = {4, 16}; // Same byte count, different shape.
  contracts[1].ports[0].elementType = "bf16";
  contracts[2].ports[1].direction = PortDirection::Output;
  contracts[2].ports[1].effect = PortEffect::Write;
  contracts[3].ports.pop_back();
  return contracts;
}

TEST(FrameworkCapabilitiesTest,
     ChangedContractCannotBorrowSuccessfulInvocation) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(bool(fixture)) << llvm::toString(fixture.takeError());
  auto compiled = compileAt(*fixture, llk::MappedStop::Executable,
                            llk::MappedBackend::Reference);
  ASSERT_TRUE(bool(compiled)) << llvm::toString(compiled.takeError());
  std::array<float, 64> left, right, output{};
  left.fill(1.0f);
  right.fill(2.0f);
  auto buffer = [](std::array<float, 64> &data) -> llk::InvocationBuffer2D {
    return {{data.data(), data.data(), 0, 8, 8, 8, 1},
            llk::InvocationElementType::F32,
            sizeof(data)};
  };
  std::array<llk::InvocationBuffer2D, 2> inputs{buffer(left), buffer(right)};
  std::array<llk::InvocationBuffer2D, 1> outputs{buffer(output)};
  llvm::Error error = compiled->executable->invoke(inputs, outputs);
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  for (float actual : output)
    EXPECT_FLOAT_EQ(actual, 3.0f);

  for (const WorkloadContract &different : differentPortContracts()) {
    auto receipt = mlir::llk::framework::describeCapabilities(
        fixture->kernel, different, *fixture->target, &fixture->analysis,
        &fixture->plan, &*compiled);
    // Each declaration fits the ABI envelope, but none is this executable's
    // ordered boundary contract. An old successful call cannot validate it.
    EXPECT_EQ(receipt.abiEligibility, CapabilityStatus::Supported);
    EXPECT_EQ(receipt.abiPreparation, CapabilityStatus::Unsupported);
    EXPECT_EQ(receipt.invocation, CapabilityStatus::Unsupported);
    EXPECT_NE(joinedReasons(receipt).find("executable ABI"), std::string::npos);
  }
}

TEST(FrameworkCapabilitiesTest,
     ContractAssociationUsesOriginalAndBoundSignature) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(bool(fixture)) << llvm::toString(fixture.takeError());
  auto sourceKernel = mlir::llk::perf::findMicroKernel(*fixture->source, "add");
  ASSERT_TRUE(bool(sourceKernel)) << llvm::toString(sourceKernel.takeError());
  for (mlir::Operation *kernel : {*sourceKernel, fixture->kernel}) {
    auto matching = mlir::llk::framework::describeCapabilities(
        kernel, fixture->contract, *fixture->target);
    EXPECT_EQ(matching.abiEligibility, CapabilityStatus::Supported);
    EXPECT_EQ(matching.abiPreparation, CapabilityStatus::NotChecked);
    EXPECT_EQ(matching.invocation, CapabilityStatus::NotChecked);
    for (const WorkloadContract &different : differentPortContracts()) {
      auto receipt = mlir::llk::framework::describeCapabilities(
          kernel, different, *fixture->target);
      EXPECT_EQ(receipt.abiEligibility, CapabilityStatus::Supported);
      EXPECT_EQ(receipt.abiPreparation, CapabilityStatus::Unsupported);
      EXPECT_EQ(receipt.invocation, CapabilityStatus::Unsupported);
      EXPECT_NE(joinedReasons(receipt).find("source signature"),
                std::string::npos);
    }
  }
}

TEST(FrameworkCapabilitiesTest, NonpositiveExtentsAreNotAbiEligible) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(bool(fixture)) << llvm::toString(fixture.takeError());
  for (int64_t extent : {-1, 0, -3}) {
    auto invalid = fixture->contract;
    invalid.ports[0].shape[0] = extent;
    auto receipt = mlir::llk::framework::describeCapabilities(
        fixture->kernel, invalid, *fixture->target);
    EXPECT_EQ(receipt.abiEligibility, CapabilityStatus::Unsupported);
    EXPECT_EQ(receipt.abiPreparation, CapabilityStatus::Unsupported);
    EXPECT_EQ(receipt.invocation, CapabilityStatus::Unsupported);
    EXPECT_NE(joinedReasons(receipt).find("positive static extents"),
              std::string::npos);
  }
}

TEST(FrameworkCapabilitiesTest, SignatureAssociationPreservesInputPortOrder) {
  std::string source = kAddSource.str();
  llvm::StringRef lastInput = "%b: tensor<8x8xf32>)";
  source.replace(source.find(lastInput.str()), lastInput.size(),
                 "%b: tensor<8x8xf32>, %unused: tensor<4x16xbf16>)");
  auto fixture = buildFixture(TargetKind::AVX2, source);
  ASSERT_TRUE(bool(fixture)) << llvm::toString(fixture.takeError());
  auto contract = addContract();
  contract.ports.insert(
      contract.ports.begin() + 2,
      {"input2", PortDirection::Input, PortEffect::Read, {4, 16}, "bf16"});
  auto matching = mlir::llk::framework::describeCapabilities(
      fixture->kernel, contract, *fixture->target);
  EXPECT_EQ(matching.abiPreparation, CapabilityStatus::NotChecked);
  std::swap(contract.ports[0], contract.ports[2]);
  auto reordered = mlir::llk::framework::describeCapabilities(
      fixture->kernel, contract, *fixture->target);
  EXPECT_EQ(reordered.abiEligibility, CapabilityStatus::Supported);
  EXPECT_EQ(reordered.abiPreparation, CapabilityStatus::Unsupported);
  EXPECT_EQ(reordered.invocation, CapabilityStatus::Unsupported);
}

TEST(FrameworkCapabilitiesTest, MalformedBundleCarriesAConcreteReason) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  ASSERT_FALSE(fixture->plan.placements.empty());

  CoveringPlan tampered = fixture->plan;
  tampered.placements.front().bundle.emitterKey = "avx2_not_declared";

  CapabilityReceipt receipt = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, *fixture->target, &fixture->analysis,
      &tampered);
  EXPECT_EQ(receipt.selectedLowering, CapabilityStatus::Unsupported);
  EXPECT_NE(joinedReasons(receipt).find("avx2_not_declared"),
            std::string::npos);
}

TEST(FrameworkCapabilitiesTest, UnsupportedAbiCarriesAConcreteReason) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  WorkloadContract rankThree = fixture->contract;
  for (mlir::llk::workload::WorkloadPort &port : rankThree.ports)
    port.shape = {2, 4, 8};

  CapabilityReceipt receipt = mlir::llk::framework::describeCapabilities(
      fixture->kernel, rankThree, *fixture->target, &fixture->analysis,
      &fixture->plan);
  EXPECT_EQ(receipt.interpretation, CapabilityStatus::Supported);
  EXPECT_EQ(receipt.mapping, CapabilityStatus::Supported);
  EXPECT_EQ(receipt.invocation, CapabilityStatus::Unsupported);
  EXPECT_EQ(receipt.abiEligibility, CapabilityStatus::Unsupported);
  EXPECT_EQ(receipt.abiPreparation, CapabilityStatus::Unsupported);
  EXPECT_EQ(receipt.hostCompatibility, CapabilityStatus::NotChecked);
  EXPECT_NE(joinedReasons(receipt).find("rank-two"), std::string::npos)
      << joinedReasons(receipt);
}

TEST(FrameworkCapabilitiesTest, IncompleteAnalysisIsUnsupportedWithReasons) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  mlir::llk::perf::SelectedKernelAnalysis incomplete = fixture->analysis;
  incomplete.complete = false;
  incomplete.incompleteReasons = {"unmodelled event resource 'vpu.0'"};

  CapabilityReceipt receipt = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, *fixture->target, &incomplete,
      &fixture->plan);
  EXPECT_EQ(receipt.interpretation, CapabilityStatus::Unsupported);
  EXPECT_NE(joinedReasons(receipt).find("unmodelled event resource"),
            std::string::npos);
}

TEST(FrameworkCapabilitiesTest, AbsentStagesAreNeverChecked) {
  auto fixture = buildFixture(TargetKind::AVX2);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  CapabilityReceipt receipt = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, *fixture->target);
  EXPECT_EQ(receipt.structural, CapabilityStatus::Supported);
  EXPECT_EQ(receipt.interpretation, CapabilityStatus::NotChecked);
  EXPECT_EQ(receipt.mapping, CapabilityStatus::NotChecked);
  EXPECT_EQ(receipt.physicalPlan, CapabilityStatus::NotChecked);
  EXPECT_EQ(receipt.referenceLowering, CapabilityStatus::NotChecked);
  EXPECT_EQ(receipt.selectedLowering, CapabilityStatus::NotChecked);
  EXPECT_EQ(receipt.abiEligibility, CapabilityStatus::Supported);
  EXPECT_EQ(receipt.abiPreparation, CapabilityStatus::NotChecked);
  EXPECT_EQ(receipt.hostCompatibility, CapabilityStatus::NotChecked);
  EXPECT_EQ(receipt.invocation, CapabilityStatus::NotChecked);
}

TEST(FrameworkCapabilitiesTest, JsonSchemaCarriesEveryStatusAndIdentity) {
  auto fixture = buildFixture(TargetKind::GenericAccelerator);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  CapabilityReceipt receipt = mlir::llk::framework::describeCapabilities(
      fixture->kernel, fixture->contract, *fixture->target, &fixture->analysis,
      &fixture->plan);
  llvm::json::Object json =
      mlir::llk::framework::capabilityReceiptToJson(receipt);

  EXPECT_EQ(json.getInteger("version"), 2u);
  for (const char *key :
       {"structural", "interpretation", "mapping", "physical_plan",
        "reference_lowering", "selected_lowering", "abi_eligibility",
        "abi_preparation", "host_compatibility", "invocation"})
    EXPECT_NE(json.get(key), nullptr) << key;
  EXPECT_NE(json.get("reasons"), nullptr);
  EXPECT_NE(json.get("source_identity"), nullptr);
  EXPECT_NE(json.get("contract_identity"), nullptr);
  EXPECT_NE(json.get("target_identity"), nullptr);
  EXPECT_NE(json.get("plan_id"), nullptr);
  ASSERT_TRUE(json.getString("structural").has_value());
  EXPECT_EQ(*json.getString("structural"), "supported");
  ASSERT_TRUE(json.getString("selected_lowering").has_value());
  EXPECT_EQ(*json.getString("selected_lowering"), "unsupported");
  ASSERT_TRUE(json.getString("invocation").has_value());
  EXPECT_EQ(*json.getString("invocation"), "not_checked");
}

} // namespace
