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
#include "LLK/Mapping/PlanBinder.h"
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

llvm::Expected<Fixture> buildFixture(TargetKind kind) {
  Fixture fixture;
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  fixture.context = std::make_unique<mlir::MLIRContext>(registry);
  fixture.context->loadAllAvailableDialects();
  fixture.source = mlir::parseSourceString<mlir::ModuleOp>(
      kAddSource, fixture.context.get());
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
  EXPECT_EQ(atLowered.referenceLowering, CapabilityStatus::Supported);
  EXPECT_EQ(atLowered.invocation, CapabilityStatus::NotChecked);
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

  EXPECT_EQ(json.getInteger("version"), 1u);
  for (const char *key :
       {"structural", "interpretation", "mapping", "physical_plan",
        "reference_lowering", "selected_lowering", "invocation"})
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
