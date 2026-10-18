//===- framework_target_portability.cpp - Target extension boundaries ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Framework C4 (issue #160). A target's name is provenance, not policy:
// renaming a target must leave every legal mapping decision, analysis verdict
// and base backend capability unchanged, while the receipt's target identity
// moves. Changing the *machine data* must change the legal set and rejection
// reasons. The tests use a test-only forwarding wrapper that is never
// registered globally, plus literal topology edits.

#include "resource_regression_fixture.h"

#include "CompletePlanEvaluation.h"

#include "LLK/Conversion/FrameworkCapabilities.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
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
#include "llvm/Support/MemoryBuffer.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

using mlir::llk::framework::CapabilityReceipt;
using mlir::llk::framework::CapabilityStatus;
using mlir::llk::mapping::BindContract;
using mlir::llk::mapping::CoveringPlan;
using mlir::llk::mapping::FileMappingTarget;
using mlir::llk::mapping::MappingSearchOptions;
using mlir::llk::mapping::MappingSearchResult;
using mlir::llk::mapping::MappingTarget;
using mlir::llk::mapping::SearchMode;
using mlir::llk::workload::PortDirection;
using mlir::llk::workload::PortEffect;
using mlir::llk::workload::WorkloadContract;

/// A test-only target whose policy is exactly `base`'s, under another name.
///
/// Every method forwards -- machine, layouts, rules, emitter lookup, codegen
/// contract, latency provider and the selected backend pipeline. Generic code
/// must never learn the difference except through `name()` and the identities
/// derived from it. The base target outlives this wrapper; it is never
/// registered globally.
class RenamedTarget final : public MappingTarget {
public:
  RenamedTarget(const MappingTarget &base, std::string name)
      : base_(base), name_(std::move(name)) {}

  llvm::StringRef name() const override { return name_; }
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
  std::optional<mlir::llk::mapping::TargetCodegenRequirements>
  codegenRequirements() const override {
    return base_.codegenRequirements();
  }
  std::unique_ptr<mlir::llk::mapping::TargetEmitter>
  createEmitter(llvm::StringRef key) const override {
    return base_.createEmitter(key);
  }
  std::unique_ptr<mlir::llk::mapping::TargetEmitter>
  createEmitter() const override {
    return base_.createEmitter();
  }
  const mlir::llk::mapping::LatencyProvider *latencyProvider() const override {
    return base_.latencyProvider();
  }
  llvm::Error
  buildSelectedBackendPipeline(mlir::OpPassManager &pm) const override {
    return base_.buildSelectedBackendPipeline(pm);
  }

private:
  const MappingTarget &base_;
  std::string name_;
};

mlir::DialectRegistry frameworkRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  return registry;
}

WorkloadContract addMulContract() {
  WorkloadContract contract;
  contract.numericalContract = "strict-f32";
  contract.ports = {
      {"a", PortDirection::Input, PortEffect::Read, {8, 8}, "f32"},
      {"b", PortDirection::Input, PortEffect::Read, {8, 8}, "f32"},
      {"output", PortDirection::Output, PortEffect::Write, {8, 8}, "f32"}};
  return contract;
}

/// Everything one framework fixture produced: the source module, the selected
/// plan, the kernel it bound and the strict analysis of that kernel.
struct FrameworkFixture {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> source;
  mlir::llk::mapping::WorkloadGraph graph;
  CoveringPlan plan;
  mlir::OwningOpRef<mlir::ModuleOp> boundModule;
  mlir::Operation *kernel = nullptr;
  mlir::llk::perf::SelectedKernelAnalysis analysis;
};

llvm::Expected<MappingSearchResult>
searchFramework(mlir::ModuleOp module,
                const mlir::llk::mapping::WorkloadGraph &graph,
                MappingTarget &target) {
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 1;
  options.requireCompleteEvaluation = true;
  options.evaluateCompletePlan = [&](const CoveringPlan &proposal) {
    return mlir::llk::mapping::evaluateCompletePlan(
        module, graph, target, proposal, BindContract::Partial, std::nullopt,
        mlir::llk::mapping::AnalysisRequirement::Complete);
  };
  mlir::llk::mapping::LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  mlir::llk::mapping::CoveringSearch search(graph, target, *module.getContext(),
                                            layoutContext, options);
  return search.search();
}

llvm::Expected<FrameworkFixture> buildFrameworkFixture(MappingTarget &target) {
  FrameworkFixture fixture;
  fixture.context = std::make_unique<mlir::MLIRContext>(frameworkRegistry());
  fixture.context->loadAllAvailableDialects();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
      llvm::MemoryBuffer::getFile(
          std::string(LLK_SOURCE_DIR) +
          "/test/Execution/Inputs/framework/add_mul.mlir");
  if (!input)
    return llvm::createStringError(input.getError(), "cannot read add_mul");
  fixture.source = mlir::parseSourceString<mlir::ModuleOp>(
      (*input)->getBuffer(), fixture.context.get());
  if (!fixture.source)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot parse add_mul");

  auto kernel = mlir::llk::perf::findMicroKernel(*fixture.source, "add_mul");
  if (!kernel)
    return kernel.takeError();
  auto graph = mlir::llk::mapping::extractWorkloadGraph(*kernel);
  if (!graph)
    return graph.takeError();
  fixture.graph = std::move(*graph);

  auto search = searchFramework(*fixture.source, fixture.graph, target);
  if (!search)
    return search.takeError();
  if (search->plans.empty()) {
    std::string reason = "no complete plan";
    if (!search->frontier.diagnostics.empty())
      reason += ": " + search->frontier.diagnostics.front().message;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), reason);
  }
  fixture.plan = search->plans.front();

  llk::MappedCompileOptions compile;
  compile.stop = llk::MappedStop::MappedMicro;
  compile.requireExecutable = false;
  auto bound =
      llk::compileMappedKernel(*fixture.source, target, fixture.plan, compile);
  if (!bound)
    return bound.takeError();
  fixture.boundModule = std::move(bound->module);
  auto boundKernel =
      mlir::llk::perf::findMicroKernel(fixture.boundModule.get(), "add_mul");
  if (!boundKernel)
    return boundKernel.takeError();
  fixture.kernel = *boundKernel;
  auto analysis = mlir::llk::perf::analyzeSelectedKernel(
      fixture.kernel, target.machine(), /*requireComplete=*/true, fixture.plan,
      fixture.graph);
  if (!analysis)
    return analysis.takeError();
  fixture.analysis = std::move(*analysis);
  return fixture;
}

CapabilityReceipt receiptFor(const FrameworkFixture &fixture,
                             const WorkloadContract &contract,
                             const MappingTarget &target,
                             const llk::MappedCompilation *compilation) {
  return mlir::llk::framework::describeCapabilities(fixture.kernel, contract,
                                                    target, &fixture.analysis,
                                                    &fixture.plan, compilation);
}

std::string selectedEngine(const CoveringPlan &plan) {
  for (const auto &placement : plan.placements)
    for (const auto &binding : placement.computeBindings)
      if (binding.first() == "vector_engine")
        return binding.second;
  return {};
}

/// Searches an issue129 fixture against a caller-supplied target, so a test can
/// vary machine data while keeping the source, rules and layouts fixed.
llvm::Expected<MappingSearchResult>
searchResourceCase(issue129::ResourceCase &c, MappingTarget &target,
                   unsigned topK) {
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = topK;
  options.requireCompleteEvaluation = true;
  options.evaluateCompletePlan = [&](const CoveringPlan &proposal) {
    return mlir::llk::mapping::evaluateCompletePlan(
        *c.source, c.graph, target, proposal, BindContract::Partial,
        std::nullopt, mlir::llk::mapping::AnalysisRequirement::Complete);
  };
  mlir::llk::mapping::LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  mlir::llk::mapping::CoveringSearch search(c.graph, target, *c.context,
                                            layoutContext, options);
  return search.search();
}

TEST(FrameworkTargetPortability,
     RenamingPreservesDecisionsAndCapabilityStatuses) {
  llvm::Expected<std::unique_ptr<MappingTarget>> base =
      mlir::llk::target::generic_accel::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(base)) << llvm::toString(base.takeError());
  RenamedTarget renamed(**base, "renamed-accelerator");

  auto baseFixture = buildFrameworkFixture(**base);
  ASSERT_TRUE(static_cast<bool>(baseFixture))
      << llvm::toString(baseFixture.takeError());
  auto renamedFixture = buildFrameworkFixture(renamed);
  ASSERT_TRUE(static_cast<bool>(renamedFixture))
      << llvm::toString(renamedFixture.takeError());

  EXPECT_TRUE(baseFixture->analysis.complete);
  EXPECT_TRUE(renamedFixture->analysis.complete);
  EXPECT_EQ(baseFixture->plan.id, renamedFixture->plan.id);

  // Every legal decision is the same: same rule, executor, compute node,
  // memories and port bindings.
  ASSERT_EQ(baseFixture->plan.placements.size(),
            renamedFixture->plan.placements.size());
  for (size_t index = 0; index < baseFixture->plan.placements.size(); ++index) {
    const auto &lhs = baseFixture->plan.placements[index];
    const auto &rhs = renamedFixture->plan.placements[index];
    EXPECT_EQ(lhs.rule, rhs.rule);
    EXPECT_EQ(lhs.executor, rhs.executor);
    EXPECT_EQ(lhs.memories, rhs.memories);
    EXPECT_EQ(lhs.computeBindings, rhs.computeBindings);
    EXPECT_EQ(lhs.portMemoryBindings, rhs.portMemoryBindings);
  }

  WorkloadContract contract = addMulContract();
  CapabilityReceipt baseReceipt =
      receiptFor(*baseFixture, contract, **base, /*compilation=*/nullptr);
  CapabilityReceipt renamedReceipt =
      receiptFor(*renamedFixture, contract, renamed, /*compilation=*/nullptr);

  EXPECT_EQ(baseReceipt.structural, renamedReceipt.structural);
  EXPECT_EQ(baseReceipt.interpretation, renamedReceipt.interpretation);
  EXPECT_EQ(baseReceipt.mapping, renamedReceipt.mapping);
  EXPECT_EQ(baseReceipt.physicalPlan, renamedReceipt.physicalPlan);
  EXPECT_EQ(baseReceipt.referenceLowering, renamedReceipt.referenceLowering);
  EXPECT_EQ(baseReceipt.selectedLowering, renamedReceipt.selectedLowering);
  EXPECT_EQ(baseReceipt.invocation, renamedReceipt.invocation);
  EXPECT_EQ(baseReceipt.sourceIdentity, renamedReceipt.sourceIdentity);
  EXPECT_EQ(baseReceipt.contractIdentity, renamedReceipt.contractIdentity);
  EXPECT_EQ(baseReceipt.planId, renamedReceipt.planId);
  // The target identity is the one thing that must move.
  EXPECT_NE(baseReceipt.targetIdentity, renamedReceipt.targetIdentity);
  EXPECT_EQ(baseReceipt.selectedLowering, CapabilityStatus::Unsupported);
  EXPECT_EQ(renamedReceipt.selectedLowering, CapabilityStatus::Unsupported);
}

TEST(FrameworkTargetPortability, RenamedAvx2PreservesBaseBackendCapability) {
  llvm::Expected<std::unique_ptr<MappingTarget>> base =
      mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(base)) << llvm::toString(base.takeError());
  RenamedTarget renamed(**base, "renamed-avx2");

  auto baseFixture = buildFrameworkFixture(**base);
  ASSERT_TRUE(static_cast<bool>(baseFixture))
      << llvm::toString(baseFixture.takeError());
  auto renamedFixture = buildFrameworkFixture(renamed);
  ASSERT_TRUE(static_cast<bool>(renamedFixture))
      << llvm::toString(renamedFixture.takeError());
  EXPECT_EQ(baseFixture->plan.id, renamedFixture->plan.id);

  llk::MappedCompileOptions compile;
  compile.entrySymbol = "add_mul";
  compile.backend = llk::MappedBackend::SelectedTarget;
  compile.stop = llk::MappedStop::Lowered;
  auto baseLowered = llk::compileMappedKernel(*baseFixture->source, **base,
                                              baseFixture->plan, compile);
  ASSERT_TRUE(static_cast<bool>(baseLowered))
      << llvm::toString(baseLowered.takeError());
  auto renamedLowered = llk::compileMappedKernel(
      *renamedFixture->source, renamed, renamedFixture->plan, compile);
  ASSERT_TRUE(static_cast<bool>(renamedLowered))
      << llvm::toString(renamedLowered.takeError());

  for (const llk::MappedCompilation *compilation :
       {&*baseLowered, &*renamedLowered}) {
    EXPECT_GT(compilation->backendGroupsRealized, 0u);
    EXPECT_EQ(compilation->referenceGroupsLowered, 0u);
    ASSERT_TRUE(compilation->codegenRequirements.has_value());
    EXPECT_EQ(compilation->codegenRequirements->architecture, "x86_64");
  }

  WorkloadContract contract = addMulContract();
  CapabilityReceipt baseReceipt =
      receiptFor(*baseFixture, contract, **base, &*baseLowered);
  CapabilityReceipt renamedReceipt =
      receiptFor(*renamedFixture, contract, renamed, &*renamedLowered);
  EXPECT_EQ(baseReceipt.selectedLowering, CapabilityStatus::Supported);
  EXPECT_EQ(renamedReceipt.selectedLowering, CapabilityStatus::Supported);
  EXPECT_EQ(baseReceipt.invocation, renamedReceipt.invocation);
  EXPECT_NE(baseReceipt.targetIdentity, renamedReceipt.targetIdentity);
}

TEST(FrameworkTargetPortability, TopologyDataChangesLegalCoverings) {
  auto twoCompute = issue129::resourceCase("two-compute");
  ASSERT_TRUE(static_cast<bool>(twoCompute))
      << llvm::toString(twoCompute.takeError());

  auto baseSearch = searchResourceCase(*twoCompute, *twoCompute->target, 0);
  ASSERT_TRUE(static_cast<bool>(baseSearch))
      << llvm::toString(baseSearch.takeError());
  std::set<std::string> baseEngines;
  for (const CoveringPlan &plan : baseSearch->plans)
    baseEngines.insert(selectedEngine(plan));
  EXPECT_EQ(baseEngines, (std::set<std::string>{"vpu.a", "vpu.b"}));

  // Removing one same-kind engine changes the legal set from machine data.
  mlir::llk::machine::MachineModel oneEngine = twoCompute->target->machine();
  llvm::erase_if(oneEngine.computes,
                 [](const auto &node) { return node.id == "vpu.b"; });
  FileMappingTarget oneEngineTarget(
      "issue129", std::move(oneEngine), twoCompute->target->layouts(),
      twoCompute->target->rules(), {"issue129_vector_add"});
  auto reducedSearch = searchResourceCase(*twoCompute, oneEngineTarget, 0);
  ASSERT_TRUE(static_cast<bool>(reducedSearch))
      << llvm::toString(reducedSearch.takeError());
  std::set<std::string> reducedEngines;
  for (const CoveringPlan &plan : reducedSearch->plans)
    reducedEngines.insert(selectedEngine(plan));
  EXPECT_EQ(reducedEngines, (std::set<std::string>{"vpu.a"}));

  // A capacity edit changes which rule is feasible, and shrinking the legal
  // memory past the plan's live peak changes the rejection reason.
  auto capacity = issue129::resourceCase("capacity-topk");
  ASSERT_TRUE(static_cast<bool>(capacity))
      << llvm::toString(capacity.takeError());
  auto baseCapacity = searchResourceCase(*capacity, *capacity->target, 1);
  ASSERT_TRUE(static_cast<bool>(baseCapacity))
      << llvm::toString(baseCapacity.takeError());
  ASSERT_EQ(baseCapacity->plans.size(), 1u);
  EXPECT_EQ(baseCapacity->plans.front().placements.front().rule, "r.large");
  EXPECT_GT(baseCapacity->frontier.plansRejectedByCapacity, 0u);

  mlir::llk::machine::MachineModel roomy = capacity->target->machine();
  for (auto &memory : roomy.memories)
    if (memory.id == "sram.0")
      memory.capacityBytes = 2048;
  FileMappingTarget roomyTarget(
      "issue129", std::move(roomy), capacity->target->layouts(),
      capacity->target->rules(), {"issue129_vector_add"});
  auto roomySearch = searchResourceCase(*capacity, roomyTarget, 0);
  ASSERT_TRUE(static_cast<bool>(roomySearch))
      << llvm::toString(roomySearch.takeError());
  bool smallIsFeasible = false;
  for (const CoveringPlan &plan : roomySearch->plans)
    for (const auto &placement : plan.placements)
      smallIsFeasible |= placement.rule == "r.small";
  EXPECT_TRUE(smallIsFeasible)
      << "enlarging SRAM must make the cheaper rule feasible";

  mlir::llk::machine::MachineModel tight = capacity->target->machine();
  for (auto &memory : tight.memories)
    if (memory.id == "dram.0")
      memory.capacityBytes = 512;
  FileMappingTarget tightTarget(
      "issue129", std::move(tight), capacity->target->layouts(),
      capacity->target->rules(), {"issue129_vector_add"});
  auto tightSearch = searchResourceCase(*capacity, tightTarget, 1);
  ASSERT_TRUE(static_cast<bool>(tightSearch))
      << llvm::toString(tightSearch.takeError());
  EXPECT_TRUE(tightSearch->plans.empty());
  EXPECT_TRUE(tightSearch->frontier.has(
      mlir::llk::mapping::DiagnosticCode::MemoryCapacityExceeded));
  EXPECT_FALSE(tightSearch->frontier.diagnostics.empty());
}

} // namespace
