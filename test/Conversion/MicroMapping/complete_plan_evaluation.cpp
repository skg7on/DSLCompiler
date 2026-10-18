//===- complete_plan_evaluation.cpp - Evaluate a complete proposal (R7) ---===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Issue #129, task R7. `evaluateCompletePlan` turns a cheap search *proposal*
// into exactly one of a finalized, physically validated, schedulable plan or a
// stable-coded rejection. These tests exercise it directly -- rather than
// through the search's callback plumbing (covered by the
// `MappingCoveringSearchTest` regressions) -- against the shared
// `capacity-topk` fixture, whose two rules bind an `8x8xf32` result with four
// live versions (1024 bytes) to a 512-byte SRAM and a 4096-byte DRAM.

#include "CompletePlanEvaluation.h"

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/StoragePlan.h"
#include "LLK/Perf/SelectedKernelAnalysis.h"

#include "resource_regression_fixture.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

using namespace mlir::llk::mapping;

namespace {

/// A proposal for every complete covering of `graph`, taken from the ordinary
/// search with *no* evaluator -- so a proposal is exactly what the search
/// synthesizes before this stage runs.
std::vector<CoveringPlan> proposalsFor(issue129::ResourceCase &c) {
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 0; // every complete covering
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  CoveringSearch search(c.graph, *c.target, *c.context, layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result) {
    ADD_FAILURE() << llvm::toString(result.takeError());
    return {};
  }
  return result->plans;
}

std::string printed(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module->print(stream);
  return stream.str();
}

// Keep the placement/routing fixture fixed and vary only the store destination
// the bound-kernel analyzer must validate. No memory node models `acc`.
void setStoreDestination(issue129::ResourceCase &c, llvm::StringRef memory) {
  auto destination = mlir::parseAttribute("#micro.memory<" + memory.str() + ">",
                                          c.context.get());
  ASSERT_TRUE(destination);
  mlir::Operation *kernel = nullptr;
  c.source->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "micro.kernel")
      kernel = op;
    if (op->getName().getStringRef() == "micro.tile_store")
      op->setAttr("dst_memory", destination);
  });
  ASSERT_NE(kernel, nullptr);
  auto graph = extractWorkloadGraph(kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());
  c.graph = std::move(*graph);
}

} // namespace

TEST(CompletePlanEvaluationTest, ExecutableRejectsSelectedAnalysisFailure) {
  auto c = issue129::resourceCase("two-hop");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());
  setStoreDestination(*c, "acc");
  const auto proposals = proposalsFor(*c);
  ASSERT_EQ(proposals.size(), 1u);
  auto evaluated =
      evaluateCompletePlan(*c->source, c->graph, *c->target, proposals.front(),
                           BindContract::Executable);
  ASSERT_TRUE(static_cast<bool>(evaluated))
      << llvm::toString(evaluated.takeError());
  EXPECT_FALSE(evaluated->plan.has_value());
  ASSERT_TRUE(evaluated->rejection.has_value());
  EXPECT_EQ(evaluated->rejection->code,
            DiagnosticCode::UnsupportedMaterialization);
  EXPECT_NE(evaluated->rejection->message.find("acc"), std::string::npos);
}

TEST(CompletePlanEvaluationTest, ExecutableRejectsIncompleteSelectedAnalysis) {
  auto c = issue129::resourceCase("two-hop");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());
  // The ordinary fixture now models its DRAM-to-L2 store. Deliberately use
  // an unmodeled DRAM-to-SRAM boundary to retain the negative control.
  setStoreDestination(*c, "sram");
  const auto proposals = proposalsFor(*c);
  ASSERT_EQ(proposals.size(), 1u);
  auto evaluated =
      evaluateCompletePlan(*c->source, c->graph, *c->target, proposals.front(),
                           BindContract::Executable);
  ASSERT_TRUE(static_cast<bool>(evaluated))
      << llvm::toString(evaluated.takeError());
  EXPECT_FALSE(evaluated->plan.has_value());
  ASSERT_TRUE(evaluated->rejection.has_value());
  EXPECT_EQ(evaluated->rejection->code,
            DiagnosticCode::UnsupportedMaterialization);
  EXPECT_NE(evaluated->rejection->message.find("incomplete"),
            std::string::npos);
}

TEST(CompletePlanEvaluationTest, PartialAnalysisFailureStillEnforcesBudget) {
  auto c = issue129::resourceCase("two-hop");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());
  setStoreDestination(*c, "acc");
  const auto proposals = proposalsFor(*c);
  ASSERT_EQ(proposals.size(), 1u);
  auto tight = evaluateCompletePlan(*c->source, c->graph, *c->target,
                                    proposals.front(), BindContract::Partial,
                                    /*memoryBudgetBytes=*/1,
                                    AnalysisRequirement::PartialReport);
  ASSERT_TRUE(static_cast<bool>(tight)) << llvm::toString(tight.takeError());
  EXPECT_FALSE(tight->plan.has_value());
  ASSERT_TRUE(tight->rejection.has_value());
  EXPECT_EQ(tight->rejection->code, DiagnosticCode::MemoryCapacityExceeded);
  EXPECT_NE(tight->rejection->message.find("budget"), std::string::npos);

  // Model incompleteness still permits an explicit partial artifact when its
  // actual finalized live footprint fits. Do not ban the fallback to hide the
  // missing budget check.
  auto roomy = evaluateCompletePlan(*c->source, c->graph, *c->target,
                                    proposals.front(), BindContract::Partial,
                                    /*memoryBudgetBytes=*/2048,
                                    AnalysisRequirement::PartialReport);
  ASSERT_TRUE(static_cast<bool>(roomy)) << llvm::toString(roomy.takeError());
  EXPECT_FALSE(roomy->rejection.has_value());
  ASSERT_TRUE(roomy->plan.has_value());
  EXPECT_FALSE(roomy->plan->diagnostics.warnings.empty());
}

// A physically legal proposal evaluates into a finalized plan: storage decided
// (allocations and movement hops), the physical verdict recorded, and a cost
// derived from the selected static analysis. The DRAM binding's 1024-byte
// footprint fits its 4096-byte memory.
TEST(CompletePlanEvaluationTest, FinalizesALegalProposal) {
  auto c = issue129::resourceCase("capacity-topk");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());
  std::vector<CoveringPlan> proposals = proposalsFor(*c);
  ASSERT_FALSE(proposals.empty());

  std::optional<CoveringPlan> dram;
  for (const CoveringPlan &proposal : proposals)
    if (proposal.placements.front().rule == "r.large")
      dram = proposal;
  ASSERT_TRUE(dram.has_value());

  llvm::Expected<CompletePlanEvaluation> evaluated = evaluateCompletePlan(
      *c->source, c->graph, *c->target, *dram, BindContract::Partial);
  ASSERT_TRUE(static_cast<bool>(evaluated))
      << llvm::toString(evaluated.takeError());
  ASSERT_FALSE(evaluated->rejection.has_value());
  ASSERT_TRUE(evaluated->plan.has_value());
  const CoveringPlan &plan = *evaluated->plan;
  EXPECT_TRUE(plan.diagnostics.physicalComplete);
  EXPECT_FALSE(plan.allocations.empty());
  EXPECT_NE(plan.graphHash, 0u);
  EXPECT_NE(plan.targetHash, 0u);
  // The final cost is the shared analysis's, so it is non-zero and carries the
  // DRAM bytes the movement does.
  EXPECT_GT(plan.totalCost.latencyCycles, 0.0);
  EXPECT_EQ(plan.scoreSource, PlanScoreSource::Schedule);
}

// A proposal whose physical occupancy overflows its memory is a *rejection*
// with the stable capacity code, not an `llvm::Error`: the search must be able
// to keep enumerating rather than stop.
TEST(CompletePlanEvaluationTest, RejectsAnOverCapacityProposal) {
  auto c = issue129::resourceCase("capacity-topk");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());
  std::vector<CoveringPlan> proposals = proposalsFor(*c);
  ASSERT_FALSE(proposals.empty());

  std::optional<CoveringPlan> sram;
  for (const CoveringPlan &proposal : proposals)
    if (proposal.placements.front().rule == "r.small")
      sram = proposal;
  ASSERT_TRUE(sram.has_value());

  llvm::Expected<CompletePlanEvaluation> evaluated = evaluateCompletePlan(
      *c->source, c->graph, *c->target, *sram, BindContract::Partial);
  ASSERT_TRUE(static_cast<bool>(evaluated))
      << llvm::toString(evaluated.takeError());
  ASSERT_FALSE(evaluated->plan.has_value());
  ASSERT_TRUE(evaluated->rejection.has_value());
  EXPECT_EQ(evaluated->rejection->code, DiagnosticCode::MemoryCapacityExceeded);
  EXPECT_NE(evaluated->rejection->message.find("sram.0"), std::string::npos);
}

// The whole-plan byte budget (design §9.3) is enforced on the *finalized*
// plan's live bytes (issue #129, task R7 review). The search's per-partial
// charge is an over-estimate it deliberately no longer rejects on, so the
// budget -- which the plan report still prints -- must be checked where the
// honest peak exists, otherwise it is silently unenforced on every
// evaluator-backed run.
TEST(CompletePlanEvaluationTest, EnforcesTheWholePlanByteBudget) {
  auto c = issue129::resourceCase("capacity-topk");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());
  std::vector<CoveringPlan> proposals = proposalsFor(*c);
  std::optional<CoveringPlan> dram;
  for (const CoveringPlan &proposal : proposals)
    if (proposal.placements.front().rule == "r.large")
      dram = proposal;
  ASSERT_TRUE(dram.has_value());

  // The DRAM binding's live peak is 1024 bytes: a 512-byte budget rejects it, a
  // 2048-byte budget admits it.
  llvm::Expected<CompletePlanEvaluation> tight = evaluateCompletePlan(
      *c->source, c->graph, *c->target, *dram, BindContract::Partial,
      /*memoryBudgetBytes=*/512);
  ASSERT_TRUE(static_cast<bool>(tight)) << llvm::toString(tight.takeError());
  ASSERT_TRUE(tight->rejection.has_value());
  EXPECT_EQ(tight->rejection->code, DiagnosticCode::MemoryCapacityExceeded);
  EXPECT_NE(tight->rejection->message.find("budget"), std::string::npos);

  llvm::Expected<CompletePlanEvaluation> roomy = evaluateCompletePlan(
      *c->source, c->graph, *c->target, *dram, BindContract::Partial,
      /*memoryBudgetBytes=*/2048);
  ASSERT_TRUE(static_cast<bool>(roomy)) << llvm::toString(roomy.takeError());
  EXPECT_FALSE(roomy->rejection.has_value());
  ASSERT_TRUE(roomy->plan.has_value());
}

// A hard storage failure that is *not* a capacity overflow must not be labelled
// one (issue #129, task R7 review): the stable code is a machine-readable
// interface, and the search keys `plansRejectedByCapacity` off it, so calling
// every storage failure a capacity refusal both lies and inflates the tally. A
// proposal with no placement at all is a structural refusal (the graph is not
// covered), so the code is the materialization code.
TEST(CompletePlanEvaluationTest, AStructuralFailureIsNotLabelledCapacity) {
  auto c = issue129::resourceCase("capacity-topk");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());
  std::vector<CoveringPlan> proposals = proposalsFor(*c);
  std::optional<CoveringPlan> dram;
  for (const CoveringPlan &proposal : proposals)
    if (proposal.placements.front().rule == "r.large")
      dram = proposal;
  ASSERT_TRUE(dram.has_value());

  CoveringPlan uncovered = *dram;
  uncovered.placements.clear(); // no placement covers the graph's node
  llvm::Expected<CompletePlanEvaluation> evaluated = evaluateCompletePlan(
      *c->source, c->graph, *c->target, uncovered, BindContract::Partial);
  ASSERT_TRUE(static_cast<bool>(evaluated))
      << llvm::toString(evaluated.takeError());
  ASSERT_TRUE(evaluated->rejection.has_value());
  EXPECT_EQ(evaluated->rejection->code,
            DiagnosticCode::UnsupportedMaterialization);
}

// Issue #129, task R7 review: a selected-kernel analysis that declares itself
// incomplete must not be scored as if it were complete. Its ordered reasons are
// recorded on the plan -- one warning each -- so the verdict travels with the
// plan instead of being discarded, exactly as `micro-perf` reports them on the
// same kernel.
TEST(CompletePlanEvaluationTest, AnIncompleteAnalysisIsRecordedNotSwallowed) {
  mlir::llk::perf::SelectedKernelAnalysis analysis;
  analysis.complete = false;
  analysis.incompleteReasons = {"a loop has non-static bounds",
                                "a value is charged as zero"};

  CoveringPlan plan;
  recordAnalysisCompleteness(analysis, plan);
  ASSERT_EQ(plan.diagnostics.warnings.size(), 2u);
  EXPECT_NE(plan.diagnostics.warnings[0].find("non-static bounds"),
            std::string::npos);
  EXPECT_NE(plan.diagnostics.warnings[1].find("charged as zero"),
            std::string::npos);
}

// The complement: a complete analysis records nothing, so the recording never
// adds noise to a plan whose work the extractor described exactly.
TEST(CompletePlanEvaluationTest, ACompleteAnalysisRecordsNoVerdict) {
  mlir::llk::perf::SelectedKernelAnalysis analysis;
  analysis.complete = true;

  CoveringPlan plan;
  recordAnalysisCompleteness(analysis, plan);
  EXPECT_TRUE(plan.diagnostics.warnings.empty());
}

// Issue #129 review finding 6: a valid provider hit the search ranked on must
// reach the plan's *final* cost, not be overwritten by the static machine
// formula. Before the fix the search ranked a plan at the measured 1,000 cycles
// (its `accumulatedCost`) while the completion analysis re-charged the static
// estimate, so an enabled measurement stopped affecting final costs/ranking.
TEST(CompletePlanEvaluationTest, AProviderHitReachesTheFinalCost) {
  class FixedProvider : public LatencyProvider {
  public:
    using LatencyProvider::lookupCycles;
    std::optional<double> lookupCycles(const OperationSignature &,
                                       const TargetContext &) const override {
      return 1000.0;
    }
  };
  FixedProvider provider;
  auto c = issue129::resourceCase("two-compute", &provider);
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 8;
  options.enableLatencyCache = true;
  auto result = issue129::searchCase(*c, options);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());

  for (const CoveringPlan &plan : result->plans) {
    ASSERT_FALSE(plan.placements.empty());
    EXPECT_TRUE(plan.placements.front().measuredCycles.has_value())
        << "the measured value must travel with the placement";
    EXPECT_DOUBLE_EQ(plan.accumulatedCost.latencyCycles, 1000.0);
    EXPECT_DOUBLE_EQ(plan.totalCost.latencyCycles, 1000.0)
        << "the final schedule must charge the measured duration";
  }
}

// Evaluation is deterministic and idempotent: two evaluations of one proposal
// produce the same decisions and therefore the same decision-only id, and
// re-finalizing a returned plan reproduces that id (the invariant the search's
// idempotence check relies on).
TEST(CompletePlanEvaluationTest, IsDeterministicAndIdempotent) {
  auto c = issue129::resourceCase("capacity-topk");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());
  std::vector<CoveringPlan> proposals = proposalsFor(*c);
  ASSERT_FALSE(proposals.empty());

  std::optional<CoveringPlan> dram;
  for (const CoveringPlan &proposal : proposals)
    if (proposal.placements.front().rule == "r.large")
      dram = proposal;
  ASSERT_TRUE(dram.has_value());

  auto first = evaluateCompletePlan(*c->source, c->graph, *c->target, *dram,
                                    BindContract::Partial);
  ASSERT_TRUE(static_cast<bool>(first)) << llvm::toString(first.takeError());
  auto second = evaluateCompletePlan(*c->source, c->graph, *c->target, *dram,
                                     BindContract::Partial);
  ASSERT_TRUE(static_cast<bool>(second)) << llvm::toString(second.takeError());
  ASSERT_TRUE(first->plan.has_value());
  ASSERT_TRUE(second->plan.has_value());
  EXPECT_EQ(first->plan->id, second->plan->id);
  EXPECT_DOUBLE_EQ(first->plan->totalCost.latencyCycles,
                   second->plan->totalCost.latencyCycles);

  // Re-finalizing a returned plan reproduces its id.
  CoveringPlan probe = *first->plan;
  llvm::Error refinalize =
      finalizeStoragePlan(c->graph, probe, c->target->machine());
  const bool refinalizeFailed = static_cast<bool>(refinalize);
  const std::string refinalizeReason =
      refinalizeFailed ? llvm::toString(std::move(refinalize)) : std::string();
  llvm::consumeError(std::move(refinalize));
  EXPECT_FALSE(refinalizeFailed) << refinalizeReason;
  EXPECT_EQ(probe.id, first->plan->id);
}

// A positive attach: a proposal whose bound kernel the machine can model
// carries the derived event snapshot, and `buildPlanEvents` consumes it as the
// plan's own stream rather than accumulating rule-local estimates. This is the
// path the shipped acceptance fixtures cannot reach (their boundary copy names
// an engine the machine does not model), so it is proven here on a kernel whose
// compute event names the engine the plan selected.
TEST(CompletePlanEvaluationTest,
     AttachesTheDerivedEventSnapshotWhenItVerifies) {
  auto c = issue129::resourceCase("two-compute");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());
  std::vector<CoveringPlan> proposals = proposalsFor(*c);
  ASSERT_FALSE(proposals.empty());

  const CoveringPlan &proposal = proposals.front();
  const std::string selected =
      proposal.placements.front().computeBindings.lookup("vector_engine");
  ASSERT_FALSE(selected.empty());

  llvm::Expected<CompletePlanEvaluation> evaluated = evaluateCompletePlan(
      *c->source, c->graph, *c->target, proposal, BindContract::Partial);
  ASSERT_TRUE(static_cast<bool>(evaluated))
      << llvm::toString(evaluated.takeError());
  ASSERT_TRUE(evaluated->plan.has_value());
  ASSERT_TRUE(evaluated->plan->analysisEvents.has_value());
  EXPECT_FALSE(evaluated->plan->analysisEvents->events.empty());

  llvm::Expected<PlanEventDAG> events =
      buildPlanEvents(*evaluated->plan, c->target->machine());
  ASSERT_TRUE(static_cast<bool>(events)) << llvm::toString(events.takeError());
  EXPECT_EQ(events->source, PlanEventSource::Snapshot);
  ASSERT_FALSE(events->events.empty());
  // The consumed stream is the materialized work's: the event names the engine
  // this plan selected, not the executor's first attachment.
  EXPECT_EQ(events->events.front().event.kind, CostEventKind::Compute);
  EXPECT_EQ(events->events.front().event.resource, selected);
}

// The two binding contexts are deliberately different (issue #129, task R7). A
// *proposal* that is not executable under `BindContract::Executable` is a
// candidate rejection -- the search records it and keeps enumerating, so a
// legal covering of the same graph is still found. A direct `bindPlan` under
// the same contract keeps failing hard; that behaviour is pinned by
// `PlanBinder.ExecutableContractRefusesAPlanThatOmitsADecision` and
// `PlanBinder.ExecutableWithoutAMaterializerIsRejected`.
TEST(CompletePlanEvaluationTest, ExecutableRefusalIsACandidateRejection) {
  auto c = issue129::resourceCase("capacity-topk");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 0;
  options.requireCompleteEvaluation = true;
  options.evaluateCompletePlan = [&c](const CoveringPlan &proposal) {
    return evaluateCompletePlan(*c->source, c->graph, *c->target, proposal,
                                BindContract::Executable);
  };
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  CoveringSearch search(c->graph, *c->target, *c->context, layoutContext,
                        options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  // The over-capacity proposal was rejected, not fatal: the search continued
  // and retained the executable covering.
  ASSERT_EQ(result->plans.size(), 1u);
  EXPECT_EQ(result->plans.front().placements.front().rule, "r.large");
  EXPECT_GT(result->frontier.plansRejectedByCapacity, 0u);
  EXPECT_FALSE(result->searchTruncated);
}

// The complementary shape: when *every* proposal is refused, the search still
// returns a result (with no plans and the refusal on the frontier) rather than
// an `llvm::Error`. `missing-memory`'s rule binds no memory over an occurrence
// the machine offers two same-kind nodes for, so the plan can never be
// physically complete and the executable binding refuses it.
TEST(CompletePlanEvaluationTest, ARefusedProposalIsNotASearchError) {
  auto c = issue129::resourceCase("missing-memory");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());

  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 0;
  options.requireCompleteEvaluation = true;
  options.evaluateCompletePlan = [&c](const CoveringPlan &proposal) {
    return evaluateCompletePlan(*c->source, c->graph, *c->target, proposal,
                                BindContract::Executable);
  };
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  CoveringSearch search(c->graph, *c->target, *c->context, layoutContext,
                        options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_FALSE(result->frontier.diagnostics.empty());
  bool refused = false;
  for (const Diagnostic &diagnostic : result->frontier.diagnostics)
    refused |= diagnostic.code == DiagnosticCode::UnsupportedMaterialization;
  EXPECT_TRUE(refused);
}

// The evaluation binds a preview onto a *private clone*: the source module is
// byte-identical before and after, however many times a proposal is evaluated.
TEST(CompletePlanEvaluationTest, LeavesTheSourceModuleUnchanged) {
  auto c = issue129::resourceCase("capacity-topk");
  ASSERT_TRUE(static_cast<bool>(c)) << llvm::toString(c.takeError());
  std::vector<CoveringPlan> proposals = proposalsFor(*c);
  ASSERT_FALSE(proposals.empty());

  const std::string before = printed(*c->source);
  for (const CoveringPlan &proposal : proposals)
    for (unsigned repeat = 0; repeat < 2; ++repeat)
      ASSERT_TRUE(static_cast<bool>(evaluateCompletePlan(
          *c->source, c->graph, *c->target, proposal, BindContract::Partial)));
  EXPECT_EQ(before, printed(*c->source));
}

namespace {
void addUnknownComputation(mlir::ModuleOp module) {
  auto kernel = mlir::llk::perf::findMicroKernel(module, "");
  ASSERT_TRUE(bool(kernel)) << llvm::toString(kernel.takeError());
  auto *context = (*kernel)->getContext();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  mlir::OpBuilder builder(context);
  builder.setInsertionPoint((*kernel)->getRegion(0).front().getTerminator());
  mlir::Value one = mlir::arith::ConstantOp::create(
      builder, (*kernel)->getLoc(), builder.getF32FloatAttr(1.0));
  mlir::arith::AddFOp::create(builder, (*kernel)->getLoc(), one, one);
}
} // namespace

TEST(CompletePlanEvaluationTest, UnknownComputationCannotEnterTopK) {
  auto c = issue129::resourceCase("two-compute");
  ASSERT_TRUE(bool(c)) << llvm::toString(c.takeError());
  addUnknownComputation(*c->source);
  MappingSearchOptions options;
  options.topK = 1;
  auto result = issue129::searchCase(*c, options);
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  bool named = false;
  for (const auto &diagnostic : result->frontier.diagnostics)
    named |= diagnostic.code == DiagnosticCode::UnsupportedMaterialization &&
             diagnostic.message.find("unmodeled_leaf: arith.addf") !=
                 std::string::npos;
  EXPECT_TRUE(named);
}

TEST(CompletePlanEvaluationTest, AnalysisOnlyRejectsExtractionFailure) {
  auto c = issue129::resourceCase("two-hop");
  ASSERT_TRUE(bool(c)) << llvm::toString(c.takeError());
  setStoreDestination(*c, "acc");
  const auto proposals = proposalsFor(*c);
  ASSERT_EQ(proposals.size(), 1u);
  auto evaluated =
      evaluateCompletePlan(*c->source, c->graph, *c->target, proposals.front(),
                           BindContract::Partial);
  ASSERT_TRUE(bool(evaluated)) << llvm::toString(evaluated.takeError());
  EXPECT_FALSE(evaluated->plan.has_value());
  ASSERT_TRUE(evaluated->rejection.has_value());
  EXPECT_EQ(evaluated->rejection->code,
            DiagnosticCode::UnsupportedMaterialization);
  EXPECT_NE(evaluated->rejection->message.find("acc"), std::string::npos);
}

TEST(CompletePlanEvaluationTest, AnalysisOnlyRejectsMissingPhysicalFacts) {
  auto c = issue129::resourceCase("missing-memory");
  ASSERT_TRUE(bool(c)) << llvm::toString(c.takeError());
  MappingSearchOptions options;
  options.topK = 1;
  auto result = issue129::searchCase(*c, options);
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(result->frontier.has(DiagnosticCode::UnsupportedMaterialization));
}

TEST(CompletePlanEvaluationTest,
     AnalysisOnlyRanksCompleteModelsWithoutBackend) {
  auto c = issue129::resourceCase("two-compute");
  ASSERT_TRUE(bool(c)) << llvm::toString(c.takeError());
  MappingSearchOptions options;
  options.topK = 1;
  auto result = issue129::searchCase(*c, options);
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->plans.size(), 1u);
  EXPECT_TRUE(result->plans.front().diagnostics.physicalComplete);
  EXPECT_TRUE(result->plans.front().analysisEvents.has_value());
  // This fixture declares synthetic emitter keys; no executable lowering is
  // installed.
  auto emitter = c->target->createEmitter("issue129_vector_add");
  ASSERT_NE(emitter, nullptr);
  EXPECT_FALSE(emitter->hasLowering());
}

TEST(CompletePlanEvaluationTest,
     IncompleteProposalDoesNotMaskCompleteAlternative) {
  class EngineProvider : public LatencyProvider {
  public:
    std::optional<double> lookupCycles(const OperationSignature &signature,
                                       const TargetContext &) const override {
      return signature.compute.find("vpu.a") != std::string::npos ? 1.0 : 100.0;
    }
  } provider;
  auto c = issue129::resourceCase("two-compute", &provider);
  ASSERT_TRUE(bool(c)) << llvm::toString(c.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> incomplete(c->source->clone());
  addUnknownComputation(*incomplete);
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 1;
  options.enableSymmetryReduction = false;
  options.requireCompleteEvaluation = true;
  unsigned refused = 0;
  options.evaluateCompletePlan = [&](const CoveringPlan &proposal) {
    const bool cheap = proposal.placements.front().computeBindings.lookup(
                           "vector_engine") == "vpu.a";
    auto evaluated =
        evaluateCompletePlan(cheap ? *incomplete : *c->source, c->graph,
                             *c->target, proposal, BindContract::Partial);
    if (evaluated && evaluated->rejection)
      ++refused;
    return evaluated;
  };
  LayoutContext layout;
  layout.rank = 2;
  layout.elementType = "f32";
  CoveringSearch search(c->graph, *c->target, *c->context, layout, options);
  auto result = search.search();
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->plans.size(), 1u);
  EXPECT_EQ(result->plans.front().placements.front().computeBindings.lookup(
                "vector_engine"),
            "vpu.b");
  EXPECT_DOUBLE_EQ(result->plans.front().totalCost.latencyCycles, 100.0);
  EXPECT_GT(refused, 0u);
  EXPECT_FALSE(result->searchTruncated);
}

TEST(CompletePlanEvaluationTest, PartialReportIsInspectableButNotRankable) {
  auto c = issue129::resourceCase("two-compute");
  ASSERT_TRUE(bool(c)) << llvm::toString(c.takeError());
  addUnknownComputation(*c->source);
  const auto proposals = proposalsFor(*c);
  ASSERT_FALSE(proposals.empty());
  auto partial = evaluateCompletePlan(
      *c->source, c->graph, *c->target, proposals.front(),
      BindContract::Partial, std::nullopt, AnalysisRequirement::PartialReport);
  ASSERT_TRUE(bool(partial)) << llvm::toString(partial.takeError());
  ASSERT_TRUE(partial->plan.has_value());
  EXPECT_FALSE(partial->plan->materialized);
  bool named = false;
  for (const auto &warning : partial->plan->diagnostics.warnings)
    named |= warning.find("unmodeled_leaf: arith.addf") != std::string::npos;
  EXPECT_TRUE(named);

  MappingSearchOptions options;
  options.topK = 1;
  options.requireCompleteEvaluation = true;
  options.evaluateCompletePlan = [&](const CoveringPlan &proposal) {
    return evaluateCompletePlan(*c->source, c->graph, *c->target, proposal,
                                BindContract::Partial, std::nullopt,
                                AnalysisRequirement::PartialReport);
  };
  LayoutContext layout;
  layout.rank = 2;
  layout.elementType = "f32";
  CoveringSearch search(c->graph, *c->target, *c->context, layout, options);
  auto result = search.search();
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->plans.empty());
  EXPECT_TRUE(result->frontier.has(DiagnosticCode::UnsupportedMaterialization));
}

TEST(CompletePlanEvaluationTest, ExecutableCannotOptIntoPartialAnalysis) {
  auto c = issue129::resourceCase("two-compute");
  ASSERT_TRUE(bool(c)) << llvm::toString(c.takeError());
  addUnknownComputation(*c->source);
  const auto proposals = proposalsFor(*c);
  ASSERT_FALSE(proposals.empty());
  auto evaluated =
      evaluateCompletePlan(*c->source, c->graph, *c->target, proposals.front(),
                           BindContract::Executable, std::nullopt,
                           AnalysisRequirement::PartialReport);
  ASSERT_TRUE(bool(evaluated)) << llvm::toString(evaluated.takeError());
  EXPECT_FALSE(evaluated->plan.has_value());
  ASSERT_TRUE(evaluated->rejection.has_value());
  EXPECT_NE(evaluated->rejection->message.find("unmodeled_leaf: arith.addf"),
            std::string::npos);
}

TEST(CompletePlanEvaluationTest, CompleteRejectsMaterializerOmissions) {
  auto c = issue129::resourceCase("two-hop");
  ASSERT_TRUE(bool(c)) << llvm::toString(c.takeError());
  auto proposals = proposalsFor(*c);
  ASSERT_EQ(proposals.size(), 1u);
  auto proposal = proposals.front();
  ASSERT_EQ(proposal.connectionPlans.size(), 1u);
  ASSERT_EQ(proposal.connectionPlans.front().consumerPorts.size(), 1u);
  // A stale endpoint names a nonexistent second input on the store. Physical
  // storage and surviving IR can still analyze, but the selected rewire failed.
  proposal.connectionPlans.front().consumerPorts.front().index = 1;
  auto complete = evaluateCompletePlan(*c->source, c->graph, *c->target,
                                       proposal, BindContract::Partial);
  ASSERT_TRUE(bool(complete)) << llvm::toString(complete.takeError());
  EXPECT_FALSE(complete->plan.has_value());
  EXPECT_TRUE(complete->rejection.has_value());
  if (complete->rejection) {
    EXPECT_EQ(complete->rejection->code,
              DiagnosticCode::UnsupportedMaterialization);
    EXPECT_NE(complete->rejection->message.find("consumer_endpoint_unresolved"),
              std::string::npos);
  }

  auto partial = evaluateCompletePlan(
      *c->source, c->graph, *c->target, proposal, BindContract::Partial,
      std::nullopt, AnalysisRequirement::PartialReport);
  ASSERT_TRUE(bool(partial)) << llvm::toString(partial.takeError());
  ASSERT_TRUE(partial->plan.has_value());
  EXPECT_FALSE(partial->plan->materialized);
  bool named = false;
  for (const auto &warning : partial->plan->diagnostics.warnings)
    named |= warning.find("consumer_endpoint_unresolved") != std::string::npos;
  EXPECT_TRUE(named);
}
