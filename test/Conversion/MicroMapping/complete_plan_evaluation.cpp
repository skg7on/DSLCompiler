//===- complete_plan_evaluation.cpp - Evaluate a complete proposal (R7) ---===//
//
// Issue #129, task R7. `evaluateCompletePlan` turns a cheap search *proposal*
// into exactly one of a finalized, physically validated, schedulable plan or a
// stable-coded rejection. These tests exercise it directly -- rather than
// through the search's callback plumbing (covered by the
// `MappingCoveringSearchTest` regressions) -- against the shared
// `capacity-topk` fixture, whose two rules bind an `8x8xf32` result with four
// live versions (1024 bytes) to a 512-byte SRAM and a 4096-byte DRAM.
//
//===----------------------------------------------------------------------===//

#include "CompletePlanEvaluation.h"

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/StoragePlan.h"

#include "resource_regression_fixture.h"

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

} // namespace

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
