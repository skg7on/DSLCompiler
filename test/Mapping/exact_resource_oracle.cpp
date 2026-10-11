//===- exact_resource_oracle.cpp - Independent exact-search oracle (R8)
//----===//
//
// Issue #129, task R8. The exact covering search's feasibility verdict must be
// provable without the search: this file enumerates the `joint-oracle`
// fixture's finite assignment space with *plain fixture arithmetic* --
// capacities and sizes stated as literals, a schedule of fixed event durations
// -- and never calls `CoveringSearch`, `finalizeStoragePlan`,
// `analyzeSelectedKernel` or a production key builder to decide what is
// feasible. The search's retained plans are then reduced to the same selection
// tuples and compared to the enumeration, so a search that keeps an infeasible
// covering, drops a feasible one, or lets insertion order decide either fails
// here.
//
// The fixture (issue #129, task R8): one worker `worker.0` with two attached
// vector engines `vpu.a`/`vpu.b`, an SRAM (512 bytes), an undersized L2 (128)
// and a DRAM (4096). Two chained `8x8xf32` vectors run four-stage pipelines, so
// each holds four simultaneous 256-byte versions (1024 bytes). The add's result
// memory is the rule's decision -- SRAM cannot hold 1024, DRAM can -- and its
// compute binding is one of the two engines. That is the whole product:
// compute (2) x memory (2) x route (2) = 8, well under the 64-combination cap.
//
// The independent arithmetic (mirrors the plan's specification):
//   needed = 4 * 256 = 1024
//   feasible iff needed <= memory.capacity
//            and (route != via-l2 or 256 <= route.middle_capacity)
// so only (vpu.a, dram.0, direct) and (vpu.b, dram.0, direct) survive: SRAM
// (512) cannot hold 1024, and the 256-byte value cannot traverse the 128-byte
// L2.
//
//===----------------------------------------------------------------------===//

#include "resource_regression_fixture.h"

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingPlan.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace mlir::llk::mapping;

/// One point of the finite assignment space: the engine that runs the add, the
/// memory its result lives in, and the route the value takes to its consumer.
struct Selection {
  std::string compute;
  std::string memory;
  std::string route;

  bool operator<(const Selection &other) const {
    return std::tie(compute, memory, route) <
           std::tie(other.compute, other.memory, other.route);
  }
  bool operator==(const Selection &other) const {
    return std::tie(compute, memory, route) ==
           std::tie(other.compute, other.memory, other.route);
  }
};

/// The fixture's memory capacities and the value sizes the arithmetic uses.
constexpr uint64_t kJointSramCapacity = 512;
constexpr uint64_t kJointL2Capacity = 128;
constexpr uint64_t kJointDramCapacity = 4096;
constexpr uint64_t kJointTileBytes = 256;  // 8 x 8 x f32
constexpr uint64_t kJointLiveVersions = 4; // four-stage pipeline residency

/// Enumerates the feasible assignments with fixture arithmetic alone -- no
/// production search, storage planner, analysis or key builder.
std::vector<Selection> enumerateFeasibleAssignments() {
  const uint64_t needed = kJointLiveVersions * kJointTileBytes;
  struct Memory {
    const char *id;
    uint64_t capacity;
  };
  struct Route {
    const char *name;
    std::optional<uint64_t> middleCapacity;
  };
  std::vector<Selection> feasible;
  for (const char *compute : {"vpu.a", "vpu.b"}) {
    for (Memory memory : {Memory{"sram.0", kJointSramCapacity},
                          Memory{"dram.0", kJointDramCapacity}}) {
      for (Route route :
           {Route{"direct", std::nullopt}, Route{"via-l2", kJointL2Capacity}}) {
        if (needed > memory.capacity)
          continue;
        if (route.middleCapacity && kJointTileBytes > *route.middleCapacity)
          continue;
        feasible.push_back(Selection{compute, memory.id, route.name});
      }
    }
  }
  return feasible;
}

/// The selection tuple each retained plan expresses, read straight off the
/// plan's own decisions: the add placement's recorded engine, the memory its
/// requirement bound, and whether any connection's route crosses the L2.
Selection selectionOf(const CoveringPlan &plan) {
  Selection selection;
  for (const PlanPlacement &placement : plan.placements) {
    for (const auto &binding : placement.computeBindings)
      if (binding.first() == "vector_engine")
        selection.compute = binding.second;
    for (const auto &memory : placement.memories)
      if (memory.first() == "dram" || selection.memory.empty())
        selection.memory = memory.second;
  }
  selection.route = "direct";
  for (const PlanConnection &connection : plan.connectionPlans)
    for (const MemoryNodeId &memory : connection.route)
      if (memory == "l2.0")
        selection.route = "via-l2";
  return selection;
}

std::multiset<Selection> selectionsOf(const MappingSearchResult &result) {
  std::multiset<Selection> selections;
  for (const CoveringPlan &plan : result.plans)
    selections.insert(selectionOf(plan));
  return selections;
}

llvm::Expected<MappingSearchResult>
searchJointOracle(llvm::StringRef caseName, SearchMode mode, unsigned topK) {
  auto c = issue129::resourceCase(caseName);
  if (!c)
    return c.takeError();
  MappingSearchOptions options;
  options.mode = mode;
  options.topK = topK;
  return issue129::searchCase(*c, options);
}

} // namespace

// The exact search's feasible set is the independently enumerated one, for
// every retention budget: with no cap (topK = 0) and with room for both plans
// (topK = 2) the sets are equal, and with topK = 1 the single retained plan is
// the best of that same set -- not a cheaper covering the evaluation rejected.
TEST(ExactResourceOracleTest, ExactSearchMatchesIndependentEnumeration) {
  const std::vector<Selection> oracle = enumerateFeasibleAssignments();
  // Premise: the arithmetic admits exactly the two DRAM/direct assignments, so
  // the test discriminates rather than passing for any set.
  ASSERT_EQ(oracle.size(), 2u);
  EXPECT_EQ(oracle[0], (Selection{"vpu.a", "dram.0", "direct"}));
  EXPECT_EQ(oracle[1], (Selection{"vpu.b", "dram.0", "direct"}));
  const std::multiset<Selection> expected(oracle.begin(), oracle.end());

  llvm::Expected<MappingSearchResult> full =
      searchJointOracle("joint-oracle", SearchMode::Exact, 0);
  ASSERT_TRUE(bool(full)) << llvm::toString(full.takeError());
  EXPECT_EQ(selectionsOf(*full), expected) << "topK=0";
  EXPECT_FALSE(full->searchTruncated) << "topK=0";

  llvm::Expected<MappingSearchResult> wide =
      searchJointOracle("joint-oracle", SearchMode::Exact, 2);
  ASSERT_TRUE(bool(wide)) << llvm::toString(wide.takeError());
  EXPECT_EQ(selectionsOf(*wide), expected) << "topK=2";
  EXPECT_FALSE(wide->searchTruncated) << "topK=2";

  // With topK = 1 the single retained plan is the *best* of that same set --
  // the ranking is by scheduled cost then id, so it is the untruncated set's
  // first plan, not the lexicographically-first selection.
  llvm::Expected<MappingSearchResult> one =
      searchJointOracle("joint-oracle", SearchMode::Exact, 1);
  ASSERT_TRUE(bool(one)) << llvm::toString(one.takeError());
  ASSERT_EQ(one->plans.size(), 1u);
  ASSERT_FALSE(full->plans.empty());
  EXPECT_EQ(selectionOf(one->plans.front()), selectionOf(full->plans.front()));
}

// A caller may declare the two same-kind engines in either order, but the
// declaration order is not a decision: the selected set and the best plan are
// identical. This is the R1 property the oracle exists to guard -- an engine
// chosen by executor order instead of by the recorded selection would flip
// here.
TEST(ExactResourceOracleTest, ReversedEngineDeclarationSelectsTheSameSet) {
  llvm::Expected<MappingSearchResult> forward =
      searchJointOracle("joint-oracle", SearchMode::Exact, 0);
  ASSERT_TRUE(bool(forward)) << llvm::toString(forward.takeError());
  llvm::Expected<MappingSearchResult> reversed =
      searchJointOracle("joint-oracle-reversed", SearchMode::Exact, 0);
  ASSERT_TRUE(bool(reversed)) << llvm::toString(reversed.takeError());

  EXPECT_EQ(selectionsOf(*forward), selectionsOf(*reversed));
  ASSERT_FALSE(forward->plans.empty());
  ASSERT_FALSE(reversed->plans.empty());
  // The best retained plan is the same decision whichever order it was declared
  // in, so its decision-only id is the same too.
  EXPECT_EQ(forward->plans.front().id, reversed->plans.front().id);
}

// A bounded beam is a heuristic, not an exhaustive enumerator: it may report
// fewer plans, but every plan it does report must be one the independent
// enumeration admits, and any shortfall must be reported as truncation rather
// than presented as the whole feasible set.
TEST(ExactResourceOracleTest, BoundedBeamOnlyReportsEnumeratedFeasiblePlans) {
  const std::vector<Selection> oracle = enumerateFeasibleAssignments();
  const std::set<Selection> admitted(oracle.begin(), oracle.end());

  llvm::Expected<MappingSearchResult> beam =
      searchJointOracle("joint-oracle", SearchMode::Beam, 0);
  ASSERT_TRUE(bool(beam)) << llvm::toString(beam.takeError());
  for (const CoveringPlan &plan : beam->plans)
    EXPECT_TRUE(admitted.count(selectionOf(plan)) == 1)
        << "the beam retained a plan the enumeration does not admit";
  if (beam->plans.size() < oracle.size())
    EXPECT_TRUE(beam->searchTruncated) << "a shortfall must be reported as "
                                          "truncation, not as the feasible set";
}

// Task R8 step 2: sequential temporaries whose *summed* sizes exceed the memory
// but whose true live peak fits must be retained. The search's own
// partial-state accounting charges every materialized output until its last
// consumer -- an over-estimate that reuse can only lower -- so a plan rejected
// on that sum would be an unsafe capacity rejection. The `sequential` fixture
// runs four iterations of one 256-byte temporary in a 256-byte L2: the peak is
// one buffer, the naive sum is four.
TEST(ExactResourceOracleTest, SequentialTemporariesAreJudgedByTheirLivePeak) {
  auto c = issue129::resourceCase("sequential");
  ASSERT_TRUE(bool(c)) << llvm::toString(c.takeError());
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 0;
  auto result = issue129::searchCase(*c, options);
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(result->plans.empty())
      << "the live peak fits; rejecting on the summed size would be unsound";

  const CoveringPlan &plan = result->plans.front();
  // Exactly one 256-byte L2 buffer, resident once: the four sequential
  // iterations reuse it. A model that charged every iteration a fresh buffer
  // (4 x 256 = 1024, over the 256-byte L2) would reject a plan whose live peak
  // is 256 -- the unsound partial-state rejection this case guards.
  uint64_t l2Allocations = 0;
  for (const StorageAllocation &allocation : plan.allocations) {
    if (allocation.memory != "l2.0")
      continue;
    ++l2Allocations;
    EXPECT_EQ(allocation.bytes, 256u);
    EXPECT_EQ(allocation.simultaneousOccurrences, 1u)
        << "a serial loop reuses one buffer; four live versions would not fit";
  }
  EXPECT_EQ(l2Allocations, 1u);
}
