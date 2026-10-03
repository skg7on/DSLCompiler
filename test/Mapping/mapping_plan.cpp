//===- mapping_plan.cpp - Plan data model and canonical ids (D1) ---------===//

#include "LLK/Mapping/MappingPlan.h"

#include <gtest/gtest.h>

using namespace mlir::llk::mapping;

TEST(MappingPlan, CandidateIdIsOrderIndependent) {
  MappingCandidate a;
  a.rule = "r";
  a.coveredNodes = {2, 0, 1};
  MappingCandidate b;
  b.rule = "r";
  b.coveredNodes = {0, 1, 2};
  EXPECT_EQ(computeCandidateId(a), computeCandidateId(b));
}

TEST(MappingPlan, CandidateIdChangesWithContent) {
  MappingCandidate a;
  a.rule = "r";
  a.coveredNodes = {0};
  MappingCandidate b;
  b.rule = "r2";
  b.coveredNodes = {0};
  EXPECT_NE(computeCandidateId(a), computeCandidateId(b));
}

TEST(MappingPlan, InstanceIdIgnoresBindingInsertionOrder) {
  CandidateInstance a;
  a.candidate = 7;
  a.executorBindings["w0"] = "core.0";
  a.executorBindings["w1"] = "core.1";
  CandidateInstance b;
  b.candidate = 7;
  b.executorBindings["w1"] = "core.1";
  b.executorBindings["w0"] = "core.0";
  EXPECT_EQ(computeInstanceId(a), computeInstanceId(b));
}

TEST(MappingPlan, PlanIdChangesWithBindingHash) {
  CoveringPlan a;
  a.sourceBindingHash = 1;
  CoveringPlan b;
  b.sourceBindingHash = 2;
  EXPECT_NE(computePlanId(a), computePlanId(b));
}

TEST(MappingPlan, PlanIdIgnoresInstanceOrder) {
  CoveringPlan a;
  a.sourceBindingHash = 5;
  a.instances = {2, 1};
  CoveringPlan b;
  b.sourceBindingHash = 5;
  b.instances = {1, 2};
  EXPECT_EQ(computePlanId(a), computePlanId(b));
}

TEST(MappingPlan, SortUniqueRemovesDuplicatesAndSorts) {
  llvm::SmallVector<WorkloadNodeId> nodes{3, 1, 3, 2};
  sortUnique(nodes);
  ASSERT_EQ(nodes.size(), 3u);
  EXPECT_EQ(nodes[0], 1u);
  EXPECT_EQ(nodes[1], 2u);
  EXPECT_EQ(nodes[2], 3u);
}

TEST(MappingPlan, ConnectionKindRoundTrips) {
  for (ConnectionKind kind :
       {ConnectionKind::Direct, ConnectionKind::Transfer,
        ConnectionKind::LayoutTransform, ConnectionKind::TransferAndTransform,
        ConnectionKind::Replicate, ConnectionKind::Reduce}) {
    EXPECT_EQ(symbolizeConnectionKind(stringifyConnectionKind(kind)), kind);
  }
}
