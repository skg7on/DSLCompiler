//===- arm_target_fixture.h - Map a fixture through the public pipeline ---===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test-support only (framework E5/E6, issues #180/#181). The acceptance for the
// ARM static targets is not "a rule matched" or "an emitter key exists": it is
// the plan an actual caller gets from the typed public pipeline -- registry
// target load, workload-graph extraction, covering search with the canonical
// complete-plan evaluator injected before retention, materialization under the
// static/partial contract, mapped-IR verification, and the shared selected
// kernel analysis carrying the plan's own storage liveness.
//
// E6 adds the two views the negative controls need: `searchFixture` stops at
// the retained set so a test can prove *why* nothing survived (and that the
// rejection is not a search cap), and `evaluatePlan` runs the same canonical
// evaluator on a caller-tampered plan copy so an altered selected identity is
// judged by the complete-plan contract rather than by a test assertion.
//
// This is the one place that sequence is assembled for the ARM tests, so a
// break anywhere along it fails in every acceptance test rather than in a
// layer test that no longer reflects the whole. It never computes an expected
// value: the tests assert the concrete decisions it returns.
//
// It is linked only into the Mapping test category, never into a production
// library, and it names no target in shared mapping/perf code.

#ifndef LLK_TEST_ARM_TARGET_FIXTURE_H
#define LLK_TEST_ARM_TARGET_FIXTURE_H

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Perf/SelectedKernelAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace arm_test {

/// Everything the public pipeline produced for one fixture and target.
///
/// `context` owns every MLIR object the other members reference and is declared
/// first, so it is destroyed last (members are destroyed in reverse order);
/// `source` and `bound.module` are owned by it, and the target, graph, plan and
/// analysis all reference objects the context owns.
struct MappedFixture {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> source;
  std::unique_ptr<mlir::llk::mapping::MappingTarget> target;
  mlir::llk::mapping::WorkloadGraph graph;
  mlir::llk::mapping::CoveringPlan plan;
  mlir::llk::mapping::BoundPlan bound;
  mlir::llk::perf::SelectedKernelAnalysis analysis;
};

/// Everything the pipeline produced up to and including the search.
///
/// It exists so a negative control can inspect the *frontier* instead of
/// reading an error string: `result.plans` is exactly the retained set (empty
/// when every proposal was rejected), `result.frontier` carries the rejection
/// codes and messages, and `result.searchTruncated` distinguishes a cap from a
/// rejection. Members are ordered so the context outlives the objects that
/// reference it.
struct SearchedFixture {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> source;
  std::unique_ptr<mlir::llk::mapping::MappingTarget> target;
  mlir::llk::mapping::WorkloadGraph graph;
  mlir::llk::mapping::MappingSearchResult result;
};

/// Maps `inputPath` (source-root-relative unless absolute) against the target
/// registered as `targetName`, optionally with a machine-profile override.
///
/// Returns an error instead of dereferencing a failed stage: target load,
/// parse/verify, kernel resolution, graph extraction, search (including "no
/// complete plan", with the frontier's diagnostics), binding, mapped-IR
/// verification and strict analysis all propagate.
///
/// `topK` bounds the retained plans; the default preserves E5's search. A
/// negative control passes 1 so a plan that survived retention could not have
/// been merely cut for budget.
llvm::Expected<MappedFixture> mapFixture(llvm::StringRef targetName,
                                         llvm::StringRef inputPath,
                                         llvm::StringRef machineOverride = {},
                                         unsigned topK = 8);

/// The pipeline through the search only. The object always represents a
/// completed search, so an empty `result.plans` is a testing observation, not a
/// helper error; target load, parse and extraction failures still propagate.
llvm::Expected<SearchedFixture>
searchFixture(llvm::StringRef targetName, llvm::StringRef inputPath,
              llvm::StringRef machineOverride = {}, unsigned topK = 1);

/// Runs the canonical complete-plan evaluator on `plan` against this fixture's
/// source, graph and target -- the same contract the search injects before
/// retention. The evaluator clones the module, so `fixture` is not mutated and
/// a test may call this once per tampered copy.
llvm::Expected<mlir::llk::mapping::CompletePlanEvaluation>
evaluatePlan(const MappedFixture &fixture,
             const mlir::llk::mapping::CoveringPlan &plan);

/// The selected placement of the fixture's contraction, or null unless the
/// graph has exactly one `micro.mma` node and the plan places it exactly once.
/// It locates the placement by graph node id, so two placements of a
/// multi-op rule cannot be mistaken for the contraction's.
const mlir::llk::mapping::PlanPlacement *
contractionPlacement(const MappedFixture &fixture);

/// Writes `yaml` to a uniquely named file in this test category's build
/// directory and returns its path. The caller owns the file (`llvm::FileRemover`
/// or an equivalent scope guard). Used to point a registered target at a
/// structurally valid but infeasible machine-profile variant.
llvm::Expected<std::string> writeMachineVariant(llvm::StringRef yaml);

} // namespace arm_test

#endif // LLK_TEST_ARM_TARGET_FIXTURE_H
