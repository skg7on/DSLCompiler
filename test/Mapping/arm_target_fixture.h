//===- arm_target_fixture.h - Map a fixture through the public pipeline ---===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test-support only (framework E5, issue #180). The acceptance for the ARM
// static targets is not "a rule matched" or "an emitter key exists": it is the
// plan an actual caller gets from the typed public pipeline -- registry target
// load, workload-graph extraction, covering search with the canonical
// complete-plan evaluator injected before retention, materialization under the
// static/partial contract, mapped-IR verification, and the shared selected
// kernel analysis carrying the plan's own storage liveness.
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

/// Maps `inputPath` (source-root-relative unless absolute) against the target
/// registered as `targetName`, optionally with a machine-profile override.
///
/// Returns an error instead of dereferencing a failed stage: target load,
/// parse/verify, kernel resolution, graph extraction, search (including "no
/// complete plan", with the frontier's diagnostics), binding, mapped-IR
/// verification and strict analysis all propagate.
llvm::Expected<MappedFixture> mapFixture(llvm::StringRef targetName,
                                         llvm::StringRef inputPath,
                                         llvm::StringRef machineOverride = {});

/// The selected placement of the fixture's contraction, or null unless the
/// graph has exactly one `micro.mma` node and the plan places it exactly once.
/// It locates the placement by graph node id, so two placements of a
/// multi-op rule cannot be mistaken for the contraction's.
const mlir::llk::mapping::PlanPlacement *
contractionPlacement(const MappedFixture &fixture);

} // namespace arm_test

#endif // LLK_TEST_ARM_TARGET_FIXTURE_H
