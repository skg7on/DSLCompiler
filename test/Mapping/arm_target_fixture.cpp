//===- arm_target_fixture.cpp - Map a fixture through the public pipeline -===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Framework E5 (issue #180). This mirrors the integration `MicroMapPass` runs,
// not the bare `CoveringSearch` a layer test can get away with:
//
//   registry target -> extract graph -> search with
//   `evaluateCompletePlan(..., BindContract::Partial, ...,
//   AnalysisRequirement::Complete)` injected before retention -> bind with the
//   canonical materializer -> `verifyMappedMicroIR` -> strict
//   `analyzeSelectedKernel` carrying the finalized plan's storage liveness.
//
// The completion evaluator is what makes a retained plan physically final:
// storage is decided, the preview is bound and analyzed, occupancy is checked,
// and only then is the plan ranked and returned. Binding the search proposal
// directly would accept a plan whose real live footprint was never measured --
// exactly the defect the ARM targets must not inherit.

#include "arm_target_fixture.h"

#include "CompletePlanEvaluation.h"
#include "MicroMappingCommon.h"

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Target/MappingTargets.h"

#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

#include <string>

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace arm_test {

using namespace mlir;
using namespace mlir::llk::mapping;
namespace perf = mlir::llk::perf;

namespace {

/// Resolve a fixture path against the source root unless it is absolute.
std::string resolveInputPath(llvm::StringRef inputPath) {
  if (llvm::sys::path::is_absolute(inputPath))
    return inputPath.str();
  llvm::SmallString<256> path(LLK_SOURCE_DIR);
  llvm::sys::path::append(path, inputPath);
  return path.str().str();
}

/// The module's single `micro.kernel`. Two kernels would make "the contraction
/// fixture" ambiguous, so this refuses rather than picking one.
llvm::Expected<Operation *> singleKernel(ModuleOp module) {
  Operation *kernel = nullptr;
  bool ambiguous = false;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.kernel")
      return;
    if (kernel)
      ambiguous = true;
    kernel = op;
  });
  if (ambiguous || !kernel)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "arm_test: the fixture must contain exactly one micro.kernel");
  return kernel;
}

} // namespace

llvm::Expected<MappedFixture> mapFixture(llvm::StringRef targetName,
                                         llvm::StringRef inputPath,
                                         llvm::StringRef machineOverride) {
  mlir::llk::target::registerAllMappingTargets();
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      createRegisteredMappingTarget(targetName, LLK_SOURCE_DIR,
                                    machineOverride);
  if (!target)
    return target.takeError();

  MappedFixture fixture;
  fixture.target = std::move(*target);
  fixture.context = std::make_unique<MLIRContext>();
  fixture.context->getOrLoadDialect<micro::MicroDialect>();

  const std::string path = resolveInputPath(inputPath);
  fixture.source = parseSourceFile<ModuleOp>(path, fixture.context.get());
  if (!fixture.source)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        ("arm_test: failed to parse or verify fixture '" + path + "'").c_str());

  llvm::Expected<Operation *> kernel = singleKernel(*fixture.source);
  if (!kernel)
    return kernel.takeError();

  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(*kernel);
  if (!graph)
    return graph.takeError();
  fixture.graph = std::move(*graph);

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  options.requireCompleteEvaluation = true;
  const uint64_t memoryBudgetBytes = options.memoryBudgetBytes;
  options.evaluateCompletePlan =
      [&, memoryBudgetBytes](const CoveringPlan &proposal)
      -> llvm::Expected<CompletePlanEvaluation> {
    return evaluateCompletePlan(*fixture.source, fixture.graph, *fixture.target,
                                proposal, BindContract::Partial,
                                memoryBudgetBytes,
                                AnalysisRequirement::Complete);
  };

  CoveringSearch search(
      fixture.graph, *fixture.target, *fixture.context,
      mlir::llk::micro_mapping_detail::deriveLayoutContext(fixture.graph),
      options);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  if (result->plans.empty()) {
    std::string message =
        "arm_test: the search produced no complete plan for '" + path + "'";
    if (result->searchTruncated)
      message += "\n  (a search cap was hit)";
    for (const mlir::llk::mapping::Diagnostic &detail :
         result->frontier.diagnostics)
      message += "\n  " + detail.message;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
  }
  fixture.plan = result->plans.front();

  // The dialect-aware half: selected movements and transforms become Micro
  // operations. The partial/static contract is the one a target without an
  // executable backend can honestly offer; it does not relax the complete
  // analysis the evaluator already required before retention.
  std::unique_ptr<PlanMaterializer> materializer =
      mlir::llk::micro_mapping_detail::createCanonicalPlanMaterializer();
  llvm::Expected<BoundPlan> bound =
      bindPlan(*fixture.source, fixture.plan, *fixture.target,
               BindContract::Partial, materializer.get());
  if (!bound)
    return bound.takeError();
  fixture.bound = std::move(*bound);

  if (llvm::Error error =
          verifyMappedMicroIR(*fixture.bound.module, *fixture.target))
    return std::move(error);

  // The strict form: the plan's own storage liveness is the occupancy verdict
  // (`storageLivenessFromPlan`), not the extraction's weaker accounting.
  llvm::Expected<perf::SelectedKernelAnalysis> analysis =
      perf::analyzeSelectedKernel(
          fixture.bound.kernel, fixture.target->machine(),
          /*requireComplete=*/true, fixture.plan, fixture.graph);
  if (!analysis)
    return analysis.takeError();
  fixture.analysis = std::move(*analysis);

  return std::move(fixture);
}

const PlanPlacement *contractionPlacement(const MappedFixture &fixture) {
  const WorkloadNode *contraction = nullptr;
  for (const WorkloadNode &node : fixture.graph.getNodes()) {
    if (node.opName != "micro.mma")
      continue;
    if (contraction)
      return nullptr; // more than one contraction: the fixture is ambiguous
    contraction = &node;
  }
  if (!contraction)
    return nullptr;

  const PlanPlacement *placement = nullptr;
  for (const PlanPlacement &candidate : fixture.plan.placements) {
    if (candidate.node != contraction->id)
      continue;
    if (placement)
      return nullptr;
    placement = &candidate;
  }
  return placement;
}

} // namespace arm_test
