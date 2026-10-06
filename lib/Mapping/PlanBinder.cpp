//===- PlanBinder.cpp - Materialize a selected plan (D7/#50) -------------===//
//
// `bindPlan` clones the source kernel and persists the selected plan as
// schema-v2 metadata (`encodeSelectedPlan`, see MappingMetadata.h). The
// concrete operations the plan implies are constructed by a `PlanMaterializer`
// -- the canonical one lives in
// `lib/Conversion/MicroMapping/PlanMaterialization.cpp`, so this translation
// unit (and the whole mapping library) never names a Micro operation.
// Verification of already-mapped Micro-IR lives in PlanVerification.cpp.

#include "LLK/Mapping/PlanBinder.h"

#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::mapping {

namespace {

constexpr llvm::StringLiteral kPlanAttr = "micro.plan";

/// The stable reason a connection is left unimplemented when no materializer
/// was supplied: a standalone mapping caller gets an honest partial plan that
/// names the missing construction path, rather than IR silently missing it.
constexpr llvm::StringLiteral kNoMaterializerReason = "no_plan_materializer";

llvm::Error bindError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// The module's single `micro.kernel`. Fails when there is none (nothing to
/// bind) or more than one (no selector exists, so binding "the first" would
/// silently ignore the rest).
llvm::Expected<mlir::Operation *> findKernel(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *, 2> kernels;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "micro.kernel")
      kernels.push_back(op);
  });
  if (kernels.empty())
    return bindError("bindPlan: the source module has no micro.kernel");
  if (kernels.size() > 1)
    return bindError("bindPlan: the source module has " +
                     std::to_string(kernels.size()) +
                     " micro.kernels; one kernel per module is required");
  return kernels.front();
}

} // namespace

llvm::Expected<BoundPlan> bindPlan(mlir::ModuleOp source,
                                   const CoveringPlan &plan,
                                   const MappingTarget &target,
                                   BindContract contract,
                                   PlanMaterializer *materializer) {
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::cast<mlir::ModuleOp>(source->clone()));
  llvm::Expected<mlir::Operation *> resolvedKernel = findKernel(*module);
  if (!resolvedKernel)
    return resolvedKernel.takeError();
  mlir::Operation *kernel = *resolvedKernel;

  mlir::MLIRContext *context = module->getContext();

  // --- persist the selected state (schema v2) ---------------------------
  // Encoded before any movement is emitted, so the recorded source-graph hash
  // is the pre-materialization identity a later reader recovers through the
  // recorded connection provenance.
  if (llvm::Error error = encodeSelectedPlan(*module, plan, target))
    return std::move(error);

  BoundPlan bound;

  // --- materialize the movement (design §18.2) -------------------------
  //
  // The mapping core cannot construct dialect operations; it delegates to the
  // supplied materializer. Without one the plan is bound metadata-only, and
  // every connection that would need a new operation is reported so the result
  // is an honest partial plan instead of IR that silently omits the movement.
  if (materializer) {
    if (llvm::Error error =
            materializer->materialize(*module, plan, target, bound))
      return std::move(error);
  } else {
    for (const PlanConnection &connection : plan.connectionPlans) {
      // A `Direct` connection is materialized by construction: the producer
      // wrote the value to the memory the consumer reads, in a layout the
      // consumer addresses, so there is nothing to emit and nothing to report.
      if (connection.kind == ConnectionKind::Direct)
        continue;
      bound.unmaterialized.push_back("value " +
                                     std::to_string(connection.value) + ": " +
                                     kNoMaterializerReason.str());
    }
  }

  // An executable contract refuses a plan it could not fully materialize: the
  // caller asked for code a backend may run, and a partial binding would omit
  // part of the selected plan without saying so in a form the backend checks.
  // The partial contract keeps the report, which is the analysis contract.
  if (contract == BindContract::Executable && !bound.unmaterialized.empty()) {
    std::string message =
        "bindPlan: the plan is not fully executable; " +
        std::to_string(bound.unmaterialized.size()) +
        " execution-affecting decision(s) could not be materialized:";
    for (const std::string &reason : bound.unmaterialized)
      message += "\n  " + reason;
    return bindError(message);
  }

  // Record materialization completeness in the persisted selection, so a reader
  // can tell a fully materialized plan from a partial one.
  if (!bound.unmaterialized.empty()) {
    if (auto planAttr =
            kernel->getAttrOfType<mlir::DictionaryAttr>(kPlanAttr)) {
      mlir::NamedAttrList updated(planAttr);
      updated.set("materialized", mlir::BoolAttr::get(context, false));
      kernel->setAttr(kPlanAttr, updated.getDictionary(context));
    }
  }

  bound.planId = plan.id;
  bound.kernel = kernel;
  bound.module = std::move(module);
  return std::move(bound);
}

} // namespace mlir::llk::mapping
