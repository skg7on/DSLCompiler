//===- CompletePlanEvaluation.h - Evaluate a complete proposal (R7) -------===//
//
// Issue #129, task R7. `CoveringSearch` synthesizes *proposals* -- cheap
// coverings of instances and connections -- and must know nothing about dialect
// conversion, plan materialization or the performance model. Whether a proposal
// is actually feasible (its memories resolve, its hop storage fits, its
// occupancy is within capacity, its kernel schedules) is only known after the
// plan is finalized and its bound kernel analyzed, and that work lives above
// both the mapping core and the performance model.
//
// This is that missing stage. `evaluateCompletePlan` turns a proposal into
// exactly one of a finalized plan or a stable-coded rejection, and is injected
// into the search as a `CompletePlanEvaluator` callback so every top-K and
// first-legal decision can be taken *after* feasibility is known.
//
// Library boundary: this translation unit lives in `LLKMicroMapping`, the layer
// above `LLKMapping` and `LLKPerf`, so the mapping core never links the
// performance model, a dialect or a JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_MICROMAPPING_COMPLETEPLANEVALUATION_H
#define LLK_CONVERSION_MICROMAPPING_COMPLETEPLANEVALUATION_H

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

namespace mlir::llk::mapping {

/// Evaluates one complete proposal against `target`, in the context of the
/// `source` module it would be bound to and the `graph` it was planned from.
///
/// The chain, in order (issue #129, task R7):
///   1. clone the proposal, record its provenance (graph/target/machine/layout/
///      rule hashes) and its materialization contract;
///   2. finalize physical storage: resolve every endpoint's map/compute/port,
///      decide the movement hops and their storage slots and steps, validate
///      occupancy against each memory's capacity, and assign the decision-only
///      v3 plan id;
///   3. bind a preview onto a *private clone* of `source` (the source module is
///      never mutated) and verify the selected decisions materialize;
///   4. run the shared selected-kernel static analysis over the bound kernel,
///      with the plan's own R5 storage liveness;
///   5. validate occupancy against the analysis's peak;
///   6. attach the derived event snapshot and the final scheduled cost, and
///      carry the plan's physical-readiness verdict;
///   7. require the durable choices to be stable: re-finalizing the plan must
///      reproduce its id. One whose id depends on the derived ordering is a
///      cycle and is rejected.
///
/// Exactly one of the returned `plan`/`rejection` is set. An `llvm::Error`
/// means an invalid invocation or an infrastructure failure and stops the
/// search.
///
/// `contract` is the binding contract the caller will bind the selected plan
/// under. Under `BindContract::Executable` the preview is an executable
/// binding: a decision the binder cannot materialize, or a value whose physical
/// memory does not resolve, is refused by `bindPlan`. That refusal is a
/// **candidate rejection** -- a property of this proposal -- so the search
/// records it with a stable code and keeps enumerating: some other covering may
/// be executable, and a cheap non-executable one must not end the search. The
/// rejection carries the binder's own message, so a caller that ends up with no
/// plan reads exactly the refusal a direct `bindPlan` under the executable
/// contract would give. Under `BindContract::Partial` an incomplete plan is
/// kept as an explicit analysis artifact -- its `physicalComplete` verdict and
/// ordered reasons are recorded -- which is what the `report-only` workflow
/// needs. Either way the strict finalize attempt runs first, so a physically
/// complete plan is always preferred.
///
/// An `llvm::Error` is reserved for a malformed invocation and for
/// infrastructure failures (a structural binder failure under the partial
/// contract, an unbuildable stream): those stop the search, because they are
/// properties of the run rather than of one candidate.
llvm::Expected<CompletePlanEvaluation>
evaluateCompletePlan(mlir::ModuleOp source, const WorkloadGraph &graph,
                     const MappingTarget &target, const CoveringPlan &proposal,
                     BindContract contract);

} // namespace mlir::llk::mapping

#endif // LLK_CONVERSION_MICROMAPPING_COMPLETEPLANEVALUATION_H
