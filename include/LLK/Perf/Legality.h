//===- Legality.h - Machine-aware candidate legality ----------------------===//
//
// Part of the M12 tuning core (issue #49).
//
// A candidate is legal when every `micro.constraint` its search space declares
// holds for the workload shape and the machine model. Each ConstraintKind maps
// to one rule here; the rule reads the parameter values it needs -- integer
// roles (BM/BN/BK, vector_width, pipeline_stages, prefetch_distance,
// num_threads) by name, symbolic roles (layout, memory_path, owner_mapping,
// fragment_shape, tail_policy) by the dialect `kind` -- and compares them
// against machine::MachineModel resources.
//
// The rules are deliberately analytical: they are the cheap, machine-only half
// of the picture, evaluated before any IR is built. The simulator (#46) still
// has the final word on a bound kernel, and reports layout and owner problems
// it finds as warnings; here they either reject a candidate or they do not.
//
// Reasons are stable strings, formatted `<constraint_kind>: <detail>`, so a
// rejection can be shown to a user and asserted on in a test. The first
// failing constraint in declaration order wins, which makes the verdict
// deterministic for a given space.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_LEGALITY_H
#define LLK_PERF_LEGALITY_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Perf/Candidate.h"
#include "LLK/Perf/SearchSpace.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::perf {

struct LegalityResult {
  bool legal = false;
  /// Stable reason string, prefixed with the constraint kind. Empty when legal.
  std::string reason;
};

/// Fraction of a memory level's capacity a schedule may occupy. Matches the
/// MVP limit in the auto-tuning spec.
inline constexpr double kSramUtilizationLimit = 0.80;

/// The workload facts a constraint is evaluated against. `originalWorkload` is
/// the pre-tiling M/N/K the export records as provenance; `contractions` holds
/// one shape per `micro.mma` the kernel performs. Either may be absent: a
/// shape-independent constraint (mapping extent, owner availability) evaluates
/// with no facts at all, while a shape-dependent one reports the specific fact
/// it is missing rather than blocking every constrained kernel.
struct BindingFacts {
  std::optional<WorkloadShape> originalWorkload;
  std::vector<WorkloadShape> contractions;
};

/// Evaluates one constraint against the facts the constraint actually needs. A
/// constraint whose referenced parameters are not all bound by `candidate` is
/// rejected rather than skipped, so a malformed candidate can never pass by
/// omission; a shape-dependent constraint whose required fact is absent is
/// rejected with a diagnostic naming that fact.
LegalityResult checkConstraint(const SearchConstraint &constraint,
                               const SearchSpace &space,
                               const Candidate &candidate,
                               const BindingFacts &facts,
                               const machine::MachineModel &machine);

/// Evaluates every constraint in declaration order and returns the first
/// rejection. A space with no constraints is legal.
LegalityResult checkLegality(const SearchSpace &space,
                             const Candidate &candidate,
                             const BindingFacts &facts,
                             const machine::MachineModel &machine);

/// Shape-based convenience overload: the single `shape` is used both as the
/// original workload (tail divisibility) and as the sole contraction (dtype and
/// fragment requirements).
LegalityResult checkConstraint(const SearchConstraint &constraint,
                               const SearchSpace &space,
                               const Candidate &candidate,
                               const WorkloadShape &shape,
                               const machine::MachineModel &machine);

/// Shape-based convenience overload of checkLegality; see above.
LegalityResult checkLegality(const SearchSpace &space,
                             const Candidate &candidate,
                             const WorkloadShape &shape,
                             const machine::MachineModel &machine);

} // namespace mlir::llk::perf

#endif // LLK_PERF_LEGALITY_H
