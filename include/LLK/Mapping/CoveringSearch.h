//===- CoveringSearch.h - Complete-plan search (D6) -----------------------===//
//
// Part of the target-independent mapping core (epic #67, workstream D6).
//
// Covering search picks one placement per workload node and one connection per
// dataflow edge, producing complete, ranked `CoveringPlan`s. It is the last
// step before a plan is bound to IR.
//
// Three modes share one interface (design §16.2): `deterministic` returns the
// first legal plan in canonical order (tests and bring-up); `beam` is the
// production default, a bounded best-first search; `exact` is branch and bound
// for small graphs and validation. Every cap is an option and is *reported* --
// reaching one sets `searchTruncated`, and no mode ever implies optimality
// after truncation.
//
// Cost is the candidate's declared lower bound plus the synthesized connection
// cost. The performance evaluator (#46) replaces that later; it does not
// change which plans are legal (design §16.2).
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_COVERINGSEARCH_H
#define LLK_MAPPING_COVERINGSEARCH_H

#include "LLK/Mapping/LlkMap.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/Placement.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mlir::llk::mapping {

enum class SearchMode {
  /// First legal plan in canonical order.
  Deterministic,
  /// Bounded best-first search; the production default.
  Beam,
  /// Branch-and-bound exact cover for small graphs.
  Exact
};

struct MappingSearchOptions {
  SearchMode mode = SearchMode::Beam;
  unsigned beamWidth = 64;
  unsigned topK = 8;
  unsigned maxCandidatesPerNode = 64;
  unsigned maxInstancesPerCandidate = 64;
  unsigned maxRoutesPerConnection = 8;
  uint64_t memoryBudgetBytes = 512ULL << 20;
  bool enableLatencyCache = true;
  bool enableSymmetryReduction = true;
  /// How complete plans are ranked (design §17.1). The default -- latency
  /// minimized -- preserves the pre-objective behaviour, so a caller that does
  /// not consult `micro.objective` is unaffected.
  ObjectiveOrder objective = {};
};

/// Why no complete plan was found (design §16.5), counted per category. The
/// counts are deterministic: the same graph always produces the same numbers.
struct FailureFrontier {
  /// Nodes with no rule in effect: either no rule names the operation, or every
  /// rule that names it was rejected by its `require` constraints.
  uint64_t nodesWithoutRules = 0;
  uint64_t candidatesWithoutPlacement = 0;
  uint64_t incompatibleInstancePairs = 0;
  uint64_t plansRejectedByCapacity = 0;
  std::vector<std::string> messages;
};

struct MappingSearchResult {
  /// Complete plans, best first, at most `topK`.
  std::vector<CoveringPlan> plans;
  /// True when any cap ended the search early.
  bool searchTruncated = false;
  FailureFrontier frontier;
  /// Partial plans the search expanded, for diagnostics.
  uint64_t expandedStates = 0;
};

/// Searches one workload graph against one target.
class CoveringSearch {
public:
  CoveringSearch(const WorkloadGraph &workload, const MappingTarget &target,
                 mlir::MLIRContext &context, const LayoutContext &layoutContext,
                 const MappingSearchOptions &options = {},
                 std::optional<SearchBinding> binding = std::nullopt);

  llvm::Expected<MappingSearchResult> search();

private:
  const WorkloadGraph &workload_;
  const MappingTarget &target_;
  mlir::MLIRContext &context_;
  LayoutContext layoutContext_;
  MappingSearchOptions options_;
  /// The search-space point this search evaluates, when known. Every emitted
  /// plan records its hash and parameters (design §8.3/§9.5); a search with no
  /// binding leaves both default so its plan ids are unchanged.
  std::optional<SearchBinding> binding_;
};

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_COVERINGSEARCH_H
