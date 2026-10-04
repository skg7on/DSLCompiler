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

#include "LLK/Mapping/Diagnostics.h"
#include "LLK/Mapping/LlkMap.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/Placement.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
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
  /// Reported with `DiagnosticCode::NoMatchingRule`.
  uint64_t nodesWithoutRules = 0;
  /// Candidates whose rule matched but placed nothing. The stable code on the
  /// accompanying diagnostic says which requirement failed
  /// (`no_legal_executor`, `no_legal_layout`, or
  /// `unsupported_compute_fragment`).
  uint64_t candidatesWithoutPlacement = 0;
  /// Chosen instance pairs no connection could join. Reported with
  /// `DiagnosticCode::NoMemoryRoute`.
  uint64_t incompatibleInstancePairs = 0;
  /// Plans a memory's own capacity or the whole-plan budget rejected.
  /// Reported with `DiagnosticCode::MemoryCapacityExceeded`.
  uint64_t plansRejectedByCapacity = 0;
  /// Stable-coded reasons (design §22.3), deterministically ordered by code
  /// then message before `search()` returns. Recurring causes -- a capacity
  /// overflow hit on every branch, a provider never caching a rule -- are
  /// recorded once each, so the frontier stays bounded; the counts above are
  /// the tallies.
  std::vector<Diagnostic> diagnostics;

  /// Every reporting event counted per stable code (design §22.2): unlike
  /// `diagnostics`, which lists each distinct (code, message) once, this
  /// tallies every occurrence, so a cause hit on many branches is visible as a
  /// count. Ordered by code (a `std::map`), so it is deterministic.
  std::map<DiagnosticCode, uint64_t> codeCounts;

  /// True when any diagnostic carries `code`.
  bool has(DiagnosticCode code) const;
};

struct MappingSearchResult {
  /// Complete plans, best first, at most `topK`.
  std::vector<CoveringPlan> plans;
  /// True when any cap ended the search early.
  bool searchTruncated = false;
  /// True when the search picked each connection's locally cheapest alternative
  /// instead of branching over the alternatives the topology offered (set in
  /// exact mode; the beam and deterministic modes are heuristic by contract).
  /// The result is then not an exhaustive joint placement/route search: a
  /// covering rejected here may still be feasible through a more expensive
  /// route combination. Distinct from `searchTruncated` -- this holds with
  /// every cap lifted -- and reported with
  /// `DiagnosticCode::ConnectionChoiceUnexplored`.
  bool connectionChoicesUnexplored = false;
  FailureFrontier frontier;
  /// Partial plans the search expanded, for diagnostics.
  uint64_t expandedStates = 0;

  // Search-wide tallies (design §22.2). These count the whole search -- not
  // just the retained top-K -- so a report can state how much of the space was
  // covered. Deterministic for identical inputs.
  /// Rule-to-candidate matches produced across all nodes.
  uint64_t candidateCount = 0;
  /// Candidate instances enumerated across all candidates.
  uint64_t instanceCount = 0;
  /// Connection alternatives synthesized across all branches.
  uint64_t routeCount = 0;
  /// Complete plans found before the top-K cap truncated `plans`.
  uint64_t planCount = 0;
};

/// Searches one workload graph against one target.
class CoveringSearch {
public:
  CoveringSearch(const WorkloadGraph &workload, const MappingTarget &target,
                 mlir::MLIRContext &context, const LayoutContext &layoutContext,
                 const MappingSearchOptions &options = {},
                 std::optional<SearchBinding> binding = std::nullopt,
                 llvm::StringMap<std::string> boundLayouts = {});

  llvm::Expected<MappingSearchResult> search();

private:
  const WorkloadGraph &workload_;
  const MappingTarget &target_;
  mlir::MLIRContext &context_;
  LayoutContext layoutContext_;
  MappingSearchOptions options_;
  /// The search-space point this search evaluates, when known. Its values
  /// constrain rule parameter resolution: a rule parameter the binding names
  /// takes only the bound value, so a binding can select among rules that
  /// differ only in a parameter choice (a pinned value that no `require`
  /// accepts makes the rule a non-match). Every emitted plan also records the
  /// binding's hash and parameters (design §8.3/§9.5); a search with no binding
  /// leaves all of that at its default, so its plan ids are unchanged.
  ///
  /// Placement and routing stay binding-independent *by design* (ruling S3): a
  /// binding names search choices, not machine resources -- the machine model
  /// owns placement -- so it never constrains where a node runs or how a
  /// connection routes. This is a finished boundary, not a half-built bridge.
  std::optional<SearchBinding> binding_;
  /// The layout the binding selects, resolved from its `layout`-kind parameter
  /// by the caller -- the pass layer, which alone can see the search space, so
  /// lib/Mapping keeps no LLKPerf dependency. A rule that declares layout
  /// requirements must offer it (`require layout ... satisfies <it>`) or it is
  /// a non-match for the node; when it does, only that layout is materialized,
  /// so the binding -- not the rule file -- decides which of the rule's
  /// declared layouts applies. A rule that declares *no* layout requirement
  /// matches unchanged: it takes on no layout obligation, so it neither offers
  /// nor contradicts the bound value. Absent leaves layout selection exactly as
  /// it was before bindings.
  llvm::StringMap<std::string> boundLayouts_;
};

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_COVERINGSEARCH_H
