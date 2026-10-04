//===- Diagnostics.h - Stable diagnostic codes (D6, §22.3) ----------------===//
//
// Part of the target-independent mapping core (epic #67). A mapping failure is
// reported by a *stable code*, not by prose: the string form is the interface a
// caller may switch on, log, or group by, while the accompanying message is
// human detail that may change between releases (design §22.3).
//
// The set is deliberately small and closed. Adding a code is a compatible
// extension; renaming its string or reusing it for a different meaning is not.
//
// Three codes are declared for the full §22.3 set but have no producer in the
// mapping search today (each is noted at its declaration):
// `no_layout_transform` and `global_constraint_failed` are decided outside the
// search, and `target_bundle_invalid` is caught when a target is loaded.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_DIAGNOSTICS_H
#define LLK_MAPPING_DIAGNOSTICS_H

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir::llk::mapping {

/// Stable diagnostic codes (design §22.3). The string each maps to is the
/// interface; see `stringifyDiagnosticCode`.
enum class DiagnosticCode {
  /// A node has no rule in effect: no rule names its operation, or every rule
  /// that names it was rejected by its `require` constraints.
  NoMatchingRule,
  /// A required layout has no solution on the target machine, so no placement
  /// can satisfy it.
  NoLegalLayout,
  /// No executor satisfies a candidate's executor requirements (or the matched
  /// executors cannot supply a required kind of memory).
  NoLegalExecutor,
  /// A plan exceeded a memory node's own capacity, or the whole-plan byte
  /// budget (design §9.3).
  MemoryCapacityExceeded,
  /// A matched executor lacks an attached compute capability a rule requires,
  /// so the rule's compute fragment is unsupported on that executor.
  UnsupportedComputeFragment,
  /// Two chosen instances could not be connected: no legal route exists
  /// between their memories (design §15.2).
  NoMemoryRoute,
  /// A connection between two layouts needs a transform that cannot be
  /// materialized. Declared for §22.3; the search has no producer today --
  /// placement turns a layout difference into a transform without testing
  /// whether one exists, so the failure surfaces later, at bind time.
  NoLayoutTransform,
  /// A search-space global constraint (`micro.constraint`) rejected a binding.
  /// Declared for §22.3; global constraints are evaluated outside this search
  /// (`lib/Perf/Legality.cpp`), so the mapping search has no producer today.
  GlobalConstraintFailed,
  /// A search cap ended the search before the space was exhausted, so no
  /// optimality may be claimed (design §16.2).
  SearchTruncated,
  /// The latency provider had no cached measurement for a piece of work
  /// (design §17.3). The static estimate stands; this is not an error.
  LatencyCacheMiss,
  /// A target bundle failed validation. Declared for §22.3; bundles are
  /// verified when a target is loaded (`verifyMappingTarget`), so the search
  /// never sees an invalid one and has no producer today.
  TargetBundleInvalid,
  /// A value's tile size could not be derived (a dynamic shape or an
  /// unmodelled type), so the search sized it with the fallback constant. The
  /// message names the value. This is not an error: the fallback keeps the
  /// plan searchable, but an assumed size must not be silent.
  AssumedValueSize,
  /// Generic mapping metadata (`micro.plan`, `micro.mapping`, `micro.routes`)
  /// is malformed: a container has the wrong attribute kind, or an entry has
  /// the wrong type. Reported by phase-2 verification with a checked cast, so
  /// invalid metadata is a diagnostic rather than an unchecked cast that aborts
  /// the process (design §25.1).
  InvalidMappingMetadata,
  /// The search chose each connection's locally cheapest alternative rather
  /// than branching over the alternatives the topology offered. The connection
  /// choice is therefore not part of an exhaustive joint placement/route
  /// search: a covering rejected here may still be feasible through a more
  /// expensive route combination. Distinct from `SearchTruncated`, which
  /// reports a *cap*; this reports an algorithmic restriction that holds even
  /// with every cap lifted. Admissible only for a *notice*, as `isRejection`
  /// classifies it.
  ConnectionChoiceUnexplored,
};

/// The stable string for `code` (for example `no_matching_rule`). Never empty.
llvm::StringRef stringifyDiagnosticCode(DiagnosticCode code);

/// The code a stable string names, or nullopt when it names none.
std::optional<DiagnosticCode> symbolizeDiagnosticCode(llvm::StringRef name);

/// A stable code plus human-readable detail. The code is the interface; the
/// message is prose and must not be parsed.
struct Diagnostic {
  DiagnosticCode code;
  std::string message;
};

/// Strict weak ordering over diagnostics: by code, then by message. Sorting by
/// this gives a deterministic order independent of how the search reached a
/// failure (design §22.1).
bool diagnosticLess(const Diagnostic &lhs, const Diagnostic &rhs);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_DIAGNOSTICS_H
