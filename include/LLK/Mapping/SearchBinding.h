//===- SearchBinding.h - Immutable point in a search space ----------------===//
//
// Part of the target-independent mapping core (issue #80, epic #67 D1).
//
// A `SearchBinding` is the handoff from the persistent search IR to the
// mapping engine: one `micro.candidate`'s parameter-to-value assignment,
// typed and immutable. It is deliberately not called a "candidate" here --
// the internal rule match the mapping engine produces is a `MappingCandidate`
// (MappingPlan.h), and keeping the two names distinct is what stops a plan's
// rule choice from being confused with the search point that led to it.
//
// The content hash is over the *values only*, so two bindings that differ
// only in their symbolic candidate name are recognisably the same point; the
// name is preserved for diagnostics and report provenance, and breaks ties in
// canonical ordering.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_SEARCHBINDING_H
#define LLK_MAPPING_SEARCHBINDING_H

#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <string>
#include <variant>

namespace mlir::llk::mapping {

/// One bound parameter value: an integer choice or a symbolic name. Keeping
/// both in one variant mirrors `micro.param`, whose choices are typed by the
/// parameter's declared kind.
using SearchValue = std::variant<int64_t, std::string>;

/// One search-space point. Immutable after `makeSearchBinding`; every plan
/// records `stableHash` so a selected plan can be traced to its binding.
struct SearchBinding {
  /// The `micro.candidate` symbol, or a derived label when the binding was
  /// generated rather than loaded.
  std::string candidateId;

  /// Parameter name to value, in no meaningful order; hashing and printing
  /// sort the keys.
  llvm::StringMap<SearchValue> values;

  /// Content hash of `values` alone (see canonicalSearchBindingString).
  uint64_t stableHash = 0;
};

/// Order-independent content hash of a binding's values. The value's kind is
/// folded in, so `x = 1` and `x = "1"` hash differently.
uint64_t computeSearchBindingHash(const llvm::StringMap<SearchValue> &values);

/// Builds a binding and computes its `stableHash`.
SearchBinding makeSearchBinding(std::string candidateId,
                                llvm::StringMap<SearchValue> values);

/// Sorted, type-tagged rendering: `a=s:z;b=i:2`. Sorted for determinism and
/// tagged so an integer and a symbolic value never print alike.
std::string canonicalSearchBindingString(const SearchBinding &binding);

/// Canonical order: by value content, then by `candidateId` as a tie-break, so
/// two bindings with the same values still order deterministically.
bool searchBindingLess(const SearchBinding &lhs, const SearchBinding &rhs);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_SEARCHBINDING_H
