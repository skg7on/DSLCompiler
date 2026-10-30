//===- ShapeContract.h - Plain symbolic shape data and binding ------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reusable symbolic shape data: a `Dimension` is either a nonnegative constant
// or a named symbol, a `ShapeContract` relates ordered named port shapes
// through a deterministic symbol table, and `bindShapeValues` checks concrete
// extents against that contract before any consumer performs shape arithmetic.
//
// This header is plain data. It depends on LLVM Support, not MLIR, so workload,
// mapping and runtime integration can all consume the same contract without
// depending on one another. No consumer passes MLIR's dynamic sentinel into
// numeric shape arithmetic: a symbol is bound by an actual port dimension.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_SHAPE_SHAPECONTRACT_H
#define LLK_SHAPE_SHAPECONTRACT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

// The shape library is below the MLIR-facing adapters and depends on LLVM
// Support only. It lives in the project's `llk` namespace tree so consumers
// that also name `mlir::llk` never see an ambiguous top-level `llk`.
namespace mlir::llk::shape {

/// Which side of a workload a shape port describes.
enum class PortDirection { Input, Output };

llvm::StringRef stringifyDirection(PortDirection direction);
std::optional<PortDirection> symbolizeDirection(llvm::StringRef text);

/// One tagged extent: a nonnegative constant or a nonempty symbol name.
///
/// Construction never rejects a value; `validateShapeContract` rejects a
/// negative constant or an empty symbol name. Two dimensions are equal only
/// when they have the same tag and the same value, so a constant never
/// silently equals a symbol that happens to spell its value.
class Dimension {
public:
  Dimension() = default;
  /// Retained as an implicit conversion so static construction keeps its
  /// existing spelling (`{2, 3, 4}` and `= 5`).
  Dimension(int64_t constant) : constant_(constant) {}

  static Dimension symbol(std::string name);

  bool isConstant() const { return !symbolic_; }
  bool isSymbol() const { return symbolic_; }
  int64_t getConstant() const { return constant_; }
  llvm::StringRef getSymbol() const { return symbol_; }

  friend bool operator==(const Dimension &lhs, const Dimension &rhs) {
    return lhs.symbolic_ == rhs.symbolic_ && lhs.constant_ == rhs.constant_ &&
           lhs.symbol_ == rhs.symbol_;
  }

private:
  int64_t constant_ = 0;
  std::string symbol_;
  bool symbolic_ = false;
};

inline std::ostream &operator<<(std::ostream &os, const Dimension &dimension) {
  if (dimension.isSymbol())
    return os << dimension.getSymbol().str();
  return os << dimension.getConstant();
}

/// An inclusive nonnegative minimum, an optional inclusive maximum and a
/// positive congruence divisor for one symbol's bound extents.
struct SymbolConstraint {
  int64_t min = 0;
  std::optional<int64_t> max;
  int64_t multipleOf = 1;

  friend bool operator==(const SymbolConstraint &lhs,
                         const SymbolConstraint &rhs) {
    return lhs.min == rhs.min && lhs.max == rhs.max &&
           lhs.multipleOf == rhs.multipleOf;
  }
};

/// One ordered, named port shape. Port order is identity and is never sorted.
struct ShapePort {
  std::string name;
  PortDirection direction = PortDirection::Input;
  std::vector<Dimension> shape;
};

/// Ordered named port shapes plus a deterministic symbol table. Reusing a
/// symbol name asserts equality; the encoding never infers equality from two
/// unrelated dynamic dimensions.
struct ShapeContract {
  std::vector<ShapePort> ports;
  std::map<std::string, SymbolConstraint> symbols;
};

/// Concrete extents supplied for one port, identified by direction and by its
/// zero-based position among the ports of that direction.
struct PortShapeValues {
  PortDirection direction = PortDirection::Input;
  uint32_t index = 0;
  std::vector<int64_t> extents;
};

/// Every declared symbol bound to its concrete extent.
using ShapeValues = std::map<std::string, int64_t>;

/// Rejects rank-zero ports, duplicate port names, negative constants, empty
/// symbol names, declared-but-unused and used-but-undeclared symbols, and
/// unsatisfiable bounds or divisors. Diagnostics are `shape_contract:`
/// prefixed and name the field that failed.
llvm::Error validateShapeContract(const ShapeContract &contract);

/// Binds concrete extents to every port of `contract` and returns the symbol
/// values. Every port must be supplied exactly once. Constant dimensions and
/// repeated symbols must agree, extents are nonnegative, and bound values must
/// satisfy their declared constraints. Extents are checked even when another
/// dimension is zero, so an empty invocation cannot hide a contradiction.
llvm::Expected<ShapeValues>
bindShapeValues(const ShapeContract &contract,
                llvm::ArrayRef<PortShapeValues> values);

/// Checks a complete symbol binding without port extents: every declared symbol
/// must occur exactly once, no unknown symbol may be present, and each bound
/// value must be nonnegative and satisfy its declared constraint. This is the
/// check a shape sample performs before an analysis clone is materialized.
llvm::Error validateShapeValues(const ShapeContract &contract,
                                const ShapeValues &values);

/// The canonical, collision-safe identity of a contract. Every variable-length
/// string is length-delimited, constants and symbols carry distinct tags, and
/// port order is preserved while the symbol table is lexical.
std::string canonicalShapeContract(const ShapeContract &contract);

//===----------------------------------------------------------------------===//
// Predicate classification (dynamic-shape design §6)
//===----------------------------------------------------------------------===//

/// How a predicate over the declared shape domain is decided:
///
///   * `Proven` -- it holds for every extent the contract admits;
///   * `Disproven` -- it holds for none, so a plan relying on it is illegal;
///   * `RuntimeGuard` -- it depends on the invocation, so the executable form
///     must carry a guard (or fall back to bounded movements) rather than
///     assume it. Symbolic divisibility is deliberately *not* automatically
///     legal: only a proven predicate lets a full-tile form replace a bounded
///     one.
enum class PredicateVerdict { Proven, Disproven, RuntimeGuard };

llvm::StringRef stringifyPredicateVerdict(PredicateVerdict verdict);

/// One predicate the mapping layer can classify without a solver.
struct ShapePredicate {
  enum class Kind {
    /// `symbol % value == 0`.
    DivisibleBy,
    /// `symbol == value`.
    EqualTo,
    /// `symbol >= value`.
    AtLeast,
  };

  Kind kind = Kind::DivisibleBy;
  std::string symbol;
  int64_t value = 1;
};

/// Classifies `predicate` against the domain `contract` declares for its
/// symbol. The answer is conservative in both directions: a predicate is
/// `Proven` only when every admissible extent satisfies it and `Disproven` only
/// when none can. Everything else -- including a symbol the contract does not
/// declare, which validation already rejects -- is `RuntimeGuard`, because a
/// fact that cannot be established is not a fact that holds.
PredicateVerdict classifyPredicate(const ShapePredicate &predicate,
                                   const ShapeContract &contract);

} // namespace mlir::llk::shape

#endif // LLK_SHAPE_SHAPECONTRACT_H
