//===- CompiledVariantIdentity.h - Replay identity and variant sets -------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamic execution identity has two levels (dynamic-shape design §7).
//
// A *compiled variant* is reusable code: the original source content, the
// symbolic contract and its version, the numerical contract, the guard domain,
// the ABI version and hash, the static schedule/mapping resources, and the
// machine/target/backend/compiler versions. Concrete invocation shapes are
// deliberately absent, so every invocation inside the declared domain reuses
// the same variant.
//
// A *measurement* is that variant plus the concrete shape observation (or the
// profile that selected it) and the measurement protocol. Code reuse never
// implies latency reuse: two shapes that run the same baseline are distinct
// measurements.
//
// Optional guarded specialization is represented by an ordered
// `DynamicVariantSet`: guarded variants first, exactly one full-domain baseline
// last, selected by the first matching guard and bounded by an explicit budget.
// The canonical form of the set is the frozen replay identity; declaration
// order is semantic and stable.
//
// This header is plain data. It depends on the plain shape contract and the
// runtime ABI constants, not on MLIR dialects or the JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_RUNTIME_COMPILEDVARIANTIDENTITY_H
#define LLK_RUNTIME_COMPILEDVARIANTIDENTITY_H

#include "LLK/Shape/ShapeContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llk {

/// The encoding version of a compiled-variant identity. Bump it whenever the
/// fields or their rendering change, so old and new artifacts cannot collide.
inline constexpr uint32_t kCompiledVariantIdentityVersion = 1;

/// The encoding version of a frozen `DynamicVariantSet` description.
inline constexpr uint32_t kDynamicVariantSetVersion = 1;

/// The reusable-code half of execution identity.
struct CompiledVariantIdentity {
  uint32_t version = kCompiledVariantIdentityVersion;
  /// Length-delimited canonical rendering of every identity field.
  std::string canonical;
  /// Deterministic FNV-1a content hash of `canonical`.
  std::string contentHash;

  bool operator==(const CompiledVariantIdentity &) const = default;
};

/// Everything that changes generated code for a dynamic kernel. Invocation
/// extents do not belong here; they belong to the measurement identity.
struct CompiledVariantIdentityInputs {
  /// Original source module content (or its faithful rendering).
  std::string sourceContent;
  /// Canonical symbolic workload/shape contract.
  std::string symbolicContract;
  /// Version of the contract encoding.
  uint32_t symbolicContractVersion = 0;
  /// Numerical contract: accumulator semantics, math mode, conversion points.
  std::string numericalContract;
  /// Range/congruence predicates a specialized variant requires. Empty for the
  /// full-domain baseline.
  std::vector<::mlir::llk::shape::ShapePredicate> guards;
  uint32_t abiVersion = 0;
  uint64_t abiHash = 0;
  /// Static schedule, selected plan and its resources.
  std::string staticSchedule;
  /// Machine and declarative target policy identity.
  std::string targetIdentity;
  /// Backend/execution identity (compiler flags, ISA features, math mode).
  std::string backendIdentity;
  /// Selected-target or reference execution mode.
  std::string executionMode;
  std::string compilerVersion;
};

/// Length-delimited canonical rendering of guard predicates in declaration
/// order. The order is preserved because a set's dispatch order is semantic.
std::string
canonicalShapeGuards(llvm::ArrayRef<::mlir::llk::shape::ShapePredicate> guards);

/// Builds the reusable-code identity. Two calls with equal inputs produce the
/// same canonical string and content hash; any changed field changes both.
CompiledVariantIdentity
makeCompiledVariantIdentity(const CompiledVariantIdentityInputs &inputs);

/// Builds the measurement identity from a compiled variant and one concrete
/// observation. `values` binds the contract's symbols, and `profileCanonical`
/// identifies the ordered sample/weight set that selected the variant. Either
/// observation is enough to separate measurements; both empty is still a
/// valid static measurement whose shapes are already part of the code content.
std::string
canonicalMeasurementIdentity(llvm::StringRef variantCanonical,
                             const ::mlir::llk::shape::ShapeValues &values,
                             llvm::StringRef profileCanonical = {});

/// One compiled variant and the shape domain it accepts.
struct GuardedVariant {
  /// Stable caller-visible identifier; unique within a set.
  std::string id;
  CompiledVariantIdentity identity;
  /// Conjunction of predicates. Empty only for the full-domain baseline.
  std::vector<::mlir::llk::shape::ShapePredicate> guards;
  /// True for the single full-domain baseline, which must be last.
  bool baseline = false;
};

/// Ordered guarded variants plus one full-domain baseline.
class DynamicVariantSet {
public:
  /// Validates and freezes `variants`.
  ///
  /// Rejects: a zero budget, more variants than the budget, no baseline, more
  /// than one baseline, a baseline that is not last or that declares guards, a
  /// guarded variant without guards, duplicate or empty ids, and malformed
  /// predicates (empty symbol, non-positive divisor).
  static llvm::Expected<DynamicVariantSet>
  create(std::vector<GuardedVariant> variants, uint32_t maxVariants);

  llvm::ArrayRef<GuardedVariant> variants() const { return variants_; }
  const GuardedVariant &baseline() const { return variants_.back(); }
  uint32_t maxVariants() const { return maxVariants_; }

  /// First variant in declaration order whose guards all hold for `values`.
  /// A predicate over an unbound symbol does not hold, so selection falls
  /// through to the baseline. The baseline therefore always matches.
  const GuardedVariant &
  select(const ::mlir::llk::shape::ShapeValues &values) const;

  /// Frozen replay identity: version, budget, and every variant's id, identity,
  /// guards and baseline flag in declaration order.
  std::string canonical() const;

  /// Deterministic FNV-1a content hash of `canonical()`.
  std::string contentHash() const;

private:
  /// A set only exists in the validated form `create` produces; an
  /// unvalidated empty set must not be able to answer `select`.
  DynamicVariantSet() = default;

  std::vector<GuardedVariant> variants_;
  uint32_t maxVariants_ = 1;
};

} // namespace llk

#endif // LLK_RUNTIME_COMPILEDVARIANTIDENTITY_H
