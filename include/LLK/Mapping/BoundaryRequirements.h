//===- BoundaryRequirements.h - Typed boundary behavior ---------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic-shape design §5/§6: a plan whose boundary work is partial -- a
// clipped load or store, or a reduction over fewer than the tile's full K
// capacity -- needs a target implementation that can realize that behaviour.
// The requirement is *typed*: generic mapping code states which behaviours the
// covered operations need, and a target's emitter answers whether it implements
// them. Generic code never reads a bundle parameter as a capability, so a rule
// file cannot impersonate a backend by spelling a parameter a particular way.
//
// The rule still selects the implementation: it names the emitter key, and the
// emitter behind that key is what opts in. A target that declares an emitter
// for an operation but has no boundary implementation inherits the rejecting
// default and the plan is refused with a reason rather than lowered into
// full-tile accesses.

#ifndef LLK_MAPPING_BOUNDARYREQUIREMENTS_H
#define LLK_MAPPING_BOUNDARYREQUIREMENTS_H

#include <string>

namespace mlir::llk::mapping {

/// The boundary behaviours one covered operation group requires.
///
/// Each flag is set only when the operation's boundary access is *not*
/// provably full: a constant valid extent equal to the tile capacity keeps the
/// existing full-tile path and asks for nothing. A runtime extent is a
/// `shape::PredicateVerdict::RuntimeGuard`, which is exactly the case the
/// bounded implementation -- not the full-tile one -- must realize.
struct BoundaryRequirements {
  /// A clipped source access into a fixed-capacity tile (`micro.tile_load`).
  bool boundedLoad = false;
  /// A clipped destination write of a fixed-capacity tile
  /// (`micro.tile_store` with valid extents).
  bool boundedStore = false;
  /// A reduction shorter than the tile's declared K capacity
  /// (`micro.mma` with a valid-K bound).
  bool partialK = false;

  /// True when any boundary behaviour is required. The empty set is the
  /// pre-existing full-tile contract and every emitter accepts it.
  bool any() const { return boundedLoad || boundedStore || partialK; }

  friend bool operator==(const BoundaryRequirements &lhs,
                         const BoundaryRequirements &rhs) {
    return lhs.boundedLoad == rhs.boundedLoad &&
           lhs.boundedStore == rhs.boundedStore && lhs.partialK == rhs.partialK;
  }
};

/// Deterministic rendering of the required behaviours, in the fixed order
/// `bounded_load, bounded_store, partial_k`, or `none` for the empty set.
/// Diagnostics name the same order, so two refusals for the same requirement
/// are byte-identical.
inline std::string renderBoundaryRequirements(const BoundaryRequirements &r) {
  std::string text;
  auto append = [&](bool present, const char *name) {
    if (!present)
      return;
    if (!text.empty())
      text += ", ";
    text += name;
  };
  append(r.boundedLoad, "bounded_load");
  append(r.boundedStore, "bounded_store");
  append(r.partialK, "partial_k");
  if (text.empty())
    text = "none";
  return text;
}

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_BOUNDARYREQUIREMENTS_H
