//===- CompiledVariantIdentity.cpp - Replay identity and variant sets -----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/CompiledVariantIdentity.h"

#include "llvm/Support/ErrorHandling.h"

#include <set>
#include <string>
#include <utility>

namespace llk {

namespace {

using ::mlir::llk::shape::ShapePredicate;
using ::mlir::llk::shape::ShapeValues;

/// Length-delimited field, so concatenations can never be ambiguous.
void appendField(std::string &out, llvm::StringRef value) {
  out += std::to_string(value.size());
  out += ':';
  out += value.str();
}

uint64_t fnv1a(llvm::StringRef value) {
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char byte : value.bytes()) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

llvm::Error variantError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dynamic_variant_set: %s",
                                 message.str().c_str());
}

llvm::StringRef predicateKindName(ShapePredicate::Kind kind) {
  switch (kind) {
  case ShapePredicate::Kind::DivisibleBy:
    return "divisible_by";
  case ShapePredicate::Kind::EqualTo:
    return "equal_to";
  case ShapePredicate::Kind::AtLeast:
    return "at_least";
  }
  llvm_unreachable("unknown shape predicate kind");
}

bool guardHolds(const ShapePredicate &predicate, const ShapeValues &values) {
  auto found = values.find(predicate.symbol);
  if (found == values.end())
    return false;
  const int64_t extent = found->second;
  if (extent < 0)
    return false;
  switch (predicate.kind) {
  case ShapePredicate::Kind::DivisibleBy:
    return predicate.value > 0 && extent % predicate.value == 0;
  case ShapePredicate::Kind::EqualTo:
    return extent == predicate.value;
  case ShapePredicate::Kind::AtLeast:
    return extent >= predicate.value;
  }
  llvm_unreachable("unknown shape predicate kind");
}

llvm::Error validateGuard(const ShapePredicate &predicate) {
  if (predicate.symbol.empty())
    return variantError("guard predicate has an empty symbol");
  if (predicate.kind == ShapePredicate::Kind::DivisibleBy &&
      predicate.value <= 0)
    return variantError("guard predicate for '" + predicate.symbol +
                        "' has a non-positive divisor");
  return llvm::Error::success();
}

} // namespace

std::string canonicalShapeGuards(llvm::ArrayRef<ShapePredicate> guards) {
  std::string out;
  appendField(out, "llk-shape-guards-v1");
  appendField(out, std::to_string(guards.size()));
  for (const ShapePredicate &guard : guards) {
    appendField(out, predicateKindName(guard.kind));
    appendField(out, guard.symbol);
    appendField(out, std::to_string(guard.value));
  }
  return out;
}

CompiledVariantIdentity
makeCompiledVariantIdentity(const CompiledVariantIdentityInputs &inputs) {
  std::string canonical;
  appendField(canonical, "llk-compiled-variant");
  appendField(canonical, std::to_string(kCompiledVariantIdentityVersion));
  appendField(canonical, inputs.sourceContent);
  appendField(canonical, inputs.symbolicContract);
  appendField(canonical, std::to_string(inputs.symbolicContractVersion));
  appendField(canonical, inputs.numericalContract);
  appendField(canonical, canonicalShapeGuards(inputs.guards));
  appendField(canonical, std::to_string(inputs.abiVersion));
  appendField(canonical, std::to_string(inputs.abiHash));
  appendField(canonical, inputs.staticSchedule);
  appendField(canonical, inputs.targetIdentity);
  appendField(canonical, inputs.backendIdentity);
  appendField(canonical, inputs.executionMode);
  appendField(canonical, inputs.compilerVersion);

  CompiledVariantIdentity identity;
  identity.canonical = std::move(canonical);
  identity.contentHash = std::to_string(fnv1a(identity.canonical));
  return identity;
}

std::string canonicalMeasurementIdentity(llvm::StringRef variantCanonical,
                                         const ShapeValues &values,
                                         llvm::StringRef profileCanonical) {
  std::string out;
  appendField(out, "llk-measurement-v2");
  appendField(out, variantCanonical);
  appendField(out, std::to_string(values.size()));
  for (const auto &entry : values) {
    appendField(out, entry.first);
    appendField(out, std::to_string(entry.second));
  }
  appendField(out, profileCanonical);
  return out;
}

llvm::Expected<DynamicVariantSet>
DynamicVariantSet::create(std::vector<GuardedVariant> variants,
                          uint32_t maxVariants) {
  if (maxVariants == 0)
    return variantError("max_variants must be positive");
  if (variants.empty())
    return variantError("a variant set needs at least one variant");
  if (variants.size() > maxVariants)
    return variantError(
        "variant budget exceeded: " + std::to_string(variants.size()) +
        " variants > max " + std::to_string(maxVariants));

  size_t baselines = 0;
  std::set<std::string> ids;
  for (size_t i = 0; i < variants.size(); ++i) {
    const GuardedVariant &variant = variants[i];
    if (variant.id.empty())
      return variantError("variant id must not be empty");
    if (!ids.insert(variant.id).second)
      return variantError("duplicate variant id '" + variant.id + "'");
    for (const ShapePredicate &guard : variant.guards)
      if (llvm::Error error = validateGuard(guard))
        return std::move(error);
    if (variant.baseline) {
      ++baselines;
      if (!variant.guards.empty())
        return variantError("the full-domain baseline cannot declare guards");
      if (i + 1 != variants.size())
        return variantError(
            "the full-domain baseline must be the final variant");
    } else if (variant.guards.empty()) {
      return variantError("guarded variant '" + variant.id +
                          "' must declare at least one guard");
    }
  }
  if (baselines == 0)
    return variantError("a variant set needs exactly one full-domain baseline");
  if (baselines > 1)
    return variantError("a variant set declares more than one baseline");

  DynamicVariantSet set;
  set.variants_ = std::move(variants);
  set.maxVariants_ = maxVariants;
  return set;
}

const GuardedVariant &
DynamicVariantSet::select(const ShapeValues &values) const {
  for (const GuardedVariant &variant : variants_) {
    if (variant.baseline)
      continue;
    bool matches = true;
    for (const ShapePredicate &guard : variant.guards) {
      if (!guardHolds(guard, values)) {
        matches = false;
        break;
      }
    }
    if (matches)
      return variant;
  }
  // Validated at construction: the final variant is the full-domain baseline,
  // so an uncovered shape always has a deterministic final choice.
  return variants_.back();
}

std::string DynamicVariantSet::canonical() const {
  std::string out;
  appendField(out, "llk-dynamic-variants");
  appendField(out, std::to_string(kDynamicVariantSetVersion));
  appendField(out, std::to_string(maxVariants_));
  appendField(out, std::to_string(variants_.size()));
  for (const GuardedVariant &variant : variants_) {
    appendField(out, variant.id);
    appendField(out, variant.identity.canonical);
    appendField(out, canonicalShapeGuards(variant.guards));
    appendField(out, variant.baseline ? "1" : "0");
  }
  return out;
}

std::string DynamicVariantSet::contentHash() const {
  return std::to_string(fnv1a(canonical()));
}

} // namespace llk
