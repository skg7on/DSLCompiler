//===- Diagnostics.cpp - Stable diagnostic codes (D6, §22.3)
//---------------===//
//
// Hand-written stringify/symbolize, matching the project idiom for a small,
// closed enum (see MicroEnums.h): a plain switch for the one direction and
// `StringSwitch` for the other, so the compiler flags a case added to the enum
// but forgotten here.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/Diagnostics.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::llk::mapping {

llvm::StringRef stringifyDiagnosticCode(DiagnosticCode code) {
  switch (code) {
  case DiagnosticCode::NoMatchingRule:
    return "no_matching_rule";
  case DiagnosticCode::NoLegalLayout:
    return "no_legal_layout";
  case DiagnosticCode::NoLegalExecutor:
    return "no_legal_executor";
  case DiagnosticCode::MemoryCapacityExceeded:
    return "memory_capacity_exceeded";
  case DiagnosticCode::UnsupportedComputeFragment:
    return "unsupported_compute_fragment";
  case DiagnosticCode::NoMemoryRoute:
    return "no_memory_route";
  case DiagnosticCode::NoLayoutTransform:
    return "no_layout_transform";
  case DiagnosticCode::GlobalConstraintFailed:
    return "global_constraint_failed";
  case DiagnosticCode::SearchTruncated:
    return "search_truncated";
  case DiagnosticCode::LatencyCacheMiss:
    return "latency_cache_miss";
  case DiagnosticCode::TargetBundleInvalid:
    return "target_bundle_invalid";
  case DiagnosticCode::AssumedValueSize:
    return "assumed_value_size";
  case DiagnosticCode::InvalidMappingMetadata:
    return "invalid_mapping_metadata";
  case DiagnosticCode::ConnectionChoiceUnexplored:
    return "connection_choice_unexplored";
  case DiagnosticCode::InvalidGatherDeclaration:
    return "invalid_gather_declaration";
  case DiagnosticCode::UnsupportedMaterialization:
    return "unsupported_materialization";
  }
  return "";
}

std::optional<DiagnosticCode> symbolizeDiagnosticCode(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<DiagnosticCode>>(name)
      .Case("no_matching_rule", DiagnosticCode::NoMatchingRule)
      .Case("no_legal_layout", DiagnosticCode::NoLegalLayout)
      .Case("no_legal_executor", DiagnosticCode::NoLegalExecutor)
      .Case("memory_capacity_exceeded", DiagnosticCode::MemoryCapacityExceeded)
      .Case("unsupported_compute_fragment",
            DiagnosticCode::UnsupportedComputeFragment)
      .Case("no_memory_route", DiagnosticCode::NoMemoryRoute)
      .Case("no_layout_transform", DiagnosticCode::NoLayoutTransform)
      .Case("global_constraint_failed", DiagnosticCode::GlobalConstraintFailed)
      .Case("search_truncated", DiagnosticCode::SearchTruncated)
      .Case("latency_cache_miss", DiagnosticCode::LatencyCacheMiss)
      .Case("target_bundle_invalid", DiagnosticCode::TargetBundleInvalid)
      .Case("assumed_value_size", DiagnosticCode::AssumedValueSize)
      .Case("invalid_mapping_metadata", DiagnosticCode::InvalidMappingMetadata)
      .Case("connection_choice_unexplored",
            DiagnosticCode::ConnectionChoiceUnexplored)
      .Case("invalid_gather_declaration",
            DiagnosticCode::InvalidGatherDeclaration)
      .Case("unsupported_materialization",
            DiagnosticCode::UnsupportedMaterialization)
      .Default(std::nullopt);
}

bool diagnosticLess(const Diagnostic &lhs, const Diagnostic &rhs) {
  if (lhs.code != rhs.code)
    return static_cast<int>(lhs.code) < static_cast<int>(rhs.code);
  return lhs.message < rhs.message;
}

bool isRejection(DiagnosticCode code) {
  switch (code) {
  // Notices: a cap, a provider gap, or an advisory assumption. None is a
  // refusal the search made.
  case DiagnosticCode::SearchTruncated:
  case DiagnosticCode::LatencyCacheMiss:
  case DiagnosticCode::AssumedValueSize:
  case DiagnosticCode::ConnectionChoiceUnexplored:
    return false;
  // Rejections: the search refused a rule, a placement, a pair, a layout, a
  // global constraint, a bundle, or a plan.
  case DiagnosticCode::NoMatchingRule:
  case DiagnosticCode::NoLegalLayout:
  case DiagnosticCode::NoLegalExecutor:
  case DiagnosticCode::MemoryCapacityExceeded:
  case DiagnosticCode::UnsupportedComputeFragment:
  case DiagnosticCode::NoMemoryRoute:
  case DiagnosticCode::NoLayoutTransform:
  case DiagnosticCode::GlobalConstraintFailed:
  case DiagnosticCode::TargetBundleInvalid:
  case DiagnosticCode::InvalidMappingMetadata:
  case DiagnosticCode::InvalidGatherDeclaration:
  case DiagnosticCode::UnsupportedMaterialization:
    return true;
  }
  llvm_unreachable("unclassified DiagnosticCode");
}

const Diagnostic *primaryRefusal(llvm::ArrayRef<Diagnostic> diagnostics) {
  const Diagnostic *first = nullptr;
  for (const Diagnostic &detail : diagnostics) {
    if (!isRejection(detail.code))
      continue;
    if (!first)
      first = &detail;
    // A plan-level refusal outranks a rule-level one whichever order the codes
    // sort in: it is the one that names the decision the search got as far as
    // choosing and then could not materialize.
    if (detail.code == DiagnosticCode::UnsupportedMaterialization ||
        detail.code == DiagnosticCode::MemoryCapacityExceeded)
      return &detail;
  }
  return first;
}

} // namespace mlir::llk::mapping
