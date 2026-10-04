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
      .Default(std::nullopt);
}

bool diagnosticLess(const Diagnostic &lhs, const Diagnostic &rhs) {
  if (lhs.code != rhs.code)
    return static_cast<int>(lhs.code) < static_cast<int>(rhs.code);
  return lhs.message < rhs.message;
}

} // namespace mlir::llk::mapping
