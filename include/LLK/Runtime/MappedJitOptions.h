#ifndef LLK_RUNTIME_MAPPEDJITOPTIONS_H
#define LLK_RUNTIME_MAPPEDJITOPTIONS_H
#include "LLK/Mapping/CodegenRequirements.h"
#include <optional>
#include <string>
namespace llk {
struct MappedJitOptions {
  std::optional<mlir::llk::mapping::TargetCodegenRequirements> selectedTarget;
  /// Stable, human-readable compiler/target/math identity supplied by the
  /// mapped compilation path. Runtime appends ABI and lowered-IR hashes.
  std::string executionIdentity;
};
} // namespace llk
#endif
