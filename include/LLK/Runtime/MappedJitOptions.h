#ifndef LLK_RUNTIME_MAPPEDJITOPTIONS_H
#define LLK_RUNTIME_MAPPEDJITOPTIONS_H
#include "LLK/Mapping/CodegenRequirements.h"
#include <optional>
namespace llk {
struct MappedJitOptions {
  std::optional<mlir::llk::mapping::TargetCodegenRequirements> selectedTarget;
};
} // namespace llk
#endif
