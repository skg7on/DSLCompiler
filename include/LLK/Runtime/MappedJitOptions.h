#ifndef LLK_RUNTIME_MAPPEDJITOPTIONS_H
#define LLK_RUNTIME_MAPPEDJITOPTIONS_H
#include "LLK/Mapping/CodegenRequirements.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
namespace llk {
/// Optional, pre-JIT evidence for a mapped compilation. The callback is only
/// invoked when a caller opts in; it sees the exact module and identity handed
/// to ORC, together with the selected-plan and ABI facts used to create it.
struct MappedJitEvidence {
  std::string entrySymbol;
  std::string executionIdentity;
  std::string targetName;
  uint64_t planId = 0;
  uint64_t machineHash = 0;
  unsigned selectedGroupsVerified = 0;
  unsigned backendGroupsRealized = 0;
  unsigned referenceGroupsLowered = 0;
  uint64_t abiHash = 0;
  std::optional<mlir::llk::mapping::TargetCodegenRequirements> selectedTarget;
  std::string llvmDialect;
  std::string llvmIR;
};

using MappedJitEvidenceSink =
    std::function<llvm::Error(const MappedJitEvidence &)>;

struct MappedJitOptions {
  std::optional<mlir::llk::mapping::TargetCodegenRequirements> selectedTarget;
  /// Stable, human-readable compiler/target/math identity supplied by the
  /// mapped compilation path. Runtime appends ABI and lowered-IR hashes.
  std::string executionIdentity;
  std::string targetName;
  uint64_t planId = 0;
  uint64_t machineHash = 0;
  unsigned selectedGroupsVerified = 0;
  unsigned backendGroupsRealized = 0;
  unsigned referenceGroupsLowered = 0;
  MappedJitEvidenceSink evidenceSink;
};
} // namespace llk
#endif
