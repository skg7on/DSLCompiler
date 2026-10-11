#ifndef LLK_MAPPING_CODEGENREQUIREMENTS_H
#define LLK_MAPPING_CODEGENREQUIREMENTS_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace mlir::llk::mapping {
struct TargetCodegenRequirements {
  std::string architecture;
  std::string cpu;
  std::vector<std::string> requiredFeatures;
};

/// Check a selected target's ISA policy against the detected host facts.
llvm::Error checkHostRequirements(const TargetCodegenRequirements &requirements,
                                  llvm::StringRef hostArchitecture,
                                  const llvm::StringMap<bool> &hostFeatures);

} // namespace mlir::llk::mapping
#endif
