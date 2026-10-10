#ifndef LLK_MAPPING_CODEGENREQUIREMENTS_H
#define LLK_MAPPING_CODEGENREQUIREMENTS_H
#include <string>
#include <vector>
namespace mlir::llk::mapping {
struct TargetCodegenRequirements {
  std::string architecture;
  std::string cpu;
  std::vector<std::string> requiredFeatures;
};
} // namespace mlir::llk::mapping
#endif
