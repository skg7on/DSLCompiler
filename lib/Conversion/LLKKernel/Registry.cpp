#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
namespace mlir::llk {
const LLKKernelLowering *LLKKernelRegistry::lookup(StringRef name) const {
  auto it = byName.find(name);
  return it == byName.end() ? nullptr : it->second;
}
void populateLLKKernelLowerings(LLKKernelRegistry &) {}
} // namespace mlir::llk
