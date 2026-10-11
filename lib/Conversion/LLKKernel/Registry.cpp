#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
namespace mlir::llk {
const LLKKernelLowering *LLKKernelRegistry::lookup(StringRef name) const {
  auto it = byName.find(name);
  return it == byName.end() ? nullptr : it->second;
}
void registerSwiGLULowering(LLKKernelRegistry &);
void populateLLKKernelLowerings(LLKKernelRegistry &r) {
  registerSwiGLULowering(r);
}
} // namespace mlir::llk
