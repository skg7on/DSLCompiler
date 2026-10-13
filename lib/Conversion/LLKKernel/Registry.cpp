#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
namespace mlir::llk {
const LLKKernelLowering *LLKKernelRegistry::lookup(StringRef name) const {
  auto it = byName.find(name);
  return it == byName.end() ? nullptr : it->second;
}
void registerSwiGLULowering(LLKKernelRegistry &);
void registerRoPELowering(LLKKernelRegistry &);
void registerAttentionLowering(LLKKernelRegistry &);
void registerMatmulLowering(LLKKernelRegistry &);
void registerLinalgMatmulLowering(LLKKernelRegistry &);
void populateLLKKernelLowerings(LLKKernelRegistry &r) {
  registerSwiGLULowering(r);       // Linalg + Micro
  registerRoPELowering(r);         // Linalg
  registerAttentionLowering(r);    // Linalg
  registerMatmulLowering(r);       // Linalg + Micro
  registerLinalgMatmulLowering(r); // Micro compatibility adapter
}
} // namespace mlir::llk
