//===- TritonCPUVerifier.h - reject unsupported GPU ops --------*- C++ -*-===//
//
// Pass header for the Triton CPU verifier. Runs before Stage 2 to reject
// GPU-only Triton constructs that the CPU lowering cannot handle.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_TRITONTOLLK_TRITONCPUVERIFIER_H
#define LLK_CONVERSION_TRITONTOLLK_TRITONCPUVERIFIER_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createTritonCPUVerifierPass();
} // namespace llk
} // namespace mlir

#endif // LLK_CONVERSION_TRITONTOLLK_TRITONCPUVERIFIER_H
