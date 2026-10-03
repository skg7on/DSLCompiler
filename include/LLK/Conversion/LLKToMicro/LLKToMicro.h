//===- LLKToMicro.h - LLK -> concrete micro.kernel lowering -----*- C++ -*-===//
//
// Export path from the LLK scheduling pipeline into tile-centric Micro-IR.
//
// The pass is non-destructive: it adds one concrete `micro.kernel` per
// supported LLK root operation and leaves the original functions alone, so the
// AVX2/JIT pipeline downstream of it is untouched.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_LLKTOMICRO_LLKTOMICRO_H
#define LLK_CONVERSION_LLKTOMICRO_LLKTOMICRO_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace llk {

/// Lowers each supported LLK root operation to a concrete `micro.kernel`.
std::unique_ptr<mlir::Pass> createLLKToMicroPass();

} // namespace llk
} // namespace mlir

#endif // LLK_CONVERSION_LLKTOMICRO_LLKTOMICRO_H
