//===- LLKToLinalg.h - Lower LLK ops to Linalg + Arith + Math ----------*- C++ -*-===//
//
// Pass header for the LLK-to-Linalg lowering conversion.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_LLKTOLINALG_H
#define LLK_CONVERSION_LLKTOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createLLKToLinalgPass();
} // namespace llk
} // namespace mlir

#endif // LLK_CONVERSION_LLKTOLINALG_H
