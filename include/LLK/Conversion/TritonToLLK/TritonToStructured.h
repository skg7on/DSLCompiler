//===- TritonToStructured.h - tt.dot → linalg.matmul ------------*- C++ -*-===//
//
// Pass header for lowering Triton compute ops to Linalg structured ops.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_TRITONTOLLK_TRITONTOSTRUCTURED_H
#define LLK_CONVERSION_TRITONTOLLK_TRITONTOSTRUCTURED_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createTritonToStructuredPass();
} // namespace llk
} // namespace mlir

#endif // LLK_CONVERSION_TRITONTOLLK_TRITONTOSTRUCTURED_H
