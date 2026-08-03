//===- AtomicToLLRT.h - tt.atomic_* → llrt runtime calls -------*- C++ -*-===//
//
// Pass header for lowering Triton atomic ops to llrt runtime calls.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_TRITONTOLLK_ATOMICTOLLRT_H
#define LLK_CONVERSION_TRITONTOLLK_ATOMICTOLLRT_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createAtomicToLLRTPass();
} // namespace llk
} // namespace mlir

#endif // LLK_CONVERSION_TRITONTOLLK_ATOMICTOLLRT_H
