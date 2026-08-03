//===- SharedMemToScratch.h - tt.alloc → memref.alloc -----------*- C++ -*-===//
//
// Pass header for lowering Triton shared-memory allocs and async copies to
// per-worker memref scratch buffers and synchronous memref.copy.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_TRITONTOLLK_SHAREDMEMTOSCRATCH_H
#define LLK_CONVERSION_TRITONTOLLK_SHAREDMEMTOSCRATCH_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createSharedMemToScratchPass();
} // namespace llk
} // namespace mlir

#endif // LLK_CONVERSION_TRITONTOLLK_SHAREDMEMTOSCRATCH_H
