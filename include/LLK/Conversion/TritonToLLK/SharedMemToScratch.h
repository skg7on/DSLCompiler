//===- SharedMemToScratch.h - tt.alloc → memref.alloc -----------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Pass header for lowering Triton shared-memory allocs and async copies to
// per-worker memref scratch buffers and synchronous memref.copy.

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
