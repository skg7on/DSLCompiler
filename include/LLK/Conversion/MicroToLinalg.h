//===- MicroToLinalg.h - Lower micro kernels to Linalg ----------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Pass header for the Micro-to-Linalg lowering conversion.
//
// `micro` is the canonical execution IR and, until this pass, a sink: nothing
// lowered it to anything the backends understand. This is the bridge that lets
// a `micro.kernel` -- hand-written, exported by `--llk-to-micro`, or bound from
// a mapping plan -- flow through the same downstream pipeline the legacy path
// uses (Linalg -> loops -> LLVM -> ORC JIT).

#ifndef LLK_CONVERSION_MICROTOLINALG_H
#define LLK_CONVERSION_MICROTOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createMicroToLinalgPass();
} // namespace llk
} // namespace mlir

#endif // LLK_CONVERSION_MICROTOLINALG_H
