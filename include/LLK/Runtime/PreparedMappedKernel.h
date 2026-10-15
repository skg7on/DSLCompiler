//===- PreparedMappedKernel.h - Prepared mapped kernel handle ---*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_RUNTIME_PREPAREDMAPPEDKERNEL_H
#define LLK_RUNTIME_PREPAREDMAPPEDKERNEL_H

#include "LLK/Runtime/KernelAbi.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <string>

namespace llk {
struct PreparedMappedKernel {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  KernelAbi abi;
  std::string entrySymbol;
};
} // namespace llk

#endif // LLK_RUNTIME_PREPAREDMAPPEDKERNEL_H
