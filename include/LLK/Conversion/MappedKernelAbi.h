//===- MappedKernelAbi.h - Mapped kernel ABI contract -----------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_MAPPEDKERNELABI_H
#define LLK_CONVERSION_MAPPEDKERNELABI_H

#include "LLK/Runtime/PreparedMappedKernel.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llk {
/// Clone a buffered module, move memref results to borrowed output parameters,
/// and insert deallocations for compiler-owned allocations.
llvm::Expected<PreparedMappedKernel>
prepareMappedKernelForInvocation(mlir::ModuleOp bufferedModule,
                                 llvm::StringRef entrySymbol);
} // namespace llk

#endif // LLK_CONVERSION_MAPPEDKERNELABI_H
