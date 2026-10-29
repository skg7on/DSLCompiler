//===- MappedInvocation.h - Typed mapped-invocation buffers -----*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_RUNTIME_MAPPEDINVOCATION_H
#define LLK_RUNTIME_MAPPEDINVOCATION_H
#include "LLK/Runtime/JitCache.h"
#include "LLK/Runtime/KernelAbi.h"
#include "llvm/ADT/ArrayRef.h"
namespace llk {
enum class InvocationElementType { F32, BF16, F16, I32, I8 };
struct InvocationBuffer2D {
  MemRef2D descriptor{};
  // Declared by the caller; this tag cannot inspect the backing bytes.
  InvocationElementType elementType = InvocationElementType::F32;
  uint64_t allocationBytes = 0; // bytes from descriptor.allocated
};
llvm::Error
validateMappedInvocation(const KernelAbi &abi,
                         llvm::ArrayRef<InvocationBuffer2D> inputs,
                         llvm::ArrayRef<InvocationBuffer2D> outputs);

/// Validates an invocation and returns the concrete extent of every symbol the
/// ABI declares.
///
/// A successful resolution means the whole invocation is admissible: arity,
/// dtype, rank, offsets, compact strides, nonnegative and non-overflowing
/// extents, allocation bounds, disjoint outputs, and every declared symbol
/// bound consistently. Version 1 returns an empty value map because it declares
/// no symbols; its validation is otherwise the same one it always performed.
/// An empty invocation (a zero extent with a null pointer and zero bytes) is
/// valid, and its zero extent still participates in symbol binding.
llvm::Expected<::mlir::llk::shape::ShapeValues>
resolveMappedShapeValues(const KernelAbi &abi,
                         llvm::ArrayRef<InvocationBuffer2D> inputs,
                         llvm::ArrayRef<InvocationBuffer2D> outputs);
} // namespace llk
#endif
