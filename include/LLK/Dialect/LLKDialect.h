//===- LLKDialect.h - LLK dialect public header -----------------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Public header for the LLK dialect.  Includes the necessary MLIR headers
// before the TableGen-generated dialect declaration.

#ifndef LLK_DIALECT_LLKDIALECT_H
#define LLK_DIALECT_LLKDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

// TableGen-generated dialect class declaration.
#include "LLK/Dialect/LLKDialect.h.inc"

#endif // LLK_DIALECT_LLKDIALECT_H
