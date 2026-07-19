//===- LLKDialect.h - LLK dialect public header -----------------*- C++ -*-===//
//
// Public header for the LLK dialect.  Includes the necessary MLIR headers
// before the TableGen-generated dialect declaration.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_DIALECT_LLKDIALECT_H
#define LLK_DIALECT_LLKDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

// TableGen-generated dialect class declaration.
#include "LLK/Dialect/LLKDialect.h.inc"

#endif // LLK_DIALECT_LLKDIALECT_H
