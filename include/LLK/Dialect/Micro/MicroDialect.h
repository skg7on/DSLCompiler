//===- MicroDialect.h - Micro dialect public header --------------*- C++
//-*-===//
//
// Public header for the Micro dialect.  Includes the necessary MLIR headers
// before the TableGen-generated dialect declaration.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_DIALECT_MICRO_MICRODIALECT_H
#define LLK_DIALECT_MICRO_MICRODIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

// TableGen-generated dialect class declaration.
#include "LLK/Dialect/Micro/MicroDialect.h.inc"

#endif // LLK_DIALECT_MICRO_MICRODIALECT_H
