//===- MicroTileInfo.h - Tile metadata read off an MLIR type -------------===//
//
// Internal to the perf library. The cost model must read shape, dtype, layout,
// memory space, and owner out of whatever a `micro` op produced or consumed,
// and those live on two different carriers: `!micro.tile` for execution values,
// and a plain shaped type for the tensors a kernel reads from and writes back
// to external memory. describeType() flattens both into one struct so no cost
// computation has to care which one it got.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_MICROTILEINFO_H
#define LLK_PERF_MICROTILEINFO_H

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"

#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

#include "LLK/Perf/TileInfo.h"

#endif // LLK_PERF_MICROTILEINFO_H
