//===- MicroDialect.cpp - Micro dialect implementation
//---------------------===//
//
// Implements the Micro dialect: registration, attribute and type
// parsing/printing, and operation initialization.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroEnums.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

//===----------------------------------------------------------------------===//
// Generated headers: class declarations
//===----------------------------------------------------------------------===//

// Dialect class declaration (no guard needed).
#include "LLK/Dialect/Micro/MicroDialect.h.inc"

// Attribute class declarations (guarded by GET_ATTRDEF_CLASSES).
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"

// Type class declarations (guarded by GET_TYPEDEF_CLASSES).
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"

// Op class declarations with full definitions (guarded by GET_OP_CLASSES).
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

//===----------------------------------------------------------------------===//
// Generated definitions: constructor, destructor
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated attribute definitions (parse/print, storage types)
// Guard must be re-defined since MicroAttributes.h.inc undef'd it.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated type definitions (storage, accessors)
// Guard must be re-defined since MicroTypes.h.inc undef'd it.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void mlir::micro::MicroDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LLK/Dialect/Micro/MicroOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "LLK/Dialect/Micro/MicroAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "LLK/Dialect/Micro/MicroTypes.cpp.inc"
      >();
}
