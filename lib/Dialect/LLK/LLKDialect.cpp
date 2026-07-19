//===- LLKDialect.cpp - LLK dialect implementation ------------------------===//
//
// Implements the LLK dialect: registration, attribute parsing/printing,
// and operation initialization.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/LLKEnums.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

//===----------------------------------------------------------------------===//
// Generated headers: class declarations
//===----------------------------------------------------------------------===//

// Dialect class declaration (no guard needed).
#include "LLK/Dialect/LLKDialect.h.inc"

// Attribute class declarations (guarded by GET_ATTRDEF_CLASSES).
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"

// Op class declarations with full definitions (guarded by GET_OP_CLASSES).
// The full definition is needed by addOperations<> in initialize().
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

//===----------------------------------------------------------------------===//
// Generated definitions: constructor, destructor
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/LLKDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated attribute definitions (parse/print, storage types)
// Guard must be re-defined since LLKAttributes.h.inc undef'd it.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void mlir::llk::LLKDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LLK/Dialect/LLKOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "LLK/Dialect/LLKAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Dialect overrides for type parsing/printing and constant materialization.
// These are stubs since the LLK dialect has no custom types and no
// constant materialization yet.
//===----------------------------------------------------------------------===//

mlir::Type mlir::llk::LLKDialect::parseType(
    mlir::DialectAsmParser &parser) const {
  parser.emitError(parser.getCurrentLocation(),
                   "LLK dialect has no custom types");
  return {};
}

void mlir::llk::LLKDialect::printType(mlir::Type type,
                                      mlir::DialectAsmPrinter &os) const {
  // Should never be called: the parser always fails.
  llvm_unreachable("LLK dialect has no custom types");
}

mlir::Operation *mlir::llk::LLKDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type,
    mlir::Location loc) {
  // No constant materialization for LLK dialect.
  return nullptr;
}
