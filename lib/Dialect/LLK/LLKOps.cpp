//===- LLKOps.cpp - LLK dialect operations ---------------------------------===//
//
// Implements the LLK dialect operations.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/LLKEnums.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

// Attribute class declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"

// Op class full declarations (undefines GET_OP_CLASSES internally).
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

//===----------------------------------------------------------------------===//
// Generated operation definitions: parse, print, verify, build, etc.
// Must re-define GET_OP_CLASSES since LLKOps.h.inc undef'd it.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom verifier for FusedSwiGLUOp.
//===----------------------------------------------------------------------===//

::llvm::LogicalResult mlir::llk::FusedSwiGLUOp::verify() {
  // No additional verification beyond the auto-generated invariants
  // (type checks, attribute presence) for now.
  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// DestinationStyleOpInterface: return the mutable init operands.
//===----------------------------------------------------------------------===//

::mlir::MutableOperandRange mlir::llk::FusedSwiGLUOp::getDpsInitsMutable() {
  return getInitMutable();
}
