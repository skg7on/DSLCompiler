//===- MicroOps.cpp - Micro dialect operations
//-----------------------------===//
//
// Implements the Micro dialect operations.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroEnums.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

// Attribute class declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"

// Type class declarations.
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"

// Op class full declarations (undefines GET_OP_CLASSES internally).
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

//===----------------------------------------------------------------------===//
// Generated operation definitions: parse, print, verify, build, etc.
// Must re-define GET_OP_CLASSES since MicroOps.h.inc undef'd it.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.cpp.inc"

using namespace mlir;
using namespace mlir::micro;

//===----------------------------------------------------------------------===//
// Custom verifier for KernelOp.
//===----------------------------------------------------------------------===//

LogicalResult KernelOp::verify() {
  // workload, if present, must be non-empty.
  if (auto workload = getWorkload())
    if (workload->empty())
      return emitOpError("workload attribute must be non-empty");

  return success();
}

//===----------------------------------------------------------------------===//
// Tile op verifiers
//===----------------------------------------------------------------------===//

static bool isSupportedVectorOp(StringRef op) {
  return llvm::StringSwitch<bool>(op)
      .Cases({"add", "sub", "mul", "div"}, true)
      .Cases({"exp", "silu", "sigmoid"}, true)
      .Cases({"max", "min", "select", "convert"}, true)
      .Default(false);
}

static bool isSupportedReduceOp(StringRef op) {
  return llvm::StringSwitch<bool>(op)
      .Cases({"sum", "max", "min", "prod"}, true)
      .Default(false);
}

LogicalResult TileViewOp::verify() {
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a tile type");

  Type sourceType = getSource().getType();
  if (!isa<ShapedType>(sourceType) && !isa<TileType>(sourceType))
    return emitOpError("source must be a tensor or tile");

  if (getShape() != resultType.getShape())
    return emitOpError("shape attribute must match result tile shape");

  if (auto layout = getLayout())
    if (resultType.getLayout() && layout != resultType.getLayout())
      return emitOpError("layout attribute must match result tile layout");

  return success();
}

LogicalResult TileAllocOp::verify() {
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a tile type");
  if (!resultType.getMemory())
    return emitOpError("tile_alloc result tile must have a memory space");
  if (resultType.getMemory().getValue() == MemorySpace::dram)
    return emitOpError("tile_alloc memory must not be dram");
  return success();
}

LogicalResult TilePartitionOp::verify() {
  auto sourceType = dyn_cast<TileType>(getSource().getType());
  if (!sourceType)
    return emitOpError("source must be a tile type");
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a tile type");

  if (getShape() != resultType.getShape())
    return emitOpError("shape attribute must match result tile shape");

  ArrayRef<int64_t> parent = sourceType.getShape();
  ArrayRef<int64_t> fragment = resultType.getShape();
  if (parent.size() != fragment.size())
    return emitOpError("fragment shape must divide parent shape");
  for (unsigned i = 0; i < parent.size(); ++i) {
    if (ShapedType::isDynamic(parent[i]) || ShapedType::isDynamic(fragment[i]))
      continue;
    if (parent[i] % fragment[i] != 0)
      return emitOpError("fragment shape must divide parent shape");
  }
  return success();
}

LogicalResult TileAsyncCopyOp::verify() {
  auto sourceType = dyn_cast<TileType>(getSource().getType());
  if (!sourceType)
    return emitOpError("source must be a tile type");
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a tile type");
  if (!isa<AsyncTokenType>(getToken().getType()))
    return emitOpError("token result must be an async token");

  if (sourceType.getMemory() &&
      sourceType.getMemory().getValue() == getDstMemory())
    return emitOpError("source and destination memory must differ");

  if (resultType.getMemory() &&
      resultType.getMemory().getValue() != getDstMemory())
    return emitOpError("result tile memory must match destination memory");

  return success();
}

LogicalResult TileStoreOp::verify() {
  auto sourceType = dyn_cast<TileType>(getSource().getType());
  if (!sourceType)
    return emitOpError("source must be a tile type");
  if (!sourceType.getMemory())
    return emitOpError("stored tile must have a memory space");
  if (sourceType.getMemory().getValue() == getDstMemory())
    return emitOpError("source and destination memory must differ");
  return success();
}

LogicalResult MmaOp::verify() {
  if (!isa<TileType>(getLhs().getType()) ||
      !isa<TileType>(getRhs().getType()) || !isa<TileType>(getAcc().getType()))
    return emitOpError("mma operands must be tile types");
  if (!isa<TileType>(getResult().getType()))
    return emitOpError("mma result must be a tile type");

  if (getShape().size() != 3)
    return emitOpError("mma shape must have exactly 3 dimensions");
  for (int64_t dim : getShape())
    if (dim <= 0)
      return emitOpError("mma shape dimensions must be positive");
  return success();
}

LogicalResult VectorOp::verify() {
  if (!isSupportedVectorOp(getOp()))
    return emitOpError("unknown vector operation");
  for (Value input : getInputs())
    if (!isa<TileType>(input.getType()))
      return emitOpError("vector operands must be tile types");
  if (!isa<TileType>(getResult().getType()))
    return emitOpError("vector result must be a tile type");
  return success();
}

LogicalResult ReduceOp::verify() {
  if (!isSupportedReduceOp(getOp()))
    return emitOpError("unknown reduce operation");
  auto inputType = dyn_cast<TileType>(getInput().getType());
  if (!inputType)
    return emitOpError("reduce input must be a tile type");
  if (!isa<TileType>(getResult().getType()))
    return emitOpError("reduce result must be a tile type");
  if (getAxis() >= inputType.getShape().size())
    return emitOpError("reduce axis must be within the input rank");
  return success();
}
