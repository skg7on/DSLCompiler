//===- MicroOps.cpp - Micro dialect operations
//-----------------------------===//
//
// Implements the Micro dialect operations.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"

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

static bool isSupportedReduceOp(StringRef op) {
  return llvm::StringSwitch<bool>(op)
      .Cases({"sum", "max", "min", "prod", "product"}, true)
      .Default(false);
}

// Returns the arity of a vector operation, or 0 for an unknown operation.
static int vectorOpArity(StringRef op) {
  return llvm::StringSwitch<int>(op)
      .Cases({"add", "sub", "mul", "div", "max", "min", "compare"}, 2)
      .Cases({"exp", "silu", "sigmoid", "reciprocal", "convert"}, 1)
      .Case("select", 3)
      .Default(0);
}

// Verifies that an optional owner attribute is carried by and agrees with the
// result tile's owner (shared by tile_partition and tile_async_copy).
template <typename OpTy>
static LogicalResult verifyOwnerAttrMatches(OpTy op, std::optional<Owner> owner,
                                            TileType resultType) {
  if (!owner)
    return success();
  if (!resultType.getOwner())
    return op.emitOpError(
        "owner attribute requires the result tile to carry an owner");
  if (*owner != resultType.getOwner().getValue())
    return op.emitOpError("owner attribute must match result tile owner");
  return success();
}

LogicalResult TileViewOp::verify() {
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a tile type");

  Type sourceType = getSource().getType();
  unsigned sourceRank = 0;
  Type sourceElementType;
  if (auto shapedType = dyn_cast<ShapedType>(sourceType)) {
    sourceRank = shapedType.getRank();
    sourceElementType = shapedType.getElementType();
  } else if (auto sourceTile = dyn_cast<TileType>(sourceType)) {
    sourceRank = sourceTile.getShape().size();
    sourceElementType = sourceTile.getElementType();
  } else {
    return emitOpError("source must be a tensor or tile");
  }

  // The view must preserve the element type.
  if (sourceElementType != resultType.getElementType())
    return emitOpError("source and result element types must match");

  if (getShape() != resultType.getShape())
    return emitOpError("shape attribute must match result tile shape");

  // Offsets, when present, must match the source rank.
  if (!getOffsets().empty() && getOffsets().size() != sourceRank)
    return emitOpError("offset count must match the source rank");

  // A layout on the op must be carried by the result tile and agree with it.
  if (auto layout = getLayout()) {
    if (!resultType.getLayout())
      return emitOpError(
          "layout attribute requires the result tile to carry a layout");
    if (layout != resultType.getLayout())
      return emitOpError("layout attribute must match result tile layout");
  }

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

  // The partition must preserve the element type.
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("source and result element types must match");

  if (getShape() != resultType.getShape())
    return emitOpError("shape attribute must match result tile shape");

  if (failed(verifyOwnerAttrMatches(*this, getOwner(), resultType)))
    return failure();

  // Fragment shape must divide the parent shape, unless a tail/mask is set.
  bool hasTail = getTail().value_or(false);
  ArrayRef<int64_t> parent = sourceType.getShape();
  ArrayRef<int64_t> fragment = resultType.getShape();
  if (parent.size() != fragment.size())
    return emitOpError("fragment shape must divide parent shape");
  if (!hasTail) {
    for (unsigned i = 0; i < parent.size(); ++i) {
      if (ShapedType::isDynamic(parent[i]) ||
          ShapedType::isDynamic(fragment[i]))
        continue;
      if (parent[i] % fragment[i] != 0)
        return emitOpError("fragment shape must divide parent shape");
    }
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

  // The result must be a materialized tile in the destination memory.
  if (!resultType.getMemory())
    return emitOpError("result tile must have a memory space");
  if (resultType.getMemory().getValue() != getDstMemory())
    return emitOpError("result tile memory must match destination memory");

  if (sourceType.getMemory() &&
      sourceType.getMemory().getValue() == getDstMemory())
    return emitOpError("source and destination memory must differ");

  // The copy must preserve shape and element type.
  if (resultType.getShape() != sourceType.getShape())
    return emitOpError("result tile shape must match source tile shape");
  if (resultType.getElementType() != sourceType.getElementType())
    return emitOpError("result tile element type must match source tile "
                       "element type");

  if (failed(verifyOwnerAttrMatches(*this, getOwner(), resultType)))
    return failure();

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
  auto lhsType = dyn_cast<TileType>(getLhs().getType());
  auto rhsType = dyn_cast<TileType>(getRhs().getType());
  auto accType = dyn_cast<TileType>(getAcc().getType());
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!lhsType || !rhsType || !accType || !resultType)
    return emitOpError("mma operands and result must be tile types");

  // shape is interpreted as [M, N, K].
  if (getShape().size() != 3)
    return emitOpError("mma shape must have exactly 3 dimensions");
  int64_t m = getShape()[0], n = getShape()[1], k = getShape()[2];
  for (int64_t dim : getShape())
    if (dim <= 0)
      return emitOpError("mma shape dimensions must be positive");

  auto dimMatches = [](int64_t expected, int64_t actual) {
    return ShapedType::isDynamic(expected) || ShapedType::isDynamic(actual) ||
           expected == actual;
  };

  // Geometric compatibility: lhs=[M,K], rhs=[K,N], acc=[M,N].
  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  ArrayRef<int64_t> accShape = accType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  if (lhsShape.size() != 2 || rhsShape.size() != 2 || accShape.size() != 2 ||
      resultShape.size() != 2)
    return emitOpError("mma operands and result must be rank-2 tiles");
  if (!dimMatches(m, lhsShape[0]) || !dimMatches(k, lhsShape[1]))
    return emitOpError("lhs tile shape must be [M, K]");
  if (!dimMatches(k, rhsShape[0]) || !dimMatches(n, rhsShape[1]))
    return emitOpError("rhs tile shape must be [K, N]");
  if (!dimMatches(m, accShape[0]) || !dimMatches(n, accShape[1]))
    return emitOpError("accumulator tile shape must be [M, N]");
  if (!dimMatches(m, resultShape[0]) || !dimMatches(n, resultShape[1]))
    return emitOpError("result tile shape must be [M, N]");

  // Dtype compatibility.
  if (dtypeOfElementType(lhsType.getElementType()) != getInput() ||
      dtypeOfElementType(rhsType.getElementType()) != getInput())
    return emitOpError("input dtype must match operand element types");
  if (dtypeOfElementType(accType.getElementType()) != getAccumulator())
    return emitOpError("accumulator dtype must match accumulator element type");
  return success();
}

LogicalResult VectorOp::verify() {
  int arity = vectorOpArity(getOp());
  if (arity == 0)
    return emitOpError("unknown vector operation");

  if (static_cast<int>(getInputs().size()) != arity)
    return emitOpError("vector operation has wrong number of operands");

  SmallVector<TileType> inputTypes;
  for (Value input : getInputs()) {
    auto inputType = dyn_cast<TileType>(input.getType());
    if (!inputType)
      return emitOpError("vector operands must be tile types");
    inputTypes.push_back(inputType);
  }
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("vector result must be a tile type");

  // Elementwise data operands (all but select's leading mask/condition) must
  // match the result shape. Element types must match except for convert and
  // compare, which legitimately change element type.
  bool elementTypeMustMatch = (getOp() != "convert" && getOp() != "compare");
  int dataStart = (getOp() == "select") ? 1 : 0;
  for (unsigned i = dataStart; i < inputTypes.size(); ++i) {
    if (inputTypes[i].getShape() != resultType.getShape())
      return emitOpError("vector operand shape must match result shape");
    if (elementTypeMustMatch &&
        inputTypes[i].getElementType() != resultType.getElementType())
      return emitOpError("vector operand element type must match result "
                         "element type");
  }
  return success();
}

LogicalResult ReduceOp::verify() {
  if (!isSupportedReduceOp(getOp()))
    return emitOpError("unknown reduce operation");
  auto inputType = dyn_cast<TileType>(getInput().getType());
  if (!inputType)
    return emitOpError("reduce input must be a tile type");
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("reduce result must be a tile type");

  uint64_t axis = getAxis();
  if (axis >= inputType.getShape().size())
    return emitOpError("reduce axis must be within the input rank");

  // Result shape must be the input shape with the reduced axis removed.
  SmallVector<int64_t> expectedResultShape;
  for (unsigned i = 0; i < inputType.getShape().size(); ++i)
    if (i != axis)
      expectedResultShape.push_back(inputType.getShape()[i]);
  if (resultType.getShape() != ArrayRef<int64_t>(expectedResultShape))
    return emitOpError("reduce result shape must be the input shape with the "
                       "reduced axis removed");
  if (resultType.getElementType() != inputType.getElementType())
    return emitOpError("reduce result element type must match input element "
                       "type");
  return success();
}
