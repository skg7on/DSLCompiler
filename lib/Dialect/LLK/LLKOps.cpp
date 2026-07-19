//===- LLKOps.cpp - LLK dialect operations ---------------------------------===//
//
// Implements the LLK dialect operations.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/LLKEnums.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
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

using namespace mlir;
using namespace mlir::llk;

//===----------------------------------------------------------------------===//
// Custom verifier for FusedSwiGLUOp.
//===----------------------------------------------------------------------===//

LogicalResult FusedSwiGLUOp::verify() {
  auto xType = getX().getType();
  auto wgType = getWg().getType();
  auto wuType = getWu().getType();
  auto initType = getInit().getType();

  if (!xType.hasRank() || xType.getRank() != 2)
    return emitOpError("X must be a 2D tensor");
  if (!wgType.hasRank() || wgType.getRank() != 2)
    return emitOpError("Wg must be a 2D tensor");
  if (!wuType.hasRank() || wuType.getRank() != 2)
    return emitOpError("Wu must be a 2D tensor");

  auto xM = xType.getDimSize(0);
  auto xK = xType.getDimSize(1);
  auto wgK = wgType.getDimSize(0);
  auto wgN = wgType.getDimSize(1);
  auto wuK = wuType.getDimSize(0);
  auto wuN = wuType.getDimSize(1);
  auto initM = initType.getDimSize(0);
  auto initN = initType.getDimSize(1);

  if (!ShapedType::isDynamic(xM) && !ShapedType::isDynamic(initM) &&
      xM != initM)
    return emitOpError("M dimension mismatch: X=")
           << xM << " vs init=" << initM;
  if (!ShapedType::isDynamic(wgN) && !ShapedType::isDynamic(initN) &&
      wgN != initN)
    return emitOpError("N dimension mismatch: Wg=")
           << wgN << " vs init=" << initN;
  if (!ShapedType::isDynamic(xK) && !ShapedType::isDynamic(wgK) && xK != wgK)
    return emitOpError("K dimension mismatch between X=")
           << xK << " and Wg=" << wgK;
  if (!ShapedType::isDynamic(xK) && !ShapedType::isDynamic(wuK) && xK != wuK)
    return emitOpError("K dimension mismatch between X=")
           << xK << " and Wu=" << wuK;
  if (!ShapedType::isDynamic(wgN) && !ShapedType::isDynamic(wuN) &&
      wgN != wuN)
    return emitOpError("N dimension mismatch between Wg=")
           << wgN << " and Wu=" << wuN;

  return success();
}

//===----------------------------------------------------------------------===//
// DestinationStyleOpInterface: return the mutable init operands.
//===----------------------------------------------------------------------===//

MutableOperandRange FusedSwiGLUOp::getDpsInitsMutable() {
  return getInitMutable();
}

//===----------------------------------------------------------------------===//
// TilingInterface stubs.
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> FusedSwiGLUOp::getLoopIteratorTypes() {
  return {utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::reduction};
}

SmallVector<Range> FusedSwiGLUOp::getIterationDomain(OpBuilder &b) {
  Location loc = getLoc();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value mDim = b.create<tensor::DimOp>(loc, getX(), zero);
  Value nDim = b.create<tensor::DimOp>(loc, getWg(), one);
  Value kDim = b.create<tensor::DimOp>(loc, getX(), one);
  return {Range{zero, mDim, one}, Range{zero, nDim, one},
          Range{zero, kDim, one}};
}

FailureOr<TilingResult> FusedSwiGLUOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // For M1: return failure to signal no tiling support yet (M2 will implement).
  return failure();
}

LogicalResult FusedSwiGLUOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  // Output uses M,N parallel dims (indices 0,1).
  resultOffsets = {offsets[0], offsets[1]};
  resultSizes = {sizes[0], sizes[1]};
  return success();
}
