//===- LLKOps.cpp - LLK dialect operations
//---------------------------------===//
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
  if (!ShapedType::isDynamic(wgN) && !ShapedType::isDynamic(wuN) && wgN != wuN)
    return emitOpError("N dimension mismatch between Wg=")
           << wgN << " and Wu=" << wuN;

  return success();
}

//===----------------------------------------------------------------------===//
// DestinationStyleOpInterface: return the mutable init operands for SwiGLU.
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

FailureOr<TilingResult>
FusedSwiGLUOp::getTiledImplementation(OpBuilder &b,
                                      ArrayRef<OpFoldResult> offsets,
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

//===----------------------------------------------------------------------===//
// DestinationStyleOpInterface: return the mutable init operands for RoPE.
//===----------------------------------------------------------------------===//

MutableOperandRange RoPEOp::getDpsInitsMutable() { return getInitMutable(); }

//===----------------------------------------------------------------------===//
// Custom verifier for RoPEOp.
//===----------------------------------------------------------------------===//

LogicalResult RoPEOp::verify() {
  auto xType = mlir::cast<ShapedType>(getX().getType());
  auto posType = mlir::cast<ShapedType>(getPositionIds().getType());

  if (!xType.hasRank() || xType.getRank() != 4)
    return emitOpError("X must be a 4D tensor [B, H, L, D]");
  if (!posType.hasRank() || posType.getRank() != 1)
    return emitOpError("position_ids must be a 1D tensor [L]");

  int64_t L = xType.getDimSize(2);
  int64_t D = xType.getDimSize(3);
  if (!ShapedType::isDynamic(D) && D % 2 != 0)
    return emitOpError("D dimension must be even, got ") << D;

  // Check position_ids length matches L.
  int64_t posLen = posType.getDimSize(0);
  if (!ShapedType::isDynamic(L) && !ShapedType::isDynamic(posLen) &&
      L != posLen)
    return emitOpError("position_ids length=")
           << posLen << " does not match X dim(2)=" << L;

  return success();
}

//===----------------------------------------------------------------------===//
// DestinationStyleOpInterface: return the mutable init operands for Attention.
//===----------------------------------------------------------------------===//

MutableOperandRange AttentionOp::getDpsInitsMutable() {
  return getInitMutable();
}

//===----------------------------------------------------------------------===//
// Custom verifier for AttentionOp.
//===----------------------------------------------------------------------===//

LogicalResult AttentionOp::verify() {
  auto qType = mlir::cast<ShapedType>(getQ().getType());
  auto kType = mlir::cast<ShapedType>(getK().getType());
  auto vType = mlir::cast<ShapedType>(getV().getType());
  auto initType = mlir::cast<ShapedType>(getInit().getType());

  if (!qType.hasRank() || qType.getRank() != 4)
    return emitOpError("Q must be a 4D tensor [B, H, Lq, D]");
  if (!kType.hasRank() || kType.getRank() != 4)
    return emitOpError("K must be a 4D tensor [B, H, Lk, D]");
  if (!vType.hasRank() || vType.getRank() != 4)
    return emitOpError("V must be a 4D tensor [B, H, Lk, D]");

  int64_t qB = qType.getDimSize(0);
  int64_t qH = qType.getDimSize(1);
  int64_t qD = qType.getDimSize(3);
  int64_t qLq = qType.getDimSize(2);
  int64_t kB = kType.getDimSize(0);
  int64_t kH = kType.getDimSize(1);
  int64_t kD = kType.getDimSize(3);
  int64_t vB = vType.getDimSize(0);
  int64_t vH = vType.getDimSize(1);
  int64_t vD = vType.getDimSize(3);
  int64_t initB = initType.getDimSize(0);
  int64_t initH = initType.getDimSize(1);
  int64_t initD = initType.getDimSize(3);
  int64_t kL = kType.getDimSize(2);
  int64_t vL = vType.getDimSize(2);
  int64_t initLq = initType.getDimSize(2);

  // B and H consistency.
  if (!ShapedType::isDynamic(qB) && !ShapedType::isDynamic(kB) && qB != kB)
    return emitOpError("B dimension mismatch: Q=") << qB << " vs K=" << kB;
  if (!ShapedType::isDynamic(qB) && !ShapedType::isDynamic(vB) && qB != vB)
    return emitOpError("B dimension mismatch: Q=") << qB << " vs V=" << vB;
  if (!ShapedType::isDynamic(qB) && !ShapedType::isDynamic(initB) &&
      qB != initB)
    return emitOpError("B dimension mismatch: Q=")
           << qB << " vs init=" << initB;
  if (!ShapedType::isDynamic(qH) && !ShapedType::isDynamic(kH) && qH != kH)
    return emitOpError("H dimension mismatch: Q=") << qH << " vs K=" << kH;
  if (!ShapedType::isDynamic(qH) && !ShapedType::isDynamic(vH) && qH != vH)
    return emitOpError("H dimension mismatch: Q=") << qH << " vs V=" << vH;
  if (!ShapedType::isDynamic(qH) && !ShapedType::isDynamic(initH) &&
      qH != initH)
    return emitOpError("H dimension mismatch: Q=")
           << qH << " vs init=" << initH;
  // Lq consistency (Q vs init).
  if (!ShapedType::isDynamic(qLq) && !ShapedType::isDynamic(initLq) &&
      qLq != initLq)
    return emitOpError("Lq dimension mismatch: Q=")
           << qLq << " vs init=" << initLq;

  if (!ShapedType::isDynamic(qD) && !ShapedType::isDynamic(kD) && qD != kD)
    return emitOpError("D dimension mismatch: Q=") << qD << " vs K=" << kD;
  if (!ShapedType::isDynamic(kD) && !ShapedType::isDynamic(vD) && kD != vD)
    return emitOpError("D dimension mismatch: K=") << kD << " vs V=" << vD;
  if (!ShapedType::isDynamic(kL) && !ShapedType::isDynamic(vL) && kL != vL)
    return emitOpError("Lk dimension mismatch: K=") << kL << " vs V=" << vL;
  if (!ShapedType::isDynamic(qD) && !ShapedType::isDynamic(initD) &&
      qD != initD)
    return emitOpError("D dimension mismatch: Q=")
           << qD << " vs init=" << initD;

  return success();
}
