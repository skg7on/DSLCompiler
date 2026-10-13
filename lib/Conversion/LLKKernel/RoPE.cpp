#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/LLKEnums.h"
#include "LLK/Transforms/Common/MathApproximation.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include <cmath>
#include <limits>

// Generated attribute and op class declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

using namespace mlir;

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"
namespace {
LogicalResult lowerRoPE(OpBuilder &builder, mlir::llk::RoPEOp op) {
  Location loc = op.getLoc();
  Value x = op.getX();
  Value pos = op.getPositionIds();
  Value init = op.getInit();
  double theta = op.getTheta().convertToDouble();

  auto xType = mlir::cast<RankedTensorType>(x.getType());
  if (!xType.hasStaticShape()) {
    op.emitError("RoPE requires static shapes");
    return failure();
  }
  int64_t B = xType.getDimSize(0);
  int64_t H = xType.getDimSize(1);
  int64_t L = xType.getDimSize(2);
  int64_t D = xType.getDimSize(3);
  int64_t halfD = D / 2;

  auto f32Type = builder.getF32Type();
  auto f64Type = builder.getF64Type();

  AffineMap idMap1D =
      AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  AffineMap idMap2D =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineMap idMap4D =
      AffineMap::getMultiDimIdentityMap(4, builder.getContext());

  llk::MathMode mode = op.getMathMode();

  // Step 1: Precompute frequency table freqs[i] = theta^(-2i/D) as f64.
  auto freqType = RankedTensorType::get({halfD}, f64Type);
  Value freqInit =
      tensor::EmptyOp::create(builder, loc, freqType, ValueRange{});
  Value freqs =
      linalg::GenericOp::create(
          builder, loc, TypeRange{freqType}, ValueRange{}, ValueRange{freqInit},
          ArrayRef<AffineMap>{idMap1D},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value i = linalg::IndexOp::create(b, l, 0);
            Value iI64 = arith::IndexCastOp::create(b, l, b.getI64Type(), i);
            Value iF64 = arith::SIToFPOp::create(b, l, f64Type, iI64);
            Value negTwo = arith::ConstantOp::create(b, l, f64Type,
                                                     b.getF64FloatAttr(-2.0));
            Value dF64 = arith::ConstantOp::create(
                b, l, f64Type, b.getF64FloatAttr(static_cast<double>(D)));
            Value exponent = arith::DivFOp::create(
                b, l, arith::MulFOp::create(b, l, negTwo, iF64), dF64);
            Value thetaF64 = arith::ConstantOp::create(
                b, l, f64Type, b.getF64FloatAttr(theta));
            Value powTheta = math::PowFOp::create(b, l, thetaF64, exponent);
            linalg::YieldOp::create(b, l, ValueRange{powTheta});
          })
          .getResult(0);

  // Step 2: Compute angles[p, i] = position_ids[p] * freqs[i] → [L, halfD] f64.
  auto angleType = RankedTensorType::get({L, halfD}, f64Type);
  Value angleInit =
      tensor::EmptyOp::create(builder, loc, angleType, ValueRange{});
  Value angles =
      linalg::GenericOp::create(
          builder, loc, TypeRange{angleType}, ValueRange{},
          ValueRange{angleInit}, ArrayRef<AffineMap>{idMap2D},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value pIdx = linalg::IndexOp::create(b, l, 0);
            Value iIdx = linalg::IndexOp::create(b, l, 1);
            Value posVal =
                tensor::ExtractOp::create(b, l, pos, ValueRange{pIdx});
            // posVal is i64; convert to f64 for multiplication.
            Value posF64 = arith::SIToFPOp::create(b, l, f64Type, posVal);
            Value freqVal =
                tensor::ExtractOp::create(b, l, freqs, ValueRange{iIdx});
            Value angle = arith::MulFOp::create(b, l, posF64, freqVal);
            linalg::YieldOp::create(b, l, ValueRange{angle});
          })
          .getResult(0);

  // Step 3: cos_table = cos(angles), sin_table = sin(angles) as f32.
  auto trigType = RankedTensorType::get({L, halfD}, f32Type);
  Value trigInit =
      tensor::EmptyOp::create(builder, loc, trigType, ValueRange{});

  auto buildTrigTable = [&](bool isCos) -> Value {
    return linalg::GenericOp::create(
               builder, loc, TypeRange{trigType}, ValueRange{angles},
               ValueRange{trigInit}, ArrayRef<AffineMap>{idMap2D, idMap2D},
               ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                             utils::IteratorType::parallel},
               [&](OpBuilder &b, Location l, ValueRange args) {
                 Value aF64 = args[0];
                 Value aF32 = arith::TruncFOp::create(b, l, f32Type, aF64);
                 Value trigVal = isCos ? llk::createApproxCos(b, l, aF32, mode)
                                       : llk::createApproxSin(b, l, aF32, mode);
                 linalg::YieldOp::create(b, l, ValueRange{trigVal});
               })
        .getResult(0);
  };

  Value cosTable = buildTrigTable(/*isCos=*/true);
  Value sinTable = buildTrigTable(/*isCos=*/false);

  // Step 4: Split X into even/odd via strided slices.
  SmallVector<OpFoldResult> evenOffsets = {
      builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0),
      builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> evenSizes = {
      builder.getIndexAttr(B), builder.getIndexAttr(H), builder.getIndexAttr(L),
      builder.getIndexAttr(halfD)};
  SmallVector<OpFoldResult> evenStrides = {
      builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1),
      builder.getIndexAttr(2)};
  SmallVector<OpFoldResult> oddOffsets = {
      builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0),
      builder.getIndexAttr(1)};

  Value xEven = tensor::ExtractSliceOp::create(builder, loc, x, evenOffsets,
                                               evenSizes, evenStrides);
  Value xOdd = tensor::ExtractSliceOp::create(builder, loc, x, oddOffsets,
                                              evenSizes, evenStrides);

  // Step 5: Rotate.
  //   x_rot[..., i] = x_even[..., i] * cos[..., i] - x_odd[..., i] * sin[...,
  //   i] y_rot[..., i] = x_odd[..., i] * cos[..., i] + x_even[..., i] *
  //   sin[..., i]
  auto halfType = RankedTensorType::get({B, H, L, halfD}, f32Type);
  Value initHalf =
      tensor::EmptyOp::create(builder, loc, halfType, ValueRange{});

  // cos/sin [L, halfD] broadcast to [B, H, L, halfD]: (b,h,p,i) → (p,i)
  AffineMap cosBroadcast =
      AffineMap::get(4, 0,
                     {getAffineDimExpr(2, builder.getContext()),
                      getAffineDimExpr(3, builder.getContext())},
                     builder.getContext());

  // x_rot = x_even * cos - x_odd * sin
  Value xRot =
      linalg::GenericOp::create(
          builder, loc, TypeRange{halfType},
          ValueRange{xEven, xOdd, cosTable, sinTable}, ValueRange{initHalf},
          ArrayRef<AffineMap>{idMap4D, idMap4D, cosBroadcast, cosBroadcast,
                              idMap4D},
          ArrayRef<utils::IteratorType>{
              utils::IteratorType::parallel, utils::IteratorType::parallel,
              utils::IteratorType::parallel, utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value even = args[0];
            Value odd = args[1];
            Value c = args[2];
            Value s = args[3];
            Value t1 = arith::MulFOp::create(b, l, even, c);
            Value t2 = arith::MulFOp::create(b, l, odd, s);
            Value result = arith::SubFOp::create(b, l, t1, t2);
            linalg::YieldOp::create(b, l, ValueRange{result});
          })
          .getResult(0);

  // y_rot = x_odd * cos + x_even * sin
  Value yRot =
      linalg::GenericOp::create(
          builder, loc, TypeRange{halfType},
          ValueRange{xEven, xOdd, cosTable, sinTable}, ValueRange{initHalf},
          ArrayRef<AffineMap>{idMap4D, idMap4D, cosBroadcast, cosBroadcast,
                              idMap4D},
          ArrayRef<utils::IteratorType>{
              utils::IteratorType::parallel, utils::IteratorType::parallel,
              utils::IteratorType::parallel, utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value even = args[0];
            Value odd = args[1];
            Value c = args[2];
            Value s = args[3];
            Value t1 = arith::MulFOp::create(b, l, odd, c);
            Value t2 = arith::MulFOp::create(b, l, even, s);
            Value result = arith::AddFOp::create(b, l, t1, t2);
            linalg::YieldOp::create(b, l, ValueRange{result});
          })
          .getResult(0);

  // Step 6: Interleave — write x_rot to even positions, y_rot to odd positions.
  Value withEven = tensor::InsertSliceOp::create(
      builder, loc, xRot, init, evenOffsets, evenSizes, evenStrides);
  Value finalResult = tensor::InsertSliceOp::create(
      builder, loc, yRot, withEven, oddOffsets, evenSizes, evenStrides);

  op.replaceAllUsesWith(finalResult);
  op.erase();
  return success();
}

class RoPELowering final : public llk::LLKKernelLowering {
public:
  StringRef getOpName() const override { return "llk.rope"; }
  LogicalResult inferContract(Operation *root,
                              llk::KernelContract &c) const override {
    auto op = cast<llk::RoPEOp>(root);
    auto type = dyn_cast<RankedTensorType>(op.getX().getType());
    if (!type || type.getRank() != 4)
      return failure();
    c.workload = "rope";
    c.M = type.getDimSize(0);
    c.N = type.getDimSize(3);
    c.K = type.getDimSize(3);
    c.inputElemType = type.getElementType();
    c.outputElemType = type.getElementType();
    c.accumulatorElemType = type.getElementType();
    c.mBucket = type.hasStaticShape() ? llk::classifyM(c.M) : -1;
    return success();
  }
  bool hasLinalgLowering() const override { return true; }
  LogicalResult lowerToLinalg(Operation *root, OpBuilder &b) const override {
    return lowerRoPE(b, cast<llk::RoPEOp>(root));
  }
};
} // namespace
namespace mlir::llk {
void registerRoPELowering(LLKKernelRegistry &r) { r.add<RoPELowering>(); }
} // namespace mlir::llk
