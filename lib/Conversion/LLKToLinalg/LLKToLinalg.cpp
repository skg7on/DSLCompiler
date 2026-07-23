//===- LLKToLinalg.cpp - Lower LLK ops to Linalg + Arith + Math -----------===//
//
// Lowers llk.fused_swiglu, llk.rope, and llk.attention into linalg named ops
// + generic ops + arith + math.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToLinalg.h"
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

namespace {

//===----------------------------------------------------------------------===//
// LLKToLinalgPass
//===----------------------------------------------------------------------===//

struct LLKToLinalgPass
    : public PassWrapper<LLKToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLKToLinalgPass)

  StringRef getArgument() const override { return "llk-to-linalg"; }
  StringRef getDescription() const override {
    return "Lower LLK ops to Linalg + Arith + Math";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override;
  void lowerSwiGLU(OpBuilder &builder, mlir::llk::FusedSwiGLUOp op);
  void lowerRoPE(OpBuilder &builder, mlir::llk::RoPEOp op);
  void lowerAttention(OpBuilder &builder, mlir::llk::AttentionOp op);
};

//===----------------------------------------------------------------------===//
// SwiGLU lowering: gate = X@Wg, up = X@Wu, SiLU(gate) * up
//===----------------------------------------------------------------------===//

void LLKToLinalgPass::lowerSwiGLU(OpBuilder &builder,
                                  mlir::llk::FusedSwiGLUOp op) {
  Location loc = op.getLoc();

  Value x = op.getX();
  Value wg = op.getWg();
  Value wu = op.getWu();
  Value init = op.getInit();

  auto xType = mlir::cast<ShapedType>(x.getType());
  auto wgType = mlir::cast<ShapedType>(wg.getType());
  int64_t M = xType.getDimSize(0);
  int64_t N = wgType.getDimSize(1);

  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(N)) {
    op.emitError("SwiGLU requires static shapes");
    signalPassFailure();
    return;
  }

  llk::MathMode mode = op.getMathMode();

  auto bf16Type = builder.getBF16Type();
  auto f32Type = builder.getF32Type();
  auto gateUpType = RankedTensorType::get({M, N}, bf16Type);

  Value bf16Zero = arith::ConstantOp::create(
      builder, loc, bf16Type, builder.getFloatAttr(bf16Type, 0.0));

  // Gate projection: X @ Wg
  Value gateInit =
      tensor::EmptyOp::create(builder, loc, gateUpType, ValueRange{});
  Value gateFilled = linalg::FillOp::create(builder, loc, ValueRange{bf16Zero},
                                            ValueRange{gateInit})
                         .getResult(0);
  Value gateResult = linalg::MatmulOp::create(builder, loc, ValueRange{x, wg},
                                              ValueRange{gateFilled})
                         .getResult(0);

  // Up projection: X @ Wu
  Value upInit =
      tensor::EmptyOp::create(builder, loc, gateUpType, ValueRange{});
  Value upFilled = linalg::FillOp::create(builder, loc, ValueRange{bf16Zero},
                                          ValueRange{upInit})
                       .getResult(0);
  Value upResult = linalg::MatmulOp::create(builder, loc, ValueRange{x, wu},
                                            ValueRange{upFilled})
                       .getResult(0);

  // Elementwise: SiLU(gate) * up, f32 internals, truncate to bf16
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(2, &getContext());

  Value siluResult =
      linalg::GenericOp::create(
          builder, loc,
          /*resultTensorTypes=*/TypeRange{gateUpType},
          /*inputs=*/ValueRange{gateResult, upResult},
          /*outputs=*/ValueRange{init},
          /*indexingMaps=*/ArrayRef<AffineMap>{idMap, idMap, idMap},
          /*iteratorTypes=*/
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel},
          /*bodyBuilder=*/
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value g = args[0]; // bf16
            Value u = args[1]; // bf16

            Value gF32 = arith::ExtFOp::create(b, l, f32Type, g);
            Value uF32 = arith::ExtFOp::create(b, l, f32Type, u);

            // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
            Value neg = arith::NegFOp::create(b, l, gF32);
            Value exp = llk::createApproxExp(b, l, neg, mode);
            Value f32One = arith::ConstantOp::create(b, l, f32Type,
                                                     b.getF32FloatAttr(1.0));
            Value den = arith::AddFOp::create(b, l, f32One, exp);
            Value invDen = arith::DivFOp::create(b, l, f32One, den);
            Value silu = arith::MulFOp::create(b, l, gF32, invDen);
            Value r = arith::MulFOp::create(b, l, silu, uF32);

            Value trunc = arith::TruncFOp::create(b, l, bf16Type, r);
            linalg::YieldOp::create(b, l, trunc);
          })
          .getResult(0);

  op.replaceAllUsesWith(siluResult);
  op.erase();
}

//===----------------------------------------------------------------------===//
// RoPE lowering: cos/sin tables → even/odd split → rotate → interleave
//===----------------------------------------------------------------------===//

void LLKToLinalgPass::lowerRoPE(OpBuilder &builder, mlir::llk::RoPEOp op) {
  Location loc = op.getLoc();
  Value x = op.getX();
  Value pos = op.getPositionIds();
  Value init = op.getInit();
  double theta = op.getTheta().convertToDouble();

  auto xType = mlir::cast<RankedTensorType>(x.getType());
  if (!xType.hasStaticShape()) {
    op.emitError("RoPE requires static shapes");
    signalPassFailure();
    return;
  }
  int64_t B = xType.getDimSize(0);
  int64_t H = xType.getDimSize(1);
  int64_t L = xType.getDimSize(2);
  int64_t D = xType.getDimSize(3);
  int64_t halfD = D / 2;

  auto f32Type = builder.getF32Type();
  auto f64Type = builder.getF64Type();

  AffineMap idMap1D = AffineMap::getMultiDimIdentityMap(1, &getContext());
  AffineMap idMap2D = AffineMap::getMultiDimIdentityMap(2, &getContext());
  AffineMap idMap4D = AffineMap::getMultiDimIdentityMap(4, &getContext());

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
  AffineMap cosBroadcast = AffineMap::get(
      4, 0,
      {getAffineDimExpr(2, &getContext()), getAffineDimExpr(3, &getContext())},
      &getContext());

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
}

//===----------------------------------------------------------------------===//
// Attention lowering: Q@K^T → scale+mask → stable softmax → @V
//===----------------------------------------------------------------------===//

void LLKToLinalgPass::lowerAttention(OpBuilder &builder,
                                     mlir::llk::AttentionOp op) {
  Location loc = op.getLoc();
  Value q = op.getQ();
  Value k = op.getK();
  Value v = op.getV();
  Value init = op.getInit();
  float scale = static_cast<float>(op.getScale().convertToDouble());
  bool causal = op.getCausalMask();

  auto qType = mlir::cast<RankedTensorType>(q.getType());
  int64_t B = qType.getDimSize(0);
  int64_t H = qType.getDimSize(1);
  int64_t Lq = qType.getDimSize(2);
  int64_t D = qType.getDimSize(3);
  int64_t Lk = mlir::cast<RankedTensorType>(k.getType()).getDimSize(2);
  int64_t BH = B * H;

  auto f32Type = builder.getF32Type();

  llk::MathMode mode = op.getMathMode();

  // Collapse B,H into a single batch dimension: [B, H, ...] → [B*H, ...].
  SmallVector<ReassociationIndices> collapse3D = {{0, 1}, {2}, {3}};
  auto q3DType = RankedTensorType::get({BH, Lq, D}, f32Type);
  auto k3DType = RankedTensorType::get({BH, Lk, D}, f32Type);
  auto v3DType = RankedTensorType::get({BH, Lk, D}, f32Type);
  auto o3DType = RankedTensorType::get({BH, Lq, D}, f32Type);

  Value q3D =
      tensor::CollapseShapeOp::create(builder, loc, q3DType, q, collapse3D);
  Value k3D =
      tensor::CollapseShapeOp::create(builder, loc, k3DType, k, collapse3D);
  Value v3D =
      tensor::CollapseShapeOp::create(builder, loc, v3DType, v, collapse3D);
  Value init3D =
      tensor::CollapseShapeOp::create(builder, loc, o3DType, init, collapse3D);

  // ---- Transpose K: [BH, Lk, D] → [BH, D, Lk] using linalg.generic ----
  auto kTransposedType = RankedTensorType::get({BH, D, Lk}, f32Type);
  Value kTransposedInit =
      tensor::EmptyOp::create(builder, loc, kTransposedType, ValueRange{});
  AffineMap kMapIn = AffineMap::getMultiDimIdentityMap(3, &getContext());
  // (b, d, lk) reads from k3D at (b, lk, d)
  AffineMap kMapOut = AffineMap::get(3, 0,
                                     {getAffineDimExpr(0, &getContext()),
                                      getAffineDimExpr(2, &getContext()),
                                      getAffineDimExpr(1, &getContext())},
                                     &getContext());
  Value kTP =
      linalg::GenericOp::create(
          builder, loc, TypeRange{kTransposedType}, ValueRange{k3D},
          ValueRange{kTransposedInit}, ArrayRef<AffineMap>{kMapOut, kMapIn},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel,
                                        utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            linalg::YieldOp::create(b, l, args[0]);
          })
          .getResult(0);

  // ---- S = Q @ K^T: contraction [BH, Lq, D] × [BH, D, Lk] → [BH, Lq, Lk] ----
  auto sType = RankedTensorType::get({BH, Lq, Lk}, f32Type);
  Value sInit = tensor::EmptyOp::create(builder, loc, sType, ValueRange{});
  Value zeroVal = arith::ConstantOp::create(builder, loc, f32Type,
                                            builder.getF32FloatAttr(0.0f));
  Value sFilled = linalg::FillOp::create(builder, loc, ValueRange{zeroVal},
                                         ValueRange{sInit})
                      .getResult(0);

  // Batched matmul via linalg.generic: S[bh,q,k] += Q[bh,q,d] * K[bh,d,k]
  // 4D iteration space: (bh, q, d, k)
  AffineMap qMapBM = AffineMap::get(4, 0,
                                    {getAffineDimExpr(0, &getContext()),
                                     getAffineDimExpr(1, &getContext()),
                                     getAffineDimExpr(2, &getContext())},
                                    &getContext());
  AffineMap kMapBM = AffineMap::get(4, 0,
                                    {getAffineDimExpr(0, &getContext()),
                                     getAffineDimExpr(2, &getContext()),
                                     getAffineDimExpr(3, &getContext())},
                                    &getContext());
  AffineMap sMapBM = AffineMap::get(4, 0,
                                    {getAffineDimExpr(0, &getContext()),
                                     getAffineDimExpr(1, &getContext()),
                                     getAffineDimExpr(3, &getContext())},
                                    &getContext());

  Value scores =
      linalg::GenericOp::create(
          builder, loc, TypeRange{sType}, ValueRange{q3D, kTP},
          ValueRange{sFilled}, ArrayRef<AffineMap>{qMapBM, kMapBM, sMapBM},
          ArrayRef<utils::IteratorType>{
              utils::IteratorType::parallel, utils::IteratorType::parallel,
              utils::IteratorType::reduction, utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value mul = arith::MulFOp::create(b, l, args[0], args[1]);
            Value add = arith::AddFOp::create(b, l, mul, args[2]);
            linalg::YieldOp::create(b, l, ValueRange{add});
          })
          .getResult(0);

  // ---- Scale + causal mask over 3D tensor [BH, Lq, Lk] ----
  Value scaleVal = arith::ConstantOp::create(builder, loc, f32Type,
                                             builder.getF32FloatAttr(scale));
  Value negInfVal = arith::ConstantOp::create(
      builder, loc, f32Type,
      builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));

  Value scaledInit = tensor::EmptyOp::create(builder, loc, sType, ValueRange{});
  AffineMap idMap3D = AffineMap::getMultiDimIdentityMap(3, &getContext());
  Value scaled =
      linalg::GenericOp::create(
          builder, loc, TypeRange{sType}, ValueRange{scores},
          ValueRange{scaledInit}, ArrayRef<AffineMap>{idMap3D, idMap3D},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel,
                                        utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value s = args[0];
            Value scaledS = arith::MulFOp::create(b, l, s, scaleVal);
            if (causal) {
              Value qIdx = linalg::IndexOp::create(b, l, 1);
              Value kIdx = linalg::IndexOp::create(b, l, 2);
              Value kGtQ = arith::CmpIOp::create(
                  b, l, arith::CmpIPredicate::sgt, kIdx, qIdx);
              Value result =
                  arith::SelectOp::create(b, l, kGtQ, negInfVal, scaledS);
              linalg::YieldOp::create(b, l, ValueRange{result});
            } else {
              linalg::YieldOp::create(b, l, ValueRange{scaledS});
            }
          })
          .getResult(0);

  // ---- Stable softmax ----
  // Step a: row-wise max over Lk.  scaled: [BH, Lq, Lk] → maxVals: [BH, Lq]
  auto maxPerRowType = RankedTensorType::get({BH, Lq}, f32Type);
  Value maxInitRow =
      tensor::EmptyOp::create(builder, loc, maxPerRowType, ValueRange{});
  Value maxFilled = linalg::FillOp::create(builder, loc, ValueRange{negInfVal},
                                           ValueRange{maxInitRow})
                        .getResult(0);

  // 3D iteration space: (bh, q, k) with k reduction.
  // Input scaled: 3D → (bh, q, k); Output max: 2D → (bh, q)
  AffineMap reduce3DIn = AffineMap::get(3, 0,
                                        {getAffineDimExpr(0, &getContext()),
                                         getAffineDimExpr(1, &getContext()),
                                         getAffineDimExpr(2, &getContext())},
                                        &getContext());
  AffineMap reduceOut2D = AffineMap::get(
      3, 0,
      {getAffineDimExpr(0, &getContext()), getAffineDimExpr(1, &getContext())},
      &getContext());

  Value maxVals =
      linalg::GenericOp::create(
          builder, loc, TypeRange{maxPerRowType}, ValueRange{scaled},
          ValueRange{maxFilled}, ArrayRef<AffineMap>{reduce3DIn, reduceOut2D},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel,
                                        utils::IteratorType::reduction},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value cur = args[0];
            Value acc = args[1];
            Value m = arith::MaximumFOp::create(b, l, cur, acc);
            linalg::YieldOp::create(b, l, ValueRange{m});
          })
          .getResult(0);

  // Step b: sum of exp(s - max) over Lk for denominator.
  Value sumInit =
      tensor::EmptyOp::create(builder, loc, maxPerRowType, ValueRange{});
  Value sumFilled = linalg::FillOp::create(builder, loc, ValueRange{zeroVal},
                                           ValueRange{sumInit})
                        .getResult(0);

  Value denominator =
      linalg::GenericOp::create(
          builder, loc, TypeRange{maxPerRowType}, ValueRange{scaled, maxVals},
          ValueRange{sumFilled},
          ArrayRef<AffineMap>{reduce3DIn, reduceOut2D, reduceOut2D},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel,
                                        utils::IteratorType::reduction},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value s = args[0];
            Value m = args[1];
            Value acc = args[2];
            Value shifted = arith::SubFOp::create(b, l, s, m);
            Value expVal = llk::createApproxExp(b, l, shifted, mode);
            Value sum = arith::AddFOp::create(b, l, expVal, acc);
            linalg::YieldOp::create(b, l, ValueRange{sum});
          })
          .getResult(0);

  // Step c: softmax weights = exp(s - max) / denominator.  3D elementwise.
  auto wType = RankedTensorType::get({BH, Lq, Lk}, f32Type);
  Value wInit = tensor::EmptyOp::create(builder, loc, wType, ValueRange{});
  // scaled: 3D (bh,q,k), maxVals: 2D (bh,q), denom: 2D (bh,q), w: 3D (bh,q,k)
  Value weights =
      linalg::GenericOp::create(
          builder, loc, TypeRange{wType},
          ValueRange{scaled, maxVals, denominator}, ValueRange{wInit},
          ArrayRef<AffineMap>{reduce3DIn, reduceOut2D, reduceOut2D, reduce3DIn},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel,
                                        utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value s = args[0];
            Value m = args[1];
            Value d = args[2];
            Value shifted = arith::SubFOp::create(b, l, s, m);
            Value expVal = llk::createApproxExp(b, l, shifted, mode);
            Value softmaxVal = arith::DivFOp::create(b, l, expVal, d);
            linalg::YieldOp::create(b, l, ValueRange{softmaxVal});
          })
          .getResult(0);

  // ---- O = softmax @ V: contraction [BH, Lq, Lk] × [BH, Lk, D] → [BH, Lq, D]
  // ---- 4D iteration space: (bh, q, k, d) O[bh,q,d] += W[bh,q,k] * V[bh,k,d]
  AffineMap wMapBM2 = AffineMap::get(4, 0,
                                     {getAffineDimExpr(0, &getContext()),
                                      getAffineDimExpr(1, &getContext()),
                                      getAffineDimExpr(2, &getContext())},
                                     &getContext());
  AffineMap vMapBM2 = AffineMap::get(4, 0,
                                     {getAffineDimExpr(0, &getContext()),
                                      getAffineDimExpr(2, &getContext()),
                                      getAffineDimExpr(3, &getContext())},
                                     &getContext());
  AffineMap oMapBM2 = AffineMap::get(4, 0,
                                     {getAffineDimExpr(0, &getContext()),
                                      getAffineDimExpr(1, &getContext()),
                                      getAffineDimExpr(3, &getContext())},
                                     &getContext());

  Value o3D =
      linalg::GenericOp::create(
          builder, loc, TypeRange{o3DType}, ValueRange{weights, v3D},
          ValueRange{init3D}, ArrayRef<AffineMap>{wMapBM2, vMapBM2, oMapBM2},
          ArrayRef<utils::IteratorType>{
              utils::IteratorType::parallel, utils::IteratorType::parallel,
              utils::IteratorType::reduction, utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value mul = arith::MulFOp::create(b, l, args[0], args[1]);
            Value add = arith::AddFOp::create(b, l, mul, args[2]);
            linalg::YieldOp::create(b, l, ValueRange{add});
          })
          .getResult(0);

  // Expand back to [B, H, Lq, D].
  auto oType = RankedTensorType::get({B, H, Lq, D}, f32Type);
  SmallVector<ReassociationIndices> expandAssoc = {{0, 1}, {2}, {3}};
  Value result =
      tensor::ExpandShapeOp::create(builder, loc, oType, o3D, expandAssoc);

  op.replaceAllUsesWith(result);
  op.erase();
}

//===----------------------------------------------------------------------===//
// Pass entry point: walk all LLK ops and dispatch lowering.
//===----------------------------------------------------------------------===//

void LLKToLinalgPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(&getContext());

  // Collect all LLK ops.
  SmallVector<mlir::llk::FusedSwiGLUOp> swigluOps;
  SmallVector<mlir::llk::RoPEOp> ropeOps;
  SmallVector<mlir::llk::AttentionOp> attentionOps;
  module.walk([&](mlir::llk::FusedSwiGLUOp op) { swigluOps.push_back(op); });
  module.walk([&](mlir::llk::RoPEOp op) { ropeOps.push_back(op); });
  module.walk([&](mlir::llk::AttentionOp op) { attentionOps.push_back(op); });

  // Lower each op kind.
  for (auto op : swigluOps) {
    builder.setInsertionPoint(op);
    lowerSwiGLU(builder, op);
  }
  for (auto op : ropeOps) {
    builder.setInsertionPoint(op);
    lowerRoPE(builder, op);
  }
  for (auto op : attentionOps) {
    builder.setInsertionPoint(op);
    lowerAttention(builder, op);
  }
}

static mlir::PassRegistration<LLKToLinalgPass> llkToLinalgPassReg;

} // namespace

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createLLKToLinalgPass() {
  return std::make_unique<LLKToLinalgPass>();
}
} // namespace llk
} // namespace mlir
