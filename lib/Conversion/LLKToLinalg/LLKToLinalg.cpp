//===- LLKToLinalg.cpp - Lower LLK ops to Linalg + Arith + Math -----------===//
//
// Lowers llk.fused_swiglu, llk.rope, and llk.attention into linalg named ops
// + generic ops + arith + math.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToLinalg.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/LLKEnums.h"

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

  auto bf16Type = builder.getBF16Type();
  auto f32Type = builder.getF32Type();
  auto gateUpType = RankedTensorType::get({M, N}, bf16Type);

  Value bf16Zero = builder.create<arith::ConstantOp>(
      loc, bf16Type, builder.getFloatAttr(bf16Type, 0.0));

  // Gate projection: X @ Wg
  Value gateInit =
      builder.create<tensor::EmptyOp>(loc, gateUpType, ValueRange{});
  Value gateFilled = builder
                         .create<linalg::FillOp>(loc, ValueRange{bf16Zero},
                                                 ValueRange{gateInit})
                         .getResult(0);
  Value gateResult = builder
                         .create<linalg::MatmulOp>(loc, ValueRange{x, wg},
                                                   ValueRange{gateFilled})
                         .getResult(0);

  // Up projection: X @ Wu
  Value upInit = builder.create<tensor::EmptyOp>(loc, gateUpType, ValueRange{});
  Value upFilled =
      builder
          .create<linalg::FillOp>(loc, ValueRange{bf16Zero}, ValueRange{upInit})
          .getResult(0);
  Value upResult = builder
                       .create<linalg::MatmulOp>(loc, ValueRange{x, wu},
                                                 ValueRange{upFilled})
                       .getResult(0);

  // Elementwise: SiLU(gate) * up, f32 internals, truncate to bf16
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(2, &getContext());

  Value siluResult =
      builder
          .create<linalg::GenericOp>(
              loc,
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

                Value gF32 = b.create<arith::ExtFOp>(l, f32Type, g);
                Value uF32 = b.create<arith::ExtFOp>(l, f32Type, u);

                // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
                Value neg = b.create<arith::NegFOp>(l, gF32);
                Value exp = b.create<math::ExpOp>(l, neg);
                Value f32One = b.create<arith::ConstantOp>(
                    l, f32Type, b.getF32FloatAttr(1.0));
                Value den = b.create<arith::AddFOp>(l, f32One, exp);
                Value invDen = b.create<arith::DivFOp>(l, f32One, den);
                Value silu = b.create<arith::MulFOp>(l, gF32, invDen);
                Value r = b.create<arith::MulFOp>(l, silu, uF32);

                Value trunc = b.create<arith::TruncFOp>(l, bf16Type, r);
                b.create<linalg::YieldOp>(l, trunc);
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

  // Step 1: Precompute frequency table freqs[i] = theta^(-2i/D) as f64.
  auto freqType = RankedTensorType::get({halfD}, f64Type);
  Value freqInit = builder.create<tensor::EmptyOp>(loc, freqType, ValueRange{});
  Value freqs =
      builder
          .create<linalg::GenericOp>(
              loc, TypeRange{freqType}, ValueRange{}, ValueRange{freqInit},
              ArrayRef<AffineMap>{idMap1D},
              ArrayRef<utils::IteratorType>{utils::IteratorType::parallel},
              [&](OpBuilder &b, Location l, ValueRange args) {
                Value i = b.create<linalg::IndexOp>(l, 0);
                Value iI64 = b.create<arith::IndexCastOp>(l, b.getI64Type(), i);
                Value iF64 = b.create<arith::SIToFPOp>(l, f64Type, iI64);
                Value negTwo = b.create<arith::ConstantOp>(
                    l, f64Type, b.getF64FloatAttr(-2.0));
                Value dF64 = b.create<arith::ConstantOp>(
                    l, f64Type, b.getF64FloatAttr(static_cast<double>(D)));
                Value exponent = b.create<arith::DivFOp>(
                    l, b.create<arith::MulFOp>(l, negTwo, iF64), dF64);
                Value thetaF64 = b.create<arith::ConstantOp>(
                    l, f64Type, b.getF64FloatAttr(theta));
                Value powTheta = b.create<math::PowFOp>(l, thetaF64, exponent);
                b.create<linalg::YieldOp>(l, ValueRange{powTheta});
              })
          .getResult(0);

  // Step 2: Compute angles[p, i] = position_ids[p] * freqs[i] → [L, halfD] f64.
  auto angleType = RankedTensorType::get({L, halfD}, f64Type);
  Value angleInit =
      builder.create<tensor::EmptyOp>(loc, angleType, ValueRange{});
  Value angles =
      builder
          .create<linalg::GenericOp>(
              loc, TypeRange{angleType}, ValueRange{}, ValueRange{angleInit},
              ArrayRef<AffineMap>{idMap2D},
              ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                            utils::IteratorType::parallel},
              [&](OpBuilder &b, Location l, ValueRange args) {
                Value pIdx = b.create<linalg::IndexOp>(l, 0);
                Value iIdx = b.create<linalg::IndexOp>(l, 1);
                Value posVal =
                    b.create<tensor::ExtractOp>(l, pos, ValueRange{pIdx});
                // posVal is i64; convert to f64 for multiplication.
                Value posF64 = b.create<arith::SIToFPOp>(l, f64Type, posVal);
                Value freqVal =
                    b.create<tensor::ExtractOp>(l, freqs, ValueRange{iIdx});
                Value angle = b.create<arith::MulFOp>(l, posF64, freqVal);
                b.create<linalg::YieldOp>(l, ValueRange{angle});
              })
          .getResult(0);

  // Step 3: cos_table = cos(angles), sin_table = sin(angles) as f32.
  auto trigType = RankedTensorType::get({L, halfD}, f32Type);
  Value trigInit = builder.create<tensor::EmptyOp>(loc, trigType, ValueRange{});

  auto buildTrigTable = [&](bool isCos) -> Value {
    return builder
        .create<linalg::GenericOp>(
            loc, TypeRange{trigType}, ValueRange{angles}, ValueRange{trigInit},
            ArrayRef<AffineMap>{idMap2D, idMap2D},
            ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                          utils::IteratorType::parallel},
            [&](OpBuilder &b, Location l, ValueRange args) {
              Value aF64 = args[0];
              Value aF32 = b.create<arith::TruncFOp>(l, f32Type, aF64);
              Value trigVal = isCos
                                  ? b.create<math::CosOp>(l, aF32).getResult()
                                  : b.create<math::SinOp>(l, aF32).getResult();
              b.create<linalg::YieldOp>(l, ValueRange{trigVal});
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

  Value xEven = builder.create<tensor::ExtractSliceOp>(loc, x, evenOffsets,
                                                       evenSizes, evenStrides);
  Value xOdd = builder.create<tensor::ExtractSliceOp>(loc, x, oddOffsets,
                                                      evenSizes, evenStrides);

  // Step 5: Rotate.
  //   x_rot[..., i] = x_even[..., i] * cos[..., i] - x_odd[..., i] * sin[...,
  //   i] y_rot[..., i] = x_odd[..., i] * cos[..., i] + x_even[..., i] *
  //   sin[..., i]
  auto halfType = RankedTensorType::get({B, H, L, halfD}, f32Type);
  Value initHalf = builder.create<tensor::EmptyOp>(loc, halfType, ValueRange{});

  // cos/sin [L, halfD] broadcast to [B, H, L, halfD]: (b,h,p,i) → (p,i)
  AffineMap cosBroadcast = AffineMap::get(
      4, 0,
      {getAffineDimExpr(2, &getContext()), getAffineDimExpr(3, &getContext())},
      &getContext());

  // x_rot = x_even * cos - x_odd * sin
  Value xRot =
      builder
          .create<linalg::GenericOp>(
              loc, TypeRange{halfType},
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
                Value t1 = b.create<arith::MulFOp>(l, even, c);
                Value t2 = b.create<arith::MulFOp>(l, odd, s);
                Value result = b.create<arith::SubFOp>(l, t1, t2);
                b.create<linalg::YieldOp>(l, ValueRange{result});
              })
          .getResult(0);

  // y_rot = x_odd * cos + x_even * sin
  Value yRot =
      builder
          .create<linalg::GenericOp>(
              loc, TypeRange{halfType},
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
                Value t1 = b.create<arith::MulFOp>(l, odd, c);
                Value t2 = b.create<arith::MulFOp>(l, even, s);
                Value result = b.create<arith::AddFOp>(l, t1, t2);
                b.create<linalg::YieldOp>(l, ValueRange{result});
              })
          .getResult(0);

  // Step 6: Interleave — write x_rot to even positions, y_rot to odd positions.
  Value withEven = builder.create<tensor::InsertSliceOp>(
      loc, xRot, init, evenOffsets, evenSizes, evenStrides);
  Value finalResult = builder.create<tensor::InsertSliceOp>(
      loc, yRot, withEven, oddOffsets, evenSizes, evenStrides);

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

  // Collapse B,H into a single batch dimension: [B, H, ...] → [B*H, ...].
  SmallVector<ReassociationIndices> collapse3D = {{0, 1}, {2}, {3}};
  auto q3DType = RankedTensorType::get({BH, Lq, D}, f32Type);
  auto k3DType = RankedTensorType::get({BH, Lk, D}, f32Type);
  auto v3DType = RankedTensorType::get({BH, Lk, D}, f32Type);
  auto o3DType = RankedTensorType::get({BH, Lq, D}, f32Type);

  Value q3D =
      builder.create<tensor::CollapseShapeOp>(loc, q3DType, q, collapse3D);
  Value k3D =
      builder.create<tensor::CollapseShapeOp>(loc, k3DType, k, collapse3D);
  Value v3D =
      builder.create<tensor::CollapseShapeOp>(loc, v3DType, v, collapse3D);
  Value init3D =
      builder.create<tensor::CollapseShapeOp>(loc, o3DType, init, collapse3D);

  // ---- Transpose K: [BH, Lk, D] → [BH, D, Lk] using linalg.generic ----
  auto kTransposedType = RankedTensorType::get({BH, D, Lk}, f32Type);
  Value kTransposedInit =
      builder.create<tensor::EmptyOp>(loc, kTransposedType, ValueRange{});
  AffineMap kMapIn = AffineMap::getMultiDimIdentityMap(3, &getContext());
  // (b, d, lk) reads from k3D at (b, lk, d)
  AffineMap kMapOut = AffineMap::get(3, 0,
                                     {getAffineDimExpr(0, &getContext()),
                                      getAffineDimExpr(2, &getContext()),
                                      getAffineDimExpr(1, &getContext())},
                                     &getContext());
  Value kTP =
      builder
          .create<linalg::GenericOp>(
              loc, TypeRange{kTransposedType}, ValueRange{k3D},
              ValueRange{kTransposedInit}, ArrayRef<AffineMap>{kMapOut, kMapIn},
              ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                            utils::IteratorType::parallel,
                                            utils::IteratorType::parallel},
              [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(l, args[0]);
              })
          .getResult(0);

  // ---- S = Q @ K^T: contraction [BH, Lq, D] × [BH, D, Lk] → [BH, Lq, Lk] ----
  auto sType = RankedTensorType::get({BH, Lq, Lk}, f32Type);
  Value sInit = builder.create<tensor::EmptyOp>(loc, sType, ValueRange{});
  Value zeroVal = builder.create<arith::ConstantOp>(
      loc, f32Type, builder.getF32FloatAttr(0.0f));
  Value sFilled =
      builder
          .create<linalg::FillOp>(loc, ValueRange{zeroVal}, ValueRange{sInit})
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
      builder
          .create<linalg::GenericOp>(
              loc, TypeRange{sType}, ValueRange{q3D, kTP}, ValueRange{sFilled},
              ArrayRef<AffineMap>{qMapBM, kMapBM, sMapBM},
              ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                            utils::IteratorType::parallel,
                                            utils::IteratorType::reduction,
                                            utils::IteratorType::parallel},
              [&](OpBuilder &b, Location l, ValueRange args) {
                Value mul = b.create<arith::MulFOp>(l, args[0], args[1]);
                Value add = b.create<arith::AddFOp>(l, mul, args[2]);
                b.create<linalg::YieldOp>(l, ValueRange{add});
              })
          .getResult(0);

  // ---- Scale + causal mask over 3D tensor [BH, Lq, Lk] ----
  Value scaleVal = builder.create<arith::ConstantOp>(
      loc, f32Type, builder.getF32FloatAttr(scale));
  Value negInfVal = builder.create<arith::ConstantOp>(
      loc, f32Type,
      builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));

  Value scaledInit = builder.create<tensor::EmptyOp>(loc, sType, ValueRange{});
  AffineMap idMap3D = AffineMap::getMultiDimIdentityMap(3, &getContext());
  Value scaled =
      builder
          .create<linalg::GenericOp>(
              loc, TypeRange{sType}, ValueRange{scores}, ValueRange{scaledInit},
              ArrayRef<AffineMap>{idMap3D, idMap3D},
              ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                            utils::IteratorType::parallel,
                                            utils::IteratorType::parallel},
              [&](OpBuilder &b, Location l, ValueRange args) {
                Value s = args[0];
                Value scaledS = b.create<arith::MulFOp>(l, s, scaleVal);
                if (causal) {
                  Value kIdx = b.create<linalg::IndexOp>(l, 1);
                  Value qIdx = b.create<linalg::IndexOp>(l, 0);
                  Value kGtQ = b.create<arith::CmpIOp>(
                      l, arith::CmpIPredicate::sgt, kIdx, qIdx);
                  Value result =
                      b.create<arith::SelectOp>(l, kGtQ, negInfVal, scaledS);
                  b.create<linalg::YieldOp>(l, ValueRange{result});
                } else {
                  b.create<linalg::YieldOp>(l, ValueRange{scaledS});
                }
              })
          .getResult(0);

  // ---- Stable softmax ----
  // Step a: row-wise max over Lk.  scaled: [BH, Lq, Lk] → maxVals: [BH, Lq]
  auto maxPerRowType = RankedTensorType::get({BH, Lq}, f32Type);
  Value maxInitRow =
      builder.create<tensor::EmptyOp>(loc, maxPerRowType, ValueRange{});
  Value maxFilled = builder
                        .create<linalg::FillOp>(loc, ValueRange{negInfVal},
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
      builder
          .create<linalg::GenericOp>(
              loc, TypeRange{maxPerRowType}, ValueRange{scaled},
              ValueRange{maxFilled},
              ArrayRef<AffineMap>{reduce3DIn, reduceOut2D},
              ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                            utils::IteratorType::parallel,
                                            utils::IteratorType::reduction},
              [&](OpBuilder &b, Location l, ValueRange args) {
                Value cur = args[0];
                Value acc = args[1];
                Value m = b.create<arith::MaximumFOp>(l, cur, acc);
                b.create<linalg::YieldOp>(l, ValueRange{m});
              })
          .getResult(0);

  // Step b: sum of exp(s - max) over Lk for denominator.
  Value sumInit =
      builder.create<tensor::EmptyOp>(loc, maxPerRowType, ValueRange{});
  Value sumFilled =
      builder
          .create<linalg::FillOp>(loc, ValueRange{zeroVal}, ValueRange{sumInit})
          .getResult(0);

  Value denominator =
      builder
          .create<linalg::GenericOp>(
              loc, TypeRange{maxPerRowType}, ValueRange{scaled, maxVals},
              ValueRange{sumFilled},
              ArrayRef<AffineMap>{reduce3DIn, reduceOut2D, reduceOut2D},
              ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                            utils::IteratorType::parallel,
                                            utils::IteratorType::reduction},
              [&](OpBuilder &b, Location l, ValueRange args) {
                Value s = args[0];
                Value m = args[1];
                Value acc = args[2];
                Value shifted = b.create<arith::SubFOp>(l, s, m);
                Value expVal = b.create<math::ExpOp>(l, shifted);
                Value sum = b.create<arith::AddFOp>(l, expVal, acc);
                b.create<linalg::YieldOp>(l, ValueRange{sum});
              })
          .getResult(0);

  // Step c: softmax weights = exp(s - max) / denominator.  3D elementwise.
  auto wType = RankedTensorType::get({BH, Lq, Lk}, f32Type);
  Value wInit = builder.create<tensor::EmptyOp>(loc, wType, ValueRange{});
  // scaled: 3D (bh,q,k), maxVals: 2D (bh,q), denom: 2D (bh,q), w: 3D (bh,q,k)
  Value weights =
      builder
          .create<linalg::GenericOp>(
              loc, TypeRange{wType}, ValueRange{scaled, maxVals, denominator},
              ValueRange{wInit},
              ArrayRef<AffineMap>{reduce3DIn, reduceOut2D, reduceOut2D,
                                  reduce3DIn},
              ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                            utils::IteratorType::parallel,
                                            utils::IteratorType::parallel},
              [&](OpBuilder &b, Location l, ValueRange args) {
                Value s = args[0];
                Value m = args[1];
                Value d = args[2];
                Value shifted = b.create<arith::SubFOp>(l, s, m);
                Value expVal = b.create<math::ExpOp>(l, shifted);
                Value softmaxVal = b.create<arith::DivFOp>(l, expVal, d);
                b.create<linalg::YieldOp>(l, ValueRange{softmaxVal});
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
      builder
          .create<linalg::GenericOp>(
              loc, TypeRange{o3DType}, ValueRange{weights, v3D},
              ValueRange{init3D},
              ArrayRef<AffineMap>{wMapBM2, vMapBM2, oMapBM2},
              ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                            utils::IteratorType::parallel,
                                            utils::IteratorType::reduction,
                                            utils::IteratorType::parallel},
              [&](OpBuilder &b, Location l, ValueRange args) {
                Value mul = b.create<arith::MulFOp>(l, args[0], args[1]);
                Value add = b.create<arith::AddFOp>(l, mul, args[2]);
                b.create<linalg::YieldOp>(l, ValueRange{add});
              })
          .getResult(0);

  // Expand back to [B, H, Lq, D].
  auto oType = RankedTensorType::get({B, H, Lq, D}, f32Type);
  SmallVector<ReassociationIndices> expandAssoc = {{0, 1}, {2}, {3}};
  Value result =
      builder.create<tensor::ExpandShapeOp>(loc, oType, o3D, expandAssoc);

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
