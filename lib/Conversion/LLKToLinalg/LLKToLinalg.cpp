//===- LLKToLinalg.cpp - Lower LLK ops to SCF + Arith + Math -------------===//
//
// M1 scalar pipeline: lowers llk.fused_swiglu into loop nests + arith/math.
//
// Decomposition:
//   1. Gate projection:  X[M,K] @ Wg[K,N]  via scf.for loops
//   2. Up projection:    X[M,K] @ Wu[K,N]  via scf.for loops
//   3. Elementwise:      SiLU(gate) * up   via scf.for loops
//
// Note: linalg ops in this MLIR 20.1.8 build have severe issues with
// operand segment sizes and 3D AffineMap creation. We use scf.for loops
// for all computation as a pragmatic M1 workaround. Full linalg lowering
// will be restored when the MLIR build is upgraded.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToLinalg.h"
#include "LLK/Dialect/LLKEnums.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Pass/Pass.h"

#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"

#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Scalar matmul: result[M,N] = sum_K( lhs[M,K] * rhs[K,N] )
// Loop order: M -> N -> K  (scalar accumulation in register)
//===----------------------------------------------------------------------===//
static Value buildScalarMatmul(OpBuilder &builder, Location loc, Value lhs,
                               Value rhs) {
  auto bf16Type = builder.getBF16Type();
  auto lhsType = lhs.getType().cast<ShapedType>();
  auto rhsType = rhs.getType().cast<ShapedType>();
  int64_t M = lhsType.getDimSize(0);
  int64_t N = rhsType.getDimSize(1);
  int64_t K = lhsType.getDimSize(1);
  auto resultType = RankedTensorType::get({M, N}, bf16Type);

  // Create index constants.  Use i32 dense constant -> extract -> index_cast.
  // Workaround: scalar arith.constant and DenseIntElementsAttr are broken
  // in this MLIR build, but parseAttribute works.
  auto idxType = builder.getIndexType();
  auto i32Type = builder.getI32Type();
  auto i32ScalarType = RankedTensorType::get({}, i32Type);
  auto makeIdx = [&](int64_t v) -> Value {
    std::string attrStr = "dense<" + std::to_string(v) + "> : tensor<i32>";
    auto parsed = mlir::parseAttribute(attrStr, builder.getContext());
    auto denseConst = builder.create<arith::ConstantOp>(
        loc, i32ScalarType, cast<TypedAttr>(parsed));
    auto i32Val =
        builder.create<tensor::ExtractOp>(loc, denseConst, ValueRange{});
    return builder.create<arith::IndexCastOp>(loc, idxType, i32Val);
  };
  Value cst0 = makeIdx(0);
  Value cst1 = makeIdx(1);
  Value cstM = makeIdx(M);
  Value cstN = makeIdx(N);
  Value cstK = makeIdx(K);
  auto bf16ScalarType = RankedTensorType::get({}, bf16Type);
  auto bf16ZeroAttr = mlir::parseAttribute("dense<0.0> : tensor<bf16>", builder.getContext());
  Value bf16Zero = builder.create<tensor::ExtractOp>(
      loc,
      builder.create<arith::ConstantOp>(
          loc, bf16ScalarType, cast<TypedAttr>(bf16ZeroAttr)),
      ValueRange{});

  Value result =
      builder.create<tensor::EmptyOp>(loc, resultType, ValueRange{});

  // for m = 0 to M-1
  auto mLoop = builder.create<scf::ForOp>(
      loc, cst0, cstM, cst1, ValueRange{result},
      [&](OpBuilder &b, Location loc, Value m, ValueRange iterArgs) {
        Value curResult = iterArgs[0];

        auto nLoop = b.create<scf::ForOp>(
            loc, cst0, cstN, cst1, ValueRange{curResult, bf16Zero},
            [&](OpBuilder &b, Location loc, Value n, ValueRange iterArgs) {
              Value curResult2 = iterArgs[0];
              Value acc = iterArgs[1];

              auto kLoop = b.create<scf::ForOp>(
                  loc, cst0, cstK, cst1, ValueRange{acc},
                  [&](OpBuilder &b, Location loc, Value k,
                      ValueRange iterArgs) {
                    Value accK = iterArgs[0];

                    Value lhsElem =
                        b.create<tensor::ExtractOp>(loc, lhs, ValueRange{m, k});
                    Value rhsElem =
                        b.create<tensor::ExtractOp>(loc, rhs, ValueRange{k, n});
                    Value mul =
                        b.create<arith::MulFOp>(loc, lhsElem, rhsElem);
                    Value add =
                        b.create<arith::AddFOp>(loc, mul, accK);

                    b.create<scf::YieldOp>(loc, ValueRange{add});
                  });

              Value finalAcc = kLoop.getResult(0);
              Value updated = b.create<tensor::InsertOp>(
                  loc, finalAcc, curResult2, ValueRange{m, n});

              b.create<scf::YieldOp>(loc,
                                      ValueRange{updated, bf16Zero});
            });

        b.create<scf::YieldOp>(loc, ValueRange{nLoop.getResult(0)});
      });

  return mLoop.getResult(0);
}

//===----------------------------------------------------------------------===//
// Scalar elementwise SiLU:  for m, n:  result[m,n] = SiLU(gate[m,n]) * up[m,n]
// Uses f32 for the SiLU computation internally.
//===----------------------------------------------------------------------===//
static Value buildScalarElementwise(OpBuilder &builder, Location loc,
                                    Value gate, Value up, Value init) {
  auto bf16Type = builder.getBF16Type();
  auto f32Type = builder.getF32Type();
  auto gateType = gate.getType().cast<ShapedType>();
  int64_t M = gateType.getDimSize(0);
  int64_t N = gateType.getDimSize(1);

  auto idxType = builder.getIndexType();
  auto i32Type = builder.getI32Type();
  auto i32ScalarType = RankedTensorType::get({}, i32Type);
  auto makeIdx = [&](int64_t v) -> Value {
    std::string attrStr = "dense<" + std::to_string(v) + "> : tensor<i32>";
    auto parsed = mlir::parseAttribute(attrStr, builder.getContext());
    auto denseConst = builder.create<arith::ConstantOp>(
        loc, i32ScalarType, cast<TypedAttr>(parsed));
    auto i32Val =
        builder.create<tensor::ExtractOp>(loc, denseConst, ValueRange{});
    return builder.create<arith::IndexCastOp>(loc, idxType, i32Val);
  };
  Value cst0 = makeIdx(0);
  Value cst1 = makeIdx(1);
  Value cstM = makeIdx(M);
  Value cstN = makeIdx(N);
  auto f32ScalarType = RankedTensorType::get({}, f32Type);
  auto f32OneAttr = mlir::parseAttribute("dense<1.0> : tensor<f32>", builder.getContext());
  Value f32One = builder.create<tensor::ExtractOp>(
      loc,
      builder.create<arith::ConstantOp>(
          loc, f32ScalarType, cast<TypedAttr>(f32OneAttr)),
      ValueRange{});

  auto mLoop = builder.create<scf::ForOp>(
      loc, cst0, cstM, cst1, ValueRange{init},
      [&](OpBuilder &b, Location loc, Value m, ValueRange iterArgs) {
        Value curResult = iterArgs[0];

        auto nLoop = b.create<scf::ForOp>(
            loc, cst0, cstN, cst1, ValueRange{curResult},
            [&](OpBuilder &b, Location loc, Value n, ValueRange iterArgs) {
              Value curResult2 = iterArgs[0];

              Value g =
                  b.create<tensor::ExtractOp>(loc, gate, ValueRange{m, n});
              Value u =
                  b.create<tensor::ExtractOp>(loc, up, ValueRange{m, n});

              // Promote to f32.
              Value gF32 = b.create<arith::ExtFOp>(loc, f32Type, g);
              Value uF32 = b.create<arith::ExtFOp>(loc, f32Type, u);

              // SiLU(g) = g / (1 + exp(-g))
              Value neg = b.create<arith::NegFOp>(loc, gF32);
              Value exp = b.create<math::ExpOp>(loc, neg);
              Value den = b.create<arith::AddFOp>(loc, f32One, exp);
              Value sigmoid = b.create<arith::DivFOp>(loc, f32One, den);
              Value silu = b.create<arith::MulFOp>(loc, gF32, sigmoid);
              Value r = b.create<arith::MulFOp>(loc, silu, uF32);
              Value cast = b.create<arith::TruncFOp>(loc, bf16Type, r);

              Value updated = b.create<tensor::InsertOp>(
                  loc, cast, curResult2, ValueRange{m, n});

              b.create<scf::YieldOp>(loc, ValueRange{updated});
            });

        b.create<scf::YieldOp>(loc, nLoop.getResults());
      });

  return mLoop.getResult(0);
}

//===----------------------------------------------------------------------===//
// LLKToLinalgPass
//===----------------------------------------------------------------------===//

struct LLKToLinalgPass
    : public PassWrapper<LLKToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLKToLinalgPass)

  StringRef getArgument() const override { return "llk-to-linalg"; }
  StringRef getDescription() const override {
    return "Lower LLK ops to SCF + Arith + Math";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<llk::FusedSwiGLUOp> opsToLower;
    module.walk(
        [&](llk::FusedSwiGLUOp op) { opsToLower.push_back(op); });

    if (opsToLower.empty())
      return;

    OpBuilder builder(&getContext());
    for (auto op : opsToLower) {
      builder.setInsertionPoint(op);
      Location loc = op.getLoc();

      Value x = op.getX();
      Value wg = op.getWg();
      Value wu = op.getWu();
      Value init = op.getInit();

      auto xType = x.getType().cast<ShapedType>();
      auto wgType = wg.getType().cast<ShapedType>();
      int64_t M = xType.getDimSize(0);
      int64_t N = wgType.getDimSize(1);

      if (ShapedType::isDynamic(M) || ShapedType::isDynamic(N)) {
        op.emitError("M1 requires static shapes");
        signalPassFailure();
        return;
      }

      // Gate projection: X @ Wg
      Value gateResult = buildScalarMatmul(builder, loc, x, wg);

      // Up projection: X @ Wu
      Value upResult = buildScalarMatmul(builder, loc, x, wu);

      // SiLU(gate) * up
      Value result =
          buildScalarElementwise(builder, loc, gateResult, upResult, init);

      op.replaceAllUsesWith(result);
      op.erase();
    }
  }
};

static mlir::PassRegistration<LLKToLinalgPass> llkToLinalgPassReg;

} // namespace

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createLLKToLinalgPass() {
  return std::make_unique<LLKToLinalgPass>();
}
} // namespace llk
} // namespace mlir
