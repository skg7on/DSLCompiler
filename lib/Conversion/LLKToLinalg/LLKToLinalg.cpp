//===- LLKToLinalg.cpp - Lower LLK ops to Linalg + Arith + Math -----------===//
//
// M1 scalar pipeline: lowers llk.fused_swiglu into linalg named ops + generic.
//
// Decomposition:
//   1. tensor.empty + linalg.fill     (accumulator init)
//   2. linalg.matmul                  (gate projection:  X @ Wg)
//   3. linalg.matmul                  (up projection:    X @ Wu)
//   4. linalg.generic                 (SiLU(gate) * up,  f32 internally,
//                                     trunc to bf16)
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

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<mlir::llk::FusedSwiGLUOp> opsToLower;
    module.walk(
        [&](mlir::llk::FusedSwiGLUOp op) { opsToLower.push_back(op); });

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

      auto xType = mlir::cast<ShapedType>(x.getType());
      auto wgType = mlir::cast<ShapedType>(wg.getType());
      int64_t M = xType.getDimSize(0);
      int64_t N = wgType.getDimSize(1);

      if (ShapedType::isDynamic(M) || ShapedType::isDynamic(N)) {
        op.emitError("M1 requires static shapes");
        signalPassFailure();
        return;
      }

      auto bf16Type = builder.getBF16Type();
      auto f32Type = builder.getF32Type();
      auto gateUpType = RankedTensorType::get({M, N}, bf16Type);

      // Zero fill value for linalg.fill.
      Value bf16Zero = builder.create<arith::ConstantOp>(
          loc, bf16Type, builder.getFloatAttr(bf16Type, 0.0));

      // -- Gate projection: X @ Wg --

      Value gateInit = builder.create<tensor::EmptyOp>(
          loc, gateUpType, ValueRange{});
      Value gateFilled = builder.create<linalg::FillOp>(
          loc, ValueRange{bf16Zero}, ValueRange{gateInit}).getResult(0);
      Value gateResult =
          builder.create<linalg::MatmulOp>(
              loc, ValueRange{x, wg}, ValueRange{gateFilled}).getResult(0);

      // -- Up projection: X @ Wu --

      Value upInit = builder.create<tensor::EmptyOp>(
          loc, gateUpType, ValueRange{});
      Value upFilled = builder.create<linalg::FillOp>(
          loc, ValueRange{bf16Zero}, ValueRange{upInit}).getResult(0);
      Value upResult =
          builder.create<linalg::MatmulOp>(
              loc, ValueRange{x, wu}, ValueRange{upFilled}).getResult(0);

      // -- Elementwise: SiLU(gate) * up, computed in f32, truncated to bf16 --

      // Affine maps: identity for all 2 inputs + 1 output (parallel loops).
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
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value g = args[0];  // bf16
                    Value u = args[1];  // bf16

                    // Promote to f32 for accurate SiLU.
                    Value gF32 = b.create<arith::ExtFOp>(loc, f32Type, g);
                    Value uF32 = b.create<arith::ExtFOp>(loc, f32Type, u);

                    // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
                    Value neg = b.create<arith::NegFOp>(loc, gF32);
                    Value exp = b.create<math::ExpOp>(loc, neg);
                    Value f32One =
                        b.create<arith::ConstantOp>(loc, f32Type,
                                                    b.getF32FloatAttr(1.0));
                    Value den = b.create<arith::AddFOp>(loc, f32One, exp);
                    Value invDen = b.create<arith::DivFOp>(loc, f32One, den);
                    Value silu = b.create<arith::MulFOp>(loc, gF32, invDen);
                    Value r = b.create<arith::MulFOp>(loc, silu, uF32);

                    // Truncate back to bf16.
                    Value trunc =
                        b.create<arith::TruncFOp>(loc, bf16Type, r);

                    b.create<linalg::YieldOp>(loc, trunc);
                  })
              .getResult(0);

      op.replaceAllUsesWith(siluResult);
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
