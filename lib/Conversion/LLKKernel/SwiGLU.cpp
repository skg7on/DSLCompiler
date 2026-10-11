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

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
namespace {
LogicalResult lowerSwiGLU(OpBuilder &builder, mlir::llk::FusedSwiGLUOp op) {
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
    return failure();
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
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(2, builder.getContext());

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
  return success();
}

class SwiGLULowering final : public llk::LLKKernelLowering {
public:
  StringRef getOpName() const override { return "llk.fused_swiglu"; }
  LogicalResult inferContract(Operation *root,
                              llk::KernelContract &c) const override {
    auto op = cast<llk::FusedSwiGLUOp>(root);
    c.workload = "fused_swiglu";
    return llk::inferContractionContract(root, op.getX(), op.getWg(),
                                         op.getResult(),
                                         op.getAccumulatorType(), c);
  }
  bool hasLinalgLowering() const override { return true; }
  LogicalResult lowerToLinalg(Operation *root, OpBuilder &b) const override {
    return lowerSwiGLU(b, cast<llk::FusedSwiGLUOp>(root));
  }
};
} // namespace
namespace mlir::llk {
void registerSwiGLULowering(LLKKernelRegistry &r) { r.add<SwiGLULowering>(); }
} // namespace mlir::llk
