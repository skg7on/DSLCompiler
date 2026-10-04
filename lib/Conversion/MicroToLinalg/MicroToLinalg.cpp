//===- MicroToLinalg.cpp - Lower micro kernels to Linalg + SCF + MemRef ---===//
//
// `micro` is the canonical execution IR, and until this pass it was a sink:
// nothing lowered it to anything a backend consumes. This is the bridge that
// lets a `micro.kernel` -- hand-written, exported by `--llk-to-micro`, or bound
// from a mapping plan -- reach the same downstream pipeline the legacy path
// uses (`Linalg -> loops -> memref -> LLVM -> ORC JIT`).
//
// Two phases, because they are different kinds of change:
//
//   1. Structure. Each `micro.kernel` becomes a `func.func` with the same
//      symbol and its body inlined, so the result is an ordinary callable the
//      downstream pipeline already knows how to handle.
//   2. Types and ops. A full dialect conversion: `!micro.tile` and the tensors
//      the kernel reads become memrefs, and every micro op is rewritten to a
//      memref/Linalg/arith equivalent. The micro dialect is marked *illegal*,
//      so an op this pass cannot lower fails the conversion loudly rather than
//      being silently left behind.
//
// This is deliberately a subset. The tile types' memory space, layout, and
// owner are placement facts: they are recorded on the mapping metadata and
// dropped here, and the target owns the memory-space mapping. Structural ops
// (`micro.for`, `micro.spatial_for`, `micro.pipeline`), `micro.mma`,
// `micro.reduce`, `micro.tile_store`, and the windowed `micro.tile_view` are
// not lowered yet -- each is a later slice, and each fails loudly until then.
//
// Kernel ABI: a `micro.kernel` has no arguments and no results. Its entry
// values are the `tensor.empty` ops its body reads, which become allocations
// here, so the lowered function is self-contained rather than callable with
// caller-provided buffers. Wiring real inputs and outputs is a later slice,
// and is what the JIT harness needs.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroToLinalg.h"

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

// Micro attribute, type, and op declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

namespace mlir {
namespace llk {
namespace {

/// `!micro.tile` and the tensors a kernel reads both become memrefs of the same
/// shape and element type. The tile's memory space, layout, and owner are
/// placement facts the target owns, so they are dropped here rather than
/// invented as memref layout/memory-space attributes. An async token lowers to
/// an `index`: it carries no data, only a dependency the erased wait names.
class MicroTypeConverter : public mlir::TypeConverter {
public:
  explicit MicroTypeConverter(mlir::MLIRContext *context) : context(context) {
    // Registration order is the *reverse* of the order `convertType` tries
    // them in (the most recently added wins), so the catch-all identity goes
    // first and the specific conversions last.
    addConversion([](Type type) -> Type { return type; });
    addConversion([this](micro::AsyncTokenType) -> Type {
      return IndexType::get(this->context);
    });
    addConversion([](RankedTensorType tensor) -> Type {
      return MemRefType::get(tensor.getShape(), tensor.getElementType());
    });
    addConversion([](micro::TileType tile) -> Type {
      return MemRefType::get(tile.getShape(), tile.getElementType());
    });
  }

private:
  mlir::MLIRContext *context;
};

/// The elementwise operations this pass can express in arith, and the arity
/// each takes. A mnemonic with no arith equivalent yet -- `silu`,
/// `reciprocal`, `exp`, a dtype conversion -- is absent, and the caller fails
/// the conversion rather than emitting something wrong.
enum class ElementwiseKind { Add, Sub, Mul, Max, Min, Negate };

std::optional<ElementwiseKind> elementwiseKind(llvm::StringRef name,
                                               size_t arity) {
  if (arity == 2) {
    if (name == "add")
      return ElementwiseKind::Add;
    if (name == "sub")
      return ElementwiseKind::Sub;
    if (name == "mul")
      return ElementwiseKind::Mul;
    if (name == "max")
      return ElementwiseKind::Max;
    if (name == "min")
      return ElementwiseKind::Min;
  }
  if (arity == 1 && name == "negate")
    return ElementwiseKind::Negate;
  return std::nullopt;
}

Value buildElementwise(OpBuilder &builder, Location loc, ElementwiseKind kind,
                       ValueRange args, Type elementType) {
  const bool floating = llvm::isa<FloatType>(elementType);
  switch (kind) {
  case ElementwiseKind::Add:
    if (floating)
      return arith::AddFOp::create(builder, loc, args[0], args[1]).getResult();
    return arith::AddIOp::create(builder, loc, args[0], args[1]).getResult();
  case ElementwiseKind::Sub:
    if (floating)
      return arith::SubFOp::create(builder, loc, args[0], args[1]).getResult();
    return arith::SubIOp::create(builder, loc, args[0], args[1]).getResult();
  case ElementwiseKind::Mul:
    if (floating)
      return arith::MulFOp::create(builder, loc, args[0], args[1]).getResult();
    return arith::MulIOp::create(builder, loc, args[0], args[1]).getResult();
  case ElementwiseKind::Max:
    if (floating)
      return arith::MaxNumFOp::create(builder, loc, args[0], args[1])
          .getResult();
    return arith::MaxSIOp::create(builder, loc, args[0], args[1]).getResult();
  case ElementwiseKind::Min:
    if (floating)
      return arith::MinNumFOp::create(builder, loc, args[0], args[1])
          .getResult();
    return arith::MinSIOp::create(builder, loc, args[0], args[1]).getResult();
  case ElementwiseKind::Negate:
    if (floating)
      return arith::NegFOp::create(builder, loc, args[0]).getResult();
    return arith::SubIOp::create(
               builder, loc,
               arith::ConstantOp::create(builder, loc,
                                         builder.getZeroAttr(elementType)),
               args[0])
        .getResult();
  }
  llvm_unreachable("all elementwise kinds handled");
}

//===----------------------------------------------------------------------===//
// Op patterns
//===----------------------------------------------------------------------===//

/// A kernel entry value is a buffer: the graph model gives it no producer, and
/// the lowered function owns it.
struct TensorEmptyOpLowering : OpConversionPattern<tensor::EmptyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    if (!type)
      return rewriter.notifyMatchFailure(op, "not convertible to a memref");
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, type);
    return success();
  }
};

/// A materialized tile is an allocation.
struct TileAllocOpLowering : OpConversionPattern<micro::TileAllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::TileAllocOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    if (!type)
      return rewriter.notifyMatchFailure(op, "not convertible to a memref");
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, type);
    return success();
  }
};

/// A logical view allocates nothing: it is the source buffer, when it covers
/// the whole source. A windowed view needs a `memref.subview`, which is a later
/// slice -- it fails loudly rather than silently reading the wrong elements.
struct TileViewOpLowering : OpConversionPattern<micro::TileViewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::TileViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto sourceType = dyn_cast<MemRefType>(adaptor.getSource().getType());
    if (!resultType || !sourceType)
      return rewriter.notifyMatchFailure(op, "not convertible to a memref");
    if (sourceType.getShape() != resultType.getShape())
      return rewriter.notifyMatchFailure(
          op, "a windowed micro.tile_view needs subview lowering, which is not "
              "implemented yet");
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

/// An elementwise fragment becomes a `linalg.generic` over the destination
/// buffer the fragment materializes into.
struct VectorOpLowering : OpConversionPattern<micro::VectorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::VectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "not convertible to a memref");
    llvm::StringRef name = op.getOp();
    Type elementType = resultType.getElementType();
    std::optional<ElementwiseKind> kind =
        elementwiseKind(name, adaptor.getInputs().size());
    if (!kind)
      return rewriter.notifyMatchFailure(
          op,
          ("elementwise op '" + name + "' has no arith lowering yet").str());
    auto out = memref::AllocOp::create(rewriter, op.getLoc(), resultType);
    SmallVector<AffineMap> maps(
        adaptor.getInputs().size() + 1,
        rewriter.getMultiDimIdentityMap(resultType.getRank()));
    SmallVector<utils::IteratorType> iterators(resultType.getRank(),
                                               utils::IteratorType::parallel);
    linalg::GenericOp::create(
        rewriter, op.getLoc(), TypeRange{}, adaptor.getInputs(),
        ValueRange{out}, maps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          linalg::YieldOp::create(
              builder, loc,
              buildElementwise(builder, loc, *kind, args, elementType));
        });
    rewriter.replaceOp(op, out);
    return success();
  }
};

/// `micro.async_copy` and `micro.tile_async_copy` are movements. The lowered
/// form is synchronous (`memref.copy`): the async token and the `micro.wait`
/// that consumes it exist to model overlap in the cost model, not to change
/// what the data movement means, so the wait is erased and the token becomes a
/// dummy `index`.
template <typename CopyOp>
struct AsyncCopyOpLowering : OpConversionPattern<CopyOp> {
  using OpConversionPattern<CopyOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto destinationType = dyn_cast<MemRefType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!destinationType)
      return rewriter.notifyMatchFailure(op, "not convertible to a memref");
    auto destination =
        memref::AllocOp::create(rewriter, op.getLoc(), destinationType);
    memref::CopyOp::create(rewriter, op.getLoc(), adaptor.getSource(),
                           destination);
    auto token = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    rewriter.replaceOp(op, ValueRange{destination, token});
    return success();
  }
};

/// A layout conversion materializes: a fresh buffer of the destination shape is
/// written from the source. The two affine maps describe *physical* layouts,
/// which this pass drops along with the tile's memory space, layout and owner,
/// so the repack is a same-shape copy here. Encoding the layouts as memref
/// layout maps -- and so making the copy do the actual repack -- is the next
/// slice, and belongs with the target's memory-space mapping.
struct TransformOpLowering : OpConversionPattern<micro::TransformOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::TransformOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "not convertible to a memref");
    auto destination =
        memref::AllocOp::create(rewriter, op.getLoc(), resultType);
    memref::CopyOp::create(rewriter, op.getLoc(), adaptor.getSource(),
                           destination);
    rewriter.replaceOp(op, destination);
    return success();
  }
};

/// The wait is a scheduling fact on a movement that is already synchronous.
struct WaitOpLowering : OpConversionPattern<micro::WaitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::WaitOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// The kernel terminator becomes the function's.
struct YieldOpLowering : OpConversionPattern<micro::YieldOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::YieldOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<func::FuncOp>(op->getParentOp()))
      return rewriter.notifyMatchFailure(op,
                                         "only a kernel terminator is lowered");
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct MicroToLinalgPass
    : public PassWrapper<MicroToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MicroToLinalgPass)

  StringRef getArgument() const override { return "micro-to-linalg"; }

  StringRef getDescription() const override {
    return "Lower micro kernels to Linalg + SCF + MemRef, so a mapped or "
           "exported micro.kernel can reach the existing LLVM/JIT pipeline";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Phase 1: structure. Each kernel becomes an ordinary function.
    llvm::SmallVector<micro::KernelOp> kernels;
    module.walk([&](micro::KernelOp kernel) { kernels.push_back(kernel); });
    for (micro::KernelOp kernel : kernels) {
      IRRewriter rewriter(context);
      rewriter.setInsertionPoint(kernel);
      // The kernel itself holds the symbol, so it must not count as a
      // collision with the function that replaces it.
      std::string name = kernel.getSymName().str();
      for (unsigned suffix = 1;; ++suffix) {
        Operation *existing = module.lookupSymbol(name);
        if (!existing || existing == kernel.getOperation())
          break;
        name = kernel.getSymName().str() + "_" + std::to_string(suffix);
      }
      auto function = func::FuncOp::create(rewriter, kernel.getLoc(), name,
                                           rewriter.getFunctionType({}, {}));
      function.getBody().takeBody(kernel.getBody());
      rewriter.eraseOp(kernel);
    }

    // Phase 2: types and ops. The micro dialect is illegal, so anything this
    // pass cannot lower fails the conversion rather than being left behind.
    MicroTypeConverter converter(context);
    ConversionTarget target(*context);
    target.addIllegalDialect<micro::MicroDialect>();
    // A kernel's entry values are `tensor.empty`; they become allocations, so
    // the op is illegal even though its dialect is otherwise legal here.
    target.addIllegalOp<tensor::EmptyOp>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, memref::MemRefDialect,
                           scf::SCFDialect, tensor::TensorDialect>();

    RewritePatternSet patterns(context);
    patterns.add<TensorEmptyOpLowering, TileAllocOpLowering, TileViewOpLowering,
                 VectorOpLowering, TransformOpLowering,
                 AsyncCopyOpLowering<micro::AsyncCopyOp>,
                 AsyncCopyOpLowering<micro::TileAsyncCopyOp>, WaitOpLowering,
                 YieldOpLowering>(converter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createMicroToLinalgPass() {
  return std::make_unique<MicroToLinalgPass>();
}

} // namespace llk
} // namespace mlir
