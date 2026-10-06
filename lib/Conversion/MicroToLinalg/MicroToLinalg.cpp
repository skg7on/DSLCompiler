//===- MicroToLinalg.cpp - Lower micro kernels to Linalg over tensors -----===//
//
// `micro` is the canonical execution IR, and until this pass it was a sink:
// nothing lowered it to anything a backend consumes. This is the bridge that
// lets a `micro.kernel` -- hand-written, exported by `--llk-to-micro`, or bound
// from a mapping plan -- reach the same downstream pipeline the legacy path
// uses (`Linalg -> tiling -> vector -> bufferization -> LLVM -> ORC JIT`).
//
// Two phases, because they are different kinds of change:
//
//   1. Structure. Each `micro.kernel` becomes a `func.func` with the same
//      symbol and its body inlined, so the result is an ordinary callable the
//      downstream pipeline already knows how to handle.
//   2. Types and ops. A full dialect conversion: `!micro.tile` becomes a ranked
//      tensor, and every micro op is rewritten to a Linalg/arith equivalent.
//      The micro dialect is marked *illegal*, so an op this pass cannot lower
//      fails the conversion loudly rather than being silently left behind.
//
// The output is deliberately *tensor*-based, not bufferized. `llk-compile`
// bufferizes tensors itself, and feeding it already-bufferized memrefs leaves
// the IR in a mixed state it cannot finish lowering -- a surviving unrealized
// `!llvm.array<4 x vector<8xf32>>` cast that LLVM translation rejects. Staying
// in tensor land hands the pipeline a `linalg.generic` over tensors, which is
// exactly what it tiles, vectorizes, and buffers on its own.
//
// This is deliberately a subset. The tile types' memory space, layout, and
// owner are placement facts: they are recorded on the mapping metadata and
// dropped here, and the target owns the memory-space mapping. Structural ops
// (`micro.for`, `micro.spatial_for`, `micro.pipeline`), `micro.mma`,
// `micro.reduce`, `micro.tile_store`, and the windowed `micro.tile_view` are
// not lowered yet -- each is a later slice, and each fails loudly until then.
//
// Kernel ABI: a `micro.kernel` has no arguments and no results. Its entry
// values are the `tensor.empty` ops its body reads, and its result values are
// unused, so the lowered function computes dead work. Binding real inputs and
// outputs is a later slice, and is what the JIT harness needs to observe a
// result.
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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
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

/// `!micro.tile` becomes a ranked tensor of the same shape and element type.
/// The tile's memory space, layout, and owner are placement facts the target
/// owns and are dropped here. An async token lowers to an `index`: it carries
/// no data, only a dependency the erased wait names.
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
    addConversion([](micro::TileType tile) -> Type {
      return RankedTensorType::get(tile.getShape(), tile.getElementType());
    });
  }

private:
  mlir::MLIRContext *context;
};

/// A pure movement, and a layout conversion once the layouts themselves are
/// dropped, both lower to a `linalg.generic` copy: a fresh tensor of the same
/// shape carrying the same elements.
Value buildTensorCopy(OpBuilder &builder, Location loc, Value source,
                      RankedTensorType resultType) {
  auto empty = tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                       resultType.getElementType());
  SmallVector<AffineMap> maps{
      builder.getMultiDimIdentityMap(resultType.getRank()),
      builder.getMultiDimIdentityMap(resultType.getRank())};
  SmallVector<utils::IteratorType> iterators(resultType.getRank(),
                                             utils::IteratorType::parallel);
  auto copy = linalg::GenericOp::create(
      builder, loc, TypeRange{resultType}, ValueRange{source},
      ValueRange{empty}, maps, iterators,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        linalg::YieldOp::create(nested, nestedLoc, args[0]);
      });
  return copy.getResult(0);
}

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

/// A materialized tile is a value with storage: in tensor land, that is an
/// `tensor.empty` of the same shape, which bufferization turns into an
/// allocation.
struct TileAllocOpLowering : OpConversionPattern<micro::TileAllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::TileAllocOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!type)
      return rewriter.notifyMatchFailure(op, "not convertible to a tensor");
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, type.getShape(),
                                                 type.getElementType());
    return success();
  }
};

/// A logical view allocates nothing: it is the source value, when it covers the
/// whole source. Covering the whole source means both an equal shape *and* no
/// window: an offset that is not provably a constant zero starts the view
/// somewhere else, so replacing it with the source would silently read the
/// wrong elements. A windowed view needs `tensor.extract_slice`, which is a
/// later slice -- it fails loudly rather than quietly dropping the offsets.
struct TileViewOpLowering : OpConversionPattern<micro::TileViewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::TileViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    auto sourceType = dyn_cast<RankedTensorType>(adaptor.getSource().getType());
    if (!resultType || !sourceType)
      return rewriter.notifyMatchFailure(op, "not convertible to a tensor");
    if (sourceType.getShape() != resultType.getShape())
      return rewriter.notifyMatchFailure(
          op, "a windowed micro.tile_view needs slice lowering, which is not "
              "implemented yet");
    // Equal shapes are not enough: the view must also start at the origin.
    // Identity is only provable when every supplied offset is a constant zero.
    // An offset the pattern cannot fold -- and a nonzero constant -- cannot
    // prove that, so it fails the match rather than being assumed zero.
    for (Value offset : adaptor.getOffsets()) {
      std::optional<int64_t> constant = getConstantIntValue(offset);
      if (!constant || *constant != 0)
        return rewriter.notifyMatchFailure(
            op, "a micro.tile_view with a nonzero or dynamic offset needs "
                "slice lowering, which is not implemented yet");
    }
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

/// An elementwise fragment becomes a `linalg.generic` producing a tensor.
struct VectorOpLowering : OpConversionPattern<micro::VectorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::VectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "not convertible to a tensor");
    llvm::StringRef name = op.getOp();
    Type elementType = resultType.getElementType();
    std::optional<ElementwiseKind> kind =
        elementwiseKind(name, adaptor.getInputs().size());
    if (!kind)
      return rewriter.notifyMatchFailure(
          op,
          ("elementwise op '" + name + "' has no arith lowering yet").str());
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(),
                                         resultType.getShape(), elementType);
    SmallVector<AffineMap> maps(
        adaptor.getInputs().size() + 1,
        rewriter.getMultiDimIdentityMap(resultType.getRank()));
    SmallVector<utils::IteratorType> iterators(resultType.getRank(),
                                               utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, op.getLoc(), TypeRange{resultType}, adaptor.getInputs(),
        ValueRange{empty}, maps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          linalg::YieldOp::create(
              builder, loc,
              buildElementwise(builder, loc, *kind, args, elementType));
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

/// A layout conversion materializes as a copy. The two affine maps describe
/// *physical* layouts, which this pass drops along with the tile's memory
/// space, layout and owner, so the repack is a same-shape copy here. Encoding
/// the layouts as tensor/memref layout maps -- and so making the copy do the
/// actual repack -- is the next slice, and belongs with the target's
/// memory-space mapping.
struct TransformOpLowering : OpConversionPattern<micro::TransformOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::TransformOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "not convertible to a tensor");
    rewriter.replaceOp(op, buildTensorCopy(rewriter, op.getLoc(),
                                           adaptor.getSource(), resultType));
    return success();
  }
};

/// `micro.async_copy` and `micro.tile_async_copy` are movements. The lowered
/// form is a synchronous copy: the async token and the `micro.wait` that
/// consumes it exist to model overlap in the cost model, not to change what the
/// data movement means, so the wait is erased and the token becomes a dummy
/// `index`.
template <typename CopyOp>
struct AsyncCopyOpLowering : OpConversionPattern<CopyOp> {
  using OpConversionPattern<CopyOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "not convertible to a tensor");
    Value copied =
        buildTensorCopy(rewriter, op.getLoc(), adaptor.getSource(), resultType);
    auto token = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    rewriter.replaceOp(op, ValueRange{copied, token});
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

/// The kernel terminator becomes the function's, carrying the values the
/// kernel yields -- the result a caller observes.
struct YieldOpLowering : OpConversionPattern<micro::YieldOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<func::FuncOp>(op->getParentOp()))
      return rewriter.notifyMatchFailure(op,
                                         "only a kernel terminator is lowered");
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getResults());
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
    return "Lower micro kernels to Linalg over tensors, so a mapped or "
           "exported "
           "micro.kernel can reach the existing LLVM/JIT pipeline";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    MicroTypeConverter converter(context);

    // Phase 1: structure and the kernel ABI. Each kernel becomes an ordinary
    // function whose signature is the one the kernel declared: the entry block
    // arguments are its inputs and the yielded values are its results. The
    // signature is read from the kernel, never inferred from its body, so an
    // internal `tensor.empty` can never be mistaken for a caller's buffer.
    llvm::SmallVector<micro::KernelOp> kernels;
    module.walk([&](micro::KernelOp kernel) { kernels.push_back(kernel); });
    for (micro::KernelOp kernel : kernels) {
      IRRewriter rewriter(context);

      // The kernel's contract is explicit. A kernel that declares no signature
      // has no ABI to lower, and inferring one from the entry tensors is
      // exactly the ambiguity this interface removed -- so it is refused with
      // a diagnostic rather than guessed at.
      FunctionType signature = kernel.getKernelFunctionType();
      if (!signature) {
        kernel.emitError(
            "kernel has no explicit signature: declare its inputs and results "
            "(`micro.kernel @name(%arg: type, ...) -> type`) before lowering "
            "it "
            "for execution");
        signalPassFailure();
        return;
      }

      llvm::SmallVector<Type> argumentTypes;
      for (Type input : signature.getInputs())
        argumentTypes.push_back(converter.convertType(input));
      llvm::SmallVector<Type> resultTypes;
      for (Type result : signature.getResults())
        resultTypes.push_back(converter.convertType(result));

      // A `micro.tile_store` is the pre-contract way of naming the output. When
      // the kernel declares results, the yield supersedes the store and it is
      // dropped rather than lowered a second time. A store with no declared
      // result has nowhere to write -- dropping it would silently discard the
      // kernel's output, so it is refused rather than guessed at.
      llvm::SmallVector<micro::TileStoreOp, 2> stores;
      kernel.getBody().walk(
          [&](micro::TileStoreOp store) { stores.push_back(store); });
      if (!stores.empty() && resultTypes.empty()) {
        kernel.emitError(
            "kernel writes through micro.tile_store but declares no result; "
            "lowering a tile store to a caller-owned output is not implemented "
            "yet");
        signalPassFailure();
        return;
      }

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
      auto function = func::FuncOp::create(
          rewriter, kernel.getLoc(), name,
          rewriter.getFunctionType(argumentTypes, resultTypes));

      // The entry block already carries the declared inputs, so moving the
      // region wholesale turns them into the function's parameters. Nothing is
      // inferred from `tensor.empty`: an internal allocation stays internal.
      for (micro::TileStoreOp store : stores)
        store.erase();

      function.getBody().takeBody(kernel.getBody());
      rewriter.eraseOp(kernel);
    }

    // Phase 2: types and ops. The micro dialect is illegal, so anything this
    // pass cannot lower fails the conversion rather than being left behind.
    ConversionTarget target(*context);
    target.addIllegalDialect<micro::MicroDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, memref::MemRefDialect,
                           scf::SCFDialect, tensor::TensorDialect>();

    RewritePatternSet patterns(context);
    patterns.add<TileAllocOpLowering, TileViewOpLowering, VectorOpLowering,
                 TransformOpLowering, AsyncCopyOpLowering<micro::AsyncCopyOp>,
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
