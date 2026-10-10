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
#include "mlir/Dialect/Math/IR/Math.h"
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

#include <limits>
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
    // A block argument whose type changes across a loop boundary -- the carried
    // value of a lowered loop -- needs a bridge while the body is still being
    // legalized. An unrealized cast is the honest placeholder: it claims no
    // semantics, and the conversion is only allowed to finish if every one of
    // them has been reconciled away by the patterns that rewrite the uses.
    auto bridge = [](OpBuilder &builder, Type type, ValueRange inputs,
                     Location loc) -> Value {
      if (inputs.size() != 1)
        return Value();
      return UnrealizedConversionCastOp::create(builder, loc, type, inputs[0])
          .getResult(0);
    };
    addTargetMaterialization(bridge);
    addSourceMaterialization(bridge);
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

/// Casts every element of `source` to `elementType`, or returns `source`
/// unchanged when it already has that type.
///
/// A contraction or an epilogue runs in one element type, so an operand of
/// another type is converted explicitly rather than left for a backend to fold
/// away -- and the direction follows the widths, because an unconditional
/// extend silently corrupts a narrowing conversion.
Value buildElementwiseCast(ConversionPatternRewriter &rewriter, Location loc,
                           Value source, Type elementType) {
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType)
    return {};
  Type from = sourceType.getElementType();
  if (from == elementType)
    return source;

  auto resultType = RankedTensorType::get(sourceType.getShape(), elementType);
  auto empty = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       elementType);
  SmallVector<AffineMap> maps{
      rewriter.getMultiDimIdentityMap(resultType.getRank()),
      rewriter.getMultiDimIdentityMap(resultType.getRank())};
  SmallVector<utils::IteratorType> iterators(resultType.getRank(),
                                             utils::IteratorType::parallel);
  auto cast = linalg::GenericOp::create(
      rewriter, loc, TypeRange{resultType}, ValueRange{source},
      ValueRange{empty}, maps, iterators,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value value = args[0];
        bool wider =
            from.getIntOrFloatBitWidth() < elementType.getIntOrFloatBitWidth();
        Value converted;
        if (isa<FloatType>(from) && isa<FloatType>(elementType))
          converted = wider ? arith::ExtFOp::create(nested, nestedLoc,
                                                    elementType, value)
                                  .getResult()
                            : arith::TruncFOp::create(nested, nestedLoc,
                                                      elementType, value)
                                  .getResult();
        else if (isa<FloatType>(from))
          converted =
              arith::FPToSIOp::create(nested, nestedLoc, elementType, value)
                  .getResult();
        else if (isa<FloatType>(elementType))
          converted =
              arith::SIToFPOp::create(nested, nestedLoc, elementType, value)
                  .getResult();
        else
          converted = wider ? arith::ExtSIOp::create(nested, nestedLoc,
                                                     elementType, value)
                                  .getResult()
                            : arith::TruncIOp::create(nested, nestedLoc,
                                                      elementType, value)
                                  .getResult();
        linalg::YieldOp::create(nested, nestedLoc, converted);
      });
  return cast.getResult(0);
}

/// Record selected identity conversions that do not need a Linalg operation.
/// Keep the mapping fact opaque here; target backends can account for the
/// structural member when validating a fused instance.
void recordStructuralMapping(Operation *source, StringRef microOp,
                             Attribute mathMode) {
  if (!source)
    return;
  auto mapping = source->getAttrOfType<DictionaryAttr>("micro.mapping");
  if (!mapping)
    return;
  ModuleOp module = source->getParentOfType<ModuleOp>();
  if (!module)
    return;

  NamedAttrList record;
  record.set("mapping", mapping);
  record.set("op", StringAttr::get(source->getContext(), microOp));
  if (mathMode)
    record.set("math_mode", mathMode);
  SmallVector<Attribute> records;
  if (auto existing =
          module->getAttrOfType<ArrayAttr>("micro.structural_mappings"))
    records.append(existing.begin(), existing.end());
  DictionaryAttr next = DictionaryAttr::get(source->getContext(), record);
  if (llvm::is_contained(records, Attribute(next)))
    return;
  records.push_back(next);
  module->setAttr("micro.structural_mappings",
                  ArrayAttr::get(source->getContext(), records));
}

/// Preserve opaque mapping provenance on the Linalg operation that replaces a
/// selected Micro compute op. The generic bridge copies facts; only a target
/// backend interprets their target-specific meaning.
void copyMappingMetadata(Operation *source, Operation *replacement,
                         StringRef microOp, Attribute mathMode) {
  if (!source || !replacement || !isa<linalg::GenericOp>(replacement))
    return;
  for (StringRef name :
       {"micro.mapping", "micro.routes", "micro.op", "micro.math_mode"})
    if (Attribute value = source->getAttr(name))
      replacement->setAttr(name, value);
  replacement->setAttr("micro.op",
                       StringAttr::get(source->getContext(), microOp));
  if (mathMode)
    replacement->setAttr("micro.math_mode", mathMode);
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
///
/// The reference form defines freshly allocated tile storage to be **zero**. A
/// real allocation's contents are unconstrained, so no correct program may
/// depend on them -- and an accumulator that started at whatever the allocator
/// left would make every numeric check of a GEMM a coin toss rather than a
/// measurement. Filling is what makes "the kernel computed this" a statement
/// about the program instead of about the heap.
struct TileAllocOpLowering : OpConversionPattern<micro::TileAllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::TileAllocOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!type)
      return rewriter.notifyMatchFailure(op, "not convertible to a tensor");
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), type.getShape(),
                                         type.getElementType());
    auto zero = arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getZeroAttr(type.getElementType()));
    rewriter.replaceOp(op, linalg::FillOp::create(rewriter, op.getLoc(),
                                                  ValueRange{zero},
                                                  ValueRange{empty})
                               .getResult(0));
    return success();
  }
};

/// A logical view allocates nothing.
///
/// When it covers the whole source from the origin it *is* the source, and
/// replacing it costs nothing. Equal shapes are not enough for that: an offset
/// that is not provably a constant zero starts the view somewhere else, so
/// identity needs both.
///
/// Anything else is a window, and a window is a `tensor.extract_slice` -- the
/// offsets and the shape become its offset/size vectors, so a dynamic offset
/// (a loop induction variable, say) stays dynamic rather than being folded to
/// whatever the compiler guessed. A window that is statically out of bounds is
/// refused here rather than turned into a slice that reads past its source.
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

    bool identity = sourceType.getShape() == resultType.getShape();
    for (Value offset : adaptor.getOffsets()) {
      std::optional<int64_t> constant = getConstantIntValue(offset);
      if (!constant || *constant != 0) {
        identity = false;
        break;
      }
    }
    if (identity) {
      rewriter.replaceOp(op, adaptor.getSource());
      return success();
    }

    if (adaptor.getOffsets().empty() ||
        adaptor.getOffsets().size() != (size_t)sourceType.getRank())
      return rewriter.notifyMatchFailure(
          op, "a windowed micro.tile_view needs one offset per source "
              "dimension");

    // A window whose start and extent are both known and leave the source is
    // rejected: silently clipping it would read elements the program did not
    // ask for, and silently over-reading is worse.
    if (sourceType.hasStaticShape() && resultType.hasStaticShape()) {
      for (auto [offset, size, extent] :
           llvm::zip(adaptor.getOffsets(), resultType.getShape(),
                     sourceType.getShape())) {
        std::optional<int64_t> constant = getConstantIntValue(offset);
        if (constant && (*constant < 0 || *constant + size > extent))
          return rewriter.notifyMatchFailure(
              op, ("view window [" + std::to_string(*constant) + ", " +
                   std::to_string(*constant + size) +
                   ") leaves a source dimension of extent " +
                   std::to_string(extent))
                      .c_str());
      }
    }

    SmallVector<OpFoldResult> offsets(adaptor.getOffsets().begin(),
                                      adaptor.getOffsets().end());
    SmallVector<OpFoldResult> sizes;
    for (int64_t dim : resultType.getShape())
      sizes.push_back(rewriter.getIndexAttr(dim));
    SmallVector<OpFoldResult> strides(sourceType.getRank(),
                                      rewriter.getIndexAttr(1));
    auto slice = tensor::ExtractSliceOp::create(rewriter, op.getLoc(),
                                                resultType, adaptor.getSource(),
                                                offsets, sizes, strides);
    rewriter.replaceOp(op, slice.getResult());
    return success();
  }
};

/// `micro.tile_partition` cuts a logical fragment out of a tile.
///
/// Which fragment it names is not in the IR: the verifier only requires the
/// fragment shape to divide the parent's, and a partition is a zero-cost
/// logical annotation rather than a movement. The reference form therefore
/// takes the leading fragment, at the origin. Inventing an offset the program
/// never wrote would be worse -- a consumed partition would then compute on
/// elements nobody asked for -- and a partition the program does not consume is
/// dead either way.
struct TilePartitionOpLowering : OpConversionPattern<micro::TilePartitionOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::TilePartitionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    auto sourceType = dyn_cast<RankedTensorType>(adaptor.getSource().getType());
    if (!resultType || !sourceType)
      return rewriter.notifyMatchFailure(op, "not convertible to a tensor");
    if (sourceType.getRank() != resultType.getRank())
      return rewriter.notifyMatchFailure(
          op, "a fragment must have its parent's rank");

    // A fragment that is the whole parent carries no information.
    if (sourceType.getShape() == resultType.getShape()) {
      rewriter.replaceOp(op, adaptor.getSource());
      return success();
    }

    SmallVector<OpFoldResult> offsets(sourceType.getRank(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (int64_t dim : resultType.getShape())
      sizes.push_back(rewriter.getIndexAttr(dim));
    SmallVector<OpFoldResult> strides(sourceType.getRank(),
                                      rewriter.getIndexAttr(1));
    auto slice = tensor::ExtractSliceOp::create(rewriter, op.getLoc(),
                                                resultType, adaptor.getSource(),
                                                offsets, sizes, strides);
    rewriter.replaceOp(op, slice.getResult());
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

    // A representation change is a per-element conversion, not an elementwise
    // arithmetic op: `convert` is spelled as a `micro.vector` like the rest of
    // the family, but what it means is a dtype cast.
    if (name == "convert") {
      if (adaptor.getInputs().size() != 1)
        return rewriter.notifyMatchFailure(op, "convert takes one operand");
      Value converted = buildElementwiseCast(
          rewriter, op.getLoc(), adaptor.getInputs()[0], elementType);
      if (!converted)
        return rewriter.notifyMatchFailure(op,
                                           "convert needs a shaped operand");
      if (converted == adaptor.getInputs()[0])
        recordStructuralMapping(op, name, op.getMathModeAttr());
      copyMappingMetadata(op, converted.getDefiningOp(), name,
                          op.getMathModeAttr());
      rewriter.replaceOp(op, converted);
      return success();
    }

    // `silu` and `sigmoid` are defined arithmetic rather than a bare arith
    // mnemonic, so they are built here from `math.exp`. The tile carries the
    // math mode the schedule chose; realising that mode differently is the
    // target's decision, and the reference form is the definition.
    if ((name == "silu" || name == "sigmoid") &&
        adaptor.getInputs().size() == 1) {
      if (!isa<FloatType>(elementType))
        return rewriter.notifyMatchFailure(op, "silu needs a float tile");
      auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(),
                                           resultType.getShape(), elementType);
      SmallVector<AffineMap> maps{
          rewriter.getMultiDimIdentityMap(resultType.getRank()),
          rewriter.getMultiDimIdentityMap(resultType.getRank())};
      SmallVector<utils::IteratorType> iterators(resultType.getRank(),
                                                 utils::IteratorType::parallel);
      bool isSilu = name == "silu";
      auto generic = linalg::GenericOp::create(
          rewriter, op.getLoc(), TypeRange{resultType},
          ValueRange{adaptor.getInputs()[0]}, ValueRange{empty}, maps,
          iterators,
          [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
            Value minusX = arith::NegFOp::create(nested, nestedLoc, args[0]);
            Value exponential = math::ExpOp::create(nested, nestedLoc,
                                                    minusX.getType(), minusX);
            Value one = arith::ConstantOp::create(
                nested, nestedLoc, nested.getFloatAttr(elementType, 1.0));
            Value denominator =
                arith::AddFOp::create(nested, nestedLoc, one, exponential);
            Value sigmoid =
                arith::DivFOp::create(nested, nestedLoc, one, denominator);
            Value result = isSilu ? arith::MulFOp::create(nested, nestedLoc,
                                                          args[0], sigmoid)
                                        .getResult()
                                  : sigmoid;
            linalg::YieldOp::create(nested, nestedLoc, result);
          });
      copyMappingMetadata(op, generic, name, op.getMathModeAttr());
      rewriter.replaceOp(op, generic.getResult(0));
      return success();
    }

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
    copyMappingMetadata(op, generic, name, op.getMathModeAttr());
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
/// kernel yields -- the result a caller observes. A loop terminator becomes
/// the loop's, carrying the values an iteration hands the next one.
struct YieldOpLowering : OpConversionPattern<micro::YieldOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *parent = op->getParentOp();
    if (isa<func::FuncOp>(parent)) {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getResults());
      return success();
    }
    if (isa<scf::ForOp>(parent)) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getResults());
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "micro.yield terminates a kernel or a lowered loop");
  }
};

/// `micro.for` and `micro.spatial_for` become the SCF loop they are modeled on.
///
/// The bounds cross unchanged. A carried value's type goes through the type
/// converter, exactly as the loop's own results do: a loop carries whatever its
/// iterations hand each other, which is a tile before lowering and a tensor
/// after.
///
/// A spatial loop lowers to a sequential `scf.for` on purpose. `scf.forall`
/// would assert that the iterations are independent, which the IR does not
/// claim: `#micro.map` says where iterations *may* be placed, not that they do
/// not communicate. Keeping the order is what the reference form promises, and
/// placing the work is the target's decision (C2).
template <typename LoopOp> struct LoopOpLowering : OpConversionPattern<LoopOp> {
  using OpConversionPattern<LoopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoopOp op, typename LoopOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> carriedTypes;
    if (failed(this->getTypeConverter()->convertTypes(
            op.getInitArgs().getTypes(), carriedTypes)))
      return rewriter.notifyMatchFailure(op, "carried types do not convert");

    auto loop = scf::ForOp::create(
        rewriter, op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs());

    // Move the source body in, drop the loop's own placeholder block, and let
    // the conversion rewrite the body's arguments: the induction variable is an
    // index either way, and a carried value goes through the same type change
    // as the loop's operands.
    rewriter.inlineRegionBefore(op.getBody(), loop->getRegion(0),
                                loop->getRegion(0).begin());
    rewriter.eraseBlock(&loop->getRegion(0).back());
    if (failed(rewriter.convertRegionTypes(&loop->getRegion(0),
                                           *this->getTypeConverter())))
      return rewriter.notifyMatchFailure(op, "loop body types do not convert");

    rewriter.replaceOp(op, loop.getResults());
    return success();
  }
};

/// `micro.pipeline` is an overlap hint, and the reference form keeps its body
/// in the order it was written: the operations happen, and realising the
/// overlap is the target's job (C2). A value the pipeline carries out is the
/// value its body yielded, so the results are rewired to those operands.
struct PipelineOpLowering : OpConversionPattern<micro::PipelineOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::PipelineOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();
    auto yield = dyn_cast<micro::YieldOp>(body.getTerminator());
    if (!yield)
      return rewriter.notifyMatchFailure(op, "body has no micro.yield");

    // The terminator is dropped rather than moved: inlining it would leave a
    // terminator in the middle of the enclosing block.
    SmallVector<Value> results(yield.getOperands());
    rewriter.eraseOp(yield);
    rewriter.inlineBlockBefore(&body, op);
    rewriter.replaceOp(op, results);
    return success();
  }
};

/// `micro.tile_store` with a destination updates it: the result is the
/// destination with the tile written at the store's offsets, which is a
/// `tensor.insert_slice` and therefore a value the enclosing loop can carry.
///
/// A store with no destination writes to a memory space and produces nothing.
/// The reference form has no value to put that write in, so it refuses rather
/// than dropping it -- a store whose effect nothing can observe is exactly what
/// the kernel ABI's result contract exists to rule out.
struct TileStoreOpLowering : OpConversionPattern<micro::TileStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::TileStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getDestination())
      return rewriter.notifyMatchFailure(
          op, "a micro.tile_store without a destination writes to memory and "
              "has no value to lower into; give it a destination to thread");

    auto sourceType = dyn_cast<RankedTensorType>(adaptor.getSource().getType());
    if (!sourceType ||
        !dyn_cast<RankedTensorType>(adaptor.getDestination().getType()))
      return rewriter.notifyMatchFailure(op, "not convertible to a tensor");

    // The slice written is the source's own shape; the offsets say where.
    SmallVector<OpFoldResult> offsets(adaptor.getOffsets().begin(),
                                      adaptor.getOffsets().end());
    SmallVector<OpFoldResult> sizes;
    for (int64_t dim : sourceType.getShape())
      sizes.push_back(rewriter.getIndexAttr(dim));
    SmallVector<OpFoldResult> strides(sourceType.getRank(),
                                      rewriter.getIndexAttr(1));

    auto insert = tensor::InsertSliceOp::create(
        rewriter, op.getLoc(), adaptor.getSource(), adaptor.getDestination(),
        offsets, sizes, strides);
    rewriter.replaceOp(op, insert.getResult());
    return success();
  }
};

/// The neutral element of a reduction: what the accumulator starts at, so the
/// first reduced value is also the answer. Starting anywhere else silently
/// folds a constant into every result.
Value neutralElement(OpBuilder &builder, Location loc, llvm::StringRef name,
                     Type elementType) {
  bool floating = isa<FloatType>(elementType);
  if (name == "sum")
    return arith::ConstantOp::create(builder, loc,
                                     builder.getZeroAttr(elementType));
  if (name == "prod" || name == "product") {
    if (floating)
      return arith::ConstantOp::create(builder, loc,
                                       builder.getFloatAttr(elementType, 1.0));
    return arith::ConstantOp::create(builder, loc,
                                     builder.getIntegerAttr(elementType, 1));
  }
  // max and min start at the extreme so that the first element always wins.
  bool largest = name == "max";
  if (floating) {
    double value = largest ? -std::numeric_limits<double>::infinity()
                           : std::numeric_limits<double>::infinity();
    return arith::ConstantOp::create(builder, loc,
                                     builder.getFloatAttr(elementType, value));
  }
  int64_t value = largest ? std::numeric_limits<int64_t>::min()
                          : std::numeric_limits<int64_t>::max();
  return arith::ConstantOp::create(builder, loc,
                                   builder.getIntegerAttr(elementType, value));
}

/// The combining operation a reduction names, at the element type's width.
Value buildReduction(OpBuilder &builder, Location loc, llvm::StringRef name,
                     Value lhs, Value rhs, Type elementType) {
  bool floating = isa<FloatType>(elementType);
  if (name == "sum")
    return floating ? arith::AddFOp::create(builder, loc, lhs, rhs).getResult()
                    : arith::AddIOp::create(builder, loc, lhs, rhs).getResult();
  if (name == "prod" || name == "product")
    return floating ? arith::MulFOp::create(builder, loc, lhs, rhs).getResult()
                    : arith::MulIOp::create(builder, loc, lhs, rhs).getResult();
  if (name == "max")
    return floating
               ? arith::MaxNumFOp::create(builder, loc, lhs, rhs).getResult()
               : arith::MaxSIOp::create(builder, loc, lhs, rhs).getResult();
  if (name == "min")
    return floating
               ? arith::MinNumFOp::create(builder, loc, lhs, rhs).getResult()
               : arith::MinSIOp::create(builder, loc, lhs, rhs).getResult();
  llvm_unreachable("ReduceOp's verifier rejects any other operation");
}

/// `micro.reduce` is a per-axis reduction, which is `linalg.reduce` with the
/// named combining operation and the neutral element as the accumulator's
/// starting value.
struct ReduceOpLowering : OpConversionPattern<micro::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    if (!resultType || !inputType)
      return rewriter.notifyMatchFailure(op, "not convertible to a tensor");

    int64_t axis = static_cast<int64_t>(op.getAxis());
    if (axis < 0 || axis >= inputType.getRank())
      return rewriter.notifyMatchFailure(op, "reduce axis is out of range");

    Type elementType = resultType.getElementType();
    llvm::StringRef name = op.getOp();
    Value neutral = neutralElement(rewriter, op.getLoc(), name, elementType);
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(),
                                         resultType.getShape(), elementType);
    Value seed = linalg::FillOp::create(rewriter, op.getLoc(),
                                        ValueRange{neutral}, ValueRange{empty})
                     .getResult(0);

    auto reduce = linalg::ReduceOp::create(
        rewriter, op.getLoc(), ValueRange{adaptor.getInput()}, ValueRange{seed},
        ArrayRef<int64_t>{axis},
        [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          linalg::YieldOp::create(nested, nestedLoc,
                                  buildReduction(nested, nestedLoc, name,
                                                 args[0], args[1],
                                                 elementType));
        });
    rewriter.replaceOp(op, reduce.getResults());
    return success();
  }
};

/// `micro.barrier` orders concurrent work. The reference form is sequential --
/// every operation happens in the order it was written -- so a barrier has
/// nothing left to order. Dropping it is not a shortcut: realising the
/// concurrency it guards is the target's job, and the target is the only layer
/// that knows which overlap it chose.
struct BarrierOpLowering : OpConversionPattern<micro::BarrierOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::BarrierOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// `micro.mma` is `acc += lhs x rhs`, which is a contraction whose accumulator
/// is its `outs`. That is exactly why the loop has to carry it: an SSA tensor
/// accumulator is only an accumulator if the value flows in and back out.
struct MmaOpLowering : OpConversionPattern<micro::MmaOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(micro::MmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhsType = dyn_cast<RankedTensorType>(adaptor.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(adaptor.getRhs().getType());
    auto accType = dyn_cast<RankedTensorType>(adaptor.getAcc().getType());
    if (!lhsType || !rhsType || !accType)
      return rewriter.notifyMatchFailure(op, "not convertible to tensors");
    if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
        accType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "a matrix contraction needs rank-2 operands");

    // The contraction runs in the accumulator's element type, so operands of a
    // narrower type are extended explicitly. Leaving the mismatch to the
    // contraction would be a dtype change nothing in the IR asked for.
    Type elementType = accType.getElementType();
    Value lhs = adaptor.getLhs();
    if (lhsType.getElementType() != elementType) {
      lhs = buildElementwiseCast(rewriter, op.getLoc(), lhs, elementType);
      if (!lhs)
        return rewriter.notifyMatchFailure(op, "lhs is not a rank-2 tensor");
    }
    Value rhs = adaptor.getRhs();
    if (rhsType.getElementType() != elementType) {
      rhs = buildElementwiseCast(rewriter, op.getLoc(), rhs, elementType);
      if (!rhs)
        return rewriter.notifyMatchFailure(op, "rhs is not a rank-2 tensor");
    }

    auto matmul = linalg::MatmulOp::create(
        rewriter, op.getLoc(), TypeRange{accType}, ValueRange{lhs, rhs},
        ValueRange{adaptor.getAcc()});
    rewriter.replaceOp(op, matmul.getResults());
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
    registry
        .insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                tensor::TensorDialect>();
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

      // A `micro.tile_store` is lowered, not dropped: with a destination it is
      // the value-producing write-back the kernel's result threads through, and
      // without one it writes to memory and is refused by its own lowering. A
      // store is only refused here when the kernel declares no result at all --
      // then nothing could observe the write, and silently discarding it is
      // exactly what the ABI contract exists to prevent.
      llvm::SmallVector<micro::TileStoreOp, 2> stores;
      kernel.getBody().walk(
          [&](micro::TileStoreOp store) { stores.push_back(store); });
      if (!stores.empty() && resultTypes.empty()) {
        kernel.emitError(
            "kernel writes through micro.tile_store but declares no result; "
            "the write would have nothing to update, so declare the output the "
            "kernel produces");
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
      function.getBody().takeBody(kernel.getBody());
      rewriter.eraseOp(kernel);
    }

    // Phase 2: types and ops. The micro dialect is illegal, so anything this
    // pass cannot lower fails the conversion rather than being left behind.
    ConversionTarget target(*context);
    target.addIllegalDialect<micro::MicroDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, math::MathDialect,
                           memref::MemRefDialect, scf::SCFDialect,
                           tensor::TensorDialect>();

    RewritePatternSet patterns(context);
    patterns.add<TileAllocOpLowering, TileViewOpLowering,
                 TilePartitionOpLowering, TileStoreOpLowering, VectorOpLowering,
                 TransformOpLowering, AsyncCopyOpLowering<micro::AsyncCopyOp>,
                 AsyncCopyOpLowering<micro::TileAsyncCopyOp>, WaitOpLowering,
                 YieldOpLowering, LoopOpLowering<micro::ForOp>,
                 LoopOpLowering<micro::SpatialForOp>, PipelineOpLowering,
                 MmaOpLowering, ReduceOpLowering, BarrierOpLowering>(converter,
                                                                     context);

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
