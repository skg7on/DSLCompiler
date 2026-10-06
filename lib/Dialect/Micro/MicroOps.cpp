//===- MicroOps.cpp - Micro dialect operations
//-----------------------------===//
//
// Implements the Micro dialect operations.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

// Attribute class declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"

// Type class declarations.
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"

// Op class full declarations (undefines GET_OP_CLASSES internally).
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

//===----------------------------------------------------------------------===//
// Generated operation definitions: parse, print, verify, build, etc.
// Must re-define GET_OP_CLASSES since MicroOps.h.inc undef'd it.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.cpp.inc"

using namespace mlir;
using namespace mlir::micro;

//===----------------------------------------------------------------------===//
// Canonical tile construction (MicroHelpers.h)
//===----------------------------------------------------------------------===//

Type mlir::micro::materializedTileType(Type source, Attribute memory) {
  auto memoryAttr = dyn_cast<MemorySpaceAttr>(memory);
  if (!memoryAttr)
    return {};

  ArrayRef<int64_t> shape;
  Type elementType;
  LayoutAttr layout;
  OwnerAttr owner;
  if (auto tile = dyn_cast<TileType>(source)) {
    shape = tile.getShape();
    elementType = tile.getElementType();
    layout = tile.getLayout();
    owner = tile.getOwner();
  } else if (auto shaped = dyn_cast<ShapedType>(source)) {
    // A shaped source must have a statically known image: a dynamic extent has
    // no tile shape to materialize, so it is refused rather than guessed.
    if (!shaped.hasStaticShape())
      return {};
    shape = shaped.getShape();
    elementType = shaped.getElementType();
  } else {
    return {};
  }

  // A destination tile needs a concrete storage footprint, so every extent must
  // be statically known; a dynamic extent has no byte size to reserve and is
  // refused rather than guessed.
  for (int64_t dim : shape)
    if (ShapedType::isDynamic(dim))
      return {};
  if (!dtypeOfElementType(elementType))
    return {};
  return TileType::get(source.getContext(), shape, elementType, layout,
                       memoryAttr, owner);
}

//===----------------------------------------------------------------------===//
// Custom assembly format and verifier for KernelOp.
//===----------------------------------------------------------------------===//

/// The logical `(shape, element type)` a type presents across the kernel
/// boundary. A tile, a tensor and a memref with the same extents and element
/// type describe the same data: memory space and layout are scheduling facts,
/// not part of a caller's ABI.
struct KernelLogicalImage {
  ArrayRef<int64_t> shape;
  Type elementType;
};

static std::optional<KernelLogicalImage> kernelLogicalImage(Type type) {
  if (auto tile = dyn_cast<micro::TileType>(type))
    return KernelLogicalImage{tile.getShape(), tile.getElementType()};
  if (auto shaped = dyn_cast<ShapedType>(type))
    return KernelLogicalImage{shaped.getShape(), shaped.getElementType()};
  return std::nullopt;
}

/// True when `declared` and `produced` describe the same logical data. Types
/// with no shaped image (an `index`, say) must match exactly.
static bool sameLogicalData(Type declared, Type produced) {
  auto lhs = kernelLogicalImage(declared);
  auto rhs = kernelLogicalImage(produced);
  if (!lhs || !rhs)
    return declared == produced;
  return lhs->shape == rhs->shape && lhs->elementType == rhs->elementType;
}

/// `micro.kernel @name` with an optional signature, then the attributes and the
/// body. The signature is `(%arg: type, ...) -> type`; omitting it entirely is
/// the legacy argumentless form, which carries no contract.
ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, getSymNameAttrName(result.name),
                             result.attributes))
    return failure();

  llvm::SmallVector<OpAsmParser::Argument> arguments;
  llvm::SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseArgumentList(arguments, OpAsmParser::Delimiter::None,
                                 /*allowType=*/true) ||
        parser.parseRParen())
      return failure();

    // The result list is optional: a kernel that produces nothing carries no
    // `->`, which is how the compiler-generated kernels currently look while
    // their output is still written by `micro.tile_store`. A single result is
    // a bare type; several are parenthesized, matching what the printer emits.
    if (succeeded(parser.parseOptionalArrow())) {
      if (succeeded(parser.parseOptionalLParen())) {
        if (parser.parseTypeList(resultTypes) || parser.parseRParen())
          return failure();
      } else if (parser.parseTypeList(resultTypes)) {
        return failure();
      }
    }

    llvm::SmallVector<Type> inputTypes;
    inputTypes.reserve(arguments.size());
    for (OpAsmParser::Argument &argument : arguments)
      inputTypes.push_back(argument.type);
    result.addAttribute(getFunctionTypeAttrName(result.name),
                        TypeAttr::get(FunctionType::get(
                            parser.getContext(), inputTypes, resultTypes)));
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, arguments, /*enableNameShadowing=*/true))
    return failure();

  // A custom parser has to add the implicit terminator itself; the declarative
  // `$body` form did it, and without this a body that omits its `micro.yield`
  // parses but fails verification.
  KernelOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void KernelOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());

  // A legacy kernel has no signature to print; an explicit one names its
  // inputs through the entry block arguments the region already carries.
  if (FunctionType signature = getKernelFunctionType()) {
    call_interface_impl::printFunctionSignature(
        printer, signature.getInputs(), /*argAttrs=*/ArrayAttr(),
        /*isVariadic=*/false, signature.getResults(),
        /*resultAttrs=*/ArrayAttr(), &getBody(), /*printEmptyResult=*/false);
  }

  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getSymNameAttrName(), getFunctionTypeAttrName()});

  printer << ' ';
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult KernelOp::verify() {
  // workload, if present, must be non-empty.
  if (auto workload = getWorkload())
    if (workload->empty())
      return emitOpError("workload attribute must be non-empty");

  // Search metadata drives the scheduler and is never executed on hardware,
  // so it does not belong in a concrete kernel.
  Operation *searchOp = nullptr;
  getBody().walk([&](Operation *op) {
    if (!searchOp &&
        isa<SearchSpaceOp, ParamOp, ConstraintOp, ObjectiveOp, CandidateOp>(op))
      searchOp = op;
  });
  if (searchOp)
    return searchOp->emitOpError(
        "search ops are not allowed inside micro.kernel");

  // An explicit signature is a contract. Hold the body to it: the entry block
  // arguments are the declared inputs and the yielded values are the results.
  FunctionType signature = getKernelFunctionType();
  if (!signature)
    return success();

  Block &entry = getBody().front();
  if (entry.getNumArguments() != signature.getNumInputs())
    return emitOpError() << "body has " << entry.getNumArguments()
                         << " block argument(s) but the signature declares "
                         << signature.getNumInputs() << " input(s)";
  for (unsigned i = 0, e = signature.getNumInputs(); i < e; ++i) {
    Type actual = entry.getArgument(i).getType();
    if (actual != signature.getInput(i))
      return emitOpError() << "block argument #" << i << " has type " << actual
                           << ", but the signature declares "
                           << signature.getInput(i);
  }

  auto yield = dyn_cast<YieldOp>(entry.getTerminator());
  if (!yield)
    return emitOpError("body must terminate with micro.yield");

  if (yield.getNumOperands() != signature.getNumResults())
    return emitOpError() << "body yields " << yield.getNumOperands()
                         << " value(s) but the signature declares "
                         << signature.getNumResults() << " result(s)";
  for (unsigned i = 0, e = signature.getNumResults(); i < e; ++i) {
    Type produced = yield.getOperand(i).getType();
    if (!sameLogicalData(signature.getResult(i), produced))
      return emitOpError() << "yielded value #" << i << " has type " << produced
                           << ", which does not match the declared result "
                           << signature.getResult(i);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Tile op verifiers
//===----------------------------------------------------------------------===//

static bool isSupportedReduceOp(StringRef op) {
  return llvm::StringSwitch<bool>(op)
      .Cases({"sum", "max", "min", "prod", "product"}, true)
      .Default(false);
}

// Returns the arity of a vector operation, or 0 for an unknown operation.
static int vectorOpArity(StringRef op) {
  return llvm::StringSwitch<int>(op)
      .Cases({"add", "sub", "mul", "div", "max", "min", "compare"}, 2)
      .Cases({"exp", "silu", "sigmoid", "reciprocal", "convert"}, 1)
      .Case("select", 3)
      .Default(0);
}

// Verifies that an optional owner attribute is carried by and agrees with the
// result tile's owner (shared by tile_partition and tile_async_copy).
template <typename OpTy>
static LogicalResult verifyOwnerAttrMatches(OpTy op, std::optional<Owner> owner,
                                            TileType resultType) {
  if (!owner)
    return success();
  if (!resultType.getOwner())
    return op.emitOpError(
        "owner attribute requires the result tile to carry an owner");
  if (*owner != resultType.getOwner().getValue())
    return op.emitOpError("owner attribute must match result tile owner");
  return success();
}

/// Returns the byte size of a tile (product of its static dimensions times the
/// element byte width), or nullopt if any dimension is dynamic.
static std::optional<int64_t> tileByteSize(TileType tile) {
  int64_t elements = 1;
  for (int64_t dim : tile.getShape()) {
    if (ShapedType::isDynamic(dim))
      return std::nullopt;
    elements *= dim;
  }
  switch (*dtypeOfElementType(tile.getElementType())) {
  case DType::f32:
  case DType::i32:
    return elements * 4;
  case DType::f16:
  case DType::bf16:
    return elements * 2;
  case DType::i8:
    return elements;
  }
  return std::nullopt;
}

LogicalResult TileViewOp::verify() {
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a tile type");

  Type sourceType = getSource().getType();
  unsigned sourceRank = 0;
  Type sourceElementType;
  if (auto shapedType = dyn_cast<ShapedType>(sourceType)) {
    sourceRank = shapedType.getRank();
    sourceElementType = shapedType.getElementType();
  } else if (auto sourceTile = dyn_cast<TileType>(sourceType)) {
    sourceRank = sourceTile.getShape().size();
    sourceElementType = sourceTile.getElementType();
  } else {
    return emitOpError("source must be a tensor or tile");
  }

  // The view must preserve the element type.
  if (sourceElementType != resultType.getElementType())
    return emitOpError("source and result element types must match");

  if (getShape() != resultType.getShape())
    return emitOpError("shape attribute must match result tile shape");

  // Offsets, when present, must match the source rank.
  if (!getOffsets().empty() && getOffsets().size() != sourceRank)
    return emitOpError("offset count must match the source rank");

  // A layout on the op must be carried by the result tile and agree with it.
  if (auto layout = getLayout()) {
    if (!resultType.getLayout())
      return emitOpError(
          "layout attribute requires the result tile to carry a layout");
    if (layout != resultType.getLayout())
      return emitOpError("layout attribute must match result tile layout");
  }

  return success();
}

LogicalResult TileAllocOp::verify() {
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a tile type");
  if (!resultType.getMemory())
    return emitOpError("tile_alloc result tile must have a memory space");
  if (resultType.getMemory().getValue() == MemorySpace::dram)
    return emitOpError("tile_alloc memory must not be dram");
  // A materialized allocation must have a statically computable byte size.
  if (!tileByteSize(resultType))
    return emitOpError(
        "tile_alloc result must have a statically computable byte size");
  return success();
}

LogicalResult TilePartitionOp::verify() {
  auto sourceType = dyn_cast<TileType>(getSource().getType());
  if (!sourceType)
    return emitOpError("source must be a tile type");
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a tile type");

  // The partition must preserve the element type.
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("source and result element types must match");

  if (getShape() != resultType.getShape())
    return emitOpError("shape attribute must match result tile shape");

  if (failed(verifyOwnerAttrMatches(*this, getOwner(), resultType)))
    return failure();

  // Fragment shape must divide the parent shape, unless a tail/mask is set.
  bool hasTail = getTail().value_or(false);
  ArrayRef<int64_t> parent = sourceType.getShape();
  ArrayRef<int64_t> fragment = resultType.getShape();
  if (parent.size() != fragment.size())
    return emitOpError("fragment shape must divide parent shape");
  if (!hasTail) {
    for (unsigned i = 0; i < parent.size(); ++i) {
      if (ShapedType::isDynamic(parent[i]) ||
          ShapedType::isDynamic(fragment[i]))
        continue;
      if (parent[i] % fragment[i] != 0)
        return emitOpError("fragment shape must divide parent shape");
    }
  }
  return success();
}

/// Validates a movement's concrete node identity (`micro.src_node` /
/// `micro.dst_node`) as *attribute shape*: a recorded node id is a non-empty
/// string. When the two abstract memories coincide, kind equality is not node
/// identity, so a same-kind movement must record two *distinct* concrete node
/// ids; the machine-aware verifier resolves those ids against the machine model
/// (link, engine, transaction/alignment facts). A same-kind move with no node
/// identity is rejected here, so `#micro.memory<sram>` -> `#micro.memory<sram>`
/// can never pass as real work on kind equality alone.
static LogicalResult verifyMovementNodeIdentity(Operation *op, bool sameSpace) {
  Attribute src = op->getAttr("micro.src_node");
  Attribute dst = op->getAttr("micro.dst_node");
  if (src && !isa<StringAttr>(src))
    return op->emitOpError() << "'micro.src_node' must be a string memory id";
  if (dst && !isa<StringAttr>(dst))
    return op->emitOpError() << "'micro.dst_node' must be a string memory id";
  if (!sameSpace)
    return success();

  auto srcId = dyn_cast_or_null<StringAttr>(src);
  auto dstId = dyn_cast_or_null<StringAttr>(dst);
  if (!srcId || !dstId || srcId.getValue().empty() || dstId.getValue().empty())
    return op->emitOpError()
           << "source and destination memory must differ: a same-kind "
              "movement requires distinct 'micro.src_node' and "
              "'micro.dst_node'";
  if (srcId.getValue() == dstId.getValue())
    return op->emitOpError()
           << "'micro.src_node' and 'micro.dst_node' must name distinct "
              "concrete memories: equal memory kinds are not node identity";
  return success();
}

LogicalResult TileAsyncCopyOp::verify() {
  auto sourceType = dyn_cast<TileType>(getSource().getType());
  if (!sourceType)
    return emitOpError("source must be a tile type");
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a tile type");
  if (!isa<AsyncTokenType>(getToken().getType()))
    return emitOpError("token result must be an async token");

  // The result must be a materialized tile in the destination memory.
  if (!resultType.getMemory())
    return emitOpError("result tile must have a memory space");
  if (resultType.getMemory().getValue() != getDstMemory())
    return emitOpError("result tile memory must match destination memory");

  // Source and destination *kinds* must differ, unless the op records two
  // distinct concrete node ids: a machine may hold several memories of one
  // abstract kind, and a move between two of them is real work even though the
  // kinds are equal. Kind equality is not node identity.
  const bool sameSpace = sourceType.getMemory() &&
                         sourceType.getMemory().getValue() == getDstMemory();
  if (failed(verifyMovementNodeIdentity(getOperation(), sameSpace)))
    return failure();

  // The copy must preserve shape and element type.
  if (resultType.getShape() != sourceType.getShape())
    return emitOpError("result tile shape must match source tile shape");
  if (resultType.getElementType() != sourceType.getElementType())
    return emitOpError("result tile element type must match source tile "
                       "element type");

  if (failed(verifyOwnerAttrMatches(*this, getOwner(), resultType)))
    return failure();

  return success();
}

ParseResult TileStoreOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  OpAsmParser::UnresolvedOperand source;
  if (parser.parseOperand(source))
    return failure();

  // Optional `into %dst[%i, %j]`: the value-producing form. Without it the
  // store stays a terminal write and carries no result.
  OpAsmParser::UnresolvedOperand destination;
  bool hasDestination = false;
  SmallVector<OpAsmParser::UnresolvedOperand> offsets;
  if (succeeded(parser.parseOptionalKeyword("into"))) {
    hasDestination = true;
    if (parser.parseOperand(destination) || parser.parseLSquare())
      return failure();
    if (failed(parser.parseOptionalRSquare())) {
      do {
        OpAsmParser::UnresolvedOperand offset;
        if (parser.parseOperand(offset))
          return failure();
        offsets.push_back(offset);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare())
        return failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  Type sourceType;
  if (parser.parseColonType(sourceType))
    return failure();
  if (parser.resolveOperand(source, sourceType, result.operands))
    return failure();

  if (hasDestination) {
    Type destinationType, resultType;
    if (parser.parseComma() || parser.parseType(destinationType) ||
        parser.parseArrow() || parser.parseType(resultType))
      return failure();
    result.addTypes(resultType);
    if (parser.resolveOperand(destination, destinationType, result.operands))
      return failure();
    for (OpAsmParser::UnresolvedOperand offset : offsets)
      if (parser.resolveOperand(offset, builder.getIndexType(),
                                result.operands))
        return failure();
  }

  // The op has two variadic operand groups, so the split between them has to
  // be recorded: no destination and no offsets is the terminal form.
  result.getOrAddProperties<TileStoreOp::Properties>().operandSegmentSizes = {
      1, hasDestination ? 1 : 0, static_cast<int32_t>(offsets.size())};
  return success();
}

void TileStoreOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getSource();
  if (getDestination()) {
    printer << " into " << getDestination() << '[';
    llvm::interleaveComma(getOffsets(), printer);
    printer << ']';
  }
  // The operand-segment split is bookkeeping the parser recomputes, so it is
  // kept out of the printed form.
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes"});
  printer << " : " << getSource().getType();
  if (getDestination())
    printer << ", " << getDestination().getType() << " -> "
            << getResult().getType();
}

LogicalResult TileStoreOp::verify() {
  auto sourceType = dyn_cast<TileType>(getSource().getType());
  if (!sourceType)
    return emitOpError("source must be a tile type");
  if (!sourceType.getMemory())
    return emitOpError("stored tile must have a memory space");
  if (sourceType.getMemory().getValue() == getDstMemory())
    return emitOpError("source and destination memory must differ");

  // A terminal write has no destination and no result; a value-producing one
  // has both, and the result is the destination it updated.
  if (!getDestination()) {
    if (getNumResults() != 0)
      return emitOpError("a terminal tile_store has no result");
    if (!getOffsets().empty())
      return emitOpError("a terminal tile_store has no offsets");
    return success();
  }

  auto destinationType = dyn_cast<TileType>(getDestination().getType());
  if (!destinationType)
    return emitOpError("destination must be a tile type");
  if (getNumResults() != 1)
    return emitOpError("a tile_store with a destination produces the updated "
                       "destination as its result");
  if (getResult().getType() != getDestination().getType())
    return emitOpError("result must have the destination's type");
  if (getOffsets().size() != destinationType.getShape().size())
    return emitOpError("offsets must name one index per destination dimension");
  return success();
}

LogicalResult MmaOp::verify() {
  auto lhsType = dyn_cast<TileType>(getLhs().getType());
  auto rhsType = dyn_cast<TileType>(getRhs().getType());
  auto accType = dyn_cast<TileType>(getAcc().getType());
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!lhsType || !rhsType || !accType || !resultType)
    return emitOpError("mma operands and result must be tile types");

  // shape is interpreted as [M, N, K].
  if (getShape().size() != 3)
    return emitOpError("mma shape must have exactly 3 dimensions");
  int64_t m = getShape()[0], n = getShape()[1], k = getShape()[2];
  for (int64_t dim : getShape())
    if (dim <= 0)
      return emitOpError("mma shape dimensions must be positive");

  auto dimMatches = [](int64_t expected, int64_t actual) {
    return ShapedType::isDynamic(expected) || ShapedType::isDynamic(actual) ||
           expected == actual;
  };

  // Geometric compatibility: lhs=[M,K], rhs=[K,N], acc=[M,N].
  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  ArrayRef<int64_t> accShape = accType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  if (lhsShape.size() != 2 || rhsShape.size() != 2 || accShape.size() != 2 ||
      resultShape.size() != 2)
    return emitOpError("mma operands and result must be rank-2 tiles");
  if (!dimMatches(m, lhsShape[0]) || !dimMatches(k, lhsShape[1]))
    return emitOpError("lhs tile shape must be [M, K]");
  if (!dimMatches(k, rhsShape[0]) || !dimMatches(n, rhsShape[1]))
    return emitOpError("rhs tile shape must be [K, N]");
  if (!dimMatches(m, accShape[0]) || !dimMatches(n, accShape[1]))
    return emitOpError("accumulator tile shape must be [M, N]");
  if (!dimMatches(m, resultShape[0]) || !dimMatches(n, resultShape[1]))
    return emitOpError("result tile shape must be [M, N]");

  // Dtype compatibility.
  if (dtypeOfElementType(lhsType.getElementType()) != getInput() ||
      dtypeOfElementType(rhsType.getElementType()) != getInput())
    return emitOpError("input dtype must match operand element types");
  if (dtypeOfElementType(accType.getElementType()) != getAccumulator())
    return emitOpError("accumulator dtype must match accumulator element type");
  return success();
}

LogicalResult VectorOp::verify() {
  int arity = vectorOpArity(getOp());
  if (arity == 0)
    return emitOpError("unknown vector operation");

  if (static_cast<int>(getInputs().size()) != arity)
    return emitOpError("vector operation has wrong number of operands");

  SmallVector<TileType> inputTypes;
  for (Value input : getInputs()) {
    auto inputType = dyn_cast<TileType>(input.getType());
    if (!inputType)
      return emitOpError("vector operands must be tile types");
    inputTypes.push_back(inputType);
  }
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("vector result must be a tile type");

  // Elementwise data operands (all but select's leading mask/condition) must
  // match the result shape. Element types must match except for convert and
  // compare, which legitimately change element type.
  bool elementTypeMustMatch = (getOp() != "convert" && getOp() != "compare");
  int dataStart = (getOp() == "select") ? 1 : 0;
  for (unsigned i = dataStart; i < inputTypes.size(); ++i) {
    if (inputTypes[i].getShape() != resultType.getShape())
      return emitOpError("vector operand shape must match result shape");
    if (elementTypeMustMatch &&
        inputTypes[i].getElementType() != resultType.getElementType())
      return emitOpError("vector operand element type must match result "
                         "element type");
  }
  return success();
}

LogicalResult ReduceOp::verify() {
  if (!isSupportedReduceOp(getOp()))
    return emitOpError("unknown reduce operation");
  auto inputType = dyn_cast<TileType>(getInput().getType());
  if (!inputType)
    return emitOpError("reduce input must be a tile type");
  auto resultType = dyn_cast<TileType>(getResult().getType());
  if (!resultType)
    return emitOpError("reduce result must be a tile type");

  uint64_t axis = getAxis();
  if (axis >= inputType.getShape().size())
    return emitOpError("reduce axis must be within the input rank");

  // Result shape must be the input shape with the reduced axis removed.
  SmallVector<int64_t> expectedResultShape;
  for (unsigned i = 0; i < inputType.getShape().size(); ++i)
    if (i != axis)
      expectedResultShape.push_back(inputType.getShape()[i]);
  if (resultType.getShape() != ArrayRef<int64_t>(expectedResultShape))
    return emitOpError("reduce result shape must be the input shape with the "
                       "reduced axis removed");
  if (resultType.getElementType() != inputType.getElementType())
    return emitOpError("reduce result element type must match input element "
                       "type");
  return success();
}

//===----------------------------------------------------------------------===//
// Gather and barrier verifiers
//===----------------------------------------------------------------------===//

/// The static shape and element type a value states, whether it is a
/// `!micro.tile` or a plain shaped type. A gather may combine either, so both
/// are read here; a value that is neither is refused by the caller.
static std::optional<std::pair<llvm::ArrayRef<int64_t>, Type>>
gatherShapeAndElement(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return std::make_pair(shaped.getShape(), shaped.getElementType());
  if (auto tile = dyn_cast<TileType>(type))
    return std::make_pair(tile.getShape(), tile.getElementType());
  return std::nullopt;
}

LogicalResult GatherOp::verify() {
  std::optional<GatherKind> kind = symbolizeGatherKind(getKind());
  if (!kind)
    return emitOpError("unknown gather kind '")
           << getKind() << "' (expected sum, max, or concat)";

  // A gather combines producers; fewer than two is not a gather.
  if (getInputs().size() < 2)
    return emitOpError("gather requires at least two inputs");

  auto result = gatherShapeAndElement(getResult().getType());
  if (!result)
    return emitOpError("gather result must be a tile or shaped type");

  SmallVector<llvm::ArrayRef<int64_t>> shapes;
  for (Value input : getInputs()) {
    auto shapeAndElement = gatherShapeAndElement(input.getType());
    if (!shapeAndElement)
      return emitOpError("gather inputs must be tile or shaped types");
    if (shapeAndElement->second != result->second)
      return emitOpError("gather input and result element types must match");
    shapes.push_back(shapeAndElement->first);
  }

  if (*kind == GatherKind::concat) {
    std::optional<int64_t> axis = getAxis();
    if (!axis)
      return emitOpError("concat gather requires an axis");
    if (*axis < 0 || static_cast<uint64_t>(*axis) >= result->first.size())
      return emitOpError("concat axis must be within the result rank");
    unsigned position = static_cast<unsigned>(*axis);
    // Every non-axis extent must match the result and be statically known; the
    // axis extent must be exactly the sum of the inputs' extents. An unknown
    // extent has no exact resulting extent, so it is refused rather than
    // guessed.
    int64_t axisExtent = 0;
    for (llvm::ArrayRef<int64_t> shape : shapes) {
      if (shape.size() != result->first.size())
        return emitOpError("concat input and result ranks must match");
      for (unsigned dim = 0; dim < shape.size(); ++dim) {
        if (dim == position)
          continue;
        if (ShapedType::isDynamic(shape[dim]) ||
            ShapedType::isDynamic(result->first[dim]) ||
            shape[dim] != result->first[dim])
          return emitOpError(
              "concat gather requires every non-axis extent to match exactly");
      }
      if (ShapedType::isDynamic(shape[position]))
        return emitOpError(
            "concat gather requires statically known input extents");
      axisExtent += shape[position];
    }
    if (ShapedType::isDynamic(result->first[position]) ||
        axisExtent != result->first[position])
      return emitOpError("concat result extent along the axis must equal the "
                         "sum of the input extents");
    return success();
  }

  // Sum and Max combine like-for-like operands and must not carry an axis.
  if (getAxis())
    return emitOpError("sum and max gather must not carry an axis");
  for (llvm::ArrayRef<int64_t> shape : shapes)
    if (shape != result->first)
      return emitOpError(
          "sum and max gather inputs and result must have the same shape");
  return success();
}

LogicalResult BarrierOp::verify() {
  if (!symbolizeBarrierScope(getScope()))
    return emitOpError("unknown barrier scope '")
           << getScope() << "' (expected executor_group)";
  return success();
}

//===----------------------------------------------------------------------===//
// Concrete loop/sync op verifiers and parsers
//===----------------------------------------------------------------------===//
/// Verifies a loop step is positive when statically known.
static LogicalResult verifyStaticPositiveStep(Operation *op, Value step) {
  if (auto constantOp = step.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
      if (!intAttr.getValue().isStrictlyPositive())
        return op->emitOpError("step must be positive");
    }
  }
  return success();
}

/// Parses the optional `iter_args(%carried = %init, ...) -> (types)` clause.
///
/// `regionArgs` already holds the induction variable, so a carried value's
/// block argument follows the ones before it. A loop that declares no
/// `iter_args` carries nothing, which is legal and simply parses to nothing.
static ParseResult
parseLoopCarriedValues(OpAsmParser &parser, OperationState &result,
                       SmallVectorImpl<OpAsmParser::Argument> &regionArgs) {
  if (failed(parser.parseOptionalKeyword("iter_args")))
    return success();
  if (parser.parseLParen())
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> inits;
  do {
    OpAsmParser::Argument carried;
    OpAsmParser::UnresolvedOperand init;
    if (parser.parseOperand(carried.ssaName) || parser.parseEqual() ||
        parser.parseOperand(init))
      return failure();
    regionArgs.push_back(carried);
    inits.push_back(init);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRParen())
    return failure();

  // The declared types are the carried values' types, and they are also the
  // loop's result types: what an iteration yields is what the loop hands back.
  SmallVector<Type> carriedTypes;
  if (parser.parseArrow() || parser.parseLParen() ||
      parser.parseTypeList(carriedTypes) || parser.parseRParen())
    return failure();
  result.addTypes(carriedTypes);

  for (unsigned i = 0, e = inits.size(); i < e; ++i) {
    regionArgs[i + 1].type = carriedTypes[i];
    if (parser.resolveOperand(inits[i], carriedTypes[i], result.operands))
      return failure();
  }
  return success();
}

/// Prints the `iter_args(...) -> (...)` clause, if the loop carries anything.
static void printLoopCarriedValues(OpAsmPrinter &printer, OperandRange initArgs,
                                   Block::BlockArgListType regionIterArgs,
                                   TypeRange resultTypes) {
  if (initArgs.empty())
    return;
  printer << " iter_args(";
  llvm::interleaveComma(
      llvm::zip(regionIterArgs, initArgs), printer, [&](auto pair) {
        printer << std::get<0>(pair) << " = " << std::get<1>(pair);
      });
  printer << ") -> (";
  llvm::interleaveComma(resultTypes, printer);
  printer << ")";
}

/// Verifies the loop-carried contract both loop ops share: as many block
/// arguments as values passed in, matching types end to end, and a terminator
/// that yields one value per hole.
template <typename LoopOp>
static LogicalResult verifyLoopCarriedValues(LoopOp op) {
  Block &entry = op.getBody().front();
  auto regionIterArgs = entry.getArguments().drop_front();
  auto initArgs = op.getInitArgs();
  auto results = op.getResults();

  if (regionIterArgs.size() != initArgs.size())
    return op.emitOpError() << "body has " << regionIterArgs.size()
                            << " loop-carried block argument(s) but the loop "
                               "passes in "
                            << initArgs.size();
  if (results.size() != initArgs.size())
    return op.emitOpError() << "loop has " << results.size()
                            << " result(s) but carries " << initArgs.size();
  for (unsigned i = 0, e = initArgs.size(); i < e; ++i) {
    if (regionIterArgs[i].getType() != initArgs[i].getType())
      return op.emitOpError()
             << "loop-carried block argument #" << i << " has type "
             << regionIterArgs[i].getType() << " but the loop passes in "
             << initArgs[i].getType();
    if (results[i].getType() != initArgs[i].getType())
      return op.emitOpError()
             << "result #" << i << " has type " << results[i].getType()
             << " but the loop carries " << initArgs[i].getType();
  }

  auto yield = dyn_cast<YieldOp>(entry.getTerminator());
  if (!yield)
    return op.emitOpError("body must terminate with micro.yield");
  if (yield.getNumOperands() != initArgs.size())
    return op.emitOpError()
           << "body yields " << yield.getNumOperands()
           << " value(s) but the loop carries " << initArgs.size();
  for (unsigned i = 0, e = initArgs.size(); i < e; ++i)
    if (yield.getOperand(i).getType() != initArgs[i].getType())
      return op.emitOpError()
             << "yielded value #" << i << " has type "
             << yield.getOperand(i).getType() << " but the loop carries "
             << initArgs[i].getType();
  return success();
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;

  // Parse `%i = lb to ub step step`.
  if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual() ||
      parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step))
    return failure();

  inductionVariable.type = builder.getIndexType();
  Type indexType = builder.getIndexType();
  if (parser.resolveOperand(lb, indexType, result.operands) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.resolveOperand(step, indexType, result.operands))
    return failure();

  SmallVector<OpAsmParser::Argument> regionArgs{inductionVariable};
  if (parseLoopCarriedValues(parser, result, regionArgs))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();
  ForOp::ensureTerminator(*body, builder, result.location);

  return success();
}

void ForOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInductionVar() << " = " << getLowerBound() << " to "
          << getUpperBound() << " step " << getStep();
  printLoopCarriedValues(printer, getInitArgs(), getRegionIterArgs(),
                         getResults().getTypes());
  printer << ' ';
  // A loop that carries nothing keeps its bare, implicit terminator out of the
  // printed form, exactly as before the carried-value contract existed.
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/!getInitArgs().empty());
}

LogicalResult ForOp::verify() {
  if (failed(verifyStaticPositiveStep(getOperation(), getStep())))
    return failure();
  return verifyLoopCarriedValues(*this);
}

ParseResult SpatialForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;

  if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual() ||
      parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step) || parser.parseKeyword("map") ||
      parser.parseEqual())
    return failure();

  MappingTargetAttr mapAttr;
  if (parser.parseAttribute(mapAttr))
    return failure();
  result.getOrAddProperties<SpatialForOp::Properties>().map = mapAttr;

  inductionVariable.type = builder.getIndexType();
  Type indexType = builder.getIndexType();
  if (parser.resolveOperand(lb, indexType, result.operands) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.resolveOperand(step, indexType, result.operands))
    return failure();

  SmallVector<OpAsmParser::Argument> regionArgs{inductionVariable};
  if (parseLoopCarriedValues(parser, result, regionArgs))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();
  SpatialForOp::ensureTerminator(*body, builder, result.location);

  return success();
}

void SpatialForOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInductionVar() << " = " << getLowerBound() << " to "
          << getUpperBound() << " step " << getStep()
          << " map = " << getMapAttr();
  printLoopCarriedValues(printer, getInitArgs(), getRegionIterArgs(),
                         getResults().getTypes());
  printer << ' ';
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/!getInitArgs().empty());
}

LogicalResult SpatialForOp::verify() {
  if (failed(verifyStaticPositiveStep(getOperation(), getStep())))
    return failure();
  return verifyLoopCarriedValues(*this);
}

ParseResult PipelineOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  // `stages = N`
  IntegerAttr stages;
  if (parser.parseKeyword("stages") || parser.parseEqual() ||
      parser.parseAttribute(stages, builder.getI64Type(), "stages",
                            result.attributes))
    return failure();

  // An optional `-> (types)` clause: a pipeline that overlaps compute with the
  // movement feeding it has to be able to hand the computed value back out.
  if (succeeded(parser.parseOptionalArrow())) {
    SmallVector<Type> resultTypes;
    if (parser.parseLParen() || parser.parseTypeList(resultTypes) ||
        parser.parseRParen())
      return failure();
    result.addTypes(resultTypes);
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();
  PipelineOp::ensureTerminator(*body, builder, result.location);
  return success();
}

void PipelineOp::print(OpAsmPrinter &printer) {
  printer << " stages = " << getStages();
  if (!getResults().empty()) {
    printer << " -> (";
    llvm::interleaveComma(getResults().getTypes(), printer);
    printer << ")";
  }
  printer.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                           /*elidedAttrs=*/{"stages"});
  printer << ' ';
  // With nothing to carry, the implicit terminator stays implicit -- the form
  // every existing fixture prints.
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/!getResults().empty());
}

LogicalResult PipelineOp::verify() {
  if (getStages() < 1)
    return emitOpError("pipeline stages must be at least 1");
  for (Operation &op : getBody().getOps())
    if (auto nested = dyn_cast<PipelineOp>(op))
      return nested.emitOpError("nested pipeline is not allowed");

  // Whatever the pipeline claims to carry has to actually be yielded.
  auto yield = dyn_cast<YieldOp>(getBody().front().getTerminator());
  if (!yield)
    return emitOpError("body must terminate with micro.yield");
  if (yield.getNumOperands() != getResults().size())
    return emitOpError() << "body yields " << yield.getNumOperands()
                         << " value(s) but the pipeline has "
                         << getResults().size() << " result(s)";
  for (unsigned i = 0, e = getResults().size(); i < e; ++i)
    if (yield.getOperand(i).getType() != getResults()[i].getType())
      return emitOpError() << "yielded value #" << i << " has type "
                           << yield.getOperand(i).getType()
                           << " but the pipeline's result is "
                           << getResults()[i].getType();
  return success();
}

LogicalResult AllocOp::verify() {
  if (!isa<ShapedType>(getResult().getType()))
    return emitOpError("result must be a shaped type");
  if (getMemory() == MemorySpace::dram)
    return emitOpError("alloc memory must not be dram");
  return success();
}

LogicalResult AsyncCopyOp::verify() {
  if (!isa<ShapedType>(getSource().getType()) ||
      !isa<ShapedType>(getResult().getType()))
    return emitOpError("source and result must be shaped types");
  if (!isa<AsyncTokenType>(getToken().getType()))
    return emitOpError("token result must be an async token");
  // Same abstract kind is legal only between two distinct concrete nodes; see
  // `verifyMovementNodeIdentity`.
  if (failed(verifyMovementNodeIdentity(getOperation(),
                                        getSrcMemory() == getDstMemory())))
    return failure();
  return success();
}

LogicalResult TransformOp::verify() {
  // `!micro.tile` is a Micro type, not an MLIR `ShapedType`, so both are
  // accepted: a bound plan's transform may be emitted over either (the workload
  // graph carries tensors; a hand-written kernel uses tiles).
  auto shapeAndElement =
      [](Type type) -> std::optional<std::pair<llvm::ArrayRef<int64_t>, Type>> {
    if (auto shaped = dyn_cast<ShapedType>(type))
      return std::make_pair(shaped.getShape(), shaped.getElementType());
    if (auto tile = dyn_cast<TileType>(type))
      return std::make_pair(tile.getShape(), tile.getElementType());
    return std::nullopt;
  };
  auto source = shapeAndElement(getSource().getType());
  auto result = shapeAndElement(getResult().getType());
  if (!source || !result)
    return emitOpError("source and result must be shaped or tile types");
  // A layout transform re-represents a value; it must not change it. Shape and
  // element type stay identical, which is what lets a consumer read the result
  // as the value it already expected. The *layout* may differ -- that is the
  // whole point of the op -- and so may the memory space and owner.
  if (source->first != result->first || source->second != result->second)
    return emitOpError(
        "source and result must have the same shape and element type: a layout "
        "transform re-represents a value, it does not change it");
  // Both maps describe the *same* value, so their logical rank (dimension
  // count) must agree. Their physical rank (result count) may differ -- a
  // blocked layout writes more physical indices than a row-major one.
  std::optional<AffineMap> src = getSrcMap();
  std::optional<AffineMap> dst = getDstMap();
  if (src && dst && src->getNumDims() != dst->getNumDims())
    return emitOpError("src_map and dst_map must have the same number of "
                       "dimensions (the value's logical rank)");
  return success();
}

LogicalResult WaitOp::verify() {
  if (getTokens().empty())
    return emitOpError("wait requires at least one token operand");
  for (Value token : getTokens())
    if (!isa<AsyncTokenType>(token.getType()))
      return emitOpError("wait operands must be async tokens");
  return success();
}

LogicalResult StoreOp::verify() {
  if (!isa<ShapedType>(getSource().getType()))
    return emitOpError("stored value must be a shaped type");
  if (getSrcMemory() == getDstMemory())
    return emitOpError("source and destination memory must differ");
  return success();
}

//===----------------------------------------------------------------------===//
// Search-space op verifiers
//===----------------------------------------------------------------------===//

/// Returns true if `kind` names a supported micro.param domain.
static bool isSupportedParamKind(StringRef kind) {
  return llvm::StringSwitch<bool>(kind)
      .Cases({"integer", "layout", "memory_path", "owner_mapping",
              "fragment_shape", "tail_policy"},
             true)
      .Default(false);
}

/// Integer domains must be positive and strictly increasing. Enforcing a
/// canonical order keeps the printed parameter reproducible, which the
/// generated candidates and FileCheck both depend on.
static LogicalResult verifyIntegerChoices(Operation *op, ArrayAttr choices) {
  std::optional<int64_t> previous;
  for (Attribute choice : choices) {
    auto integer = dyn_cast<IntegerAttr>(choice);
    if (!integer)
      return op->emitOpError("integer parameter choices must be integers");
    int64_t value = integer.getInt();
    if (value <= 0)
      return op->emitOpError("integer parameter choices must be positive");
    if (previous && value <= *previous)
      return op->emitOpError(
          "integer parameter choices must be unique and strictly increasing");
    previous = value;
  }
  return success();
}

/// Validates one choice of a symbolic domain against the Micro vocabulary, so
/// a search space cannot name layouts, memory spaces, or owners that the rest
/// of the dialect would reject.
static LogicalResult verifySymbolicChoice(Operation *op, StringRef kind,
                                          StringRef choice) {
  if (kind == "layout") {
    if (!symbolizeLayoutKind(choice))
      return op->emitOpError("layout choice '")
             << choice << "' is not a known layout kind";
    return success();
  }

  if (kind == "memory_path") {
    llvm::SmallVector<StringRef, 4> spaces;
    choice.split(spaces, ':');
    if (spaces.size() < 2)
      return op->emitOpError("memory_path choice '")
             << choice << "' must join at least two memory spaces with ':'";
    for (StringRef space : spaces)
      if (!symbolizeMemorySpace(space))
        return op->emitOpError("memory_path choice '")
               << choice << "' has unknown memory space '" << space << "'";
    return success();
  }

  if (kind == "owner_mapping") {
    llvm::SmallVector<StringRef, 4> owners;
    choice.split(owners, '/');
    if (owners.size() < 2)
      return op->emitOpError("owner_mapping choice '")
             << choice << "' must join at least two owners with '/'";
    for (StringRef owner : owners)
      if (!symbolizeOwner(owner))
        return op->emitOpError("owner_mapping choice '")
               << choice << "' has unknown owner '" << owner << "'";
    return success();
  }

  if (kind == "fragment_shape") {
    llvm::SmallVector<StringRef, 3> dimensions;
    choice.split(dimensions, 'x');
    bool valid = dimensions.size() == 3;
    for (StringRef dimension : dimensions) {
      int64_t value;
      if (dimension.getAsInteger(10, value) || value <= 0) {
        valid = false;
        break;
      }
    }
    if (!valid)
      return op->emitOpError("fragment_shape choice '")
             << choice << "' must be a positive MxNxK triple";
    return success();
  }

  // tail_policy: masked tails are the only strategy in the MVP.
  if (choice != "mask")
    return op->emitOpError("tail_policy choice '")
           << choice << "' is not supported (only 'mask')";
  return success();
}

LogicalResult ParamOp::verify() {
  if (getName().empty())
    return emitOpError("parameter name must be non-empty");

  StringRef kind = getKind();
  if (!isSupportedParamKind(kind))
    return emitOpError("unsupported parameter kind '") << kind << "'";

  ArrayAttr choices = getChoices();
  if (choices.empty())
    return emitOpError("parameter must declare at least one choice");

  if (kind == "integer")
    return verifyIntegerChoices(getOperation(), choices);

  for (Attribute choice : choices) {
    auto text = dyn_cast<StringAttr>(choice);
    if (!text)
      return emitOpError("symbolic parameter choices must be strings");
    if (failed(verifySymbolicChoice(getOperation(), kind, text.getValue())))
      return failure();
  }
  return success();
}

/// Returns true if `kind` names a supported micro.constraint legality rule.
static bool isSupportedConstraintKind(StringRef kind) {
  return llvm::StringSwitch<bool>(kind)
      .Cases({"sram_capacity", "acc_capacity", "mma_compatible",
              "mapping_extent", "tail_supported", "vector_width_supported",
              "tile_hierarchy_compatible", "layout_supported",
              "owner_supported", "fragment_compatible", "pipeline_live_tiles"},
             true)
      .Default(false);
}

/// Returns true if `metric` is a metric candidates can be ranked by.
static bool isSupportedObjectiveMetric(StringRef metric) {
  return llvm::StringSwitch<bool>(metric)
      .Cases({"latency_cycles", "dram_bytes", "sram_bytes",
              "matrix_utilization", "dma_utilization", "capacity_spill_bytes"},
             true)
      .Default(false);
}

LogicalResult ConstraintOp::verify() {
  if (!isSupportedConstraintKind(getKind()))
    return emitOpError("unsupported constraint kind '") << getKind() << "'";

  ArrayAttr params = getParams();
  if (params.empty())
    return emitOpError("constraint must reference at least one parameter");

  for (Attribute param : params)
    if (!isa<StringAttr>(param))
      return emitOpError("constraint parameters must be strings");

  return success();
}

LogicalResult CandidateOp::verify() {
  if (getSymName().empty())
    return emitOpError("candidate name must be non-empty");

  for (NamedAttribute binding : getBindings())
    if (!isa<IntegerAttr, StringAttr>(binding.getValue()))
      return emitOpError("candidate binding for '")
             << binding.getName().strref()
             << "' must be an integer or a string";

  return success();
}

LogicalResult ObjectiveOp::verify() {
  StringRef direction = getDirection();
  if (direction != "minimize" && direction != "maximize")
    return emitOpError("objective direction must be 'minimize' or 'maximize'");

  if (!isSupportedObjectiveMetric(getMetric()))
    return emitOpError("unsupported objective metric '") << getMetric() << "'";

  std::optional<ArrayAttr> secondary = getSecondary();
  if (!secondary)
    return success();

  llvm::StringSet<> seen;
  for (Attribute metricAttr : *secondary) {
    auto metric = dyn_cast<StringAttr>(metricAttr);
    if (!metric)
      return emitOpError("secondary metrics must be strings");
    if (!isSupportedObjectiveMetric(metric.getValue()))
      return emitOpError("unsupported secondary metric '")
             << metric.getValue() << "'";
    if (!seen.insert(metric.getValue()).second)
      return emitOpError("duplicate secondary metric '")
             << metric.getValue() << "'";
  }

  return success();
}

/// Returns true if `choices` contains `value`. Integer choices are compared by
/// value so that an i32 spelling still matches an i64 one.
static bool choiceContains(ArrayAttr choices, Attribute value) {
  auto integer = dyn_cast<IntegerAttr>(value);
  for (Attribute choice : choices) {
    if (integer) {
      if (auto other = dyn_cast<IntegerAttr>(choice))
        if (other.getInt() == integer.getInt())
          return true;
      continue;
    }
    if (choice == value)
      return true;
  }
  return false;
}

/// Checks one candidate against the declared parameters: every parameter is
/// bound exactly once, with a value of the parameter's domain type drawn from
/// its declared choices.
static LogicalResult verifyCandidate(CandidateOp candidate,
                                     const llvm::StringMap<ParamOp> &params) {
  DictionaryAttr bindings = candidate.getBindings();
  if (bindings.empty())
    return candidate.emitOpError("candidate must bind at least one parameter");

  for (NamedAttribute binding : bindings) {
    StringRef name = binding.getName().strref();
    auto param = params.find(name);
    if (param == params.end())
      return candidate.emitOpError("candidate binds unknown parameter '")
             << name << "'";

    ParamOp declaration = param->second;
    bool isInteger = declaration.getKind() == "integer";

    if (isInteger && !isa<IntegerAttr>(binding.getValue()))
      return candidate.emitOpError("candidate binding for integer parameter '")
             << name << "' must be an integer";
    if (!isInteger && !isa<StringAttr>(binding.getValue()))
      return candidate.emitOpError("candidate binding for parameter '")
             << name << "' must be a string";

    if (!choiceContains(declaration.getChoices(), binding.getValue()))
      return candidate.emitOpError("candidate value for '")
             << name << "' is not one of the declared choices";
  }

  for (auto &entry : params)
    if (!bindings.get(entry.getKey()))
      return candidate.emitOpError("candidate does not bind parameter '")
             << entry.getKey() << "'";

  return success();
}

LogicalResult SearchSpaceOp::verify() {
  if (getWorkload().empty())
    return emitOpError("workload must be non-empty");

  llvm::StringMap<ParamOp> params;
  llvm::StringSet<> candidateNames;
  unsigned objectiveCount = 0;

  // First pass: structural rules and the parameter declarations.
  for (Operation &op : getBody().front().without_terminator()) {
    if (auto param = dyn_cast<ParamOp>(op)) {
      if (!params.try_emplace(param.getName(), param).second)
        return param.emitOpError("duplicate parameter name '")
               << param.getName() << "'";
      continue;
    }
    if (auto objective = dyn_cast<ObjectiveOp>(op)) {
      if (++objectiveCount > 1)
        return objective.emitOpError("at most one micro.objective is allowed");
      continue;
    }
    if (auto candidate = dyn_cast<CandidateOp>(op)) {
      if (!candidateNames.insert(candidate.getSymName()).second)
        return candidate.emitOpError("duplicate candidate name '")
               << candidate.getSymName() << "'";
      continue;
    }
    if (isa<ConstraintOp>(op))
      continue;
    return op.emitOpError(
        "only search ops are allowed in a micro.search_space");
  }

  if (params.empty())
    return emitOpError("search space requires at least one micro.param");

  // Second pass: references between records, which need the parameter set.
  for (Operation &op : getBody().front().without_terminator()) {
    if (auto constraint = dyn_cast<ConstraintOp>(op)) {
      for (Attribute param : constraint.getParams()) {
        StringRef name = cast<StringAttr>(param).getValue();
        if (params.count(name))
          continue;
        unsigned matches = 0;
        for (const auto &entry : params) {
          auto role = entry.second->getAttrOfType<StringAttr>("role");
          if (role && role.getValue() == name)
            ++matches;
        }
        if (matches == 0)
          return constraint.emitOpError(
                     "constraint references unknown parameter '")
                 << name << "'";
        if (matches > 1)
          return constraint.emitOpError(
                     "constraint references ambiguous role '")
                 << name << "'";
      }
      continue;
    }
    if (auto candidate = dyn_cast<CandidateOp>(op))
      if (failed(verifyCandidate(candidate, params)))
        return failure();
  }

  return success();
}
