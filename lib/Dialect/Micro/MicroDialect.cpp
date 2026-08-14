//===- MicroDialect.cpp - Micro dialect implementation
//---------------------===//
//
// Implements the Micro dialect: registration, attribute and type
// parsing/printing, and operation initialization.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroEnums.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

//===----------------------------------------------------------------------===//
// Generated headers: class declarations
//===----------------------------------------------------------------------===//

// Dialect class declaration (no guard needed).
#include "LLK/Dialect/Micro/MicroDialect.h.inc"

// Attribute class declarations (guarded by GET_ATTRDEF_CLASSES).
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"

// Type class declarations (guarded by GET_TYPEDEF_CLASSES).
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"

// Op class declarations with full definitions (guarded by GET_OP_CLASSES).
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

//===----------------------------------------------------------------------===//
// Generated definitions: constructor, destructor
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated attribute definitions (parse/print, storage types)
// Guard must be re-defined since MicroAttributes.h.inc undef'd it.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated type definitions (storage, accessors)
// Guard must be re-defined since MicroTypes.h.inc undef'd it.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void mlir::micro::MicroDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LLK/Dialect/Micro/MicroOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "LLK/Dialect/Micro/MicroAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "LLK/Dialect/Micro/MicroTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// LayoutAttr custom parse/print/verify
//===----------------------------------------------------------------------===//

using namespace mlir;
using namespace mlir::micro;

Attribute LayoutAttr::parse(AsmParser &parser, Type type) {
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLess())
    return {};

  // Parse the layout kind keyword.
  std::string kindStr;
  if (parser.parseKeywordOrString(&kindStr))
    return {};
  auto kindOpt = symbolizeLayoutKind(kindStr);
  if (!kindOpt) {
    parser.emitError(parser.getCurrentLocation(),
                     "unknown layout kind '" + kindStr + "'");
    return {};
  }
  uint32_t kind = static_cast<uint32_t>(*kindOpt);

  DenseI64ArrayAttr block;
  int64_t vector = 0, align = 0, banks = 0;
  StringAttr swizzle;

  // Parse optional named parameters.
  SmallVector<StringRef, 4> seenParams;
  while (succeeded(parser.parseOptionalComma())) {
    StringRef name;
    if (parser.parseKeyword(&name))
      return {};
    if (parser.parseEqual())
      return {};
    if (llvm::is_contained(seenParams, name)) {
      parser.emitError(parser.getCurrentLocation(),
                       "duplicate layout parameter '" + name + "'");
      return {};
    }
    seenParams.push_back(name);
    if (name == "block") {
      if (parser.parseAttribute(block))
        return {};
    } else if (name == "vector") {
      if (parser.parseInteger(vector))
        return {};
    } else if (name == "align") {
      if (parser.parseInteger(align))
        return {};
    } else if (name == "banks") {
      if (parser.parseInteger(banks))
        return {};
    } else if (name == "swizzle") {
      std::string swizzleStr;
      if (parser.parseString(&swizzleStr))
        return {};
      swizzle = StringAttr::get(parser.getContext(), swizzleStr);
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "unknown layout parameter '" + name + "'");
      return {};
    }
  }

  if (parser.parseGreater())
    return {};

  return LayoutAttr::getChecked([&]() { return parser.emitError(loc); },
                                parser.getContext(), kind, block, vector, align,
                                banks, swizzle);
}

void LayoutAttr::print(AsmPrinter &printer) const {
  printer << "<" << stringifyLayoutKind(static_cast<LayoutKind>(getKind()));
  if (getBlock())
    printer << ", block = " << getBlock();
  if (getVector() != 0)
    printer << ", vector = " << getVector();
  if (getAlign() != 0)
    printer << ", align = " << getAlign();
  if (getBanks() != 0)
    printer << ", banks = " << getBanks();
  if (getSwizzle())
    printer << ", swizzle = " << getSwizzle();
  printer << ">";
}

LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 uint32_t kind, DenseI64ArrayAttr block,
                                 int64_t vector, int64_t align, int64_t banks,
                                 StringAttr swizzle) {
  // Kind must be a known layout kind.
  if (kind > static_cast<uint32_t>(LayoutKind::swizzled))
    return emitError() << "unknown layout kind";
  // Block extents must be positive.
  if (block)
    for (int64_t extent : block.asArrayRef())
      if (extent <= 0)
        return emitError() << "block extents must be positive";
  // Vector width must be positive when set.
  if (vector < 0)
    return emitError() << "vector width must be positive";
  // Align must be positive and a power of two when set.
  if (align != 0) {
    if (align <= 0)
      return emitError() << "align must be positive";
    if ((align & (align - 1)) != 0)
      return emitError() << "align must be a power of two";
  }
  // Banks must be positive when set.
  if (banks < 0)
    return emitError() << "banks must be positive";
  // Kind/parameter consistency.
  if (static_cast<LayoutKind>(kind) == LayoutKind::blocked &&
      (!block || block.asArrayRef().empty()))
    return emitError() << "blocked layout requires a block parameter";
  if (static_cast<LayoutKind>(kind) == LayoutKind::vectorized && vector <= 0)
    return emitError() << "vectorized layout requires a positive vector width";
  // Swizzle requires a swizzled layout kind.
  if (swizzle && static_cast<LayoutKind>(kind) != LayoutKind::swizzled)
    return emitError() << "swizzle requires a swizzled layout";
  if (static_cast<LayoutKind>(kind) == LayoutKind::swizzled && !swizzle)
    return emitError() << "swizzled layout requires a swizzle parameter";
  return success();
}

//===----------------------------------------------------------------------===//
// TileType custom parse/print/verify
//===----------------------------------------------------------------------===//

/// Returns true for the MLIR element types that map to a supported Micro
/// dtype (f32, f16, bf16, i32, i8).
static bool isSupportedTileElementType(Type elementType) {
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return floatType.isBF16() || floatType.isF16() || floatType.isF32();
  if (auto intType = dyn_cast<IntegerType>(elementType))
    return intType.getWidth() == 8 || intType.getWidth() == 32;
  return false;
}

Type TileType::parse(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLess())
    return {};

  // Parse the shape as a dimension list (`dim` (`x` `dim`)* `x`), leaving the
  // element type as the next token. parseDimensionList handles the lexer quirk
  // where `x` is juxtaposed with the following token (e.g. "x64").
  SmallVector<int64_t> shape;
  if (parser.parseDimensionList(shape))
    return {};

  Type elementType;
  if (parser.parseType(elementType))
    return {};

  LayoutAttr layout;
  MemorySpaceAttr memory;
  OwnerAttr owner;

  // Parse optional layout, memory, and owner parameters.
  SmallVector<StringRef, 3> seenParams;
  while (succeeded(parser.parseOptionalComma())) {
    StringRef name;
    if (parser.parseKeyword(&name))
      return {};
    if (parser.parseEqual())
      return {};
    if (llvm::is_contained(seenParams, name)) {
      parser.emitError(parser.getCurrentLocation(),
                       "duplicate tile parameter '" + name + "'");
      return {};
    }
    seenParams.push_back(name);
    if (name == "layout") {
      if (parser.parseAttribute(layout))
        return {};
    } else if (name == "memory") {
      if (parser.parseAttribute(memory))
        return {};
    } else if (name == "owner") {
      if (parser.parseAttribute(owner))
        return {};
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "unknown tile parameter '" + name + "'");
      return {};
    }
  }

  if (parser.parseGreater())
    return {};

  return TileType::getChecked([&]() { return parser.emitError(loc); },
                              parser.getContext(), shape, elementType, layout,
                              memory, owner);
}

void TileType::print(AsmPrinter &printer) const {
  printer << "<";
  bool first = true;
  for (int64_t dim : getShape()) {
    if (!first)
      printer << "x";
    first = false;
    if (ShapedType::isDynamic(dim))
      printer << "?";
    else
      printer << dim;
  }
  if (!getShape().empty())
    printer << "x";
  printer << getElementType();
  if (getLayout())
    printer << ", layout = " << getLayout();
  if (getMemory())
    printer << ", memory = " << getMemory();
  if (getOwner())
    printer << ", owner = " << getOwner();
  printer << ">";
}

LogicalResult TileType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<int64_t> shape, Type elementType,
                               LayoutAttr layout, MemorySpaceAttr memory,
                               OwnerAttr owner) {
  // A tile is a shaped value; reject rank-0 (scalar) tiles.
  if (shape.empty())
    return emitError() << "tile must have at least one dimension";
  // Dimensions must be positive or dynamic.
  for (int64_t dim : shape)
    if (!ShapedType::isDynamic(dim) && dim <= 0)
      return emitError() << "tile dimensions must be positive or dynamic";
  // Element type must map to a supported Micro dtype.
  if (!isSupportedTileElementType(elementType))
    return emitError() << "unsupported tile element type";
  return success();
}
