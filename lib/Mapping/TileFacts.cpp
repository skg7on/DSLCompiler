//===- TileFacts.cpp - Bytes and alignment of a moving value --------------===//
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/TileFacts.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::llk::mapping {

namespace {

std::string printedType(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return stream.str();
}

/// Bytes one element of `element` occupies. A float or integer type states a
/// bit width; an index type does not (its width is target-dependent), and any
/// other element type is not modelled here, so both yield `std::nullopt`.
std::optional<uint64_t> elementByteWidth(mlir::Type element) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(element))
    return static_cast<uint64_t>(floatType.getWidth()) / 8;
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(element))
    return (static_cast<uint64_t>(intType.getWidth()) + 7) / 8;
  return std::nullopt;
}

} // namespace

mlir::Type tileAsTensor(mlir::Type type) {
  if (!type)
    return {};
  std::string printed = printedType(type);
  llvm::StringRef text(printed);
  if (!text.consume_front("!micro.tile<"))
    return {};
  size_t end = text.find_first_of(",>");
  if (end == llvm::StringRef::npos)
    return {};
  std::string wrapped = ("tensor<" + text.take_front(end) + ">").str();
  return mlir::parseType(wrapped, type.getContext());
}

mlir::Type elementTypeOf(mlir::Type type) {
  if (!type)
    return {};
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    return shaped.getElementType();
  if (mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(type))
    return type;
  if (mlir::Type tile = tileAsTensor(type))
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(tile))
      return shaped.getElementType();
  return {};
}

std::optional<llvm::SmallVector<int64_t, 4>> staticShapeOf(mlir::Type type) {
  if (!type)
    return std::nullopt;
  mlir::Type candidate = type;
  if (!mlir::isa<mlir::ShapedType>(candidate))
    candidate = tileAsTensor(type);
  if (!candidate)
    return std::nullopt;
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(candidate))
    if (shaped.hasStaticShape())
      return llvm::SmallVector<int64_t, 4>(shaped.getShape());
  return std::nullopt;
}

TileFacts tileFactsFor(mlir::Type valueType) {
  TileFacts facts;
  if (!valueType)
    return facts;
  // A `!micro.tile` is opaque to this core, so it is read through its printed
  // form; a shaped type is already what this needs.
  mlir::Type candidate = valueType;
  if (!mlir::isa<mlir::ShapedType>(candidate))
    candidate = tileAsTensor(valueType);
  if (!candidate)
    return facts;
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(candidate);
  if (!shaped || !shaped.hasStaticShape())
    return facts;
  std::optional<uint64_t> width = elementByteWidth(shaped.getElementType());
  if (!width || *width == 0)
    return facts;
  int64_t elements = shaped.getNumElements();
  if (elements < 0)
    return facts;
  // Checked: a shape whose byte count does not fit is reported unknown rather
  // than wrapping to a small, wrong size. An unknown footprint is not a smaller
  // one. LLVM's `MathExtras` overflow helpers are signed-only, so the builtin
  // is used directly.
  uint64_t bytes = 0;
  if (__builtin_mul_overflow(static_cast<uint64_t>(elements), *width, &bytes))
    return facts;
  facts.bytes = bytes;
  facts.alignment = *width;
  facts.known = true;
  return facts;
}

} // namespace mlir::llk::mapping
