//===- AVX2BundleLowering.cpp - AVX2 bundle lowering (issue #67, C2) ------===//
//
// The AVX2 target's lowering: what a *selected* bundle means for the code this
// backend emits.
//
// The split this file exists to make real is the one §18.3 draws. Verification
// asks "is this bundle complete?"; lowering asks "what code is it?". A target
// that can answer the first and not the second is a configuration file
// pretending to be a backend, which is exactly what the rest of the mapping
// core cannot tell apart on its own -- so `TargetEmitter::lower` rejects by
// default and only a target that overrides it here is executable.
//
// Everything in this file is AVX2 policy: the vector width a bundle selected,
// the element types this backend has arithmetic for, and the physical layout
// the lowered tile carries. None of it lives in the mapping core.
//
//===----------------------------------------------------------------------===//

#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

// The Micro attribute, type, and op classes. A target lowerer rewrites Micro
// operations, so the AVX2 package is where the dialect's own declarations are
// pulled in -- the generic mapping core never sees them.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

namespace mlir::llk::target::avx2 {
namespace {

llvm::Error avx2Error(std::string message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "avx2_lowering: " + std::move(message));
}

/// Renders a type for a diagnostic. `mlir::Type` has no Twine conversion, so a
/// message that names one goes through the printer.
std::string typeToString(mlir::Type type) {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  type.print(stream);
  return buffer;
}

/// The emitter keys whose lowering is the arithmetic tile-vector family. The
/// rest of the declared keys cover movement and contraction, which the
/// reference bridge still carries; claiming them here would emit nothing real.
bool isArithmeticKey(llvm::StringRef key) {
  return key == "avx2_vector_add" || key == "avx2_vector_convert" ||
         key == "avx2_vector_silu" || key == "avx2_vector_mul";
}

/// The element types the AVX2 arithmetic emitters implement. A target's rules
/// are dtype-parameterized, so a bundle can name an element type this backend
/// has no vector path for -- that is a target-readiness failure, not a parse
/// error, and it has to be reported rather than assumed away.
bool hasArithmeticPath(mlir::Type elementType) {
  return elementType.isF32() || elementType.isBF16();
}

/// The physical vector width a bundle selected.
///
/// The rule's `param VW` is persisted onto the bundle as an integer attribute,
/// so the width is read back from the *selected* plan rather than re-derived.
/// A bundle that omits it, or carries a width this backend cannot emit, did not
/// select a complete implementation.
llvm::Expected<int64_t>
resolveVectorWidth(const mapping::TargetBundle &bundle) {
  if (!bundle.parameters)
    return avx2Error(
        "bundle '" + bundle.name +
        "' carries no parameters, so no vector width was selected");
  mlir::Attribute raw = bundle.parameters.get("VW");
  if (!raw)
    return avx2Error("bundle '" + bundle.name +
                     "' declares no 'VW' parameter, so no vector width was "
                     "selected");
  auto width = mlir::dyn_cast<mlir::IntegerAttr>(raw);
  if (!width)
    return avx2Error("bundle parameter 'VW' is not an integer");
  int64_t value = width.getInt();
  if (value < 1 || value > 16 ||
      !llvm::isPowerOf2_64(static_cast<uint64_t>(value)))
    return avx2Error("vector width " + std::to_string(value) +
                     " is not a power of two in [1, 16]");
  return value;
}

/// The emitter a `MappingTarget` hands out for one key. It is the declared
/// shape check plus the AVX2 lowering; a key with no lowering implementation
/// still reports one, because the check and the implementation are different
/// promises.
class AVX2Emitter : public mapping::DeclaredTargetEmitter {
public:
  AVX2Emitter(std::string key, std::vector<std::string> declaredKeys)
      : DeclaredTargetEmitter(std::move(key), std::move(declaredKeys)) {}

  llvm::Error lower(llvm::ArrayRef<mlir::Operation *> coveredOps,
                    const mapping::TargetBundle &bundle,
                    const mapping::TargetLoweringContext &context,
                    mlir::RewriterBase &rewriter) const override;
};

} // namespace

llvm::Error AVX2Emitter::lower(llvm::ArrayRef<mlir::Operation *> coveredOps,
                               const mapping::TargetBundle &bundle,
                               const mapping::TargetLoweringContext &context,
                               mlir::RewriterBase &rewriter) const {
  // Validate the whole contract before rewriting a single operation. A bundle
  // this target cannot honor must leave `coveredOps` untouched rather than
  // half-lowered, so every rejection below happens before the first write.
  if (llvm::Error error = verify(bundle))
    return error;

  if (coveredOps.empty())
    return avx2Error("no covered operations to lower");

  if (!isArithmeticKey(key()))
    return avx2Error("emitter '" + key().str() +
                     "' has no lowering implementation for its covered "
                     "operations yet");

  llvm::Expected<int64_t> vectorWidth = resolveVectorWidth(bundle);
  if (!vectorWidth)
    return vectorWidth.takeError();

  // The selected plan placed this work on a vector engine. Lowering onto a
  // machine that offers none would emit for a resource the search never chose.
  bool hasVectorEngine = llvm::any_of(context.machine.computes,
                                      [](const machine::ComputeNode &node) {
                                        return node.kind == "vector_engine";
                                      });
  if (!hasVectorEngine)
    return avx2Error("machine offers no vector_engine for emitter '" +
                     key().str() + "'");

  for (mlir::Operation *op : coveredOps) {
    auto vector = mlir::dyn_cast<micro::VectorOp>(op);
    if (!vector)
      return avx2Error("covered operation '" +
                       op->getName().getStringRef().str() +
                       "' is not a micro.vector op");
    auto resultType =
        mlir::dyn_cast<micro::TileType>(vector.getResult().getType());
    if (!resultType)
      return avx2Error("micro.vector result is not a tile");
    if (!hasArithmeticPath(resultType.getElementType()))
      return avx2Error("element type '" +
                       typeToString(resultType.getElementType()) +
                       "' has no AVX2 arithmetic implementation");
  }

  // The contract holds. Apply it: the bundle's vector width becomes the
  // result tile's physical layout, which is the target-owned decision the
  // reference bridge is not allowed to make.
  for (mlir::Operation *op : coveredOps) {
    auto vector = mlir::cast<micro::VectorOp>(op);
    auto resultType = mlir::cast<micro::TileType>(vector.getResult().getType());

    micro::LayoutAttr loweredLayout = micro::LayoutAttr::get(
        rewriter.getContext(),
        static_cast<uint32_t>(micro::LayoutKind::vectorized),
        mlir::DenseI64ArrayAttr(), *vectorWidth, /*align=*/0, /*banks=*/0,
        mlir::StringAttr());
    micro::TileType loweredType =
        micro::TileType::get(rewriter.getContext(), resultType.getShape(),
                             resultType.getElementType(), loweredLayout,
                             resultType.getMemory(), resultType.getOwner());

    // The replacement takes the lowered op's place: the rewriter is told where
    // to build before it is asked to, so the new op lands where the old one was
    // rather than nowhere.
    rewriter.setInsertionPoint(vector);
    auto replacement = micro::VectorOp::create(
        rewriter, vector.getLoc(), loweredType, vector.getOp(),
        vector.getInputs(), vector.getMathModeAttr());
    rewriter.replaceOp(vector, replacement.getResult());
  }

  return llvm::Error::success();
}

std::unique_ptr<mapping::TargetEmitter> createAVX2Emitter(llvm::StringRef key) {
  std::vector<std::string> declared;
  for (llvm::StringLiteral declaredKey : emitterKeys())
    declared.push_back(declaredKey.str());
  return std::make_unique<AVX2Emitter>(key.str(), std::move(declared));
}

} // namespace mlir::llk::target::avx2
