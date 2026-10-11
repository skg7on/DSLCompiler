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

/// The emitter whose bundle is a *group*: one rule implementing a whole matched
/// subgraph. Its lowering applies the same target-owned physical decision to
/// every operation the group covers, which is why it takes the whole covered
/// array rather than one operation -- a fused bundle that rewrote only its
/// anchor would leave the rest of the match at the reference layout.
bool isFusedKey(llvm::StringRef key) {
  return key == "avx2_fused_convert_silu_mul";
}

bool isBackendHandoffKey(llvm::StringRef key) {
  return key == "avx2_mma" || key == "avx2_reduce" || key == "avx2_copy" ||
         key == "avx2_tile_copy" || key == "avx2_tile_store";
}

mlir::ArrayAttr
encodeSelectedConnections(const mapping::TargetLoweringContext &context,
                          mlir::MLIRContext *mlirContext) {
  mlir::Builder builder(mlirContext);
  llvm::SmallVector<mlir::Attribute> encoded;
  for (const mapping::PlanConnection &connection : context.connections) {
    mlir::NamedAttrList record;
    record.set("id", builder.getI64IntegerAttr(connection.id));
    record.set("value", builder.getI64IntegerAttr(connection.value));
    record.set("kind", builder.getStringAttr(
                           mapping::stringifyConnectionKind(connection.kind)));
    auto encodeIds = [&](llvm::ArrayRef<uint64_t> ids) {
      llvm::SmallVector<mlir::Attribute> values;
      for (uint64_t id : ids)
        values.push_back(builder.getI64IntegerAttr(id));
      return builder.getArrayAttr(values);
    };
    llvm::SmallVector<mlir::Attribute> route;
    for (const mapping::MemoryNodeId &memory : connection.route)
      route.push_back(builder.getStringAttr(memory));
    record.set("route", builder.getArrayAttr(route));
    record.set("storage_ids", encodeIds(connection.storageIds));
    llvm::SmallVector<mlir::Attribute> engines;
    for (const mapping::ExecutorId &engine : connection.engines)
      engines.push_back(builder.getStringAttr(engine));
    record.set("engines", builder.getArrayAttr(engines));
    llvm::SmallVector<mlir::Attribute> hops;
    for (const mapping::PlanMovementHop &hop : connection.hops) {
      mlir::NamedAttrList movement;
      movement.set("index", builder.getI64IntegerAttr(hop.index));
      movement.set("src_memory", builder.getStringAttr(hop.srcMemory));
      movement.set("dst_memory", builder.getStringAttr(hop.dstMemory));
      movement.set("engine", builder.getStringAttr(hop.engine));
      movement.set("source_storage_id",
                   builder.getI64IntegerAttr(hop.sourceStorageId));
      movement.set("destination_storage_id",
                   builder.getI64IntegerAttr(hop.destinationStorageId));
      movement.set("movement_step",
                   builder.getI64IntegerAttr(hop.movementStep));
      movement.set("wait_step", builder.getI64IntegerAttr(hop.waitStep));
      hops.push_back(builder.getDictionaryAttr(movement));
    }
    record.set("hops", builder.getArrayAttr(hops));
    encoded.push_back(builder.getDictionaryAttr(record));
  }
  return builder.getArrayAttr(encoded);
}

/// The element types the AVX2 arithmetic emitters implement. A target's rules
/// are dtype-parameterized, so a bundle can name an element type this backend
/// has no vector path for -- that is a target-readiness failure, not a parse
/// error, and it has to be reported rather than assumed away.
bool hasArithmeticPath(mlir::Type elementType) {
  return elementType.isF32() || elementType.isBF16();
}

/// Checks a candidate width and returns it, or reports why this backend cannot
/// emit it.
llvm::Expected<int64_t> checkVectorWidth(mlir::Attribute raw,
                                         llvm::StringRef source) {
  auto width = mlir::dyn_cast<mlir::IntegerAttr>(raw);
  if (!width)
    return avx2Error(
        ("vector width from " + source + " is not an integer").str());
  int64_t value = width.getInt();
  if (value < 1 || value > 16 ||
      !llvm::isPowerOf2_64(static_cast<uint64_t>(value)))
    return avx2Error("vector width " + std::to_string(value) +
                     " is not a power of two in [1, 16]");
  return value;
}

/// The physical vector width the selected implementation uses.
///
/// The width is a property of the *layout* the search solved: the rule requires
/// its operand to satisfy a layout whose `VW` parameter the solver fixed, and
/// the binder persists that solution on the operation. Reading it from there is
/// what makes the lowering follow the selection -- a bundle that happened to
/// carry a width of its own would otherwise override the solved layout.
///
/// A bundle that does carry `VW` is still honoured, because that is the shape a
/// bundle with parameters has; the layout solution is the fallback that the
/// real pipeline always takes.
llvm::Expected<int64_t> resolveVectorWidth(const mapping::TargetBundle &bundle,
                                           mlir::Operation *op) {
  std::optional<int64_t> bundleWidth;
  if (bundle.parameters)
    if (mlir::Attribute raw = bundle.parameters.get("VW")) {
      llvm::Expected<int64_t> width =
          checkVectorWidth(raw, "bundle '" + bundle.name + "'");
      if (!width)
        return width.takeError();
      bundleWidth = *width;
    }

  auto mapping = op->getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
  auto solutions =
      mapping ? mapping.getAs<mlir::DictionaryAttr>("layout_parameters")
              : mlir::DictionaryAttr();
  std::optional<int64_t> layoutWidth;
  if (solutions) {
    for (mlir::NamedAttribute entry : solutions) {
      auto solution = mlir::dyn_cast<mlir::DictionaryAttr>(entry.getValue());
      if (!solution)
        continue;
      if (mlir::Attribute raw = solution.get("VW")) {
        llvm::Expected<int64_t> width =
            checkVectorWidth(raw, "layout '" + entry.getName().str() + "'");
        if (!width)
          return width.takeError();
        if (layoutWidth && *layoutWidth != *width)
          return avx2Error("selected layouts disagree on vector width");
        layoutWidth = *width;
      }
    }
  }

  if (bundleWidth && layoutWidth && *bundleWidth != *layoutWidth)
    return avx2Error(
        "bundle vector width conflicts with selected layout width");
  if (bundleWidth)
    return *bundleWidth;
  if (layoutWidth)
    return *layoutWidth;

  return avx2Error(
      "no vector width was selected: bundle '" + bundle.name +
      "' carries none and the operation records no solved layout that has one");
}

/// The emitter a `MappingTarget` hands out for one key. It is the declared
/// shape check plus the AVX2 lowering; a key with no lowering implementation
/// still reports one, because the check and the implementation are different
/// promises.
class AVX2Emitter : public mapping::DeclaredTargetEmitter {
public:
  AVX2Emitter(std::string key, std::vector<std::string> declaredKeys)
      : DeclaredTargetEmitter(std::move(key), std::move(declaredKeys)) {}

  /// Only the arithmetic family has a lowering here; the movement and
  /// contraction keys are declared because rules emit them, and the reference
  /// bridge carries those until this target implements them.
  bool hasLowering() const override {
    return isArithmeticKey(key()) || isFusedKey(key()) ||
           isBackendHandoffKey(key());
  }

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

  // These selected operations are consumed by the AVX2 Vector backend pass or
  // by the host ABI's explicit movement lowering. Keep their Micro form and
  // provenance intact until that consumer runs; the handoff is not a claim of
  // native matrix or accelerator-memory instructions.
  if (isBackendHandoffKey(key())) {
    for (mlir::Operation *op : coveredOps) {
      llvm::StringRef name = op->getName().getStringRef();
      bool supported =
          (key() == "avx2_mma" && name == "micro.mma") ||
          (key() == "avx2_reduce" && name == "micro.reduce") ||
          (key() == "avx2_copy" && name == "micro.async_copy") ||
          (key() == "avx2_tile_copy" && name == "micro.tile_async_copy") ||
          (key() == "avx2_tile_store" && name == "micro.tile_store");
      if (!supported)
        return avx2Error("emitter '" + key().str() +
                         "' cannot hand off covered operation '" + name.str() +
                         "'");
    }
    bool needsVectorWidth = key() == "avx2_mma" || key() == "avx2_reduce";
    if (needsVectorWidth) {
      llvm::Expected<int64_t> width =
          resolveVectorWidth(bundle, coveredOps.front());
      if (!width)
        return width.takeError();
    }
    for (mlir::Operation *op : coveredOps) {
      op->setAttr("llk.avx2.selected_handoff",
                  mlir::UnitAttr::get(rewriter.getContext()));
      if (key() == "avx2_copy" || key() == "avx2_tile_copy" ||
          key() == "avx2_tile_store")
        op->setAttr("micro.selected_connections",
                    encodeSelectedConnections(context, rewriter.getContext()));
    }
    return llvm::Error::success();
  }

  if (!isArithmeticKey(key()) && !isFusedKey(key()))
    return avx2Error("emitter '" + key().str() +
                     "' has no lowering implementation for its covered "
                     "operations yet");

  // The selected width is a property of the operation's solved layout, so it
  // is read from the operation rather than guessed from the bundle name.
  llvm::Expected<int64_t> vectorWidth =
      resolveVectorWidth(bundle, coveredOps.front());
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

  if (isFusedKey(key())) {
    if (coveredOps.size() != 3)
      return avx2Error("fused convert-silu-mul lowering requires exactly three "
                       "covered operations");
    auto convert = mlir::dyn_cast<micro::VectorOp>(coveredOps[0]);
    auto silu = mlir::dyn_cast<micro::VectorOp>(coveredOps[1]);
    auto multiply = mlir::dyn_cast<micro::VectorOp>(coveredOps[2]);
    if (!convert || !silu || !multiply || convert.getOp() != "convert" ||
        silu.getOp() != "silu" || multiply.getOp() != "mul")
      return avx2Error("fused lowering requires the ordered convert -> silu -> "
                       "mul operation group");
    if (convert.getInputs().size() != 1 || silu.getInputs().size() != 1 ||
        multiply.getInputs().size() != 2 ||
        silu.getInputs().front() != convert.getResult() ||
        multiply.getInputs().front() != silu.getResult())
      return avx2Error("fused lowering requires connected convert -> silu -> "
                       "mul dataflow");

    auto sourceTile =
        mlir::dyn_cast<micro::TileType>(convert.getInputs().front().getType());
    auto convertedTile =
        mlir::dyn_cast<micro::TileType>(convert.getResult().getType());
    if (!sourceTile || !convertedTile ||
        !convertedTile.getElementType().isF32() ||
        !(sourceTile.getElementType().isF32() ||
          sourceTile.getElementType().isBF16()))
      return avx2Error("fused convert supports only f32 -> f32 and bf16 -> "
                       "f32 conversions");

    llvm::SmallPtrSet<mlir::Operation *, 4> members;
    for (mlir::Operation *op : coveredOps)
      members.insert(op);
    for (mlir::Value intermediate : {convert.getResult(), silu.getResult()}) {
      bool internalUse = false;
      bool externalUse = false;
      for (mlir::OpOperand &use : intermediate.getUses()) {
        internalUse |= members.contains(use.getOwner());
        externalUse |= !members.contains(use.getOwner());
      }
      if (internalUse && externalUse)
        return avx2Error("fused intermediate has a live external use");
    }
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
    // The target-owned physical rewrite must retain the selected instance and
    // bundle provenance for the later AVX2 Linalg/Vector backend pass.
    replacement->setAttrs(vector->getAttrs());
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
