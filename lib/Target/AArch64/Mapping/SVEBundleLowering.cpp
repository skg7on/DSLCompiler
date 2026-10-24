//===- SVEBundleLowering.cpp - Selected SVE bundle lowering --------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The SVE target's lowering: what a *selected* bundle means for the code this
// backend emits.
//
// Verification asks "is this bundle complete?"; lowering asks "what code is
// it?". Only the f32 vector add/multiply family has an implementation here.
// Those two emitters do not rewrite the covered Micro operation: they validate
// the whole bundle and the covered operation group, then mark it as handed off
// so the selected backend pass that runs after the generic Micro to Linalg
// conversion can realize it as scalable Vector IR. Every other declared key
// keeps the base emitter's explicit `target_lowering_unsupported` rejection, so
// a rule file cannot impersonate an accelerator backend.

#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include "LLK/Dialect/Micro/MicroDialect.h" // IWYU pragma: keep
#include "LLK/Dialect/Micro/MicroEnums.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

// The Micro attribute, type, and op classes. A target lowerer rewrites Micro
// operations, so the SVE package is where the dialect's own declarations are
// pulled in -- the generic mapping core never sees them.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

namespace mlir::llk::target::sve {
namespace {

llvm::Error sveError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "sve_lowering_unsupported: " + message);
}

/// The emitter a `MappingTarget` hands out for one key. It is the declared
/// shape check plus this target's lowering; a key with no lowering still
/// reports one, because the check and the implementation are different
/// promises.
class SVEEmitter : public mapping::DeclaredTargetEmitter {
public:
  SVEEmitter(std::string key, std::vector<std::string> declaredKeys)
      : DeclaredTargetEmitter(std::move(key), std::move(declaredKeys)) {}

  bool hasLowering() const override {
    return key() == "sve_vector_add" || key() == "sve_vector_mul";
  }

  /// The executable subset is not the whole key: `sve_vector_add` and
  /// `sve_vector_mul` cover f32, bf16 and i8 rules, but this backend only
  /// implements the rank-two f32 signature. Reporting the key alone would
  /// make a portable reference compilation call `lower` and fail on every
  /// other dtype, so the per-group question is answered per group.
  bool
  hasLoweringFor(llvm::ArrayRef<mlir::Operation *> coveredOps) const override {
    if (!hasLowering())
      return false;
    // An empty group is not lowerable: `lower({})` rejects it, and this query
    // must answer the same question as `lower` rather than the opposite.
    if (coveredOps.empty())
      return false;
    return !static_cast<bool>(validateGroup(coveredOps));
  }

  llvm::Error lower(llvm::ArrayRef<mlir::Operation *> coveredOps,
                    const mapping::TargetBundle &bundle,
                    const mapping::TargetLoweringContext &context,
                    mlir::RewriterBase &rewriter) const override {
    if (llvm::Error error = verify(bundle))
      return error;
    if (!hasLowering())
      return DeclaredTargetEmitter::lower(coveredOps, bundle, context,
                                          rewriter);
    // Validate the whole group before the first write. A group this target
    // cannot honor must leave `coveredOps` untouched rather than partly
    // marked as handed off.
    if (llvm::Error error = validateGroup(coveredOps))
      return error;
    for (mlir::Operation *op : coveredOps)
      op->setAttr("llk.sve.selected_handoff",
                  mlir::UnitAttr::get(rewriter.getContext()));
    return llvm::Error::success();
  }

private:
  /// Checks that `coveredOps` is exactly the signature this key implements: a
  /// `micro.vector` add/multiply on a rank-two f32 tile. Pure; it is both the
  /// capability query and `lower`'s pre-write validation.
  llvm::Error
  validateGroup(llvm::ArrayRef<mlir::Operation *> coveredOps) const {
    if (coveredOps.empty())
      return sveError("no covered operations to lower");
    llvm::StringRef expected = key() == "sve_vector_add" ? "add" : "mul";
    for (mlir::Operation *op : coveredOps) {
      auto vector = mlir::dyn_cast<micro::VectorOp>(op);
      if (!vector || vector.getOp() != expected)
        return sveError("emitter '" + key() +
                        "' cannot lower covered "
                        "operation '" +
                        op->getName().getStringRef() + "'");
      auto result =
          mlir::dyn_cast<micro::TileType>(vector.getResult().getType());
      if (!result)
        return sveError("emitter '" + key() + "' needs a tile result");
      // The backend pass requires a static rank-two shape; agreeing here keeps
      // a group the emitter accepts from being skipped as unrealized later.
      if (result.getShape().size() != 2 ||
          mlir::ShapedType::isDynamicShape(result.getShape()))
        return sveError("emitter '" + key() +
                        "' supports static rank-two tiles only");
      if (!result.getElementType().isF32())
        return sveError("emitter '" + key() + "' supports f32 tiles only");
    }
    return llvm::Error::success();
  }
};

} // namespace

std::unique_ptr<mapping::TargetEmitter> createSVEEmitter(llvm::StringRef key) {
  std::vector<std::string> declared;
  for (llvm::StringRef declaredKey : emitterKeys())
    declared.push_back(declaredKey.str());
  return std::make_unique<SVEEmitter>(key.str(), std::move(declared));
}

} // namespace mlir::llk::target::sve
