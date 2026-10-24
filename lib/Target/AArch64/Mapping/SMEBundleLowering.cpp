//===- SMEBundleLowering.cpp - Selected SME bundle lowering --------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The SME target's lowering boundary. Verification asks whether a bundle is
// complete; these emitters answer what code the selected group is. The f32
// 16x16x16 initialized contraction and the fixture's movement keys are the
// only supported signatures. Movement groups are carried by the generic
// Micro-to-Linalg and Linalg-to-loops bridge; the emitter's claim keeps them
// selected rather than reference-lowered. Every other key keeps the base
// emitter's explicit rejection.

#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"

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

#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

namespace mlir::llk::target::sme {
namespace {

llvm::Error smeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "sme_lowering_unsupported: " + message);
}

bool isMovementKey(llvm::StringRef key) {
  return key == "sme_copy" || key == "sme_tile_copy" || key == "sme_tile_store";
}

class SMEEmitter : public mapping::DeclaredTargetEmitter {
public:
  SMEEmitter(std::string key, std::vector<std::string> declaredKeys)
      : DeclaredTargetEmitter(std::move(key), std::move(declaredKeys)) {}

  bool hasLowering() const override {
    return key() == "sme_mma_za" || isMovementKey(key());
  }

  bool
  hasLoweringFor(llvm::ArrayRef<mlir::Operation *> coveredOps) const override {
    if (!hasLowering() || coveredOps.empty())
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
    if (llvm::Error error = validateGroup(coveredOps))
      return error;
    for (mlir::Operation *op : coveredOps)
      op->setAttr("llk.sme.selected_handoff",
                  mlir::UnitAttr::get(rewriter.getContext()));
    return llvm::Error::success();
  }

private:
  llvm::Error
  validateGroup(llvm::ArrayRef<mlir::Operation *> coveredOps) const {
    if (coveredOps.empty())
      return smeError("no covered operations to lower");
    if (isMovementKey(key())) {
      llvm::StringRef expected = key() == "sme_copy" ? "micro.async_copy"
                                 : key() == "sme_tile_copy"
                                     ? "micro.tile_async_copy"
                                     : "micro.tile_store";
      for (mlir::Operation *op : coveredOps)
        if (op->getName().getStringRef() != expected)
          return smeError("emitter '" + key() +
                          "' cannot lower covered operation '" +
                          op->getName().getStringRef() + "'");
      return llvm::Error::success();
    }

    if (coveredOps.size() != 1)
      return smeError("the contraction group must cover exactly one micro.mma");
    auto mma = mlir::dyn_cast<micro::MmaOp>(coveredOps.front());
    if (!mma)
      return smeError("emitter 'sme_mma_za' needs a micro.mma");
    if (mma.getShape() != llvm::ArrayRef<int64_t>{16, 16, 16})
      return smeError("emitter 'sme_mma_za' supports the 16x16x16 shape only");
    if (mma.getInput() != micro::DType::f32 ||
        mma.getAccumulator() != micro::DType::f32)
      return smeError("emitter 'sme_mma_za' supports f32 inputs and "
                      "accumulator only");
    auto result = mlir::dyn_cast<micro::TileType>(mma.getResult().getType());
    if (!result || result.getShape() != llvm::ArrayRef<int64_t>{16, 16} ||
        !result.getElementType().isF32())
      return smeError("emitter 'sme_mma_za' needs a static 16x16 f32 tile "
                      "result");
    return llvm::Error::success();
  }
};

} // namespace

std::unique_ptr<mapping::TargetEmitter> createSMEEmitter(llvm::StringRef key) {
  std::vector<std::string> declared;
  for (llvm::StringRef declaredKey : emitterKeys())
    declared.push_back(declaredKey.str());
  return std::make_unique<SMEEmitter>(key.str(), std::move(declared));
}

} // namespace mlir::llk::target::sme
