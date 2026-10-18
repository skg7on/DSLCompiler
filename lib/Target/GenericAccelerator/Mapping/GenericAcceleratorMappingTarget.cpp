//===- GenericAcceleratorMappingTarget.cpp - second target (D7) -----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.h"

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"

#include "mlir/IR/Operation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"

namespace mlir::llk::target::generic_accel {
namespace {

bool isF32Tile(mlir::Type type) {
  auto tile = mlir::dyn_cast<mlir::micro::TileType>(type);
  return tile && tile.getElementType().isF32();
}

/// The accelerator's emitters only verify: declaring a key is not implementing
/// a backend, and no hardware execution is claimed. They add the one check a
/// bundle cannot express -- the covered operation's variant and f32 tiles.
class GenericAcceleratorEmitter : public mapping::DeclaredTargetEmitter {
public:
  GenericAcceleratorEmitter(std::string key, std::vector<std::string> declared)
      : DeclaredTargetEmitter(std::move(key), std::move(declared)) {}

  llvm::Error
  verifyOperations(llvm::ArrayRef<mlir::Operation *> coveredOps,
                   const mapping::TargetBundle &bundle) const override;
};

llvm::Error GenericAcceleratorEmitter::verifyOperations(
    llvm::ArrayRef<mlir::Operation *> coveredOps,
    const mapping::TargetBundle &bundle) const {
  if (llvm::Error error = verify(bundle))
    return error;

  auto fail = [&](const llvm::Twine &detail) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "accelerator coverage: emitter '" +
                                       key().str() + "': " + detail);
  };
  const bool multiply = key() == "accel_vector_mul";
  const bool reduceSum = key() == "accel_reduce_sum";
  if (!multiply && !reduceSum)
    return llvm::Error::success();
  if (coveredOps.empty())
    return fail("no covered operations");

  for (mlir::Operation *op : coveredOps) {
    llvm::StringRef name = op->getName().getStringRef();
    mlir::StringAttr variant = op->getAttrOfType<mlir::StringAttr>("op");
    if (multiply) {
      if (name != "micro.vector")
        return fail(llvm::Twine("expected micro.vector, got '") + name + "'");
      if (!variant || variant.getValue() != "mul")
        return fail("expected the 'mul' variant");
      for (mlir::Value operand : op->getOperands())
        if (!isF32Tile(operand.getType()))
          return fail("operands must be f32 tiles");
      if (op->getNumResults() != 1 || !isF32Tile(op->getResult(0).getType()))
        return fail("result must be an f32 tile");
      continue;
    }
    if (name != "micro.reduce")
      return fail(llvm::Twine("expected micro.reduce, got '") + name + "'");
    if (!variant || variant.getValue() != "sum")
      return fail("expected the 'sum' variant");
    if (op->getNumOperands() != 1 || !isF32Tile(op->getOperand(0).getType()))
      return fail("operand must be an f32 tile");
    if (op->getNumResults() != 1 || !isF32Tile(op->getResult(0).getType()))
      return fail("result must be an f32 tile");
  }
  return llvm::Error::success();
}

std::unique_ptr<mapping::TargetEmitter>
createGenericAcceleratorEmitter(llvm::StringRef key) {
  std::vector<std::string> declared;
  for (llvm::StringRef declaredKey : emitterKeys())
    declared.push_back(declaredKey.str());
  return std::make_unique<GenericAcceleratorEmitter>(key.str(),
                                                     std::move(declared));
}

/// The accelerator target wraps the loaded configuration and supplies the
/// plugin's own emitters. It deliberately has no codegen contract: this target
/// is analysis-only, and selected execution stays unsupported.
class GenericAcceleratorTarget : public mapping::MappingTarget {
public:
  explicit GenericAcceleratorTarget(
      std::unique_ptr<mapping::MappingTarget> configuration)
      : configuration_(std::move(configuration)) {}

  llvm::StringRef name() const override { return configuration_->name(); }
  const machine::MachineModel &machine() const override {
    return configuration_->machine();
  }
  const mapping::LayoutRegistry &layouts() const override {
    return configuration_->layouts();
  }
  const mapping::RuleRegistry &rules() const override {
    return configuration_->rules();
  }
  bool isKnownEmitter(llvm::StringRef key) const override {
    return configuration_->isKnownEmitter(key);
  }
  const mapping::LatencyProvider *latencyProvider() const override {
    return configuration_->latencyProvider();
  }
  std::unique_ptr<mapping::TargetEmitter>
  createEmitter(llvm::StringRef key) const override {
    if (!configuration_->isKnownEmitter(key))
      return nullptr;
    return createGenericAcceleratorEmitter(key);
  }
  std::unique_ptr<mapping::TargetEmitter> createEmitter() const override {
    llvm::ArrayRef<llvm::StringRef> keys = emitterKeys();
    if (keys.empty())
      return nullptr;
    return createEmitter(keys.front());
  }

private:
  std::unique_ptr<mapping::MappingTarget> configuration_;
};

} // namespace

llvm::ArrayRef<llvm::StringRef> emitterKeys() {
  static const llvm::StringRef kKeys[] = {"accel_vector_add",
                                          "accel_vector_mul", "accel_mxu",
                                          "accel_copy", "accel_reduce_sum"};
  return kKeys;
}

llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath) {
  std::string root = configurationRoot.str();
  std::string machine = machinePath.empty()
                            ? root + "/machines/generic-ai-accel-v2.yaml"
                            : machinePath.str();
  std::vector<std::string> keys;
  for (llvm::StringRef key : emitterKeys())
    keys.push_back(key.str());
  llvm::Expected<std::unique_ptr<mapping::MappingTarget>> configuration =
      mapping::loadMappingTarget(
          "generic-ai-accel", machine,
          root + "/mapping/generic-ai-accel/layouts.llkmap",
          root + "/mapping/generic-ai-accel/rules.llkmap", std::move(keys));
  if (!configuration)
    return configuration.takeError();
  return std::make_unique<GenericAcceleratorTarget>(std::move(*configuration));
}

} // namespace mlir::llk::target::generic_accel
