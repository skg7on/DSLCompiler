//===- TargetCoverageValidation.cpp - Validate covered operations --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetCoverageValidation.h"

#include "LLK/Mapping/MappingMetadata.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace mlir::llk::mapping {
namespace {

llvm::Error coverageError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// Reads the selected bundle the binder recorded on a covered operation. A
/// mapped operation without one has lost part of its selected state.
llvm::Expected<TargetBundle> readSelectedBundle(mlir::Operation *op) {
  mlir::DictionaryAttr mapping =
      op->getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
  if (!mapping)
    return coverageError("operation '" + op->getName().getStringRef() +
                         "' carries no micro.mapping metadata");
  llvm::Expected<std::string> name =
      readMetadataString(mapping, "bundle", "micro.mapping");
  if (!name)
    return name.takeError();
  llvm::Expected<std::string> emitter =
      readMetadataString(mapping, "emitter", "micro.mapping");
  if (!emitter)
    return emitter.takeError();
  TargetBundle bundle;
  bundle.name = *name;
  bundle.emitterKey = *emitter;
  bundle.parameters = mapping.getAs<mlir::DictionaryAttr>("bundle_parameters");
  return bundle;
}

bool sameBundle(const TargetBundle &lhs, const TargetBundle &rhs) {
  return lhs.name == rhs.name && lhs.emitterKey == rhs.emitterKey &&
         lhs.parameters == rhs.parameters;
}

} // namespace

llvm::Error verifyTargetOperationCoverage(mlir::ModuleOp bound,
                                          const CoveringPlan &plan,
                                          const MappingTarget &target) {
  std::map<WorkloadNodeId, const PlanPlacement *> placementForNode;
  std::map<InstanceId, unsigned> expectedOperations;
  for (const PlanPlacement &placement : plan.placements) {
    if (!placementForNode.emplace(placement.node, &placement).second)
      return coverageError("selected plan repeats a placement for node " +
                           llvm::Twine(placement.node));
    ++expectedOperations[placement.instance];
  }

  struct Group {
    const PlanPlacement *placement = nullptr;
    llvm::SmallVector<mlir::Operation *, 4> operations;
  };
  std::map<InstanceId, Group> groups;
  std::map<WorkloadNodeId, mlir::Operation *> operationForNode;
  std::string error;

  bound.walk([&](mlir::Operation *op) {
    if (!error.empty() || !op->hasAttr("micro.mapping"))
      return;
    mlir::DictionaryAttr mapping =
        op->getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
    if (!mapping) {
      error = "operation '" + op->getName().getStringRef().str() +
              "' has malformed micro.mapping metadata";
      return;
    }
    mlir::IntegerAttr nodeAttr = mapping.getAs<mlir::IntegerAttr>("node");
    mlir::IntegerAttr instanceAttr =
        mapping.getAs<mlir::IntegerAttr>("instance");
    if (!nodeAttr || !instanceAttr ||
        nodeAttr.getValue().getActiveBits() > 64 ||
        instanceAttr.getValue().getActiveBits() > 64) {
      error = "operation '" + op->getName().getStringRef().str() +
              "' has no valid selected node and instance identity";
      return;
    }
    WorkloadNodeId node = nodeAttr.getValue().getZExtValue();
    InstanceId instance = instanceAttr.getValue().getZExtValue();
    auto placement = placementForNode.find(node);
    if (placement == placementForNode.end() ||
        placement->second->instance != instance) {
      error = "selected instance " + std::to_string(instance) +
              " has no matching plan placement for node " +
              std::to_string(node);
      return;
    }
    if (!operationForNode.emplace(node, op).second) {
      error = "selected instance " + std::to_string(instance) +
              " repeats mapped node " + std::to_string(node);
      return;
    }
    llvm::Expected<TargetBundle> bundle = readSelectedBundle(op);
    if (!bundle) {
      error = llvm::toString(bundle.takeError());
      return;
    }
    if (!sameBundle(*bundle, placement->second->bundle)) {
      error = "selected instance " + std::to_string(instance) + " rule '" +
              placement->second->rule +
              "' has bundle metadata that disagrees with its plan";
      return;
    }
    auto [group, inserted] = groups.try_emplace(instance);
    if (inserted) {
      group->second.placement = placement->second;
      group->second.operations.push_back(op);
      return;
    }
    if (!sameBundle(*bundle, group->second.placement->bundle) ||
        group->second.placement->rule != placement->second->rule) {
      error = "selected instance " + std::to_string(instance) +
              " has inconsistent bundle or rule metadata";
      return;
    }
    group->second.operations.push_back(op);
  });

  if (!error.empty())
    return coverageError(error);

  if (operationForNode.size() != placementForNode.size())
    for (const auto &[node, placement] : placementForNode)
      if (!operationForNode.count(node))
        return coverageError("selected instance " +
                             llvm::Twine(placement->instance) + " rule '" +
                             placement->rule + "' is missing mapped node " +
                             llvm::Twine(node));

  for (const auto &[instance, group] : groups) {
    if (group.operations.size() != expectedOperations[instance])
      return coverageError("selected instance " + llvm::Twine(instance) +
                           " rule '" + group.placement->rule +
                           "' has a different number of mapped operations "
                           "and plan placements");
    std::unique_ptr<TargetEmitter> emitter =
        target.createEmitter(group.placement->bundle.emitterKey);
    if (!emitter)
      return coverageError(
          "selected emitter '" + group.placement->bundle.emitterKey +
          "' is not declared by target '" + target.name().str() + "'");
    if (llvm::Error failure = emitter->verifyOperations(
            group.operations, group.placement->bundle))
      return coverageError("selected instance " + llvm::Twine(instance) +
                           " rule '" + group.placement->rule +
                           "' coverage validation failed: " +
                           llvm::toString(std::move(failure)));
  }
  return llvm::Error::success();
}

} // namespace mlir::llk::mapping
