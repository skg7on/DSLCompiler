//===- MappedCompilation.cpp - Compile a selected mapping plan (C5) -------===//
//
// The shared sequence, in the order the stages depend on each other:
//
//   bind (clone + materialize + persist selected state)
//     -> verify the mapped IR against the target
//       -> let the target's emitters rewrite what they implement
//         -> MicroToLinalg, bufferize, roll into loops
//           -> apply the calling convention and compile
//
// Each step refuses to guess about the next one. The binder will not invent a
// connection it cannot materialize, verification will not resolve an id the
// target does not declare, and the emitters are asked whether they lower before
// they are called -- so "the target ran this kernel" and "the reference bridge
// ran this kernel" stay different statements.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MappedCompilation.h"

#include "LLK/Conversion/MicroToLinalg.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/PlanBinder.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

// The canonical plan materializer lives beside this file, in the one library
// that owns the Micro dialect on the mapping path.
#include "MicroMappingCommon.h"
#include "selected_group_lowering_test_hooks.h"

#if LLVM_VERSION_MAJOR >= 21
using BufOpts = mlir::bufferization::OneShotBufferizePassOptions;
#else
using BufOpts = mlir::bufferization::OneShotBufferizationOptions;
#endif

namespace llk {

namespace {

using mlir::llk::mapping::BindContract;
using mlir::llk::mapping::MappingTarget;
using mlir::llk::mapping::PlanMaterializer;
using mlir::llk::mapping::TargetBundle;
using mlir::llk::mapping::TargetLoweringContext;

llvm::Error compileError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// Reads the selected bundle a covered operation carries.
///
/// The mapping metadata is the only record of what the search chose, so an
/// operation with no bundle is not "nothing to lower here" -- it is a mapped
/// kernel that lost part of its plan, and saying so is better than carrying it
/// silently.
llvm::Expected<TargetBundle> readSelectedBundle(mlir::Operation *op) {
  mlir::DictionaryAttr mapping =
      op->getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
  if (!mapping)
    return compileError("operation '" + op->getName().getStringRef().str() +
                        "' carries no micro.mapping, so its selected bundle is "
                        "not recorded");

  TargetBundle bundle;
  llvm::Expected<std::string> name = mlir::llk::mapping::readMetadataString(
      mapping, "bundle", "micro.mapping");
  if (!name)
    return name.takeError();
  bundle.name = *name;

  llvm::Expected<std::string> emitter = mlir::llk::mapping::readMetadataString(
      mapping, "emitter", "micro.mapping");
  if (!emitter)
    return emitter.takeError();
  bundle.emitterKey = *emitter;

  bundle.parameters = mapping.getAs<mlir::DictionaryAttr>("bundle_parameters");
  return bundle;
}

/// Lets the target's emitters rewrite the operations they implement.
///
/// An emitter that only verifies its bundle is left alone, and the reference
/// bridge lowers its operation later. That is a real difference and it is
/// counted rather than glossed: a compilation whose operations were all carried
/// by the bridge has said nothing about this target's code generation.
llvm::Error
lowerSelectedOperations(mlir::ModuleOp module, const MappingTarget &target,
                        const mlir::llk::mapping::CoveringPlan &plan,
                        MappedCompilation &compilation) {
  using namespace mlir::llk::mapping;
  struct SelectedGroup {
    InstanceId instance = 0;
    TargetBundle bundle;
    std::string rule;
    llvm::SmallVector<mlir::Operation *> operations;
    llvm::SmallVector<PlanPlacement, 1> placements;
    llvm::SmallVector<PlanConnection, 1> connections;
    std::unique_ptr<TargetEmitter> emitter;
  };

  std::map<WorkloadNodeId, const PlanPlacement *> placementForNode;
  std::map<InstanceId, std::vector<const PlanPlacement *>>
      placementsForInstance;
  for (const PlanPlacement &placement : plan.placements) {
    if (!placementForNode.emplace(placement.node, &placement).second)
      return compileError("selected plan repeats placement for node " +
                          std::to_string(placement.node));
    placementsForInstance[placement.instance].push_back(&placement);
  }

  std::map<InstanceId, SelectedGroup> groups;
  std::map<WorkloadNodeId, mlir::Operation *> operationForNode;
  std::string metadataError;
  module.walk([&](mlir::Operation *op) {
    if (!metadataError.empty() || !op->hasAttr("micro.mapping"))
      return;
    auto mapping = op->getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
    if (!mapping) {
      metadataError = "operation '" + op->getName().getStringRef().str() +
                      "' has malformed micro.mapping metadata";
      return;
    }
    auto nodeAttr = mapping.getAs<mlir::IntegerAttr>("node");
    auto instanceAttr = mapping.getAs<mlir::IntegerAttr>("instance");
    if (!nodeAttr || !instanceAttr ||
        nodeAttr.getValue().getActiveBits() > 64 ||
        instanceAttr.getValue().getActiveBits() > 64) {
      metadataError = "operation '" + op->getName().getStringRef().str() +
                      "' has no valid selected node and instance identity";
      return;
    }
    WorkloadNodeId node = nodeAttr.getValue().getZExtValue();
    InstanceId instance = instanceAttr.getValue().getZExtValue();
    auto placement = placementForNode.find(node);
    if (placement == placementForNode.end() ||
        placement->second->instance != instance) {
      metadataError = "selected instance " + std::to_string(instance) +
                      " has no matching plan placement for node " +
                      std::to_string(node);
      return;
    }
    if (!operationForNode.emplace(node, op).second) {
      metadataError = "selected instance " + std::to_string(instance) +
                      " repeats mapped node " + std::to_string(node);
      return;
    }
    llvm::Expected<TargetBundle> bundle = readSelectedBundle(op);
    if (!bundle) {
      metadataError = llvm::toString(bundle.takeError());
      return;
    }
    const TargetBundle &selected = placement->second->bundle;
    if (bundle->name != selected.name ||
        bundle->emitterKey != selected.emitterKey ||
        bundle->parameters != selected.parameters) {
      metadataError = "selected instance " + std::to_string(instance) +
                      " rule '" + placement->second->rule +
                      "' has bundle metadata that disagrees with its plan";
      return;
    }

    auto [groupIt, inserted] = groups.try_emplace(instance);
    SelectedGroup &group = groupIt->second;
    if (inserted) {
      group.instance = instance;
      group.bundle = *bundle;
      group.rule = placement->second->rule;
    } else if (group.bundle.name != bundle->name ||
               group.bundle.emitterKey != bundle->emitterKey ||
               group.bundle.parameters != bundle->parameters ||
               group.rule != placement->second->rule) {
      metadataError = "selected instance " + std::to_string(instance) +
                      " has inconsistent bundle or rule metadata";
      return;
    }
    group.operations.push_back(op);
  });
  if (!metadataError.empty())
    return compileError(metadataError);

  if (operationForNode.size() != placementForNode.size())
    for (const auto &[node, placement] : placementForNode)
      if (!operationForNode.count(node))
        return compileError("selected instance " +
                            std::to_string(placement->instance) + " rule '" +
                            placement->rule + "' is missing mapped node " +
                            std::to_string(node));

  // Validate group membership, SSA order, control/effect boundaries and
  // external intermediate uses before resolving or invoking any emitter.
  for (auto &[instance, group] : groups) {
    auto expected = placementsForInstance.find(instance);
    if (expected == placementsForInstance.end() ||
        expected->second.size() != group.operations.size())
      return compileError("selected instance " + std::to_string(instance) +
                          " rule '" + group.rule +
                          "' has a different number of mapped operations and "
                          "plan placements");
    for (const PlanPlacement *placement : expected->second)
      group.placements.push_back(*placement);

    llvm::SmallPtrSet<mlir::Operation *, 8> members;
    for (mlir::Operation *op : group.operations)
      members.insert(op);
    llvm::SmallVector<mlir::Operation *> sourceOrder(group.operations.begin(),
                                                     group.operations.end());
    llvm::SmallPtrSet<mlir::Operation *, 8> emitted;
    group.operations.clear();
    while (group.operations.size() < sourceOrder.size()) {
      bool added = false;
      for (mlir::Operation *op : sourceOrder) {
        if (emitted.contains(op))
          continue;
        bool ready = true;
        for (mlir::Value operand : op->getOperands())
          if (mlir::Operation *def = operand.getDefiningOp())
            if (members.contains(def) && !emitted.contains(def)) {
              ready = false;
              break;
            }
        if (ready) {
          group.operations.push_back(op);
          emitted.insert(op);
          added = true;
        }
      }
      if (!added)
        return compileError("selected instance " + std::to_string(instance) +
                            " rule '" + group.rule +
                            "' has cyclic or unresolved SSA order");
    }

    mlir::Block *block = group.operations.front()->getBlock();
    for (mlir::Operation *op : group.operations)
      if (op->getBlock() != block)
        return compileError("selected instance " + std::to_string(instance) +
                            " rule '" + group.rule +
                            "' crosses a region or block boundary");
    for (mlir::Operation *op : group.operations)
      for (mlir::Value result : op->getResults()) {
        bool internalUse = false;
        bool externalUse = false;
        for (mlir::OpOperand &use : result.getUses()) {
          internalUse |= members.contains(use.getOwner());
          externalUse |= !members.contains(use.getOwner());
        }
        if (internalUse && externalUse)
          return compileError("selected instance " + std::to_string(instance) +
                              " rule '" + group.rule +
                              "' has a live external use of an internal value");
      }

    llvm::DenseSet<mlir::Operation *> memberSet;
    for (mlir::Operation *op : group.operations)
      memberSet.insert(op);
    bool betweenMembers = false;
    for (mlir::Operation &operation : *block) {
      if (memberSet.contains(&operation)) {
        betweenMembers = true;
        continue;
      }
      if (!betweenMembers)
        continue;
      bool laterMember = false;
      for (auto it = std::next(operation.getIterator()); it != block->end();
           ++it)
        if (memberSet.contains(&*it)) {
          laterMember = true;
          break;
        }
      if (!laterMember)
        break;
      if (!mlir::isMemoryEffectFree(&operation))
        return compileError("selected instance " + std::to_string(instance) +
                            " rule '" + group.rule +
                            "' crosses an intervening side effect");
    }
  }

  auto connectionTouches = [&](const PlanConnection &connection,
                               const SelectedGroup &group) {
    if (llvm::is_contained(connection.consumers, group.instance))
      return true;
    auto touchesPort = [&](const PortRef &port) {
      auto found = placementForNode.find(port.node);
      return found != placementForNode.end() &&
             found->second->instance == group.instance;
    };
    if (connection.producerPort && touchesPort(*connection.producerPort))
      return true;
    for (PortRef port : connection.producerPorts)
      if (touchesPort(port))
        return true;
    for (PortRef port : connection.consumerPorts)
      if (touchesPort(port))
        return true;
    return false;
  };
  for (const PlanConnection &connection : plan.connectionPlans)
    for (auto &[instance, group] : groups)
      if (connectionTouches(connection, group))
        group.connections.push_back(connection);

  // Verify every bundle and resolve every emitter before the first rewrite.
  for (auto &[instance, group] : groups) {
    group.emitter = target.createEmitter(group.bundle.emitterKey);
    if (!group.emitter)
      return compileError("selected emitter '" + group.bundle.emitterKey +
                          "' is not declared by target '" +
                          target.name().str() + "'");
    if (llvm::Error error = group.emitter->verify(group.bundle))
      return compileError(
          "selected instance " + std::to_string(instance) + " rule '" +
          group.rule +
          "' bundle validation failed: " + llvm::toString(std::move(error)));
  }
  compilation.selectedGroupsVerified += groups.size();

  mlir::IRRewriter rewriter(module.getContext());
  for (auto &[instance, group] : groups) {
    TargetLoweringContext context{target.machine(), group.placements,
                                  group.connections, instance};
    if (!group.emitter->hasLowering()) {
      compilation.referenceLowered += group.operations.size();
      ++compilation.referenceGroupsLowered;
      continue;
    }
    if (llvm::Error error = group.emitter->lower(group.operations, group.bundle,
                                                 context, rewriter))
      return compileError("selected instance " + std::to_string(instance) +
                          " rule '" + group.rule + "' lowering failed: " +
                          llvm::toString(std::move(error)));
    compilation.targetLowered += group.operations.size();
    ++compilation.backendGroupsRealized;
  }
  return llvm::Error::success();
}

/// Registers the bufferizable-op external models One-Shot Bufferize needs.
///
/// They are extensions rather than passes, so a context that never saw them has
/// dialects with no bufferization interface and the pass reports "op was not
/// bufferized" for perfectly ordinary tensors. A caller of this entry point
/// should not have to know that, so it is done here.
void registerBufferizationInterfaces(mlir::MLIRContext *context) {
  mlir::DialectRegistry registry;
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  context->appendDialectRegistry(registry);
}

/// Buffers the mapped kernel and rolls its loops: the Linalg form the existing
/// backend pipeline already consumes.
mlir::LogicalResult lowerToBackendForm(mlir::ModuleOp module) {
  registerBufferizationInterfaces(module.getContext());
  mlir::PassManager pm(module.getContext());
  pm.addPass(mlir::llk::createMicroToLinalgPass());
  BufOpts options;
  options.bufferizeFunctionBoundaries = true;
  // The calling convention is a row-major descriptor, so a function boundary
  // has to keep an identity layout. Left to the analysis, a kernel whose tiles
  // are re-represented by `micro.transform` can acquire a dynamic strided
  // signature -- which the C interface cannot then materialize, and which no
  // caller could describe.
  options.functionBoundaryTypeConversion =
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  return pm.run(module);
}

} // namespace

namespace testing {
llvm::Error
lowerSelectedOperationsForTest(mlir::ModuleOp module,
                               const MappingTarget &target,
                               const mlir::llk::mapping::CoveringPlan &plan,
                               MappedCompilation &compilation) {
  return lowerSelectedOperations(module, target, plan, compilation);
}
} // namespace testing

/// The tail every compilation shares: Linalg and loops, then the calling
/// convention, then the code.
llvm::Expected<MappedCompilation>
finishCompilation(MappedCompilation compilation,
                  const MappedCompileOptions &options) {
  // A source that came from the LLK export carries the semantic function
  // beside the kernel it produced. Only the kernel is compiled: the semantic
  // source is the reference the export was made from, and its high-level ops
  // have no lowering on this path -- leaving it in would fail the bufferization
  // for a function nobody asked to run.
  llvm::SmallVector<mlir::Operation *> semanticSources;
  compilation.module->walk([&](mlir::Operation *op) {
    if (!mlir::isa<mlir::func::FuncOp>(op))
      return;
    bool semantic = false;
    op->walk([&](mlir::Operation *inner) {
      if (inner->getName().getStringRef().starts_with("llk."))
        semantic = true;
    });
    if (semantic)
      semanticSources.push_back(op);
  });
  for (mlir::Operation *op : semanticSources)
    op->erase();

  if (mlir::failed(lowerToBackendForm(*compilation.module)))
    return compileError(
        "lowering the mapped kernel to Linalg and loops failed");
  compilation.stopped = MappedStop::Lowered;
  if (options.stop == MappedStop::Lowered)
    return compilation;

  // The calling convention, and then the code. `createMappedExecutable` takes
  // the module by value and keeps nothing from it, so the module this
  // compilation held is handed over here.
  //
  // The entry symbol is required rather than guessed: after lowering a module
  // may hold the source function and the kernel that replaced it, and picking
  // one by position would compile a different program than the caller asked
  // for.
  if (options.entrySymbol.empty())
    return compileError("compiling to an executable needs the entry symbol to "
                        "name; a module may hold more than one function");

  llvm::Expected<std::unique_ptr<MappedExecutable>> executable =
      createMappedExecutable(*compilation.module, options.entrySymbol);
  if (!executable)
    return executable.takeError();
  compilation.executable = std::move(*executable);
  compilation.stopped = MappedStop::Executable;
  compilation.module = nullptr;
  return compilation;
}

llvm::Expected<MappedCompilation>
compileMappedKernel(mlir::ModuleOp source, const MappingTarget &target,
                    const mlir::llk::mapping::CoveringPlan &plan,
                    const MappedCompileOptions &options) {
  if (options.backend != MappedBackend::Reference)
    return compileError("selected-target backend is not available yet");

  MappedCompilation compilation;

  // Binding clones the source and writes the selected state onto the clone, so
  // the caller's module is untouched whatever happens next.
  std::unique_ptr<PlanMaterializer> materializer =
      mlir::llk::micro_mapping_detail::createCanonicalPlanMaterializer();
  llvm::Expected<mlir::llk::mapping::BoundPlan> bound =
      mlir::llk::mapping::bindPlan(source, plan, target,
                                   options.requireExecutable
                                       ? BindContract::Executable
                                       : BindContract::Partial,
                                   materializer.get());
  if (!bound)
    return bound.takeError();
  compilation.module = std::move(bound->module);
  compilation.stopped = MappedStop::MappedMicro;
  if (options.stop == MappedStop::MappedMicro)
    return compilation;

  if (llvm::Error error =
          mlir::llk::mapping::verifyMappedMicroIR(*compilation.module, target))
    return std::move(error);

  if (llvm::Error error = lowerSelectedOperations(*compilation.module, target,
                                                  plan, compilation))
    return std::move(error);
  compilation.stopped = MappedStop::TargetLowered;
  if (options.stop == MappedStop::TargetLowered)
    return compilation;

  return finishCompilation(std::move(compilation), options);
}

llvm::Expected<MappedCompilation>
compileConcreteMicroKernel(mlir::ModuleOp source,
                           const MappedCompileOptions &options) {
  if (options.backend != MappedBackend::Reference)
    return compileError("selected-target backend is not available yet");

  if (options.stop == MappedStop::MappedMicro ||
      options.stop == MappedStop::TargetLowered)
    return compileError("a concrete Micro kernel has no plan to bind and no "
                        "target to lower it against; ask for `lowered` or "
                        "`executable`");

  MappedCompilation compilation;
  compilation.module = source.clone();
  return finishCompilation(std::move(compilation), options);
}

} // namespace llk
