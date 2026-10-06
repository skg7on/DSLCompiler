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
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"

#include <string>

// The canonical plan materializer lives beside this file, in the one library
// that owns the Micro dialect on the mapping path.
#include "MicroMappingCommon.h"

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
llvm::Error lowerSelectedOperations(mlir::ModuleOp module,
                                    const MappingTarget &target,
                                    MappedCompilation &compilation) {
  // The lowering context carries the selected facts. Only the machine is
  // populated here: the placements and connections a target might need are the
  // bound plan's, and the emitter interface takes them as arrays so a lowerer
  // that reads them can be handed them without changing this signature.
  TargetLoweringContext context{target.machine(), {}, {}};

  llvm::SmallVector<mlir::Operation *> covered;
  module.walk([&](mlir::Operation *op) {
    if (op->hasAttr("micro.mapping"))
      covered.push_back(op);
  });

  mlir::IRRewriter rewriter(module.getContext());
  for (mlir::Operation *op : covered) {
    llvm::Expected<TargetBundle> bundle = readSelectedBundle(op);
    if (!bundle)
      return bundle.takeError();

    std::unique_ptr<mlir::llk::mapping::TargetEmitter> emitter =
        target.createEmitter(bundle->emitterKey);
    if (!emitter)
      return compileError("selected emitter '" + bundle->emitterKey +
                          "' is not declared by target '" +
                          target.name().str() + "'");

    if (!emitter->hasLowering()) {
      ++compilation.referenceLowered;
      continue;
    }
    if (llvm::Error error = emitter->lower({op}, *bundle, context, rewriter))
      return error;
    ++compilation.targetLowered;
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

/// The tail every compilation shares: Linalg and loops, then the calling
/// convention, then the code.
llvm::Expected<MappedCompilation>
finishCompilation(MappedCompilation compilation,
                  const MappedCompileOptions &options) {
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

  if (llvm::Error error =
          lowerSelectedOperations(*compilation.module, target, compilation))
    return std::move(error);
  compilation.stopped = MappedStop::TargetLowered;
  if (options.stop == MappedStop::TargetLowered)
    return compilation;

  return finishCompilation(std::move(compilation), options);
}

llvm::Expected<MappedCompilation>
compileConcreteMicroKernel(mlir::ModuleOp source,
                           const MappedCompileOptions &options) {
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
